"""
predicate_normalization.py — embedding-based predicate normalization

The model emits one relation under many phrasings — 'uses' / 'use', 'is used
for' / 'used for' / 'are used in', 'is a type of' / 'are types of' — which
leaves thousands of predicate types for a few tens of thousands of relations.
Lexical folding barely touches this (measured: 4022 predicates, 7 removed),
because the redundancy is semantic, so predicates are embedded and clustered by
cosine similarity instead.

Embeddings are local (sentence-transformers, the same MiniLM weights as
SAPPHIRE_EMBEDDING_MODEL), so the pass costs no tokens and runs in seconds.

Cosine alone is not a safe merge test. It rates 'learns from' / 'cannot learn
from' and 'is evaluated on' / 'is used to evaluate' as near-identical, because
they share nearly every token — but the first pair differs in truth value and
the second in argument order. So similarity only nominates CANDIDATES, and an
NLI cross-encoder decides. Contradiction and non-entailment are exactly what
that model is trained to detect, which is why a trained judgement replaces
hand-written syntax rules here: a regex over predicate strings has to be
extended for every phrasing a new corpus invents.

Two predicates merge only on MUTUAL entailment: 'X p1 Y' must entail 'X p2 Y'
and the reverse. One-way entailment is a hypernym, not a synonym.

A cheap 'by' pre-filter runs first. Measured on a 192-document corpus, 242
predicates end in 'by' and every frequent one is a true inverse ('is used by',
'is authored by'), so dropping those pairs before inference is free precision.

Crawler-agnostic: the graph payload is the only contract.

Contents:
  Voice       is_inverse
  Candidates  candidate_pairs
  NLI gate    entailed_pairs
  Clustering  predicate_groups
  Normalize   normalize_predicates
"""

import collections
import os
import re

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import CrossEncoder, SentenceTransformer

from graph_merge import (apply_merge_map, clusters_from_map, merge_map_from_groups,
                         name_counts)

# Same weights as SAPPHIRE_EMBEDDING_MODEL — Xenova/all-MiniLM-L12-v2 is the
# ONNX port of this checkpoint, so predicate vectors share the corpus space.
EMBEDDING_MODEL = os.environ.get("GRAPH_EMBEDDING_MODEL",
                                 "sentence-transformers/all-MiniLM-L12-v2")
# Cosine floor for NOMINATING a pair. Deliberately lower than a merge threshold
# would be: this only has to supply recall, since the NLI gate supplies the
# precision. Measured on 4022 predicates: 0.75 nominates 5321 pairs, 0.85 only
# 1688 — and the 0.75-0.85 band is where 'uses'/'use' style variants live.
CANDIDATE_SIMILARITY = float(os.environ.get("PREDICATE_CANDIDATE_SIMILARITY", "0.75"))
NLI_MODEL = os.environ.get("PREDICATE_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
# Minimum entailment probability, required in BOTH directions.
ENTAILMENT_FLOOR = float(os.environ.get("PREDICATE_ENTAILMENT_FLOOR", "0.5"))
# Set PREDICATE_NLI=0 to fall back to cosine alone at SIMILARITY (no model
# download, no inference) — the old behaviour, kept for offline runs.
USE_NLI = os.environ.get("PREDICATE_NLI", "1") != "0"
SIMILARITY = float(os.environ.get("PREDICATE_SIMILARITY", "0.85"))
# Average-linkage cut on the approval graph. 0.5 means a group holds together
# only while more than half its internal pairs were approved.
_LINKAGE_DISTANCE = float(os.environ.get("PREDICATE_LINKAGE_DISTANCE", "0.5"))

# Predicates are fragments, so they are read into a frame before inference — an
# NLI model scores sentences, not phrases. Two frames are used and entailment is
# required under BOTH, because the choice of arguments changes the verdict and
# the two fail on different pairs. Measured on a 10-pair probe: contentless
# X/Y wrongly merged contains/is contained in (0.826) and uses/is also known as
# (0.921), which the noun frame rejects (0.002, 0.184); the noun frame wrongly
# merged is a type of/is a (0.991), which X/Y rejects (0.051). Requiring both
# cut wrong merges from 3 to 1 with no loss of correct ones. Fully concrete
# arguments ("the Materials Project <pred> cerium oxide") were far worse — 5 of
# 6 wrong — because a plausible-sounding sentence is entailed too easily.
_FRAMES = (("X", "Y"), ("the method", "the dataset"))

# 'by' as the final word marks the agent of a passive relation.
_INVERSE = re.compile(r"\bby$")


def is_inverse(predicate: str) -> bool:
    """True when the predicate names its agent, making it a relation's inverse."""
    return bool(_INVERSE.search(predicate.strip()))


def candidate_pairs(predicates: list[str], similarity: float,
                    model_name: str) -> list[tuple[int, int]]:
    """Index pairs worth judging: cosine-similar and of the same voice."""
    model = SentenceTransformer(model_name)
    vectors = model.encode(predicates, normalize_embeddings=True,
                           batch_size=256, show_progress_bar=False)
    # Vectors are unit length, so the dot product IS cosine similarity.
    similar = (vectors @ vectors.T) >= similarity
    left, right = np.where(np.triu(similar, 1))

    voice = np.array([is_inverse(predicate) for predicate in predicates])
    same_voice = voice[left] == voice[right]
    return list(zip(left[same_voice].tolist(), right[same_voice].tolist()))


def entailed_pairs(predicates: list[str], pairs: list[tuple[int, int]],
                   model_name: str, floor: float) -> list[tuple[int, int]]:
    """Keep only the pairs that entail each other in both directions.

    Both directions are required because one-way entailment is a hypernym
    ('is a type of' entails 'is a', not the reverse), and merging those loses
    the more specific claim. A contradiction — which is what a negation scores
    against its positive — fails in both directions.
    """
    if not pairs:
        return []
    model = CrossEncoder(model_name)
    # Label order varies between checkpoints, so it is read off the config
    # rather than assumed.
    labels = model.model.config.id2label
    entailment = next(index for index, label in labels.items()
                      if label.lower() == "entailment")

    survivors = pairs
    for subject, obj in _FRAMES:
        if not survivors:
            break
        framed = [f"{subject} {predicate} {obj}." for predicate in predicates]
        forward = [(framed[left], framed[right]) for left, right in survivors]
        backward = [(framed[right], framed[left]) for left, right in survivors]
        scores = model.predict(forward + backward, batch_size=64,
                               apply_softmax=True, show_progress_bar=False)
        split = len(survivors)
        survivors = [pair for index, pair in enumerate(survivors)
                     if scores[index][entailment] >= floor
                     and scores[split + index][entailment] >= floor]
    return survivors


def _average_linkage_labels(approved, distance: float):
    """Cluster the approval graph by average linkage.

    Distance is 0 for an approved pair and 1 for anything else, so a cluster
    only forms while the AVERAGE pair inside it is approved — the fraction of
    approved pairs must stay above 1 - distance.
    """
    if not approved.any():
        return list(range(len(approved)))
    clustering = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                         linkage="average",
                                         distance_threshold=distance)
    return clustering.fit_predict(1.0 - approved.astype(np.float64))


def predicate_groups(predicates: list[str], similarity: float, model_name: str,
                     use_nli: bool = USE_NLI, nli_model: str = NLI_MODEL,
                     floor: float = ENTAILMENT_FLOOR) -> list[list[str]]:
    """Group predicates that mean the same thing.

    Cosine nominates, the NLI gate decides, and connected components takes the
    transitive closure of the surviving pairs.
    """
    pairs = candidate_pairs(predicates, similarity, model_name)
    if use_nli:
        pairs = entailed_pairs(predicates, pairs, nli_model, floor)

    size = len(predicates)
    approved = np.zeros((size, size), dtype=bool)
    for left, right in pairs:
        approved[left, right] = approved[right, left] = True

    # Connected components would be SINGLE linkage: A~B and B~C chain A to C
    # even when A and C were never approved. Measured on 4022 predicates, that
    # produced one 223-member group spanning 'are derived from' to 'are designed
    # for'. Average linkage instead requires most of a group to be pairwise
    # approved, so a chain has to earn its way into a cluster.
    labels = _average_linkage_labels(approved, _LINKAGE_DISTANCE)
    groups = collections.defaultdict(list)
    for predicate, label in zip(predicates, labels):
        groups[label].append(predicate)
    return list(groups.values())


def normalize_predicates(graph: dict,
                         similarity: float = None,
                         model_name: str = EMBEDDING_MODEL) -> tuple[dict, dict]:
    """Fold synonymous predicates onto a single representative.

    Returns (normalized, stats). The merge map lands in the payload as
    predicateClusters, so which phrasings collapsed is recoverable.
    """
    predicates = sorted(graph["edges"])
    if len(predicates) < 2:
        return dict(graph), {"predicatesMerged": 0, "groups": 0}

    # The cosine floor means different things in the two modes: a candidate
    # threshold when the NLI gate follows, a merge threshold when it doesn't.
    if similarity is None:
        similarity = CANDIDATE_SIMILARITY if USE_NLI else SIMILARITY
    groups = predicate_groups(predicates, similarity, model_name)
    counts = name_counts(graph["relations"], "predicate")
    merge_map = merge_map_from_groups(groups, counts)
    normalized, stats = apply_merge_map(graph, predicate_map=merge_map)

    normalized["predicateClusters"] = clusters_from_map(merge_map)
    stats["groups"] = len(normalized["predicateClusters"])
    return normalized, stats
