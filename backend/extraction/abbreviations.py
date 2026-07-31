"""
abbreviations.py — merge acronyms into their expansions

Papers define their own abbreviations — "density functional theory (DFT)" —
so the mapping is stated in the source rather than inferred. That makes this
the one entity-resolution signal that is evidence rather than similarity, and
it closes the gap neither lexical keys nor embeddings can: measured on a
192-document corpus, cosine scored 'DFT'/'density functional theory' at 0.269,
BELOW unrelated pairs, while the corpus states the definition 121 times.

Extraction runs elsewhere. scispaCy's AbbreviationDetector (Schwartz & Hearst,
2003) needs numpy 1.x, which would downgrade this environment, so it lives in
its own venv behind scripts/extract_abbreviations.py and hands over a JSON
file. This module only consumes that file, so the main pipeline keeps its
dependency set and degrades to a no-op when the file is absent.

A short form is merged only when one expansion DOMINATES its definitions, since
the same acronym can mean different things in different papers — measured here:
GP is Gaussian process in most papers and genetic programming in one, BO is
Bayesian optimization and once bridging oxygen. Below the threshold the
acronym is left alone rather than guessed at.

Crawler-agnostic: the graph payload is the only contract.

Contents:
  Loading   load_definitions
  Mapping   dominant_expansions, abbreviation_merge_map
  Applying  merge_abbreviations
"""

import collections
import json
import os
from pathlib import Path

from entity_resolution import resolution_key
from graph_merge import apply_merge_map, clusters_from_map, fold_clusters

# Share of a short form's definitions that one expansion must hold before the
# merge is trusted. High on purpose: an acronym with a genuinely split meaning
# should stay unmerged rather than be resolved by majority vote.
DOMINANCE = float(os.environ.get("ABBREVIATION_DOMINANCE", "0.8"))
DEFINITIONS_FILE = "abbreviations.json"


def is_abbreviation(short_form: str) -> bool:
    """Whether a short form is shaped like an acronym at all.

    Schwartz-Hearst validates by character subsequence, which an ordinary word
    can satisfy by accident. Measured on a 192-document corpus: 'atomic
    coordinates (atoms)' and 'unit cell (unit)' both validated, folding a
    general concept into a narrow one, and 'Lithium (Li)' would have merged an
    element symbol into a word.

    Counting capitals rather than demanding all of them keeps the mixed-case
    acronyms this literature is full of — GANs, NequIP, MEGNet, GNoME, MLIPs —
    while still rejecting the three above (no capitals, and Li has only one).
    """
    return len(short_form) >= 2 and sum(c.isupper() for c in short_form) >= 2


def load_definitions(data_dir: Path) -> dict[str, dict[str, int]]:
    """{short form: {long form: times defined}} as written by the extractor.

    Returns {} when the file is absent, so a corpus that never ran the
    extractor simply skips this pass.
    """
    path = Path(data_dir) / DEFINITIONS_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dominant_expansions(definitions: dict[str, dict[str, int]],
                        dominance: float) -> dict[str, str]:
    """{short form: the one expansion that carries `dominance` of its definitions}.

    Expansions are pooled on their resolution key first, so 'Machine Learning'
    and 'machine learning' count as the same reading rather than splitting the
    vote and pushing a clear case below the threshold.
    """
    dominant = {}
    for short_form, long_forms in definitions.items():
        if not is_abbreviation(short_form):
            continue
        pooled = collections.Counter()
        surface = {}
        for long_form, count in long_forms.items():
            key = resolution_key(long_form)
            pooled[key] += count
            # Keep the most-seen spelling of each reading as its display form.
            if key not in surface or count > long_forms.get(surface[key], 0):
                surface[key] = long_form
        total = sum(pooled.values())
        if not total:
            continue
        key, count = pooled.most_common(1)[0]
        if count / total >= dominance:
            dominant[short_form] = surface[key]
    return dominant


def abbreviation_merge_map(graph: dict, definitions: dict[str, dict[str, int]],
                           dominance: float = DOMINANCE) -> dict[str, str]:
    """{name: representative} folding acronyms into their expansions.

    Only pairs where BOTH sides are already nodes are merged — an expansion the
    graph never extracted is not worth inventing a node for. The expansion is
    always the representative, so the graph reads without decoding acronyms.
    """
    by_key = collections.defaultdict(list)
    for entity in graph["entities"]:
        by_key[resolution_key(entity)].append(entity)
    entities = set(graph["entities"])

    merge_map = {}
    for short_form, long_form in dominant_expansions(definitions, dominance).items():
        if short_form not in entities:
            continue
        matches = by_key.get(resolution_key(long_form))
        if not matches:
            continue
        representative = matches[0]
        if representative == short_form:
            continue
        merge_map[short_form] = representative
        merge_map[representative] = representative
    return merge_map


def merge_abbreviations(graph: dict, data_dir: Path,
                        dominance: float = DOMINANCE) -> tuple[dict, dict]:
    """Fold acronyms into their expansions. Returns (merged, stats)."""
    definitions = load_definitions(data_dir)
    if not definitions:
        return dict(graph), {"entitiesMerged": 0, "groups": 0, "definitions": 0}

    merge_map = abbreviation_merge_map(graph, definitions, dominance)
    merged, stats = apply_merge_map(graph, entity_map=merge_map)

    clusters = clusters_from_map(merge_map)
    # Fold into the entity clusters the lexical pass already recorded, so one
    # field explains every entity merge regardless of which pass made it. This
    # pass runs second, so it can merge away a name the lexical pass made a
    # representative — fold_clusters absorbs that group rather than orphaning it.
    merged["entityClusters"] = fold_clusters(merged.get("entityClusters") or {}, clusters)

    stats["groups"] = len(clusters)
    stats["definitions"] = len(definitions)
    return merged, stats
