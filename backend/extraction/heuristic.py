"""
heuristic.py — select the k most important documents

Orchestration only: every scoring primitive lives in heuristic_utils.py; this
file owns the PARAMETERS and the pipeline glue (read doclings/categories →
score → rank → write heuristic_output.json).

Crawler-agnostic. It builds no citation graph of its own — sapphire's citations
stage writes citations.json and this file consumes it when present. That is the
whole reason this lives in extraction/ rather than sapphire/: a general-document
crawler has no references to build a graph from, and used to be unable to rank
at all without importing academic code.

Scoring model:
  final_score = ALPHA * bm25_component + (1 - ALPHA) * pagerank_score

  bm25_component — blend of two sub-signals:
    - representativeness: mean of the TOP_M_CHUNKS best chunk-level BM25
      scores against the document's cluster keywords (read straight from
      categories.json — generate_categories.js is the single source of
      keyword truth). Top-m chunk scoring ranks documents on the density of
      their best material: long docs get no credit for filler, and (unlike
      a plain per-chunk average) broad documents aren't punished for
      off-topic appendices. See topm_chunk_representativeness().
      Normalized WITHIN each cluster (percentile among cluster members),
      not globally: every doc is graded against its own cluster's keywords,
      so cross-cluster raw scores were never comparable — a singleton
      trivially aces the exam its own vocabulary wrote.
    - novelty: average IDF of a document's unique non-hapax vocabulary vs.
      the full corpus — counterweight to representativeness, which rewards
      typicality. Percentile-normalized globally.

  pagerank_score — PageRank over the citation graph, read from citations.json
    if the citations stage ran (sapphire only). OPTIONAL: a crawler with no
    references produces no such file, and the score then falls back to the
    BM25 component alone rather than blending against a constant.

    That fallback matters. An edgeless graph gives every document the same
    PageRank, which percentile-normalizes to exactly 0.5 for all of them, so
    the old unconditional blend added a fixed 0.5*(1-ALPHA) offset to every
    score: ordering survived but the range was squashed and the reported
    pagerankScore was a fiction. Omitting the term is the honest form.

Top-k selection uses per-cluster quotas, proportional to cluster size
(largest-remainder apportionment): a cluster holding 40% of the corpus gets
~40% of the slots, each filled by its highest-scoring members. Singletons
only compete for remainder slots, which neutralizes their within-cluster
percentile of 1.0.

Writes data/heuristic_output.json:
  topK        : [{docId, filename, cluster, finalScore, bm25Score,
                  bm25Representativeness, bm25Novelty, pagerankScore}]
  chunkScores : {chunkId: score}  every chunk scored against its document's
                cluster keywords, for kg_graph.py to pick a summary document's
                best material without relying on academic section headings

chunkScores is ADDITIVE — bm25Representativeness still comes from 180-token
windows over the body text, not from these chunks. Scoring real chunks for
representativeness too would change document ranking, and this stage's output
has to stay stable for corpora that were already built.

Citation edges are NOT written here; sapphire/citations.py owns them.

Dependencies: networkx (pip install networkx).
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from heuristic_utils import (
    BM25,
    compute_pagerank,
    novelty_score,
    percentile_normalize,
    tokenise,
    top_terms,
    topm_chunk_representativeness,
)

# ---------------------------------------------------------------------------
# Parameters — the single place tunables live
# ---------------------------------------------------------------------------

ROOT            = Path(__file__).resolve().parents[2]
DATA_DIR        = ROOT / os.environ.get("DATA_DIR", "data")
DOCLINGS_PATH   = DATA_DIR / "doclings.json"
CATEGORIES_PATH = DATA_DIR / "categories.json"
# Written by sapphire/citations.py. Absent for crawlers that have no references,
# which is the signal to rank on BM25 alone.
CITATIONS_PATH  = DATA_DIR / "citations.json"
# The chunk store (embed.js). Exported without vectors for this stage — only
# chunk text and ids are needed, and the float arrays are ~85% of the file.
CHUNKS_PATH     = DATA_DIR / "embeddings.json"
OUTPUT_PATH     = DATA_DIR / "heuristic_output.json"

# Every tunable below is an env override (see .env, HEURISTIC_* block). The
# literal here is the default and must stay in sync with .env — Python is
# spawned by the pipeline with the env inherited, but a bare `python
# heuristic.py` gets no .env at all and falls back to these.
K              = int(os.environ.get("HEURISTIC_K", "2"))                   # top-k documents to select
ALPHA          = float(os.environ.get("HEURISTIC_ALPHA", "0.5"))           # weight of the BM25 component vs. PageRank
                                                                           # 0.5: both signals inform on a densely-cited corpus.
                                                                           # Raise it when intra-corpus citations are sparse —
                                                                           # PageRank goes near-constant there (see .env)
NOVELTY_WEIGHT = float(os.environ.get("HEURISTIC_NOVELTY_WEIGHT", "0.2"))  # how much rare vocabulary boosts the final score

BM25_K1        = float(os.environ.get("HEURISTIC_BM25_K1", "1.5"))         # term-frequency saturation
BM25_B         = float(os.environ.get("HEURISTIC_BM25_B", "0.75"))         # length normalization strength

CHUNK_WORDS         = int(os.environ.get("HEURISTIC_CHUNK_WORDS", "180"))  # window size (tokens) for chunk-level
                                                                          # scoring — matches the pipeline's CHUNK_SIZE
TOP_M_CHUNKS        = int(os.environ.get("HEURISTIC_TOP_M_CHUNKS", "5"))   # how many best chunks define representativeness
KEYWORDS_N          = int(os.environ.get("HEURISTIC_KEYWORDS_N", "20"))    # fallback keywords for uncategorized docs

# Bibliography headings, shared with extract.py / kg_graph.py / regex_utils.js —
# one env var so the four copies can't drift apart.
_REF_HEADINGS = frozenset(
    heading.strip().lower()
    for heading in os.environ.get(
        "PIPELINE_REF_HEADINGS",
        "references,bibliography,works cited,literature cited,citations").split(",")
    if heading.strip()
)

# HEURISTIC_MIN_KEY_LENGTH / HEURISTIC_MIN_CONTAINED_TITLE moved to
# sapphire/citations.py with the graph builder they configure.
PAGERANK_DAMPING = float(os.environ.get("HEURISTIC_PAGERANK_DAMPING", "0.85"))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _score_chunks(bm25: BM25, clusters: list[list[str]], cluster_kws: list[list[str]],
                  cluster_of: dict[str, int]) -> dict[str, float]:
    """{chunkId: BM25 score} for every chunk, against its document's cluster keywords.

    Additive to the document ranking above — nothing here feeds finalScore. It
    exists so kg_graph.py can pick a summary document's best chunks by score
    rather than by matching 'abstract'/'conclusion' headings, which a report or
    manual does not have.

    Scored the same way topm_chunk_representativeness scores its windows (same
    BM25, same avgdl pinned to CHUNK_WORDS), so a chunk's score is comparable to
    the window scores that decided its document's rank.

    Returns {} when the chunk store is absent — the stage still ranks, and
    kg_graph.py falls back to heading matching.
    """
    if not CHUNKS_PATH.exists():
        print("[heuristic] no chunk store — skipping chunk scores", file=sys.stderr)
        return {}
    chunks = json.loads(CHUNKS_PATH.read_text(encoding='utf-8')).get("chunks", [])
    scores: dict[str, float] = {}
    for chunk in chunks:
        cluster_idx = cluster_of.get(chunk.get("docId"))
        if cluster_idx is None:
            continue
        chunk_tokens = tokenise(chunk.get("text") or "")
        if not chunk_tokens:
            continue
        scores[chunk["id"]] = round(
            bm25.score_tokens(chunk_tokens, cluster_kws[cluster_idx],
                              avgdl=float(CHUNK_WORDS)), 4)
    print(f"[heuristic] scored {len(scores)} chunk(s) across {len(clusters)} cluster(s)")
    return scores


def run(k: int = K) -> None:
    if not DOCLINGS_PATH.exists():
        print(f"[heuristic] {DOCLINGS_PATH} not found — run extract.py first.", file=sys.stderr)
        sys.exit(1)

    doclings: dict = json.loads(DOCLINGS_PATH.read_text(encoding='utf-8'))
    if not doclings:
        print("[heuristic] doclings.json is empty — nothing to score.")
        return

    doc_ids   = list(doclings.keys())
    filenames = {doc_id: doclings[doc_id]["filename"] for doc_id in doc_ids}

    # Tokenize body text only — reference sections would leak author names and
    # cited titles into BM25 novelty. Headings are normalized so numbered
    # bibliography headings ('7. References') don't slip through.
    def _norm_heading(heading: str) -> str:
        normalized = heading.lower().strip()
        normalized = re.sub(r'^[\divxlc]+[\.\)]?\s+', '', normalized)
        return normalized.rstrip(' .:')

    def _body_text(docling_entry: dict) -> str:
        sections = docling_entry.get("sections", [])
        body_sections = [section["text"] for section in sections
                         if _norm_heading(section.get("heading", "")) not in _REF_HEADINGS
                         and (section.get("text") or "").strip()]
        return " ".join(body_sections) if body_sections else docling_entry.get("text", "")

    tokenised = {doc_id: tokenise(_body_text(doclings[doc_id])) for doc_id in doc_ids}

    bm25 = BM25([tokenised[doc_id] for doc_id in doc_ids], k1=BM25_K1, b=BM25_B)

    # Clusters + keywords come from categories.json (single source of keyword
    # truth). Docs absent from it form one pseudo-cluster graded against
    # corpus-wide fallback keywords.
    fallback_kws = top_terms(bm25, n=KEYWORDS_N)
    clusters: list[list[str]] = []      # member doc_ids per cluster
    cluster_kws: list[list[str]] = []   # keywords per cluster (parallel)
    if CATEGORIES_PATH.exists():
        categories = json.loads(CATEGORIES_PATH.read_text(encoding='utf-8'))
        for category in categories.get("categories", []):
            member_ids = [member["docId"] for member in category.get("members", [])
                          if member["docId"] in tokenised]
            if not member_ids:
                continue
            clusters.append(member_ids)
            cluster_kws.append(category.get("keywords") or fallback_kws)
    categorized_ids = {doc_id for members in clusters for doc_id in members}
    uncategorized_ids = [doc_id for doc_id in doc_ids if doc_id not in categorized_ids]
    if uncategorized_ids:
        clusters.append(uncategorized_ids)
        cluster_kws.append(fallback_kws)

    # Representativeness: percentile-normalized within each cluster (see file
    # header). Novelty: corpus-level, normalized globally.
    norm_repr: dict[str, float] = {}
    cluster_of: dict[str, int] = {}
    for cluster_idx, members in enumerate(clusters):
        raw_repr = {
            doc_id: topm_chunk_representativeness(
                bm25, tokenised[doc_id], cluster_kws[cluster_idx],
                chunk_words=CHUNK_WORDS, top_m=TOP_M_CHUNKS,
            )
            for doc_id in members
        }
        norm_repr.update(percentile_normalize(raw_repr))
        for doc_id in members:
            cluster_of[doc_id] = cluster_idx

    raw_novelty  = {doc_id: novelty_score(bm25, tokenised[doc_id]) for doc_id in doc_ids}
    norm_novelty = percentile_normalize(raw_novelty)
    bm25_component = {
        doc_id: (1 - NOVELTY_WEIGHT) * norm_repr[doc_id] + NOVELTY_WEIGHT * norm_novelty[doc_id]
        for doc_id in doc_ids
    }

    # Citation graph, when a crawler produced one. Rebuilding the adjacency from
    # the edge list is equivalent to the builder's own return value: isolated
    # documents are missing as keys here, but compute_pagerank seeds every node
    # from doc_ids before reading any edges, so the scores are identical.
    norm_pagerank: dict[str, float] | None = None
    if CITATIONS_PATH.exists():
        citation_edges = json.loads(CITATIONS_PATH.read_text(encoding='utf-8')).get("edges", [])
        adjacency: dict[str, set[str]] = {}
        for edge in citation_edges:
            adjacency.setdefault(edge["source"], set()).add(edge["target"])
        print(f"[heuristic] citations.json: {len(citation_edges)} edge(s) — blending PageRank")
        norm_pagerank = percentile_normalize(
            compute_pagerank(adjacency, doc_ids, damping=PAGERANK_DAMPING)
        )
    else:
        print("[heuristic] no citations.json — ranking on BM25 alone")

    def _final_score(doc_id: str) -> float:
        if norm_pagerank is None:
            return round(bm25_component[doc_id], 4)
        return round(ALPHA * bm25_component[doc_id]
                     + (1 - ALPHA) * norm_pagerank[doc_id], 4)

    scored = [
        {
            "docId":                  doc_id,
            "filename":               filenames[doc_id],
            "cluster":                cluster_of[doc_id],
            "finalScore":             _final_score(doc_id),
            "bm25Score":              round(bm25_component[doc_id], 4),
            "bm25Representativeness": round(norm_repr[doc_id], 4),
            "bm25Novelty":            round(norm_novelty[doc_id], 4),
            # null rather than 0.5 when there is no graph: reporting a real
            # percentile for a score nothing produced is the misleading part.
            "pagerankScore":          None if norm_pagerank is None
                                      else round(norm_pagerank[doc_id], 4),
        }
        for doc_id in doc_ids
    ]

    # Per-cluster quotas, proportional to cluster size (largest-remainder
    # apportionment — see file header for why this neutralizes singletons).
    k = min(k, len(doc_ids))
    cluster_sizes = [len(members) for members in clusters]
    exact_quotas = [k * size / len(doc_ids) for size in cluster_sizes]
    quotas = [int(exact) for exact in exact_quotas]
    remaining_slots = k - sum(quotas)
    # Ties broken by larger cluster, then index — deterministic run-to-run.
    remainder_order = sorted(
        range(len(clusters)),
        key=lambda cluster_idx: (-(exact_quotas[cluster_idx] - quotas[cluster_idx]),
                                 -cluster_sizes[cluster_idx], cluster_idx))
    while remaining_slots > 0:
        for cluster_idx in remainder_order:
            if remaining_slots == 0:
                break
            if quotas[cluster_idx] < cluster_sizes[cluster_idx]:
                quotas[cluster_idx] += 1
                remaining_slots -= 1

    scored_by_cluster: dict[int, list[dict]] = {}
    for scored_doc in scored:
        scored_by_cluster.setdefault(scored_doc["cluster"], []).append(scored_doc)
    top_k: list[dict] = []
    for cluster_idx, members in scored_by_cluster.items():
        members.sort(key=lambda doc: -doc["finalScore"])
        top_k.extend(members[:quotas[cluster_idx]])
    top_k.sort(key=lambda doc: -doc["finalScore"])

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "k":           k,
        "topK":        top_k,
        "chunkScores": _score_chunks(bm25, clusters, cluster_kws, cluster_of),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[heuristic] Top-{k} documents:")
    for top_doc in top_k:
        print(f"  {top_doc['filename']}  (score={top_doc['finalScore']})")
    print(f"[heuristic] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Rank documents by BM-25 (top-m chunk representativeness + novelty) + citation PageRank"
    )
    parser.add_argument("--k", type=int, default=K, help=f"Number of top documents to select (default {K})")
    args = parser.parse_args()
    run(k=args.k)