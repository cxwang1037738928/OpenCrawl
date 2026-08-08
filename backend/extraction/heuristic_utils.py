"""
heuristic_utils.py — reusable scoring primitives for heuristic.py

Every function here is pure with respect to configuration: all tunables
arrive as arguments (no module-level constants), so heuristic.py owns the
parameters and this module owns the mechanics. Nothing here reads or
writes files.

Crawler-agnostic on purpose: nothing here knows about papers, references or
DOIs. The citation-specific half lives in sapphire/citation_utils.py, and
keeping the two apart is what lets a non-academic crawler reuse this ranking.

compute_pagerank lives here rather than with the citation code because the
ranker owns scoring — it takes an adjacency as an argument and returns
scores, so the citation-shaped input stays at the boundary.

Contents:
  Tokenisation      tokenise
  BM25              BM25, top_terms
  Doc scoring       topm_chunk_representativeness, novelty_score
  Normalization     percentile_normalize
  Graph scoring     compute_pagerank
"""

import math
import re
from collections import defaultdict

import networkx as nx

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","this","that","these","those","it","its","i","we","you","he",
    "she","they","their","our","us","not","no","so","if","than","then",
}


def tokenise(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 2]


# ---------------------------------------------------------------------------
# BM-25
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, corpus: list[list[str]], k1: float, b: float):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.avgdl = sum(len(doc_tokens) for doc_tokens in corpus) / max(self.N, 1)
        self.df: dict[str, int] = defaultdict(int)
        self.tf_per_doc: list[dict[str, int]] = []

        for doc_tokens in corpus:
            term_freq: dict[str, int] = defaultdict(int)
            counted_terms: set[str] = set()
            for term in doc_tokens:
                term_freq[term] += 1
                if term not in counted_terms:
                    self.df[term] += 1
                    counted_terms.add(term)
            self.tf_per_doc.append(term_freq)

    def idf(self, term: str) -> float:
        doc_freq = self.df.get(term, 0)
        return math.log((self.N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def score_tokens(self, doc_tokens: list[str], query: list[str],
                     avgdl: float | None = None) -> float:
        """
        BM25 score of a token sequence against a query term set.
        `avgdl` overrides the corpus average document length — pass the
        expected CHUNK length when scoring chunk-sized windows, so length
        normalization is calibrated to the unit actually being scored
        rather than to full documents.
        """
        avgdl = avgdl if avgdl is not None else self.avgdl
        term_freq: dict[str, int] = defaultdict(int)
        for token in doc_tokens:
            term_freq[token] += 1
        doc_length = len(doc_tokens)
        score = 0.0
        for term in set(query):
            freq = term_freq.get(term, 0)
            if freq == 0:
                continue
            term_idf = self.idf(term)
            tf_norm = freq * (self.k1 + 1) / (
                freq + self.k1 * (1 - self.b + self.b * doc_length / avgdl))
            score += term_idf * tf_norm
        return score


def top_terms(bm25: BM25, n: int) -> list[str]:
    """The n terms with the highest summed IDF across all docs (corpus-wide fallback keywords)."""
    term_scores: dict[str, float] = defaultdict(float)
    for doc_term_freq in bm25.tf_per_doc:
        for term in doc_term_freq.keys():
            term_scores[term] += bm25.idf(term)
    # Secondary alphabetical key so ties at the top-n cutoff resolve the same
    # way every run (dict/set iteration order is hash-seed dependent).
    return [term for term, _ in
            sorted(term_scores.items(), key=lambda item: (-item[1], item[0]))[:n]]


# ---------------------------------------------------------------------------
# Document scoring
# ---------------------------------------------------------------------------

def topm_chunk_representativeness(bm25: BM25, doc_tokens: list[str],
                                  query: list[str], chunk_words: int,
                                  top_m: int) -> float:
    """
    Mean of the top-m fixed-window BM25 scores (avgdl pinned to the window
    size). Ranks a document on the density of its best material: whole-doc
    scoring rewards keyword coverage, which grows with length.

    The divisor is ALWAYS top_m (missing windows count as zero) so every
    document is graded on the same scale — dividing by the actual window
    count would let a 150-word fragment with one dense window outrank a
    paper with m strong sections.
    """
    if not doc_tokens:
        return 0.0
    windows = [doc_tokens[window_start:window_start + chunk_words]
               for window_start in range(0, len(doc_tokens), chunk_words)]
    # Drop a trailing fragment window when the doc has full windows to
    # spare — a 30-token tail scores erratically under chunk-calibrated
    # normalization.
    if len(windows) > 1 and len(windows[-1]) < chunk_words // 4:
        windows.pop()
    window_scores = sorted(
        (bm25.score_tokens(window, query, avgdl=float(chunk_words)) for window in windows),
        reverse=True,
    )[:top_m]
    return sum(window_scores) / top_m


def novelty_score(bm25: BM25, doc_tokens: list[str]) -> float:
    """
    Average IDF of a document's unique vocabulary — counterweight to
    representativeness, which rewards typicality. Hapax terms (df == 1) are
    excluded: on scanned corpora they're OCR artifacts, not vocabulary.
    """
    scoreable_terms = [term for term in set(doc_tokens) if bm25.df.get(term, 0) >= 2]
    if not scoreable_terms:
        return 0.0
    return sum(bm25.idf(term) for term in scoreable_terms) / len(scoreable_terms)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def percentile_normalize(scores: dict[str, float]) -> dict[str, float]:
    """
    Percentile-rank normalization to [0, 1] (max-normalization compresses
    heavy-tailed signals like PageRank toward zero). Ties get the average of
    the ranks they span, compared at 12 decimals so solver float noise
    doesn't split a genuine tie.
    """
    if not scores:
        return {}
    ordered = sorted(scores.items(), key=lambda item: item[1])
    doc_count = len(ordered)
    if doc_count == 1:
        return {ordered[0][0]: 1.0}

    normalized: dict[str, float] = {}
    tie_start = 0
    while tie_start < doc_count:
        tie_end = tie_start
        while (tie_end + 1 < doc_count
               and round(ordered[tie_end + 1][1], 12) == round(ordered[tie_start][1], 12)):
            tie_end += 1
        avg_rank = (tie_start + tie_end) / 2
        for tied_idx in range(tie_start, tie_end + 1):
            normalized[ordered[tied_idx][0]] = avg_rank / (doc_count - 1)
        tie_start = tie_end + 1
    return normalized


# ---------------------------------------------------------------------------
# Graph scoring
# ---------------------------------------------------------------------------

def compute_pagerank(adjacency: dict[str, set[str]], doc_ids: list[str],
                     damping: float = 0.85) -> dict[str, float]:
    """
    PageRank over the citation graph. Edge src -> tgt means "src cites
    tgt": rank flows through INCOMING edges, so being cited drives score
    and long reference lists don't. Uniform scores on an edgeless graph.
    """
    citation_graph = nx.DiGraph()
    citation_graph.add_nodes_from(doc_ids)
    for source_id, target_ids in adjacency.items():
        for target_id in target_ids:
            if target_id in citation_graph:
                citation_graph.add_edge(source_id, target_id)

    if citation_graph.number_of_edges() == 0:
        return {doc_id: 1.0 / len(doc_ids) for doc_id in doc_ids}

    return nx.pagerank(citation_graph, alpha=damping)

