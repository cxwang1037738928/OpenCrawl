"""
heuristic_utils.py — reusable scoring / graph primitives for heuristic.py

Every function here is pure with respect to configuration: all tunables
arrive as arguments (no module-level constants), so heuristic.py owns the
parameters and this module owns the mechanics. Nothing here reads or
writes files.

Contents:
  Tokenisation      tokenise
  BM25              BM25, top_terms
  Doc scoring       topm_chunk_representativeness, novelty_score
  Normalization     percentile_normalize
  Citation graph    build_connectivity, compute_pagerank
"""

import math
import re
import sys
from collections import Counter, defaultdict

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
# Citation connectivity + PageRank
# ---------------------------------------------------------------------------

def _surname(name: str) -> str:
    """Normalized surname: 'Gomez, Aidan N.' and 'Aidan N. Gomez' → 'gomez'."""
    name = name.strip().lower()
    if "," in name:
        name = name.split(",", 1)[0]
    name_tokens = re.findall(r"[a-zà-öø-ÿ'\-]{2,}", name)
    return name_tokens[-1] if name_tokens else ""


def _norm_title(title: str) -> str:
    """Lowercase, fold punctuation to spaces, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def _norm_doi(doi: str) -> str:
    """Lowercase; strip URL/'doi:' prefix and trailing punctuation."""
    normalized = doi.strip().lower()
    normalized = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", normalized)
    return normalized.strip().rstrip(".")


def _titles_match(corpus_title: str, ref_title: str, min_contained: int) -> bool:
    """Bidirectional containment; the contained side must be at least
    `min_contained` chars so short generic titles ('networks') don't hit
    every longer title containing the word."""
    if corpus_title == ref_title:
        return True
    if corpus_title in ref_title and len(corpus_title) >= min_contained:
        return True
    if ref_title in corpus_title and len(ref_title) >= min_contained:
        return True
    return False


def _index_tokens(norm_title: str) -> set[str]:
    """Informative tokens (no stopwords / 1-2 char tokens) for the inverted
    index; all-stopword titles fall back to their full token set."""
    informative_tokens = {token for token in norm_title.split()
                          if len(token) > 2 and token not in _STOPWORDS}
    return informative_tokens or set(norm_title.split())


def build_connectivity(doclings: dict, min_key_length: int,
                       min_contained_length: int = 15) -> dict[str, set[str]]:
    """
    Directed citation adjacency: source_docId -> target_docIds it cites.
    Two phases, unioned:

      Phase 1 — exact DOI match on crossrefReferences (certain edges).
      Phase 2 — fuzzy title match on GROBID parsedReferences: bidirectional
        containment (_titles_match) + shared author surname when both sides
        have authors.

    There is deliberately NO created-year check. One used to reject a target
    dated more than a year after the citing paper, on the theory that a paper
    cannot cite the future. It cost more than it caught: extract.py's `created`
    comes from a date scanned out of the PDF's front matter, and arXiv re-stamps
    its PDFs, so preprints carry the regeneration date rather than the
    publication date. Measured on this corpus, three cond-mat papers were dated
    2018/2018/2021 for work published in 2002/2003/2006, and that silently
    deleted a correct edge — an exact title match with five shared author
    surnames AND a Crossref-resolved DOI — because the target looked 2 years
    "newer" than the paper citing it. Title + author agreement is far stronger
    evidence than a scraped date, so the date no longer gets a vote.

    Phase-2 candidates come from an exact-title hash join plus an inverted
    token index (>= 2 shared informative tokens, or 1 for single-token
    titles) — never a full O(refs x titles) scan. Containment implies the
    contained side's tokens all appear in the containing side, so the
    token filter can't drop a true match.
    """
    title_lookup: dict[str, str] = {}          # normalized title -> doc_id
    doi_lookup: dict[str, str] = {}
    surnames_of: dict[str, set[str]] = {}      # doc_id -> author surnames

    for doc_id, docling_entry in doclings.items():
        metadata = docling_entry.get("metadata", {})
        norm_title = _norm_title(metadata.get("title") or "")
        if norm_title and len(norm_title) >= min_key_length:
            if norm_title in title_lookup:
                # e.g. arXiv + published version — keep the first, deterministically
                print(f"[heuristic] WARNING: duplicate corpus title {norm_title!r} — "
                      f"keeping {title_lookup[norm_title]}, ignoring {doc_id} as a citation target",
                      file=sys.stderr)
            else:
                title_lookup[norm_title] = doc_id
        surnames_of[doc_id] = {
            surname for surname in (_surname(author) for author in (metadata.get("authors") or [])
                                    if len(author.strip()) >= min_key_length)
            if surname
        }
        norm_doi = _norm_doi(metadata.get("doi") or "")
        if norm_doi:
            doi_lookup[norm_doi] = doc_id

    # Inverted token index over corpus titles (phase-2 candidate generation).
    title_of: dict[str, str] = {target_id: norm_title
                                for norm_title, target_id in title_lookup.items()}
    token_index: dict[str, set[str]] = defaultdict(set)
    single_token_targets: dict[str, set[str]] = defaultdict(set)
    for norm_title, target_id in title_lookup.items():
        title_tokens = _index_tokens(norm_title)
        for token in title_tokens:
            token_index[token].add(target_id)
        if len(title_tokens) == 1:
            # can never reach the 2-shared-token bar — tracked separately
            single_token_targets[next(iter(title_tokens))].add(target_id)

    adjacency: dict[str, set[str]] = {doc_id: set() for doc_id in doclings}

    # Phase 1: exact DOI matching
    for doc_id, docling_entry in doclings.items():
        for reference in (docling_entry.get("crossrefReferences") or []):
            ref_doi = _norm_doi(reference.get("doi") or "")
            if not ref_doi:
                continue
            target_id = doi_lookup.get(ref_doi)
            if target_id and target_id != doc_id:
                adjacency[doc_id].add(target_id)

    # Phase 2: fuzzy title matching
    for doc_id, docling_entry in doclings.items():
        for reference in (docling_entry.get("parsedReferences") or []):
            ref_title = _norm_title(reference.get("title") or "")
            if not ref_title or len(ref_title) < min_key_length:
                continue

            ref_tokens = _index_tokens(ref_title)
            min_shared_tokens = 1 if len(ref_tokens) == 1 else 2
            shared_token_counts = Counter()
            for token in ref_tokens:
                for target_id in token_index.get(token, ()):
                    shared_token_counts[target_id] += 1
            candidates = {target_id for target_id, shared in shared_token_counts.items()
                          if shared >= min_shared_tokens}
            for token in ref_tokens:
                candidates |= single_token_targets.get(token, set())
            exact_match_id = title_lookup.get(ref_title)
            if exact_match_id:
                candidates.add(exact_match_id)

            ref_surnames = {
                surname for surname in (_surname(author)
                                        for author in reference.get("authors", []) if author.strip())
                if surname
            }

            for target_id in candidates:
                if target_id == doc_id:
                    continue
                if not _titles_match(title_of[target_id], ref_title, min_contained_length):
                    continue
                # No extracted authors on the target → accept the title match
                # alone rather than silently dropping the edge.
                target_surnames = surnames_of.get(target_id, set())
                if target_surnames and ref_surnames:
                    if not (ref_surnames & target_surnames):
                        continue
                adjacency[doc_id].add(target_id)

    return adjacency


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