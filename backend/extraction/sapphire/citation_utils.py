"""
citation_utils.py — citation-graph primitives (sapphire only)

Split out of heuristic_utils.py: matching a reference string to another
document in the corpus needs DOIs, GROBID-parsed reference lists and author
surnames, none of which a general document has. The ranking half stayed in
extraction/heuristic_utils.py so crawlers without references can still use it.

Pure with respect to configuration — all tunables arrive as arguments. The
one side effect is a stderr warning on duplicate corpus titles.

Contents:
  Normalization   _surname, _norm_title, _norm_doi
  Matching        _titles_match, _index_tokens
  Graph           build_connectivity
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The ranking half owns the stopword list; importing it rather than copying
# keeps one definition, since a drifted copy would silently change which
# title tokens are indexed and therefore which citation edges are found.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heuristic_utils import _STOPWORDS   # noqa: E402

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
