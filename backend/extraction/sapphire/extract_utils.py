"""
extract_utils.py — pure parsing / formatting primitives for extract.py

Every function here turns already-fetched input — a docling document, a TEI
element, or a text blob — into plain Python data. Nothing here makes a network
call, reads documents.json, or writes doclings.json: extract.py owns the
converters, the GROBID/Ollama calls and the config; this module owns the
mechanics of turning their output into sections, tables, references, metadata
and dates.

Contents:
  Sections / tables   extract_sections, extract_tables
  JSON / title clean  parse_json, clean_boilerplate_title
  docling metadata    extract_metadata
  References          extract_references (+ tiered helpers, regex splitter)
  TEI / dates         tei_text, tei_persname, valid_created, scan_created
"""

import json
import os
import re
from datetime import datetime, timezone

from docling_core.types.doc.labels import DocItemLabel


# ---------------------------------------------------------------------------
# Sections + tables
# ---------------------------------------------------------------------------

def extract_sections(doc) -> list[dict]:
    """Walk the document body and collect (heading, body-text, page-range)."""
    sections = []
    current_heading = None
    current_text_parts = []
    current_pages: list[int] = []

    def _flush():
        if current_heading is None and not current_text_parts:
            return
        sections.append({
            "heading": current_heading or "",
            "text": " ".join(current_text_parts),
            # 1-based inclusive [first, last]; None when docling has no prov
            "pages": [min(current_pages), max(current_pages)] if current_pages else None,
        })

    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        text  = (getattr(item, "text", "") or "").strip()
        if not text:
            continue

        prov = getattr(item, "prov", None) or []
        page = getattr(prov[0], "page_no", None) if prov else None

        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            _flush()
            current_heading = text
            current_text_parts = []
            current_pages = [page] if page else []
        else:
            current_text_parts.append(text)
            if page:
                current_pages.append(page)

    _flush()
    return sections


def extract_tables(doc) -> list[dict]:
    """[{text, page}] — page is 1-based, None without provenance.
    (Older extracts stored plain strings; chunker.js accepts both.)"""
    tables = []
    for table in doc.tables:
        try:
            table_text = table.export_to_markdown()
        except Exception:
            table_text = str(table)
        provenance = getattr(table, "prov", None) or []
        page = getattr(provenance[0], "page_no", None) if provenance else None
        tables.append({"text": table_text, "page": page})
    return tables


# ---------------------------------------------------------------------------
# Sanitizing
# ---------------------------------------------------------------------------

def strip_nul(value):
    """Recursively remove NUL (U+0000) from every string in a nested structure.

    Postgres cannot store NUL in `text` or `jsonb` — an insert fails outright
    with `22P05: unsupported Unicode escape sequence`, which killed the ingest
    of a whole extraction batch. This is not a Prisma quirk; it is a hard
    Postgres limitation, so the character can never be allowed downstream.

    NULs get here from the PDF, not from us: some fonts map a glyph to U+0000,
    and docling faithfully decodes it. Measured on this corpus, 3 of 20
    documents carried 180 of them, all where an en-dash belonged —
    '[1 \\x00 4]' for a '[1-4]' citation range.

    They are DROPPED rather than replaced: the character conveys nothing, and
    substituting the dash we happen to have seen here would be guessing at a
    glyph that differs per font. Applied to the whole docling record, so the
    text, markdown, sections, tables and metadata that feed Postgres, the
    chunker and the graph prompts are all clean at the source.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {key: strip_nul(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_nul(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# JSON + title cleaning
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)


def parse_json(llm_response: str):
    """Strip markdown fences and parse JSON; return None on failure."""
    try:
        return json.loads(_CODE_FENCE.sub("", llm_response).strip())
    except Exception:
        return None


METADATA_SKIP = frozenset({
    "abstract", "introduction", "background", "methods", "results",
    "discussion", "conclusion", "conclusions", "references", "bibliography",
    "acknowledgements", "acknowledgments", "appendix", "supplementary",
    "related work", "future work", "limitations", "keywords", "overview",
    "summary", "notation", "funding", "license", "orcid",
})

# Non-title heading patterns: copyright lines, publisher banners, etc.
NOT_TITLE_RE = re.compile(
    r'(?i)^('
    r'provided proper attribution|permission to reproduce|all rights reserved|'
    r'published by|journal of|proceedings of|transactions on|'
    r'vol\.|volume\s+\d|issue\s+\d|doi:|arxiv:'
    r')',
)


def clean_boilerplate_title(title: str) -> str | None:
    """Strip leading copyright/publisher boilerplate GROBID sometimes glues
    onto a title — a polluted title poisons chunk embeddings, category
    vectors, and citation matching. Returns None when nothing survives."""
    title_sentences = re.split(r'(?<=[.!?])\s+', title.strip())
    while title_sentences and NOT_TITLE_RE.match(title_sentences[0]):
        title_sentences.pop(0)
    cleaned = " ".join(title_sentences).strip()
    return cleaned or None


# ---------------------------------------------------------------------------
# Metadata fallback from docling labels (used when Ollama is unavailable)
# ---------------------------------------------------------------------------

def extract_metadata(doc) -> dict:
    """Extract title/authors/abstract from docling item labels with header fallbacks."""
    meta = {"title": None, "authors": [], "abstract": None}
    section_count = 0
    _SKIP = frozenset({
        "abstract", "introduction", "background", "methods", "results",
        "discussion", "conclusion", "conclusions", "references", "bibliography",
        "acknowledgements", "acknowledgments", "appendix", "supplementary",
        "related work", "future work", "limitations", "orcid", "license",
        "terms", "funding", "keywords", "overview", "notation", "summary",
    })
    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        text  = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        if label == DocItemLabel.TITLE and meta["title"] is None:
            meta["title"] = text
        if label == DocItemLabel.SECTION_HEADER:
            section_count += 1
            heading_lower = text.lower().strip()
            if meta["title"] is None and heading_lower not in _SKIP and len(text.split()) <= 15:
                meta["title"] = text
            elif (section_count <= 6 and not meta["authors"]
                  and 1 <= len(text.split()) <= 5
                  and heading_lower not in _SKIP
                  and all(re.match(r'^[A-Z][a-zA-Z\-\.]*$', word) for word in text.split())):
                meta["authors"].append(text)
            if heading_lower.startswith("abstract"):
                meta["abstract"] = ""
        elif meta["abstract"] == "" and label not in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            meta["abstract"] = text
    return meta


# ---------------------------------------------------------------------------
# References — shared constants
# ---------------------------------------------------------------------------

# Shared with heuristic.py / kg_graph.py / regex_utils.js via one env var so the
# four copies can't drift apart.
_REF_SECTION_HEADINGS = frozenset(
    heading.strip().lower()
    for heading in os.environ.get(
        "PIPELINE_REF_HEADINGS",
        "references,bibliography,works cited,literature cited,citations").split(",")
    if heading.strip()
)

# Caption lines that leak into the refs section from PDF text extraction.
_REF_CAPTION_RE = re.compile(r'(?im)^(?:Figure|Table|Scheme|Chart|Fig\.?|Eq\.?)\s*\d+\s*[:.]')
# Spurious glued double-numbering left after marker stripping, e.g. "18.Sierra".
_REF_LEADING_NUM_RE = re.compile(r'^\d{1,3}\.(?=\S)')
# Leading numbered/bracketed marker on an individual entry, e.g. "[3] ", "(3) ", "3. ".
_REF_MARKER_PREFIX_RE = re.compile(r'^(?:[-–—•*]\s+)?(?:\[\d+\]|\(\d+\)|\d+[a-zA-Z]?\.)\s+')

# Docling labels that are never bibliography content — dropped during the
# section-walk fallback (running headers/footers, captions, footnotes, ...).
_REF_DROP_LABELS = frozenset({
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.TABLE,
})


def _norm_heading(heading: str) -> str:
    """Normalize a heading for refs-section matching: lowercase, strip
    leading numbering ('7. References', 'VII. References') and trailing
    punctuation."""
    normalized = heading.lower().strip()
    normalized = re.sub(r'^[\divxlc]+[\.\)]?\s+', '', normalized)   # '7. ' / 'vii. ' / '7) '
    return normalized.rstrip(' .:')


def _is_ref_heading(heading: str) -> bool:
    return _norm_heading(heading) in _REF_SECTION_HEADINGS


def _finalize_entries(entries: list[str]) -> list[str]:
    """Post-process split entries: strip spurious glued leading numbers and
    drop caption-only / empty entries. Applied to every extraction tier."""
    finalized = []
    for entry in entries:
        entry = _REF_LEADING_NUM_RE.sub('', entry.strip()).strip()
        if entry and not _REF_CAPTION_RE.match(entry):
            finalized.append(entry)
    return finalized


# ---------------------------------------------------------------------------
# References — regex fallback (Tier 3, text-dump splitting)
# ---------------------------------------------------------------------------

def _ref_section_from_text(full_text: str) -> str | None:
    """Extract the references section from the full document text, preserving
    per-reference line breaks.
    extract_sections joins items with ' ' (spaces), destroying the line breaks
    between individual references.  This function works directly on
    export_to_text() output so each numbered entry stays on its own line.
    Returns everything after the last known refs heading to end-of-document.
    """
    heading_pattern = '|'.join(
        re.escape(heading) for heading in sorted(_REF_SECTION_HEADINGS, key=len, reverse=True)
    )
    heading_matches = list(re.finditer(rf'(?im)^(?:{heading_pattern})\s*\n', full_text))
    if not heading_matches:
        return None
    return full_text[heading_matches[-1].end():].strip()


def _split_references_regex(text: str) -> list[str]:
    """Split a references-section blob into individual entries.
    Preferred input is text from _ref_section_from_text() which preserves
    newlines; extract_sections() space-joins items so step 2 exists as a
    fallback for that case.
    Tries in order:
    1. Numbered/bracketed markers at LINE START — well-formatted plain text
       ([1] IEEE/ACS, (1) ACS-alt, 1. APA/Vancouver/CSE, 1A. AIP,
        optionally preceded by a bullet/dash e.g. "- [1]")
    2. Inline N. markers — handles space-joined text where line breaks are lost
       (lookbehind prevents matching abbreviations like J., No., and URLs doi:10.)
    3. Inline bracket markers [N] — IEEE style after reflow
    4. Blank-line paragraph separation — APA/MLA/Chicago author-date
    5. Author-date line starts (Lastname, F.)
    6. One reference per physical line

    All strategies run through _finalize_entries(), which drops figure/table
    captions and strips spurious glued leading numbers (e.g. "18.Sierra").
    """
    text = text.strip()
    if not text:
        return []

    def _spans_to_entries(marker_spans):
        entries = []
        for span_idx, (_, marker_end) in enumerate(marker_spans):
            entry_end = (marker_spans[span_idx + 1][0]
                         if span_idx + 1 < len(marker_spans) else len(text))
            body = text[marker_end:entry_end].strip()
            if not body:
                continue
            # split off embedded caption lines so they don't pollute a reference
            entry_lines = []
            for raw_line in body.split('\n'):
                line = raw_line.strip()
                if _REF_CAPTION_RE.match(line):   # caption -> flush current, drop caption
                    if entry_lines:
                        entries.append(' '.join(entry_lines).strip())
                        entry_lines = []
                    continue
                entry_lines.append(line)
            if entry_lines:
                entries.append(' '.join(entry_lines).strip())
        return entries

    # 1. Line-start markers: [1], (1), 1., 1A. — optional leading bullet/dash "- [1]"
    marker_spans = [(match.start(), match.end()) for match in
                    re.finditer(r'(?m)^[ \t]*(?:[-–—•*]\s+)?'
                                r'(?:\[\d+\]|\(\d+\)|\d+[a-zA-Z]?\.)\s+', text)]
    if len(marker_spans) >= 2:
        entries = _spans_to_entries(marker_spans)
        if entries:
            return _finalize_entries(entries)

    # 2. Inline N. — space-joined paragraph (e.g. from extract_sections)
    #    (?<![a-zA-Z:]) avoids J., No., Fig., doi:10.
    #    (?=[A-Z\("[]) avoids matching volume/page numbers before lowercase
    marker_spans = [(match.start(), match.end()) for match in
                    re.finditer(r'(?<![a-zA-Z:])\d{1,3}\.\s+(?=[A-Z\("[])', text)]
    if len(marker_spans) >= 2:
        entries = _spans_to_entries(marker_spans)
        if entries:
            return _finalize_entries(entries)

    # 3. Inline bracket markers [N]
    marker_spans = [(match.start(), match.end())
                    for match in re.finditer(r'\[\d+\]\s+', text)]
    if len(marker_spans) >= 2:
        entries = _spans_to_entries(marker_spans)
        if entries:
            return _finalize_entries(entries)

    # 4. Blank-line paragraph separation
    paragraphs = [paragraph.strip() for paragraph in re.split(r'\n[ \t]*\n', text)
                  if len(paragraph.strip()) > 15]
    if len(paragraphs) >= 2:
        return _finalize_entries(paragraphs)

    # 5. Author-date: each entry starts on a new line with Lastname, First
    entry_starts = [match.start() for match in
                    re.finditer(r'(?m)^[A-Z][a-z\-]{1,25},\s+[A-Z]', text)]
    if len(entry_starts) >= 2:
        entries = []
        for start_idx, entry_start in enumerate(entry_starts):
            entry_end = (entry_starts[start_idx + 1]
                         if start_idx + 1 < len(entry_starts) else len(text))
            body = text[entry_start:entry_end].strip()
            if len(body) > 15:
                entries.append(body)
        if entries:
            return _finalize_entries(entries)

    # 6. One reference per line
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
    return _finalize_entries(lines) if len(lines) >= 2 else ([text] if text else [])


# ---------------------------------------------------------------------------
# References — tiered extraction (structure first, regex last)
# ---------------------------------------------------------------------------

def _clean_structured_entry(text: str) -> str:
    """Clean a single structure-derived entry: collapse internal line breaks
    (wrapped lines within one item) and strip a leading '[3] ' / '3. ' marker."""
    joined = ' '.join(line.strip() for line in text.splitlines() if line.strip())
    return _REF_MARKER_PREFIX_RE.sub('', joined).strip()


def _refs_from_labels(doc) -> list[str]:
    """Tier 1: docling's layout model labeled individual bibliography entries
    as DocItemLabel.REFERENCE.  Cleanest path — no splitting heuristics."""
    refs = []
    for doc_item in getattr(doc, "texts", []):
        if getattr(doc_item, "label", None) == DocItemLabel.REFERENCE:
            item_text = (getattr(doc_item, "text", "") or "").strip()
            if item_text:
                refs.append(_clean_structured_entry(item_text))
    return _finalize_entries(refs)


def _refs_from_section_walk(doc) -> list[str]:
    """Tier 2: walk body reading order; collect body-layer items between the
    References heading and the next section heading, dropping items whose
    labels mark them as non-bibliographic (captions, headers, footers, ...)."""
    collected = []
    in_refs = False
    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        text  = (getattr(item, "text", "") or "").strip()
        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            if in_refs:
                break                       # next section started — refs done
            in_refs = _is_ref_heading(text)
            continue
        if not in_refs or not text or label in _REF_DROP_LABELS:
            continue
        collected.append(text)

    if not collected:
        return []
    if len(collected) == 1:
        # Single blob — docling merged the entries; delegate to the splitter.
        return _split_references_regex(collected[0])
    # If each collected item still bundles several entries (multi-line),
    # split those; otherwise treat one item = one reference.
    entries = []
    for collected_item in collected:
        item_lines = [line.strip() for line in collected_item.splitlines() if line.strip()]
        if (len(item_lines) > 1
                and sum(bool(_REF_MARKER_PREFIX_RE.match(line)) for line in item_lines) >= 2):
            entries.extend(_split_references_regex(collected_item))
        else:
            entries.append(_clean_structured_entry(collected_item))
    return _finalize_entries(entries)


def _refs_from_text_dump(doc, sections: list[dict] | None = None) -> list[str]:
    """Tier 3: regex over the flat text export (last resort)."""
    full_text = doc.export_to_text()
    ref_section_text = _ref_section_from_text(full_text)
    if not ref_section_text and sections:
        ref_section_text = next(
            (section["text"] for section in reversed(sections)
             if _is_ref_heading(section.get("heading", ""))),
            None,
        )
    return _split_references_regex(ref_section_text) if ref_section_text else []


def extract_references(doc, sections: list[dict] | None = None) -> list[str]:
    """Extract bibliography entries, preferring docling's structure over
    text-dump regex:
      Tier 1 — REFERENCE-labelled items (layout model segmentation)
      Tier 2 — reading-order walk of the References section, label-filtered
      Tier 3 — regex splitting of the flat text export
    A tier's result is accepted only if it yields >= 2 entries, since a
    single 'entry' usually means the tier saw one merged blob."""
    refs = _refs_from_labels(doc)
    if len(refs) >= 2:
        return refs
    refs = _refs_from_section_walk(doc)
    if len(refs) >= 2:
        return refs
    return _refs_from_text_dump(doc, sections)


# ---------------------------------------------------------------------------
# TEI helpers + creation-date scanning (GROBID output → plain data)
# ---------------------------------------------------------------------------

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

_MONTH_NUM = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}

# Years 1600–2099: wide enough for any citable document, narrow enough that
# page numbers and grant IDs rarely collide.
_YEAR = r"(?:1[6-9]|20)\d{2}"


def valid_created(year: int, month: int | None) -> bool:
    """A creation date can't be in the future: reject candidates later than
    the current UTC year/month (a bare year equal to the current year is
    allowed — the month is simply unknown)."""
    now = datetime.now(timezone.utc)
    if year > now.year:
        return False
    if year == now.year and month is not None and month > now.month:
        return False
    return True


def scan_created(text: str) -> dict | None:
    """Closest plausible {year, month|null} date to the current date mentioned in `text`.

    Callers pass the first-page region only (~3000 chars: arXiv stamp,
    'Received:', journal line) — body text is full of in-text citation years
    ('(Hochreiter & Schmidhuber, 1997)') that would make every paper look as
    old as its oldest reference. Future dates are rejected via
    valid_created; among survivors the closest wins, with a known month
    beating an unknown one in the same year.
    """
    candidates: list[tuple[int, int | None]] = []
    # 'June 2017', 'Jun. 2017', 'August 15, 2017' (month before year)
    for match in re.finditer(
            rf"\b([A-Za-z]{{3,9}})\.?\s+(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s+)?({_YEAR})\b",
            text):
        month = _MONTH_NUM.get(match.group(1)[:3].lower())
        if month:
            candidates.append((int(match.group(2)), month))
    # ISO '2017-06' / '2017-06-12'
    for match in re.finditer(rf"\b({_YEAR})-(\d{{2}})\b", text):
        month = int(match.group(2))
        if 1 <= month <= 12:
            candidates.append((int(match.group(1)), month))
    # Bare years (month unknown)
    for match in re.finditer(rf"\b({_YEAR})\b", text):
        candidates.append((int(match.group(1)), None))

    valid_dates = [(year, month) for year, month in candidates
                   if valid_created(year, month)]
    if not valid_dates:
        return None
    # Latest surviving date wins (= closest to now, since future dates are already
    # rejected). An unknown month sorts to 0 so a known month still beats it in
    # the same year — with `or 12` a bare '2017' would outrank 'June 2017'.
    year, month = max(valid_dates, key=lambda date: (date[0], date[1] or 0))
    return {"year": year, "month": month}


def tei_text(element) -> str:
    """Flattened text content of a TEI element (None-safe)."""
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def tei_persname(persname_el) -> str:
    """'First Middle Last' from a TEI <persName> element."""
    forenames = [tei_text(forename)
                 for forename in persname_el.findall("tei:forename", TEI_NS)]
    surname   = tei_text(persname_el.find("tei:surname", TEI_NS))
    name_parts = [part for part in forenames + [surname] if part]
    return " ".join(name_parts)
