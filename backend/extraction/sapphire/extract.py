"""
extract.py

Converts every PDF tracked in data/documents.json into a structured
DoclingDocument.  Digital PDFs go through docling's standard pipeline;
scanned/mixed PDFs use docling's OCR pipeline backed by Tesseract.

GROBID (CRF models) runs alongside docling on the same PDF: docling owns
text/markdown/sections/tables, GROBID owns metadata + references. GROBID is
REQUIRED, not preferred: it is the only source of authors and parsed
references, so extraction raises when the server is unreachable rather than
producing documents that look complete but carry no authors and no citation
edges. Title and abstract still fall back to the document's own section
structure when GROBID's header model leaves them empty.

Orchestration + config only: the pure parsing primitives (sections, tables,
references, TEI/date scanning, docling-label metadata) live in
extract_utils.py; this file owns the converters, the GROBID network
calls, the tunables, and the per-document pipeline glue.

Output: data/doclings.json  — dict keyed by docId, each entry holds:
  text             : full extracted plain text (docling)
  markdown         : structured markdown (docling)
  sections         : [{heading, text}] extracted from the document hierarchy
  tables           : [str] table text, one entry per table
  references       : [str] raw bibliographic reference strings (GROBID)
  parsedReferences : [{title, authors, raw}] structured refs (GROBID CRF) —
                     lets heuristic.py skip its LLM reference-parsing pass
  metadata         : {title, authors, abstract, created} (GROBID header
                     model; structural fallback for title/abstract only —
                     authors come from GROBID alone; abstract falls back to the
                     first 200 words of body text as a last resort — Crossref
                     enrichment overwrites it when a real abstract exists),
                     plus doi added later by doi_regex.js.
                     created is {year, month|null} — GROBID's TEI header date
                     when present, else the earliest plausible date in the
                     first-page text, never later than the current UTC
                     year/month; Crossref's issued date (search_doi.js)
                     overwrites it when available. heuristic.py uses it to
                     reject citation edges that point forward in time.
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone

import requests

# Force UTF-8 console streams: Windows defaults to cp1252, and a non-Latin-1
# char in a print inside the per-document try/except misfiles a converted doc
# as an extraction error.
for _console_stream in (sys.stdout, sys.stderr):
    _reconfigure_stream = getattr(_console_stream, "reconfigure", None)
    if callable(_reconfigure_stream):
        try:
            _reconfigure_stream(encoding="utf-8")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parents[3]
DATA_DIR     = ROOT / os.environ.get("DATA_DIR", "data")
DOCUMENTS_META = DATA_DIR / "documents.json"
DOCLINGS_OUT   = DATA_DIR / "doclings.json"

# OCR_PATH: directory containing the Tesseract binary (e.g. C:\Program Files\Tesseract-OCR\).
# When unset, assumes "tesseract" is already on the system PATH (Linux/Docker).
_ocr_dir = os.environ.get("OCR_PATH", "").strip()
if _ocr_dir:
    _tesseract_exe = "tesseract.exe" if os.name == "nt" else "tesseract"
    TESSERACT_CMD = str(Path(_ocr_dir) / _tesseract_exe)
else:
    TESSERACT_CMD = "tesseract"

METADATA_WORDS   = int(os.environ.get("METADATA_WORDS", "800"))
# Last-resort abstract: first N words of body text when nothing else yielded one.
ABSTRACT_FALLBACK_WORDS = int(os.environ.get("EXTRACT_ABSTRACT_FALLBACK_WORDS", "200"))
# Text-layer probe (_choose_converter): pages sampled and the character count
# that marks a PDF as digital. A scanned document is scanned throughout, so a
# few pages settle it; measured on 192 papers, digital ones yield thousands of
# characters in the first three pages and scans yield essentially none.
TEXT_LAYER_SAMPLE_PAGES = int(os.environ.get("EXTRACT_TEXT_LAYER_PAGES", "3"))
TEXT_LAYER_MIN_CHARS    = int(os.environ.get("EXTRACT_TEXT_LAYER_MIN_CHARS", "200"))
# First-page region scanned for a creation date. Body text is full of in-text
# citation years, so the scan is capped to the front matter.
CREATED_SCAN_CHARS = int(os.environ.get("EXTRACT_CREATED_SCAN_CHARS", "3000"))


# GROBID server (CRF models — run the lightweight image):
#   docker run -d --name grobid --restart unless-stopped -p 8070:8070 \
#     -e JDK_JAVA_OPTIONS="-XX:-UseContainerSupport" lfoppiano/grobid:0.8.0
# The JDK flag works around a cgroup-v2 NullPointerException that crash-loops
# the container's JVM under newer Docker Desktop/WSL2 — keep it if recreating.
# The full grobid/grobid image swaps in deep-learning models; we
# deliberately use the CRF-only image for speed and low memory.
GROBID_URL     = os.environ.get("GROBID_URL", "http://localhost:8070")
GROBID_TIMEOUT = int(os.environ.get("GROBID_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# docling imports + pure parsing primitives (extract_utils)
# ---------------------------------------------------------------------------

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions

from extract_utils import (
    NOT_TITLE_RE,
    METADATA_SKIP,
    TEI_NS,
    clean_boilerplate_title,
    extract_metadata,
    extract_references,
    extract_sections,
    extract_tables,
    scan_created,
    strip_nul,
    tei_persname,
    tei_text,
    valid_created,
)

# DOCLING_PAGE_BATCH: pages docling processes concurrently (its default is 4).
# Each in-flight page holds its rasterized image + model tensors in RAM, so on
# memory-tight machines a large scanned PDF OOMs the C++ preprocessor
# (std::bad_alloc). Set 1 in .env to minimize peak memory; unset to keep
# docling's default.
_page_batch = os.environ.get("DOCLING_PAGE_BATCH", "").strip()
if _page_batch:
    try:
        from docling.datamodel.settings import settings as _docling_settings
        _docling_settings.perf.page_batch_size = max(1, int(_page_batch))
        print(f"[extract] docling page_batch_size={_docling_settings.perf.page_batch_size}")
    except Exception as exc:
        print(f"[extract] could not set DOCLING_PAGE_BATCH: {exc}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Converters (built once, reused across documents)
# ---------------------------------------------------------------------------

def _make_converter(ocr: bool) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = ocr
    if ocr:
        pipeline_options.ocr_options = TesseractCliOcrOptions(
            force_full_page_ocr=False, tesseract_cmd=TESSERACT_CMD)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

_converter_digital = _make_converter(ocr=False)
_converter_ocr     = _make_converter(ocr=True)


def _has_text_layer(pdf_path: str) -> bool:
    """Does this PDF carry extractable text, or is it page images?

    Sampled, not exhaustive: a scanned document is scanned throughout, and a
    250-page book would otherwise cost more to classify than to convert. Uses
    pypdfium2, which docling already depends on, so this adds no dependency and
    costs ~15ms per document (measured: 0.09s across six papers).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:                       # no backend → assume the safe path
        return False
    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception:
        return False
    try:
        characters = 0
        for page_index in range(min(TEXT_LAYER_SAMPLE_PAGES, len(pdf))):
            characters += len(pdf[page_index].get_textpage().get_text_bounded().strip())
            if characters >= TEXT_LAYER_MIN_CHARS:
                return True
        return False
    except Exception:
        return False
    finally:
        pdf.close()


def _choose_converter(doc_meta: dict) -> DocumentConverter:
    """Digital pipeline for PDFs with a text layer, OCR for the rest.

    This used to read a per-document page report from enhance_pdf.js and fall
    back to OCR when none existed. That stage is deprecated and no longer runs,
    so the fallback became unconditional: every digital paper was OCR'd page by
    page, at roughly a minute each and producing nothing but Tesseract "Too few
    characters" noise, because the text was already there to read.
    """
    file_path = doc_meta.get("filePath", "")
    return _converter_digital if _has_text_layer(file_path) else _converter_ocr


# ---------------------------------------------------------------------------
# Structural metadata (fills fields GROBID's header model left empty)
# ---------------------------------------------------------------------------

def _structural_metadata(sections: list[dict]) -> dict | None:
    """Title and abstract from the document's own section structure.

    No model involved. Title/abstract were always found this way — small LLMs
    are unreliable at 'is this the title?' — and the Ollama author pass that
    used to sit here is gone, so GROBID is the only author source.
    """
    if not sections:
        return None

    title: str | None = None
    for section in sections[:7]:
        heading = (section.get("heading") or "").strip()
        if (heading
                and heading.lower().strip() not in METADATA_SKIP
                and not NOT_TITLE_RE.match(heading)):
            title = heading
            break

    abstract: str | None = None
    for section in sections:
        if (section.get("heading") or "").lower().strip().startswith("abstract"):
            abstract = (section.get("text") or "").strip() or None
            break

    if title is None and abstract is None:
        return None
    # No authors: separating a name list from its affiliations is what the LLM
    # was for, and structure alone cannot do it. GROBID's CRF header model is now
    # the only author source, and extract_document raises when it is unavailable,
    # so there is nothing left for a guess to rescue.
    return {"title": title, "authors": [], "abstract": abstract}


# ---------------------------------------------------------------------------
# GROBID — title / authors / abstract / references (CRF models)
# ---------------------------------------------------------------------------
# GROBID consumes the raw PDF over HTTP and returns TEI XML. It replaces the
# regex + LLM metadata/reference parsing as the PRIMARY path; the docling/LLM
# heuristics (extract_utils) survive only as fallbacks for when the server is down.


def _grobid_alive() -> bool:
    try:
        return requests.get(f"{GROBID_URL}/api/isalive", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def _grobid_header(pdf_path: str) -> dict | None:
    """Title/authors/abstract from GROBID's processHeaderDocument (CRF header
    model). Returns None on any transport or parse failure."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            response = requests.post(
                f"{GROBID_URL}/api/processHeaderDocument",
                files={"input": (os.path.basename(pdf_path), pdf_file, "application/pdf")},
                data={"consolidateHeader": "0"},
                # 0.8.x defaults this endpoint to BibTeX — demand TEI XML
                headers={"Accept": "application/xml"},
                timeout=GROBID_TIMEOUT,
            )
        if response.status_code != 200:
            return None
        tei_root = ET.fromstring(response.text)
    except (requests.RequestException, ET.ParseError):
        return None

    raw_title = tei_text(tei_root.find(".//tei:titleStmt/tei:title", TEI_NS))
    title = clean_boilerplate_title(raw_title) if raw_title else None

    authors = []
    for author_el in tei_root.findall(
            ".//tei:sourceDesc//tei:biblStruct//tei:author/tei:persName", TEI_NS):
        author_name = tei_persname(author_el)
        if author_name and author_name not in authors:
            authors.append(author_name)

    abstract = tei_text(tei_root.find(".//tei:profileDesc/tei:abstract", TEI_NS)) or None

    # Publication date: prefer the machine-readable @when attribute; fall back
    # to scanning the element's text ('June 2017'). scan_created applies the
    # same future-date cap as the first-page fallback.
    created = None
    date_el = tei_root.find(".//tei:publicationStmt/tei:date", TEI_NS)
    if date_el is None:
        date_el = tei_root.find(
            ".//tei:sourceDesc//tei:biblStruct//tei:imprint/tei:date", TEI_NS)
    if date_el is not None:
        when = date_el.get("when") or ""
        when_match = re.match(r"(\d{4})(?:-(\d{2}))?", when)
        if when_match and valid_created(
                int(when_match.group(1)),
                int(when_match.group(2)) if when_match.group(2) else None):
            created = {"year": int(when_match.group(1)),
                       "month": int(when_match.group(2)) if when_match.group(2) else None}
        else:
            created = scan_created(tei_text(date_el))

    if title is None and not authors and abstract is None:
        return None
    return {"title": title, "authors": authors, "abstract": abstract,
            "created": created}


def _grobid_references(pdf_path: str) -> tuple[list[str], list[dict]] | None:
    """Bibliography via GROBID's processReferences (CRF citation model).
    Returns (raw_strings, parsed) where parsed is [{title, authors, raw}] —
    already structured, so heuristic.py needs no LLM pass. None on failure."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            response = requests.post(
                f"{GROBID_URL}/api/processReferences",
                files={"input": (os.path.basename(pdf_path), pdf_file, "application/pdf")},
                data={"consolidateCitations": "0", "includeRawCitations": "1"},
                headers={"Accept": "application/xml"},
                timeout=GROBID_TIMEOUT,
            )
        if response.status_code != 200:
            return None
        tei_root = ET.fromstring(response.text)
    except (requests.RequestException, ET.ParseError):
        return None

    raw_refs: list[str] = []
    parsed:   list[dict] = []
    for bibl in tei_root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS):
        # Article title lives in <analytic>; for books/theses only <monogr>
        # exists, so fall back to it.
        title_el = (bibl.find("tei:analytic/tei:title", TEI_NS)
                    if bibl.find("tei:analytic", TEI_NS) is not None else None)
        if title_el is None or not tei_text(title_el):
            title_el = bibl.find("tei:monogr/tei:title", TEI_NS)
        title = tei_text(title_el)

        authors = []
        for persname_el in bibl.findall(".//tei:author/tei:persName", TEI_NS):
            author_name = tei_persname(persname_el)
            if author_name and author_name not in authors:
                authors.append(author_name)

        raw_citation = tei_text(bibl.find("tei:note[@type='raw_reference']", TEI_NS))
        if not raw_citation:
            raw_citation = " ".join(part for part in [", ".join(authors), title] if part)

        if raw_citation:
            raw_refs.append(raw_citation)
        if title or authors:
            parsed.append({"title": title, "authors": authors, "raw": raw_citation})

    return raw_refs, parsed


def _grobid_extract(pdf_path: str) -> dict | None:
    """Header + references in one call.

    RAISES when the server is unreachable. GROBID is now the only source of
    authors and structured references, so continuing without it would silently
    produce documents with no authors and no citation edges — a corpus that
    looks complete and quietly cannot support a citation graph. Failing the run
    is recoverable; a plausible-looking empty corpus is not.

    Still returns None when the server is UP but has nothing for this PDF (an
    image-only scan: GROBID does not OCR, so the header comes back empty and
    processReferences 204s). That is one bad document, not a broken pipeline, so
    the structural/docling path handles it.
    """
    if not _grobid_alive():
        raise RuntimeError(
            f"GROBID is not answering at {GROBID_URL}. It is the only source of "
            f"authors and parsed references — start it (docker compose up -d grobid) "
            f"and re-run.")
    header = _grobid_header(pdf_path)
    refs   = _grobid_references(pdf_path)
    if header is None and refs is None:
        print("[extract]   GROBID returned no content (scanned PDF with no text "
              "layer? GROBID does not OCR) — falling back to structural parsing",
              file=sys.stderr)
        return None
    raw_refs, parsed = refs if refs is not None else ([], [])
    return {
        "metadata":         header,
        "references":       raw_refs,
        "parsedReferences": parsed,
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_document(doc_meta: dict) -> dict:
    file_path = doc_meta.get("filePath", "")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # GROBID and docling consume the same PDF in parallel: GROBID runs on its
    # server over HTTP while docling churns locally in this process.
    converter = _choose_converter(doc_meta)
    with ThreadPoolExecutor(max_workers=1) as pool:
        grobid_future = pool.submit(_grobid_extract, file_path)
        conversion = converter.convert(file_path)
        grobid = grobid_future.result()

    doc = conversion.document

    full_text = doc.export_to_text()
    sections  = extract_sections(doc)

    # Metadata: GROBID first, merged PER FIELD — a partial GROBID header still
    # gets its gaps filled by the structural/LLM path, then by docling labels.
    grobid_meta = (grobid or {}).get("metadata") or {}
    title    = grobid_meta.get("title")
    authors  = grobid_meta.get("authors") or []
    abstract = grobid_meta.get("abstract")

    if not (title and authors and abstract):
        structural_meta = _structural_metadata(sections) or {}
        title    = title    or structural_meta.get("title")
        abstract = abstract or structural_meta.get("abstract")
    if not (title and authors and abstract):
        docling_meta = extract_metadata(doc)
        title    = title    or docling_meta.get("title")
        authors  = authors  or docling_meta.get("authors") or []
        abstract = abstract or docling_meta.get("abstract")

    # Last-resort abstract: first 200 words of body text (Crossref enrichment
    # overwrites it later). Title deliberately gets NO body-text fallback —
    # citation matching is by containment, and a snippet posing as a title
    # would fabricate edges.
    if not abstract:
        body_snippet = " ".join(full_text.split()[:ABSTRACT_FALLBACK_WORDS])
        if body_snippet:
            abstract = body_snippet
            print(f"[extract]   no abstract found — using first "
                  f"{ABSTRACT_FALLBACK_WORDS} words of body text", file=sys.stderr)

    # Created date: GROBID TEI date, else first-page scan (NOT the whole body
    # — in-text citation years would date every paper by its oldest reference).
    created = grobid_meta.get("created") or scan_created(full_text[:CREATED_SCAN_CHARS])

    metadata = {"title": title, "authors": authors, "abstract": abstract,
                "created": created}

    refs        = (grobid or {}).get("references") or []
    parsed_refs = (grobid or {}).get("parsedReferences") or []
    if not refs:
        refs = extract_references(doc, sections)

    # strip_nul over the WHOLE record, not per field: NUL comes from the PDF's
    # own font tables and lands anywhere docling put text. Postgres rejects it
    # outright, so it must not survive past extraction (see strip_nul).
    return strip_nul({
        "docId":            doc_meta["docId"],
        "filename":         doc_meta["filename"],
        "filePath":         file_path,
        "extractedAt":      datetime.now(timezone.utc).isoformat(),
        "text":             full_text,
        "markdown":         doc.export_to_markdown(),
        "sections":         sections,
        "tables":           extract_tables(doc),
        "references":       refs,
        "parsedReferences": parsed_refs,
        "metadata":         metadata,
    })


def run(force: bool = False) -> None:
    if not DOCUMENTS_META.exists():
        print(f"[extract] {DOCUMENTS_META} not found — run the ingest API first.", file=sys.stderr)
        sys.exit(1)

    docs_meta = json.loads(DOCUMENTS_META.read_text(encoding='utf-8')).get("documents", {})
    if not docs_meta:
        print("[extract] No documents in documents.json. Nothing to do.")
        return

    existing: dict = {}
    if DOCLINGS_OUT.exists():
        try:
            existing = json.loads(DOCLINGS_OUT.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, ValueError):
            existing = {}

    results   = dict(existing)
    extracted = []
    skipped   = 0
    errors    = []

    for doc_id, doc_meta in docs_meta.items():
        if not force and doc_id in existing:
            print(f"[extract] Skipping {doc_meta['filename']} (already extracted)")
            skipped += 1
            continue

        if doc_meta.get("status") not in ("completed", "pending", None):
            print(f"[extract] Skipping {doc_meta['filename']} (status={doc_meta.get('status')})")
            skipped += 1
            continue

        print(f"[extract] Processing {doc_meta['filename']} ...")
        try:
            results[doc_id] = convert_document(doc_meta)
            extracted.append(doc_id)
            print(f"[extract]   → {len(results[doc_id]['sections'])} sections, "
                  f"{len(results[doc_id]['references'])} references")
        except Exception as exc:
            print(f"[extract]   ERROR: {exc}", file=sys.stderr)
            errors.append({"docId": doc_id, "filename": doc_meta.get("filename"),
                           "error": str(exc)})

    DOCLINGS_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOCLINGS_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[extract] Wrote {len(results)} documents to {DOCLINGS_OUT}")

    # Per-run summary for the API: doclings.json can't distinguish this run's
    # work from historical entries, and failed docs never appear in it.
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "force":       force,
        "extracted":   extracted,
        "skipped":     skipped,
        "errors":      errors,
    }
    (DATA_DIR / "extract_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    if errors:
        print(f"[extract] {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"  {error['filename']}: {error['error']}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract text from PDFs via docling")
    parser.add_argument("--force", action="store_true", help="Re-extract already-processed documents")
    args = parser.parse_args()
    run(force=args.force)
