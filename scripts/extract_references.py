"""
extract_references.py — dump every reference cited by the PDFs in
tests/test-input-synthesis to JSON.

Nothing is reparsed here: each PDF goes through the sapphire extract stage's
own convert_document() (docling for the document, GROBID for header +
bibliography), and this script only reshapes its parsedReferences into one row
per (reference, citing paper):

  id             1-based counter over every reference found, all PDFs together
  referenceName  reference title (GROBID citation model), raw string when the
                 model found authors but no title
  authors        authors of the REFERENCED paper
  referencedBy   title of the citing PDF, its filename when no title parsed

GROBID must be running: per-reference authors come from its citation model,
and docling's own extract_references() yields raw strings with no author
split, so an unreachable server exits 1 rather than writing author-less rows.

  docker run -d --name grobid --restart unless-stopped -p 8070:8070 \
    -e JDK_JAVA_OPTIONS="-XX:-UseContainerSupport" lfoppiano/grobid:0.8.0

Run: python scripts/extract_references.py [--input DIR] [--output FILE]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[1]
SAPPHIRE_DIR = ROOT / "backend" / "extraction" / "sapphire"
INPUT_DIR    = ROOT / "tests" / "test-input-synthesis"
OUTPUT_PATH  = INPUT_DIR / "references.json"

# A .pyc dropped into backend/ restarts the dev:all watch server mid-run
# (see scripts/dev.mjs) — same reason pipeline.js sets PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
# extract.py imports extract_utils flat, so its directory must be importable.
sys.path.insert(0, str(SAPPHIRE_DIR))

import extract  # noqa: E402  — the sys.path setup above has to run first

# convert_document picks its converter from the enhance report
# (data/enhanced/<docId>.json), which this script never generates; that default
# is the OCR pipeline, far slower than these digital PDFs need.
extract._choose_converter = lambda doc_meta: extract._converter_digital


def doc_meta_for(pdf_path: Path) -> dict:
    """documents.json-shaped meta for convert_document. docId is the sha256
    prefix the ingest API assigns (backend/routes/documents.js), so a PDF keeps
    the same id here as in a real collection."""
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return {"docId": sha256[:16], "filename": pdf_path.name, "filePath": str(pdf_path)}


def run(input_dir: Path, output_path: Path) -> None:
    pdf_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_paths:
        print(f"[refs] No PDFs in {input_dir}. Nothing to do.", file=sys.stderr)
        sys.exit(1)

    if not extract._grobid_alive():
        print(f"[refs] GROBID unreachable at {extract.GROBID_URL} — per-reference "
              "authors come from its citation model. Start the server and re-run.",
              file=sys.stderr)
        sys.exit(1)

    references = []
    errors     = []

    for pdf_path in pdf_paths:
        print(f"[refs] Processing {pdf_path.name} ...")
        try:
            docling_entry = extract.convert_document(doc_meta_for(pdf_path))
        except Exception as exc:
            print(f"[refs]   ERROR: {exc}", file=sys.stderr)
            errors.append({"filename": pdf_path.name, "error": str(exc)})
            continue

        # GROBID's header model can come back without a title — fall back to the
        # filename so a reference is never attributed to an empty citing paper.
        referenced_by = (docling_entry["metadata"].get("title") or "").strip() or pdf_path.name

        parsed_references = docling_entry["parsedReferences"]
        for parsed_reference in parsed_references:
            references.append({
                # 1-based, in the order references are found across all PDFs
                "id":            len(references) + 1,
                "referenceName": parsed_reference.get("title") or parsed_reference.get("raw") or "",
                "authors":       parsed_reference.get("authors") or [],
                "referencedBy":  referenced_by,
            })
        print(f"[refs]   → {len(parsed_references)} references from {referenced_by}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(references, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[refs] Wrote {len(references)} references from "
          f"{len(pdf_paths) - len(errors)}/{len(pdf_paths)} PDFs to {output_path}")

    if errors:
        print(f"[refs] {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"  {error['filename']}: {error['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract every reference from a directory of PDFs via docling + GROBID")
    parser.add_argument("--input", type=Path, default=INPUT_DIR,
                        help=f"Directory of PDFs to parse (default {INPUT_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"JSON file to write (default {OUTPUT_PATH})")
    args = parser.parse_args()
    run(input_dir=args.input.resolve(), output_path=args.output.resolve())
