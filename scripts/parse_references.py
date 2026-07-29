"""
parse_references.py — steps 1-2: turn the PDFs in tests/test-input-synthesis
into a reference list.

Two sources, best first:
  Semantic Scholar  when the paper is indexed there its reference list has
                    clean titles, plus the cited paper's arXiv id / DOI, which
                    lets the fetch stage skip searching entirely
  GROBID            otherwise — it parses any PDF, but mislabels author lists
                    as article titles and glues words together often enough to
                    cost matches downstream, so it is the fallback

Writes tests/test-input-synthesis/references.json:
  [{id, referenceName, authors, referencedBy, arxivId, doi, source}]

GROBID is needed for the header title (which identifies the paper to Semantic
Scholar) and for the fallback parse:
  docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0

Run: python scripts/parse_references.py [--input DIR] [--output FILE]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT         = Path(__file__).resolve().parents[1]
SAPPHIRE_DIR = ROOT / "backend" / "extraction" / "sapphire"
INPUT_DIR    = ROOT / "tests" / "test-input-synthesis"
OUTPUT_PATH  = INPUT_DIR / "references.json"

# Force UTF-8 console streams, same guard extract.py opens with: Windows
# defaults to cp1252 and an accented title in a print would kill the run.
for _console_stream in (sys.stdout, sys.stderr):
    _reconfigure_stream = getattr(_console_stream, "reconfigure", None)
    if callable(_reconfigure_stream):
        try:
            _reconfigure_stream(encoding="utf-8")
        except Exception:
            pass

sys.dont_write_bytecode = True          # a .pyc in backend/ restarts the dev:all watcher
sys.path.insert(0, str(SAPPHIRE_DIR))

# extract.py builds its docling converters at import (~8s) even though only its
# GROBID helpers are used here; the alternative is a second copy of the TEI
# parsing, which is exactly what this reuse is meant to avoid.
import extract                          # noqa: E402 — sys.path setup must run first

S2_PAGE  = 100
S2_DELAY = 1.5
TIMEOUT  = 30

session = requests.Session()
session.headers["User-Agent"] = "OpenCrawl/1.0"


def s2_references(title: str) -> list[dict] | None:
    """Semantic Scholar's reference list for a paper title, or None when it
    does not know the paper. Match first, then page through the references."""
    time.sleep(S2_DELAY)
    matched = session.get("https://api.semanticscholar.org/graph/v1/paper/search/match",
                          params={"query": title, "fields": "title,externalIds"},
                          timeout=TIMEOUT)
    if matched.status_code != 200:      # 404 = no match, 429 = throttled
        return None
    matches = matched.json().get("data") or []
    if not matches:
        return None
    paper_id = matches[0].get("paperId")

    references, offset = [], 0
    while paper_id:
        time.sleep(S2_DELAY)
        response = session.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references",
            params={"fields": "title,authors,externalIds", "limit": S2_PAGE, "offset": offset},
            timeout=TIMEOUT)
        if response.status_code != 200:
            break
        page = response.json().get("data") or []
        references.extend(reference.get("citedPaper") or {} for reference in page)
        if len(page) < S2_PAGE:
            break
        offset += S2_PAGE
    return references or None


def run(input_dir: Path, output_path: Path) -> None:
    pdf_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_paths:
        print(f"[parse] No PDFs in {input_dir}.", file=sys.stderr)
        sys.exit(1)
    if not extract._grobid_alive():
        print(f"[parse] GROBID unreachable at {extract.GROBID_URL} — start it and re-run.",
              file=sys.stderr)
        sys.exit(1)

    references: list[dict] = []
    for pdf_path in pdf_paths:
        print(f"[parse] {pdf_path.name}")
        header = extract._grobid_header(str(pdf_path)) or {}
        citing_paper = (header.get("title") or "").strip() or pdf_path.stem

        parsed = s2_references(citing_paper)
        source = "s2"
        if parsed is None:
            grobid_refs = extract._grobid_references(str(pdf_path))
            parsed = (grobid_refs[1] if grobid_refs else []) or []
            source = "grobid"
        print(f"[parse]   {len(parsed)} references via {source} (citing: {citing_paper[:52]})")

        for reference in parsed:
            title = (reference.get("title") or "").strip()
            if not title:
                continue
            external_ids = reference.get("externalIds") or {}
            authors = reference.get("authors") or []
            references.append({
                "id":            len(references) + 1,
                "referenceName": title,
                # S2 gives {name}; GROBID gives plain strings
                "authors":       [author.get("name", "") if isinstance(author, dict) else author
                                  for author in authors],
                "referencedBy":  citing_paper,
                "arxivId":       external_ids.get("ArXiv", "") or "",
                "doi":           external_ids.get("DOI", "") or "",
                "source":        source,
            })

    output_path.write_text(json.dumps(references, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[parse] Wrote {len(references)} references to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse reference lists out of the input PDFs")
    parser.add_argument("--input", type=Path, default=INPUT_DIR,
                        help=f"Directory of source PDFs (default {INPUT_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"references.json to write (default {OUTPUT_PATH})")
    args = parser.parse_args()
    run(input_dir=args.input.resolve(), output_path=args.output.resolve())
