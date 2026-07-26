"""
get_reference_from_arXiv.py — fetch the arXiv PDF of every reference in
tests/test-input-synthesis/references.json (written by extract_references.py).

Each distinct reference title — deduplicated with the citation graph's own
_norm_title (heuristic_utils) — is searched on arXiv, and the first result
whose title clears MIN_COSINE is downloaded to tests/test-synthesis-references.
Anything that clears nothing is recorded in tests/test-input-synthesis/
missing.json, together with the best-scoring title arXiv did return, so a near
miss is distinguishable from a paper that simply isn't on arXiv.

Similarity is a cosine over term counts from the pipeline's tokenise(), the
same tokenizer heuristic.py scores documents with: at 0.95 the two titles share
essentially all of their informative words. The pipeline's MiniLM embedder
(backend/retriever/embedder.js) is JS-only and not callable from here.

arxiv 4.0.0 dropped Result.download_pdf, so the PDF itself is fetched with
requests — already an extraction dependency. Re-running re-downloads.

Run: python scripts/get_reference_from_arXiv.py
       [--references FILE] [--download-dir DIR]
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import arxiv
import requests

ROOT            = Path(__file__).resolve().parents[1]
SAPPHIRE_DIR    = ROOT / "backend" / "extraction" / "sapphire"
REFERENCES_PATH = ROOT / "tests" / "test-input-synthesis" / "references.json"
DOWNLOAD_DIR    = ROOT / "tests" / "test-synthesis-references"

# A .pyc dropped into backend/ restarts the dev:all watch server mid-run
# (see scripts/dev.mjs) — same reason pipeline.js sets PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
sys.path.insert(0, str(SAPPHIRE_DIR))

from heuristic_utils import _norm_title, tokenise  # noqa: E402 — sys.path setup first

MAX_RESULTS = 10      # search results scanned per reference
MIN_COSINE  = 0.95    # title similarity a result must beat to count as the same paper
PDF_TIMEOUT = 60


def title_cosine(reference_name: str, result_title: str) -> float:
    """Cosine between two titles over pipeline-tokenised term counts."""
    reference_counts = Counter(tokenise(reference_name))
    result_counts    = Counter(tokenise(result_title))
    shared_terms = set(reference_counts) & set(result_counts)
    if not shared_terms:
        return 0.0
    dot_product    = sum(reference_counts[term] * result_counts[term] for term in shared_terms)
    reference_norm = math.sqrt(sum(count ** 2 for count in reference_counts.values()))
    result_norm    = math.sqrt(sum(count ** 2 for count in result_counts.values()))
    return dot_product / (reference_norm * result_norm)


def download_pdf(result, download_dir: Path) -> Path:
    """Save a result's PDF as <arxiv id>.pdf ('/' in old-style ids → '_')."""
    pdf_path = download_dir / f"{result.get_short_id().replace('/', '_')}.pdf"
    response = requests.get(result.pdf_url, timeout=PDF_TIMEOUT)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)
    return pdf_path


def run(references_path: Path, download_dir: Path) -> None:
    if not references_path.exists():
        print(f"[arxiv] {references_path} not found — run extract_references.py first.",
              file=sys.stderr)
        sys.exit(1)
    references = json.loads(references_path.read_text(encoding="utf-8"))

    # One search per distinct title; every row id sharing that title rides along
    # so missing.json can name all of them.
    ids_by_title:  dict[str, list[int]] = {}
    name_by_title: dict[str, str]       = {}
    for reference in references:
        norm_title = _norm_title(reference["referenceName"])
        if not norm_title:
            continue
        ids_by_title.setdefault(norm_title, []).append(reference["id"])
        name_by_title.setdefault(norm_title, reference["referenceName"])

    download_dir.mkdir(parents=True, exist_ok=True)
    missing_path = references_path.parent / "missing.json"
    client = arxiv.Client(page_size=MAX_RESULTS)

    downloaded = 0
    missing: list[dict] = []
    print(f"[arxiv] {len(ids_by_title)} distinct references from {len(references)} rows")

    for norm_title, reference_ids in ids_by_title.items():
        reference_name = name_by_title[norm_title]
        match, match_score = None, 0.0
        best_title, best_score = "", 0.0
        try:
            for result in client.results(arxiv.Search(query=reference_name,
                                                      max_results=MAX_RESULTS)):
                score = title_cosine(reference_name, result.title)
                if score > best_score:
                    best_title, best_score = result.title, score
                if score > MIN_COSINE:
                    match, match_score = result, score
                    break
            pdf_path = download_pdf(match, download_dir) if match and match.pdf_url else None
        except Exception as exc:
            missing.append({"ids": reference_ids, "referenceName": reference_name,
                            "bestMatch": best_title, "bestScore": round(best_score, 4),
                            "error": str(exc)})
            print(f"[arxiv] error {reference_name[:60]}: {exc}", file=sys.stderr)
            continue

        if pdf_path is None:
            missing.append({"ids": reference_ids, "referenceName": reference_name,
                            "bestMatch": best_title, "bestScore": round(best_score, 4)})
            print(f"[arxiv] miss  {reference_name[:60]} (best {best_score:.2f})")
            continue

        downloaded += 1
        print(f"[arxiv] ok    {reference_name[:60]} → {pdf_path.name} ({match_score:.2f})")

    missing_path.write_text(json.dumps(missing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[arxiv] Downloaded {downloaded} PDF(s) to {download_dir}")
    print(f"[arxiv] Wrote {len(missing)} unmatched reference(s) to {missing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the arXiv PDF of every reference in references.json")
    parser.add_argument("--references", type=Path, default=REFERENCES_PATH,
                        help=f"references.json to read (default {REFERENCES_PATH})")
    parser.add_argument("--download-dir", type=Path, default=DOWNLOAD_DIR,
                        help=f"Directory PDFs are saved to (default {DOWNLOAD_DIR})")
    args = parser.parse_args()
    run(references_path=args.references.resolve(), download_dir=args.download_dir.resolve())
