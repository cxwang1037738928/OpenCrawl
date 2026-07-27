"""
get_arxiv_references.py — every reference of the two review papers, fetched as
a PDF wherever an open copy exists.

Reference lists come from metadata, never from parsing the PDFs: GROBID's parse
of these two mislabels author lists as article titles, which poisons every
downstream query. Semantic Scholar returns the same bibliographies with clean
titles and, for a quarter of them, the cited paper's arXiv id.

Crossref cannot supply the lists — arXiv DOIs (10.48550/arXiv.*) are
DataCite-registered, so /works/10.48550/arXiv.2508.03278 is a 404 for both
papers, and OpenAlex holds the works with an empty referenced_works.

Per reference, in order:
  externalIds.ArXiv  → fetch that id directly, no search at all
  no id              → ti:"<title>" AND au:<surname> on arXiv. The fielded
                       syntax returns a known paper at rank 1, where an
                       unquoted query ORs its terms and floods; it is an exact
                       phrase match, so a damaged title yields 0 results
  no arXiv copy      → DOI → OpenAlex + Unpaywall open-access locations,
                       landing pages rewritten to their real download URLs

Downloads are verified by %PDF magic bytes — publisher links routinely answer
a .pdf URL with an HTML paywall page.

Everything stays under scripts/: arxiv_pdfs/ (arXiv and other open copies
alike) and arxiv_references.json, which records every reference with its
status — downloaded / download_failed / no_oa_pdf / not_found.

CROSSREF_MAILTO (.env) joins the OpenAlex/Unpaywall polite pools.

Run: python scripts/get_arxiv_references.py [--source ARXIV_ID ...] [--resume]
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import arxiv
import requests

SCRIPTS_DIR  = Path(__file__).resolve().parent
ROOT         = SCRIPTS_DIR.parent
SAPPHIRE_DIR = ROOT / "backend" / "extraction" / "sapphire"
DOWNLOAD_DIR = SCRIPTS_DIR / "arxiv_pdfs"
REPORT_PATH  = SCRIPTS_DIR / "arxiv_references.json"

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

from heuristic_utils import tokenise    # noqa: E402 — sys.path setup must run first

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")          # CROSSREF_MAILTO lives there
except ImportError:
    pass

# The two reviews being parsed, by arXiv id.
SOURCE_PAPERS = ["2508.03278", "2503.18975"]

CROSSREF_MAILTO = os.environ.get("CROSSREF_MAILTO", "").strip()
MIN_COSINE    = 0.90   # title similarity an arXiv search hit must clear
S2_PAGE       = 100    # max references per Semantic Scholar page
S2_DELAY      = 1.5    # unauthenticated S2 throttles hard
TIMEOUT       = 30
RETRIES       = 3      # attempts per API call before the reference errors out
RETRY_DELAY   = 5

session = requests.Session()
session.headers["User-Agent"] = (f"OpenCrawl/1.0 (mailto:{CROSSREF_MAILTO})"
                                 if CROSSREF_MAILTO else "OpenCrawl/1.0")


def title_cosine(reference_title: str, candidate_title: str) -> float:
    """Cosine between two titles over pipeline-tokenised term counts."""
    reference_counts = Counter(tokenise(reference_title))
    candidate_counts = Counter(tokenise(candidate_title))
    shared_terms = set(reference_counts) & set(candidate_counts)
    if not shared_terms:
        return 0.0
    dot_product    = sum(reference_counts[term] * candidate_counts[term] for term in shared_terms)
    reference_norm = math.sqrt(sum(count ** 2 for count in reference_counts.values()))
    candidate_norm = math.sqrt(sum(count ** 2 for count in candidate_counts.values()))
    return dot_product / (reference_norm * candidate_norm)


def get_with_retry(url: str, params: dict) -> requests.Response:
    """GET with backoff over 429s AND dropped connections: an API stops
    accepting connections outright once a client bursts, which a status-code
    check never sees."""
    last_error = ""
    for attempt in range(RETRIES):
        try:
            response = session.get(url, params=params, timeout=TIMEOUT)
            if response.status_code != 429:
                return response
            last_error = "429"
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"{url} unreachable after {RETRIES} attempts ({last_error})")


# ---------------------------------------------------------------------------
# Reference lists + arXiv lookup
# ---------------------------------------------------------------------------

def s2_references(source_id: str) -> list[dict]:
    """Every reference Semantic Scholar lists for an arXiv paper."""
    references, offset = [], 0
    while True:
        time.sleep(S2_DELAY)
        response = session.get(
            f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{source_id}/references",
            params={"fields": "title,authors,externalIds", "limit": S2_PAGE, "offset": offset},
            timeout=TIMEOUT)
        if response.status_code != 200:
            print(f"[refs] S2 returned {response.status_code} for {source_id} at offset {offset}",
                  file=sys.stderr)
            break
        page = response.json().get("data") or []
        references.extend(reference.get("citedPaper") or {} for reference in page)
        if len(page) < S2_PAGE:
            break
        offset += S2_PAGE
    return references


def arxiv_id_by_search(title: str, authors: list[str], client: arxiv.Client) -> tuple[str, float]:
    """Find a title on arXiv with the fielded syntax → (arxiv id, cosine).

    ti: is an exact phrase match, so punctuation is stripped first and the
    author is only added when it looks like a real surname — a mangled one
    silently zeroes the query. Falls back to ti: alone.
    """
    clean_title = re.sub(r"[^A-Za-z0-9 ]", " ", title).strip()
    if not clean_title:
        return "", 0.0
    surname = authors[0].split()[-1] if authors and authors[0].split() else ""
    queries = [f'ti:"{clean_title}" AND au:{surname}'] if len(surname) > 2 else []
    queries.append(f'ti:"{clean_title}"')

    for query in queries:
        try:
            results = list(client.results(arxiv.Search(query=query, max_results=5)))
        except Exception:
            continue
        for result in results:
            score = title_cosine(title, result.title)
            if score >= MIN_COSINE:
                return result.get_short_id(), score
    return "", 0.0


# ---------------------------------------------------------------------------
# Open-access lookup for references with no arXiv copy
# ---------------------------------------------------------------------------

def openalex_pdf_urls(doi: str) -> list[str]:
    """Every open-access PDF URL OpenAlex knows for a DOI, best location first.
    A list, not one URL: the best location is often the publisher's own copy,
    and ACS/RSC/SSRN answer 403 to any script — a repository mirror listed
    beside it usually does not."""
    params = {"mailto": CROSSREF_MAILTO} if CROSSREF_MAILTO else {}
    response = get_with_retry(f"https://api.openalex.org/works/doi:{doi}", params)
    if response.status_code != 200:
        return []
    work = response.json()
    candidates = [(work.get("best_oa_location") or {}).get("pdf_url"),
                  *((location or {}).get("pdf_url") for location in work.get("locations") or []),
                  work.get("open_access", {}).get("oa_url")]
    return [candidate for candidate in candidates if candidate]


def unpaywall_pdf_urls(doi: str) -> list[str]:
    """OA locations Unpaywall lists for a DOI. Its crawl differs from
    OpenAlex's, so it often knows a repository copy OpenAlex does not."""
    if not CROSSREF_MAILTO:            # the API requires a contact address
        return []
    response = get_with_retry(f"https://api.unpaywall.org/v2/{doi}",
                              {"email": CROSSREF_MAILTO})
    if response.status_code != 200:
        return []
    locations = response.json().get("oa_locations") or []
    return [url for url in (location.get("url_for_pdf") for location in locations) if url]


def repository_pdf_urls(pdf_url: str) -> list[str]:
    """Direct file URLs for landing pages that get recorded as PDF links.
    PMC and OSTI publish a predictable download path; figshare needs one API
    call to name the file. Returns [] for anything else."""
    pmc_match = re.search(r"(?:/pmc/articles/|pmc\.ncbi\.nlm\.nih\.gov/articles/)(PMC\d+|\d+)",
                          pdf_url)
    if pmc_match:
        return [f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_match.group(1)}/pdf/"]

    osti_match = re.search(r"osti\.gov/(?:biblio|servlets/purl)/(\d+)", pdf_url)
    if osti_match:
        return [f"https://www.osti.gov/servlets/purl/{osti_match.group(1)}"]

    figshare_match = re.search(r"figshare\.com/articles/[^/]+/[^/]+/(\d+)", pdf_url)
    if figshare_match:
        try:
            response = session.get(
                f"https://api.figshare.com/v2/articles/{figshare_match.group(1)}/files",
                timeout=TIMEOUT)
            return [file_entry["download_url"] for file_entry in response.json()
                    if file_entry.get("download_url")]
        except Exception:
            return []
    return []


def candidate_urls(pdf_urls: list[str]) -> list[str]:
    """Ordered, de-duplicated download candidates: real file URLs first, then
    the DOI resolver, which is a redirect to a landing page rather than a file
    and only ever worth trying when nothing else is left."""
    expanded: list[str] = []
    for pdf_url in pdf_urls:
        for candidate in [*repository_pdf_urls(pdf_url), pdf_url]:
            if candidate and candidate not in expanded:
                expanded.append(candidate)
    return sorted(expanded, key=lambda url: "doi.org/" in url)


def download_pdf(pdf_url: str, pdf_path: Path) -> bool:
    """Save pdf_url when it really is a PDF — publisher links often answer a
    .pdf URL with an HTML paywall page."""
    try:
        response = session.get(pdf_url, timeout=TIMEOUT, allow_redirects=True)
    except requests.RequestException:
        return False
    if response.status_code != 200 or not response.content.startswith(b"%PDF"):
        return False
    pdf_path.write_bytes(response.content)
    return True


# ---------------------------------------------------------------------------
# Per-reference resolution
# ---------------------------------------------------------------------------

def resolve_reference(entry: dict, download_dir: Path, client: arxiv.Client) -> dict:
    """Fetch one reference: arXiv id → arXiv search → open-access by DOI."""
    entry.setdefault("pdfUrl", "")      # reports written before this stage existed
    try:
        if not entry["arxivId"]:
            found_id, score = arxiv_id_by_search(entry["title"], entry["authors"], client)
            if found_id:
                entry.update(arxivId=found_id, matchScore=round(score, 4),
                             source="arxiv-search")

        if entry["arxivId"]:
            file_stem = entry["arxivId"].replace("/", "_")
            pdf_urls  = [f"https://arxiv.org/pdf/{entry['arxivId']}"]
        elif entry["doi"]:
            file_stem = entry["doi"].replace("/", "_")
            pdf_urls  = candidate_urls(openalex_pdf_urls(entry["doi"])
                                       + unpaywall_pdf_urls(entry["doi"]))
            if pdf_urls:
                entry["source"] = "oa-doi"
        else:
            entry["status"] = "not_found"
            print(f"[refs] {entry['status']:15s} {entry['title'][:60]}")
            return entry

        if not pdf_urls:
            entry["status"] = "no_oa_pdf"
        else:
            entry["status"] = "download_failed"
            for pdf_url in pdf_urls:
                entry["pdfUrl"] = pdf_url
                if download_pdf(pdf_url, download_dir / f"{file_stem}.pdf"):
                    entry["status"] = "downloaded"
                    break
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = str(exc)

    print(f"[refs] {entry['status']:15s} {entry['title'][:50]:50s} "
          f"{(entry['arxivId'] or entry['doi'])[:28]:28s} {entry['source']}")
    return entry


def run(source_ids: list[str], download_dir: Path, report_path: Path,
        resume: bool) -> None:
    """--resume re-runs only what a previous report failed to download, using
    the titles and ids already stored rather than re-querying Semantic Scholar."""
    download_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=5, delay_seconds=3)

    if resume:
        if not report_path.exists():
            print(f"[refs] {report_path} not found — nothing to resume.", file=sys.stderr)
            sys.exit(1)
        previous   = json.loads(report_path.read_text(encoding="utf-8"))
        downloaded = [entry for entry in previous if entry["status"] == "downloaded"]
        pending    = [entry for entry in previous if entry["status"] != "downloaded"]
        print(f"[refs] resuming {len(pending)} of {len(previous)} references "
              f"({len(downloaded)} already downloaded)")
    else:
        # Dedup across both reviews — they share part of their bibliography.
        references_by_title: dict[str, dict] = {}
        for source_id in source_ids:
            references = s2_references(source_id)
            print(f"[refs] {source_id}: {len(references)} references from Semantic Scholar")
            for reference in references:
                title = (reference.get("title") or "").strip()
                if not title:
                    continue
                existing = references_by_title.get(title.lower())
                if existing:
                    existing["referencedBy"].append(source_id)
                    continue
                external_ids = reference.get("externalIds") or {}
                references_by_title[title.lower()] = {
                    "title":        title,
                    "authors":      [author.get("name", "")
                                     for author in reference.get("authors") or []],
                    "arxivId":      external_ids.get("ArXiv", "") or "",
                    "doi":          external_ids.get("DOI", "") or "",
                    "referencedBy": [source_id],
                    "matchScore":   1.0 if external_ids.get("ArXiv") else 0.0,
                    "source":       "s2-externalIds" if external_ids.get("ArXiv") else "",
                    "pdfUrl":       "",
                    "status":       "pending",
                }
        downloaded = []
        pending    = list(references_by_title.values())
        print(f"[refs] {len(pending)} distinct references to fetch")

    report = downloaded + [resolve_reference(entry, download_dir, client)
                           for entry in pending]
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[refs] {dict(Counter(entry['status'] for entry in report))}")
    print(f"[refs] PDFs in {download_dir}, report in {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch PDFs for every reference of the two review papers")
    parser.add_argument("--source", nargs="+", default=SOURCE_PAPERS,
                        help=f"arXiv ids of the citing papers (default {' '.join(SOURCE_PAPERS)})")
    parser.add_argument("--download-dir", type=Path, default=DOWNLOAD_DIR,
                        help=f"Directory PDFs are saved to (default {DOWNLOAD_DIR})")
    parser.add_argument("--report", type=Path, default=REPORT_PATH,
                        help=f"Report file (default {REPORT_PATH})")
    parser.add_argument("--resume", action="store_true",
                        help="Re-run only the references --report has not downloaded")
    args = parser.parse_args()
    run(source_ids=args.source, download_dir=args.download_dir.resolve(),
        report_path=args.report.resolve(), resume=args.resume)
