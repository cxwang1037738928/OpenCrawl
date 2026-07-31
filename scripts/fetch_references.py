"""
fetch_references.py — steps 3-4: fetch a copy of every reference in
references.json, trying each source in turn.

Sources are tried in this order, cheapest and most reliable first. The first
one that yields bytes wins, and the report records which:

  1. arXiv id from the reference metadata → arxiv.org/pdf, no search at all
  2. arXiv fielded search  ti:"<title>" AND au:<surname> — an unquoted query
     ORs its terms and floods, and ti: is an exact phrase, so a damaged title
     returns nothing rather than the wrong paper
  3. Crossref  query.bibliographic, title + authors → DOI, when the reference
     arrived without one
  4. OpenAlex  DOI → best_oa_location and every other location it lists
  5. Unpaywall DOI → oa_locations (a different crawl from OpenAlex's)
  6. Semantic Scholar  DOI → openAccessPdf
  7. CORE      DOI → extracted full TEXT. Not a PDF: CORE's downloadUrl and
     /download/pdf both 404 and its recorded repository urls are stale, but
     the text comes straight from the API past every publisher block

PDFs land in <input>/pdf/, CORE text in <input>/txt/, and everything not
retrieved in <input>/missing.json with the reason. fetch_report.json holds
every reference's status so --resume revisits only what is still missing.

Landing pages (PMC, OSTI, figshare) are rewritten to real download URLs, and
every file is verified by its %PDF magic bytes — publisher links routinely
answer a .pdf URL with an HTML paywall page.

CROSSREF_MAILTO and CORE_API_KEY come from .env.

Run: python scripts/fetch_references.py [--references FILE] [--resume]
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

ROOT         = Path(__file__).resolve().parents[1]
SAPPHIRE_DIR = ROOT / "backend" / "extraction" / "sapphire"
INPUT_DIR    = ROOT / "tests" / "test-input-synthesis"
REFERENCES_PATH = INPUT_DIR / "references.json"
PDF_DIR      = INPUT_DIR / "pdf"
TXT_DIR      = INPUT_DIR / "txt"
MISSING_PATH = INPUT_DIR / "missing.json"
REPORT_PATH  = INPUT_DIR / "fetch_report.json"

for _console_stream in (sys.stdout, sys.stderr):
    _reconfigure_stream = getattr(_console_stream, "reconfigure", None)
    if callable(_reconfigure_stream):
        try:
            _reconfigure_stream(encoding="utf-8")
        except Exception:
            pass

sys.dont_write_bytecode = True          # a .pyc in backend/ restarts the dev:all watcher
sys.path.insert(0, str(SAPPHIRE_DIR))

from heuristic_utils import _norm_title, tokenise   # noqa: E402 — sys.path first

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

CROSSREF_MAILTO = os.environ.get("CROSSREF_MAILTO", "").strip()
CORE_API_KEY    = os.environ.get("CORE_API_KEY", "").strip()

MIN_COSINE    = 0.90   # title similarity a search hit must clear to be trusted
CANDIDATES    = 20     # Crossref rows scanned — the right paper is often deep
TIMEOUT       = 30
RETRIES       = 3
RETRY_DELAY   = 5
REQUEST_DELAY = 0.5    # pace Crossref; a no-delay burst gets connections refused
S2_DELAY      = 1.5    # unauthenticated Semantic Scholar throttles hard
CORE_DELAY    = 6      # CORE allows ~10 calls per window
MIN_FULLTEXT  = 2000   # shorter than this is an abstract, not an article

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
# Source 2-3: arXiv search, Crossref resolution
# ---------------------------------------------------------------------------

def arxiv_id_by_search(title: str, authors: list[str], client: arxiv.Client) -> tuple[str, float]:
    """Find a title on arXiv with the fielded syntax → (arxiv id, cosine).
    The author is only added when it looks like a real surname — a mangled one
    silently zeroes the query."""
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


def crossref_doi(reference_name: str, authors: list[str]) -> tuple[str, float]:
    """Best Crossref match → (doi, cosine of ITS title against reference_name).

    query.bibliographic matches a whole citation string, so the authors go into
    the query while scoring stays on the title alone. The title-only query runs
    too, since author tokens sometimes pull Crossref off the paper entirely."""
    queries = [f"{reference_name} {' '.join(authors)}".strip()] if authors else []
    queries.append(reference_name)

    best_doi, best_score = "", 0.0
    for query in queries:
        params = {"query.bibliographic": query, "rows": CANDIDATES, "select": "DOI,title"}
        if CROSSREF_MAILTO:
            params["mailto"] = CROSSREF_MAILTO
        time.sleep(REQUEST_DELAY)
        response = get_with_retry("https://api.crossref.org/works", params)
        if response.status_code != 200:
            continue
        for item in response.json().get("message", {}).get("items", []):
            score = title_cosine(reference_name, (item.get("title") or [""])[0])
            if score > best_score:
                best_doi, best_score = item.get("DOI", ""), score
        if best_score >= MIN_COSINE:
            break
    return best_doi, best_score


# ---------------------------------------------------------------------------
# Sources 4-7: open-access lookups by DOI
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
    """OA locations Unpaywall lists for a DOI — a different crawl from
    OpenAlex's, so it often knows a repository copy OpenAlex does not."""
    if not CROSSREF_MAILTO:            # the API requires a contact address
        return []
    response = get_with_retry(f"https://api.unpaywall.org/v2/{doi}", {"email": CROSSREF_MAILTO})
    if response.status_code != 200:
        return []
    locations = response.json().get("oa_locations") or []
    return [url for url in (location.get("url_for_pdf") for location in locations) if url]


def semantic_scholar_pdf_urls(doi: str) -> list[str]:
    """S2's own open-access link for a DOI — a third crawl again."""
    time.sleep(S2_DELAY)
    try:
        response = session.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                               params={"fields": "openAccessPdf"}, timeout=TIMEOUT)
    except requests.RequestException:
        return []
    if response.status_code != 200:
        return []
    pdf_url = ((response.json().get("openAccessPdf") or {}).get("url") or "")
    return [pdf_url] if pdf_url else []


def core_fulltext(doi: str) -> str:
    """CORE's extracted full text for a DOI ('' when it has none).

    Not a PDF route: CORE's downloadUrl and /download/pdf/<id> both 404 and the
    repository urls it records are stale. The key rides on this request only —
    never on the session, which also talks to publishers."""
    if not CORE_API_KEY:
        return ""
    response = None
    for attempt in range(RETRIES):
        try:
            response = session.get("https://api.core.ac.uk/v3/search/works",
                                   params={"q": f'doi:"{doi.lower()}"', "limit": 1},
                                   headers={"Authorization": f"Bearer {CORE_API_KEY}"},
                                   timeout=TIMEOUT)
        except requests.RequestException:
            return ""
        if response.status_code != 429:
            break
        time.sleep(CORE_DELAY * (attempt + 2))
    if response is None or response.status_code != 200:
        return ""
    results = response.json().get("results") or []
    return (results[0].get("fullText") or "") if results else ""


# ---------------------------------------------------------------------------
# Download plumbing
# ---------------------------------------------------------------------------

def repository_pdf_urls(pdf_url: str) -> list[str]:
    """Direct file URLs for landing pages that get recorded as PDF links.
    PMC and OSTI publish a predictable download path; figshare needs one API
    call to name the file."""
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
    and only worth trying when nothing else is left."""
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
# Per-reference resolution: every source in turn
# ---------------------------------------------------------------------------

def fetch_reference(entry: dict, pdf_dir: Path, txt_dir: Path,
                    client: arxiv.Client) -> dict:
    """Try each source in order until one yields a file."""
    entry.setdefault("pdfUrl", "")
    entry.setdefault("matchScore", 1.0 if entry.get("arxivId") else 0.0)
    entry.setdefault("source", "metadata-arxiv" if entry.get("arxivId") else "")
    try:
        # 2. arXiv search when the metadata carried no id
        if not entry["arxivId"]:
            found_id, score = arxiv_id_by_search(entry["referenceName"],
                                                 entry["authors"], client)
            if found_id:
                entry.update(arxivId=found_id, matchScore=round(score, 4),
                             source="arxiv-search")

        # 1./2. an arXiv id is a direct, unblocked download
        if entry["arxivId"]:
            file_stem = entry["arxivId"].replace("/", "_")
            pdf_urls  = [f"https://arxiv.org/pdf/{entry['arxivId']}"]
        else:
            # 3. resolve a DOI when the reference arrived without one
            if not entry["doi"]:
                doi, score = crossref_doi(entry["referenceName"], entry["authors"])
                if doi and score >= MIN_COSINE:
                    entry.update(doi=doi, matchScore=round(score, 4), source="crossref")
            if not entry["doi"]:
                entry["status"] = "not_found"
                print(f"[fetch] {entry['status']:15s} {entry['referenceName'][:56]}")
                return entry
            # 4-6. every open-access location the three crawls know
            file_stem = entry["doi"].replace("/", "_")
            pdf_urls  = candidate_urls(openalex_pdf_urls(entry["doi"])
                                       + unpaywall_pdf_urls(entry["doi"])
                                       + semantic_scholar_pdf_urls(entry["doi"]))
            if pdf_urls and not entry["source"]:
                entry["source"] = "oa-doi"

        entry["status"] = "no_oa_pdf" if not pdf_urls else "download_failed"
        for pdf_url in pdf_urls:
            entry["pdfUrl"] = pdf_url
            if download_pdf(pdf_url, pdf_dir / f"{file_stem}.pdf"):
                entry["status"] = "downloaded"
                break

        # 7. no host would serve a file — take CORE's extracted text instead
        if entry["status"] != "downloaded" and entry["doi"]:
            article_text = core_fulltext(entry["doi"])
            entry["fullTextChars"] = len(article_text)
            if len(article_text) >= MIN_FULLTEXT:
                (txt_dir / f"{file_stem}.txt").write_text(article_text, encoding="utf-8")
                entry.update(status="fulltext_only", source="core")
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = str(exc)

    print(f"[fetch] {entry['status']:15s} {entry['referenceName'][:48]:48s} "
          f"{(entry.get('arxivId') or entry.get('doi'))[:26]:26s} {entry['source']}")
    return entry


def run(references_path: Path, pdf_dir: Path, txt_dir: Path,
        report_path: Path, missing_path: Path, resume: bool) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=5, delay_seconds=3)

    if resume and report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        done    = [entry for entry in previous
                   if entry["status"] in ("downloaded", "fulltext_only")]
        pending = [entry for entry in previous
                   if entry["status"] not in ("downloaded", "fulltext_only")]
        print(f"[fetch] resuming {len(pending)} of {len(previous)} ({len(done)} already held)")
    else:
        if not references_path.exists():
            print(f"[fetch] {references_path} not found — run parse_references.py first.",
                  file=sys.stderr)
            sys.exit(1)
        references = json.loads(references_path.read_text(encoding="utf-8"))
        # One fetch per distinct title; every citing paper rides along.
        by_title: dict[str, dict] = {}
        for reference in references:
            key = _norm_title(reference["referenceName"])
            if not key:
                continue
            if key in by_title:
                by_title[key]["referencedBy"] = list(
                    {*by_title[key]["referencedBy"], reference["referencedBy"]})
                continue
            by_title[key] = {**reference,
                             "referencedBy": [reference["referencedBy"]],
                             "status": "pending"}
        done, pending = [], list(by_title.values())
        print(f"[fetch] {len(pending)} distinct references from {len(references)} rows")

    report = done + [fetch_reference(entry, pdf_dir, txt_dir, client) for entry in pending]
    report.sort(key=lambda entry: entry.get("id", 0))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    missing = [{"id": entry.get("id"), "referenceName": entry["referenceName"],
                "doi": entry.get("doi", ""), "reason": entry["status"],
                "referencedBy": entry.get("referencedBy", [])}
               for entry in report
               if entry["status"] not in ("downloaded", "fulltext_only")]
    missing_path.write_text(json.dumps(missing, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[fetch] {dict(Counter(entry['status'] for entry in report))}")
    print(f"[fetch] PDFs in {pdf_dir}, text in {txt_dir}")
    print(f"[fetch] {len(missing)} unfetched references in {missing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch every reference in references.json, trying each source in turn")
    parser.add_argument("--references", type=Path, default=REFERENCES_PATH,
                        help=f"references.json to read (default {REFERENCES_PATH})")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--txt-dir", type=Path, default=TXT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--missing", type=Path, default=MISSING_PATH)
    parser.add_argument("--resume", action="store_true",
                        help="Revisit only the references not yet held as a PDF or text")
    args = parser.parse_args()
    run(references_path=args.references.resolve(), pdf_dir=args.pdf_dir.resolve(),
        txt_dir=args.txt_dir.resolve(), report_path=args.report.resolve(),
        missing_path=args.missing.resolve(), resume=args.resume)
