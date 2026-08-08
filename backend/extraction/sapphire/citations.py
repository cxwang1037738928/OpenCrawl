"""
citations.py — build the corpus citation graph (sapphire only)

Split out of heuristic.py. The graph was always a pure function of doclings —
build_connectivity reads metadata titles/authors/DOIs and GROBID's parsed
reference lists, and touches nothing the ranking computes — but it lived inside
the ranker, so a crawler with no references could not rank without importing
academic code.

Now it is its own stage. Sapphire runs it; topaz does not, and the ranker simply
finds no citations.json and scores on BM25 alone.

Must run AFTER extract: doi_regex.js and search_doi.js add crossrefReferences
during that stage, which is the DOI phase's entire input.

Reads  <DATA_DIR>/doclings.json
Writes <DATA_DIR>/citations.json  {generatedAt, edges: [{source, target}]}

  python citations.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from citation_utils import build_connectivity

ROOT          = Path(__file__).resolve().parents[3]
DATA_DIR      = ROOT / os.environ.get("DATA_DIR", "data")
DOCLINGS_PATH = DATA_DIR / "doclings.json"
OUTPUT_PATH   = DATA_DIR / "citations.json"

# Moved here from heuristic.py with the builder they configure.
MIN_KEY_LENGTH      = int(os.environ.get("HEURISTIC_MIN_KEY_LENGTH", "4"))        # skip title/author match keys shorter
                                                                                 # than this; short surnames over-match
MIN_CONTAINED_TITLE = int(os.environ.get("HEURISTIC_MIN_CONTAINED_TITLE", "15"))  # proper title containment (one inside
                          # the other) only counts when the contained title has at least this many chars — blocks short
                          # generic titles ('networks') from matching every longer title that mentions the word


def run() -> None:
    if not DOCLINGS_PATH.exists():
        print(f"[citations] {DOCLINGS_PATH} not found — run extract first.", file=sys.stderr)
        sys.exit(1)

    doclings: dict = json.loads(DOCLINGS_PATH.read_text(encoding='utf-8'))
    if not doclings:
        print("[citations] doclings.json is empty — nothing to link.")
        return

    print("[citations] Building citation connectivity graph ...")
    adjacency = build_connectivity(doclings, min_key_length=MIN_KEY_LENGTH,
                                   min_contained_length=MIN_CONTAINED_TITLE)
    edges = [{"source": source_id, "target": target_id}
             for source_id, target_ids in adjacency.items() for target_id in target_ids]

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "edges":       edges,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')

    cited = len({edge["target"] for edge in edges})
    print(f"[citations] {len(edges)} edge(s) over {len(doclings)} document(s); "
          f"{cited} document(s) cited at least once")
    print(f"[citations] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
