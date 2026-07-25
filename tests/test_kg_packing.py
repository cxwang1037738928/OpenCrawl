"""
test_kg_packing.py — pipeline stage 6, batching only (no model calls)

Checks the invariants kg_graph.py's packing must hold before any Ollama call
is made. test_kg_graph.js runs the real thing but takes hours, so a regression
in the batching would otherwise only surface after a very long wait.

Run:  .venv/Scripts/python tests/test_kg_packing.py
      npm run test:kg-packing

Prerequisite: tests/test-output/embeddings.json (produced by test_embed.js).
Needs no Ollama and loads no model.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "tests" / "test-output"

# Must be set before importing kg_graph: it resolves DATA_DIR and the tunables
# at import time, exactly as the pipeline's spawned process would.
os.environ["DATA_DIR"] = str(TEST_DATA)
sys.path.insert(0, str(ROOT / "backend" / "extraction"))

import kg_graph  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  — ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


store = json.loads((TEST_DATA / "embeddings.json").read_text(encoding="utf-8"))
chunks = store["chunks"]
doc_ids = list(dict.fromkeys(c["docId"] for c in chunks))[:5]
budget = kg_graph.CALL_MAX_CHARS

batches, chunks_used = kg_graph._call_batches(chunks, doc_ids, budget)
calls = [kg_graph._batch_text(b) for b in batches]   # rendered prompt per batch
print(f"[test_kg_packing] {chunks_used} chunk(s) over {len(doc_ids)} doc(s) "
      f"-> {len(calls)} call(s), budget {budget}\n")

# --- 1. every call fits the budget ------------------------------------------
oversized = [len(c) for c in calls if len(c) > budget]
check("every call is within KG_CALL_MAX_CHARS", not oversized,
      f"{len(oversized)} over budget: {oversized[:3]}" if oversized else
      f"max {max(len(c) for c in calls)} / {budget}")

# --- 2. no call spans two documents -----------------------------------------
# The header names the document, so a mixed call would carry two DOCUMENT lines.
mixed = [c for c in calls if c.count("DOCUMENT: ") > 1]
check("no call mixes two documents", not mixed, f"{len(mixed)} mixed")

# --- 3. header appears exactly once per call --------------------------------
missing = [c for c in calls if c.count("DOCUMENT: ") != 1]
check("each call states its header exactly once", not missing,
      f"{len(missing)} call(s) without exactly one header")

selected = [c for c in chunks
            if c["docId"] in set(doc_ids)
            and kg_graph._norm_heading(c.get("heading") or "") not in kg_graph._REF_HEADINGS
            and kg_graph._clean_body(c)]
selected.sort(key=lambda c: (doc_ids.index(c["docId"]), c.get("chunkIndex", 0)))
all_bodies = "\n\n".join(call.split("\n---\n", 1)[-1] for call in calls)

# --- 4. the embed-time prefix is gone from every chunk body -----------------
# Asserted as "no body STARTS with its prefix", not "the prefix appears
# nowhere in the call": a paper's own front matter legitimately repeats its
# title in the body text ("Attention Is All You Need. Ashish Vaswani..."), so
# a substring search reports a leak that isn't one.
prefixed = [c["id"] for c in selected
            if (c.get("prefixLen") or 0)
            and (kg_graph._clean_body(c).startswith(c["text"][:c["prefixLen"]].strip())
                 or len(kg_graph._clean_body(c)) >= len(c["text"].strip()))]
check("chunk prefixes are stripped from call bodies", not prefixed,
      f"{len(prefixed)} chunk(s) kept their prefix: {prefixed[:3]}" if prefixed
      else f"{sum(1 for c in selected if c.get('prefixLen'))} prefixed chunk(s) cleaned")

# --- 5. no UNDER-BUDGET chunk is split --------------------------------------
# A chunk whose own body exceeds the budget is deliberately split (its pieces
# still appear, just not as one contiguous span); every chunk that FITS must
# appear whole in some call. Concatenated call bodies are the haystack.
fits = [c for c in selected if len(kg_graph._clean_body(c)) <= budget - 200]
intact = sum(1 for c in fits if kg_graph._clean_body(c) in all_bodies)
oversized_ct = len(selected) - len(fits)
check("every under-budget chunk appears whole in some call", intact == len(fits),
      f"{intact}/{len(fits)} intact ({oversized_ct} oversized chunk(s) split by design)")

# --- 6. packing actually packs ----------------------------------------------
check("packing reduced call count well below chunk count",
      len(calls) < len(selected) / 3,
      f"{len(selected)} chunks -> {len(calls)} calls "
      f"({len(selected) / max(len(calls), 1):.1f}x fewer)")

fills = [len(c) for c in calls]
print(f"\n[test_kg_packing] mean fill {sum(fills) / len(fills):.0f}/{budget} chars "
      f"({100 * (sum(fills) / len(fills)) / budget:.0f}%)")
print(f"[test_kg_packing] sample header:\n    "
      + calls[0].split("\n---\n")[0].replace("\n", "\n    "))

if failures:
    print(f"\n[test_kg_packing] {len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\n[test_kg_packing] all checks passed.")
