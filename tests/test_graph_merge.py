"""
test_graph_merge.py — cluster bookkeeping across two merge passes

The entity passes run in sequence (entity_resolution, then abbreviations), so
the second can merge away a name the first made a representative. Checks that
the recorded clusters survive that: every key names a live entity, and no member
is stranded under a key the graph no longer contains.

Run:  .venv/Scripts/python tests/test_graph_merge.py
      npm run test:graph-merge

Pure functions only — no model, no corpus, no Ollama.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "extraction"))

from graph_merge import fold_clusters   # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  — ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


print("[test_graph_merge] fold_clusters\n")

# --- 1. the AI case: a representative merged away by the later pass ----------
# Lexical made 'AI' represent {A.I., AI}; abbreviations then fold 'AI' into its
# expansion. 'A.I.' must travel with it.
lexical = {
    "AI": ["A.I.", "AI"],
    "artificial intelligence": ["Artificial intelligence", "artificial intelligence"],
}
abbreviation = {"artificial intelligence": ["AI", "artificial intelligence"]}
folded = fold_clusters(lexical, abbreviation)

check("dead representative key is dropped", "AI" not in folded,
      f"keys: {sorted(folded)}")
check("its members move to the surviving representative",
      set(folded["artificial intelligence"]) >= {"A.I.", "AI"},
      str(folded["artificial intelligence"]))
check("the earlier pass's own members are kept",
      {"Artificial intelligence", "artificial intelligence"}
      <= set(folded["artificial intelligence"]))
check("nothing is lost overall",
      set(folded["artificial intelligence"])
      == {"A.I.", "AI", "Artificial intelligence", "artificial intelligence"},
      str(folded["artificial intelligence"]))

# --- 2. untouched clusters carry over unchanged ------------------------------
lexical = {"AI": ["A.I.", "AI"], "DFT": ["DFT", "dft"]}
folded = fold_clusters(lexical, {"artificial intelligence": ["AI"]})
check("an unrelated cluster is untouched", folded.get("DFT") == ["DFT", "dft"],
      str(folded.get("DFT")))

# --- 3. no orphans: every key is reachable, no member points at a dead key ---
folded = fold_clusters({"AI": ["A.I.", "AI"], "MLIPs": ["ML-IPs", "MLIPs"]},
                       {"artificial intelligence": ["AI"],
                        "machine learning interatomic potentials": ["MLIPs"]})
orphans = [key for key in folded if any(key in members
                                        for other, members in folded.items()
                                        if other != key)]
check("no key is also a member of another cluster", not orphans, str(orphans))

# --- 4. transitive: three passes deep still collapses to one key -------------
# Not reachable with today's two passes; guards a third being added later.
folded = fold_clusters({"A": ["a", "A"], "B": ["A", "B"]}, {"C": ["B", "C"]})
check("a three-deep chain collapses onto the final representative",
      set(folded.get("C", [])) == {"a", "A", "B", "C"} and "A" not in folded
      and "B" not in folded, f"{folded}")

# --- 5. the input mapping is not mutated -------------------------------------
# merge_abbreviations passes the payload's own dict; dict(graph) is a shallow
# copy, so mutating in place would reach back into the caller's graph.
original = {"AI": ["A.I.", "AI"]}
fold_clusters(original, {"artificial intelligence": ["AI"]})
check("the existing mapping is left unmodified", original == {"AI": ["A.I.", "AI"]},
      str(original))

# --- 6. members stay sorted and deduplicated ---------------------------------
folded = fold_clusters({"X": ["b", "X"]}, {"X": ["a", "b", "X"]})
check("members are sorted and deduplicated", folded["X"] == ["X", "a", "b"],
      str(folded["X"]))

print(f"\n[test_graph_merge] {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
