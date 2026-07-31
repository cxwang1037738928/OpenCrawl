"""
extract_abbreviations.py — pull "long form (SHORT)" definitions out of a corpus

Runs in .venv-abbrev, NOT the main environment: scispaCy pins numpy 1.x and
installing it alongside would downgrade the numpy 2.x that torch, scipy and
sentence-transformers are built against. Keeping it isolated costs one manual
step and protects the pipeline.

  .venv-abbrev/Scripts/python scripts/extract_abbreviations.py <collection-id>

Reads  <DATA_DIR>/doclings.json  (docId -> {text, ...})
Writes <DATA_DIR>/abbreviations.json  {short form: {long form: times defined}}

Counting definitions rather than collapsing them here is deliberate: the same
acronym means different things in different papers, and only the consumer
(backend/extraction/abbreviations.py) knows what dominance to demand before
trusting one reading.
"""

import collections
import json
import sys
from pathlib import Path

import spacy
from scispacy.abbreviation import AbbreviationDetector  # noqa: F401  (registers the pipe)

ROOT = Path(__file__).resolve().parents[1]
# A blank pipeline is enough: the detector needs a tokeniser, not a model, so
# no scientific model download is required.
MAX_CHARS = 1_000_000


def build_pipeline():
    nlp = spacy.blank("en")
    nlp.max_length = MAX_CHARS
    nlp.add_pipe("abbreviation_detector")
    return nlp


def definitions_for(nlp, text: str) -> list[tuple[str, str]]:
    """(short form, long form) pairs stated in one document."""
    pairs = []
    for start in range(0, len(text), MAX_CHARS):
        doc = nlp(text[start:start + MAX_CHARS])
        for abbreviation in doc._.abbreviations:
            short_form = str(abbreviation).strip()
            long_form = str(abbreviation._.long_form).strip()
            if short_form and long_form and len(long_form) > len(short_form):
                pairs.append((short_form, long_form))
    return pairs


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: extract_abbreviations.py <collection-id>", file=sys.stderr)
        sys.exit(1)

    data_dir = ROOT / "data" / "collections" / sys.argv[1]
    doclings = json.loads((data_dir / "doclings.json").read_text(encoding="utf-8"))

    nlp = build_pipeline()
    definitions: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    documents_with_definitions = 0

    for index, (doc_id, record) in enumerate(doclings.items(), start=1):
        pairs = definitions_for(nlp, record.get("text") or "")
        if pairs:
            documents_with_definitions += 1
        for short_form, long_form in pairs:
            definitions[short_form][long_form] += 1
        if index % 25 == 0:
            print(f"  {index}/{len(doclings)} documents, {len(definitions)} short forms")

    output = {short: dict(counts) for short, counts in sorted(definitions.items())}
    (data_dir / "abbreviations.json").write_text(json.dumps(output, indent=2),
                                                 encoding="utf-8")
    total = sum(sum(counts.values()) for counts in definitions.values())
    print(f"[abbreviations] {documents_with_definitions}/{len(doclings)} documents defined "
          f"{total} abbreviation(s), {len(definitions)} distinct short form(s)")
    # ASCII only: this runs under the Windows console default (cp1252), which
    # cannot encode an arrow and would raise AFTER the file is safely written.
    print(f"[abbreviations] wrote {data_dir / 'abbreviations.json'}")


if __name__ == "__main__":
    main()
