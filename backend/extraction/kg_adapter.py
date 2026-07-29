"""
kg_adapter.py — a fault-tolerant dspy output parser for the graph stage.

WHY THIS EXISTS

kg-gen asks the model for `relations: list[Relation]`, and dspy validates that
field as ONE unit: `TypeAdapter(list[Relation]).validate_python(candidate)`
(dspy/adapters/utils.py). A single bad element — `object` emitted as a list, a
missing `predicate`, a null — raises, and all forty triples in the response are
discarded. The failure then escalates through three more layers before anything
is salvaged:

  1. dspy retries the whole call with JSONAdapter          (+1 model call)
  2. kg-gen re-prompts with a looser signature, then runs a
     ChainOfThought "fix the relations" repair pass        (+2 model calls)
  3. kg_graph.py halves the batch and recurses, then walks
     a temperature ladder at the leaves        (many more model calls)

Measured on a 3-call smoke run: 32 model requests where 6 would do — ~81% of
the model work was spent recovering from one bad element in one list.

Every layer above pays an LLM round-trip to fix something a local `for` loop
can fix for free: the response usually parses to a perfectly good LIST, and
only one or two of its elements are malformed. This adapter drops those
elements and keeps the rest.

HOW IT HOOKS IN

dspy.Predict resolves its adapter as `settings.adapter or ChatAdapter()`
(dspy/predict/predict.py), so `dspy.configure(adapter=TolerantChatAdapter())`
replaces the parser process-wide with NO change to kg-gen, to the prompts, or
to the signatures. The output field stays typed `list[Relation]`, so dspy still
generates the format instructions from the annotation — this buys the leniency
without inheriting the job of describing the JSON shape to the model.

WHAT IT DOES NOT CHANGE

The happy path is untouched: parse() delegates to ChatAdapter first and only
runs salvage code when dspy would otherwise have raised. When salvage recovers
NOTHING (the response held no list at all — prose, or truncated past the first
element), the original error is re-raised, so kg_graph.py's split-and-retry
safety net still gets its turn. Leniency is added at the element level only;
it never converts a genuinely failed call into a silent empty success.

Dropped elements are COUNTED, not hidden (see `dropped_elements`): the loud
failure this replaces is what revealed that 12KB calls were broken, and a
parser that quietly turns "40 triples, 25 malformed" into 15 would have hidden
that. kg_graph.py reports the totals at the end of a run.
"""

from typing import Any, get_args, get_origin

import json_repair
from pydantic import TypeAdapter

from dspy.adapters.chat_adapter import ChatAdapter, field_header_pattern
from dspy.adapters.utils import parse_value
from dspy.signatures.signature import Signature


class TolerantChatAdapter(ChatAdapter):
    """ChatAdapter that validates list-typed output fields element-wise.

    Counters (read by kg_graph.py after a run):
      dropped_elements  malformed items discarded from otherwise-good lists
      salvaged_fields   fields rescued this way (each one is a model call, and
                        usually a whole 3-layer cascade, that did NOT happen)
      failed_parses     parses salvage could not rescue, which fall through to
                        the original error and kg_graph.py's split/retry
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropped_elements = 0
        self.salvaged_fields = 0
        self.failed_parses = 0

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        try:
            return super().parse(signature, completion)
        except Exception:
            salvaged = self._salvage(signature, completion)
            if salvaged is None:
                self.failed_parses += 1
                raise      # unsalvageable — let the caller's retry ladder run
            return salvaged

    # ── salvage ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sections(completion: str) -> list[tuple[str | None, str]]:
        """Split a completion into (field_name, text) on dspy's own field
        headers. Mirrors ChatAdapter.parse's splitting and reuses dspy's
        compiled `field_header_pattern`, so the two cannot drift on format."""
        sections: list[tuple[str | None, list[str]]] = [(None, [])]
        for line in completion.splitlines():
            header_match = field_header_pattern.match(line.strip())
            if header_match:
                trailing_content = line[header_match.end():].strip()
                sections.append((header_match.group(1),
                                 [trailing_content] if trailing_content else []))
            else:
                sections[-1][1].append(line)
        return [(name, "\n".join(lines).strip()) for name, lines in sections]

    def _salvage(self, signature: type[Signature], completion: str) -> dict[str, Any] | None:
        """Re-parse a completion dspy rejected, dropping only the bad elements.

        Returns None — meaning "could not rescue this, re-raise" — when a field
        is missing entirely, is not a list type, or held nothing valid.
        """
        parsed_fields: dict[str, Any] = {}
        for field_name, raw_text in self._sections(completion):
            if field_name in parsed_fields or field_name not in signature.output_fields:
                continue
            annotation = signature.output_fields[field_name].annotation
            # Most fields in a failed response are fine; only the one that
            # raised needs the element-wise path.
            try:
                parsed_fields[field_name] = parse_value(raw_text, annotation)
                continue
            except Exception:
                pass
            kept_elements = self._parse_list_elementwise(raw_text, annotation)
            if kept_elements is None:
                return None
            parsed_fields[field_name] = kept_elements

        # dspy requires every output field to be present; a partial parse is
        # not a usable Prediction, so treat it as unsalvageable.
        if parsed_fields.keys() != signature.output_fields.keys():
            return None
        return parsed_fields

    def _parse_list_elementwise(self, raw_text: str, annotation: Any) -> list | None:
        """Validate a `list[T]` field one element at a time, dropping failures.

        json_repair (dspy's own choice, already a dependency) handles the
        syntactic damage first — unclosed brackets, trailing commas, markdown
        fences — and notably recovers the COMPLETE elements of a list that was
        truncated by the output-token cap, which today costs the whole call.
        What is left after that is schema damage in individual elements, which
        is what this drops.
        """
        if get_origin(annotation) is not list:
            return None                    # not a list field — nothing to salvage
        item_types = get_args(annotation)
        if len(item_types) != 1:
            return None

        candidate = json_repair.loads(raw_text)
        if not isinstance(candidate, list):
            return None                    # no list in there at all (prose, empty)

        item_validator = TypeAdapter(item_types[0])
        kept_elements: list = []
        dropped_count = 0
        for element in candidate:
            try:
                kept_elements.append(item_validator.validate_python(element))
            except Exception:
                dropped_count += 1

        # Nothing survived: this is a genuinely failed extraction, not a list
        # with a few bad apples. Re-raising keeps the split/retry safety net —
        # a shorter batch may well succeed where this one did not.
        if not kept_elements:
            return None

        self.dropped_elements += dropped_count
        self.salvaged_fields += 1
        return kept_elements
