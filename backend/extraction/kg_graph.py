"""
kg_graph.py — knowledge graph over a collection's documents, via kg-gen.

Replaces the old citation/section graph (build_graph.js). kg-gen prompts an
LLM for (subject, predicate, object) triples, so the graph holds concepts and
their relations instead of document structure.

Two kinds of document, so the whole corpus is represented without paying
full-text price for all of it:

  FULL TEXT   — the top KG_FULL_TEXT_FRACTION (default 0.4 = 40%) of the
                collection by heuristic.py rank. Every non-bibliography chunk
                is sent.
  SUMMARY     — every other document. Only its title, abstract and conclusion
                are sent (see _summary_entries), which is where a paper states
                what it contributes and what it concluded. Cost per summary
                document is one or two calls instead of dozens, so the corpus
                stays connected to the graph without dominating the runtime.

The unit of extraction is a BATCH of consecutive chunks from ONE document,
packed greedily up to the model's per-call char budget (see "Model + context
sizing" below). Chunks are the embed stage's docling chunks (structure-aware —
never crossing a section, let alone a document boundary), so a batch is a run
of adjacent text in its original order.

Packing rules:
  - Documents are never mixed in one call. Two papers in one prompt invite
    the model to emit triples linking a concept in paper A to one in paper B
    that no text asserts, and aggregate() below cannot tell those apart from
    real edges.
  - A chunk is never split across calls. The chunk that would overflow the
    budget starts the next call instead, which re-states the header. Only a
    chunk exceeding the budget ON ITS OWN is split (rare — chunks are ~1KB
    against a 6KB budget on the local model).
  - The embed stage's "title — heading" prefix is STRIPPED from every chunk
    (via prefixLen) and re-stated ONCE per call as a header. Embedded in
    every chunk it was both repetition and noise: the small model extracted
    the document title as an entity from every chunk, so the title became a
    first-class node in the graph. The header carries the same grounding at
    1/N the repetition. The merged prefix+body text is left untouched in the
    chunk store — retrieval is tuned on it (see chunker.js), and prefixLen
    lets the two consumers diverge without storing the text twice.

Model + context sizing:
  KG_MODEL names the model; documents/model_metadata.json gives its
  context_length in tokens. Nothing here is hard-coded to one model — the
  per-call char budget and the output-token cap are DERIVED from that window,
  so switching KG_MODEL to a long-context hosted model automatically packs
  bigger calls (fewer calls, less repeated header, faster) and switching back
  to a small local model shrinks them again. See _model_profile().

  A bare tag (no "/") is a local Ollama model: its usable window is capped by
  KG_NUM_CTX, because Ollama sizes its KV cache to num_ctx and on a small GPU
  that cache evicts model layers (measured on a 4GB GTX 970: 15/27 layers
  resident at 2048, 5/27 at 16384, decode 13.5 -> 9.0 tok/s). A prefixed id
  (e.g. "gemini/gemini-2.5-flash") is routed by LiteLLM to that provider,
  where there is no local KV cache to trade against and the model's full
  window is usable — bounded only by KG_CALL_MAX_CHARS_CEILING.

Budget choice is a reliability trade, not just speed: kg-gen validates each
call's relations as one typed list, so a single malformed triple from a small
model voids the whole call — a probability that rises with input size. ~6KB
validates reliably on the local 3B model; 12KB failed on every call. A call
that still fails is HALVED and retried (_generate_with_retry), shortening the
relation list until it validates and salvaging the pieces that do, rather than
dropping every chunk in the batch. The document title, which the model
re-extracts from the header, is filtered from the final graph
(_write_graph_json).

Batch graphs are unioned with kg-gen's aggregate(). The stage depends on
embed: extract → embed → heuristic → graph.

Reads:
  <DATA_DIR>/embeddings.json        — chunk store (embed.js / chunker.js)
  <DATA_DIR>/heuristic_output.json  — top-k doc ranking (heuristic.py); decides
                     WHICH documents get full text. Absent → rank order falls
                     back to chunk-store order.
  <DATA_DIR>/doclings.json          — optional; supplies metadata.abstract for
                     summary documents, whose abstract is often not a section
                     of its own and so never became a chunk. Absent → summary
                     documents fall back to their abstract-headed chunks.
  documents/model_metadata.json     — [{id, name, context_length}] per model

Writes:
  <DATA_DIR>/graph.json      — {createdAt, model, entities, edges, relations,
                                relationDocIds, sourceDocIds, fullTextDocIds,
                                summaryDocIds, chunksProcessed, calls,
                                callsFailed, callsCompleted, complete}.
                                relationDocIds is index-aligned with relations:
                                relationDocIds[i] names every document the i-th
                                triple came from, so a claim can be cited and a
                                triple attested by several papers is visible as
                                such. Rewritten atomically after EVERY call, not
                                just at the end — a crash leaves a valid partial
                                graph. callsCompleted < calls and complete=false
                                mark a mid-run partial.
  <DATA_DIR>/graph.raw.json  — the same payload as it stood BEFORE entity
                                resolution and predicate normalization, so
                                their thresholds can be re-tuned later without
                                re-running extraction. Merging is lossy.
  <DATA_DIR>/kg_view.html    — standalone interactive visualization (kg-gen),
                                written once at the end (whole-graph render)

After each per-call flush the marker line _PROGRESS_MARKER is printed to
stdout; the Node parent (routes/pipeline.js) watches for it and ingests the
partial graph.json into Postgres, saving the graph once per call.

Env:
  KG_MODEL             model id; a bare tag is prefixed with 'ollama_chat/' for
                       LiteLLM, which kg-gen routes through. Should appear in
                       documents/model_metadata.json so its context is known.
  MODEL_METADATA_PATH  default documents/model_metadata.json
  OLLAMA_URL           default http://localhost:11434
  KG_FULL_TEXT_FRACTION  share of the corpus graphed from full text (default
                       0.4); the rest contribute title + abstract + conclusion
  KG_NUM_CTX           local-model context cap in tokens (default 8192) — the
                       VRAM trade above, applied as min() with the model's own
                       context_length. Ignored for hosted models.
  KG_CALL_MAX_CHARS    force a fixed per-call char budget; unset (or 0) derives
                       it from the model's context window
  KG_MAX_SPLIT_DEPTH   how many times a failing call is halved (default 3)
  KG_RETRY_TEMPERATURES  retry ladder for malformed-triple failures
  KG_SUMMARY_HEADINGS  headings that make a section part of a summary document's
                       input (default abstract,conclusion,concluding remarks)

Reference/bibliography chunks are excluded — citation strings would flood the
graph with author/title noise. Each chunk piece is retried over the
temperature ladder; a piece that fails every temperature is skipped (and
counted) rather than failing the multi-hour stage.
"""

import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dspy
from kg_gen import KGGen
from kg_gen.models import Graph
from kg_gen.utils.visualize_kg import visualize
from dspy.utils.usage_tracker import track_usage

from kg_adapter import TolerantChatAdapter
from abbreviations import merge_abbreviations
from entity_resolution import resolve_entities
from graph_merge import aligned_sources
from predicate_normalization import normalize_predicates
from sapphire.prune_authors import prune_author_nodes

# Replaces dspy's output parser process-wide (dspy.Predict resolves
# `settings.adapter or ChatAdapter()`), so a relation list with one malformed
# element loses that element instead of all forty — and the 3-layer repair
# cascade below never fires for it. kg-gen, its prompts and its signatures are
# untouched: the output field stays typed, so dspy still writes the format
# instructions. See kg_adapter.py. Set KG_TOLERANT_PARSE=0 to run the stock
# parser (the A/B this was measured with).
_TOLERANT_PARSE = os.environ.get("KG_TOLERANT_PARSE", "1").strip() not in ("0", "false", "")
_TOLERANT_ADAPTER = TolerantChatAdapter() if _TOLERANT_PARSE else None

# Printed on its own line after each per-call flush of graph.json. The Node
# parent (routes/pipeline.js) watches stdout for this and ingests the partial
# graph into Postgres, so the graph is saved once per call, not once per run.
_PROGRESS_MARKER = "@@KG_GRAPH_SAVED@@"

# Force UTF-8 console streams, as extract.py does: the pipeline pipes stdout,
# which on Windows defaults to cp1252, and a non-Latin-1 char in a print here
# would fail the stage after the graph was already built.
for _console_stream in (sys.stdout, sys.stderr):
    _reconfigure_stream = getattr(_console_stream, "reconfigure", None)
    if callable(_reconfigure_stream):
        try:
            _reconfigure_stream(encoding="utf-8")
        except Exception:
            pass

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / os.environ.get("DATA_DIR", "data")
MODEL_METADATA_PATH = ROOT / os.environ.get(
    "MODEL_METADATA_PATH", os.path.join("documents", "model_metadata.json"))

KG_MODEL   = os.environ.get("KG_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Share of the collection graphed from FULL TEXT, by heuristic rank. The rest
# of the corpus is still graphed, from title + abstract + conclusion only, so
# every document reaches the graph — this fraction only decides how much of
# each one does. It is the main latency lever: full-text documents cost dozens
# of calls each, summary documents one or two.
FULL_TEXT_FRACTION = float(os.environ.get("KG_FULL_TEXT_FRACTION", "0.4"))

# Context window, in tokens, assumed for a model missing from
# model_metadata.json. Deliberately small: under-guessing packs smaller calls
# (slower, still correct), over-guessing silently overflows the window and
# Ollama drops the front of the prompt with no error.
FALLBACK_CONTEXT_TOKENS = int(os.environ.get("KG_FALLBACK_CONTEXT_TOKENS", "8192"))
# Local-model context cap — see the VRAM trade in the module docstring. Applied
# as min() with the model's own context_length, so it can only ever shrink the
# window, never claim one the model doesn't have.
LOCAL_NUM_CTX = int(os.environ.get("KG_NUM_CTX", "8192"))
# Rough chars-per-token for English prose, used to turn a token window into a
# char budget. 4 is the usual figure for GPT-family BPE and close enough here:
# the budget is a target, not a hard ceiling (calls that overflow get split).
CHARS_PER_TOKEN = float(os.environ.get("KG_CHARS_PER_TOKEN", "4"))
# Share of the usable window spent on packed body text. The rest holds kg-gen's
# prompt scaffolding, the entity list it feeds back on the relation pass, and
# the model's own output. 0.18 reproduces the measured-good 6000-char budget at
# the local 8192-token window (8192 × 0.18 × 4 ≈ 5900).
INPUT_WINDOW_FRACTION = float(os.environ.get("KG_INPUT_WINDOW_FRACTION", "0.18"))
# Share of the usable window left for the model's output (kg-gen's max_tokens).
# Its own default is 16000, which OVERFLOWS an 8192-token local window; sizing
# it from the window instead keeps prompt + output inside the context.
OUTPUT_WINDOW_FRACTION = float(os.environ.get("KG_OUTPUT_WINDOW_FRACTION", "0.5"))
MAX_OUTPUT_TOKENS = int(os.environ.get("KG_MAX_OUTPUT_TOKENS", "16000"))
# Absolute ceiling on a derived per-call budget. A million-token hosted window
# would otherwise pack an entire paper into one call: the relation list gets so
# long that one malformed triple (which voids the whole list) becomes likely
# again, and a failure costs the whole document. ~48KB is a large call that
# still fails cheaply.
CALL_MAX_CHARS_CEILING = int(os.environ.get("KG_CALL_MAX_CHARS_CEILING", "48000"))
# Set to force a fixed budget regardless of model; 0/unset derives it.
CALL_MAX_CHARS_OVERRIDE = int(os.environ.get("KG_CALL_MAX_CHARS", "0") or 0)
# A batch that fails validation is halved and retried (see _generate_with_retry)
# up to this depth, then a single small body falls back to the temperature
# ladder. 3 halvings take a 6KB call down to <1KB.
_MAX_SPLIT_DEPTH = int(os.environ.get("KG_MAX_SPLIT_DEPTH", "3"))
# Don't split a single body below this — smaller than a chunk gains nothing and
# a null-triple failure at this size is the temperature ladder's job. Also the
# floor for a derived per-call budget: below it, packing does nothing.
_MIN_SPLIT_CHARS = int(os.environ.get("KG_MIN_SPLIT_CHARS", "1500"))
# Thinking budget for hosted reasoning models (Gemini 3.x, gpt-5, …), passed to
# LiteLLM as reasoning_effort. Default 'minimal' — MEASURED on gemini-3.6-flash
# over a real extraction prompt: thinking is 54% of billed output at the default
# budget and buys nothing here, because listing entities and triples is a
# mechanical read of the source, not a reasoning problem.
#
#   effort     thinking  answer  billed out  entities found
#   default       1,336   1,137       2,473              87
#   high          2,358   1,321       3,679              90
#   minimal           0   1,188       1,188              97
#
# Zero thinking halved the bill AND found more entities. Empty string leaves
# the provider default. Ignored for local models: Ollama has no such param and
# LiteLLM would forward it.
REASONING_EFFORT = os.environ.get("KG_REASONING_EFFORT", "minimal").strip()
# graph.json is republished after every call while the Node parent reads it, and
# on Windows a replace over an open file fails. See _write_json_atomic — 6
# attempts backing off 0.15s, 0.30s, … covers a reader holding the handle for
# well over a second.
_REPLACE_ATTEMPTS = int(os.environ.get("KG_REPLACE_ATTEMPTS", "6"))
_REPLACE_BACKOFF_SECONDS = float(os.environ.get("KG_REPLACE_BACKOFF", "0.15"))
# Temperatures tried in order until kg-gen returns a valid graph for a chunk.
_RETRY_TEMPERATURES = tuple(
    float(value) for value in
    os.environ.get("KG_RETRY_TEMPERATURES", "0.0,0.4,0.7").split(",") if value.strip()
)

# Bibliography headings, shared with extract.py / heuristic.py / regex_utils.js.
_REF_HEADINGS = frozenset(
    heading.strip().lower()
    for heading in os.environ.get(
        "PIPELINE_REF_HEADINGS",
        "references,bibliography,works cited,literature cited,citations").split(",")
    if heading.strip()
)

# Sections a SUMMARY document contributes. Matched as substrings of the
# normalized heading, so '6. Conclusions' and 'Conclusion and Future Work'
# both count — papers name this section a dozen ways and an exact-match set
# would silently drop most of them.
_SUMMARY_HEADINGS = tuple(
    heading.strip().lower()
    for heading in os.environ.get(
        "KG_SUMMARY_HEADINGS", "abstract,conclusion,concluding remarks").split(",")
    if heading.strip()
)
_ABSTRACT_HEADING = "abstract"

# Which crawler this run belongs to (injected by routes/pipeline.js). Sapphire
# summarises by section heading; anything else has no such vocabulary and
# summarises by the ranking stage's chunk scores instead.
_CRAWLER = os.environ.get("CRAWLER", "sapphire").strip().lower()
# Share of a summary document's chunks kept when selecting by score.
_SUMMARY_CHUNK_FRACTION = float(os.environ.get("KG_SUMMARY_CHUNK_FRACTION", "0.4"))
# Chars one summary document may contribute. A real abstract plus conclusion is
# well under this; the cap matters when docling folded the BIBLIOGRAPHY into the
# conclusion section (no separate 'References' heading to filter on), which
# happens in this corpus. Measured there: the genuine conclusion is the first
# ~3.4KB and every chunk after it is citation strings. Keeping the head of the
# section keeps the argument and drops the debris.
_SUMMARY_MAX_CHARS = int(os.environ.get("KG_SUMMARY_MAX_CHARS", "4000"))

# Environment variable holding the API key for each hosted provider, keyed by
# the id prefix LiteLLM routes on. Absent from this map (or unset) is not fatal:
# LiteLLM reads the standard provider env vars itself.
_PROVIDER_KEY_ENV = {
    "gemini":    "GEMINI_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _norm_heading(heading: str) -> str:
    """Lowercase, drop leading numbering ('7. References'), strip trailing punctuation."""
    normalized = re.sub(r'^[\divxlc]+[\.\)]?\s+', '', heading.lower().strip())
    return normalized.rstrip(' .:')


def _litellm_model(model: str) -> str:
    """kg-gen calls LiteLLM, which needs a provider prefix; bare tags are Ollama."""
    return model if "/" in model else f"ollama_chat/{model}"


def _is_local_model(model: str) -> bool:
    """True for models served by the local Ollama daemon. A bare tag is one by
    convention (_litellm_model prefixes it); an explicit 'ollama.../' id is too."""
    return "/" not in model or model.split("/", 1)[0].startswith("ollama")


def _load_model_catalog() -> list[dict]:
    """[{id, name, context_length}] from model_metadata.json — the single place
    a model's context window is declared. A missing or malformed file is not
    fatal: the model just falls back to FALLBACK_CONTEXT_TOKENS."""
    try:
        catalog = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    models = catalog.get("models") if isinstance(catalog, dict) else catalog
    return [model for model in (models or []) if isinstance(model, dict) and model.get("id")]


def _model_profile(model_id: str) -> dict:
    """Everything the run needs to size itself to ONE model.

    context_length comes from model_metadata.json; every other number is
    derived from it, so a new model is a metadata entry rather than a code
    change. Keys:
      litellmId       provider-prefixed id kg-gen/LiteLLM is called with
      name            human label from the catalog (or the raw id)
      contextTokens   the model's declared window
      windowTokens    the window this run will actually use (local models are
                      capped by KG_NUM_CTX — the KV-cache/VRAM trade)
      numCtx          num_ctx to send to Ollama, or None for hosted models
                      (an unknown param there would be rejected by LiteLLM)
      callMaxChars    header + body packed into one kg-gen call
      maxOutputTokens kg-gen's max_tokens, kept inside the window
      apiKey          provider key when one is configured, else None
    """
    catalog_entry = next(
        (model for model in _load_model_catalog()
         if model["id"] == model_id or model["id"] == _litellm_model(model_id)),
        None)
    context_tokens = int((catalog_entry or {}).get("context_length") or FALLBACK_CONTEXT_TOKENS)

    is_local = _is_local_model(model_id)
    window_tokens = min(context_tokens, LOCAL_NUM_CTX) if is_local else context_tokens

    call_max_chars = CALL_MAX_CHARS_OVERRIDE or int(
        window_tokens * INPUT_WINDOW_FRACTION * CHARS_PER_TOKEN)
    call_max_chars = max(_MIN_SPLIT_CHARS, min(call_max_chars, CALL_MAX_CHARS_CEILING))

    max_output_tokens = max(512, min(int(window_tokens * OUTPUT_WINDOW_FRACTION),
                                     MAX_OUTPUT_TOKENS))

    litellm_id = _litellm_model(model_id)
    key_env = _PROVIDER_KEY_ENV.get(litellm_id.split("/", 1)[0])
    # STRIPPED: the key goes straight into an HTTP header, and a .env line with
    # a trailing space (invisible, and easy to leave behind) makes an illegal
    # header value. LiteLLM surfaces that as APIConnectionError — which reads
    # like the network is down, not like a config typo, and every retry in the
    # ladder above reproduces it identically.
    api_key = (os.environ.get(key_env) or "").strip() if key_env else ""

    return {
        "litellmId":       litellm_id,
        "name":            (catalog_entry or {}).get("name") or model_id,
        "contextTokens":   context_tokens,
        "windowTokens":    window_tokens,
        "numCtx":          window_tokens if is_local else None,
        "callMaxChars":    call_max_chars,
        "maxOutputTokens": max_output_tokens,
        "isLocal":         is_local,
        "knownModel":      catalog_entry is not None,
        "apiKey":          api_key or None,
    }


def _make_kg_client(profile: dict) -> KGGen:
    """kg-gen client wired to the right backend. Local models go to the Ollama
    daemon with its placeholder key; hosted ones carry the provider key when
    .env has it, and otherwise let LiteLLM find its own standard env var."""
    if profile["isLocal"]:
        return KGGen(model=profile["litellmId"], api_base=OLLAMA_URL, api_key="ollama",
                     temperature=0.0, max_tokens=profile["maxOutputTokens"])
    return KGGen(model=profile["litellmId"], api_key=profile["apiKey"],
                 temperature=0.0, max_tokens=profile["maxOutputTokens"])


def _ranked_doc_ids(data_dir: Path) -> list[str]:
    """docIds of the heuristic's top-k, in rank order; [] when the ranking isn't
    available (no heuristic run yet, or an unreadable/empty file) so the caller
    falls back to chunk-store order."""
    heuristic_path = data_dir / "heuristic_output.json"
    if not heuristic_path.exists():
        return []
    try:
        top_k = json.loads(heuristic_path.read_text(encoding="utf-8")).get("topK", [])
    except (json.JSONDecodeError, ValueError):
        return []
    return [entry["docId"] for entry in top_k if entry.get("docId")]


def _chunk_scores(data_dir: Path) -> dict[str, float]:
    """{chunkId: score} from the ranking stage, or {} when unavailable.

    Only the non-sapphire summary path uses these. Empty is a valid state — an
    older heuristic_output.json predates the field — and the caller then falls
    back to heading-based summaries.
    """
    heuristic_path = data_dir / "heuristic_output.json"
    if not heuristic_path.exists():
        return {}
    try:
        return json.loads(heuristic_path.read_text(encoding="utf-8")).get("chunkScores") or {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _split_documents(chunk_doc_ids: list[str], ranked_doc_ids: list[str],
                     fraction: float) -> tuple[list[str], list[str]]:
    """(full_text_ids, summary_ids) — the top `fraction` of the corpus by
    heuristic rank gets full text, everything else gets summarized.

    The ranking only ORDERS the corpus here; it no longer decides who is in the
    graph, because summary documents are graphed too. When heuristic.py emitted
    fewer documents than the quota (HEURISTIC_K below the fraction), the
    remaining full-text slots are filled in chunk-store order so the fraction
    still holds — arbitrary among unranked docs, but never fewer than asked for.
    """
    if not chunk_doc_ids:
        return [], []
    known = set(chunk_doc_ids)
    ranked = [doc_id for doc_id in ranked_doc_ids if doc_id in known]

    full_count = max(1, math.ceil(len(chunk_doc_ids) * fraction))
    full_ids = ranked[:full_count]
    if len(full_ids) < full_count:
        for doc_id in chunk_doc_ids:
            if len(full_ids) >= full_count:
                break
            if doc_id not in full_ids:
                full_ids.append(doc_id)

    # Summary docs keep rank order where it exists, then chunk-store order, so
    # the incremental flush covers the more important documents first.
    selected = set(full_ids)
    summary_ids = [doc_id for doc_id in ranked if doc_id not in selected]
    summary_ids += [doc_id for doc_id in chunk_doc_ids
                    if doc_id not in selected and doc_id not in set(summary_ids)]
    return full_ids, summary_ids


def _abstracts(data_dir: Path) -> dict[str, str]:
    """docId → abstract from doclings.json (extract.py's metadata).

    Read from the extract stage rather than the chunk store because a paper's
    abstract is frequently NOT a section of its own — it sits in the front
    matter — so it never became a chunk with an 'Abstract' heading. Missing
    file → {}, and _summary_entries falls back to abstract-headed chunks.
    """
    doclings_path = data_dir / "doclings.json"
    if not doclings_path.exists():
        return {}
    try:
        doclings = json.loads(doclings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {}
    return {
        doc_id: (entry.get("metadata") or {}).get("abstract", "").strip()
        for doc_id, entry in doclings.items()
        if (entry.get("metadata") or {}).get("abstract", "").strip()
    }


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split one chunk into near-equal pieces under max_chars, cutting on word
    boundaries. Called only for chunks larger than the context guard."""
    piece_count = math.ceil(len(text) / max_chars)
    words = text.split(" ")
    words_per_piece = math.ceil(len(words) / piece_count)
    return [
        " ".join(words[start:start + words_per_piece])
        for start in range(0, len(words), words_per_piece)
    ]


def _clean_body(chunk: dict) -> str:
    """Chunk text with the embed stage's 'title — heading\\n' prefix removed.
    prefixLen is written by chunker.js and carried through embed.js and the
    Chunk rows, so the prefix can be dropped exactly without re-deriving it."""
    text = chunk.get("text") or ""
    return text[chunk.get("prefixLen") or 0:].strip()


def _doc_title(chunk: dict) -> str:
    """Document title, read back off the stripped prefix ('title — heading').
    Taken from the chunk store so this stage needs no extra input file; falls
    back to the filename when a document had no title at extract time."""
    prefix = (chunk.get("text") or "")[: chunk.get("prefixLen") or 0]
    title = prefix.split("—")[0].strip() if "—" in prefix else prefix.strip()
    return title or (chunk.get("filename") or "").strip()


# A "batch" is one call's worth of work, kept STRUCTURED (not pre-rendered) so
# a batch that fails validation can be halved and re-rendered — see
# _generate_with_retry. Keys: title (str), headings (list[str]), bodies
# (list[str], one clean chunk body each, in document order).

def _render_call(title: str, headings: list[str], bodies: list[str]) -> str:
    """One call's prompt text: the document header once, then the bodies in
    document order. Blank line between bodies so the model sees where one
    chunk ends — they are adjacent but not necessarily continuous prose."""
    header_lines = []
    if title:
        header_lines.append(f"DOCUMENT: {title}")
    if headings:
        header_lines.append("SECTIONS: " + "; ".join(headings))
    body = "\n\n".join(bodies)
    return f"{chr(10).join(header_lines)}\n---\n{body}" if header_lines else body


def _batch_text(batch: dict) -> str:
    return _render_call(batch["title"], batch["headings"], batch["bodies"])


def _document_batches(title: str, entries: list[tuple[str, str]],
                      max_chars: int, doc_id: str) -> list[dict]:
    """Greedy pack of ONE document's (heading, body) entries into batches.

    Fills a batch with consecutive entries until the next one would push it past
    max_chars; that entry starts the next batch (which re-states the header)
    rather than being split. An entry that exceeds max_chars on its own — no
    packing can help it — is split into multiple bodies under the same header.

    Takes (heading, body) pairs rather than chunks so full-text and summary
    documents share one packer: the only difference between them is which
    entries they hand in.

    doc_id rides on every batch so the triples a call produces can be attributed
    to the paper they came from. One document per batch is what makes that exact
    rather than a guess — see _call_batches.
    """
    batches: list[dict] = []
    bodies: list[str] = []
    headings: list[str] = []

    def flush() -> None:
        nonlocal bodies, headings
        if bodies:
            batches.append({"title": title, "headings": headings,
                            "bodies": bodies, "docId": doc_id})
        bodies, headings = [], []

    for heading, body in entries:
        if not body:
            continue
        heading = (heading or "").strip()
        # Headings accumulate per batch, so the header names every section the
        # batch spans — the grounding the per-chunk prefix used to carry.
        trial_headings = (headings + [heading]
                          if heading and heading not in headings else headings)
        if bodies and len(_render_call(title, trial_headings, bodies + [body])) > max_chars:
            flush()
            trial_headings = [heading] if heading else []
        if not bodies and len(_render_call(title, trial_headings, [body])) > max_chars:
            # Too big even alone: split into one batch PER piece (each under the
            # budget), all sharing the header. Budget the split against the
            # space the header leaves.
            overhead = len(_render_call(title, trial_headings, [""]))
            for piece in _split_oversized(body, max(max_chars - overhead, 1000)):
                batches.append({"title": title, "headings": trial_headings,
                                "bodies": [piece], "docId": doc_id})
            continue
        bodies.append(body)
        headings = trial_headings
    flush()
    return batches


def _full_text_entries(doc_chunks: list[dict]) -> list[tuple[str, str]]:
    """(heading, body) for every chunk of a full-text document."""
    return [((chunk.get("heading") or "").strip(), _clean_body(chunk))
            for chunk in doc_chunks]


def _scored_summary_entries(doc_chunks: list[dict], abstract: str,
                            chunk_scores: dict[str, float]) -> list[tuple[str, str]]:
    """(heading, body) for a summary document, chosen by SCORE rather than heading.

    For crawlers whose documents have no academic section vocabulary. Heading
    matching finds nothing in a manual or a report, which would leave those
    documents contributing only the metadata abstract — and for a non-academic
    document that abstract is itself just the first 200 words of body text, so
    the graph would see a snippet of every document past the full-text quota.

    Takes the document's best _SUMMARY_CHUNK_FRACTION of chunks by the ranking
    stage's BM25-against-cluster-keywords score, then restores document order so
    the model reads them the way they were written. Falls back to the heading
    path when the ranker produced no scores.
    """
    scored = [chunk for chunk in doc_chunks if chunk.get("id") in chunk_scores]
    if not scored:
        return _summary_entries(doc_chunks, abstract)

    keep_count = max(1, round(len(scored) * _SUMMARY_CHUNK_FRACTION))
    best_ids = {
        chunk["id"] for chunk in
        sorted(scored, key=lambda chunk: -chunk_scores[chunk["id"]])[:keep_count]
    }

    entries: list[tuple[str, str]] = []
    if abstract:
        entries.append(("Abstract", abstract))
    used_chars = len(abstract or "")
    for chunk in doc_chunks:                      # document order, not score order
        if chunk.get("id") not in best_ids:
            continue
        body = _clean_body(chunk)
        if not body:
            continue
        if entries and used_chars + len(body) > _SUMMARY_MAX_CHARS:
            break
        entries.append(((chunk.get("heading") or "").strip(), body))
        used_chars += len(body)

    if not entries and doc_chunks:
        entries.append(((doc_chunks[0].get("heading") or "").strip(),
                        _clean_body(doc_chunks[0])))
    return [(heading, body) for heading, body in entries if body]


def _summary_entries(doc_chunks: list[dict], abstract: str) -> list[tuple[str, str]]:
    """(heading, body) for a SUMMARY document: abstract + conclusion only.

    The title is not an entry — _render_call already states it as the header of
    every call, and repeating it in the body is exactly what made the title a
    node in the graph (see the module docstring).

    The abstract comes from doclings metadata when available; the matching
    abstract-headed chunks are then skipped so the same text isn't sent twice.
    It is also exempt from the _SUMMARY_MAX_CHARS budget — the budget exists to
    truncate an over-long conclusion, not to drop the abstract.

    A document with neither an abstract nor a conclusion heading falls back to
    its first chunk — the opening of a paper is abstract-like — so no document
    silently contributes nothing.
    """
    entries: list[tuple[str, str]] = []
    used_chars = 0
    if abstract:
        entries.append(("Abstract", abstract))
        used_chars += len(abstract)

    for chunk in doc_chunks:
        normalized = _norm_heading(chunk.get("heading") or "")
        if not any(pattern in normalized for pattern in _SUMMARY_HEADINGS):
            continue
        if abstract and _ABSTRACT_HEADING in normalized:
            continue        # already carried by the metadata abstract
        body = _clean_body(chunk)
        if not body:
            continue
        # Chunks arrive in document order, so this keeps the HEAD of the
        # conclusion — the argument — and drops whatever trails it.
        if entries and used_chars + len(body) > _SUMMARY_MAX_CHARS:
            break
        entries.append(((chunk.get("heading") or "").strip(), body))
        used_chars += len(body)

    if not entries and doc_chunks:
        entries.append(((doc_chunks[0].get("heading") or "").strip(),
                        _clean_body(doc_chunks[0])))
    return [(heading, body) for heading, body in entries if body]


def _call_batches(chunks: list[dict], full_ids: list[str], summary_ids: list[str],
                  abstracts: dict[str, str], max_chars: int,
                  chunk_scores: dict[str, float] | None = None) -> tuple[list[dict], int]:
    """Structured per-call batches for the whole corpus, bibliography chunks
    dropped. Documents are batched independently and emitted full-text first
    (rank order), so no batch ever spans two documents and the incremental flush
    covers the most important material earliest. Returns (batches, chunks_used),
    where chunks_used counts only the chunks actually sent."""
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        if _norm_heading(chunk.get("heading") or "") in _REF_HEADINGS:
            continue
        if not _clean_body(chunk):
            continue
        by_doc.setdefault(chunk.get("docId"), []).append(chunk)
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda chunk: chunk.get("chunkIndex", 0))

    full_text_ids = set(full_ids)
    batches: list[dict] = []
    chunks_used = 0
    for doc_id in list(full_ids) + list(summary_ids):
        doc_chunks = by_doc.get(doc_id)
        if not doc_chunks:
            continue
        is_full_text = doc_id in full_text_ids
        if is_full_text:
            entries = _full_text_entries(doc_chunks)
        elif _CRAWLER == "sapphire":
            # A paper's abstract and conclusion are the author's own summary of
            # the argument, which scoring is unlikely to improve on.
            entries = _summary_entries(doc_chunks, abstracts.get(doc_id, ""))
        else:
            entries = _scored_summary_entries(doc_chunks, abstracts.get(doc_id, ""),
                                              chunk_scores or {})
        entries = [(heading, body) for heading, body in entries if body]
        if not entries:
            continue
        # Full-text documents send every chunk; a summary document sends a
        # handful of entries, which is what makes it cheap.
        chunks_used += len(doc_chunks) if is_full_text else len(entries)
        batches.extend(_document_batches(_doc_title(doc_chunks[0]), entries,
                                         max_chars, doc_id))
    return batches, chunks_used


def _split_batch(batch: dict) -> list[dict]:
    """Halve a batch for retry after a validation failure. Splits on body
    boundaries when it has several (the common case); a single oversized body
    is split in two on word boundaries. Returns [batch] when it can't be split
    further (one small body) so the caller stops recursing."""
    bodies, headings, title = batch["bodies"], batch["headings"], batch["title"]
    doc_id = batch["docId"]
    if len(bodies) > 1:
        mid = len(bodies) // 2
        return [{"title": title, "headings": headings, "bodies": bodies[:mid], "docId": doc_id},
                {"title": title, "headings": headings, "bodies": bodies[mid:], "docId": doc_id}]
    if len(bodies) == 1 and len(bodies[0]) > _MIN_SPLIT_CHARS:
        halves = _split_oversized(bodies[0], math.ceil(len(bodies[0]) / 2))
        return [{"title": title, "headings": headings, "bodies": [half], "docId": doc_id}
                for half in halves]
    return [batch]


def _try_generate(kg: KGGen, profile: dict, text: str, temperature: float):
    """One kg-gen call at one temperature → Graph, or None on failure.

    num_ctx and temperature are set on the LM's kwargs (forwarded verbatim to
    LiteLLM and Ollama) rather than passed to generate(): passing temperature=
    makes kg-gen rebuild its LM WITHOUT num_ctx, silently dropping back to
    Ollama's default window. dspy keys its cache on these kwargs, so varying
    temperature here still defeats the temp-0 cache the way passing it would.

    num_ctx is Ollama-specific and only sent for local models — a hosted
    provider would reject the unknown parameter.
    """
    try:
        if profile["numCtx"]:
            kg.lm.kwargs["num_ctx"] = profile["numCtx"]
        # Hosted only: KGGen.__init__ refuses reasoning_effort for anything
        # outside the gpt-5 family, so it is set here on the LM kwargs (the same
        # back door num_ctx uses) rather than through the constructor.
        if REASONING_EFFORT and not profile["isLocal"]:
            kg.lm.kwargs["reasoning_effort"] = REASONING_EFFORT
        kg.lm.kwargs["temperature"] = temperature
        return kg.generate(input_data=text)
    except Exception as exc:  # dspy/pydantic ValidationError on malformed triples
        print(f"[kg_graph]     generate failed at temperature={temperature} "
              f"({type(exc).__name__})", file=sys.stderr)
        return None


def _generate_with_retry(kg: KGGen, profile: dict, batch: dict, depth: int = 0):
    """One batch → Graph, degrading gracefully on validation failure.

    The failure that matters here is structural, not stochastic: kg-gen asks
    the model for a typed list[Relation] and validates it as ONE unit, so a
    single malformed triple (e.g. an object emitted as a list) discards the
    whole list. That probability rises with input size — more text → more
    entities → a longer relation list → near-certain at least one bad element.
    A bigger context window does not help, and re-sampling at a new temperature
    usually reproduces it, so the temperature ladder alone would burn three
    full-length calls to drop every chunk in the batch.

    Instead: try temp 0 once; on failure SPLIT the batch in half and recurse,
    which shortens each relation list until it validates and salvages the
    pieces that do. Only when a batch is already a single small body — nothing
    left to split — fall back to the temperature ladder for the genuinely
    stochastic null-triple case the ladder was meant for. Sub-graphs from the
    halves are unioned. Returns None only when every piece fails.
    """
    graph = _try_generate(kg, profile, _batch_text(batch), _RETRY_TEMPERATURES[0])
    if graph is not None:
        return graph

    if depth < _MAX_SPLIT_DEPTH:
        halves = _split_batch(batch)
        if len(halves) > 1:
            print(f"[kg_graph]     splitting batch ({len(batch['bodies'])} bodies) "
                  f"and retrying halves", file=sys.stderr)
            merged = None
            for half in halves:
                sub = _generate_with_retry(kg, profile, half, depth + 1)
                if sub is not None:
                    merged = sub if merged is None else kg.aggregate([merged, sub])
            return merged

    # Unsplittable (one small body) and still failing: the stochastic case the
    # temperature ladder targets — re-sample at the higher temperatures.
    for temperature in _RETRY_TEMPERATURES[1:]:
        graph = _try_generate(kg, profile, _batch_text(batch), temperature)
        if graph is not None:
            return graph
    return None


def _norm_entity(value: str) -> str:
    """Whitespace-collapsed lowercase, for comparing entity strings."""
    return re.sub(r"\s+", " ", value).strip().lower()


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON via temp file + replace, retrying the replace on Windows.

    os.replace is atomic on both platforms, but on WINDOWS it raises
    PermissionError (WinError 5) when the destination is still OPEN in another
    process. That is not hypothetical here: the Node parent
    (routes/pipeline.js) reads graph.json on every _PROGRESS_MARKER to ingest
    the partial graph, and the final flush follows the last marker with no gap
    — so the collision is systematic rather than unlucky, and it killed a
    completed 32-call run at the very last write.

    The reader holds the handle for milliseconds, so a short backoff clears it.
    Retrying is safe: the temp file is fully written before the first attempt,
    so every attempt publishes identical, complete content.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                tmp_path.unlink(missing_ok=True)   # don't leave debris behind
                raise
            time.sleep(_REPLACE_BACKOFF_SECONDS * (attempt + 1))


def _salvage_totals() -> tuple[int, int, int]:
    """(dropped_elements, salvaged_fields, failed_parses) from the tolerant
    parser, or zeros when it's disabled. Recorded in graph.json so a run says
    how much malformed output it absorbed rather than hiding it."""
    if _TOLERANT_ADAPTER is None:
        return 0, 0, 0
    return (_TOLERANT_ADAPTER.dropped_elements,
            _TOLERANT_ADAPTER.salvaged_fields,
            _TOLERANT_ADAPTER.failed_parses)


def _tracker_totals(tracker) -> tuple[int, int, int]:
    """(prompt_tokens, output_tokens, model_requests) summed over the run's LM
    calls, read from dspy's usage tracker. Each usage entry is one real model
    request; dspy cache hits cost nothing and never reach the tracker."""
    prompt_tokens = output_tokens = model_requests = 0
    for entries in tracker.usage_data.values():
        model_requests += len(entries)
        for entry in entries:
            prompt_tokens += entry.get("prompt_tokens") or 0
            output_tokens += entry.get("completion_tokens") or 0
    return prompt_tokens, output_tokens, model_requests


def _write_graph_json(data_dir: Path, profile: dict, full_ids: list[str],
                      summary_ids: list[str], chunks_used: int, calls_total: int,
                      calls_failed: int, entities: set, relation_sources: dict,
                      edges: set, completed: int, drop_titles: frozenset = frozenset(),
                      prompt_tokens: int = 0, output_tokens: int = 0,
                      model_requests: int = 0) -> dict:
    """Build the payload from the running union and write graph.json atomically.

    Cleaning happens HERE, into a fresh payload, so the raw accumulator sets are
    never mutated between calls. Three filters:
      - blank entities and malformed triples the small model leaks past kg-gen's
        validation;
      - entities equal to a source document's TITLE, and any triple that touches
        one. The title sits in each call's header for grounding, and the model
        re-extracts it as an entity (confirmed: 'Attention Is All You Need' came
        back as a node). Stripping it from the chunk body wasn't enough — the
        header reintroduced it — so it is removed from the graph here. Headings
        are NOT dropped: some ('Scaled Dot-Product Attention') are real concepts.
      - author names lifted out of reference lists, dropped on the finished
        payload by sapphire/prune_authors.py.

    The write is temp-file + os.replace so a reader (the Node ingest) never sees
    a half-written file — os.replace is atomic on Windows and POSIX alike.
    'completed' < calls_total marks a partial, mid-run graph.
    """
    dropped_elements, salvaged_fields, failed_parses = _salvage_totals()
    ok = lambda value: (isinstance(value, str) and value.strip()
                        and _norm_entity(value) not in drop_titles)
    clean_entities  = sorted(entity for entity in entities if ok(entity))
    clean_relations = sorted([list(relation) for relation in relation_sources
                              if len(relation) == 3 and all(ok(part) for part in relation)])
    clean_edges     = sorted(edges)

    payload = {
        "createdAt":       datetime.now(timezone.utc).isoformat(),
        "model":           profile["litellmId"],
        # Context sizing this run used, so a graph records the budget it was
        # built under (the numbers move with KG_MODEL).
        "contextTokens":   profile["windowTokens"],
        "callMaxChars":    profile["callMaxChars"],
        # Every document that reached the graph, then the split: full-text docs
        # sent every chunk, summary docs sent title + abstract + conclusion.
        "sourceDocIds":    list(full_ids) + list(summary_ids),
        "fullTextDocIds":  list(full_ids),
        "summaryDocIds":   list(summary_ids),
        "chunksProcessed": chunks_used,
        # Failures are per CALL now, not per chunk — one failed call drops
        # every chunk packed into it, so the two are no longer interchangeable.
        "calls":           calls_total,
        "callsFailed":     calls_failed,
        # Progress: completed < calls means this is a partial, mid-run graph.
        "callsCompleted":  completed,
        "complete":        completed >= calls_total,
        # LM token usage so far (dspy usage tracker; cache hits aren't counted).
        "promptTokens":    prompt_tokens,
        "outputTokens":    output_tokens,
        "totalTokens":     prompt_tokens + output_tokens,
        "modelRequests":   model_requests,
        # Tolerant parsing (kg_adapter.py): malformed list elements dropped,
        # fields rescued that way, and parses it could not rescue.
        "tolerantParse":   _TOLERANT_PARSE,
        "droppedElements": dropped_elements,
        "salvagedFields":  salvaged_fields,
        "failedParses":    failed_parses,
        "entities":        clean_entities,
        "edges":           clean_edges,
        "relations":       clean_relations,
        # Index-aligned with relations: relationDocIds[i] lists every document
        # the i-th triple was extracted from. One batch never spans two
        # documents, so the attribution is exact rather than inferred.
        "relationDocIds":  aligned_sources(clean_relations, relation_sources),
    }

    # Runs on every flush, not just the last one, so the partial graphs on disk
    # and in the DB match the final one. Recomputes edges from the survivors.
    payload, prune_stats = prune_author_nodes(payload)
    payload["authorNodesDropped"] = prune_stats["authorNodes"]

    _write_json_atomic(data_dir / "graph.json", payload)
    return payload


def build_kg(data_dir: Path = DATA_DIR) -> dict:
    embeddings_path = data_dir / "embeddings.json"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"{embeddings_path} not found — run the embed stage first")
    chunks = json.loads(embeddings_path.read_text(encoding="utf-8")).get("chunks", [])
    if not chunks:
        raise ValueError("embeddings.json has no chunks — run the embed stage first")

    profile = _model_profile(KG_MODEL)
    if not profile["knownModel"]:
        print(f"[kg_graph] {KG_MODEL} is not in {MODEL_METADATA_PATH.name} — assuming a "
              f"{FALLBACK_CONTEXT_TOKENS}-token context", file=sys.stderr)

    # Rank decides WHICH documents get full text; every document is graphed.
    chunk_doc_ids = list(dict.fromkeys(chunk.get("docId") for chunk in chunks))
    ranked_ids = _ranked_doc_ids(data_dir)
    full_ids, summary_ids = _split_documents(chunk_doc_ids, ranked_ids, FULL_TEXT_FRACTION)
    # How many full-text slots the ranking actually decided; the rest were
    # filled in chunk-store order because the ranking ran out.
    rank_driven = len([doc_id for doc_id in full_ids if doc_id in set(ranked_ids)])
    if not ranked_ids:
        print("[kg_graph] no top-k ranking — full-text documents chosen in chunk-store order")
    elif rank_driven < len(full_ids):
        print(f"[kg_graph] ranking decided {rank_driven} of {len(full_ids)} full-text "
              f"slot(s); the rest were filled in chunk-store order — raise HEURISTIC_K "
              f"to keep the choice rank-driven", file=sys.stderr)
    summary_style = ("title+abstract+conclusion" if _CRAWLER == "sapphire"
                     else f"title + best {_SUMMARY_CHUNK_FRACTION:.0%} of chunks by score")
    print(f"[kg_graph] {len(chunk_doc_ids)} document(s): {len(full_ids)} full text "
          f"({FULL_TEXT_FRACTION:.0%}), {len(summary_ids)} {summary_style}")

    abstracts = _abstracts(data_dir)
    chunk_scores = {} if _CRAWLER == "sapphire" else _chunk_scores(data_dir)
    if _CRAWLER != "sapphire" and not chunk_scores:
        print("[kg_graph] no chunkScores in heuristic_output.json — summary documents "
              "fall back to heading matching, which finds little outside papers",
              file=sys.stderr)
    batches, chunks_used = _call_batches(chunks, full_ids, summary_ids,
                                         abstracts, profile["callMaxChars"],
                                         chunk_scores=chunk_scores)
    if not batches:
        raise ValueError("no chunk text available — nothing to build a graph from")

    packed = sum(len(_batch_text(batch)) for batch in batches)
    print(f"[kg_graph] {chunks_used} chunk(s) → {len(batches)} call(s) "
          f"({packed / len(batches):.0f} avg chars, budget {profile['callMaxChars']}, "
          f"window {profile['windowTokens']} tokens of {profile['contextTokens']}), "
          f"model={profile['litellmId']}")

    # Document titles sit in every call's header for grounding; the model
    # re-extracts them as entities, so drop them from the graph on write.
    drop_titles = frozenset(_norm_entity(batch["title"])
                            for batch in batches if batch["title"])

    data_dir.mkdir(parents=True, exist_ok=True)
    kg = _make_kg_client(profile)
    # Process-wide parser swap. Must be set before any Predict call; kg-gen's
    # own `with dspy.context(lm=...)` only overrides the LM, so this survives it.
    if _TOLERANT_ADAPTER is not None:
        dspy.configure(adapter=_TOLERANT_ADAPTER)
        print("[kg_graph] tolerant parsing ON — malformed relations are dropped "
              "element-wise instead of voiding the call")
    else:
        print("[kg_graph] tolerant parsing OFF (KG_TOLERANT_PARSE=0) — stock dspy parser")

    # Running union, not a list of every call's Graph: memory stays flat in the
    # number of calls, and the accumulated graph can be flushed to disk after
    # each call instead of only at the end. The raw sets are the accumulator;
    # cleaning happens per flush into a fresh payload, never mutating them.
    entities: set = set()
    # Relations are keyed by triple with their source documents as the value —
    # the same de-duplication a set gives, plus the provenance. A triple the
    # corpus states more than once accumulates one docId per paper, which is
    # what makes corroboration countable downstream.
    relation_sources: dict[tuple, set[str]] = {}
    edges: set = set()
    failed_calls = 0

    # track_usage records prompt/output tokens for every real LM call in the
    # block; flush reads the running total so token counts land in graph.json
    # next to the graph (and in each partial flush), not just at the end.
    with track_usage() as tracker:
        def flush(completed: int) -> dict:
            prompt_tokens, output_tokens, model_requests = _tracker_totals(tracker)
            return _write_graph_json(data_dir, profile, full_ids, summary_ids, chunks_used,
                                     len(batches), failed_calls, entities, relation_sources,
                                     edges, completed=completed, drop_titles=drop_titles,
                                     prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                                     model_requests=model_requests)

        for call_idx, batch in enumerate(batches):
            print(f"[kg_graph]   call {call_idx + 1}/{len(batches)} "
                  f"({len(_batch_text(batch))} chars, {len(batch['bodies'])} body/bodies)")
            call_graph = _generate_with_retry(kg, profile, batch)
            if call_graph is None:
                failed_calls += 1
                print(f"[kg_graph]   call {call_idx + 1} failed (even after splitting) — skipped",
                      file=sys.stderr)
                continue
            entities.update(call_graph.entities)
            for relation in call_graph.relations:
                relation_sources.setdefault(tuple(relation), set()).add(batch["docId"])
            edges.update(call_graph.edges)
            # Flush the graph-so-far after every successful call. The atomic write
            # means a crash (or a kill mid-run) leaves a valid partial graph, and
            # the marker line tells the Node parent to ingest it into the DB now —
            # the graph becomes progressively durable instead of all-or-nothing at
            # the end of a multi-hour run.
            flush(call_idx + 1)
            print(_PROGRESS_MARKER, flush=True)

        if not entities:
            raise RuntimeError(f"kg-gen produced no valid graph for any of {len(batches)} call(s)")

        # Final flush + the whole-graph visualization (regenerating it per call
        # would be wasted work — it is meaningless for a partial graph).
        payload = flush(len(batches))

    # Snapshot before the merge passes. Merging is a one-way door: once two
    # predicates collapse, the relations no longer record which of them they
    # used, so a threshold can never be re-tuned downward. This file is what
    # makes the merge passes re-runnable without paying for extraction again.
    _write_json_atomic(data_dir / "graph.raw.json", payload)
    print(f"[kg_graph] pre-merge snapshot → {data_dir / 'graph.raw.json'}")

    # Whole-graph cleanup, once per run. Both passes need the finished graph —
    # predicate clustering compares every predicate against every other — and
    # neither makes a model call, so they run here rather than per flush.
    payload, entity_stats = resolve_entities(payload)
    print(f"[kg_graph] entity resolution: {entity_stats['entitiesMerged']} entity/entities "
          f"merged across {entity_stats['groups']} group(s)")
    # After the lexical pass, so an acronym is matched against already-canonical
    # spellings. Silently a no-op when scripts/extract_abbreviations.py has not
    # been run for this collection — it lives in its own venv.
    payload, abbreviation_stats = merge_abbreviations(payload, data_dir)
    print(f"[kg_graph] abbreviations: {abbreviation_stats['entitiesMerged']} acronym(s) "
          f"folded into their expansion across {abbreviation_stats['groups']} group(s) "
          f"from {abbreviation_stats['definitions']} defined short form(s)")
    payload, predicate_stats = normalize_predicates(payload)
    print(f"[kg_graph] predicate normalization: {predicate_stats['predicatesMerged']} "
          f"predicate(s) merged across {predicate_stats['groups']} group(s)")
    _write_json_atomic(data_dir / "graph.json", payload)

    # The cluster maps drive more than a counter in the view: kg-gen colours
    # merged nodes by group and marks each representative, so they have to be
    # handed to Graph or the visualization reports every cluster as empty.
    visualize(Graph(entities=set(payload["entities"]),
                    relations={tuple(relation) for relation in payload["relations"]},
                    edges=set(payload["edges"]),
                    entity_clusters=payload.get("entityClusters"),
                    edge_clusters=payload.get("predicateClusters")),
              str(data_dir / "kg_view.html"), open_in_browser=False)

    print(f"[kg_graph] {len(payload['entities'])} entities, "
          f"{len(payload['relations'])} relations → {data_dir / 'graph.json'} "
          f"({payload['authorNodesDropped']} author node(s) pruned)")
    print(f"[kg_graph] {payload['totalTokens']} tokens "
          f"({payload['promptTokens']} prompt + {payload['outputTokens']} output) "
          f"over {payload['modelRequests']} model request(s)")
    if _TOLERANT_PARSE:
        print(f"[kg_graph] tolerant parse: {payload['droppedElements']} malformed "
              f"element(s) dropped across {payload['salvagedFields']} rescued field(s); "
              f"{payload['failedParses']} parse(s) fell through to the retry ladder")
    return payload


if __name__ == "__main__":
    try:
        build_kg()
    except Exception as exc:
        print(f"[kg_graph] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
