"""
kg_graph.py — knowledge graph over a collection's documents, via kg-gen.

Replaces the old citation/section graph (build_graph.js). kg-gen prompts an
LLM for (subject, predicate, object) triples, so the graph holds concepts and
their relations instead of document structure.

The unit of extraction is a BATCH of consecutive chunks from ONE document,
packed greedily up to KG_CALL_MAX_CHARS. Chunks are the embed stage's docling
chunks (structure-aware — never crossing a section, let alone a document
boundary), so a batch is a run of adjacent text in its original order.

Packing rules:
  - Documents are never mixed in one call. Two papers in one prompt invite
    the model to emit triples linking a concept in paper A to one in paper B
    that no text asserts, and aggregate() below cannot tell those apart from
    real edges.
  - A chunk is never split across calls. The chunk that would overflow the
    budget starts the next call instead, which re-states the header. Only a
    chunk exceeding the budget ON ITS OWN is split (rare — chunks are ~1KB
    against a 6KB budget).
  - The embed stage's "title — heading" prefix is STRIPPED from every chunk
    (via prefixLen) and re-stated ONCE per call as a header. Embedded in
    every chunk it was both repetition and noise: the small model extracted
    the document title as an entity from every chunk, so the title became a
    first-class node in the graph. The header carries the same grounding at
    1/N the repetition. The merged prefix+body text is left untouched in the
    chunk store — retrieval is tuned on it (see chunker.js), and prefixLen
    lets the two consumers diverge without storing the text twice.

Budget choice is a reliability trade, not just speed: kg-gen validates each
call's relations as one typed list, so a single malformed triple from the
small model voids the whole call — a probability that rises with input size.
6KB validates reliably here; 12KB failed on every call. A call that still
fails is HALVED and retried (_generate_with_retry), shortening the relation
list until it validates and salvaging the pieces that do, rather than dropping
every chunk in the batch. The document title, which the model re-extracts from
the header, is filtered from the final graph (_write_graph_json).

Batch graphs are unioned with kg-gen's aggregate(). The stage depends on
embed: extract → embed → heuristic → graph.

Reads:
  <DATA_DIR>/embeddings.json        — chunk store (embed.js / chunker.js)
  <DATA_DIR>/heuristic_output.json  — top-k doc ranking (heuristic.py); when
                     present, only chunks of the TOP_DOCUMENTS highest-ranked
                     docs are graphed. Absent → every document (fallback).

Writes:
  <DATA_DIR>/graph.json      — {createdAt, model, entities, edges, relations,
                                sourceDocIds, chunksProcessed, calls,
                                callsFailed, callsCompleted, complete}.
                                Rewritten atomically after EVERY call, not just
                                at the end — a crash leaves a valid partial
                                graph. callsCompleted < calls and complete=false
                                mark a mid-run partial.
  <DATA_DIR>/kg_view.html    — standalone interactive visualization (kg-gen),
                                written once at the end (whole-graph render)

After each per-call flush the marker line _PROGRESS_MARKER is printed to
stdout; the Node parent (routes/pipeline.js) watches for it and ingests the
partial graph.json into Postgres, saving the graph once per call.

Env:
  KG_MODEL           Ollama model tag; a bare tag is prefixed with
                     'ollama_chat/' for LiteLLM, which kg-gen routes through.
  OLLAMA_URL         default http://localhost:11434
  TOP_DOCUMENTS      how many ranked docs to graph (default 2)
  KG_CALL_MAX_CHARS  chars of header+body packed into one call (default 6000);
                     a call that fails validation is halved and retried
  KG_MAX_SPLIT_DEPTH how many times a failing call is halved (default 3)
  KG_NUM_CTX         Ollama context window in tokens (default 8192). MUST be
                     large enough for KG_CALL_MAX_CHARS plus the entity list
                     and the model's own output, or Ollama silently drops the
                     front of the prompt — a smaller graph with no error.
  KG_RETRY_TEMPERATURES  retry ladder for malformed-triple failures

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
from datetime import datetime, timezone
from pathlib import Path

from kg_gen import KGGen
from kg_gen.models import Graph
from kg_gen.utils.visualize_kg import visualize
from dspy.utils.usage_tracker import track_usage

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

KG_MODEL   = os.environ.get("KG_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# How many of the heuristic's ranked documents the graph is built from. Capped
# by however many heuristic.py actually emitted (HEURISTIC_K).
TOP_DOCUMENTS = int(os.environ.get("TOP_DOCUMENTS", "2"))
# Chars of header + body packed into one kg-gen call. Bigger is faster per
# char (the model's output grows sublinearly with input), BUT kg-gen validates
# each call's relations as one typed list, and a single malformed triple voids
# the whole list — a probability that climbs with input size. Measured on this
# corpus with the 3B model: ~6KB calls validate reliably, 12KB calls failed
# every time (40 entities → a long relation list → one bad element). 6000 is
# the reliable-throughput knee; _generate_with_retry splits any call that still
# fails, so this is a target, not a hard ceiling.
CALL_MAX_CHARS = int(os.environ.get("KG_CALL_MAX_CHARS", "6000"))
# A batch that fails validation is halved and retried (see _generate_with_retry)
# up to this depth, then a single small body falls back to the temperature
# ladder. 3 halvings take a 6KB call down to <1KB.
_MAX_SPLIT_DEPTH = int(os.environ.get("KG_MAX_SPLIT_DEPTH", "3"))
# Don't split a single body below this — smaller than a chunk gains nothing and
# a null-triple failure at this size is the temperature ladder's job.
_MIN_SPLIT_CHARS = int(os.environ.get("KG_MIN_SPLIT_CHARS", "1500"))
# Ollama context window, in tokens. Nothing set this before, so calls ran at
# Ollama's default (2048-4096) — fine for one ~1KB chunk, far too small once
# chunks are packed. At ~4 chars/token, CALL_MAX_CHARS=12000 is ~3000 tokens
# of prompt; the window must also hold the entity list and the output.
# Raising this is NOT free: Ollama sizes its KV cache to num_ctx, and on a
# small GPU that cache evicts model layers (measured here: 15/27 layers on
# GPU at 2048, only 5/27 at 16384, decode 13.5 -> 9.0 tok/s). 8192 is the
# balance point — it holds a packed call with headroom and keeps 10/27 layers
# resident. Lower it if the model is fully GPU-resident and you want speed.
NUM_CTX = int(os.environ.get("KG_NUM_CTX", "8192"))
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


def _norm_heading(heading: str) -> str:
    """Lowercase, drop leading numbering ('7. References'), strip trailing punctuation."""
    normalized = re.sub(r'^[\divxlc]+[\.\)]?\s+', '', heading.lower().strip())
    return normalized.rstrip(' .:')


def _litellm_model(model: str) -> str:
    """kg-gen calls LiteLLM, which needs a provider prefix; bare tags are Ollama."""
    return model if "/" in model else f"ollama_chat/{model}"


def _top_k_doc_ids(data_dir: Path) -> list[str] | None:
    """docIds of the heuristic's top-k, in rank order; None when the ranking
    isn't available (no heuristic run yet, or an unreadable/empty file) so the
    caller falls back to every document."""
    heuristic_path = data_dir / "heuristic_output.json"
    if not heuristic_path.exists():
        return None
    try:
        top_k = json.loads(heuristic_path.read_text(encoding="utf-8")).get("topK", [])
    except (json.JSONDecodeError, ValueError):
        return None
    doc_ids = [entry["docId"] for entry in top_k if entry.get("docId")]
    return doc_ids or None


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


def _document_batches(doc_chunks: list[dict], max_chars: int) -> list[dict]:
    """Greedy pack of ONE document's chunks into batches.

    Fills a batch with consecutive chunks until the next one would push it past
    max_chars; that chunk starts the next batch (which re-states the header)
    rather than being split. A chunk that exceeds max_chars on its own — no
    packing can help it — is split into multiple bodies under the same header.
    """
    title = _doc_title(doc_chunks[0])
    batches: list[dict] = []
    bodies: list[str] = []
    headings: list[str] = []

    def flush() -> None:
        nonlocal bodies, headings
        if bodies:
            batches.append({"title": title, "headings": headings, "bodies": bodies})
        bodies, headings = [], []

    for chunk in doc_chunks:
        body = _clean_body(chunk)
        if not body:
            continue
        heading = (chunk.get("heading") or "").strip()
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
                batches.append({"title": title, "headings": trial_headings, "bodies": [piece]})
            continue
        bodies.append(body)
        headings = trial_headings
    flush()
    return batches


def _call_batches(chunks: list[dict], selected_doc_ids: list[str],
                  max_chars: int) -> tuple[list[dict], int]:
    """Structured per-call batches for the selected docs, bibliography chunks
    dropped. Documents are batched independently and emitted in rank order, so
    no batch ever spans two documents. Returns (batches, chunks_used)."""
    selected = set(selected_doc_ids)
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        if chunk.get("docId") not in selected:
            continue
        if _norm_heading(chunk.get("heading") or "") in _REF_HEADINGS:
            continue
        if not _clean_body(chunk):
            continue
        by_doc.setdefault(chunk["docId"], []).append(chunk)

    batches: list[dict] = []
    chunks_used = 0
    for doc_id in selected_doc_ids:
        doc_chunks = by_doc.get(doc_id)
        if not doc_chunks:
            continue
        doc_chunks.sort(key=lambda chunk: chunk.get("chunkIndex", 0))
        chunks_used += len(doc_chunks)
        batches.extend(_document_batches(doc_chunks, max_chars))
    return batches, chunks_used


def _split_batch(batch: dict) -> list[dict]:
    """Halve a batch for retry after a validation failure. Splits on body
    boundaries when it has several (the common case); a single oversized body
    is split in two on word boundaries. Returns [batch] when it can't be split
    further (one small body) so the caller stops recursing."""
    bodies, headings, title = batch["bodies"], batch["headings"], batch["title"]
    if len(bodies) > 1:
        mid = len(bodies) // 2
        return [{"title": title, "headings": headings, "bodies": bodies[:mid]},
                {"title": title, "headings": headings, "bodies": bodies[mid:]}]
    if len(bodies) == 1 and len(bodies[0]) > _MIN_SPLIT_CHARS:
        halves = _split_oversized(bodies[0], math.ceil(len(bodies[0]) / 2))
        return [{"title": title, "headings": headings, "bodies": [half]} for half in halves]
    return [batch]


def _try_generate(kg: KGGen, text: str, temperature: float):
    """One kg-gen call at one temperature → Graph, or None on failure.

    num_ctx and temperature are set on the LM's kwargs (forwarded verbatim to
    LiteLLM and Ollama) rather than passed to generate(): passing temperature=
    makes kg-gen rebuild its LM WITHOUT num_ctx, silently dropping back to
    Ollama's default window. dspy keys its cache on these kwargs, so varying
    temperature here still defeats the temp-0 cache the way passing it would.
    """
    try:
        kg.lm.kwargs["num_ctx"] = NUM_CTX
        kg.lm.kwargs["temperature"] = temperature
        return kg.generate(input_data=text)
    except Exception as exc:  # dspy/pydantic ValidationError on malformed triples
        print(f"[kg_graph]     generate failed at temperature={temperature} "
              f"({type(exc).__name__})", file=sys.stderr)
        return None


def _generate_with_retry(kg: KGGen, batch: dict, depth: int = 0):
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
    graph = _try_generate(kg, _batch_text(batch), _RETRY_TEMPERATURES[0])
    if graph is not None:
        return graph

    if depth < _MAX_SPLIT_DEPTH:
        halves = _split_batch(batch)
        if len(halves) > 1:
            print(f"[kg_graph]     splitting batch ({len(batch['bodies'])} bodies) "
                  f"and retrying halves", file=sys.stderr)
            merged = None
            for half in halves:
                sub = _generate_with_retry(kg, half, depth + 1)
                if sub is not None:
                    merged = sub if merged is None else kg.aggregate([merged, sub])
            return merged

    # Unsplittable (one small body) and still failing: the stochastic case the
    # temperature ladder targets — re-sample at the higher temperatures.
    for temperature in _RETRY_TEMPERATURES[1:]:
        graph = _try_generate(kg, _batch_text(batch), temperature)
        if graph is not None:
            return graph
    return None


def _norm_entity(value: str) -> str:
    """Whitespace-collapsed lowercase, for comparing entity strings."""
    return re.sub(r"\s+", " ", value).strip().lower()


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


def _write_graph_json(data_dir: Path, model: str, selected_ids: list[str],
                      chunks_used: int, calls_total: int, calls_failed: int,
                      entities: set, relations: set, edges: set,
                      completed: int, drop_titles: frozenset = frozenset(),
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

    The write is temp-file + os.replace so a reader (the Node ingest) never sees
    a half-written file — os.replace is atomic on Windows and POSIX alike.
    'completed' < calls_total marks a partial, mid-run graph.
    """
    ok = lambda value: (isinstance(value, str) and value.strip()
                        and _norm_entity(value) not in drop_titles)
    clean_entities  = sorted(entity for entity in entities if ok(entity))
    clean_relations = sorted([list(relation) for relation in relations
                              if len(relation) == 3 and all(ok(part) for part in relation)])
    clean_edges     = sorted(edges)

    payload = {
        "createdAt":       datetime.now(timezone.utc).isoformat(),
        "model":           model,
        "sourceDocIds":    selected_ids,
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
        "entities":        clean_entities,
        "edges":           clean_edges,
        "relations":       clean_relations,
    }

    tmp_path = data_dir / "graph.json.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, data_dir / "graph.json")
    return payload


def build_kg(data_dir: Path = DATA_DIR) -> dict:
    embeddings_path = data_dir / "embeddings.json"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"{embeddings_path} not found — run the embed stage first")
    chunks = json.loads(embeddings_path.read_text(encoding="utf-8")).get("chunks", [])
    if not chunks:
        raise ValueError("embeddings.json has no chunks — run the embed stage first")

    # Doc order: heuristic rank when available, else chunk-store order.
    chunk_doc_ids = list(dict.fromkeys(chunk.get("docId") for chunk in chunks))
    top_k_ids = _top_k_doc_ids(data_dir)
    if top_k_ids:
        selected_ids = [doc_id for doc_id in top_k_ids
                        if doc_id in set(chunk_doc_ids)][:TOP_DOCUMENTS]
        print(f"[kg_graph] top-k ranking found — graphing {len(selected_ids)} "
              f"of {len(chunk_doc_ids)} document(s) (TOP_DOCUMENTS={TOP_DOCUMENTS})")
    else:
        selected_ids = chunk_doc_ids
        print(f"[kg_graph] no top-k ranking — graphing all {len(selected_ids)} document(s)")

    batches, chunks_used = _call_batches(chunks, selected_ids, CALL_MAX_CHARS)
    if not batches:
        raise ValueError("no chunk text available — nothing to build a graph from")

    model = _litellm_model(KG_MODEL)
    packed = sum(len(_batch_text(batch)) for batch in batches)
    print(f"[kg_graph] {chunks_used} chunk(s) → {len(batches)} call(s) "
          f"({packed / len(batches):.0f} avg chars, budget {CALL_MAX_CHARS}, "
          f"num_ctx={NUM_CTX}), model={model}")

    # Document titles sit in every call's header for grounding; the model
    # re-extracts them as entities, so drop them from the graph on write.
    drop_titles = frozenset(_norm_entity(batch["title"])
                            for batch in batches if batch["title"])

    data_dir.mkdir(parents=True, exist_ok=True)
    kg = KGGen(model=model, api_base=OLLAMA_URL, api_key="ollama", temperature=0.0)

    # Running union, not a list of every call's Graph: memory stays flat in the
    # number of calls, and the accumulated graph can be flushed to disk after
    # each call instead of only at the end. The raw sets are the accumulator;
    # cleaning happens per flush into a fresh payload, never mutating them.
    entities: set = set()
    relations: set = set()
    edges: set = set()
    failed_calls = 0

    # track_usage records prompt/output tokens for every real LM call in the
    # block; flush reads the running total so token counts land in graph.json
    # next to the graph (and in each partial flush), not just at the end.
    with track_usage() as tracker:
        def flush(completed: int) -> dict:
            prompt_tokens, output_tokens, model_requests = _tracker_totals(tracker)
            return _write_graph_json(data_dir, model, selected_ids, chunks_used,
                                     len(batches), failed_calls, entities, relations,
                                     edges, completed=completed, drop_titles=drop_titles,
                                     prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                                     model_requests=model_requests)

        for call_idx, batch in enumerate(batches):
            print(f"[kg_graph]   call {call_idx + 1}/{len(batches)} "
                  f"({len(_batch_text(batch))} chars, {len(batch['bodies'])} chunk(s))")
            call_graph = _generate_with_retry(kg, batch)
            if call_graph is None:
                failed_calls += 1
                print(f"[kg_graph]   call {call_idx + 1} failed (even after splitting) — skipped",
                      file=sys.stderr)
                continue
            entities.update(call_graph.entities)
            relations.update(call_graph.relations)
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
    visualize(Graph(entities=set(payload["entities"]),
                    relations={tuple(relation) for relation in payload["relations"]},
                    edges=set(payload["edges"])),
              str(data_dir / "kg_view.html"), open_in_browser=False)

    print(f"[kg_graph] {len(payload['entities'])} entities, "
          f"{len(payload['relations'])} relations → {data_dir / 'graph.json'}")
    print(f"[kg_graph] {payload['totalTokens']} tokens "
          f"({payload['promptTokens']} prompt + {payload['outputTokens']} output) "
          f"over {payload['modelRequests']} model request(s)")
    return payload


if __name__ == "__main__":
    try:
        build_kg()
    except Exception as exc:
        print(f"[kg_graph] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
