/**
 * routes/pipeline.js — /api/collections/:collectionId/pipeline/*  (req.collection set upstream)
 *
 * Controls the extraction/ranking pipeline for ONE collection. Postgres is
 * the canonical store; every stage runs against the collection's scratch
 * directory (data/collections/<collectionId>/) — inputs are exported from the DB before the stage,
 * outputs ingested back after (pipeline/collectionStore.js). The stage
 * implementations themselves are unchanged and file-based.
 *
 * Stage order and persistence:
 *   1. enhance    — per-document page enhancement (excluded from /run)
 *   2. extract    — docling PDF → Document.docling rows
 *   3. citations  — SAPPHIRE ONLY: reference lists → Collection.citationGraph
 *   4. embed      — chunk + MiniLM encode → Chunk rows
 *   5. categorize — clusters → Collection.categories/.docVectors + Chunk.category
 *   6. heuristic  — BM25 (+ PageRank when a citation graph exists) top-k:
 *                   printed to the server console only (not persisted)
 *   7. graph      — kg-gen entity/relation graph over the embed stage's
 *                   chunks → Collection.knowledgeGraph + .knowledgeGraphHtml
 *
 * Which stages run depends on the collection's CRAWLER — see indexStagesFor.
 * Only sapphire's documents carry reference lists, so only sapphire builds a
 * citation graph; other crawlers rank on BM25 alone and skip GROBID entirely.
 *
 * /run covers the indexing stages ONLY. The graph is its own endpoint,
 * /build-graph, because it is a different kind of job: indexing is minutes and
 * every later stage depends on it, while the graph is a long LLM run that
 * nothing else depends on and that you re-run on its own when you change
 * KG_MODEL or the full-text fraction. Bundling them meant no one could re-index
 * without paying for the graph, and the frontend exposes them as two buttons.
 */

import 'dotenv/config';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn }  from 'child_process';
import { Router } from 'express';

import { annotateDois }       from '../extraction/sapphire/doi_regex.js';
import { enrichDoclings }     from '../extraction/sapphire/search_doi.js';
import { embedAll }           from '../extraction/embed.js';
import { generateCategories } from '../extraction/generate_categories.js';
import { processDocument }    from '../parser/cleaning/enhance_pdf.js';
import { prisma } from '../db.js';
import {
  scratchDir, readScratchJson,
  exportDocumentsMeta, exportDoclings, exportEmbeddings, exportCategories,
  ingestDoclings, ingestChunks, ingestCategories, ingestCitations,
  ingestGraph, ingestGraphProgress,
} from '../pipeline/collectionStore.js';

// ── Paths ─────────────────────────────────────────────────────────────────────

const __dirname  = path.dirname(fileURLToPath(import.meta.url));
const ROOT       = path.resolve(__dirname, '..', '..');
const EXTRACT_PY   = path.join(ROOT, 'backend', 'extraction', 'sapphire', 'extract.py');
// Ranking is crawler-agnostic and lives one level up; the citation graph it
// used to build is sapphire-only and now has its own stage.
const HEURISTIC_PY = path.join(ROOT, 'backend', 'extraction', 'heuristic.py');
const CITATIONS_PY = path.join(ROOT, 'backend', 'extraction', 'sapphire', 'citations.py');
const KG_GRAPH_PY  = path.join(ROOT, 'backend', 'extraction', 'kg_graph.py');
const PYTHON     = process.env.PYTHON || 'python';
// Enhancement reports are keyed by docId (content hash), so one global dir is
// shared by all collections — the same PDF gets the same report everywhere.
const ENHANCED_DIR = path.resolve(ROOT, process.env.ENHANCED_DIR || 'data/enhanced');

// ── Helpers ───────────────────────────────────────────────────────────────────

const wrap = fn => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

// Line emitted by kg_graph.py after each per-call flush of graph.json (mirrors
// _PROGRESS_MARKER there). Seeing it on stdout means a partial graph is on disk
// and ready to ingest.
const KG_GRAPH_SAVED = '@@KG_GRAPH_SAVED@@';

/**
 * Spawn a child process against a collection's scratch dir; reject (502) on
 * exit≠0. onLine, when given, is called once per complete stdout line (used to
 * catch the KG_GRAPH_SAVED marker) — stdout is still forwarded verbatim for
 * display, this only adds line-boundary detection on top.
 */
function spawnAsync(cmd, args, collectionId, { onLine, crawler, forceOcr } = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      cwd: ROOT,
      env: {
        ...process.env,
        DATA_DIR: scratchDir(collectionId),
        // Which crawler this run belongs to. extract.py consults GROBID only
        // for sapphire; kg_graph.py picks its summary strategy from it.
        CRAWLER: crawler || 'sapphire',
        // Per-run, so one collection can be re-OCR'd while another indexes
        // normally. Only spelled out when set — otherwise extract.py's own
        // default (off) applies, and an inherited '' would read as unset anyway.
        ...(forceOcr ? { EXTRACT_FORCE_OCR: '1' } : {}),
        ENHANCED_DIR,
        // Don't drop .pyc files into backend/ — under `npm run dev:all` a new
        // file there restarts the watch server mid-run (see scripts/dev.mjs).
        PYTHONDONTWRITEBYTECODE: '1',
        // Python block-buffers stdout when it is a pipe, so per-document
        // progress sat in an 8KB buffer while a multi-hour stage ran: the log
        // showed one document while the worker was many further on, and a
        // healthy run was indistinguishable from a hung one.
        PYTHONUNBUFFERED: '1',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stderr = '';
    let lineBuf = '';
    proc.stdout.on('data', stdoutChunk => {
      const text = stdoutChunk.toString();
      process.stdout.write(text);              // unchanged live passthrough
      if (!onLine) return;
      lineBuf += text;                         // reassemble lines across chunks
      let newlineIdx;
      while ((newlineIdx = lineBuf.indexOf('\n')) !== -1) {
        onLine(lineBuf.slice(0, newlineIdx));
        lineBuf = lineBuf.slice(newlineIdx + 1);
      }
    });
    proc.stderr.on('data', stderrChunk => {
      const text = stderrChunk.toString();
      stderr += text;
      process.stderr.write(text);
    });

    proc.on('close', exitCode => {
      if (exitCode !== 0) {
        const err = new Error(`${path.basename(cmd)} exited with code ${exitCode}`);
        err.status = 502;
        err.detail = stderr.slice(-500);
        return reject(err);
      }
      resolve();
    });

    proc.on('error', err => {
      err.status = 502;
      reject(err);
    });
  });
}

/** Missing-input errors become 503 (run the earlier stage first). */
function toHttp(err) {
  if (!err.status && /not found/i.test(err.message)) err.status = 503;
  return err;
}

// ── Stage runners ─────────────────────────────────────────────────────────────
//
// Each runner takes (collection, params), runs export → stage → ingest, and returns
// the JSON summary the route responds with. /run reuses them verbatim.

// What /run executes, in order. 'graph' is deliberately NOT here — see the
// file header. STAGE_ORDER is the full dependency chain, for anything that
// needs to reason about the pipeline as a whole.
//
// Per crawler, because the stages a corpus needs depend on what its documents
// ARE: 'citations' matches reference lists against corpus titles and DOIs, which
// only sapphire's documents have. Declared as lists rather than decided by
// conditionals inside the runners so what actually executes stays readable.
// Citations must follow extract — doi_regex.js and search_doi.js populate
// crossrefReferences there, and that is the DOI phase's only input.
const SAPPHIRE_INDEX_STAGES = ['extract', 'citations', 'embed', 'categorize', 'heuristic'];
const GENERAL_INDEX_STAGES  = ['extract', 'embed', 'categorize', 'heuristic'];

export const indexStagesFor = (crawler) =>
  (crawler === 'sapphire' ? SAPPHIRE_INDEX_STAGES : GENERAL_INDEX_STAGES);

// Superset, for callers reasoning about the pipeline as a whole (route
// registration, status). No collection runs every one of these.
export const INDEX_STAGE_ORDER = SAPPHIRE_INDEX_STAGES;
export const STAGE_ORDER = [...INDEX_STAGE_ORDER, 'graph'];

export const STAGES = {
  async extract(collection, { force = false, forceOcr = false } = {}) {
    await exportDocumentsMeta(collection);
    await exportDoclings(collection);           // lets extract.py skip done docs
    await spawnAsync(PYTHON, [EXTRACT_PY, ...(force ? ['--force'] : [])], collection.id,
                     { crawler: collection.crawler, forceOcr });
    // DOI regex + Crossref lookup are academic enrichment: they resolve
    // reference strings to registered works, which is meaningless for a
    // document that has no reference list.
    if (collection.crawler === 'sapphire') {
      await annotateDois(scratchDir(collection.id));
      try {
        await enrichDoclings(scratchDir(collection.id));   // network enrichment — never fatal
      } catch (err) {
        console.warn(`[pipeline] Crossref enrichment skipped: ${err.message}`);
      }
    }
    await ingestDoclings(collection);
    const report = await readScratchJson(collection.id, 'extract_report.json');
    return { extracted: report.extracted.length, skipped: report.skipped, errors: report.errors };
  },

  // Sapphire only. Pure function of doclings, so it needs nothing but the
  // extract stage's output — and produces the citations.json the ranker blends
  // PageRank from when it exists.
  async citations(collection) {
    await exportDoclings(collection);
    await spawnAsync(PYTHON, [CITATIONS_PY], collection.id, { crawler: collection.crawler });
    await ingestCitations(collection);
    const output = await readScratchJson(collection.id, 'citations.json');
    return { edges: output.edges.length };
  },

  async embed(collection, { force = false } = {}) {
    await exportDoclings(collection);
    await exportEmbeddings(collection);         // lets embed.js skip fresh docs
    await embedAll({ force, dataDir: scratchDir(collection.id) });
    await ingestChunks(collection);
    const embeddingStore = await readScratchJson(collection.id, 'embeddings.json');
    return {
      chunks:     embeddingStore.chunks.length,
      docs:       embeddingStore.metadata.totalDocs,
      model:      embeddingStore.metadata.model,
      dimensions: embeddingStore.metadata.dimensions,
    };
  },

  async categorize(collection, { threshold } = {}) {
    await exportDoclings(collection);
    const result = await generateCategories(threshold, scratchDir(collection.id));
    await ingestCategories(collection);
    return {
      threshold:  result.threshold,
      categories: result.categories.length,
      docs:       result.categories.reduce(
        (total, category) => total + category.members.length, 0),
    };
  },

  async heuristic(collection, { k = parseInt(process.env.HEURISTIC_K || '2', 10) } = {}) {
    await exportDoclings(collection);
    await exportCategories(collection);
    // Chunk text, no vectors: the ranker scores every chunk against its
    // cluster's keywords and never looks at an embedding, and the float arrays
    // are the great majority of this file's bytes.
    await exportEmbeddings(collection, { vectors: false });
    // The graph stage reads this ranking to decide which documents it reads in
    // full (the top KG_FULL_TEXT_FRACTION of the corpus). A ranking shorter
    // than that quota leaves kg_graph.py filling the gap in arbitrary order, so
    // rank at least that many here — k stays whichever is larger, and an
    // explicit k from the caller can still raise it.
    const documentCount = await prisma.document.count({ where: { collectionId: collection.id } });
    const graphNeeds = Math.ceil(documentCount * parseFloat(process.env.KG_FULL_TEXT_FRACTION || '0.4'));
    const rankedK = Math.max(k, graphNeeds);
    await spawnAsync(PYTHON, [HEURISTIC_PY, '--k', String(rankedK)], collection.id,
                     { crawler: collection.crawler });
    const output = await readScratchJson(collection.id, 'heuristic_output.json');
    // Not persisted for now — printed here only. chunkScores is omitted from
    // the log: it holds one entry per chunk (16k+ on a large corpus) and would
    // bury the ranking it accompanies.
    const { chunkScores, ...loggable } = output;
    console.log(`[heuristic] collection ${collection.id} output:\n${JSON.stringify(loggable, null, 2)}`);
    return { k: output.k, topK: output.topK,
             chunksScored: Object.keys(chunkScores || {}).length };
  },

  async graph(collection) {
    // kg_graph.py consumes the embed stage's chunks (packed into calls), so
    // the graph depends on embed: extract → embed → … → graph. It flushes a
    // partial graph.json and prints KG_GRAPH_SAVED after every call; each
    // marker ingests that partial into the DB, so a multi-hour run is saved
    // incrementally instead of only on completion.
    await exportEmbeddings(collection);
    // Doclings too: the documents graphed as title+abstract+conclusion take
    // their abstract from docling metadata, because a paper's abstract is
    // usually front matter rather than a section, so it never became a chunk.
    await exportDoclings(collection);

    // Serialize the per-call ingests onto one chain: two Collection.update
    // calls never overlap, and a transient failure is swallowed (the next
    // flush supersedes it; the final ingestGraph below is authoritative).
    let ingestChain = Promise.resolve();
    const onLine = (line) => {
      if (!line.includes(KG_GRAPH_SAVED)) return;
      ingestChain = ingestChain
        .then(() => ingestGraphProgress(collection))
        .catch(err => console.warn(`[pipeline] partial graph ingest skipped: ${err.message}`));
    };

    await spawnAsync(PYTHON, [KG_GRAPH_PY], collection.id,
                     { onLine, crawler: collection.crawler });
    await ingestChain;              // let the last partial ingest settle
    await ingestGraph(collection);  // final: graph + kg_view.html, authoritative
    const graph = await readScratchJson(collection.id, 'graph.json');
    return {
      model:        graph.model,
      entities:     graph.entities.length,
      edges:        graph.edges.length,
      relations:    graph.relations.length,
      docs:         graph.sourceDocIds.length,
      // How the corpus was split: full-text documents sent every chunk,
      // summarized ones only title + abstract + conclusion.
      fullTextDocs: graph.fullTextDocIds?.length ?? graph.sourceDocIds.length,
      summaryDocs:  graph.summaryDocIds?.length ?? 0,
      chunks:       graph.chunksProcessed,
      // Chunks are packed many-per-call now, so failures count CALLS: one
      // failed call drops every chunk batched into it.
      calls:        graph.calls,
      callsFailed:  graph.callsFailed,
    };
  },
};

// ── Router ────────────────────────────────────────────────────────────────────

export const pipelineRouter = Router();

// GET /status — last-run timestamp per stage from the DB (null = never ran).
pipelineRouter.get('/status', wrap(async (req, res) => {
  const newestDoc = await prisma.document.findFirst({
    where: { collectionId: req.collection.id, extractedAt: { not: null } },
    orderBy: { extractedAt: 'desc' },
    select: { extractedAt: true },
  });
  res.json({
    doclings:   newestDoc?.extractedAt ?? null,
    embeddings: req.collection.embeddingsMeta?.updated ?? null,
    categories: req.collection.categories?.generatedAt ?? null,
    citations:  req.collection.citationGraph?.generatedAt ?? null,
    heuristic:  null,   // printed only, not persisted
    graph:      req.collection.knowledgeGraph?.createdAt ?? null,
  });
}));

// POST /enhance { docId, dpi? } — per-document page enhancement; run before
// extract so extract.py can route scanned pages to the OCR converter.
pipelineRouter.post('/enhance', wrap(async (req, res) => {
  const { docId, dpi = 300 } = req.body ?? {};
  if (!docId) throw httpError(400, '"docId" is required');

  const doc = await prisma.document.findFirst({ where: { collectionId: req.collection.id, docId } });
  if (!doc) throw httpError(404, `Document "${docId}" not found`);

  res.json(await processDocument(doc.filePath, { docId, dpi }));
}));

// POST /extract { force?, forceOcr? }
pipelineRouter.post('/extract', wrap(async (req, res) => {
  res.json(await STAGES.extract(req.collection, {
    force:    req.body?.force === true,
    forceOcr: req.body?.forceOcr === true,
  }));
}));

// POST /embed { force? }
pipelineRouter.post('/embed', wrap(async (req, res) => {
  res.json(await STAGES.embed(req.collection, { force: req.body?.force === true })
    .catch(err => { throw toHttp(err); }));
}));

// POST /categorize { threshold } — cosine similarity in (0, 1], required.
pipelineRouter.post('/categorize', wrap(async (req, res) => {
  const { threshold } = req.body ?? {};
  if (typeof threshold !== 'number' || isNaN(threshold) || threshold <= 0 || threshold > 1) {
    throw httpError(400, '"threshold" must be a number in (0, 1]');
  }
  res.json(await STAGES.categorize(req.collection, { threshold }).catch(err => { throw toHttp(err); }));
}));

// POST /citations — sapphire only; the corpus citation graph on its own.
pipelineRouter.post('/citations', wrap(async (req, res) => {
  if (req.collection.crawler !== 'sapphire') {
    throw httpError(400, `Citation graphs need reference lists — the ${req.collection.crawler} `
                       + 'crawler does not extract them');
  }
  res.json(await STAGES.citations(req.collection).catch(err => { throw toHttp(err); }));
}));

// POST /heuristic { k? }
pipelineRouter.post('/heuristic', wrap(async (req, res) => {
  const k = parseInt(req.body?.k ?? process.env.HEURISTIC_K ?? '2', 10);
  if (!Number.isInteger(k) || k < 1) throw httpError(400, '"k" must be a positive integer');
  res.json(await STAGES.heuristic(req.collection, { k }));
}));

// POST /build-graph — the knowledge-graph run on its own (its own frontend
// button). Long: an LLM call per packed batch of chunks.
pipelineRouter.post('/build-graph', wrap(async (req, res) => {
  res.json(await STAGES.graph(req.collection).catch(err => { throw toHttp(err); }));
}));

// POST /run { threshold?, k?, force?, forceOcr? } — the INDEXING stages (2–5) in
// order; a failing stage stops the run and the response shows how far it got.
// The graph is not part of this — POST /build-graph runs it.
pipelineRouter.post('/run', wrap(async (req, res) => {
  const {
    threshold = parseFloat(process.env.CATEGORIES_SIMILARITY || '0.75'),
    k         = parseInt(process.env.HEURISTIC_K || '2', 10),
    force     = false,
    forceOcr  = false,
  } = req.body ?? {};

  if (typeof threshold !== 'number' || threshold <= 0 || threshold > 1) {
    throw httpError(400, '"threshold" must be a number in (0, 1]');
  }

  const params = {
    extract:    { force, forceOcr },
    citations:  {},
    embed:      { force },
    categorize: { threshold },
    heuristic:  { k },
  };

  const stages = {};
  // Which stages this corpus needs depends on its crawler — a general-document
  // collection has no references, so no citation stage.
  for (const stageName of indexStagesFor(req.collection.crawler)) {
    try {
      // Re-read the collection row: earlier stages update fields later ones export.
      const collection = await prisma.collection.findUnique({ where: { id: req.collection.id } });
      stages[stageName] = { ok: true, ...(await STAGES[stageName](collection, params[stageName])) };
    } catch (err) {
      stages[stageName] = { ok: false, error: err.message };
      break;
    }
  }

  res.json({ stages });
}));
