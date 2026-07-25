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
 *   3. embed      — chunk + MiniLM encode → Chunk rows
 *   4. categorize — clusters → Collection.categories/.docVectors + Chunk.category
 *   5. heuristic  — BM25 + PageRank top-k: printed to the server console
 *                   only (not persisted)
 *   6. graph      — kg-gen entity/relation graph over the embed stage's
 *                   chunks → Collection.knowledgeGraph + .knowledgeGraphHtml
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
  ingestDoclings, ingestChunks, ingestCategories, ingestGraph, ingestGraphProgress,
} from '../pipeline/collectionStore.js';

// ── Paths ─────────────────────────────────────────────────────────────────────

const __dirname  = path.dirname(fileURLToPath(import.meta.url));
const ROOT       = path.resolve(__dirname, '..', '..');
const EXTRACT_PY   = path.join(ROOT, 'backend', 'extraction', 'sapphire', 'extract.py');
const HEURISTIC_PY = path.join(ROOT, 'backend', 'extraction', 'sapphire', 'heuristic.py');
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
function spawnAsync(cmd, args, collectionId, { onLine } = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      cwd: ROOT,
      env: {
        ...process.env,
        DATA_DIR: scratchDir(collectionId),
        ENHANCED_DIR,
        // Don't drop .pyc files into backend/ — under `npm run dev:all` a new
        // file there restarts the watch server mid-run (see scripts/dev.mjs).
        PYTHONDONTWRITEBYTECODE: '1',
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

export const STAGE_ORDER = ['extract', 'embed', 'categorize', 'heuristic', 'graph'];

export const STAGES = {
  async extract(collection, { force = false } = {}) {
    await exportDocumentsMeta(collection);
    await exportDoclings(collection);           // lets extract.py skip done docs
    await spawnAsync(PYTHON, [EXTRACT_PY, ...(force ? ['--force'] : [])], collection.id);
    await annotateDois(scratchDir(collection.id));
    try {
      await enrichDoclings(scratchDir(collection.id));   // network enrichment — never fatal
    } catch (err) {
      console.warn(`[pipeline] Crossref enrichment skipped: ${err.message}`);
    }
    await ingestDoclings(collection);
    const report = await readScratchJson(collection.id, 'extract_report.json');
    return { extracted: report.extracted.length, skipped: report.skipped, errors: report.errors };
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
    await spawnAsync(PYTHON, [HEURISTIC_PY, '--k', String(k)], collection.id);
    const output = await readScratchJson(collection.id, 'heuristic_output.json');
    // Not persisted for now — printed here only.
    console.log(`[heuristic] collection ${collection.id} output:\n${JSON.stringify(output, null, 2)}`);
    return { k: output.k, topK: output.topK, edges: output.edges.length };
  },

  async graph(collection) {
    // kg_graph.py consumes the embed stage's chunks (packed into calls), so
    // the graph depends on embed: extract → embed → … → graph. It flushes a
    // partial graph.json and prints KG_GRAPH_SAVED after every call; each
    // marker ingests that partial into the DB, so a multi-hour run is saved
    // incrementally instead of only on completion.
    await exportEmbeddings(collection);

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

    await spawnAsync(PYTHON, [KG_GRAPH_PY], collection.id, { onLine });
    await ingestChain;              // let the last partial ingest settle
    await ingestGraph(collection);  // final: graph + kg_view.html, authoritative
    const graph = await readScratchJson(collection.id, 'graph.json');
    return {
      model:        graph.model,
      entities:     graph.entities.length,
      edges:        graph.edges.length,
      relations:    graph.relations.length,
      docs:         graph.sourceDocIds.length,
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

// POST /extract { force? }
pipelineRouter.post('/extract', wrap(async (req, res) => {
  res.json(await STAGES.extract(req.collection, { force: req.body?.force === true }));
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

// POST /heuristic { k? }
pipelineRouter.post('/heuristic', wrap(async (req, res) => {
  const k = parseInt(req.body?.k ?? process.env.HEURISTIC_K ?? '2', 10);
  if (!Number.isInteger(k) || k < 1) throw httpError(400, '"k" must be a positive integer');
  res.json(await STAGES.heuristic(req.collection, { k }));
}));

// POST /build-graph
pipelineRouter.post('/build-graph', wrap(async (req, res) => {
  res.json(await STAGES.graph(req.collection).catch(err => { throw toHttp(err); }));
}));

// POST /run { threshold?, k?, force? } — stages 2–6 in order; a failing stage
// stops the run and the response shows how far it got.
pipelineRouter.post('/run', wrap(async (req, res) => {
  const {
    threshold = parseFloat(process.env.CATEGORIES_SIMILARITY || '0.75'),
    k         = parseInt(process.env.HEURISTIC_K || '2', 10),
    force     = false,
  } = req.body ?? {};

  if (typeof threshold !== 'number' || threshold <= 0 || threshold > 1) {
    throw httpError(400, '"threshold" must be a number in (0, 1]');
  }

  const params = {
    extract:    { force },
    embed:      { force },
    categorize: { threshold },
    heuristic:  { k },
    graph:      {},
  };

  const stages = {};
  for (const stageName of STAGE_ORDER) {
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
