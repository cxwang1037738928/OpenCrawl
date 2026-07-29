/**
 * corpus.js — corpus data for the frontend, read from Postgres.
 *
 * collectionCorpusRouter — mounted at /api/collections/:collectionId/corpus
 * (req.collection set by the collections router):
 *   GET /embedding-map   — the collection's document points projected to 3D
 *         and 2D (UMAP over Collection.docVectors) plus mutual-kNN edges with
 *         cosine similarities. The frontend re-runs union-find over these
 *         edges as the threshold slider moves, reproducing the backend
 *         clustering exactly with no server round-trip.
 *   GET /graph           — Collection.knowledgeGraph passthrough (kg-gen
 *         entities/edges/relations).
 *   GET /graph/view      — kg-gen's standalone interactive page, as HTML.
 *   GET /chunks/:chunkId — one indexed chunk (text, pages, prefixLen) so the
 *         viewer can locate and highlight it in the PDF.
 *
 * modelsRouter — mounted at /api/corpus (global, not chat-scoped):
 *   GET  /models   — installed Ollama models, the knowledge-graph model
 *         catalog (documents/model_metadata.json), and current per-role choices.
 *   POST /settings — persist per-role model choices to .env and process.env.
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import express from 'express';
import { UMAP } from 'umap-js';
import { prisma } from '../db.js';

const ROOT     = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const ENV_PATH = path.join(ROOT, '.env');
// The knowledge-graph model catalog: [{id, name, context_length}]. kg_graph.py
// reads the same file to size its calls to the selected model's context window,
// so a model is switchable exactly when it is listed here.
const MODEL_METADATA_PATH = path.resolve(
  ROOT, process.env.MODEL_METADATA_PATH || 'documents/model_metadata.json');

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const MUTUAL_K   = parseInt(process.env.CATEGORIES_MUTUAL_K || '10', 10);

// Model roles the frontend may configure. Keys are the .env variable names;
// descriptions surface in the Models tab.
export const MODEL_ROLES = {
  METADATA_MODEL:         'Title/author/abstract fallback when GROBID is down (extract.py)',
  EXTRACTION_MODEL:       'Other extraction tasks (extract.py)',
  QUERY_CLASSIFIER_MODEL: 'Query classification (parse_user_query.js)',
  KG_MODEL:               'Knowledge-graph construction (kg-gen, kg_graph.py)',
  REASONING_MODEL:        'Answer synthesis over retrieved chunks (/api/chat)',
};

const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

// ---------------------------------------------------------------------------
// Embedding map
// ---------------------------------------------------------------------------

// Vectors are L2-normalized (generate_categories.js), so dot = cosine.
function dot(vecA, vecB) {
  let sum = 0;
  for (let componentIdx = 0; componentIdx < vecA.length; componentIdx++) {
    sum += vecA[componentIdx] * vecB[componentIdx];
  }
  return sum;
}

// Deterministic PRNG (mulberry32) so the UMAP layout is stable across
// requests and reloads instead of reshuffling on every fetch.
function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state |= 0; state = (state + 0x6D2B79F5) | 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

/** Per-axis min-max scale to [-1, 1] so the scene size is data-independent. */
function normalizeCoords(coords) {
  const dims = coords[0].length;
  for (let axis = 0; axis < dims; axis++) {
    let min = Infinity, max = -Infinity;
    for (const point of coords) {
      min = Math.min(min, point[axis]);
      max = Math.max(max, point[axis]);
    }
    const span = max - min || 1;
    for (const point of coords) point[axis] = ((point[axis] - min) / span) * 2 - 1;
  }
  return coords;
}

function project(vectors, nComponents, seed) {
  const docCount = vectors.length;
  if (docCount === 1) return [nComponents === 3 ? [0, 0, 0] : [0, 0]];
  // umap-js requires nNeighbors < docCount — impossible with 2 points, so
  // place them apart directly instead of crashing the embedding map.
  if (docCount === 2) {
    return nComponents === 3 ? [[-1, 0, 0], [1, 0, 0]] : [[-1, 0], [1, 0]];
  }
  const umap = new UMAP({
    nComponents,
    // UMAP requires nNeighbors < docCount; 15 is the library default sweet spot.
    nNeighbors: Math.max(2, Math.min(15, docCount - 1)),
    minDist: 0.15,
    random: mulberry32(seed),
  });
  return normalizeCoords(umap.fit(vectors));
}

/** Mutual-kNN pairs with cosine sims — the same gate generate_categories.js
 * clusters with, so browser-side union-find over these edges reproduces the
 * backend clustering at any threshold. O(n²) sims, O(n·K) shipped. */
function mutualKnnEdges(vectors, k) {
  const docCount = vectors.length;
  const nearestNeighbours = [];
  for (let docIdx = 0; docIdx < docCount; docIdx++) {
    const sims = [];
    for (let otherIdx = 0; otherIdx < docCount; otherIdx++) {
      if (otherIdx !== docIdx) sims.push([otherIdx, dot(vectors[docIdx], vectors[otherIdx])]);
    }
    sims.sort((simA, simB) => simB[1] - simA[1]);
    nearestNeighbours.push(new Map(sims.slice(0, k)));
  }
  const edges = [];
  for (let docIdx = 0; docIdx < docCount; docIdx++) {
    for (const [otherIdx, sim] of nearestNeighbours[docIdx]) {
      if (otherIdx > docIdx && nearestNeighbours[otherIdx].has(docIdx)) {
        edges.push({ i: docIdx, j: otherIdx, sim: Math.round(sim * 10000) / 10000 });
      }
    }
  }
  return edges;
}

export const collectionCorpusRouter = express.Router();

// Projection is the expensive part — memoize per collection on the vectors'
// generatedAt stamp so repeat requests are free until the corpus changes.
const mapCacheByCollection = new Map();

collectionCorpusRouter.get('/embedding-map', wrap(async (req, res) => {
  const docVectors = req.collection.docVectors;
  if (!docVectors?.docs?.length) {
    throw httpError(404, 'No document vectors — run the categorize stage first');
  }

  const cached = mapCacheByCollection.get(req.collection.id);
  if (cached?.generatedAt !== docVectors.generatedAt) {
    const vectors = docVectors.docs.map((doc) => doc.vector);
    const p3 = project(vectors, 3, 1337);
    const p2 = project(vectors, 2, 1337);
    mapCacheByCollection.set(req.collection.id, {
      generatedAt: docVectors.generatedAt,
      mutualK: MUTUAL_K,
      defaultThreshold: parseFloat(process.env.CATEGORIES_SIMILARITY || '0.75'),
      points: docVectors.docs.map((doc, docIdx) => ({
        docId:    doc.docId,
        filename: doc.filename,
        title:    doc.title || doc.filename,
        p3:       p3[docIdx].map((coord) => Math.round(coord * 1000) / 1000),
        p2:       p2[docIdx].map((coord) => Math.round(coord * 1000) / 1000),
      })),
      edges: mutualKnnEdges(vectors, MUTUAL_K),
    });
  }
  res.json(mapCacheByCollection.get(req.collection.id));
}));

collectionCorpusRouter.get('/graph', wrap(async (req, res) => {
  if (!req.collection.knowledgeGraph) {
    throw httpError(404, 'No knowledge graph — run the build-graph stage first');
  }
  res.json(req.collection.knowledgeGraph);
}));

// kg-gen's standalone page. Sent as text, not a static file: collections are
// per-user, and an iframe src can't carry the auth header this route needs.
collectionCorpusRouter.get('/graph/view', wrap(async (req, res) => {
  // Not on req.collection — loadOwnedCollection omits it (see collections.js).
  const { knowledgeGraphHtml } = await prisma.collection.findUniqueOrThrow({
    where:  { id: req.collection.id },
    select: { knowledgeGraphHtml: true },
  });
  if (!knowledgeGraphHtml) {
    throw httpError(404, 'No knowledge graph — run the build-graph stage first');
  }
  res.type('html').send(knowledgeGraphHtml);
}));

collectionCorpusRouter.get('/chunks/:chunkId', wrap(async (req, res) => {
  const chunk = await prisma.chunk.findFirst({
    where: { collectionId: req.collection.id, chunkId: req.params.chunkId },
  });
  if (!chunk) throw httpError(404, `Unknown chunk "${req.params.chunkId}"`);
  res.json({
    id:           chunk.chunkId,
    docId:        chunk.docId,
    filename:     chunk.filename,
    pages:        chunk.pages,
    prefixLen:    chunk.prefixLen,
    chunkIndex:   chunk.chunkIndex,
    heading:      chunk.heading,
    sectionIndex: chunk.sectionIndex,
    chunkType:    chunk.chunkType,
    text:         chunk.text,
  });
}));

// ---------------------------------------------------------------------------
// Models (global — not chat-scoped)
// ---------------------------------------------------------------------------

export const modelsRouter = express.Router();

async function ollamaModels() {
  try {
    const abortController = new AbortController();
    const timeoutTimer = setTimeout(() => abortController.abort(), 3000);
    const response = await fetch(`${OLLAMA_URL}/api/tags`, { signal: abortController.signal });
    clearTimeout(timeoutTimer);
    if (!response.ok) return null;
    const body = await response.json();
    return (body.models || []).map((model) => model.name).sort();
  } catch {
    return null; // Ollama down — the frontend falls back to free-text input
  }
}

/**
 * Models the graph stage knows the context window of. Returned so the Models
 * tab can offer them as a pick-list instead of free text: an unlisted model
 * still works, but kg_graph.py has to assume a small context for it.
 */
async function knowledgeGraphModels() {
  try {
    const catalog = JSON.parse(await fs.readFile(MODEL_METADATA_PATH, 'utf-8'));
    return (catalog.models || [])
      .filter((model) => model?.id)
      .map((model) => ({
        id:            model.id,
        name:          model.name || model.id,
        contextLength: model.context_length ?? null,
      }));
  } catch {
    return [];   // missing/malformed catalog — the tab falls back to free text
  }
}

modelsRouter.get('/models', wrap(async (req, res) => {
  const [installed, kgModels] = await Promise.all([ollamaModels(), knowledgeGraphModels()]);
  const roles = {};
  for (const key of Object.keys(MODEL_ROLES)) roles[key] = process.env[key] || null;
  res.json({
    ollamaUp: installed !== null,
    ollamaUrl: OLLAMA_URL,
    installed: installed || [],
    kgModels,
    roles,
    descriptions: MODEL_ROLES,
  });
}));

/**
 * Persist role→model choices. .env stays the single source of truth: existing
 * KEY= lines are replaced in place (comments and ordering preserved), missing
 * keys are appended, and process.env is updated so pipeline spawns inherit the
 * new values without a server restart. Temp-file + rename so a crash mid-write
 * can't truncate .env.
 */
modelsRouter.post('/settings', wrap(async (req, res) => {
  const updates = req.body || {};
  const roleKeys = Object.keys(updates);
  if (roleKeys.length === 0) throw httpError(400, 'Empty settings payload');

  for (const roleKey of roleKeys) {
    if (!(roleKey in MODEL_ROLES)) throw httpError(400, `Unknown setting "${roleKey}"`);
    const modelName = updates[roleKey];
    if (typeof modelName !== 'string' || !modelName.trim() || /[\r\n]/.test(modelName)) {
      throw httpError(400, `Invalid value for ${roleKey}`);
    }
  }

  let envContents;
  try {
    envContents = await fs.readFile(ENV_PATH, 'utf-8');
  } catch {
    throw httpError(500, '.env not found at project root');
  }

  const envLines = envContents.split(/\r?\n/);
  const replacedKeys = new Set();
  for (let lineIdx = 0; lineIdx < envLines.length; lineIdx++) {
    const keyMatch = envLines[lineIdx].match(/^([A-Z_][A-Z0-9_]*)=/);
    if (keyMatch && roleKeys.includes(keyMatch[1])) {
      envLines[lineIdx] = `${keyMatch[1]}=${updates[keyMatch[1]].trim()}`;
      replacedKeys.add(keyMatch[1]);
    }
  }
  const appendKeys = roleKeys.filter((roleKey) => !replacedKeys.has(roleKey));
  if (appendKeys.length) {
    if (envLines[envLines.length - 1]?.trim() !== '') envLines.push('');
    envLines.push('# Models set via the frontend Models tab');
    for (const roleKey of appendKeys) envLines.push(`${roleKey}=${updates[roleKey].trim()}`);
  }

  const tempPath = `${ENV_PATH}.tmp`;
  await fs.writeFile(tempPath, envLines.join('\n'), 'utf-8');
  await fs.rename(tempPath, ENV_PATH);

  for (const roleKey of roleKeys) process.env[roleKey] = updates[roleKey].trim();

  const roles = {};
  for (const key of Object.keys(MODEL_ROLES)) roles[key] = process.env[key] || null;
  res.json({ ok: true, roles });
}));
