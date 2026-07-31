/**
 * routes/collections.js — /api/collections (login required; server.js
 * applies `requireAuth` before this router).
 *
 *   GET    /                — the owner's collections
 *   POST   /                — create {name (required), crawler?}; orb color
 *                             is auto-assigned from the series palette
 *   DELETE /:collectionId   — delete the collection and EVERYTHING under it:
 *                             chats/history, documents, chunks, artifacts
 *                             (DB cascade) + uploaded PDFs + scratch dir
 *
 * Sub-resources (ownership-checked by loadOwnedCollection → req.collection):
 *   /:collectionId/documents — upload/list/delete PDFs        (documents.js)
 *   /:collectionId/pipeline  — run extraction stages          (pipeline.js)
 *   /:collectionId/corpus    — embedding map / graph / chunks (corpus.js)
 */

import 'dotenv/config';
import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { prisma } from '../db.js';
import { documentsRouter } from './documents.js';
import { pipelineRouter } from './pipeline.js';
import { collectionCorpusRouter } from './corpus.js';
import { removeScratch } from '../pipeline/collectionStore.js';

const ROOT        = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const UPLOADS_DIR = path.resolve(ROOT, process.env.UPLOADS_DIR || 'uploads');
const CRAWLERS    = ['sapphire', 'ruby', 'topaz'];

// Orb palette — mirrors SERIES.colors in frontend/src/lib/theme.js.
const ORB_COLORS = ['#199e70', '#c98500', '#d55181', '#d95926', '#3987e5', '#008300', '#9085e9', '#e66767'];

const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

/** List/summary shape — leaves out the heavy JSON artifacts. */
const collectionSummary = (collection) => ({
  id:        collection.id,
  name:      collection.name,
  color:     collection.color,
  crawler:   collection.crawler,
  createdAt: collection.createdAt,
  documents: collection._count?.documents ?? undefined,
});

export const collectionsRouter = Router();

collectionsRouter.get('/', wrap(async (req, res) => {
  const collections = await prisma.collection.findMany({
    where: { userId: req.user.id },
    orderBy: { createdAt: 'asc' },
    include: { _count: { select: { documents: true } } },
  });
  res.json({ collections: collections.map(collectionSummary) });
}));

collectionsRouter.post('/', wrap(async (req, res) => {
  const { name, crawler = 'sapphire' } = req.body ?? {};
  if (typeof name !== 'string' || !name.trim()) throw httpError(400, '"name" is required');
  if (!CRAWLERS.includes(crawler)) {
    throw httpError(400, `"crawler" must be one of: ${CRAWLERS.join(', ')}`);
  }
  const existingCount = await prisma.collection.count({ where: { userId: req.user.id } });
  const collection = await prisma.collection.create({
    data: {
      name: name.trim(),
      crawler,
      color: ORB_COLORS[existingCount % ORB_COLORS.length],
      userId: req.user.id,
    },
  });
  res.status(201).json({ collection: collectionSummary(collection) });
}));

/** Everything below /:collectionId is ownership-checked here. */
const loadOwnedCollection = wrap(async (req, res, next) => {
  const collectionId = parseInt(req.params.collectionId, 10);
  if (!Number.isInteger(collectionId)) throw httpError(400, 'collectionId must be an integer');
  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId: req.user.id },
    // The rendered graph page is ~60KB and only one route wants it; that route
    // reads it directly rather than making every sub-resource carry it.
    omit: { knowledgeGraphHtml: true },
  });
  if (!collection) throw httpError(404, `No collection ${collectionId}`);
  req.collection = collection;
  next();
});

collectionsRouter.use('/:collectionId', loadOwnedCollection);

collectionsRouter.delete('/:collectionId', wrap(async (req, res) => {
  // DB cascade removes chats (+ their history), documents, and chunks.
  await prisma.collection.delete({ where: { id: req.collection.id } });
  await fs.rm(path.join(UPLOADS_DIR, String(req.collection.id)), { recursive: true, force: true })
    .catch(() => {});
  await removeScratch(req.collection.id);
  res.json({ ok: true, id: req.collection.id });
}));

collectionsRouter.use('/:collectionId/documents', documentsRouter);
collectionsRouter.use('/:collectionId/pipeline', pipelineRouter);
collectionsRouter.use('/:collectionId/corpus', collectionCorpusRouter);
