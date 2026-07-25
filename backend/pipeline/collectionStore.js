/**
 * pipeline/collectionStore.js — bridge between Postgres (canonical storage)
 * and a collection's scratch directory (the JSON exchange format the pipeline
 * stages read/write, unchanged from the old file-based layout).
 *
 * Every stage run is: export the stage's inputs from the DB into
 * data/collections/<collectionId>/, run the stage against that directory,
 * ingest the stage's output back into the DB. Scratch files are disposable.
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { Prisma } from '@prisma/client';
import { prisma } from '../db.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

/** Absolute scratch directory for one collection's pipeline runs. */
export const scratchDir = (collectionId) =>
  path.join(ROOT, 'data', 'collections', String(collectionId));

async function writeJson(filePath, data) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf-8');
}

export async function readScratchJson(collectionId, filename) {
  return JSON.parse(await fs.readFile(path.join(scratchDir(collectionId), filename), 'utf-8'));
}

/** Deletes a collection's scratch dir (used when the collection is deleted). */
export async function removeScratch(collectionId) {
  await fs.rm(scratchDir(collectionId), { recursive: true, force: true }).catch(() => {});
}

// ── Exports: DB → scratch ─────────────────────────────────────────────────────

/** documents.json — what extract.py iterates over. */
export async function exportDocumentsMeta(collection) {
  const rows = await prisma.document.findMany({ where: { collectionId: collection.id } });
  const documents = {};
  for (const doc of rows) {
    documents[doc.docId] = {
      docId:     doc.docId,
      filename:  doc.filename,
      filePath:  doc.filePath,
      sha256:    doc.sha256,
      pageCount: doc.pageCount,
      status:    doc.status,
    };
  }
  await writeJson(path.join(scratchDir(collection.id), 'documents.json'), { documents });
}

/** doclings.json — extracted content; lets extract.py skip already-done docs. */
export async function exportDoclings(collection) {
  const rows = await prisma.document.findMany({
    where: { collectionId: collection.id, docling: { not: Prisma.DbNull } },
  });
  const doclings = {};
  for (const doc of rows) doclings[doc.docId] = doc.docling;
  await writeJson(path.join(scratchDir(collection.id), 'doclings.json'), doclings);
}

/** embeddings.json — chunk store; lets embed.js skip fresh docs. */
export async function exportEmbeddings(collection) {
  const rows = await prisma.chunk.findMany({
    where: { collectionId: collection.id },
    orderBy: [{ docId: 'asc' }, { chunkIndex: 'asc' }],
  });
  const store = {
    chunks: rows.map((row) => ({
      id:           row.chunkId,
      docId:        row.docId,
      filename:     row.filename,
      pages:        row.pages,
      prefixLen:    row.prefixLen,
      chunkIndex:   row.chunkIndex,
      heading:      row.heading,
      sectionIndex: row.sectionIndex,
      chunkType:    row.chunkType,
      text:         row.text,
      embedding:    row.embedding,
      ingestedAt:   row.ingestedAt.toISOString(),
    })),
    metadata: collection.embeddingsMeta ?? {},
  };
  await writeJson(path.join(scratchDir(collection.id), 'embeddings.json'), store);
}

/** categories.json — cluster keywords (heuristic.py input). */
export async function exportCategories(collection) {
  if (!collection.categories) return;
  await writeJson(path.join(scratchDir(collection.id), 'categories.json'), collection.categories);
}

// ── Ingests: scratch → DB ─────────────────────────────────────────────────────

/** doclings.json → Document rows (docling body, title/authors, status). */
export async function ingestDoclings(collection) {
  const doclings = await readScratchJson(collection.id, 'doclings.json');
  for (const [docId, docling] of Object.entries(doclings)) {
    await prisma.document.updateMany({
      where: { collectionId: collection.id, docId },
      data: {
        docling,
        title:       docling.metadata?.title || null,
        authors:     docling.metadata?.authors || [],
        status:      'completed',
        extractedAt: docling.extractedAt ? new Date(docling.extractedAt) : new Date(),
      },
    });
  }
}

/** embeddings.json → Chunk rows (full replace) + Collection.embeddingsMeta. */
export async function ingestChunks(collection) {
  const store = await readScratchJson(collection.id, 'embeddings.json');
  const docs = await prisma.document.findMany({
    where: { collectionId: collection.id },
    select: { id: true, docId: true },
  });
  const documentIdByDocId = new Map(docs.map((doc) => [doc.docId, doc.id]));

  const rows = (store.chunks || [])
    .filter((chunk) => documentIdByDocId.has(chunk.docId))
    .map((chunk) => ({
      chunkId:      chunk.id,
      collectionId: collection.id,
      documentId:   documentIdByDocId.get(chunk.docId),
      docId:        chunk.docId,
      filename:     chunk.filename,
      chunkIndex:   chunk.chunkIndex,
      text:         chunk.text,
      heading:      chunk.heading ?? null,
      chunkType:    chunk.chunkType ?? null,
      sectionIndex: chunk.sectionIndex ?? null,
      pages:        chunk.pages ?? undefined,
      prefixLen:    chunk.prefixLen ?? 0,
      embedding:    chunk.embedding,
    }));

  await prisma.$transaction([
    prisma.chunk.deleteMany({ where: { collectionId: collection.id } }),
    prisma.chunk.createMany({ data: rows }),
    prisma.collection.update({
      where: { id: collection.id },
      data: { embeddingsMeta: store.metadata ?? {}, corpusUpdatedAt: new Date() },
    }),
  ]);
}

/** categories.json + doc_vectors.json → Collection fields + Chunk.category. */
export async function ingestCategories(collection) {
  const categories = await readScratchJson(collection.id, 'categories.json');
  const docVectors = await readScratchJson(collection.id, 'doc_vectors.json').catch(() => null);

  await prisma.collection.update({
    where: { id: collection.id },
    data: {
      categories,
      docVectors: docVectors ?? Prisma.DbNull,
      corpusUpdatedAt: new Date(),
    },
  });

  // Label every chunk with its document's cluster keywords.
  for (const category of categories.categories || []) {
    const label = (category.keywords || []).slice(0, 3).join(', ') || `cluster ${category.index}`;
    await prisma.chunk.updateMany({
      where: {
        collectionId: collection.id,
        docId: { in: category.members.map((member) => member.docId) },
      },
      data: { category: label },
    });
  }
}

/** graph.json + kg_view.html → Collection.knowledgeGraph/.knowledgeGraphHtml. */
export async function ingestGraph(collection) {
  const graph = await readScratchJson(collection.id, 'graph.json');
  const html = await fs.readFile(
    path.join(scratchDir(collection.id), 'kg_view.html'), 'utf-8');
  await prisma.collection.update({
    where: { id: collection.id },
    data: { knowledgeGraph: graph, knowledgeGraphHtml: html, corpusUpdatedAt: new Date() },
  });
}

/**
 * graph.json → Collection.knowledgeGraph, for the PARTIAL graph flushed after
 * each kg-gen call. Graph only — kg_view.html is a whole-graph render kg_graph.py
 * writes once at the end, so it isn't touched here (the last full ingestGraph
 * sets it). Tolerates a not-yet-written or torn read: kg_graph.py writes
 * graph.json atomically, but the very first marker can race the file, and a
 * mid-run partial must never fail the stage — the next call's flush supersedes
 * it, and the final ingestGraph is authoritative regardless.
 */
export async function ingestGraphProgress(collection) {
  let graph;
  try {
    graph = await readScratchJson(collection.id, 'graph.json');
  } catch {
    return;   // file not there yet / mid-write — skip this tick
  }
  await prisma.collection.update({
    where: { id: collection.id },
    data: { knowledgeGraph: graph, corpusUpdatedAt: new Date() },
  });
}
