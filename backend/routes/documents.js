/**
 * routes/documents.js — per-collection PDF management, mounted at
 * /api/collections/:collectionId/documents (req.collection is set by the
 * collections router).
 *
 *   GET    /            — list the collection's documents
 *   POST   /            — upload one or more PDFs (multipart field "files")
 *   GET    /:docId/pdf  — stream the original PDF
 *   DELETE /:docId      — remove the document row + file (chunks cascade)
 *
 * Uploads are validated (magic bytes, size, page count, per-collection
 * duplicate hash) and stored under uploads/<collectionId>/; docId = sha256
 * prefix, so the same PDF gets the same id in every collection.
 */

import 'dotenv/config';
import { Router } from 'express';
import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import multer from 'multer';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';
import { prisma } from '../db.js';

const ROOT        = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const UPLOADS_DIR = path.resolve(ROOT, process.env.UPLOADS_DIR || 'uploads');
const MAX_FILE_SIZE = parseInt(process.env.MAX_PDF_SIZE_MB || '50', 10) * 1024 * 1024;

const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_FILE_SIZE, files: 20 },
});

/** Validate one uploaded buffer; returns {error} or the fields for a Document row. */
async function inspectPdf(file) {
  if (!file.originalname.toLowerCase().endsWith('.pdf')) {
    return { error: 'Not a .pdf file' };
  }
  if (file.buffer.subarray(0, 4).toString('ascii') !== '%PDF') {
    return { error: 'Not a valid PDF (bad magic bytes)' };
  }
  let pageCount;
  try {
    pageCount = (await pdfParse(file.buffer, { max: 0 })).numpages;
  } catch {
    return { error: 'PDF appears corrupted — could not parse it' };
  }
  const sha256 = crypto.createHash('sha256').update(file.buffer).digest('hex');
  return { sha256, docId: sha256.slice(0, 16), pageCount };
}

export const documentsRouter = Router();

documentsRouter.get('/', wrap(async (req, res) => {
  const docs = await prisma.document.findMany({
    where: { collectionId: req.collection.id },
    orderBy: { createdAt: 'asc' },
    select: {
      docId: true, filename: true, title: true, authors: true,
      status: true, pageCount: true, extractedAt: true, createdAt: true,
    },
  });
  res.json({ documents: docs });
}));

documentsRouter.post('/', upload.array('files'), wrap(async (req, res) => {
  const files = req.files || [];
  if (files.length === 0) throw httpError(400, 'No files uploaded (multipart field "files")');

  const collectionDir = path.join(UPLOADS_DIR, String(req.collection.id));
  await fs.mkdir(collectionDir, { recursive: true });

  const results = [];
  for (const file of files) {
    const inspected = await inspectPdf(file);
    if (inspected.error) {
      results.push({ filename: file.originalname, ok: false, error: inspected.error });
      continue;
    }
    const duplicate = await prisma.document.findFirst({
      where: { collectionId: req.collection.id, sha256: inspected.sha256 },
    });
    if (duplicate) {
      results.push({
        filename: file.originalname, ok: false,
        error: `Already in this collection as "${duplicate.filename}"`,
      });
      continue;
    }

    const filePath = path.join(collectionDir, `${inspected.docId}_${path.basename(file.originalname)}`);
    await fs.writeFile(filePath, file.buffer);
    const doc = await prisma.document.create({
      data: {
        docId:        inspected.docId,
        collectionId: req.collection.id,
        filename:  file.originalname,
        filePath,
        sha256:    inspected.sha256,
        pageCount: inspected.pageCount,
      },
    });
    results.push({ filename: file.originalname, ok: true, docId: doc.docId });
  }

  const uploaded = results.filter((result) => result.ok).length;
  res.status(uploaded ? 201 : 400).json({ uploaded, results });
}));

documentsRouter.get('/:docId/pdf', wrap(async (req, res) => {
  const doc = await prisma.document.findFirst({
    where: { collectionId: req.collection.id, docId: req.params.docId },
  });
  if (!doc) throw httpError(404, `Unknown document "${req.params.docId}"`);
  try {
    await fs.access(doc.filePath);
  } catch {
    throw httpError(404, `Source PDF missing on disk: ${doc.filename}`);
  }
  res.type('application/pdf');
  res.sendFile(path.resolve(doc.filePath));
}));

documentsRouter.delete('/:docId', wrap(async (req, res) => {
  const doc = await prisma.document.findFirst({
    where: { collectionId: req.collection.id, docId: req.params.docId },
  });
  if (!doc) throw httpError(404, `Unknown document "${req.params.docId}"`);
  await prisma.document.delete({ where: { id: doc.id } }); // chunks cascade
  await fs.rm(doc.filePath, { force: true }).catch(() => {});
  res.json({ ok: true, docId: doc.docId });
}));
