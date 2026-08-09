/**
 * routes/chat.js — /api/chats/:chatId/chat  (req.chat with its collection
 * included, and req.user, set upstream)
 *
 * RAG chat over the chat's collection. The frontend embeds the user's
 * question in the browser (same MiniLM model as the corpus) and sends the
 * vector along; retrieval + Ollama answering happen here (retriever/).
 *
 * POST /
 *   Request:  { content: string, queryEmbedding: number[] }
 *   Response: { reply, model, sources: [{chunkId, docId, filename, heading, pages, sim, boost, score, quotes}] }
 *             sources[n-1] is what a [n] citation marker in reply refers to;
 *             quotes = verbatim spans in reply that this source grounds (the
 *             PDF viewer highlights exactly these when the citation is clicked)
 *   Errors:   400 bad payload · 502 Ollama failure · 503 missing corpus/model
 *
 * The conversation persists on the Chat row: both the user message and the
 * assistant reply (with sources) are appended to Chat.conversation.
 *
 * chat_log.txt (prompt → retrieved-context → response trios at the repo root)
 * is written for the ADMIN USER ONLY.
 */

import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { retrieve, retrieveFacts, answer } from '../retriever/retriever.js';
import { prisma } from '../db.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
// Repo root, NOT backend/: the dev watcher watches backend/ (--watch-path),
// so a log written there would restart the server on every chat.
const CHAT_LOG_PATH = path.join(ROOT, 'chat_log.txt');

/**
 * Append one prompt → retrieved context → response trio to chat_log.txt.
 * Fire-and-forget: a logging failure must never fail the chat itself.
 */
function logChatTrio({ question, chunks, reply, model }) {
  const context = chunks
    .map((chunk, chunkIdx) =>
      `[${chunkIdx + 1}] ${chunk.filename}${chunk.heading ? ` — ${chunk.heading}` : ''}` +
      ` (score ${chunk.score})\n${chunk.text}`)
    .join('\n\n');
  const entry = [
    '='.repeat(78),
    `[${new Date().toISOString()}] model=${model}`,
    '',
    '--- prompt ---',
    question,
    '',
    '--- retrieved context ---',
    context || '(no chunks retrieved)',
    '',
    '--- response ---',
    reply,
    '', '',
  ].join('\n');
  return fs.appendFile(CHAT_LOG_PATH, entry, 'utf-8')
    .catch((err) => console.warn('[chat] could not write chat_log.txt:', err.message));
}

const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

export const chatRouter = Router();

chatRouter.post('/', wrap(async (req, res) => {
  const { content, queryEmbedding, docIds } = req.body ?? {};

  if (typeof content !== 'string' || !content.trim()) {
    return res.status(400).json({ error: '"content" must be a non-empty string' });
  }
  if (!Array.isArray(queryEmbedding)
      || queryEmbedding.some((component) => typeof component !== 'number')) {
    return res.status(400).json({ error: '"queryEmbedding" must be a number array (computed in the browser)' });
  }
  // Optional: the documents this question is about. Present, it is a hard
  // retrieval scope — no other document reaches the model. Omitted, the
  // retriever infers document names from the question and only boosts them.
  if (docIds !== undefined
      && (!Array.isArray(docIds) || docIds.some((docId) => typeof docId !== 'string' || !docId.trim()))) {
    return res.status(400).json({ error: '"docIds" must be an array of document id strings' });
  }

  // LLM history = stored conversation (roles + text only) + the new question.
  const conversation = Array.isArray(req.chat.conversation) ? req.chat.conversation : [];
  const messages = [
    ...conversation.map(({ role, content: text }) => ({ role, content: text })),
    { role: 'user', content },
  ];

  const chunks = await retrieve(req.chat.collection, queryEmbedding, content, { docIds });
  // Graph facts are additive: chunk retrieval is untouched, and a collection
  // with no graph (or a question naming no known entity) answers exactly as before.
  const { facts } = await retrieveFacts(req.chat.collection, content);
  const { reply, model, quotesByChunk } = await answer(messages, chunks, facts);

  if (req.user.isAdmin) logChatTrio({ question: content, chunks, reply, model });

  // Full text + embedding stay server-side (text is large; embedding is a
  // grounding artifact the browser has no use for).
  const sources = chunks.map(({ text, embedding, ...sourceMeta }, chunkIdx) =>
    ({ ...sourceMeta, quotes: quotesByChunk[chunkIdx] }));

  await prisma.chat.update({
    where: { id: req.chat.id },
    data: {
      conversation: [
        ...conversation,
        { role: 'user', content },
        { role: 'assistant', content: reply, sources },
      ],
      // First exchange titles the chat after the question.
      ...(conversation.length === 0 && req.chat.title === 'New chat'
        ? { title: content.trim().slice(0, 60) }
        : {}),
    },
  });

  res.json({ reply, model, sources });
}));
