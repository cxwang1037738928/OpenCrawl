/**
 * test_KG_pass.js — compare the node sets of two knowledge graphs
 *
 * Loads two of the demo user's collections and reports the nodes each graph has
 * that the OTHER is missing. A node counts as present in the other graph if some
 * node there matches it exactly, or embeds within SIMILARITY_THRESHOLD cosine —
 * so 'Encoder' vs 'encoder' or 'self-attention' vs 'self attention' are not
 * flagged as missing. Missing nodes from both directions are written to
 * tests/KG_log.txt as one table.
 *
 * Run:  node tests/test_KG_pass.js <collectionA> <collectionB>
 *   Each argument is a collection id (number) or a case-insensitive substring of
 *   its name. With no arguments, lists the demo user's collections and exits.
 *
 * Needs Postgres up (the collections live there) and downloads the embedding
 * model on first run. Nodes are embedded with the same model the pipeline uses
 * for chunks, so the two share a vector space.
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from '@xenova/transformers';
import { prisma } from '../backend/db.js';

const ROOT       = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const LOG_PATH   = path.join(ROOT, 'tests', 'KG_log.txt');
const DEMO_EMAIL = 'demo@gmail.com';
// A node is present in the other graph on an exact match, or on cosine >= this.
const SIMILARITY_THRESHOLD = 0.8;
// Same model embed.js uses for chunks, so node vectors land in its space.
const MODEL = process.env.SAPPHIRE_EMBEDDING_MODEL || 'Xenova/all-MiniLM-L12-v2';
// Labels encoded per forward pass — mirrors EMBED_BATCH_SIZE, caps peak memory.
const BATCH_SIZE = 32;

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

let _extractor = null;

// Node labels -> normalized vectors (so cosine is a plain dot product).
async function embed(labels) {
  if (!_extractor) {
    console.log(`[test_KG_pass] loading ${MODEL} ...`);
    _extractor = await pipeline('feature-extraction', MODEL, { quantized: true });
  }
  const vectors = [];
  for (let batchStart = 0; batchStart < labels.length; batchStart += BATCH_SIZE) {
    const batch = labels.slice(batchStart, batchStart + BATCH_SIZE);
    const output = await _extractor(batch, { pooling: 'mean', normalize: true });
    const dims = output.data.length / batch.length;
    for (let labelIdx = 0; labelIdx < batch.length; labelIdx++) {
      vectors.push(Array.from(output.data.slice(labelIdx * dims, (labelIdx + 1) * dims)));
    }
  }
  return vectors;
}

function cosine(vectorA, vectorB) {
  let dot = 0;
  for (let dim = 0; dim < vectorA.length; dim++) dot += vectorA[dim] * vectorB[dim];
  return dot;
}

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

// selector is a numeric id or a case-insensitive name substring.
function resolveCollection(collections, selector) {
  const byId = collections.find((collection) => String(collection.id) === selector);
  if (byId) return byId;
  const matches = collections.filter((collection) =>
    collection.name.toLowerCase().includes(selector.toLowerCase()));
  if (matches.length === 1) return matches[0];
  if (matches.length === 0) throw new Error(`no collection matches "${selector}"`);
  throw new Error(`"${selector}" is ambiguous: `
    + matches.map((collection) => `#${collection.id} ${collection.name}`).join('; '));
}

// A collection's graph nodes, deduped on trimmed text (first surface form kept).
function nodesOf(collection) {
  const entities = collection.knowledgeGraph?.entities;
  if (!Array.isArray(entities)) {
    throw new Error(`collection #${collection.id} "${collection.name}" has no knowledgeGraph.entities`);
  }
  const byText = new Map();
  for (const entity of entities) {
    const text = String(entity).trim();
    if (text && !byText.has(text)) byText.set(text, text);
  }
  return [...byText.values()];
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

// Nodes in `fromNodes` missing from `otherNodes`: no exact match AND best cosine
// below the threshold. Returns [{ node, nearest, similarity }].
function missingNodes(fromNodes, fromVectors, otherNodes, otherVectors) {
  const otherExact = new Set(otherNodes);
  const missing = [];
  fromNodes.forEach((node, nodeIdx) => {
    if (otherExact.has(node)) return;                    // exact match — present
    let bestSimilarity = -1;
    let nearest = null;
    for (let otherIdx = 0; otherIdx < otherVectors.length; otherIdx++) {
      const similarity = cosine(fromVectors[nodeIdx], otherVectors[otherIdx]);
      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        nearest = otherNodes[otherIdx];
      }
    }
    if (bestSimilarity < SIMILARITY_THRESHOLD) {
      missing.push({ node, nearest, similarity: bestSimilarity });
    }
  });
  return missing;
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

function renderTable(collectionA, missingFromB, collectionB, missingFromA) {
  const escape = (text) => String(text).replace(/\|/g, '\\|').replace(/\s+/g, ' ').trim();
  const timestamp = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
  const row = (presentIn, missingFrom, entry) =>
    `| ${presentIn} | ${missingFrom} | ${escape(entry.node)} `
    + `| ${escape(entry.nearest ?? '—')} | ${entry.similarity < 0 ? '—' : entry.similarity.toFixed(3)} |`;

  const lines = [
    '='.repeat(100),
    `KG node comparison — ${timestamp}`,
    `  A: #${collectionA.id} "${collectionA.name}"`,
    `  B: #${collectionB.id} "${collectionB.name}"`,
    `  present = exact match OR cosine >= ${SIMILARITY_THRESHOLD}  (model ${MODEL})`,
    `  missing from B: ${missingFromB.length}    missing from A: ${missingFromA.length}`,
    '='.repeat(100),
    '',
    '| Present in | Missing from | Node | Nearest node in other KG | Best cosine |',
    '|------------|--------------|------|--------------------------|-------------|',
  ];
  for (const entry of missingFromB) lines.push(row('A', 'B', entry));
  for (const entry of missingFromA) lines.push(row('B', 'A', entry));
  lines.push('');
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const demoUser = await prisma.user.findUnique({ where: { email: DEMO_EMAIL } });
  if (!demoUser) throw new Error(`demo user ${DEMO_EMAIL} not found — run "npm run db:seed" first`);

  const collections = await prisma.collection.findMany({
    where: { userId: demoUser.id },
    orderBy: { id: 'asc' },
  });

  const [selectorA, selectorB] = process.argv.slice(2);
  if (!selectorA || !selectorB) {
    console.log("[test_KG_pass] demo user's collections (pass two as arguments):");
    for (const collection of collections) {
      const nodeCount = collection.knowledgeGraph?.entities?.length ?? 0;
      console.log(`  #${collection.id}  ${collection.name}  (${nodeCount} nodes)`);
    }
    console.log('\nUsage: node tests/test_KG_pass.js <collectionA> <collectionB>');
    return;
  }

  const collectionA = resolveCollection(collections, selectorA);
  const collectionB = resolveCollection(collections, selectorB);
  const nodesA = nodesOf(collectionA);
  const nodesB = nodesOf(collectionB);
  console.log(`[test_KG_pass] A #${collectionA.id} "${collectionA.name}": ${nodesA.length} nodes`);
  console.log(`[test_KG_pass] B #${collectionB.id} "${collectionB.name}": ${nodesB.length} nodes`);

  const vectorsA = await embed(nodesA);
  const vectorsB = await embed(nodesB);

  const missingFromB = missingNodes(nodesA, vectorsA, nodesB, vectorsB);
  const missingFromA = missingNodes(nodesB, vectorsB, nodesA, vectorsA);
  console.log(`[test_KG_pass] ${missingFromB.length} node(s) in A missing from B; `
            + `${missingFromA.length} node(s) in B missing from A`);

  await fs.writeFile(LOG_PATH, renderTable(collectionA, missingFromB, collectionB, missingFromA), 'utf-8');
  console.log(`[test_KG_pass] wrote ${LOG_PATH}`);
}

main()
  .catch((err) => { console.error('[test_KG_pass]', err.message); process.exitCode = 1; })
  .finally(() => prisma.$disconnect());
