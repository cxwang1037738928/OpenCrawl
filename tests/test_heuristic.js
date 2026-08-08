/**
 * test_heuristic.js — pipeline stage 5: BM25 + citation PageRank top-k selection
 *
 * Spawns heuristic.py to rank documents and build the citation graph.
 * Outputs go to tests/test-output/.
 *
 * Run:  node tests/test_heuristic.js [--k 5]
 *
 * Prerequisite: tests/test-output/doclings.json, embeddings.json, categories.json
 *   (produced by test_extract.js, test_embed.js, test_generate_categories.js)
 *
 * Outputs:
 *   tests/test-output/heuristic_output.json
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');

process.env.DATA_DIR = TEST_DATA;

await fs.mkdir(TEST_DATA, { recursive: true });

const HEURISTIC_PY = path.join(ROOT, 'backend', 'extraction', 'heuristic.py');
const CITATIONS_PY = path.join(ROOT, 'backend', 'extraction', 'sapphire', 'citations.py');
const PYTHON       = process.env.PYTHON || 'python';

const argv    = process.argv;
const kArgIdx = argv.indexOf('--k');
const k       = kArgIdx !== -1 ? argv[kArgIdx + 1] : '5';

/** Spawn one pipeline script and resolve on a clean exit. */
function runStage(scriptPath, label, args = []) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON, [scriptPath, ...args], { stdio: 'inherit', cwd: ROOT });
    proc.on('close', (exitCode) => exitCode !== 0
      ? reject(new Error(`${label} exited with code ${exitCode}`))
      : resolve());
    proc.on('error', (err) => reject(new Error(`Failed to spawn ${label}: ${err.message}`)));
  });
}

const start = Date.now();
// Citations first: the ranker reads citations.json to blend PageRank, and skips
// that term entirely when the file is absent.
console.log(`[test_heuristic] Spawning citations.py (${PYTHON}; GROBID parsedReferences, no LLM) ...\n`);
await runStage(CITATIONS_PY, 'citations.py');

console.log(`\n[test_heuristic] Spawning heuristic.py --k ${k} ...\n`);
await runStage(HEURISTIC_PY, 'heuristic.py', ['--k', k]);

const heuristic = JSON.parse(await fs.readFile(path.join(TEST_DATA, 'heuristic_output.json'), 'utf-8'));
// Citation edges live in their own stage output now, not in the ranking.
const citations = JSON.parse(
  await fs.readFile(path.join(TEST_DATA, 'citations.json'), 'utf-8')).edges;

const alpha = parseFloat(process.env.HEURISTIC_ALPHA || '0.5');
console.log(`\n[test_heuristic] Top-k breakdown `
          + `(final = ${alpha}·bm25 + ${(1 - alpha).toFixed(2)}·pagerank):`);
for (const rankedDoc of heuristic.topK) {
  console.log(
    `  ${rankedDoc.finalScore.toFixed(4)}  ${rankedDoc.filename}` +
    `  (repr=${rankedDoc.bm25Representativeness} novelty=${rankedDoc.bm25Novelty} pr=${rankedDoc.pagerankScore})`
  );
}
console.log(`  ${citations.length} citation edge(s) across the corpus`);
console.log(`  ${Object.keys(heuristic.chunkScores || {}).length} chunk(s) scored`);
if (citations.length === 0) {
  console.warn('  WARNING: zero citation edges — PageRank is uniform, so ranking is');
  console.warn('           effectively BM25-only. Usual causes: missing metadata.title/');
  console.warn('           authors (see test_extract.js coverage summary), or empty');
  console.warn('           parsedReferences arrays (GROBID down during extraction).');
}

const elapsed   = ((Date.now() - start) / 1000).toFixed(2);
const timestamp = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_log.txt'),
  `[${timestamp}] test_heuristic           : ${elapsed}s\n`,
  'utf-8',
);
console.log(`\nDone in ${elapsed}s. Run test_kg_graph.js next.`);
