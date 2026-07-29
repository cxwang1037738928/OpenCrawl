/**
 * test_kg_graph.js — pipeline stage 6: kg-gen knowledge graph
 *
 * Spawns kg_graph.py to build the entity/relation graph over the embed
 * stage's chunks (one kg-gen call per chunk). Outputs go to tests/test-output/.
 *
 * Run:  node tests/test_kg_graph.js
 *
 * Prerequisite: tests/test-output/embeddings.json (produced by test_embed.js);
 * tests/test-output/heuristic_output.json (test_heuristic.js) narrows it to
 * the top-ranked docs. Also needs Ollama running with KG_MODEL pulled.
 *
 * Outputs:
 *   tests/test-output/graph.json
 *   tests/test-output/kg_view.html
 *
 * RUNTIME: hours, not seconds — this is the slowest stage in the suite by a
 * wide margin. Cost is (packed batches) × 2 sequential model calls each
 * (kg-gen extracts entities, then relations), so it scales with the fixture
 * corpus, not with anything this file does. Two knobs shorten it: lower
 * KG_FULL_TEXT_FRACTION, so more of the corpus is graphed from its abstract
 * and conclusion instead of full text; or point KG_MODEL at a longer-context
 * model listed in documents/model_metadata.json, which packs more text per
 * call and so makes fewer of them. Because it is this slow there is no timeout
 * here on purpose: a run that looks hung is usually just working, so check
 * kg_graph.py's per-call progress lines before killing it.
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');

// Must be set BEFORE the spawn below: that call passes no `env`, so the child
// inherits this process's env, and kg_graph.py resolves DATA_DIR once at import
// time. Without this the stage would read and overwrite the real data/ dir.
process.env.DATA_DIR = TEST_DATA;

await fs.mkdir(TEST_DATA, { recursive: true });

const KG_GRAPH_PY = path.join(ROOT, 'backend', 'extraction', 'kg_graph.py');
const PYTHON      = process.env.PYTHON || 'python';

const start = Date.now();
console.log(`[test_kg_graph] Spawning kg_graph.py (${PYTHON}; model ${process.env.KG_MODEL}) ...\n`);

await new Promise((resolve, reject) => {
  // stdio 'inherit' rather than piping: kg_graph.py prints a line per chunk, and
  // over a multi-hour run that live progress is the only signal the stage is
  // advancing. Piping would buffer it until exit. The trade-off is that stderr
  // isn't captured, so failures surface as an exit code plus whatever already
  // scrolled past — unlike routes/pipeline.js, which pipes to keep the tail for
  // its HTTP error response.
  const proc = spawn(PYTHON, [KG_GRAPH_PY], { stdio: 'inherit', cwd: ROOT });
  // 'close' fires after the child's stdio has flushed; 'exit' can beat the last
  // progress lines to the console. Both handlers are needed — 'error' covers the
  // spawn itself failing (bad PYTHON path), which never produces a 'close'.
  proc.on('close', (exitCode) => exitCode !== 0
    ? reject(new Error(`kg_graph.py exited with code ${exitCode}`))
    : resolve());
  proc.on('error', (err) => reject(new Error(`Failed to spawn kg_graph.py: ${err.message}`)));
});

// Both reads double as the assertion: kg_graph.py exiting 0 only means the run
// finished, not that it wrote its outputs, so a missing or malformed file must
// fail the test here rather than pass silently. kg_view.html is only stat'd —
// its contents are kg-gen's, so size alone is the useful signal.
const graph = JSON.parse(await fs.readFile(path.join(TEST_DATA, 'graph.json'), 'utf-8'));
const viewHtml = await fs.stat(path.join(TEST_DATA, 'kg_view.html'));

console.log('\n[test_kg_graph] Graph summary:');
console.log(`  ${graph.entities.length} entities, ${graph.relations.length} relations, ` +
            `${graph.edges.length} relation types`);
console.log(`  over ${graph.sourceDocIds.length} document(s) ` +
            `(${graph.fullTextDocIds?.length ?? graph.sourceDocIds.length} full text, ` +
            `${graph.summaryDocIds?.length ?? 0} title+abstract+conclusion), ` +
            `${graph.chunksProcessed} chunk(s) in ${graph.calls} call(s), ` +
            `${graph.callsFailed} call(s) failed`);
console.log(`  model ${graph.model} — ${graph.callMaxChars} chars/call ` +
            `at a ${graph.contextTokens}-token window`);
console.log(`  tokens: ${graph.totalTokens ?? 0} ` +
            `(${graph.promptTokens ?? 0} prompt + ${graph.outputTokens ?? 0} output) ` +
            `over ${graph.modelRequests ?? 0} model request(s)`);
console.log(`  kg_view.html: ${viewHtml.size} bytes`);
// Sample triples: relations are sorted in graph.json, so these five are always
// the same ones for a given graph — an eyeball check that entities read like
// concepts ("self-attention") and not citation debris ("J. Smith", "In ACL").
// The latter means bibliography chunks are reaching kg-gen and the heading
// filter (PIPELINE_REF_HEADINGS) missed this corpus's reference heading.
for (const [subject, predicate, object] of graph.relations.slice(0, 5)) {
  console.log(`    ${subject} —[${predicate}]→ ${object}`);
}
// Warn rather than throw: an empty graph is a real result the run should still
// report (and callsFailed above tells you which of these it was), not a crash
// after a run this long. Non-zero callsFailed with non-zero relations is
// normal — kg_graph.py skips calls the model never returns valid triples for,
// though each one now costs every chunk that was packed into it.
if (graph.relations.length === 0) {
  console.warn('  WARNING: zero relations — the model returned no triples. Usual causes:');
  console.warn('           Ollama down, KG_MODEL not pulled, or every chunk failed validation.');
}

// Appended (not overwritten) so tests/test_log.txt keeps a history: this
// stage's duration is the one worth trending, since it moves with corpus size,
// KG_MODEL and how much of the model Ollama fits on the GPU.
const elapsed   = ((Date.now() - start) / 1000).toFixed(2);
const timestamp = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_log.txt'),
  `[${timestamp}] test_kg_graph            : ${elapsed}s\n`,
  'utf-8',
);

// Token usage, appended to its own history file. kg_graph.py records it from
// dspy's usage tracker into graph.json, so it's read here, not re-measured. A
// re-run on unchanged input legitimately logs ~0 tokens: dspy serves those
// calls from cache, so no model work (and no token cost) actually happened.
const promptTokens  = graph.promptTokens  ?? 0;
const outputTokens  = graph.outputTokens  ?? 0;
const totalTokens   = graph.totalTokens   ?? (promptTokens + outputTokens);
const modelRequests = graph.modelRequests ?? 0;
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_token_usage.txt'),
  `[${timestamp}] test_kg_graph : ${totalTokens} tokens ` +
  `(${promptTokens} prompt + ${outputTokens} output) over ${modelRequests} request(s), ` +
  `${graph.entities.length} entities / ${graph.relations.length} relations\n`,
  'utf-8',
);
console.log(`\nDone in ${elapsed}s. All outputs in tests/test-output/.`);
