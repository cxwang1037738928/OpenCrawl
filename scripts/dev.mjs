/**
 * dev.mjs — the whole stack in one command: `npm run dev:all`
 *
 *   Ollama   — SKIPPED when Azure AI Foundry is configured, since chat answers
 *              come from the hosted deployment. Otherwise started only if
 *              nothing is answering on OLLAMA_URL, then REASONING_MODEL is
 *              preloaded so the first chat isn't paying a cold model load on
 *              top of generation
 *   Backend  — node --watch-path=backend backend/server.js
 *              (:3000, DATA_DIR=tests/test-output)
 *   Frontend — vite dev server                  (:5173, proxies /api + /models)
 *
 * Node, not bash: on Windows `npm run` hands a shell script to whatever `bash`
 * resolves to, which can be WSL's bash — a different filesystem (/mnt/c, not
 * /c) where the Windows Ollama install is invisible. Node has no such ambiguity.
 *
 * Ctrl-C stops the backend and frontend. Ollama is a daemon and is left running:
 * killing it would make every run pay its (sometimes >60s) cold start again.
 *
 *   npm run dev:all
 *   DATA_DIR=data npm run dev:all      # override the corpus directory
 */

import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

// Read DATA_DIR BEFORE dotenv, so we can tell "the user set it on the command
// line" from ".env happens to define it". dotenv would overwrite neither, but it
// does populate process.env, and .env's DATA_DIR=data is not what dev:all wants:
// this script is the dev:test workflow, whose corpus lives in tests/test-output.
// Hence the dynamic import — a static one would run before this line.
const DATA_DIR_FROM_CLI = process.env.DATA_DIR;
await import('dotenv/config');

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const OLLAMA_URL      = process.env.OLLAMA_URL || 'http://localhost:11434';
const REASONING_MODEL = (process.env.REASONING_MODEL || '').trim();
const PORT            = process.env.PORT || '3000';
const DATA_DIR        = DATA_DIR_FROM_CLI || 'tests/test-output';

// Chat answers come from Azure AI Foundry when these three are set (see
// retriever.js), so there is nothing to start or preload locally. Ollama is
// still what serves a LOCAL KG_MODEL and extract.py's author fallback, so it is
// skipped rather than removed — a clone without Azure credentials still gets the
// old behaviour.
const AZURE_CHAT = Boolean(
  (process.env.MICROSOFT_AZURE_PROJECT_ENDPOINT || '').trim()
  && (process.env.MICROSOFT_AZURE_API_KEY || '').trim()
  && (process.env.AZURE_DEPLOYMENT_NAME || '').trim());

const log = (message) => console.log(`\x1b[35m[dev]\x1b[0m ${message}`);

/**
 * Read every file under dir once, BEFORE the backend watcher exists.
 *
 * On NTFS a read updates the file's last-access time whenever the stored one
 * is more than an hour old — i.e. on the first run after boot — and libuv's
 * file watcher reports last-access updates as changes. Without this, node's
 * watcher restarts on its own module loads: a restart storm on the first run
 * of the day that kills whatever request is in flight (the ECONNRESET on the
 * first /api/chat). Warming the atimes here makes that burst happen while
 * nothing is watching; the Ollama checks below give the flurry time to settle.
 */
async function warmAtimes(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true, recursive: true })
    .catch(() => []);
  await Promise.all(entries
    .filter((entry) => entry.isFile())
    .map((entry) => fs.readFile(path.join(entry.parentPath, entry.name)).catch(() => {})));
}

/** True when Ollama answers. Generous timeout: /api/tags is slow while a model loads. */
async function ollamaUp() {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`, { signal: AbortSignal.timeout(5000) });
    return response.ok;
  } catch {
    return false;
  }
}

const children = [];

/** Spawn a long-running child with its output prefixed and colored. */
function run(name, color, command, args, env = {}) {
  const child = spawn(command, args, {
    cwd: ROOT,
    env: { ...process.env, ...env },
    // shell:true so `npm` resolves to npm.cmd on Windows without hard-coding it.
    shell: process.platform === 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const prefix = `\x1b[${color}m[${name}]\x1b[0m `;
  const forward = (stream, sink) => stream.on('data', (bytes) => {
    for (const line of bytes.toString().split('\n')) {
      if (line.trim()) sink.write(prefix + line + '\n');
    }
  });
  forward(child.stdout, process.stdout);
  forward(child.stderr, process.stderr);
  child.on('exit', (code) => log(`${name} exited (code ${code})`));
  children.push(child);
  return child;
}

let shuttingDown = false;
function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log('');
  log('stopping backend + frontend (ollama left running)…');
  for (const child of children) {
    if (child.exitCode !== null || !child.pid) continue;
    // Windows: child.kill() leaves grandchildren alive (npm spawns node, which
    // then holds the port). /T kills the whole tree.
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
    } else {
      child.kill('SIGTERM');
    }
  }
  setTimeout(() => process.exit(0), 500);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// ── Ollama ───────────────────────────────────────────────────────────────────

await warmAtimes(path.join(ROOT, 'backend'));

if (AZURE_CHAT) {
  log(`chat → Azure AI Foundry (${process.env.AZURE_DEPLOYMENT_NAME}) — not starting ollama`);
  // Not fatal, only worth knowing: these degrade quietly rather than failing.
  // extract.py's LLM author fallback swallows the connection error, and the
  // Models tab simply reports ollamaUp:false.
  if (!(await ollamaUp())) {
    log('note: ollama is not running. A LOCAL KG_MODEL and extract.py\'s author');
    log('      fallback need it — start it with "ollama serve" before a pipeline run.');
  }
} else if (await ollamaUp()) {
  log(`ollama already up at ${OLLAMA_URL}`);
} else {
  log('starting ollama serve…');
  // detached + unref: ollama outlives this script on purpose, so the next run
  // finds it warm. shell:true resolves it from the same PATH your terminal uses.
  const ollama = spawn('ollama', ['serve'], {
    detached: true, stdio: 'ignore', shell: process.platform === 'win32',
  });
  ollama.on('error', () => {
    log('could not launch "ollama" — it is not on this shell\'s PATH.');
    log('Start it yourself (ollama serve) and re-run, or install it.');
    process.exit(1);
  });
  ollama.unref();

  // A cold start on Windows can take well over a minute (runtime init, GPU probe).
  const deadline = Date.now() + 120_000;
  while (!(await ollamaUp())) {
    if (Date.now() > deadline) {
      log(`ollama still not answering at ${OLLAMA_URL} after 120s — giving up.`);
      process.exit(1);
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  log(`ollama up at ${OLLAMA_URL}`);
}

if (AZURE_CHAT) {
  // Nothing to preload: the deployment is always warm, and REASONING_MODEL is
  // not consulted on the Azure path.
} else if (!REASONING_MODEL) {
  log('WARNING: REASONING_MODEL is unset — chat will 503 until you pick one in the Models tab.');
} else {
  log(`preloading ${REASONING_MODEL}…`);
  try {
    // Empty prompt + keep_alive is Ollama's "load into memory and stay there"
    // call. `ollama run` would open an interactive REPL this script can't drive.
    const response = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: REASONING_MODEL, keep_alive: '1h' }),
      signal: AbortSignal.timeout(300_000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
    log(`${REASONING_MODEL} resident (1h keep-alive)`);
  } catch (err) {
    log(`WARNING: could not preload ${REASONING_MODEL} — ${err.message}`);
    log(`Is it pulled?  ollama pull ${REASONING_MODEL}`);
  }
}

// ── Backend + frontend ───────────────────────────────────────────────────────

// Watch the JS server dirs individually, NOT all of backend/ — and pointedly
// NOT backend/extraction, where the pipeline's Python lives. A pipeline run
// spawns python, which reads those .py files; on NTFS a read bumps the file's
// atime, which libuv reports as a change, restarting the server mid-run and
// killing the (multi-minute) /pipeline/run request with ECONNRESET. Excluding
// extraction/ keeps the watcher blind to that churn (and to any __pycache__).
// Trade-off: editing a JS module under backend/extraction/ (embed.js,
// generate_categories.js, chunker.js, sapphire/*.js) needs a manual restart.
const WATCH_PATHS = [
  'backend/server.js', 'backend/db.js', 'backend/prompts.js',
  'backend/middleware', 'backend/parser', 'backend/pipeline',
  'backend/retriever', 'backend/routes',
];
// NO_WATCH=1 runs the backend without --watch at all. The scoped watch above
// stops the pipeline's OWN files from triggering restarts, but the server's
// watched .js still go atime-stale after ~1h, and a long run's event-loop
// starvation lets libuv fire those stale-atime reads as changes — the same
// "cold start" restart storm warmAtimes fights, now hitting a run instead of
// boot. There's no way to tell node --watch to ignore atime, so for a long
// pipeline run, launch with NO_WATCH=1 (backend hot-reload off) to guarantee
// the run isn't interrupted; edit-and-restart the backend by hand meanwhile.
const NO_WATCH = process.env.NO_WATCH === '1' || process.env.NO_WATCH === 'true';
const apiArgs = NO_WATCH
  ? ['backend/server.js']
  : [...WATCH_PATHS.map((p) => `--watch-path=${p}`), 'backend/server.js'];

log(`backend  → http://localhost:${PORT}   (DATA_DIR=${DATA_DIR}${NO_WATCH ? ', no-watch' : ''})`);
run('api', '36', process.execPath, apiArgs, { DATA_DIR });

log('frontend → http://localhost:5173');
run('web', '32', 'npm', ['--prefix', 'frontend', 'run', 'dev']);

log('all up. Ctrl-C to stop.');
