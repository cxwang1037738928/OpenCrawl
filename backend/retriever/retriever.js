/**
 * retriever.js — RAG retrieval + answer synthesis for /api/chats/:chatId/chat
 *
 * The query embedding arrives from the BROWSER (the frontend embeds the
 * user's question with the same MiniLM model the corpus was embedded with),
 * so retrieval here is pure math over the collection's Chunk rows — no model loads.
 *
 * Scoring: (cosine × category boost) + lexical BM25 blend.
 *   - category boost: if a query token matches a keyword of a category
 *     (categories.json), every chunk of every doc in that category gets a 5%
 *     boost per matching keyword, multiplicative: 1.05^matches.
 *   - BM25 over chunk text, normalized to the query's best-scoring chunk and
 *     weighted by LEXICAL_WEIGHT. Pure-semantic ranking fails meta-queries
 *     ("which document mentions quantum chemicals?") — the question framing
 *     dominates the embedding while the literal term pins the exact chunk.
 *
 * Answering: Ollama chat with REASONING_MODEL from .env; retrieved chunks
 * are injected as context in the system prompt. The response is streamed so
 * the timeout can key on inactivity instead of total time — a small model on
 * CPU needs ~110s for a long answer, which is slow but not hung.
 *
 * Exports:
 *   retrieve(collection, queryEmbedding, queryText, {topK}) — ranked chunks
 *   answer(messages, chunks)                          — Ollama completion string
 */

import 'dotenv/config';
import { tokenise } from '../extraction/regex_utils.js';
import { getPrompt } from '../prompts.js';
import { embedTexts } from './embedder.js';
import { buildGraphIndex, graphFacts } from './graph_retriever.js';
import { prisma } from '../db.js';

const OLLAMA_URL     = process.env.OLLAMA_URL || 'http://localhost:11434';
// per matched category keyword, multiplicative
const KEYWORD_BOOST  = parseFloat(process.env.RETRIEVER_KEYWORD_BOOST || '1.05');
// Multiplier on chunks of a document the query names outright. A corpus of
// near-identical studies gives the vector channel nothing to separate them
// ("1.4 Vitro_geno" and "2.7 Vitro_geno" are the same assay), and tokenise()
// drops digits so BM25 cannot either — measured on a 20-report corpus, only
// 10% of retrieved chunks came from the study the question named. A boost
// rather than a filter: a name recognised in the question is a guess, so a
// wrong one should cost ranking, not hide the corpus. Set to 1 to disable.
//
// Measured on that corpus (220 questions, top-10): 1.5 → 57% of retrieved
// chunks from the named study, 2.0 → 75%, 3.0 → 87% but the retrieved set
// starts shrinking, because a boosted top score drags SCORE_FLOOR up with it
// and prunes the rest of the corpus away. 2.0 is the last value that still
// returns a full top-K.
const DOC_BOOST      = parseFloat(process.env.RETRIEVER_DOC_BOOST || '2.0');
const LEXICAL_WEIGHT = parseFloat(process.env.RETRIEVER_LEXICAL_WEIGHT || '0.3');
const BM25_K1        = parseFloat(process.env.RETRIEVER_BM25_K1 || '1.5');   // term-frequency saturation
const BM25_B         = parseFloat(process.env.RETRIEVER_BM25_B  || '0.75');  // length normalization strength
const DEFAULT_TOP_K  = parseInt(process.env.RETRIEVER_TOP_K || '8', 10);
// Relevance floor: keep only chunks scoring within this fraction of the best
// one. Without it every query returns exactly topK chunks, so a question with
// one real hit also ships 7 near-misses the model has to explain away.
const SCORE_FLOOR = parseFloat(process.env.RETRIEVER_SCORE_FLOOR || '0.5');
// Inactivity timeout, NOT a wall-clock deadline: we stream from Ollama and
// reset this on every token. A long answer is more work, not a hang — a fixed
// deadline killed perfectly healthy 110s generations.
const OLLAMA_IDLE_TIMEOUT = parseInt(process.env.OLLAMA_IDLE_TIMEOUT_MS || '60000', 10);
// Penalty on tokens the reply has already used. Measured on Phi-4-mini-instruct
// at temperature 0.2 with no penalty set: an answer fell into a token loop
// mid-sentence ("and and and 2 and…") and ran on for ~1,500 characters of
// nothing. A small model at low temperature has little to pull it out of a loop,
// and nothing in the request discouraged reuse. Deliberately NOT paired with a
// max_tokens cap: a cap truncates the collapse but also cuts off a legitimately
// long synthesis mid-sentence, and the penalty addresses the cause instead.
const FREQUENCY_PENALTY = parseFloat(process.env.RETRIEVER_FREQUENCY_PENALTY || '0.3');

// ---------------------------------------------------------------------------
// Corpus cache (per collection, invalidated when corpusUpdatedAt changes)
// ---------------------------------------------------------------------------

const _cacheByCollection = new Map();

async function loadCorpus(collection) {
  const rows = await prisma.chunk.findMany({
    where: { collectionId: collection.id },
    orderBy: [{ docId: 'asc' }, { chunkIndex: 'asc' }],
  });
  // The graph rides this cache rather than being loaded per query: it is on the
  // same Collection row, and ingestGraph bumps corpusUpdatedAt on every write,
  // which is already this cache's invalidation key. No second staleness path.
  const { knowledgeGraph, documents } = await prisma.collection.findUniqueOrThrow({
    where:  { id: collection.id },
    select: {
      knowledgeGraph: true,
      // Names ride this same read so resolving "regarding 1.4 Vitro_geno" to a
      // docId costs no extra query and is invalidated with the rest of the cache.
      documents: { select: { docId: true, filename: true, title: true } },
    },
  });
  if (rows.length === 0) {
    const err = new Error('This collection has no indexed chunks — upload PDFs and run the pipeline first');
    err.status = 503;
    throw err;
  }

  const chunks = rows.map((row) => ({
    id:        row.chunkId,
    docId:     row.docId,
    filename:  row.filename,
    pages:     row.pages,
    prefixLen: row.prefixLen,
    heading:   row.heading,
    text:      row.text,
    embedding: row.embedding,
  }));

  return {
    dims: collection.embeddingsMeta?.dimensions,
    chunks,
    categories: collection.categories?.categories || [],   // absent → no keyword boost
    docIndex: buildDocIndex(documents),
    lexical: buildLexicalIndex(chunks),
    // null when the collection has no graph yet — graph facts are then skipped
    // and the chat behaves exactly as it did before.
    graph: buildGraphIndex(knowledgeGraph),
  };
}

/** Per-chunk term frequencies + document frequencies for BM25. */
function buildLexicalIndex(chunks) {
  const docFreq = new Map();
  const chunkTermFreqs = chunks.map((chunk) => {
    const termFreq = new Map();
    for (const token of tokenise(chunk.text || '')) termFreq.set(token, (termFreq.get(token) || 0) + 1);
    for (const token of termFreq.keys()) docFreq.set(token, (docFreq.get(token) || 0) + 1);
    return termFreq;
  });
  const avgChunkLength = chunkTermFreqs.reduce(
    (total, termFreq) => total + termFreq.size, 0) / (chunkTermFreqs.length || 1);
  return { docFreq, chunkTermFreqs, avgChunkLength, chunkCount: chunks.length };
}

/**
 * BM25 score of every chunk against the query tokens, with two twists:
 *   - idf is SQUARED so rare terms dominate. Meta-queries ("which document
 *     mentions quantum chemicals") carry several medium-rarity words
 *     (document, part, mentions) whose classic-BM25 sum outweighs the one
 *     df=1 term the user actually cares about.
 *   - unmatched plural query tokens fall back to their singular ("chemicals"
 *     → "chemical") since the corpus index is unstemmed.
 */
function bm25Scores(queryTokens, { docFreq, chunkTermFreqs, avgChunkLength, chunkCount }) {
  const scores = new Float64Array(chunkTermFreqs.length);
  const matchedTerms = new Set();
  for (const token of queryTokens) {
    if (docFreq.has(token)) matchedTerms.add(token);
    else if (token.endsWith('s') && docFreq.has(token.slice(0, -1))) {
      matchedTerms.add(token.slice(0, -1));
    }
  }
  for (const term of matchedTerms) {
    const termDocFreq = docFreq.get(term);
    const idf = Math.log(1 + (chunkCount - termDocFreq + 0.5) / (termDocFreq + 0.5)) ** 2;
    for (let chunkIdx = 0; chunkIdx < chunkTermFreqs.length; chunkIdx++) {
      const termFreq = chunkTermFreqs[chunkIdx].get(term);
      if (!termFreq) continue;
      scores[chunkIdx] += idf * (termFreq * (BM25_K1 + 1)) /
        (termFreq + BM25_K1 * (1 - BM25_B + BM25_B
          * (chunkTermFreqs[chunkIdx].size / avgChunkLength)));
    }
  }
  return scores;
}

async function getCorpus(collection) {
  const stamp = collection.corpusUpdatedAt?.getTime() ?? 0;
  const cached = _cacheByCollection.get(collection.id);
  if (cached?.stamp === stamp) return cached.data;
  const data = await loadCorpus(collection);
  _cacheByCollection.set(collection.id, { stamp, data });
  return data;
}

// ---------------------------------------------------------------------------
// Retrieval
// ---------------------------------------------------------------------------

// Embeddings are L2-normalized, so dot = cosine.
function dot(vecA, vecB) {
  let sum = 0;
  for (let componentIdx = 0; componentIdx < vecA.length; componentIdx++) {
    sum += vecA[componentIdx] * vecB[componentIdx];
  }
  return sum;
}

// ---------------------------------------------------------------------------
// Named-document resolution
// ---------------------------------------------------------------------------

// Filenames are never part of chunk text and tokenise() discards digits, so on
// their own neither channel can tell "1.1 Dermal_sens.pdf" from "2.10
// Dermal_sens.pdf" — both reduce to ["dermal","sens","pdf"]. Everything below
// works on the raw query instead, keeping the numbers that do the separating.

const wordsIn   = (text) => new Set(String(text || '').toLowerCase().match(/[a-z]+/g) || []);
const numbersIn = (text) => new Set(String(text || '').match(/\d+(?:\.\d+)+|\d+/g) || []);

/** Filename with separators flattened: "1.10_Oral_Repd.pdf" → "1.10 oral repd". */
const normalizeDocName = (text) => String(text || '')
  .toLowerCase().replace(/\.pdf$/, '').replace(/[_\s]+/g, ' ').trim();

/**
 * One entry per document: the flattened filename, its leading number, the words
 * of its name, and its title only when the title is long and wordy enough to be
 * worth matching. Extraction frequently yields "ORIGINAL", "Gentlemen:" or an
 * OCR smear, and those must never be match candidates; a verbatim hit on a long
 * title, by contrast, is safe by construction.
 */
function buildDocIndex(documents = []) {
  return documents.map((doc) => {
    const name = normalizeDocName(doc.filename);
    const [first, ...rest] = name.split(' ');
    const numbered = /^\d+(?:\.\d+)*$/.test(first);
    const title = String(doc.title || '').toLowerCase().replace(/\s+/g, ' ').trim();
    const letters = (title.match(/[a-z ]/g) || []).length;
    return {
      docId:  doc.docId,
      name:   name.length >= 5 ? name : null,     // a 2-char name would match anything
      number: numbered ? first : null,
      words:  (numbered ? rest : name.split(' ')).filter((word) => word.length > 2),
      title:  title.length >= 25 && letters / title.length >= 0.6 ? title : null,
    };
  });
}

/**
 * The documents a query names outright, as a Set of docIds. Empty unless the
 * evidence is strong: the whole filename, or its number together with one of
 * its own words, or a long title verbatim. A bare "dermal" must never match —
 * most of a toxicology corpus shares it. A query naming three or more documents
 * resolves to none rather than collapsing onto one of them.
 */
export function resolveDocIds(queryText, docIndex = []) {
  if (!docIndex.length || !queryText) return new Set();
  const query   = normalizeDocName(queryText);
  const words   = wordsIn(queryText);
  const numbers = numbersIn(queryText);

  const matched = docIndex.filter((doc) =>
    (doc.name && query.includes(doc.name))
    || (doc.number && numbers.has(doc.number) && doc.words.some((word) => words.has(word)))
    || (doc.title && query.includes(doc.title)));

  return new Set(matched.length <= 2 ? matched.map((doc) => doc.docId) : []);
}

/**
 * Per-doc multiplicative boost from category keyword matches:
 * 1.05^(query tokens ∩ category keywords) for every doc in that category.
 */
function keywordBoosts(queryText, categories) {
  const queryTokens = new Set(tokenise(queryText || ''));
  const boostByDocId = new Map();
  for (const category of categories) {
    const matchCount = (category.keywords || [])
      .filter((keyword) => queryTokens.has(keyword)).length;
    if (matchCount === 0) continue;
    const boostFactor = KEYWORD_BOOST ** matchCount;
    for (const member of category.members || []) boostByDocId.set(member.docId, boostFactor);
  }
  return boostByDocId;
}

/**
 * Rank one collection's chunks against a query.
 * @param {object}   collection     — Collection row (id, corpusUpdatedAt, categories, embeddingsMeta)
 * @param {number[]} queryEmbedding — L2-normalized, same model/dims as the corpus
 * @param {string}   queryText      — raw question, for keyword matching
 * @param {string[]} [opts.docIds]  — documents the caller asserts the query is about.
 *                                    Present: a hard scope, nothing else is considered.
 *                                    Omitted: names found in queryText are boosted instead.
 * @returns top-K [{docId, filename, heading, text, sim, boost, score}]
 */
export async function retrieve(collection, queryEmbedding, queryText,
  { topK = DEFAULT_TOP_K, docIds } = {}) {
  const corpus = await getCorpus(collection);
  if (corpus.dims && queryEmbedding.length !== corpus.dims) {
    const err = new Error(`query embedding has ${queryEmbedding.length} dims; corpus uses ${corpus.dims}`);
    err.status = 400;
    throw err;
  }

  // Two different claims, deliberately handled differently. An explicit docIds
  // is the caller asserting scope — honour it exactly, because a question like
  // "does this study name a test guideline?" is answered wrongly, not partially,
  // if another study's guidelines are in the context. A name merely recognised
  // in the question text is a guess, so it only boosts.
  const scope    = Array.isArray(docIds) && docIds.length ? new Set(docIds) : null;
  const targeted = scope ? null : resolveDocIds(queryText, corpus.docIndex);
  const inScope  = (chunk) => !scope || scope.has(chunk.docId);

  const boostByDocId = keywordBoosts(queryText, corpus.categories);
  const bm25 = bm25Scores(tokenise(queryText || ''), corpus.lexical);
  // Normalize BM25 to the query's best chunk so the blend weight is stable
  // across queries of different rarity — measured within the scope, since a
  // strong match in an excluded document would otherwise flatten the lexical
  // channel for everything that survives the filter.
  let bm25Max = 0;
  corpus.chunks.forEach((chunk, chunkIdx) => {
    if (inScope(chunk) && bm25[chunkIdx] > bm25Max) bm25Max = bm25[chunkIdx];
  });
  bm25Max = bm25Max || 1;

  const scored = [];
  corpus.chunks.forEach((chunk, chunkIdx) => {
    if (!inScope(chunk)) return;              // out of scope — skip the dot product too
    const sim   = dot(queryEmbedding, chunk.embedding);
    const boost = (boostByDocId.get(chunk.docId) || 1)
      * (targeted?.has(chunk.docId) ? DOC_BOOST : 1);
    const lex   = bm25[chunkIdx] / bm25Max;
    scored.push({ chunk, sim, boost, lex, score: sim * boost + LEXICAL_WEIGHT * lex });
  });

  // Silently answering from the whole corpus would be worse than failing: the
  // caller asked for specific documents and would not know it did not get them.
  if (scope && !scored.length) {
    const err = new Error(`no indexed chunks for the requested documents: ${[...scope].join(', ')}`);
    err.status = 400;
    throw err;
  }

  scored.sort((a, b) => b.score - a.score);

  // Drop the long tail below the floor; the top chunk always survives.
  const cutoff = Math.max(scored[0]?.score * SCORE_FLOOR || 0, 0);
  const kept = scored.slice(0, topK).filter((c, idx) => idx === 0 || c.score >= cutoff);

  return kept.map(({ chunk, sim, boost, lex, score }) => ({
    chunkId:   chunk.id,
    docId:     chunk.docId,
    filename:  chunk.filename,
    heading:   chunk.heading,
    pages:     chunk.pages ?? null,
    text:      chunk.text,
    embedding: chunk.embedding,   // kept for citation grounding; chat.js strips it from sources
    sim:       Math.round(sim * 10000) / 10000,
    boost:     Math.round(boost * 10000) / 10000,
    lex:       Math.round(lex * 10000) / 10000,
    score:     Math.round(score * 10000) / 10000,
  }));
}

/**
 * Knowledge-graph facts for a query, from the graph cached with the corpus.
 * Empty when the collection has no graph or the query mentions no known entity.
 * @returns {Promise<{facts: object[], seeds: string[]}>}
 */
export async function retrieveFacts(collection, queryText, options = {}) {
  const corpus = await getCorpus(collection);
  if (!corpus.graph) return { facts: [], seeds: [] };
  return graphFacts(queryText, corpus.graph, options);
}

// ---------------------------------------------------------------------------
// Answer synthesis — provider chosen by REASONING_MODEL's id
// ---------------------------------------------------------------------------

// Foundry exposes an OpenAI-compatible API at <resource>/openai/v1. The .env
// value is the PROJECT endpoint (.../api/projects/<name>), which is a different
// path on the same host, so only its origin is reused.
const AZURE_PROJECT_ENDPOINT = (process.env.MICROSOFT_AZURE_PROJECT_ENDPOINT || '').trim();
const AZURE_API_KEY = (process.env.MICROSOFT_AZURE_API_KEY || '').trim();
const AZURE_DEPLOYMENT = (process.env.AZURE_DEPLOYMENT_NAME || '').trim();

// Google serves an OpenAI-compatible endpoint, so Gemini reuses this file's
// streaming, abort and idle-timer handling rather than adding a second SDK.
const GOOGLE_BASE_URL = (process.env.GEMINI_BASE_URL
  || 'https://generativelanguage.googleapis.com/v1beta/openai/').trim();
const GEMINI_API_KEY = (process.env.GEMINI_API_KEY || '').trim();
// Gemini's thinking budget, as the OpenAI-compatible layer spells it.
const CHAT_REASONING_EFFORT = (process.env.CHAT_REASONING_EFFORT || 'low').trim();

const azureConfigured = () =>
  Boolean(AZURE_PROJECT_ENDPOINT && AZURE_API_KEY && AZURE_DEPLOYMENT);

/**
 * Which provider serves `model`, decided by the id alone.
 *
 * The id IS the setting: the Models tab writes REASONING_MODEL and nothing
 * else, so a global "Azure wins when configured" rule would let a user pick
 * Gemini and still be answered by the Foundry deployment. Azure credentials
 * staying in .env is therefore no longer enough to route there — the model has
 * to name it.
 */
function chatProvider(model) {
  if (/^gemini[/-]/i.test(model)) return 'google';
  if (AZURE_DEPLOYMENT && model === AZURE_DEPLOYMENT) return 'azure';
  return 'ollama';
}

/** litellm-style "gemini/gemini-3.1-flash-lite" → the bare id Google's API wants. */
const googleModelId = (model) => model.replace(/^gemini\//i, '');

// maxRetries 0 on both hosted clients: the SDK's default retry on 429 waits
// BEFORE the stream starts, so no token ever arrives to reset the inactivity
// timer — the request is aborted at OLLAMA_IDLE_TIMEOUT and the real cause (a
// rate limit, with a clear remedy) is reported as "Request was aborted".
let _azureClient = null;
async function azureClient() {
  if (_azureClient) return _azureClient;
  const { default: OpenAI } = await import('openai');
  const baseURL = `${new URL(AZURE_PROJECT_ENDPOINT).origin}/openai/v1`;
  _azureClient = new OpenAI({ baseURL, apiKey: AZURE_API_KEY, maxRetries: 0 });
  return _azureClient;
}

let _googleClient = null;
async function googleClient() {
  if (_googleClient) return _googleClient;
  const { default: OpenAI } = await import('openai');
  _googleClient = new OpenAI({ baseURL: GOOGLE_BASE_URL, apiKey: GEMINI_API_KEY, maxRetries: 0 });
  return _googleClient;
}

/**
 * Fold the system prompt into the LATEST user message.
 *
 * Foundry deployments can discard role:'system' silently. Measured on
 * Phi-4-mini-instruct: prompt_tokens stayed at 16 whether the system message
 * held 87 characters or 18,027, and the model answered as though it had no
 * context — no citations, no graph facts, questions answered from its own
 * memory. The identical payload in a user message is counted correctly and used
 * correctly at 18k chars. There is no error, so this cannot be detected from a
 * response; folding unconditionally is the only safe default.
 *
 * The LATEST user turn, not the first: in a long conversation the prompt would
 * otherwise drift far from the question it governs. The cost is resending the
 * context every turn, which is the reliable side of that trade.
 *
 * `reminder` is appended AFTER the question. Folding puts the rules ahead of the
 * entire context, which buries them: measured on Phi-4-mini-instruct with 10
 * excerpts and 25 graph facts, the citation rule lands at character 97 and the
 * question at 14,499, and the reply cites nothing at all. The same request with
 * the rule restated after the question cites six times. Only the FORMAT is
 * repeated, not the grounding policy — the policy is long, and the failure is
 * the model forgetting the notation, not forgetting to stay grounded.
 */
function foldSystemIntoLatestUser(system, messages, reminder = '') {
  const folded = messages.map((message) => ({ ...message }));
  const suffix = reminder ? `\n\n${reminder}` : '';
  for (let index = folded.length - 1; index >= 0; index--) {
    if (folded[index].role === 'user') {
      folded[index].content = `${system}\n\n---\n\n${folded[index].content}${suffix}`;
      return folded;
    }
  }
  return [...folded, { role: 'user', content: `${system}${suffix}` }];
}

/**
 * Append the reminder after the latest user message, leaving `system` in its own
 * role. For providers that honour a system message, so the rules are not pushed
 * ahead of the whole context in the first place.
 */
function appendReminder(messages, reminder) {
  const copy = messages.map((message) => ({ ...message }));
  if (!reminder) return copy;
  for (let index = copy.length - 1; index >= 0; index--) {
    if (copy[index].role === 'user') {
      copy[index].content = `${copy[index].content}\n\n${reminder}`;
      return copy;
    }
  }
  return copy;
}

/** The citation format, restated after the question. See foldSystemIntoLatestUser. */
function citationReminder(factCount) {
  const reminder = 'Reminder: cite every claim with its excerpt number in square '
    + 'brackets, exactly like [1] or [2], or [1, 3] for several.';
  // Included for the same reason, though it did NOT recover [G] in testing —
  // that marker fails for its own reasons, not because the rule went unread.
  return factCount
    ? `${reminder} Append [G] to any claim that rests on one of the listed relationships.`
    : reminder;
}

/**
 * Stream a completion from an OpenAI-compatible hosted provider.
 *
 * Both Foundry and Google speak this shape, so the difference between them is
 * data (client, model id, whether the system prompt survives as its own role,
 * whether a thinking budget applies) rather than a second copy of the streaming
 * and error handling.
 *
 * Hosted, so the 2048-token window Ollama silently imposes locally does not
 * apply — the deployment serves the model's real context. The failure mode moves
 * from truncation to rate limiting: a quota rejects an oversized request outright
 * with 429 rather than quietly dropping the front of the prompt, which is the
 * better of the two.
 */
async function completeViaHosted({ label, client, model, fold, reasoningEffort,
                                   frequencyPenalty, system, messages, reminder,
                                   abortController, armIdleTimer }) {
  let reply = '';
  armIdleTimer();
  try {
    const stream = await client.chat.completions.create({
      model,
      // Providers that honour role:'system' keep it there, so the rules are not
      // separated from the question by the whole context. Only the reminder is
      // appended after the question in that case.
      messages: fold
        ? foldSystemIntoLatestUser(system, messages, reminder)
        : [{ role: 'system', content: system }, ...appendReminder(messages, reminder)],
      temperature: 0.2,
      // Omitted rather than zeroed where unsupported: Google's OpenAI layer
      // rejects frequency_penalty outright with a bodyless 400.
      ...(frequencyPenalty ? { frequency_penalty: frequencyPenalty } : {}),
      ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
      stream: true,
    }, { signal: abortController.signal });
    for await (const part of stream) {
      armIdleTimer();                      // progress — restart the idle clock
      reply += part.choices?.[0]?.delta?.content || '';
    }
  } catch (err) {
    const status = err.status;
    const httpError = new Error(
      status === 429
        ? `${label} rate limit hit for ${model} — raise its quota or retrieve fewer chunks`
        : status === 401 || status === 403
          ? `${label} rejected its API key`
          : err.name === 'AbortError'
            ? `${label} sent nothing for ${OLLAMA_IDLE_TIMEOUT / 1000}s`
            : `${label} request failed: ${err.message}`);
    httpError.status = status === 429 ? 429 : 502;
    throw httpError;
  }
  return reply;
}

/**
 * Answer the conversation using the retrieved chunks as grounding context.
 * @param {{role: string, content: string}[]} messages — chat history, last = question
 * @param {Awaited<ReturnType<typeof retrieve>>} chunks
 * @returns {Promise<{reply: string, model: string, quotesByChunk: string[][]}>}
 *          quotesByChunk[i] = verbatim quotes in the reply grounded by chunks[i]
 */
export async function answer(messages, chunks, facts = []) {
  const model = (process.env.REASONING_MODEL || '').trim();
  if (!model) {
    const err = new Error('REASONING_MODEL is not set — pick one in the Models tab');
    err.status = 503;
    throw err;
  }

  const context = chunks
    .map((chunk, chunkIdx) =>
      `[${chunkIdx + 1}] ${chunk.filename}${chunk.heading ? ` — ${chunk.heading}` : ''}\n${chunk.text}`)
    .join('\n\n');

  // Graph facts are a SEPARATE block, unnumbered. Numbering them would invite
  // the model to cite them as [n], and repairCitations would then try to ground
  // a triple against chunk text it never appears in — see the [G] note below.
  const graphContext = facts.length
    ? facts.map((fact) => `- ${fact.subject} ${fact.predicate} ${fact.object}`).join('\n')
    : '';

  // The citation format the prompt demands is load-bearing: the frontend
  // turns [n] into a link into the PDF, so a model that writes [n1] or names
  // the file inline produces an answer with no working citations.
  const system = getPrompt(facts.length ? 'chat_system_graph' : 'chat_system',
                           { context, graphContext });

  // Stream so the timeout can be based on inactivity rather than total time:
  // a slow model writing a long answer is working, not hung.
  const abortController = new AbortController();
  let idleTimer;
  const armIdleTimer = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => abortController.abort(), OLLAMA_IDLE_TIMEOUT);
  };

  const provider = chatProvider(model);

  if (provider === 'google' || provider === 'azure') {
    const google = provider === 'google';
    if (google && !GEMINI_API_KEY) {
      const err = new Error(`GEMINI_API_KEY is not set — ${model} cannot be reached`);
      err.status = 503;
      throw err;
    }
    if (!google && !azureConfigured()) {
      const err = new Error(`${model} needs MICROSOFT_AZURE_PROJECT_ENDPOINT, `
                          + 'MICROSOFT_AZURE_API_KEY and AZURE_DEPLOYMENT_NAME');
      err.status = 503;
      throw err;
    }
    let reply;
    try {
      reply = await completeViaHosted({
        label:  google ? 'Google' : 'Azure',
        client: await (google ? googleClient() : azureClient()),
        model:  google ? googleModelId(model) : AZURE_DEPLOYMENT,
        // Foundry deployments silently discard role:'system' (see
        // foldSystemIntoLatestUser); Gemini honours it, so it keeps the rules
        // out in front of the context rather than buried behind it.
        fold:   !google,
        // Thinking budget, Gemini only — Foundry's deployment rejects the field.
        reasoningEffort: google ? CHAT_REASONING_EFFORT : null,
        // Repetition control, Azure only for the mirror-image reason. It exists
        // for a 3.8B model that fell into a token loop; Gemini does not need it.
        frequencyPenalty: google ? null : FREQUENCY_PENALTY,
        system,
        messages,
        reminder: citationReminder(facts.length),
        abortController,
        armIdleTimer,
      });
    } finally {
      clearTimeout(idleTimer);             // owned here, same as the Ollama path
    }
    const normalized = normalizeGraphMarkers(
      normalizeCitations(reply.trim(), chunks.length), facts.length);
    const { text, quotesByChunk } = await repairCitations(normalized, chunks);
    return { reply: text, model: google ? model : AZURE_DEPLOYMENT, quotesByChunk };
  }

  let response;
  armIdleTimer();
  try {
    response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: abortController.signal,
      body: JSON.stringify({
        model,
        stream: true,
        messages: [{ role: 'system', content: system }, ...messages],
        // Same knob, Ollama's spelling of it; unknown options are ignored, so an
        // older build simply behaves as before rather than erroring.
        options: { temperature: 0.2, frequency_penalty: FREQUENCY_PENALTY },
      }),
    });
  } catch (err) {
    clearTimeout(idleTimer);
    const stalled = err.name === 'AbortError';
    const httpError = new Error(stalled
      ? `Ollama at ${OLLAMA_URL} sent nothing for ${OLLAMA_IDLE_TIMEOUT / 1000}s`
      : `Ollama unreachable at ${OLLAMA_URL} (${err.message})`);
    httpError.status = 502;
    throw httpError;
  }

  if (!response.ok) {
    clearTimeout(idleTimer);
    const detail = await response.text().catch(() => '');
    const httpError = new Error(
      `Ollama HTTP ${response.status}${detail ? `: ${detail.slice(0, 200)}` : ''}`);
    httpError.status = 502;
    throw httpError;
  }

  // NDJSON: one JSON object per line, each carrying a token in message.content.
  let reply = '';
  let pendingLine = '';
  try {
    const decoder = new TextDecoder();
    for await (const streamBytes of response.body) {
      armIdleTimer();                          // progress — restart the idle clock
      pendingLine += decoder.decode(streamBytes, { stream: true });
      const lines = pendingLine.split('\n');
      pendingLine = lines.pop() ?? '';         // keep the trailing partial line
      for (const line of lines) {
        if (!line.trim()) continue;
        let streamEvent;
        try { streamEvent = JSON.parse(line); } catch { continue; }
        if (streamEvent.error) throw new Error(streamEvent.error);
        reply += streamEvent.message?.content || '';
      }
    }
  } catch (err) {
    const stalled = err.name === 'AbortError';
    const httpError = new Error(stalled
      ? `Ollama stalled — no output for ${OLLAMA_IDLE_TIMEOUT / 1000}s`
      : `Ollama stream failed: ${err.message}`);
    httpError.status = 502;
    throw httpError;
  } finally {
    clearTimeout(idleTimer);
  }

  // Repair before returning: the model's numbering is a guess, not an index.
  // normalizeGraphMarkers runs FIRST so [graph]/[g] become [G] before anything
  // else looks at markers, and last-resort strips [G] when no facts were sent —
  // a marker for context the model never received is noise.
  const normalized = normalizeGraphMarkers(
    normalizeCitations(reply.trim(), chunks.length), facts.length);
  const { text: repairedReply, quotesByChunk } = await repairCitations(normalized, chunks);
  return { reply: repairedReply, model, quotesByChunk };
}

// ---------------------------------------------------------------------------
// Citation repair
// ---------------------------------------------------------------------------

const CITE_MARKER   = /\[\s*\d+(?:\s*,\s*\d+)*\s*\]/g;
const QUOTED_SPAN   = /["“”']([^"“”']{25,})["“”']/g;
// words of literal overlap that pin a QUOTE to a chunk
const VERBATIM_RUN  = parseInt(process.env.RETRIEVER_VERBATIM_RUN || '8', 10);
// chars each side of a marker searched for its quote
const CLAIM_WINDOW  = parseInt(process.env.RETRIEVER_CLAIM_WINDOW || '400', 10);
// A paraphrase (no quote marks) is grounded when EITHER signal clears its bar:
//   - MIN_OVERLAP: fraction of the claim's content words present literally in a
//     chunk. Catches claims that reuse the source's vocabulary.
//   - GROUNDING_SIM: cosine between the claim and the chunk in embedding space.
//     Catches claims reworded in the model's own words, which share few literal
//     words but stay semantically close.
// Both, not just cosine: a short, generic true paraphrase can score LOWER on
// cosine than a plausible on-topic fabrication (measured: "…are enumerable"
// 0.45 vs a fabricated "…tabulated by Babbage on the difference engine" 0.48),
// but the true one reuses corpus words and clears MIN_OVERLAP. A claim is
// flagged [!] only when it fails both. GROUNDING_SIM is deliberately high so
// topical-but-unsupported claims still fall through.
const MIN_OVERLAP   = parseFloat(process.env.RETRIEVER_MIN_OVERLAP || '0.5');
const GROUNDING_SIM = parseFloat(process.env.RETRIEVER_GROUNDING_SIM || '0.6');

// Space-free comparison, same trick as the PDF viewer: immune to how each side
// hyphenates, spaces, or line-breaks the text.
const spaceless = (text) => tokenise(text || '').join('');

/**
 * Which retrieved chunk does this claim actually come from?
 * A literal run of VERBATIM_RUN words (or the whole claim, when shorter)
 * outranks any amount of loose word overlap; `verbatim` reports whether the
 * winning chunk contains such a run, since quotes are only trustable verbatim.
 */
function bestChunkFor(claim, chunkIndex) {
  const claimTokens = tokenise(claim);
  if (claimTokens.length < 4) return { chunkIdx: -1, score: 0, verbatim: false };
  const runLength = Math.min(VERBATIM_RUN, claimTokens.length);

  let best = { chunkIdx: -1, score: 0, verbatim: false };
  chunkIndex.forEach((indexed, chunkIdx) => {
    let verbatim = 0;
    for (let start = 0; start + runLength <= claimTokens.length; start++) {
      if (indexed.spaceless.includes(claimTokens.slice(start, start + runLength).join(''))) {
        verbatim = 1;
        break;
      }
    }
    const overlap = claimTokens.filter((token) => indexed.tokens.has(token)).length
      / claimTokens.length;
    const score = verbatim + overlap;
    if (score > best.score) best = { chunkIdx, score, verbatim: verbatim === 1 };
  });
  return best;
}

/** Quoted spans with their positions: "..." and blockquote ("> ...") lines. */
function quotedSpans(text) {
  const spans = [...text.matchAll(QUOTED_SPAN)].map((quote) => ({
    text: quote[1],
    start: quote.index,
    end: quote.index + quote[0].length,
  }));

  let lineStart = 0;
  for (const line of text.split('\n')) {
    if (/^\s*>\s?\S/.test(line)) {
      spans.push({
        text: line.replace(/^\s*>\s?/, ''),
        start: lineStart,
        end: lineStart + line.length,
      });
    }
    lineStart += line.length + 1;
  }
  return spans;
}

/** Chars between a span and a marker; 0 when the marker sits inside the span. */
function gapTo(span, markerStart, markerEnd) {
  if (span.end < markerStart) return markerStart - span.end;
  if (span.start > markerEnd) return span.start - markerEnd;
  return 0;
}

/**
 * The claim a marker refers to, when nothing nearby is quoted. Small models
 * routinely park the marker on its OWN line, after the sentence it supports:
 *
 *     - a computable number's decimal can be written down by a machine.
 *     [1]
 *
 * so we can't just split on \n and take the fragment touching the marker —
 * that fragment is the blank indentation, and the claim looks empty (which
 * flagged every such citation as unsupported). Instead: the claim is the last
 * NON-BLANK sentence before the marker; if it already ends a sentence (. ! ?)
 * that is the whole claim. Only a marker sitting mid-sentence glues on the
 * fragment that follows.
 */
function sentenceAround(text, markerStart, markerEnd) {
  const sentences = (segment) => segment
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

  const before = sentences(text.slice(Math.max(0, markerStart - CLAIM_WINDOW), markerStart));
  const lastBefore = before[before.length - 1] || '';
  if (/[.!?]$/.test(lastBefore)) return lastBefore;

  // Glue on only what continues the SAME line. A marker at end-of-line (or on
  // its own line) belongs to the sentence before it — reaching into the next
  // line pulls an unrelated claim in and tanks the grounding of both.
  const restOfLine = text.slice(markerEnd, markerEnd + CLAIM_WINDOW).split('\n')[0];
  const after = sentences(restOfLine);
  return `${lastBefore} ${after[0] || ''}`.trim();
}

/**
 * Re-point every [n] at the chunk whose text the claim actually came from. The
 * model quotes one excerpt and cites another (and copies the source paper's own
 * [12]-style refs verbatim), so a number in the reply cannot be trusted as an
 * index into `chunks`.
 *
 * A marker no chunk supports is flagged rather than trusted or dropped: it keeps
 * the model's own number but as [n!] (frontend renders a clickable warning), so
 * the reader can still open what it points at while knowing we couldn't verify
 * it. A marker with no usable number becomes a bare [!].
 *
 * Returns { text, quotesByChunk } — quotesByChunk[i] lists the verbatim quotes
 * that grounded chunk i, so the PDF viewer can highlight the quoted lines
 * themselves instead of guessing from query similarity.
 *
 * `embedClaims(texts) → vectors` (default: the MiniLM backend embedder) supplies
 * the semantic side of paraphrase grounding. Injected so tests can stub it and
 * so a caller without a model degrades to lexical-only grounding.
 */
export const UNSUPPORTED_MARKER = '[!]';
export async function repairCitations(text, chunks, embedClaims = embedTexts) {
  const quotesByChunk = chunks.map(() => []);
  if (!chunks.length) return { text, quotesByChunk };
  const chunkIndex = chunks.map((chunk) => ({
    spaceless: spaceless(chunk.text),
    tokens: new Set(tokenise(chunk.text || '')),
    embedding: chunk.embedding || null,
  }));
  const spans = quotedSpans(text);

  // Resolve each marker to a quote (verbatim path) or a paraphrase claim.
  const markers = [...text.matchAll(CITE_MARKER)].map((marker) => {
    const markerStart = marker.index;
    const markerEnd = markerStart + marker[0].length;
    // Cite the NEAREST quote, not the best-scoring one in the neighbourhood —
    // otherwise a strong quote two paragraphs away steals its neighbour's marker.
    const quote = spans
      .map((span) => ({ span, gap: gapTo(span, markerStart, markerEnd) }))
      .filter((candidate) => candidate.gap <= CLAIM_WINDOW)
      .sort((a, b) => a.gap - b.gap)[0];
    const claim = quote ? null : sentenceAround(text, markerStart, markerEnd);
    // The model's own cited excerpt(s), kept so an ungrounded marker can still
    // link to what it points at instead of collapsing to a bare warning.
    const citedNumbers = (marker[0].match(/\d+/g) || [])
      .map(Number).filter((num) => num >= 1 && num <= chunks.length);
    return { markerStart, markerEnd, quote, claim, citedNumbers };
  });

  // Batch-embed the paraphrase claims once (strip stray [n] markers first so
  // they don't pollute the vector). Only touched when there are paraphrases,
  // so a quotes-only reply never loads the model.
  const claimTexts = [...new Set(markers
    .map((marker) => marker.claim)
    .filter((claim) => claim && claim.length >= 8)
    .map((claim) => claim.replace(CITE_MARKER, ' ').replace(/\s+/g, ' ').trim()))];
  const claimVectors = new Map();
  if (claimTexts.length) {
    try {
      const vectors = await embedClaims(claimTexts);
      claimTexts.forEach((claim, claimIdx) => claimVectors.set(claim, vectors[claimIdx]));
    } catch (err) {
      console.warn('[retriever] claim embedding failed — grounding lexically only:', err.message);
    }
  }

  // Best chunk for a paraphrase claim by cosine, and the similarity itself.
  const bestBySimilarity = (claim) => {
    const vector = claimVectors.get(claim.replace(CITE_MARKER, ' ').replace(/\s+/g, ' ').trim());
    if (!vector) return { chunkIdx: -1, sim: 0 };
    let best = { chunkIdx: -1, sim: 0 };
    chunkIndex.forEach((indexed, chunkIdx) => {
      if (!indexed.embedding) return;
      const sim = dot(vector, indexed.embedding);
      if (sim > best.sim) best = { chunkIdx, sim };
    });
    return best;
  };

  let repaired = '';
  let cursor = 0;
  for (const { markerStart, markerEnd, quote, claim, citedNumbers } of markers) {
    let target = -1;
    if (quote) {
      // A quote claims to be verbatim, so it must literally appear in a chunk —
      // verbatim run required. Word overlap and cosine are NOT enough: a quote
      // built from common words matches many chunks, and a reworded "quote" is
      // still a fabricated quote. If no chunk contains it, the model made it up.
      const best = bestChunkFor(quote.span.text, chunkIndex);
      if (best.verbatim) {
        target = best.chunkIdx;
        if (!quotesByChunk[target].includes(quote.span.text)) {
          quotesByChunk[target].push(quote.span.text);
        }
      }
    } else {
      // Paraphrase: grounded if it reuses the chunk's words (lexical overlap) OR
      // is semantically close to it (cosine). The model rewords the source, so
      // literal overlap alone flagged genuine paraphrases as fabricated.
      const lexical = bestChunkFor(claim, chunkIndex);
      if (lexical.score >= MIN_OVERLAP) {
        target = lexical.chunkIdx;
      } else {
        const semantic = bestBySimilarity(claim);
        if (semantic.sim >= GROUNDING_SIM) target = semantic.chunkIdx;
      }
    }

    repaired += text.slice(cursor, markerStart);
    // Grounded → the verified excerpt. Ungrounded → keep the model's own
    // citation but flag it ([n!]), so the reader can still open what it points
    // at while seeing it's unverified; only a marker with no usable number
    // drops to a bare [!].
    if (target >= 0) repaired += `[${target + 1}]`;
    else if (citedNumbers.length) repaired += `[${citedNumbers.join(', ')}!]`;
    else repaired += UNSUPPORTED_MARKER;
    cursor = markerEnd;
  }
  repaired += text.slice(cursor);

  // Tidy spacing the rewrite may have doubled up.
  return {
    text: repaired.replace(/ {2,}/g, ' ').replace(/ ([.,;:)])/g, '$1'),
    quotesByChunk,
  };
}

/**
 * Coerce the graph marker into the single [G] form the frontend styles, and drop
 * it entirely when no facts were in the context.
 *
 * [G] is intentionally invisible to repairCitations: a triple is the extraction
 * model's paraphrase of a paper, so it appears verbatim in no chunk. Sent
 * through the grounding check it would either be flagged as hallucinated — a
 * true, sourced fact reported as fabricated — or clear the cosine bar against
 * some unrelated chunk and be silently rewritten into a [n] pointing at text
 * that does not support it. CITE_MARKER only matches digits, so [G] passes
 * through untouched; this function just tidies the spellings models emit.
 */
function normalizeGraphMarkers(text, factCount) {
  const tidied = text.replace(/\[\s*(?:G|g|graph|Graph|KG|kg)\s*\]/g, '[G]');
  return factCount ? tidied : tidied.replace(/\[G\]/g, '');
}

/**
 * Coerce the citation markers small models actually emit into the [n] form
 * the frontend links on: [n1], [N1], [#1], [source 1], [1](...) → [1].
 * Markers pointing past the excerpt count are left alone (they'd link nowhere).
 */
function normalizeCitations(text, excerptCount) {
  return text
    .replace(/\[\s*(?:n|N|#|source|excerpt|ref)\s*[.:]?\s*(\d+(?:\s*,\s*\d+)*)\s*\]/g, '[$1]')
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]\([^)]*\)/g, '[$1]')
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (marker, markerNumbers) => {
      const validNumbers = markerNumbers.split(',')
        .map((number) => parseInt(number, 10))
        .filter((number) => number >= 1 && number <= excerptCount);
      return validNumbers.length ? `[${validNumbers.join(', ')}]` : marker;
    });
}
