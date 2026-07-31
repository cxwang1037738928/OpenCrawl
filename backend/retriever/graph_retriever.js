/**
 * graph_retriever.js — knowledge-graph side of RAG: query → entities → triples
 *
 * Runs in the request path next to retriever.js, deliberately: matching a query
 * against the graph is dictionary lookup plus two hops over ~24k edges, which is
 * microseconds in memory. Spawning Python per chat message to do it would cost
 * more in interpreter startup than the work itself.
 *
 * Query entities are found by GAZETTEER, not by NER. An off-the-shelf NER model
 * predicts PERSON/ORG/LOC and would not recognise 'band gap' or
 * 'Perdew-Burke-Ernzerhof' at all, whereas the graph already knows every name it
 * can answer about. Cluster members are indexed alongside representatives, so a
 * query saying 'DFT' or 'NequIP' still finds the node those merged into.
 *
 * Expansion is TWO hops but refuses to travel through hub entities. Measured on
 * a 192-document corpus: unrestricted 2-hop from any single seed reaches 41-54%
 * of the corpus, and unbounded expansion reaches 99% — because 57% of the graph
 * is one connected component. Hubs are the cause ('machine learning' has degree
 * 399, in 60 of 192 papers), so they are allowed as seeds but never as
 * stepping stones.
 *
 * Triples are ranked by specificity × corroboration:
 *   specificity  idf = log(N / documents the seed appears in). A seed in 60/192
 *                papers barely narrows anything; one in 3/192 nearly names the
 *                paper. Measured spread: 1.16 to 4.16.
 *   corroboration  how many papers state the same triple. A weak signal in
 *                practice — measured, 98.9% of triples come from one paper —
 *                so it acts as a tiebreaker rather than a driver.
 *
 * Exports:
 *   buildGraphIndex(graph)                  — gazetteer + adjacency, built once per corpus
 *   matchSeeds(queryText, index)            — exact entity mentions in a query
 *   matchSeedsByEmbedding(spans, index, ..) — fallback for unmatched spans
 *   graphFacts(queryText, index, opts)      — ranked triples with their documents
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { embedTexts } from './embedder.js';

// Curated artifact names — table headers and generic nouns, NOT a length or digit
// rule: in a materials corpus the short names are element symbols (Li, Ti, C, O)
// and among the most important entities in the graph.
const JUNK_SEEDS_PATH = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)), '..', 'extraction', 'junk_seeds.txt');

let _junkSeeds = null;
function junkSeeds() {
  if (_junkSeeds) return _junkSeeds;
  try {
    _junkSeeds = new Set(fs.readFileSync(JUNK_SEEDS_PATH, 'utf-8')
      .split('\n').map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#')));
  } catch {
    // A missing list must not disable graph retrieval, only its filtering.
    console.warn(`[graph_retriever] no junk seed list at ${JUNK_SEEDS_PATH}`);
    _junkSeeds = new Set();
  }
  return _junkSeeds;
}

// Hops out from a seed. Two, with hub gating; see the header.
const HOPS = parseInt(process.env.GRAPH_HOPS || '2', 10);
// An entity appearing in more than this many documents is a hub: usable as a
// seed, never traversed through. 192-doc corpus: only 2 entities exceed 20.
const HUB_DOC_FREQ = parseInt(process.env.GRAPH_HUB_DOC_FREQ || '20', 10);
// Documents an entity must appear in before it may SEED a query. Extraction
// artifacts sit in exactly one paper, and because idf = log(N/df) that made them
// score HIGHEST — measured, the word "a" seeded at idf 5.26 while
// "graph neural networks" scored 3.06. Requiring two papers removes that whole
// class without touching real but rare entities, which the corpus repeats.
const MIN_SEED_DOC_FREQ = parseInt(process.env.GRAPH_MIN_SEED_DOC_FREQ || '2', 10);
// Multiplied into a fact's score for every hop past the first. Without it a
// second-hop fact scores like a first-hop one, and second-hop facts vastly
// outnumber them — measured on a CGCNN question, the top 8 filled with band-gap
// facts reached at hop 2 while 'CGCNN are invariant to unit cells', a direct
// neighbour and the actual answer, never surfaced.
const HOP_DECAY = parseFloat(process.env.GRAPH_HOP_DECAY || '0.5');
// Triples handed to the model. The context already carries 8 chunks, and an
// unbounded 2-hop expansion can reach thousands of triples.
const MAX_FACTS = parseInt(process.env.GRAPH_MAX_FACTS || '25', 10);
// Longest entity name, in words, the gazetteer will try to match.
const MAX_NGRAM = parseInt(process.env.GRAPH_MAX_NGRAM || '6', 10);
// Cosine floor for the embedding fallback. High, because MiniLM is unreliable on
// short entity strings — measured on this corpus, 'DFT' vs 'density functional
// theory' scored 0.269, BELOW unrelated pairs. Exact matching carries the load.
const EMBED_SIMILARITY = parseFloat(process.env.GRAPH_EMBED_SIMILARITY || '0.75');

// Mirrors entity_resolution.resolution_key in Python: joiners and punctuation
// always fold, case folds only above 4 chars and outside formula shapes, so CO
// stays distinct from Co. \p{L}\p{N} rather than \w to match Python's
// unicode-aware \w — subscripted formulae would otherwise lose characters.
const JOINERS = /[\s\-_]+/gu;
const PUNCTUATION = /[^\p{L}\p{N}_]/gu;
const FORMULA = /^(?=.*\d)(?:[A-Z][a-z]?\d*|[()[\]·.+-]|\d)+$/u;

export function resolutionKey(entity) {
  const stripped = (entity || '').trim();
  const key = stripped.replace(JOINERS, '').replace(PUNCTUATION, '');
  return stripped.length > 4 && !FORMULA.test(stripped) ? key.toLowerCase() : key;
}

/**
 * Gazetteer + adjacency for one graph payload, built once and cached with the
 * corpus. Nothing here is per-query.
 */
export function buildGraphIndex(graph) {
  if (!graph?.entities?.length || !graph?.relations?.length) return null;
  const relations = graph.relations;
  const relationDocIds = graph.relationDocIds || relations.map(() => []);

  // Which relations touch an entity, and which documents it appears in.
  // Self-loops are skipped: merging an acronym into its expansion turns
  // "NequIP is Neural Equivariant Interatomic Potentials" into "X is X", and
  // because every paper defining the acronym states it, those carry high
  // corroboration and would otherwise top every ranking with nothing said.
  const junk = junkSeeds();
  const relationsOf = new Map();
  const docsOf = new Map();
  relations.forEach((relation, relationIdx) => {
    if (relation[0] === relation[2]) return;
    // Dropped outright, not merely blocked as a seed: a triple ending in a table
    // header is noise wherever it surfaces.
    if (junk.has(relation[0]) || junk.has(relation[2])) return;
    const docIds = relationDocIds[relationIdx] || [];
    for (const entity of [relation[0], relation[2]]) {
      if (!relationsOf.has(entity)) relationsOf.set(entity, []);
      relationsOf.get(entity).push(relationIdx);
      if (!docsOf.has(entity)) docsOf.set(entity, new Set());
      for (const docId of docIds) docsOf.get(entity).add(docId);
    }
  });

  // Gazetteer: every entity, plus every merged surface form pointing at the
  // representative it collapsed into. Without the cluster members a query
  // saying 'NequIP' finds nothing, because that node is now its expansion.
  const entitySet = new Set(graph.entities.filter((entity) => !junk.has(entity)));
  const keyToEntity = new Map();
  for (const entity of graph.entities) {
    if (junk.has(entity)) continue;
    const key = resolutionKey(entity);
    if (key) keyToEntity.set(key, entity);
  }
  for (const [representative, members] of Object.entries(graph.entityClusters || {})) {
    // A representative that is no longer an entity is a STALE cluster key: the
    // lexical pass names a group after one form, then the abbreviation pass
    // merges that form into an expansion without retiring the old key. Mapping
    // to it would hand back an entity that has no relations — the symptom being
    // a query for 'NequIP' finding a seed and then zero facts.
    if (!entitySet.has(representative)) continue;
    for (const member of members) {
      const key = resolutionKey(member);
      // Never shadow a real entity with a merged alias of a different one.
      if (key && !keyToEntity.has(key)) keyToEntity.set(key, representative);
    }
  }

  const documentCount = (graph.sourceDocIds || []).length
    || new Set(relationDocIds.flat()).size || 1;

  return { relations, relationDocIds, relationsOf, docsOf, keyToEntity, documentCount };
}

/** log(N / df) for a seed — how much finding it narrows the corpus. */
export function specificity(entity, index) {
  const docFreq = index.docsOf.get(entity)?.size || 1;
  return Math.log(index.documentCount / docFreq);
}

const isHub = (entity, index) =>
  (index.docsOf.get(entity)?.size || 0) > HUB_DOC_FREQ;

/**
 * Entity mentions in a query, longest match first.
 *
 * Longest-first and non-overlapping on purpose: matching every n-gram turns
 * "density functional theory" into four seeds (the phrase plus 'density',
 * 'theory', 'functional'), and the generic ones are hubs that swamp the ranking.
 * Returns { seeds, unmatched } — unmatched spans are what the embedding
 * fallback can try.
 */
export function matchSeeds(queryText, index) {
  const words = (queryText || '').split(/\s+/).filter(Boolean);
  const seeds = new Map();
  const unmatched = [];
  let wordIdx = 0;

  while (wordIdx < words.length) {
    let matched = 0;
    for (let span = Math.min(MAX_NGRAM, words.length - wordIdx); span >= 1; span--) {
      const phrase = words.slice(wordIdx, wordIdx + span).join(' ');
      const entity = index.keyToEntity.get(resolutionKey(phrase));
      if (entity) {
        seeds.set(entity, Math.max(seeds.get(entity) || 0, span));
        matched = span;
        break;
      }
    }
    if (matched) {
      wordIdx += matched;
    } else {
      unmatched.push(words[wordIdx]);
      wordIdx += 1;
    }
  }
  return { seeds: [...seeds.keys()], unmatched };
}

/**
 * Entities semantically near a query span, for spans the gazetteer missed.
 *
 * Deliberately NOT the primary path: it embeds every entity name, which is
 * seconds of work and tens of MB, and cosine on short names is unreliable (see
 * EMBED_SIMILARITY). Enable with GRAPH_EMBED_FALLBACK=1 and expect it to help
 * with multi-word paraphrases, not acronyms — those are what entityClusters is
 * for.
 */
export async function matchSeedsByEmbedding(spans, index, embed = embedTexts) {
  if (!spans.length) return [];
  const entities = [...new Set(index.relations.flatMap((r) => [r[0], r[2]]))];
  index._entityVectors ??= await embed(entities);
  const spanVectors = await embed(spans);

  const found = new Set();
  for (const spanVector of spanVectors) {
    let best = { entity: null, sim: 0 };
    index._entityVectors.forEach((entityVector, entityIdx) => {
      let sim = 0;
      for (let dim = 0; dim < spanVector.length; dim++) sim += spanVector[dim] * entityVector[dim];
      if (sim > best.sim) best = { entity: entities[entityIdx], sim };
    });
    if (best.entity && best.sim >= EMBED_SIMILARITY) found.add(best.entity);
  }
  return [...found];
}

/**
 * Ranked triples for a query, each with the documents that state it.
 *
 * Seeds may be hubs; the frontier may not. Every triple keeps the specificity of
 * the best seed that reached it, so a fact found via a rare entity outranks the
 * same fact found via 'machine learning'.
 */
export function graphFacts(queryText, index,
                           { maxFacts = MAX_FACTS, hops = HOPS, hopDecay = HOP_DECAY } = {}) {
  if (!index) return { facts: [], seeds: [] };
  const { seeds: matched } = matchSeeds(queryText, index);
  // The floor is applied HERE rather than in matchSeeds so callers can still
  // inspect what a query mentioned, even when it is too rare to expand from.
  const seeds = matched.filter(
    (entity) => (index.docsOf.get(entity)?.size || 0) >= MIN_SEED_DOC_FREQ);
  if (!seeds.length) return { facts: [], seeds: [] };

  // relationIdx → best specificity of any seed that reached it.
  const reached = new Map();
  const visited = new Set(seeds);
  let frontier = seeds.map((entity) => ({ entity, weight: specificity(entity, index) }));

  for (let hop = 0; hop < hops && frontier.length; hop++) {
    const next = [];
    for (const { entity, weight } of frontier) {
      for (const relationIdx of index.relationsOf.get(entity) || []) {
        if ((reached.get(relationIdx) || 0) < weight) reached.set(relationIdx, weight);
        const [subject, , object] = index.relations[relationIdx];
        for (const neighbour of [subject, object]) {
          // Hubs terminate a path: expanding through them is what takes 2-hop
          // expansion to half the corpus.
          if (visited.has(neighbour) || isHub(neighbour, index)) continue;
          visited.add(neighbour);
          // Decay applies on the way OUT, so a fact keeps the weight of the hop
          // that reached it rather than the hop it leads to.
          next.push({ entity: neighbour, weight: weight * hopDecay });
        }
      }
    }
    frontier = next;
  }

  const facts = [...reached.entries()]
    .map(([relationIdx, seedSpecificity]) => {
      const docIds = index.relationDocIds[relationIdx] || [];
      return {
        subject:   index.relations[relationIdx][0],
        predicate: index.relations[relationIdx][1],
        object:    index.relations[relationIdx][2],
        docIds,
        // specificity x corroboration, already decayed by distance from the seed.
        score: seedSpecificity * Math.max(docIds.length, 1),
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, maxFacts);

  return { facts, seeds };
}
