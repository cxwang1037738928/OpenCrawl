/**
 * chunker.js — structure-aware chunking for DoclingDocument extracts
 *
 * Two entry points:
 *
 *   chunkDocument(entry, opts)  — PREFERRED. Takes one doclings.json entry
 *     (the output of extract.py) and chunks along the document's own
 *     structure: section boundaries are never crossed, every chunk carries
 *     its section heading as an embedded prefix (so the embedding "knows"
 *     where the text came from), tiny adjacent sections are merged so you
 *     don't get fragment chunks, and tables are emitted as standalone
 *     chunks instead of being shredded mid-row.
 *
 *   chunkText(text, chunkSize, overlap) — fallback for plain text with no
 *     structure (keeps the signature embed.js already uses). Sentence-aware
 *     sliding window: chunks end on sentence boundaries where possible and
 *     overlap by whole sentences, not by an arbitrary character cut.
 *
 * Sizes are in WORDS, not characters. Note the embedding model
 * (all-MiniLM-L12-v2) truncates input around 256 word-piece tokens, i.e.
 * roughly 180–200 English words. CHUNK_SIZE much above ~200 means the tail
 * of every chunk is silently thrown away at embed time.
 */

import 'dotenv/config';

const DEFAULTS = {
  chunkSize: process.env.CHUNK_SIZE ? parseInt(process.env.CHUNK_SIZE) : 180,
  // 0 by default: chunks already stop at docling section boundaries, so the
  // sliding window never straddles a topic change, and splitSentences() below
  // ends chunks on sentence boundaries — the failure overlap normally guards
  // against (a sentence cut in half) cannot happen here. Overlap cost what it
  // bought: ~18% of corpus text stored and processed twice, duplicate
  // near-identical chunks competing for RETRIEVER_TOP_K slots, and deflated
  // BM25 idf (a term in an overlap region has df=2, and retriever.js squares
  // idf). Set CHUNK_OVERLAP > 0 to restore it; the carry logic below is intact
  // and a re-embed is required either way.
  overlap:   process.env.CHUNK_OVERLAP ? parseInt(process.env.CHUNK_OVERLAP) : 0,
  // Sections shorter than this (words) are merged into their neighbour.
  minSectionMerge: parseInt(process.env.CHUNKER_MIN_SECTION_MERGE || '60', 10),
  // A table chunk is truncated past this many words.
  maxTableWords:   parseInt(process.env.CHUNKER_MAX_TABLE_WORDS  || '300', 10),
};

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

const wordCount = (text) => (text.match(/\S+/g) || []).length;

/**
 * Split text into sentences. Deliberately conservative: splits on
 * .!? followed by whitespace + capital/opening char, and avoids splitting
 * on common abbreviations and initials ("et al.", "Fig.", "e.g.", "J. Smith")
 * that are everywhere in academic prose.
 */
function splitSentences(text) {
  const ABBREV = /(?:et al|e\.g|i\.e|cf|vs|fig|figs|eq|eqs|sec|ref|refs|no|vol|pp|dr|mr|ms|prof|jr|sr|approx)\.$/i;
  const sentences = [];
  let sentenceStart = 0;
  const sentenceBoundary = /[.!?]+["')\]]*\s+(?=[A-Z0-9("'\[])/g;
  let boundary;
  while ((boundary = sentenceBoundary.exec(text)) !== null) {
    const candidate = text.slice(sentenceStart, boundary.index + boundary[0].length);
    const trimmed = candidate.trimEnd();
    // Don't split right after an abbreviation or a single-letter initial
    if (ABBREV.test(trimmed) || /\b[A-Z]\.$/.test(trimmed)) continue;
    if (trimmed) sentences.push(trimmed);
    sentenceStart = boundary.index + boundary[0].length;
  }
  const tail = text.slice(sentenceStart).trim();
  if (tail) sentences.push(tail);
  return sentences.length ? sentences : (text.trim() ? [text.trim()] : []);
}

/**
 * Sentence-aware sliding window over plain text.
 * Returns string[] — same shape embed.js already expects.
 */
export function chunkText(text, chunkSize = DEFAULTS.chunkSize, overlap = DEFAULTS.overlap) {
  if (!text || !text.trim()) return [];
  const sentences = splitSentences(text);
  const chunks = [];
  let currentSentences = [];
  let currentWordCount = 0;

  const flush = () => {
    if (!currentSentences.length) return;
    chunks.push(currentSentences.join(' '));
    let carriedSentences = [];
    let carriedWordCount = 0;
    for (let sentenceIdx = currentSentences.length - 1; sentenceIdx >= 0 && carriedWordCount < overlap; sentenceIdx--) {
      carriedSentences.unshift(currentSentences[sentenceIdx]);
      carriedWordCount += wordCount(currentSentences[sentenceIdx]);
    }
    // Overlap must never be the whole chunk, or we'd loop forever
    if (carriedWordCount >= currentWordCount) carriedSentences = [];
    currentSentences = carriedSentences;
    currentWordCount = currentSentences.reduce((total, sentence) => total + wordCount(sentence), 0);
  };

  for (const sentence of sentences) {
    const sentenceWordCount = wordCount(sentence);
    if (sentenceWordCount >= chunkSize) {
      // Pathological sentence (equations, mangled OCR): hard-split by words
      flush();
      const words = sentence.split(/\s+/);
      for (let windowStart = 0; windowStart < words.length; windowStart += chunkSize - overlap) {
        chunks.push(words.slice(windowStart, windowStart + chunkSize).join(' '));
      }
      currentSentences = [];
      currentWordCount = 0;
      continue;
    }
    if (currentWordCount + sentenceWordCount > chunkSize && currentWordCount > 0) flush();
    // After flush the carry is kept as overlap; if overlap + sentence still
    // overflows (large sentence near the chunk limit), drop the carry so the
    // final chunk stays within chunkSize.
    if (currentWordCount + sentenceWordCount > chunkSize) { currentSentences = []; currentWordCount = 0; }
    currentSentences.push(sentence);
    currentWordCount += sentenceWordCount;
  }
  if (currentSentences.length) chunks.push(currentSentences.join(' '));
  return chunks;
}

// ---------------------------------------------------------------------------
// Structure-aware chunking over a doclings.json entry
// ---------------------------------------------------------------------------

/**
 * chunkDocument(entry, opts) → [{ text, heading, sectionIndex, type, pages, prefixLen }]
 *
 * entry is one value from doclings.json:
 *   { text, markdown, sections: [{heading, text, pages}], tables: [str], metadata }
 *
 * pages is the section's 1-based [first, last] PDF page range (null when the
 * extractor recorded no provenance); prefixLen is the length in characters of
 * the embedded "title — heading\n" prefix, so a viewer can strip it before
 * locating the chunk body in the source PDF.
 *
 * Strategy:
 *   1. Merge runs of tiny sections (< minSectionMerge words) into one unit so
 *      front-matter fragments don't become their own near-empty chunks.
 *   2. Within each unit, sentence-chunk the body and prefix every chunk with
 *      its heading path — the heading text participates in the embedding,
 *      which measurably helps retrieval for queries phrased like headings
 *      ("related work on X", "evaluation methodology").
 *   3. Emit each table as ONE chunk (truncated if huge). Splitting a
 *      markdown table mid-row produces chunks that embed as noise.
 *   4. If the entry has no usable sections, fall back to chunkText on
 *      entry.text so nothing silently drops out of the index.
 */
export function chunkDocument(entry, opts = {}) {
  const { chunkSize, overlap, minSectionMerge, maxTableWords } = { ...DEFAULTS, ...opts };
  const sections = (entry.sections || []).filter(
    (section) => (section.text || '').trim() || (section.heading || '').trim());

  if (sections.length === 0) {
    return chunkText(entry.text || '', chunkSize, overlap).map((text) => ({
      text,
      heading: null,
      sectionIndex: null,
      type: 'text',
      pages: null,
      prefixLen: 0,
    }));
  }

  const mergePages = (rangeA, rangeB) => {
    if (!rangeA) return rangeB || null;
    if (!rangeB) return rangeA;
    return [Math.min(rangeA[0], rangeB[0]), Math.max(rangeA[1], rangeB[1])];
  };

  // --- 1. merge tiny adjacent sections -------------------------------------
  const units = []; // { heading, text, sectionIndex, pages }
  for (let sectionIdx = 0; sectionIdx < sections.length; sectionIdx++) {
    const section = sections[sectionIdx];
    const body = (section.text || '').trim();
    const heading = (section.heading || '').trim();
    const bodyWordCount = wordCount(body);

    const previousUnit = units[units.length - 1];
    if (previousUnit && wordCount(previousUnit.text) < minSectionMerge && bodyWordCount < minSectionMerge) {
      previousUnit.text = [previousUnit.text, heading ? `${heading}. ${body}` : body]
        .filter(Boolean).join(' ');
      previousUnit.pages = mergePages(previousUnit.pages, section.pages || null);
    } else {
      units.push({ heading, text: body, sectionIndex: sectionIdx, pages: section.pages || null });
    }
  }

  // --- 2. chunk each unit, prefixing the heading ----------------------------
  const title = entry.metadata?.title || '';
  const chunks = [];
  for (const unit of units) {
    const prefixParts = [title, unit.heading].filter(Boolean);
    const prefix = prefixParts.length ? `${prefixParts.join(' — ')}\n` : '';
    const prefixWordCount = wordCount(prefix);

    const bodyChunks = chunkText(unit.text, Math.max(chunkSize - prefixWordCount, 50), overlap);
    if (bodyChunks.length === 0 && unit.heading) continue;   // heading-only section
    for (const body of bodyChunks) {
      chunks.push({
        text: prefix + body,
        heading: unit.heading || null,
        sectionIndex: unit.sectionIndex,
        type: 'text',
        pages: unit.pages,
        prefixLen: prefix.length,
      });
    }
  }

  // --- 3. tables as standalone chunks ---------------------------------------
  // Tables are {text, page} since page provenance landed; older extracts
  // stored plain markdown strings.
  (entry.tables || []).forEach((table) => {
    const tableText = ((typeof table === 'string' ? table : table?.text) || '').trim();
    if (!tableText) return;
    const page = typeof table === 'object' ? table?.page ?? null : null;
    const tableWords = tableText.split(/\s+/);
    const truncated = tableWords.length > maxTableWords
      ? tableWords.slice(0, maxTableWords).join(' ') + ' …'
      : tableText;
    const tablePrefix = title ? `${title} — Table\n` : '';
    chunks.push({
      text: tablePrefix + truncated,
      heading: 'Table',
      sectionIndex: null,
      type: 'table',
      pages: page ? [page, page] : null,
      prefixLen: tablePrefix.length,
    });
  });

  return chunks;
}

export default { chunkText, chunkDocument };