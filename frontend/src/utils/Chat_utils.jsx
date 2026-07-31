/**
 * Chat_utils.jsx — non-component helpers for Chat.jsx: citation-marker
 * parsing/rendering, the unsupported-citation tooltip, minimal inline
 * markdown, query embedding, and conversation serialization.
 */

import { embedTexts } from '../lib/embedder.js';

/** Embed the question in-browser with the corpus MiniLM model. */
export async function embedQuery(text, onStatus) {
  onStatus('embedding your question…');
  const [queryVector] = await embedTexts([text], onStatus);
  return queryVector;
}

// Verified citations. CITE_MARKER_RE matches a whole [n] marker (plus the
// [n1]/[#1]/[ref 1] variants small models drift into); CITE_NUMBERS_RE anchors
// it and captures the numbers.
export const CITE_MARKER_RE = /\[\s*(?:n|N|#|source|excerpt|ref)?\s*[.:]?\s*\d+(?:\s*,\s*\d+)*\s*\]/;
export const CITE_NUMBERS_RE = /^\[\s*(?:n|N|#|source|excerpt|ref)?\s*[.:]?\s*(\d+(?:\s*,\s*\d+)*)\s*\]$/;

// Ungrounded citations (retriever.js): [n!] still points at the model's cited
// excerpt but is flagged; a bare [!] is one with no usable excerpt number.
export const UNSUPPORTED_MARKER = '[!]';
export const FLAGGED_CITE_RE = /\[\s*\d+(?:\s*,\s*\d+)*\s*!\s*\]/;
export const FLAGGED_NUMBERS_RE = /^\[\s*(\d+(?:\s*,\s*\d+)*)\s*!\s*\]$/;
export const UNSUPPORTED_TIP =
  'This information is possibly hallucinated. No retrieved excerpt contains this exact sentence.';

// [G] (retriever.js) — the claim leaned on a knowledge-graph triple rather than
// excerpt text. Not a hallucination flag: the fact has documents behind it, but
// a triple is machine-extracted, so it carries no verbatim text to verify
// against and is exempt from the excerpt grounding check.
export const GRAPH_MARKER = '[G]';
export const GRAPH_MARKER_RE = /\[G\]/;
export const GRAPH_TIP =
  'This information was synthesized with triplets from the Knowledge graph and could contain errors';

// One capture group, so split() hands markers back interleaved with the prose.
export const SEGMENT_RE = new RegExp(
  `(${CITE_MARKER_RE.source}|${FLAGGED_CITE_RE.source}|\\[!\\]|${GRAPH_MARKER_RE.source})`, 'g');

// The disclaimer tooltip is position:fixed and placed here on hover/focus: the
// chat scroller clips overflow, so centering it on the marker pushed it off the
// edge. Clamp it to the viewport (arrow stays on the marker), flipping below
// only when there's no room above.
const TIP_MARGIN = 8;
export function positionTip(event) {
  const marker = event.currentTarget;
  const tip = marker.querySelector('.unsupported-tip');
  if (!tip) return;
  const markerRect = marker.getBoundingClientRect();
  const tipRect = tip.getBoundingClientRect();
  const markerCenterX = markerRect.left + markerRect.width / 2;
  const left = Math.max(TIP_MARGIN,
    Math.min(markerCenterX - tipRect.width / 2, window.innerWidth - tipRect.width - TIP_MARGIN));
  const fitsAbove = markerRect.top - tipRect.height - TIP_MARGIN >= 0;
  tip.style.left = `${left}px`;
  tip.style.top = `${fitsAbove ? markerRect.top - tipRect.height - 8 : markerRect.bottom + 8}px`;
  tip.style.setProperty('--arrow-left',
    `${Math.max(12, Math.min(markerCenterX - left, tipRect.width - 12))}px`);
  tip.classList.toggle('tip-below', !fitsAbove);
}

/**
 * The claim sentence each marker sits in — one entry per marker occurrence, in
 * document order, so clicking a specific marker can highlight what ITS sentence
 * refers to rather than every sentence citing the chunk. Works on the block's
 * markdown-stripped text with the markers themselves removed.
 *
 * Code spans are dropped whole (not just their backticks): inlineMd renders
 * code content without citation buttons, so counting a marker inside one here
 * would desync every later marker in the block from its claim.
 *
 * A sentence that is nothing but markers (models often write "claim. [1]",
 * which the splitter separates) inherits the previous sentence as its claim.
 */
export function markerCitingSentences(blockText) {
  const plain = (blockText || '').replace(/`[^`]+`/g, ' ').replace(/\*\*|\*/g, '');
  const claims = [];
  let lastClaim = '';
  for (const sentence of plain.split(/(?<=[.!?])\s+/)) {
    const claim = sentence.replace(SEGMENT_RE, ' ').replace(/\s+/g, ' ').trim();
    if (claim) lastClaim = claim;
    const markerCount = (sentence.match(SEGMENT_RE) || []).length;
    for (let occurrence = 0; occurrence < markerCount; occurrence++) {
      claims.push(claim || lastClaim);
    }
  }
  return claims;
}

/**
 * Red, non-clickable "!" carrying the hover/focus disclaimer — the flag itself.
 * Reused for a bare [!] and for the mark trailing a flagged [n!] citation.
 */
export function unsupportedMark(key) {
  // Custom tooltip (not title=): the native one is slow to appear and easy to
  // miss; positionTip clamps it to the viewport on hover/focus.
  return (
    <span className="citation-unsupported" key={key}
          tabIndex={0} role="img" aria-label={UNSUPPORTED_TIP}
          onMouseEnter={positionTip} onFocus={positionTip}>
      !
      <span className="unsupported-tip" role="tooltip" aria-hidden="true">
        {UNSUPPORTED_TIP}
      </span>
    </span>
  );
}

/**
 * Yellow, non-clickable "!" marking a claim built on graph triples. Deliberately
 * not a link: a triple cites documents rather than a span in one, so there is no
 * single place in a PDF to open.
 */
export function graphMark(key) {
  return (
    <span className="citation-graph" key={key}
          tabIndex={0} role="img" aria-label={GRAPH_TIP}
          onMouseEnter={positionTip} onFocus={positionTip}>
      !
      <span className="unsupported-tip" role="tooltip" aria-hidden="true">
        {GRAPH_TIP}
      </span>
    </span>
  );
}

/**
 * [n] → a blue button opening the cited chunk. [n!] → the same blue button(s)
 * followed by a red non-clickable "!" flagging it unverified. [!] → the flag
 * alone, with no chunk to open. [G] → a yellow "!" marking graph-derived
 * synthesis, which may trail an excerpt citation as [3][G].
 * `claimCursor` = { claims, next }: the block's citing sentences, consumed one
 * per marker in document order so each marker carries its own claim.
 */
export function citeInline(text, sources, query, onCitation, keyBase, claimCursor) {
  return text.split(SEGMENT_RE).map((part, partIdx) => {
    if (part === UNSUPPORTED_MARKER) {
      if (claimCursor) claimCursor.next += 1;   // [!] is a marker occurrence too
      return unsupportedMark(`${keyBase}u${partIdx}`);
    }
    // [G] does NOT advance the claim cursor: it annotates the same claim as any
    // [n] beside it, so consuming a slot would misalign every later marker.
    if (part === GRAPH_MARKER) return graphMark(`${keyBase}g${partIdx}`);
    const flaggedMatch = part.match(FLAGGED_NUMBERS_RE);
    const citeMatch = flaggedMatch || part.match(CITE_NUMBERS_RE);
    if (!citeMatch) return part;                        // plain prose
    const citing = claimCursor ? claimCursor.claims[claimCursor.next++] : undefined;
    if (!sources?.length || !onCitation) return part;
    const sourceNumbers = citeMatch[1].split(',').map((number) => parseInt(number, 10));
    if (sourceNumbers.some((sourceNumber) => sourceNumber < 1 || sourceNumber > sources.length)) return part;
    return (
      <span className="citation-group" key={`${keyBase}c${partIdx}`}>
        {sourceNumbers.map((sourceNumber) => {
          const source = sources[sourceNumber - 1];
          return (
            <button
              key={sourceNumber}
              className="citation-link"
              title={`${source.filename}${source.heading ? ` — ${source.heading}` : ''}${source.pages ? ` · p.${source.pages[0]}` : ''}`}
              onClick={() => onCitation(source, query, citing)}
            >
              {sourceNumber}
            </button>
          );
        })}
        {/* Flagged: the blue link(s) still open the chunk; the trailing red !
            (with disclaimer) marks the citation as unverified. */}
        {flaggedMatch && unsupportedMark(`${keyBase}f${partIdx}`)}
      </span>
    );
  });
}

/** Inline markdown the models actually emit: **bold**, *italic*, `code`. */
export function inlineMd(text, keyBase, cite) {
  const nodes = [];
  const emphasisPattern = /\*\*(.+?)\*\*|(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)|`([^`]+)`/g;
  let plainStart = 0;
  let emphasisMatch;
  while ((emphasisMatch = emphasisPattern.exec(text)) !== null) {
    if (emphasisMatch.index > plainStart) {
      nodes.push(cite(text.slice(plainStart, emphasisMatch.index), `${keyBase}t${plainStart}`));
    }
    const key = `${keyBase}m${emphasisMatch.index}`;
    const [whole, boldText, italicText, codeText] = emphasisMatch;
    if (boldText !== undefined)        nodes.push(<strong key={key}>{cite(boldText, key)}</strong>);
    else if (italicText !== undefined) nodes.push(<em key={key}>{cite(italicText, key)}</em>);
    else                               nodes.push(<code key={key}>{codeText}</code>);
    plainStart = emphasisMatch.index + whole.length;
  }
  if (plainStart < text.length) nodes.push(cite(text.slice(plainStart), `${keyBase}t${plainStart}`));
  return nodes;
}

/** Excerpt numbers the reply cites, split by marker kind: `verified` from [n]
 * markers, `flagged` from unverified [n!] ones. Bare [!] carries no number. */
export function citedMarkerNumbers(text) {
  const verified = new Set();
  const flagged = new Set();
  for (const [marker] of (text || '').matchAll(SEGMENT_RE)) {
    const flaggedMatch = marker.match(FLAGGED_NUMBERS_RE);
    const citeMatch = flaggedMatch || marker.match(CITE_NUMBERS_RE);
    if (!citeMatch) continue;
    const bucket = flaggedMatch ? flagged : verified;
    for (const number of citeMatch[1].split(',')) bucket.add(parseInt(number, 10));
  }
  return { verified, flagged };
}

/** Serialize the thread back to the server's conversation shape: role/content
 * (+ sources on replies). Local-only fields (query, model) and error bubbles
 * — which the server never stored — are dropped. */
export function toConversation(thread) {
  return thread
    .filter((message) => !message.error)
    .map(({ role, content, sources }) =>
      (sources ? { role, content, sources } : { role, content }));
}
