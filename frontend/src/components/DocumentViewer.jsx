/**
 * DocumentViewer.jsx — Documents tab: collections + their PDFs + citation
 * targets.
 *
 * The sidebar (via portal) lists the owner's collections (colored orb each;
 * + adds one, hover × deletes with a warning). Clicking a collection opens
 * its document list with upload / delete controls and the two run buttons —
 * "Index documents" (extract → embed → categorize → rank) and "Build knowledge
 * graph", kept separate because the graph is a far longer LLM run that nothing
 * else depends on. The main pane renders the selected PDF with pdf.js, pages
 * lazily rendered on scroll.
 *
 * Citation deep-links: App passes `target` = { docId, chunkId, quotes, citing,
 * query, nonce } when a [n] marker (or source chip) is clicked in Chat. The
 * viewer opens that document, locates the chunk's text in the PDF text layer
 * and lays a light yellow highlight over it, scrolled into view. Location
 * strategy: strip the chunk's embedded heading prefix (prefixLen), normalize
 * both sides, search the chunk's recorded page range first (±1), then the whole
 * document — chunks indexed before page provenance existed still resolve.
 *
 * Highlight priority, all scoped to the specific citation clicked (`citing` =
 * the sentence that [n] sat in, so a chunk cited by several sentences lights a
 * different span per citation): verbatim `quotes` contained in the citing
 * sentence → the chunk sentence scoring best against the citing sentence
 * (in-browser cosine + keyword bonus, near-ties included) → sentences above the
 * loose threshold against `query` (chip clicks, or a citation with no claim) →
 * the whole chunk.
 */

import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import * as pdfjsLib from 'pdfjs-dist';
import {
  getDocuments, getChunk, documentPdfUrl,
  uploadDocuments, deleteDocument, runPipeline, buildGraph, authHeaders,
  getDeploymentConfig,
} from '../api.js';
import {
  normWords, indexPage, itemsInRange, matchOnPage, pickSentences, embedFocus,
} from '../utils/DocumentViewer_utils.jsx';

pdfjsLib.GlobalWorkerOptions.workerSrc =
  new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString();

const SCALE = 1.4;
const HIGHLIGHT_COLOR = 'rgba(255, 235, 59, 0.42)';   // light yellow

// ---------------------------------------------------------------------------
// Single page: lazy canvas render + highlight overlay
// ---------------------------------------------------------------------------

function PdfPage({ pdf, pageNum, size, highlightIds, onPageEl }) {
  const holderRef = useRef(null);
  const [rendered, setRendered] = useState(null);   // { canvasUrl?, rects }

  useEffect(() => {
    const holderEl = holderRef.current;
    if (!holderEl) return;
    let cancelled = false;

    const observer = new IntersectionObserver(async ([entry]) => {
      if (!entry.isIntersecting) return;
      observer.disconnect();

      const page = await pdf.getPage(pageNum);
      const viewport = page.getViewport({ scale: SCALE });
      const canvas = document.createElement('canvas');
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      await page.render({ canvas, canvasContext: canvas.getContext('2d'), viewport }).promise;
      if (cancelled) return;

      let highlightRects = [];
      if (highlightIds?.length) {
        const textContent = await page.getTextContent();
        highlightRects = highlightIds
          .map((itemIdx) => textContent.items[itemIdx])
          .filter(Boolean)
          .map((textItem) => {
            const deviceTransform = pdfjsLib.Util.transform(viewport.transform, textItem.transform);
            const glyphHeight = Math.hypot(deviceTransform[2], deviceTransform[3]);
            return {
              left: deviceTransform[4],
              top: deviceTransform[5] - glyphHeight,
              width: textItem.width * viewport.scale,
              height: glyphHeight * 1.18,
            };
          });
      }
      if (!cancelled) setRendered({ canvasUrl: canvas.toDataURL('image/png'), rects: highlightRects });
    }, { rootMargin: '600px' });

    observer.observe(holderEl);
    return () => { cancelled = true; observer.disconnect(); };
  }, [pdf, pageNum, highlightIds]);

  return (
    <div
      className="pdf-page"
      ref={(holderEl) => { holderRef.current = holderEl; if (onPageEl) onPageEl(pageNum, holderEl); }}
      style={{ width: size.width, height: size.height }}
    >
      {rendered ? (
        <>
          <img src={rendered.canvasUrl} alt={`Page ${pageNum}`} width={size.width} height={size.height} />
          {rendered.rects.map((highlightRect, rectIdx) => (
            <div
              key={rectIdx}
              className="pdf-highlight"
              style={{ ...highlightRect, background: HIGHLIGHT_COLOR }}
            />
          ))}
        </>
      ) : (
        <span className="pdf-page-num">{pageNum}</span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Viewer
// ---------------------------------------------------------------------------

export default function DocumentViewer({
  collectionId, collections, onSelectCollection, onCreateCollection,
  onDeleteCollection, onPipelineDone, onBusyChange, controlsEl, active, target,
}) {
  const [docs, setDocs] = useState(null);
  const [error, setError] = useState(null);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [pdf, setPdf] = useState(null);
  const [pageSizes, setPageSizes] = useState([]);
  const [highlights, setHighlights] = useState({});     // pageNum -> itemIds
  const [status, setStatus] = useState(null);
  const [search, setSearch] = useState('');             // sidebar filter text
  const [jobStatus, setJobStatus] = useState(null);     // upload/pipeline progress line
  // Which long job is in flight: 'upload' | 'index' | 'graph' | null. Drives
  // the action buttons' disabled state; 'index' and 'graph' are also reported
  // upward, where they lock app navigation for the duration.
  const [busyJob, setBusyJob] = useState(null);
  // Declared with the other state, ABOVE the control markup: collectionControls
  // is a const evaluated during render and reads `demo`, so a later declaration
  // is a temporal-dead-zone throw that unmounts the whole app.
  const [demo, setDemo] = useState(false);
  const pageEls = useRef(new Map());
  const pendingScrollPage = useRef(null);               // pageNum to scroll to once its element mounts
  const fileInputRef = useRef(null);

  const refreshDocs = () =>
    getDocuments(collectionId)
      .then((response) => { setError(null); setDocs(response.documents); })
      .catch((err) => setError(err.message));

  useEffect(() => {
    if (collectionId) refreshDocs();
  }, [collectionId]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    let live = true;
    getDeploymentConfig().then((config) => { if (live) setDemo(Boolean(config.demo)); });
    return () => { live = false; };
  }, []);

  // Upload the picked PDFs into this collection, then refresh the list. The
  // corpus only sees them after the pipeline runs.
  async function onFilesPicked(event) {
    const files = [...event.target.files];
    event.target.value = '';                       // same files re-pickable later
    if (!files.length) return;
    setJobStatus(`uploading ${files.length} file(s)…`);
    setBusyJob('upload');
    try {
      const { results } = await uploadDocuments(collectionId, files);
      const failed = results.filter((result) => !result.ok);
      setJobStatus(failed.length
        ? `${failed.length} rejected — ${failed[0].filename}: ${failed[0].error}`
        : `${results.length} uploaded — index them next`);
      refreshDocs();
    } catch (err) {
      setJobStatus(`upload failed: ${err.message}`);
    } finally {
      setBusyJob(null);
    }
  }

  // Indexing pipeline (extract → embed → categorize → heuristic) for this
  // collection. Extraction is the slow stage — minutes for large PDFs. The
  // knowledge graph is NOT part of this: it's a much longer LLM run that
  // nothing else depends on, so it has its own button below.
  async function runIndexPipeline() {
    setJobStatus('indexing… (extraction can take minutes)');
    setBusyJob('index');
    onBusyChange?.('index');   // locks app navigation until the run settles
    try {
      const { stages } = await runPipeline(collectionId);
      const failedStage = Object.entries(stages).find(([, stage]) => !stage.ok);
      setJobStatus(failedStage
        ? `${failedStage[0]} failed: ${failedStage[1].error}`
        : 'indexing complete — build the knowledge graph next');
      refreshDocs();
      // Tell the app so the embedding-space / knowledge-graph tabs refetch.
      if (!failedStage) onPipelineDone?.();
    } catch (err) {
      setJobStatus(`indexing failed: ${err.message}`);
    } finally {
      setBusyJob(null);
      onBusyChange?.(null);
    }
  }

  // Knowledge graph, on its own. Needs the indexing run first (it reads the
  // embed stage's chunks); the backend answers 503 when those are missing.
  async function runGraphBuild() {
    setJobStatus('building knowledge graph… (an LLM call per batch of chunks)');
    setBusyJob('graph');
    onBusyChange?.('graph');
    try {
      const result = await buildGraph(collectionId);
      setJobStatus(
        `graph built — ${result.entities} entities, ${result.relations} relations ` +
        `over ${result.fullTextDocs} full-text + ${result.summaryDocs} summarized document(s)`);
      onPipelineDone?.();
    } catch (err) {
      setJobStatus(`graph build failed: ${err.message}`);
    } finally {
      setBusyJob(null);
      onBusyChange?.(null);
    }
  }

  async function removeDoc(doc) {
    if (!window.confirm(`Remove "${doc.filename}" from this collection?`)) return;
    try {
      await deleteDocument(collectionId, doc.docId);
      if (selectedDocId === doc.docId) { setSelectedDocId(null); setPdf(null); }
      refreshDocs();
    } catch (err) {
      setJobStatus(`delete failed: ${err.message}`);
    }
  }

  // Load the selected PDF and measure every page for stable lazy-scroll.
  useEffect(() => {
    if (!selectedDocId) return;
    let cancelled = false;
    setPdf(null);
    setPageSizes([]);
    (async () => {
      try {
        // pdf.js v6 only reads src.url — the positional-string form of
        // getDocument(url) from v4 silently resolves to "no url given".
        // wasmUrl: JBIG2/JPX scans decode in wasm; without it scanned pages
        // render blank white. Vendored by npm run fetch:model.
        const doc = await pdfjsLib.getDocument({
          url: documentPdfUrl(collectionId, selectedDocId),
          httpHeaders: authHeaders(),   // the PDF route requires the JWT
          wasmUrl: '/models/pdfjs/wasm/',
          standardFontDataUrl: '/models/pdfjs/standard_fonts/',
        }).promise;
        if (cancelled) return;
        const sizes = [];
        for (let pageNum = 1; pageNum <= doc.numPages; pageNum++) {
          const viewport = (await doc.getPage(pageNum)).getViewport({ scale: SCALE });
          sizes.push({ width: viewport.width, height: viewport.height });
        }
        if (cancelled) return;
        setPdf(doc);
        setPageSizes(sizes);
      } catch (err) {
        if (!cancelled) setStatus(`could not load PDF: ${err.message}`);
      }
    })();
    return () => { cancelled = true; };
  }, [selectedDocId]);

  // Citation deep-link: open the doc, locate the chunk, highlight + scroll.
  useEffect(() => {
    if (!target) return;
    setHighlights({});
    setSelectedDocId(target.docId);
    setStatus('locating cited passage…');
  }, [target?.nonce]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!target || !pdf || selectedDocId !== target.docId) return;
    let cancelled = false;
    (async () => {
      try {
        const chunk = await getChunk(collectionId, target.chunkId);
        // Strip the embedded "title — heading\n" prefix; legacy chunks lack
        // prefixLen, but the prefix convention always ends at the first \n.
        const text = chunk.text || '';
        const body = chunk.prefixLen != null
          ? text.slice(chunk.prefixLen)
          : text.slice(text.indexOf('\n') + 1);
        const bodyWords = normWords(body);

        // Recorded page range (±1) first, then everything else.
        const recordedPages = [];
        if (chunk.pages) {
          for (let pageNum = Math.max(1, chunk.pages[0] - 1);
               pageNum <= Math.min(pdf.numPages, chunk.pages[1] + 1); pageNum++) {
            recordedPages.push(pageNum);
          }
        }
        const candidatePages = [...recordedPages];
        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
          if (!candidatePages.includes(pageNum)) candidatePages.push(pageNum);
        }

        // One text-layer index per page per click, shared by every lookup below.
        const pageIndexes = new Map();
        const indexFor = async (pageNum) => {
          if (!pageIndexes.has(pageNum)) {
            const textContent = await (await pdf.getPage(pageNum)).getTextContent();
            pageIndexes.set(pageNum, indexPage(textContent));
          }
          return pageIndexes.get(pageNum);
        };

        // Anchor: the first candidate page where the chunk's body matches. A
        // chunk that crosses a page break only matches its HEAD here — its tail
        // sentences live on the next page, which locate() handles below.
        let anchorPage = null;
        let bodyRange = null;
        for (const pageNum of candidatePages) {
          const pageIndex = await indexFor(pageNum);
          if (cancelled) return;
          bodyRange = matchOnPage(pageIndex, bodyWords);
          if (bodyRange) { anchorPage = pageNum; break; }
        }

        if (anchorPage === null) {
          // Text not locatable (scanned page, heavy equations): land on the page.
          const fallbackPage = chunk.pages?.[0] ?? 1;
          setStatus('passage could not be pinpointed — showing its page');
          pendingScrollPage.current = fallbackPage;
          pageEls.current.get(fallbackPage)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
          return;
        }
        const anchorIndex = pageIndexes.get(anchorPage);

        // Locate a space-free needle: near the anchor range first (the same
        // words elsewhere shouldn't steal it), then anywhere on the anchor
        // page, then on the chunk's other recorded pages — a page-spanning
        // chunk keeps its tail sentences on the NEXT page, and pinning the
        // search to the anchor range used to highlight the wrong head text.
        const locate = async (needleJoined) => {
          if (needleJoined.length < 12) return null;
          const nearAt = anchorIndex.joined.indexOf(
            needleJoined, Math.max(0, bodyRange.start - 300));
          if (nearAt !== -1 && nearAt < bodyRange.end + 300) {
            return { pageNum: anchorPage, at: nearAt };
          }
          const anchorAt = anchorIndex.joined.indexOf(needleJoined);
          if (anchorAt !== -1) return { pageNum: anchorPage, at: anchorAt };
          for (const pageNum of recordedPages) {
            if (pageNum === anchorPage) continue;
            const pageIndex = await indexFor(pageNum);
            const at = pageIndex.joined.indexOf(needleJoined);
            if (at !== -1) return { pageNum, at };
          }
          return null;
        };

        // Highlight priority, all scoped to the SPECIFIC citation clicked:
        //   1. verbatim quotes from this citation's own sentence,
        //   2. the chunk sentence(s) scoring best against the citing sentence,
        //   3. against the query (chip clicks, or a citation with no claim),
        //   4. the whole matched chunk region.
        const highlightsByPage = {};   // pageNum -> text item ids
        const addHighlight = ({ pageNum, at }, needleLength) => {
          const itemIds = itemsInRange(pageIndexes.get(pageNum), at, at + needleLength);
          if (!highlightsByPage[pageNum]) highlightsByPage[pageNum] = [];
          highlightsByPage[pageNum].push(...itemIds);
        };

        // Only the quotes this citation's sentence actually contains — so
        // sentence 3's quote doesn't light up when you click sentence 1's [n].
        const citingJoined = target.citing ? normWords(target.citing).join('') : '';
        const relevantQuotes = (target.quotes || []).filter((quoteText) => {
          if (!citingJoined) return true;   // chip click: no sentence to scope by
          const quoteJoined = normWords(quoteText).join('');
          return quoteJoined.length >= 12 && citingJoined.includes(quoteJoined);
        });
        for (const quoteText of relevantQuotes) {
          const quoteJoined = normWords(quoteText).join('');
          const found = await locate(quoteJoined);
          if (cancelled) return;
          if (found) addHighlight(found, quoteJoined.length);
        }

        if (!Object.keys(highlightsByPage).length) {
          const citingFocus = await embedFocus(target.citing);
          if (cancelled) return;
          let focus = citingFocus || target.query;
          // A restored conversation stores only the query TEXT — embed it now.
          if (focus && !focus.embedding) focus = await embedFocus(focus.text);
          if (cancelled) return;
          if (focus?.embedding) {
            // precise only with a citing sentence — a question is too vague
            // to pin one sentence, so chip clicks keep the loose band.
            const pickedSentences = await pickSentences(body, focus, setStatus, !!citingFocus);
            if (cancelled) return;
            for (const sentence of pickedSentences) {
              const found = await locate(normWords(sentence).join(''));
              if (cancelled) return;
              if (found) addHighlight(found, normWords(sentence).join('').length);
            }
          }
        }

        // Nothing located → the whole matched body region on the anchor page.
        if (!Object.keys(highlightsByPage).length) {
          highlightsByPage[anchorPage] =
            itemsInRange(anchorIndex, bodyRange.start, bodyRange.end);
        }

        setHighlights(Object.fromEntries(Object.entries(highlightsByPage)
          .map(([pageNum, itemIds]) => [pageNum, [...new Set(itemIds)]])));
        setStatus(null);
        // Scroll to the first highlighted page in reading order.
        const scrollPage = Math.min(...Object.keys(highlightsByPage).map(Number));
        pendingScrollPage.current = scrollPage;
        const pageEl = pageEls.current.get(scrollPage);
        if (pageEl) {
          pageEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
          pendingScrollPage.current = null;
        }
      } catch (err) {
        if (!cancelled) setStatus(`citation lookup failed: ${err.message}`);
      }
    })();
    return () => { cancelled = true; };
  }, [target?.nonce, pdf]);   // eslint-disable-line react-hooks/exhaustive-deps

  const registerPage = (pageNum, pageEl) => {
    if (pageEl) pageEls.current.set(pageNum, pageEl);
    else pageEls.current.delete(pageNum);
    if (pageEl && pendingScrollPage.current === pageNum) {
      pendingScrollPage.current = null;
      pageEl.scrollIntoView({ block: 'start' });
    }
  };

  // Plain lexical filter: the search text must appear literally in the title,
  // an author, or the filename (case-insensitive). Filename matters: title and
  // authors only exist after the extract stage, and un-indexed docs are listed
  // by filename.
  const needle = search.trim().toLowerCase();
  const shownDocs = (docs || []).filter((doc) => {
    if (!needle) return true;
    const searchableText = [doc.title || '', doc.filename || '', ...(doc.authors || [])]
      .join(' ')
      .toLowerCase();
    return searchableText.includes(needle);
  });

  // Collections list: orb + name; + creates one (name prompt), hover × deletes
  // (with a warning). Clicking a collection opens its document list below.
  const collectionControls = (
    <div className="collection-list">
      <div className="control-label chat-list-head">
        Collections
        {!demo && (
          <button className="chat-new" onClick={onCreateCollection} title="New collection">+</button>
        )}
      </div>
      {(collections || []).map((collection) => (
        <div
          key={collection.id}
          className={`chat-item ${collection.id === collectionId ? 'active' : ''}`}
          onClick={() => onSelectCollection(collection)}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => { if (event.key === 'Enter') onSelectCollection(collection); }}
        >
          <span className="orb" style={{ background: collection.color }} />
          <span className="chat-item-title" title={collection.name}>{collection.name}</span>
          {!demo && (
            <button
              className="chat-delete"
              title="Delete collection (and everything in it)"
              onClick={(event) => { event.stopPropagation(); onDeleteCollection(collection); }}
            >
              ×
            </button>
          )}
        </div>
      ))}
      {(!collections || collections.length === 0) && (
        <div className="doc-list-error">
          {demo ? 'No collections in this demo.' : 'No collections yet — hit + to create one.'}
        </div>
      )}
    </div>
  );

  const jobBusy = busyJob !== null;

  // Demo deployments serve a fixed corpus: no upload, no pipeline, no deletion.
  // The routes 403 anyway (middleware/demo.js); hiding the controls keeps the UI
  // from advertising actions that cannot succeed.
  const documentControls = collectionId && (
    <div className="doc-list">
      <div className="control-label">Documents</div>
      {!demo && (
      <div className="doc-actions">
        <button className="btn btn-small" onClick={() => fileInputRef.current?.click()}>
          Upload PDFs
        </button>
        <button
          className="btn btn-small"
          onClick={runIndexPipeline}
          disabled={jobBusy || !docs?.length}
          title="Extract, embed, categorize and rank this collection's PDFs"
        >
          Index documents
        </button>
        <button
          className="btn btn-small"
          onClick={runGraphBuild}
          disabled={jobBusy || !docs?.length}
          title="Build the knowledge graph from the indexed chunks (long — one LLM call per batch)"
        >
          Build knowledge graph
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={onFilesPicked}
        />
      </div>
      )}
      {jobStatus && <div className="doc-job-status">{jobStatus}</div>}
      {error && <div className="doc-list-error">{error}</div>}
      <input
        className="doc-search"
        type="search"
        value={search}
        placeholder="Search title, author, or filename…"
        aria-label="Search documents by title, author, or filename"
        onChange={(event) => setSearch(event.target.value)}
      />
      {shownDocs.map((doc) => (
        <div
          key={doc.docId}
          className={`doc-item ${selectedDocId === doc.docId ? 'active' : ''}`}
          onClick={() => { setHighlights({}); setStatus(null); setSelectedDocId(doc.docId); }}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === 'Enter') { setHighlights({}); setStatus(null); setSelectedDocId(doc.docId); }
          }}
          title={doc.filename}
        >
          <span className="doc-item-title">
            {doc.title || doc.filename}
            {doc.status !== 'completed' && <span className="doc-item-pending"> · not indexed</span>}
          </span>
          {doc.authors?.length > 0 && (
            <span className="doc-item-authors">{doc.authors.slice(0, 3).join(', ')}</span>
          )}
          {!demo && (
            <button
              className="doc-delete"
              title="Remove from this chat"
              onClick={(event) => { event.stopPropagation(); removeDoc(doc); }}
            >
              ×
            </button>
          )}
        </div>
      ))}
      {docs?.length > 0 && shownDocs.length === 0 && (
        <div className="doc-list-error">No document matches “{search.trim()}”.</div>
      )}
      {docs && docs.length === 0 && (
        <div className="doc-list-error">
          {demo ? 'No documents in this collection.' : 'No documents yet — upload PDFs, then index them.'}
        </div>
      )}
    </div>
  );

  return (
    <div className="pdf-wrap">
      {active && controlsEl && createPortal(
        <>
          {collectionControls}
          {documentControls}
        </>,
        controlsEl,
      )}
      {status && <div className="pdf-status">{status}</div>}
      {!collectionId ? (
        <div className="viz-empty">
          <h2>Collections</h2>
          <p>Select a collection in the sidebar — or hit + to create one — then upload PDFs, index them, and build the knowledge graph.</p>
        </div>
      ) : !selectedDocId ? (
        <div className="viz-empty">
          <h2>Documents</h2>
          <p>Pick a document from the sidebar, or click a citation in Chat to jump straight to the cited passage.</p>
        </div>
      ) : !pdf ? (
        <div className="viz-empty"><p>loading PDF…</p></div>
      ) : (
        <div className="pdf-scroll">
          {pageSizes.map((size, pageIdx) => (
            <PdfPage
              key={`${selectedDocId}_${pageIdx + 1}`}
              pdf={pdf}
              pageNum={pageIdx + 1}
              size={size}
              highlightIds={highlights[pageIdx + 1] || null}
              onPageEl={registerPage}
            />
          ))}
        </div>
      )}
    </div>
  );
}
