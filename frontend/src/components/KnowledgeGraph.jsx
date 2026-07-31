import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { getGraph, getGraphHtml } from '../api.js';

/**
 * Knowledge graph tab — renders the standalone interactive page kg-gen
 * generates (kg_graph.py) inside a sandboxed iframe, with the entity/relation
 * counts from the JSON alongside it in the sidebar.
 *
 * The page is injected via srcDoc rather than loaded by URL: collections are
 * per-user, and an iframe src can't carry the JWT the route requires.
 */
export default function KnowledgeGraph({ collectionId, corpusVersion, controlsEl, active }) {
  const [graphHtml, setGraphHtml] = useState(null);
  const [graphSummary, setGraphSummary] = useState(null);
  const [error, setError] = useState(null);

  // Refetches when the collection changes and when a pipeline run finishes.
  useEffect(() => {
    if (!collectionId) return;
    setError(null);
    setGraphHtml(null);
    Promise.all([getGraphHtml(collectionId), getGraph(collectionId)])
      .then(([html, summary]) => { setGraphHtml(html); setGraphSummary(summary); })
      .catch((err) => setError(err.message));
  }, [collectionId, corpusVersion]);

  if (!collectionId) {
    return (
      <div className="viz-empty">
        <h2>No collection selected</h2>
        <p>Select a collection in the Documents tab to see its knowledge graph.</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="viz-empty">
        <h2>No knowledge graph yet</h2>
        <p>{error}</p>
        <p>Index the collection in the Documents tab, then hit “Build knowledge graph”.</p>
      </div>
    );
  }
  if (!graphHtml) return <div className="viz-empty"><p>Loading knowledge graph…</p></div>;

  const controls = graphSummary && (
    <div>
      <span className="control-label">Graph</span>
      <div className="legend">
        <div className="legend-row">
          <span className="legend-title">{graphSummary.entities.length} entities</span>
        </div>
        <div className="legend-row">
          <span className="legend-title">{graphSummary.relations.length} relations</span>
        </div>
        <div className="legend-row">
          <span className="legend-title">{graphSummary.edges.length} relation types</span>
        </div>
        <div className="legend-row">
          <span className="legend-title">{graphSummary.sourceDocIds.length} documents</span>
        </div>
        {/* How much of each document was read: full text for the top-ranked
            share of the corpus, title + abstract + conclusion for the rest.
            Absent on graphs built before the split existed. */}
        {graphSummary.fullTextDocIds && (
          <div className="legend-row">
            <span className="legend-title">
              {graphSummary.fullTextDocIds.length} full text ·{' '}
              {graphSummary.summaryDocIds?.length ?? 0} summarized
            </span>
          </div>
        )}
      </div>
      <p style={{ color: 'var(--ink-muted)', fontSize: 12, marginTop: 10 }}>
        {graphSummary.model}
      </p>
    </div>
  );

  return (
    <div className="viz-fill">
      <iframe
        title="Knowledge Graph"
        srcDoc={graphHtml}
        // No allow-same-origin: the page stays in an opaque origin and can't
        // reach the app's localStorage (where the JWT lives).
        sandbox="allow-scripts"
        style={{ width: '100%', height: '100%', border: 'none', borderRadius: 8 }}
      />
      {active && controlsEl && createPortal(controls, controlsEl)}
    </div>
  );
}
