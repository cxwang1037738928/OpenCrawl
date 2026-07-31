/**
 * PipelineOverlay — full-app scrim shown while a long collection job is in
 * flight, for either of the two: 'index' (extract → embed → categorize → rank)
 * or 'graph' (the kg-gen knowledge graph).
 *
 * The run is one long (minutes to hours) request with no per-stage streaming,
 * so this blocks interaction rather than showing a progress bar: navigating
 * away unmounts the Documents tab and orphans the request. The scrim
 * intercepts every click, so the user can't tab away mid-run.
 */

const JOBS = {
  index: {
    heading: 'Indexing documents…',
    what:    'extract → embed → categorize → rank',
    note:    'Extraction is the slow part — minutes for large or scanned PDFs. ' +
             'Keep this tab open; navigation is paused until it finishes.',
  },
  graph: {
    heading: 'Building knowledge graph…',
    what:    'one model call per packed batch of chunks',
    note:    'This is the longest job in the app — the top-ranked documents are ' +
             'graphed from full text and the rest from their abstract and ' +
             'conclusion. The graph is saved after every call, so progress ' +
             'survives an interruption. Keep this tab open; navigation is ' +
             'paused until it finishes.',
  },
};

export default function PipelineOverlay({ job, collectionName }) {
  const { heading, what, note } = JOBS[job] ?? JOBS.index;
  return (
    <div className="pipeline-overlay" role="alertdialog" aria-live="assertive"
         aria-label={heading}>
      <div className="pipeline-overlay-card">
        <div className="pipeline-spinner" aria-hidden="true" />
        <h2>{heading}</h2>
        <p>
          {collectionName ? <><strong>{collectionName}</strong>: </> : 'This collection: '}
          {what}.
        </p>
        <p className="pipeline-overlay-note">{note}</p>
      </div>
    </div>
  );
}
