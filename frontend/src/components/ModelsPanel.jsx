import { useEffect, useState } from 'react';
import { getModels, saveSettings } from '../api.js';

// Stable identity: RoleCard's effect depends on `catalog`, so a fresh []
// literal per render would re-run it on every render.
const NO_CATALOG = [];

const ROLE_LABELS = {
  METADATA_MODEL:         'Metadata extraction',
  EXTRACTION_MODEL:       'Content extraction',
  QUERY_CLASSIFIER_MODEL: 'Query classifier',
  KG_MODEL:               'Knowledge-graph builder',
  REASONING_MODEL:        'Reasoning',
};

/** Human-readable context window, e.g. 1048576 → "1024k tokens". */
const formatContext = (tokens) =>
  (tokens >= 1024 ? `${Math.round(tokens / 1024)}k` : `${tokens}`) + ' tokens';

function RoleCard({ role, description, value, installed, catalog, onSaved }) {
  const [draft, setDraft] = useState(value || '');
  const [saveState, setSaveState] = useState('idle'); // idle | saving | ok | err
  const [saveMessage, setSaveMessage] = useState('');

  // A role with a catalog gets a pick-list; "Custom…" drops back to free text
  // so an unlisted model is still selectable — it just works blind, since the
  // graph stage sizes its calls from the catalog's context_length.
  const [freeText, setFreeText] = useState(false);

  useEffect(() => {
    setDraft(value || '');
    setFreeText(catalog.length > 0 && !!value && !catalog.some((model) => model.id === value));
  }, [value, catalog]);

  const dirty = draft.trim() !== (value || '');
  const catalogEntry = catalog.find((model) => model.id === draft.trim());
  const usePicker = catalog.length > 0 && !freeText;

  const apply = async () => {
    setSaveState('saving');
    try {
      const saved = await saveSettings({ [role]: draft.trim() });
      onSaved(saved.roles);
      setSaveState('ok');
      setSaveMessage('Saved to .env');
    } catch (err) {
      setSaveState('err');
      setSaveMessage(err.message);
    }
  };

  return (
    <div className="model-card">
      <h3>{ROLE_LABELS[role] || role}</h3>
      <p className="desc">{description}</p>
      <div className="model-row">
        {usePicker ? (
          <select
            value={draft}
            onChange={(event) => {
              if (event.target.value === '__custom__') { setFreeText(true); return; }
              setDraft(event.target.value);
              setSaveState('idle');
            }}
            aria-label={`${ROLE_LABELS[role] || role} model`}
          >
            {!catalogEntry && <option value={draft}>{draft || '— none —'}</option>}
            {catalog.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
                {model.contextLength ? ` · ${formatContext(model.contextLength)}` : ''}
              </option>
            ))}
            <option value="__custom__">Custom…</option>
          </select>
        ) : (
          <input
            type="text"
            list="installed-models"
            value={draft}
            placeholder="model name, e.g. phi4"
            onChange={(event) => { setDraft(event.target.value); setSaveState('idle'); }}
            aria-label={`${ROLE_LABELS[role] || role} model`}
          />
        )}
        <button className="btn" onClick={apply} disabled={!dirty || !draft.trim() || saveState === 'saving'}>
          {saveState === 'saving' ? 'Saving…' : 'Apply'}
        </button>
      </div>
      {saveState === 'ok' && !dirty && <p className="save-note ok">✓ {saveMessage}</p>}
      {saveState === 'err' && <p className="save-note err">{saveMessage}</p>}
      {catalog.length > 0 && catalogEntry?.contextLength && (
        <p className="save-note" style={{ color: 'var(--ink-muted)' }}>
          {formatContext(catalogEntry.contextLength)} of context — the graph stage
          sizes each call from this.
        </p>
      )}
      {catalog.length > 0 && draft.trim() && !catalogEntry && (
        <p className="save-note" style={{ color: 'var(--ink-muted)' }}>
          Not in documents/model_metadata.json — the graph stage will assume a small
          context window. Add it there to use the model’s real one.
        </p>
      )}
      {installed.length > 0 && value && !installed.includes(value) && !catalogEntry?.id?.includes('/') && (
        <p className="save-note" style={{ color: 'var(--ink-muted)' }}>
          Current model isn’t in Ollama’s installed list.
        </p>
      )}
    </div>
  );
}

export default function ModelsPanel() {
  const [modelsInfo, setModelsInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModels().then(setModelsInfo).catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="viz-empty"><p>{error}</p></div>;
  if (!modelsInfo) return <div className="viz-empty"><p>Loading models…</p></div>;

  return (
    <div className="models-wrap">
      <div className="models-inner">
        <h2 className="page-title">Models</h2>
        <p className="page-sub">
          Which model each pipeline role uses. Changes persist to .env and apply to
          the next run. The knowledge-graph role picks from the models in
          documents/model_metadata.json — local Ollama tags and hosted providers
          alike — because that stage sizes its calls to the model’s context window.
        </p>

        {!modelsInfo.ollamaUp && (
          <div className="banner">
            Ollama is unreachable at {modelsInfo.ollamaUrl} — the installed-model list is
            unavailable, but you can still type a model name and apply it.
          </div>
        )}

        <datalist id="installed-models">
          {modelsInfo.installed.map((modelName) => <option key={modelName} value={modelName} />)}
        </datalist>

        {Object.entries(modelsInfo.descriptions).map(([role, description]) => (
          <RoleCard
            key={role}
            role={role}
            description={description}
            value={modelsInfo.roles[role]}
            installed={modelsInfo.installed}
            // Only the graph role has a catalog: it's the one stage that must
            // know a model's context window to size its calls.
            catalog={role === 'KG_MODEL' ? (modelsInfo.kgModels || NO_CATALOG) : NO_CATALOG}
            onSaved={(roles) => setModelsInfo((prev) => ({ ...prev, roles }))}
          />
        ))}
      </div>
    </div>
  );
}
