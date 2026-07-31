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

function RoleCard({ role, description, value, options, onSaved }) {
  const [draft, setDraft] = useState(value || '');
  const [saveState, setSaveState] = useState('idle'); // idle | saving | ok | err
  const [saveMessage, setSaveMessage] = useState('');

  useEffect(() => { setDraft(value || ''); }, [value]);

  const dirty = draft.trim() !== (value || '');
  const selected = options.find((model) => model.id === draft.trim());
  // A configured model that is not in this role's list — an id left in .env by
  // hand, or an Ollama tag that is no longer installed. Kept as an option so the
  // select shows the truth rather than silently appearing to be something else.
  const unlisted = draft.trim() && !selected;

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
        <select
          value={draft}
          disabled={options.length === 0}
          onChange={(event) => { setDraft(event.target.value); setSaveState('idle'); }}
          aria-label={`${ROLE_LABELS[role] || role} model`}
        >
          {!draft.trim() && <option value="">— none —</option>}
          {unlisted && <option value={draft}>{draft} — not available for this role</option>}
          {options.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
              {model.contextLength ? ` · ${formatContext(model.contextLength)}` : ''}
            </option>
          ))}
        </select>
        <button className="btn" onClick={apply} disabled={!dirty || !draft.trim() || saveState === 'saving'}>
          {saveState === 'saving' ? 'Saving…' : 'Apply'}
        </button>
      </div>
      {saveState === 'ok' && !dirty && <p className="save-note ok">✓ {saveMessage}</p>}
      {saveState === 'err' && <p className="save-note err">{saveMessage}</p>}
      {options.length === 0 && (
        <p className="save-note" style={{ color: 'var(--ink-muted)' }}>
          No models available for this role — it runs on Ollama, which is unreachable.
        </p>
      )}
      {selected?.contextLength && (
        <p className="save-note" style={{ color: 'var(--ink-muted)' }}>
          {formatContext(selected.contextLength)} of context.
        </p>
      )}
      {unlisted && (
        <p className="save-note" style={{ color: 'var(--ink-muted)' }}>
          Configured in .env but not offered here — pick one of the listed models
          to change it.
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
          the next run. The reasoning and knowledge-graph roles choose from the
          three supported models — the reasoning role routes by model id across
          Ollama, Google and Azure — while the extraction roles run on Ollama only
          and list its installed tags.
        </p>
        {!modelsInfo.ollamaUp && (
          <div className="banner">
            Ollama is unreachable at {modelsInfo.ollamaUrl} — roles that run on it have
            no models to choose from until it is started.
          </div>
        )}

        {Object.entries(modelsInfo.descriptions).map(([role, description]) => (
          <RoleCard
            key={role}
            role={role}
            description={description}
            value={modelsInfo.roles[role]}
            // Per-role: the backend decides which models a role's caller can
            // actually reach, so the UI never offers an unreachable one.
            options={modelsInfo.roleOptions?.[role] || NO_CATALOG}
            onSaved={(roles) => setModelsInfo((prev) => ({ ...prev, roles }))}
          />
        ))}
      </div>
    </div>
  );
}
