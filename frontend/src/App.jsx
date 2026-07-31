import { useEffect, useRef, useState } from 'react';
import AuthPage from './components/AuthPage.jsx';
import Chat from './components/Chat.jsx';
import DocumentViewer from './components/DocumentViewer.jsx';
import EmbeddingSpace from './components/EmbeddingSpace.jsx';
import KnowledgeGraph from './components/KnowledgeGraph.jsx';
import ModelsPanel from './components/ModelsPanel.jsx';
import PipelineOverlay from './components/PipelineOverlay.jsx';
import { useCrawler, CRAWLERS } from './lib/theme.js';
import {
  getToken, getMe, clearToken, getCollections, createCollection,
  deleteCollection, getChats, createChat, deleteChat,
} from './api.js';

const GEM = (color, size = 12) => (
  <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
    <path d="M8 1.5 14.5 8 8 14.5 1.5 8Z" fill={color} opacity="0.9" />
    <path d="M8 1.5 11.2 8 8 14.5 4.8 8Z" fill="#fff" opacity="0.22" />
  </svg>
);

const ICONS = {
  chat: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" aria-hidden="true">
      <path d="M2.5 3.5h11v7h-6l-3 3v-3h-2z" />
    </svg>
  ),
  docs: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 1.5h6l2.5 2.5v10.5h-8.5z M10 1.5v2.5h2.5" />
      <path d="M6 8h4 M6 10.5h4" strokeLinecap="round" />
    </svg>
  ),
  space: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <circle cx="4" cy="5" r="2" /><circle cx="11.5" cy="3.5" r="1.6" />
      <circle cx="12.5" cy="10" r="2.2" /><circle cx="5" cy="12" r="1.6" />
    </svg>
  ),
  graph: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
      <circle cx="3.5" cy="8" r="2" /><circle cx="12.5" cy="3.5" r="2" /><circle cx="12.5" cy="12.5" r="2" />
      <path d="M5.3 7.2 10.7 4.3 M5.3 8.8 10.7 11.7" />
    </svg>
  ),
  models: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" aria-hidden="true">
      <path d="M2 5h12 M2 11h12" />
      <circle cx="6" cy="5" r="1.8" fill="var(--surface)" />
      <circle cx="10.5" cy="11" r="1.8" fill="var(--surface)" />
    </svg>
  ),
};

const TABS = [
  { id: 'chat',   label: 'Chat' },
  { id: 'docs',   label: 'Documents' },
  { id: 'space',  label: 'Embedding Space' },
  { id: 'graph',  label: 'Knowledge Graph' },
  { id: 'models', label: 'Models' },
];

function ComingSoon({ crawler }) {
  const crawlerInfo = CRAWLERS[crawler];
  return (
    <div className="viz-empty">
      {GEM(crawlerInfo.accent, 44)}
      <h2>{crawlerInfo.name}</h2>
      <p>The {crawlerInfo.tagline} crawler is on the roadmap — its pipeline hasn't landed yet.</p>
      <p>Switch back to Sapphire to explore the academic corpus.</p>
    </div>
  );
}

/** Sidebar chat history: each chat carries its collection's colored orb. */
function ChatList({ chats, selectedChatId, onSelect, onNew, onDelete, disabled }) {
  return (
    <div className={`chat-list ${disabled ? 'is-locked' : ''}`} aria-disabled={disabled}>
      <div className="control-label chat-list-head">
        Chats
        <button className="chat-new" onClick={onNew} disabled={disabled}
                title="New chat on the selected collection">+</button>
      </div>
      {chats.map((chat) => (
        <div
          key={chat.id}
          className={`chat-item ${chat.id === selectedChatId ? 'active' : ''}`}
          onClick={() => onSelect(chat)}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => { if (event.key === 'Enter') onSelect(chat); }}
        >
          <span
            className="orb"
            style={{ background: chat.collection?.color || '#5b6a84' }}
            title={chat.collection?.name}
          />
          <span className="chat-item-title" title={chat.title}>{chat.title}</span>
          <button
            className="chat-delete"
            title="Delete chat"
            onClick={(event) => { event.stopPropagation(); onDelete(chat); }}
          >
            ×
          </button>
        </div>
      ))}
      {chats.length === 0 && (
        <div className="chat-list-empty">No chats yet — pick a collection in Documents, then hit +.</div>
      )}
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState(null);            // null = not logged in
  const [authChecked, setAuthChecked] = useState(false);
  const [collections, setCollections] = useState([]);
  const [selectedCollectionId, setSelectedCollectionId] = useState(null);
  const [chats, setChats] = useState([]);
  const [selectedChatId, setSelectedChatId] = useState(null);
  const [tab, setTab] = useState('chat');
  // A pipeline run is one long request tied to the mounted Documents tab, so
  // while it's in flight we lock navigation (switching collection/chat/tab
  // would unmount it and orphan the request). DocumentViewer drives this, and
  // names the job — 'index' or 'graph' — so the scrim can say which is running.
  const [runningJob, setRunningJob] = useState(null);
  const pipelineBusy = runningJob !== null;
  const [controlsEl, setControlsEl] = useState(null);
  const [docTarget, setDocTarget] = useState(null);   // { docId, chunkId, quotes, citing, query, nonce }
  // Bumped when a pipeline run finishes so the embedding-space and
  // knowledge-graph tabs refetch without a remount.
  const [corpusVersion, setCorpusVersion] = useState(0);
  // Counter, not Date.now(): two clicks in the same millisecond must still
  // produce distinct nonces or the second deep-link is silently ignored.
  const citationClickCount = useRef(0);
  const { crawler, setCrawler } = useCrawler();
  const live = crawler === 'sapphire';

  // Boot: validate any stored token; without a valid session the app shows
  // the login page. Data loads once a user is confirmed.
  useEffect(() => {
    if (!getToken()) { setAuthChecked(true); return; }
    getMe()
      .then((response) => {
        setUser(response.user);
        if (response.user) loadOwnerData();
      })
      .catch(() => clearToken())
      .finally(() => setAuthChecked(true));
  }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  // Warn before a full page unload (reload / close) while a run is in flight —
  // leaving abandons it. In-app navigation is blocked separately, below.
  useEffect(() => {
    if (!pipelineBusy) return;
    const warn = (event) => { event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [pipelineBusy]);

  async function loadOwnerData() {
    try {
      const [{ collections: loadedCollections }, { chats: loadedChats }] =
        await Promise.all([getCollections(), getChats()]);
      setCollections(loadedCollections);
      setChats(loadedChats);
      if (loadedChats.length) selectChat(loadedChats[0]);
      else if (loadedCollections.length) setSelectedCollectionId(loadedCollections[0].id);
    } catch (err) {
      console.error('[app] could not load collections/chats:', err);
    }
  }

  const selectChat = (chat) => {
    setSelectedChatId(chat.id);
    setDocTarget(null);              // targets point into the previous corpus
    if (chat.collection) {
      setSelectedCollectionId(chat.collection.id);
      if (CRAWLERS[chat.collection.crawler]) setCrawler(chat.collection.crawler);
    }
  };

  const newChat = async () => {
    if (!selectedCollectionId) {
      window.alert('Select (or create) a collection in the Documents tab first — every chat runs on one.');
      setTab('docs');
      return;
    }
    const { chat } = await createChat(selectedCollectionId);
    setChats((previous) => [chat, ...previous]);
    selectChat(chat);
    setTab('chat');
  };

  const removeChat = async (chat) => {
    if (!window.confirm(`Delete "${chat.title}"?`)) return;
    await deleteChat(chat.id);
    const remaining = chats.filter((other) => other.id !== chat.id);
    setChats(remaining);
    if (chat.id === selectedChatId) {
      setSelectedChatId(null);
      if (remaining.length) selectChat(remaining[0]);
    }
  };

  const selectCollection = (collection) => {
    setSelectedCollectionId(collection.id);
    if (CRAWLERS[collection.crawler]) setCrawler(collection.crawler);
  };

  // New collection: name is required; then the user lands in the upload UI.
  const newCollection = async () => {
    const name = window.prompt('Collection name:');
    if (name === null) return;
    if (!name.trim()) { window.alert('A collection needs a name.'); return; }
    const { collection } = await createCollection(name.trim(), crawler);
    setCollections((previous) => [...previous, collection]);
    selectCollection(collection);
    setTab('docs');
  };

  const removeCollection = async (collection) => {
    const warning = `Delete collection "${collection.name}"?\n\nThis permanently removes ` +
      'its documents, embeddings, knowledge graph, and every chat (with history) that uses it.';
    if (!window.confirm(warning)) return;
    await deleteCollection(collection.id);
    setCollections((previous) => previous.filter((other) => other.id !== collection.id));
    // Its chats are gone server-side (cascade); drop them here too.
    const remainingChats = chats.filter((chat) => chat.collection?.id !== collection.id);
    setChats(remainingChats);
    if (selectedCollectionId === collection.id) setSelectedCollectionId(null);
    if (!remainingChats.some((chat) => chat.id === selectedChatId)) setSelectedChatId(null);
  };

  /** A chat's first message retitles it server-side; mirror that in the list. */
  const retitleChat = (chatId, title) => {
    setChats((previous) => previous.map((chat) =>
      (chat.id === chatId && chat.title === 'New chat' ? { ...chat, title } : chat)));
  };

  const logout = () => {
    clearToken();
    window.location.reload();
  };

  // A [n] citation (or source chip) was clicked in Chat: jump to the cited
  // chunk in the Documents tab. `citing` = the specific sentence that [n] sat
  // in, so the viewer highlights the part of the chunk THAT sentence refers to.
  const openCitation = (source, query, citing) => {
    setDocTarget({
      docId: source.docId,
      chunkId: source.chunkId,
      quotes: source.quotes?.length ? source.quotes : null,
      citing: citing || null,
      query: query || null,
      nonce: ++citationClickCount.current,
    });
    setTab('docs');
  };

  if (!authChecked) return null;
  if (!user) {
    return <AuthPage onAuth={(loggedInUser) => { setUser(loggedInUser); loadOwnerData(); }} />;
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1 className="wordmark">OpenCrawl</h1>
        </div>

        <nav className="tab-rail" aria-label="Views">
          {TABS.map((tabDef) => (
            <button
              key={tabDef.id}
              className={`tab-btn ${tab === tabDef.id ? 'active' : ''}`}
              onClick={() => setTab(tabDef.id)}
              disabled={pipelineBusy}
            >
              {ICONS[tabDef.id]}
              {tabDef.label}
            </button>
          ))}
        </nav>

        <hr className="sidebar-divider" />

        <ChatList
          chats={chats}
          selectedChatId={selectedChatId}
          onSelect={(chat) => { selectChat(chat); setTab('chat'); }}
          onNew={newChat}
          onDelete={removeChat}
          disabled={pipelineBusy}
        />

        <hr className="sidebar-divider" />

        {/* Active tab portals its contextual controls here. */}
        <div className="sidebar-controls" ref={setControlsEl} />

        <div className="sidebar-footer">
          <span className="footer-user" title={user.email}>{user.email}</span>
          <button className="footer-logout" onClick={logout} disabled={pipelineBusy}>log out</button>
        </div>
      </aside>

      <main className="main">
        {/* Crawler switcher — sets the crawler for NEW collections + the theme. */}
        <div className="crawler-switch" role="group" aria-label="Crawler">
          {Object.entries(CRAWLERS).map(([crawlerId, crawlerInfo]) => (
            <button
              key={crawlerId}
              className={`crawler-btn ${crawler === crawlerId ? 'active' : ''}`}
              onClick={() => setCrawler(crawlerId)}
              disabled={pipelineBusy}
              title={`${crawlerInfo.name} — ${crawlerInfo.tagline}${crawlerInfo.ready ? '' : ' (coming soon)'}`}
            >
              {GEM(crawlerInfo.accent)}
              <span>{crawlerInfo.name}</span>
            </button>
          ))}
        </div>

        {!live ? (
          <ComingSoon crawler={crawler} />
        ) : (
          <>
            {/* Tabs stay mounted so camera position / graph layout / chat
                history survive tab switches; hidden ones are display:none. */}
            <div className="viz-fill" style={{ display: tab === 'chat' ? 'block' : 'none' }}>
              {selectedChatId ? (
                <Chat
                  key={selectedChatId}
                  chatId={selectedChatId}
                  active={tab === 'chat'}
                  onCitation={openCitation}
                  onFirstMessage={retitleChat}
                />
              ) : (
                <div className="viz-empty">
                  <h2>No chat open</h2>
                  <p>Select a collection in the Documents tab, then hit + in Chats to start one.</p>
                </div>
              )}
            </div>
            <div className="viz-fill" style={{ display: tab === 'docs' ? 'block' : 'none' }}>
              {controlsEl && (
                <DocumentViewer
                  key={selectedCollectionId ?? 'none'}
                  collectionId={selectedCollectionId}
                  collections={collections}
                  onSelectCollection={selectCollection}
                  onCreateCollection={newCollection}
                  onDeleteCollection={removeCollection}
                  onPipelineDone={() => setCorpusVersion((version) => version + 1)}
                  onBusyChange={setRunningJob}
                  controlsEl={controlsEl}
                  active={tab === 'docs'}
                  target={docTarget}
                />
              )}
            </div>
            <div className="viz-fill" style={{ display: tab === 'space' ? 'block' : 'none' }}>
              {controlsEl && (
                <EmbeddingSpace
                  key={selectedCollectionId ?? 'none'}
                  collectionId={selectedCollectionId}
                  corpusVersion={corpusVersion}
                  controlsEl={controlsEl}
                  active={tab === 'space'}
                />
              )}
            </div>
            <div className="viz-fill" style={{ display: tab === 'graph' ? 'block' : 'none' }}>
              {controlsEl && (
                <KnowledgeGraph
                  key={selectedCollectionId ?? 'none'}
                  collectionId={selectedCollectionId}
                  corpusVersion={corpusVersion}
                  controlsEl={controlsEl}
                  active={tab === 'graph'}
                />
              )}
            </div>
            <div className="viz-fill" style={{ display: tab === 'models' ? 'block' : 'none' }}>
              <ModelsPanel active={tab === 'models'} />
            </div>
          </>
        )}
      </main>

      {pipelineBusy && (
        <PipelineOverlay
          job={runningJob}
          collectionName={collections.find((c) => c.id === selectedCollectionId)?.name}
        />
      )}
    </div>
  );
}
