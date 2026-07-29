// Thin fetch wrappers over the Express API. Every request (except login /
// register) carries the JWT from localStorage; a 401 clears the token and
// reloads so the app lands back on the login page.

const TOKEN_KEY = 'opencrawl_token';

export const getToken   = () => localStorage.getItem(TOKEN_KEY);
export const setToken   = (token) => localStorage.setItem(TOKEN_KEY, token);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

/** Authorization header — also used by pdf.js when fetching PDFs. */
export function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers || {}) },
  });
  if (response.status === 401 && getToken()) {
    clearToken();
    window.location.reload();   // expired token → login page
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

/** Same auth/401 handling as request(), for routes that return HTML. */
async function requestText(url) {
  const response = await fetch(url, { headers: authHeaders() });
  if (response.status === 401 && getToken()) {
    clearToken();
    window.location.reload();
  }
  const body = await response.text();
  if (!response.ok) {
    // Errors come back as JSON even though success is HTML.
    let message = `HTTP ${response.status}`;
    try { message = JSON.parse(body).error ?? message; } catch { /* not JSON */ }
    throw new Error(message);
  }
  return body;
}

const postJson = (url, payload) =>
  request(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

// ── Auth ─────────────────────────────────────────────────────────────────────

export const register = (email, password) => postJson('/api/auth/register', { email, password });
export const login    = (email, password) => postJson('/api/auth/login', { email, password });
export const getMe    = () => request('/api/auth/me');

// ── Collections ──────────────────────────────────────────────────────────────

export const getCollections   = () => request('/api/collections');
export const createCollection = (name, crawler) => postJson('/api/collections', { name, crawler });
export const deleteCollection = (collectionId) =>
  request(`/api/collections/${collectionId}`, { method: 'DELETE' });

// ── Chats (each bound to a collection) ───────────────────────────────────────

export const getChats   = () => request('/api/chats');
export const createChat = (collectionId) => postJson('/api/chats', { collectionId });
export const getChat    = (chatId) => request(`/api/chats/${chatId}`);
export const deleteChat = (chatId) => request(`/api/chats/${chatId}`, { method: 'DELETE' });
export const updateChat = (chatId, fields) =>
  request(`/api/chats/${chatId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(fields),
  });

// ── Documents (per collection) ───────────────────────────────────────────────

export const getDocuments   = (collectionId) => request(`/api/collections/${collectionId}/documents`);
export const deleteDocument = (collectionId, docId) =>
  request(`/api/collections/${collectionId}/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' });
export const documentPdfUrl = (collectionId, docId) =>
  `/api/collections/${collectionId}/documents/${encodeURIComponent(docId)}/pdf`;

export function uploadDocuments(collectionId, files) {
  const form = new FormData();
  for (const file of files) form.append('files', file);
  return request(`/api/collections/${collectionId}/documents`, { method: 'POST', body: form });
}

// ── Pipeline (per collection) ────────────────────────────────────────────────

// Indexing only: extract → embed → categorize → rank. The knowledge graph is
// a separate, much longer job with its own endpoint (and its own button).
export const runPipeline       = (collectionId, params = {}) =>
  postJson(`/api/collections/${collectionId}/pipeline/run`, params);
export const buildGraph        = (collectionId) =>
  postJson(`/api/collections/${collectionId}/pipeline/build-graph`, {});
export const getPipelineStatus = (collectionId) =>
  request(`/api/collections/${collectionId}/pipeline/status`);

// ── Corpus (per collection) ──────────────────────────────────────────────────

export const getEmbeddingMap = (collectionId) => request(`/api/collections/${collectionId}/corpus/embedding-map`);
export const getGraph        = (collectionId) => request(`/api/collections/${collectionId}/corpus/graph`);
export const getGraphHtml    = (collectionId) => requestText(`/api/collections/${collectionId}/corpus/graph/view`);
export const getChunk        = (collectionId, chunkId) =>
  request(`/api/collections/${collectionId}/corpus/chunks/${encodeURIComponent(chunkId)}`);

// ── Models (global) ──────────────────────────────────────────────────────────

export const getModels    = () => request('/api/corpus/models');
export const saveSettings = (updates) => postJson('/api/corpus/settings', updates);

// ── Chat (RAG) ───────────────────────────────────────────────────────────────

export const postChat = (chatId, payload) => postJson(`/api/chats/${chatId}/chat`, payload);
