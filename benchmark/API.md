# OpenCrawl HTTP API Specification

Complete reference for every HTTP endpoint the OpenCrawl backend exposes, written for a benchmark
harness driving the pipeline programmatically.

**30 JSON endpoints** across 8 route groups, plus 2 unauthenticated static mounts. Every field
name, error string, and status code below is taken verbatim from `backend/routes/*.js`.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Global conventions](#2-global-conventions)
3. [Endpoint reference](#3-endpoint-reference)
   - [3.1 `/api/auth`](#31-apiauth--3-endpoints-public)
   - [3.2 `/api/corpus`](#32-apicorpus--2-endpoints)
   - [3.3 `/api/collections`](#33-apicollections--3-endpoints)
   - [3.4 `…/documents`](#34-apicollectionscollectioniddocuments--4-endpoints)
   - [3.5 `…/pipeline`](#35-apicollectionscollectionidpipeline--8-endpoints)
   - [3.6 `…/corpus`](#36-apicollectionscollectionidcorpus--4-endpoints)
   - [3.7 `/api/chats`](#37-apichats--5-endpoints)
   - [3.8 `/api/chats/:chatId/chat`](#38-apichatschatidchat--1-endpoint)
   - [3.9 Static mounts](#39-static-mounts-unauthenticated)
4. [End-to-end benchmark walkthrough](#4-end-to-end-benchmark-walkthrough)
5. [Environment reference](#5-environment-reference)
6. [Known gaps and benchmark gotchas](#6-known-gaps-and-benchmark-gotchas)

---

## 1. Quick start

### Base URL

| Mode | URL | Notes |
| --- | --- | --- |
| Backend direct | `http://localhost:3000` | `PORT` env var, default `3000` (`backend/server.js:27`) |
| Via Vite dev server | `http://localhost:5173` | Proxies `/api` and `/models` → `:3000` (`frontend/vite.config.js:26-33`) |

Benchmark against `http://localhost:3000` directly — the Vite proxy adds nothing but a hop.

### Bring-up

```bash
npm run db:up        # docker compose up -d postgres  (Postgres 16 on host port 5433)
npm run db:migrate   # prisma migrate dev
npm run db:seed      # node prisma/seed.js  — creates the accounts below
npm start            # node backend/server.js  → http://localhost:3000
```

Use `npm start` rather than `npm run dev` for benchmarking. `npm run dev` runs
`node --watch-path=backend`, which restarts the server when anything under `backend/` changes and
will kill an in-flight pipeline run.

### Demo credentials

Seeded by `prisma/seed.js:12-15`. The seed is idempotent and **resets the password on every run**,
so `npm run db:seed` always restores these exact values.

| Account | Email | Password | `isAdmin` |
| --- | --- | --- | --- |
| **Demo** | `demo@gmail.com` | `demo123` | `false` |
| Admin | `admin@gmail.com` | `admin123` | `true` |

**Benchmark as `demo@gmail.com`.** There is no admin-only route and no `requireAdmin` middleware
anywhere in the codebase. The `isAdmin` flag changes exactly one behavior: when the user is an
admin, `POST /api/chats/:chatId/chat` also appends a prompt → retrieved-context → response trio to
`chat_log.txt` at the repo root (`backend/routes/chat.js:92`). That is a fire-and-forget disk
write; the HTTP response is byte-for-byte identical either way.

Passwords are bcrypt (cost 10). Registering your own account via `POST /api/auth/register` works
equally well and avoids depending on seed state — note the 6-character minimum.

### Authenticate

```bash
TOKEN=$(curl -s -X POST http://localhost:3000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"demo@gmail.com","password":"demo123"}' | jq -r .token)

curl -s http://localhost:3000/api/collections -H "Authorization: Bearer $TOKEN"
```

Python equivalent:

```python
import requests
BASE = "http://localhost:3000"
r = requests.post(f"{BASE}/api/auth/login",
                  json={"email": "demo@gmail.com", "password": "demo123"})
r.raise_for_status()
token = r.json()["token"]
H = {"Authorization": f"Bearer {token}"}

print(requests.get(f"{BASE}/api/collections", headers=H).json())
```

---

## 2. Global conventions

### Authentication

Stateless JWT in a header. No cookies, no sessions, no server-side revocation.

```
Authorization: Bearer <token>
```

| Property | Value | Source |
| --- | --- | --- |
| Algorithm | HS256 (jsonwebtoken default) | `backend/middleware/auth.js:17-21` |
| Payload | `{ sub: <user id>, email, isAdmin }` | `auth.js:18` |
| Secret | `JWT_SECRET`, default `opencrawl-local-dev-secret` | `auth.js:13` |
| TTL | `JWT_TTL`, default `7d` | `auth.js:14` |
| Header prefix | Literal `Bearer ` — anything else is treated as absent | `auth.js:27` |

A missing, malformed, expired, or invalid token on any guarded route returns:

```
401  {"error":"Not logged in"}
```

`requireAuth` is applied at mount time in `backend/server.js:40-42` to `/api/corpus`,
`/api/collections`, and `/api/chats`. `/api/auth` is public.

Logout is client-side only — a token stays valid until it expires.

### Error envelope

Every error response is a flat JSON object with a single `error` string
(`backend/server.js:70-74`):

```json
{ "error": "human-readable message" }
```

The status is `err.status` when a route set one, otherwise `500`. Unknown routes return:

```
404  {"error":"No route GET /api/nope"}
```

The one exception to the JSON-everywhere rule: `GET …/corpus/graph/view` returns `text/html` on
success (its 404 is still JSON).

### Ownership

Resources are scoped to the authenticated user. Collections are owned via `Collection.userId`;
documents, chunks, and chats inherit ownership through their collection.

Accessing another user's resource returns **404, not 403** — the API does not distinguish
"missing" from "not yours" (`collections.js:93`, `chats.js:81`).

Path ids are parsed with `parseInt`. A non-integer id fails before the lookup:

| Route family | Bad-id error |
| --- | --- |
| `/api/collections/:collectionId/*` | `400 {"error":"collectionId must be an integer"}` |
| `/api/chats/:chatId/*` | `400 {"error":"chatId must be an integer"}` |

Because `parseInt` is lenient, `/api/collections/12abc` resolves to collection `12`. Only inputs
with no leading digits (e.g. `abc`) produce `NaN` and the 400.

### Request bodies

`express.json()` is applied globally (`server.js:35`) with its **default 100 kb limit**. This
matters for exactly two payloads:

- `POST /api/chats/:chatId/chat` — the 384-float `queryEmbedding` serializes to roughly 8 kb. Fine.
- `PATCH /api/chats/:chatId` — a long `conversation` array with embedded `sources` can approach the
  limit. Exceeding it produces a `PayloadTooLargeError` surfaced as `413` by the global handler.

All bodies are `application/json` except `POST …/documents`, which is `multipart/form-data`.

### CORS

`app.use(cors())` with no options (`server.js:34`) — `Access-Control-Allow-Origin: *`, no
credentials, no origin allowlist. A browser-based harness can call the API from any origin.

### No rate limiting

There is no `express-rate-limit`, `helmet`, or any throttle anywhere in the backend. A harness can
hammer `/api/auth/login` or fire concurrent pipeline runs without being throttled — including
concurrent `POST …/pipeline/build-graph` calls on the same collection, which will interleave their
writes to `Collection.knowledgeGraph`. Serialize pipeline calls per collection yourself.

### No streaming — set long client timeouts

There is no SSE, no WebSocket, and no job-polling endpoint. Long stages are **blocking synchronous
POSTs** that hold the HTTP connection open for their entire duration:

| Endpoint | Typical duration |
| --- | --- |
| `POST …/pipeline/extract` | seconds to minutes per document |
| `POST …/pipeline/run` | minutes |
| `POST …/pipeline/build-graph` | **minutes to hours** — one LLM call per packed chunk batch |
| `GET …/corpus/embedding-map` | seconds on first call (UMAP), free thereafter |

Configure your HTTP client accordingly (`requests` has no default timeout, which is what you want
here; `httpx` defaults to 5 s and **will** cut off a graph build).

Progress for the graph stage is observable only indirectly: `kg_graph.py` flushes a partial graph
after every LLM call, so polling `GET …/corpus/graph` from a second connection shows
`callsCompleted` climbing while `complete` is still `false`.

---

## 3. Endpoint reference

Per-endpoint format: source location, auth, request, success response, every error, side effects.
The `401 Not logged in` and ownership errors from [§2](#2-global-conventions) apply to every
guarded endpoint and are not repeated.

### Summary

| # | Method | Path | Auth |
| --- | --- | --- | --- |
| 1 | POST | `/api/auth/register` | — |
| 2 | POST | `/api/auth/login` | — |
| 3 | GET | `/api/auth/me` | optional |
| 4 | GET | `/api/corpus/models` | required |
| 5 | POST | `/api/corpus/settings` | required |
| 6 | GET | `/api/collections` | required |
| 7 | POST | `/api/collections` | required |
| 8 | DELETE | `/api/collections/:collectionId` | required + owner |
| 9 | GET | `/api/collections/:collectionId/documents` | required + owner |
| 10 | POST | `/api/collections/:collectionId/documents` | required + owner |
| 11 | GET | `/api/collections/:collectionId/documents/:docId/pdf` | required + owner |
| 12 | DELETE | `/api/collections/:collectionId/documents/:docId` | required + owner |
| 13 | GET | `/api/collections/:collectionId/pipeline/status` | required + owner |
| 14 | POST | `/api/collections/:collectionId/pipeline/enhance` | required + owner |
| 15 | POST | `/api/collections/:collectionId/pipeline/extract` | required + owner |
| 16 | POST | `/api/collections/:collectionId/pipeline/embed` | required + owner |
| 17 | POST | `/api/collections/:collectionId/pipeline/categorize` | required + owner |
| 18 | POST | `/api/collections/:collectionId/pipeline/heuristic` | required + owner |
| 19 | POST | `/api/collections/:collectionId/pipeline/build-graph` | required + owner |
| 20 | POST | `/api/collections/:collectionId/pipeline/run` | required + owner |
| 21 | GET | `/api/collections/:collectionId/corpus/embedding-map` | required + owner |
| 22 | GET | `/api/collections/:collectionId/corpus/graph` | required + owner |
| 23 | GET | `/api/collections/:collectionId/corpus/graph/view` | required + owner |
| 24 | GET | `/api/collections/:collectionId/corpus/chunks/:chunkId` | required + owner |
| 25 | GET | `/api/chats` | required |
| 26 | POST | `/api/chats` | required |
| 27 | GET | `/api/chats/:chatId` | required + owner |
| 28 | PATCH | `/api/chats/:chatId` | required + owner |
| 29 | DELETE | `/api/chats/:chatId` | required + owner |
| 30 | POST | `/api/chats/:chatId/chat` | required + owner |

---

### 3.1 `/api/auth` — 3 endpoints (public)

Source: `backend/routes/auth.js`. Mounted without `requireAuth` (`server.js:39`).

Shared credential validation (`auth.js:22-32`) applies to both `register` and `login`:

- `email` — must be a string matching `/^[^\s@]+@[^\s@]+\.[^\s@]+$/`; trimmed and lowercased before
  use.
- `password` — must be a string of **length ≥ 6**.

Both checks run *before* any DB lookup, so a login attempt with a 5-character password returns
`400`, not `401`.

The `user` object returned everywhere in this group is (`auth.js:19`):

```json
{ "id": 1, "email": "demo@gmail.com", "isAdmin": false }
```

---

#### 1. `POST /api/auth/register`

`auth.js:36` · Auth: none

**Request**

| Field | Type | Required | Validation |
| --- | --- | --- | --- |
| `email` | string | yes | email regex; trimmed + lowercased |
| `password` | string | yes | length ≥ 6 |

**Success — `201`**

```json
{ "token": "eyJhbGciOiJIUzI1NiIs...", "user": { "id": 3, "email": "a@b.com", "isAdmin": false } }
```

**Errors**

| Status | Message |
| --- | --- |
| 400 | `A valid email is required` |
| 400 | `Password must be at least 6 characters` |
| 409 | `That email is already registered` |

**Side effects** — creates a `User` row with `bcrypt.hash(password, 10)`. New users are always
`isAdmin: false`; there is no API path to create an admin.

---

#### 2. `POST /api/auth/login`

`auth.js:46` · Auth: none

**Request** — identical to register.

**Success — `200`**

```json
{ "token": "eyJhbGciOiJIUzI1NiIs...", "user": { "id": 2, "email": "demo@gmail.com", "isAdmin": false } }
```

**Errors**

| Status | Message |
| --- | --- |
| 400 | `A valid email is required` |
| 400 | `Password must be at least 6 characters` |
| 401 | `Invalid email or password` |

The 401 is shared by "no such user" and "wrong password" — no user enumeration.

---

#### 3. `GET /api/auth/me`

`auth.js:57` · Auth: **optional (soft)**

Uses `userFromRequest`, not `requireAuth`. It **never returns 401**. A missing header, an expired
token, a garbage token, and a valid token for a since-deleted user all produce the same body:

**Success — `200`**

```json
{ "user": { "id": 2, "email": "demo@gmail.com", "isAdmin": false } }
```

or

```json
{ "user": null }
```

Do not use this endpoint to assert that auth rejection works — assert against any guarded endpoint
instead. It is a useful liveness probe: it needs a DB connection but no credentials.

---

### 3.2 `/api/corpus` — 2 endpoints

Source: `backend/routes/corpus.js` (`modelsRouter`). Global, not collection-scoped.
Auth: required for both.

The five configurable model roles (`MODEL_ROLES`, `corpus.js:44-50`):

| Key | Purpose |
| --- | --- |
| `METADATA_MODEL` | Title/author/abstract fallback when GROBID is down (`extract.py`) |
| `EXTRACTION_MODEL` | Other extraction tasks (`extract.py`) |
| `QUERY_CLASSIFIER_MODEL` | Query classification (`parse_user_query.js`) |
| `KG_MODEL` | Knowledge-graph construction (kg-gen, `kg_graph.py`) |
| `REASONING_MODEL` | Answer synthesis over retrieved chunks (chat) |

---

#### 4. `GET /api/corpus/models`

`corpus.js:296` · Auth: required · No params

**Success — `200`**

```json
{
  "ollamaUp": true,
  "ollamaUrl": "http://localhost:11434",
  "installed": ["ministral-3:3b-instruct-2512-q4_K_M"],
  "kgModels": [{ "id": "...", "name": "...", "contextLength": 8192 }],
  "roles": {
    "METADATA_MODEL": "ministral-3:3b-instruct-2512-q4_K_M",
    "EXTRACTION_MODEL": "ministral-3:3b-instruct-2512-q4_K_M",
    "QUERY_CLASSIFIER_MODEL": "ministral-3:3b-instruct-2512-q4_K_M",
    "KG_MODEL": "ministral-3:3b-instruct-2512-q4_K_M",
    "REASONING_MODEL": "gemini/gemini-3.1-flash-lite"
  },
  "roleOptions": {
    "METADATA_MODEL": [{ "id": "...", "name": "...", "contextLength": null }],
    "EXTRACTION_MODEL": [], "QUERY_CLASSIFIER_MODEL": [],
    "KG_MODEL": [], "REASONING_MODEL": []
  },
  "descriptions": { "METADATA_MODEL": "Title/author/abstract fallback ..." }
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `ollamaUp` | boolean | `false` when Ollama is unreachable or exceeds the 3 s timeout |
| `ollamaUrl` | string | `OLLAMA_URL`, default `http://localhost:11434` |
| `installed` | string[] | Ollama tag names, sorted. `[]` when Ollama is down |
| `kgModels` | object[] | From `documents/model_metadata.json`. `[]` if missing/malformed |
| `roles` | object | Current value of each role from `process.env`; each is `string \| null` |
| `roleOptions` | object | Pick-list per role. Ollama-sourced roles list installed tags; `KG_MODEL` and `REASONING_MODEL` use a curated chat list |
| `descriptions` | object | The `MODEL_ROLES` table above |

**Errors** — none beyond auth. Both external reads degrade silently: Ollama unreachable →
`ollamaUp: false`, `installed: []`; catalog missing → `kgModels: []`.

**Side effects** — outbound `GET {OLLAMA_URL}/api/tags` with a 3 s `AbortController`.

This is the cheapest way for a benchmark to assert which models are configured before deciding
whether a graph or chat stage is expected to succeed.

---

#### 5. `POST /api/corpus/settings`

`corpus.js:332` · Auth: required (**not** admin-gated)

**Request** — a flat object whose keys must all be `MODEL_ROLES` keys. Partial updates allowed.

```json
{ "REASONING_MODEL": "gemini/gemini-3.1-flash-lite", "KG_MODEL": "ministral-3:3b-instruct-2512-q4_K_M" }
```

Each value must be a non-empty string containing no `\r` or `\n`. Values are `.trim()`ed.

**Success — `200`**

```json
{ "ok": true, "roles": { "METADATA_MODEL": "...", "EXTRACTION_MODEL": "...",
  "QUERY_CLASSIFIER_MODEL": "...", "KG_MODEL": "...", "REASONING_MODEL": "..." } }
```

`roles` always contains all five keys, not just the ones you sent.

**Errors**

| Status | Message |
| --- | --- |
| 400 | `Empty settings payload` |
| 400 | `Unknown setting "<key>"` |
| 400 | `Invalid value for <KEY>` |
| 500 | `.env not found at project root` |

**Side effects — destructive, read before using.** This endpoint **rewrites the repo-root `.env`
file in place**: matching `KEY=` lines are replaced (comments and ordering preserved), missing keys
are appended under a `# Models set via the frontend Models tab` comment. Written via temp file +
`fs.rename`, so it cannot truncate on crash. It also mutates `process.env` so subsequent pipeline
spawns inherit the new values without a restart.

A benchmark that switches models via this endpoint is **permanently modifying the developer's
`.env`**. Snapshot the file first and restore it afterward, or set model roles by editing `.env`
directly before server start.

---

### 3.3 `/api/collections` — 3 endpoints

Source: `backend/routes/collections.js`. Auth: required.

Collection summary shape (`collections.js:45-52`), used by both list and create:

```json
{ "id": 1, "name": "My corpus", "color": "#199e70", "crawler": "sapphire",
  "createdAt": "2026-08-06T12:00:00.000Z", "documents": 12 }
```

`documents` is a count. It is present on list responses and **absent** on create (the field is
`undefined`, so JSON serialization drops it).

---

#### 6. `GET /api/collections`

`collections.js:56` · Auth: required

**Success — `200`**

```json
{ "collections": [ { "id": 1, "name": "...", "color": "#199e70", "crawler": "sapphire",
                     "createdAt": "...", "documents": 12 } ] }
```

Scoped to `req.user.id`, ordered `createdAt` ascending. Empty array for a new user.

---

#### 7. `POST /api/collections`

`collections.js:65` · Auth: required

**Request**

| Field | Type | Required | Default | Validation |
| --- | --- | --- | --- | --- |
| `name` | string | yes | — | non-empty after `.trim()` |
| `crawler` | string | no | `"sapphire"` | one of `sapphire`, `ruby`, `topaz` |

**Success — `201`**

```json
{ "collection": { "id": 7, "name": "My corpus", "color": "#3987e5",
                  "crawler": "sapphire", "createdAt": "..." } }
```

**Errors**

| Status | Message |
| --- | --- |
| 400 | `"name" is required` |
| 400 | `"crawler" must be one of: sapphire, ruby, topaz` |

**Side effects** — `color` is auto-assigned round-robin from an 8-entry palette indexed by the
user's existing collection count (`ORB_COLORS`, `collections.js:34`), so it is deterministic:
the Nth collection a user creates gets `ORB_COLORS[N % 8]`.

Only `sapphire` is actually implemented. `ruby` and `topaz` are accepted values whose pipeline
stages are stubs — creating a collection with them succeeds, then extraction behaves as sapphire.

---

#### 8. `DELETE /api/collections/:collectionId`

`collections.js:100` · Auth: required + ownership

**Success — `200`**

```json
{ "ok": true, "id": 7 }
```

**Errors**

| Status | Message |
| --- | --- |
| 400 | `collectionId must be an integer` |
| 404 | `No collection <id>` |

**Side effects — irreversible cascade.** Deletes the `Collection` row, which cascades to all chats
(and their conversation history), documents, and chunks; then removes `uploads/<collectionId>/`
and the scratch directory `data/collections/<collectionId>/`. Both filesystem removals are
best-effort and never fail the request.

This is the correct way to reset state between benchmark runs.

> There is **no `GET /api/collections/:collectionId`**. `DELETE` is the only method registered on
> that exact path. A `GET` still runs the ownership middleware first, so the response depends on
> the id:
>
> - unknown or not yours → `404 {"error":"No collection <id>"}`
> - valid and yours → falls through to the global handler:
>   `404 {"error":"No route GET /api/collections/<id>"}`
>
> Both are JSON — the SPA fallback regex excludes anything under `/api/`, so an API path never
> returns HTML. Read a single collection's metadata from the list endpoint instead.

---

### 3.4 `/api/collections/:collectionId/documents` — 4 endpoints

Source: `backend/routes/documents.js`. Auth: required + collection ownership on all four.

`docId` is the **first 16 hex characters of the file's SHA-256** (`documents.js:58`), so the same
PDF has the same `docId` in every collection — useful for cross-collection benchmark assertions.

---

#### 9. `GET /api/collections/:collectionId/documents`

`documents.js:63` · Auth: required + owner

**Success — `200`**

```json
{ "documents": [ {
  "docId": "a3f1c0d92e4b7a15",
  "filename": "paper.pdf",
  "title": "Some Paper Title",
  "authors": ["Doe, J."],
  "status": "completed",
  "pageCount": 14,
  "extractedAt": "2026-08-06T12:34:56.000Z",
  "createdAt": "2026-08-06T12:00:00.000Z"
} ] }
```

Ordered `createdAt` ascending. `status` is one of `pending`, `processing`, `completed`, `failed`
(default `pending` until extraction runs). `title`, `pageCount`, `extractedAt` are `null` until
extraction populates them; `authors` is a JSON array defaulting to `[]`.

---

#### 10. `POST /api/collections/:collectionId/documents`

`documents.js:75` · Auth: required + owner · **`Content-Type: multipart/form-data`**

**Request**

| Property | Value |
| --- | --- |
| Field name | **`files`** (repeat for multiple) — `upload.array('files')` |
| Max files per request | 20 |
| Max bytes per file | `MAX_PDF_SIZE_MB` × 1 MiB, default **50 MB** |
| Storage | multer memory storage — the whole batch is buffered in RAM |

Per-file validation, in order (`inspectPdf`, `documents.js:44-59`):

1. filename ends with `.pdf` (case-insensitive)
2. first 4 bytes are `%PDF`
3. parsable by `pdf-parse` (yields `numpages`)
4. no existing document in *this* collection with the same full SHA-256

**Success — `201` when at least one file succeeded, `400` when none did.** The body is identical
in both cases:

```json
{ "uploaded": 2, "results": [
  { "filename": "good.pdf", "ok": true,  "docId": "a3f1c0d92e4b7a15" },
  { "filename": "bad.txt",  "ok": false, "error": "Not a .pdf file" }
] }
```

`uploaded` counts only `ok: true` entries. **Partial success is reported as `201` with failures
inside `results`** — a benchmark must inspect every `results[].ok`, not just the status code.

Per-file `error` strings:

| Message | Cause |
| --- | --- |
| `Not a .pdf file` | extension check |
| `Not a valid PDF (bad magic bytes)` | first 4 bytes ≠ `%PDF` |
| `PDF appears corrupted — could not parse it` | `pdf-parse` threw |
| `Already in this collection as "<filename>"` | duplicate SHA-256 in this collection |

**Request-level errors**

| Status | Message | Cause |
| --- | --- | --- |
| 400 | `No files uploaded (multipart field "files")` | zero files, or wrong field name |
| 400 | *(body as above, `uploaded: 0`)* | every file failed validation |
| 500 | `File too large` | multer size limit exceeded |
| 500 | `Too many files` | more than 20 files |

The two 500s are **not** a documentation error: multer throws a `MulterError` with no `.status`
property, so the global error handler defaults it to 500 rather than 413. Assert on 500 + the
message, not on 413.

**Side effects** — creates `uploads/<collectionId>/`, writes each accepted file as
`<docId>_<basename>`, inserts a `Document` row with `status: "pending"`.

Example:

```python
files = [("files", ("a.pdf", open("a.pdf", "rb"), "application/pdf")),
         ("files", ("b.pdf", open("b.pdf", "rb"), "application/pdf"))]
r = requests.post(f"{BASE}/api/collections/{cid}/documents", headers=H, files=files)
assert r.status_code == 201
assert all(x["ok"] for x in r.json()["results"]), r.json()
```

---

#### 11. `GET /api/collections/:collectionId/documents/:docId/pdf`

`documents.js:119` · Auth: required + owner

**Path params** — `docId` string (the 16-char hash prefix).

**Success — `200`**, `Content-Type: application/pdf`, raw file bytes via `res.sendFile`.

**Errors**

| Status | Message |
| --- | --- |
| 404 | `Unknown document "<docId>"` |
| 404 | `Source PDF missing on disk: <filename>` |

The second case means the DB row survived but the file did not — worth asserting separately.

Note this route requires the `Authorization` header like any other, so it cannot be used as a bare
`<iframe src>`. Fetch it programmatically.

---

#### 12. `DELETE /api/collections/:collectionId/documents/:docId`

`documents.js:133` · Auth: required + owner

**Success — `200`**

```json
{ "ok": true, "docId": "a3f1c0d92e4b7a15" }
```

**Errors** — `404 Unknown document "<docId>"`.

**Side effects** — deletes the `Document` row (chunks cascade) and unlinks the file. The file
removal is best-effort.

Deleting a document does **not** rebuild the embeddings, categories, or graph — those keep
referencing the removed `docId` until the corresponding stage re-runs.

---

### 3.5 `/api/collections/:collectionId/pipeline` — 8 endpoints

Source: `backend/routes/pipeline.js`. Auth: required + collection ownership on all eight.

#### How stages execute

Every stage follows **export → run → ingest**: inputs are exported from Postgres into the
collection's scratch directory `data/collections/<collectionId>/`, the stage runs against those
files, and outputs are ingested back into Postgres (`backend/pipeline/collectionStore.js`).

Python stages are spawned as subprocesses (`spawnAsync`, `pipeline.js:79`) with:

- command: `process.env.PYTHON || 'python'`
- `cwd`: repo root
- env: inherited `process.env` plus `DATA_DIR=data/collections/<id>`, `ENHANCED_DIR`,
  `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`

Child stdout and stderr are piped to the **server console**, not to the HTTP response. A benchmark
that needs stage logs must capture the server's stdout.

#### Shared failure modes

| Status | Message | Cause |
| --- | --- | --- |
| 502 | `<script>.py exited with code <n>` | Python stage exited non-zero. The last 500 chars of stderr are attached to the error object as `.detail` but **are not serialized into the response body** — check the server console |
| 502 | *(spawn error message)* | Python interpreter not found — check `PYTHON` |
| 503 | *(message containing "not found")* | A required input file is missing — run the earlier stage first |

The 503 remap (`toHttp`, `pipeline.js:136`) is applied on **`/embed`, `/categorize`, and
`/build-graph` only**. `/extract` and `/heuristic` do not call it, so a missing-input error there
surfaces as a raw `500`. This asymmetry is real; do not assert 503 uniformly.

#### External preconditions

| Stage | Requires |
| --- | --- |
| `extract` | **GROBID reachable at `GROBID_URL` (default `http://localhost:8070`)** — it is the only source of authors and parsed references, and extraction raises without it. Also reaches Crossref over the network for enrichment (non-fatal). |
| `embed`, `categorize` | Nothing external — MiniLM runs in-process |
| `heuristic` | Nothing external |
| `build-graph` | `KG_MODEL` set and reachable (Ollama, or a hosted route via dspy/litellm) |

---

#### 13. `GET /api/collections/:collectionId/pipeline/status`

`pipeline.js:263` · Auth: required + owner · No params

**Success — `200`** — last-run ISO timestamp per stage, `null` = never ran.

```json
{
  "doclings":   "2026-08-06T12:34:56.000Z",
  "embeddings": "2026-08-06T12:40:00.000Z",
  "categories": "2026-08-06T12:41:00.000Z",
  "heuristic":  null,
  "graph":      "2026-08-06T14:10:00.000Z"
}
```

| Field | Source |
| --- | --- |
| `doclings` | max `Document.extractedAt` in the collection |
| `embeddings` | `Collection.embeddingsMeta.updated` |
| `categories` | `Collection.categories.generatedAt` |
| `heuristic` | **hardcoded `null`** |
| `graph` | `Collection.knowledgeGraph.createdAt` |

**`heuristic` is permanently `null`.** The stage's output is console-logged and never persisted
(`pipeline.js:207-208`, `:273`). Do not treat `null` there as "has not run" — there is no way to
observe heuristic completion through this endpoint. Use the `POST …/pipeline/heuristic` response
body instead.

This endpoint reads only the already-loaded collection row plus one indexed document query — it is
cheap enough to poll while a stage runs on another connection.

---

#### 14. `POST /api/collections/:collectionId/pipeline/enhance`

`pipeline.js:280` · Auth: required + owner

Per-document page rasterization and enhancement. **A retired stage** — still routed, but excluded
from `/run` and not required by any later stage.

**Request**

| Field | Type | Required | Default |
| --- | --- | --- | --- |
| `docId` | string | yes | — |
| `dpi` | number | no | `300` |

**Success — `200`** — the enhancement report (`enhance_pdf.js:363`):

```json
{
  "docId": "a3f1c0d92e4b7a15",
  "pdfPath": "uploads/7/a3f1c0d92e4b7a15_paper.pdf",
  "numPages": 14,
  "dpi": 300,
  "processedAt": "2026-08-06T12:00:00.000Z",
  "pages": [ {
    "pageNumber": 1,
    "pageType": "text",
    "textDensity": 0.42,
    "dimensions": { "width": 2550, "height": 3300, "dpi": 300 },
    "enhancement": { "deskewAngle": 0.0, "binarizationThreshold": 172 },
    "blank": false
  } ]
}
```

**Errors**

| Status | Message |
| --- | --- |
| 400 | `"docId" is required` |
| 404 | `Document "<docId>" not found` |

Note the falsy check: `docId: ""` and `docId: 0` both yield the 400.

**Side effects** — rasterizes every page (memory-heavy, sequential), writes `page_<n>.png` files
into `ENHANCED_DIR/<docId>/` and the report to `ENHANCED_DIR/<docId>.json`. `ENHANCED_DIR` defaults
to `data/enhanced` and is **global, not per-collection** — reports are keyed by content hash.

---

#### 15. `POST /api/collections/:collectionId/pipeline/extract`

`pipeline.js:291` · Auth: required + owner

**Request**

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `force` | boolean | no | `false` | Only the literal `true` counts (`req.body?.force === true`); `"true"` is falsy here. Passes `--force` to `extract.py`, re-extracting documents already done |

**Success — `200`**

```json
{ "extracted": 3, "skipped": 10, "errors": [] }
```

`extracted` is a count; `skipped` and `errors` are passed through from `extract_report.json`
verbatim (`skipped` is a number, `errors` an array).

**Errors** — the shared 502 cases. Note `/extract` does **not** apply the 503 remap.

**Side effects** — exports document metadata and existing doclings to scratch, spawns
`backend/extraction/sapphire/extract.py`, stamps DOIs (`annotateDois`), enriches via Crossref
(`enrichDoclings` — network failures are caught and logged, never fatal), then ingests results into
`Document.docling` / `Document.extractedAt` and bumps `Collection.corpusUpdatedAt`.

Requires GROBID. This is usually the longest stage after `build-graph`.

---

#### 16. `POST /api/collections/:collectionId/pipeline/embed`

`pipeline.js:296` · Auth: required + owner

**Request**

| Field | Type | Required | Default |
| --- | --- | --- | --- |
| `force` | boolean | no | `false` — same `=== true` strictness as extract |

**Success — `200`**

```json
{ "chunks": 812, "docs": 13, "model": "Xenova/all-MiniLM-L12-v2", "dimensions": 384 }
```

**Errors** — shared 502 cases, plus `503` when doclings are missing (run `extract` first).

**Side effects** — chunks documents and encodes with MiniLM **in-process** (no subprocess), then
replaces all `Chunk` rows for the collection inside a transaction and updates
`Collection.embeddingsMeta`.

`dimensions` here is the number your `queryEmbedding` must match in
[endpoint 30](#30-post-apichatschatidchat). Read it from this response rather than hardcoding 384.

---

#### 17. `POST /api/collections/:collectionId/pipeline/categorize`

`pipeline.js:302` · Auth: required + owner

**Request**

| Field | Type | Required | Validation |
| --- | --- | --- | --- |
| `threshold` | number | **yes** | `typeof === 'number'`, not `NaN`, and `0 < threshold ≤ 1` |

Unlike `/run`, this endpoint has **no default** — omitting `threshold` is a 400.

**Success — `200`**

```json
{ "threshold": 0.75, "categories": 4, "docs": 13 }
```

`docs` is the total membership across all categories.

**Errors**

| Status | Message |
| --- | --- |
| 400 | `"threshold" must be a number in (0, 1]` |
| 503 | missing inputs — run `embed` first |

**Side effects** — writes `Collection.categories`, `Collection.docVectors`, and per-row
`Chunk.category`. Invalidates the memoized embedding map (via `docVectors.generatedAt`).

---

#### 18. `POST /api/collections/:collectionId/pipeline/heuristic`

`pipeline.js:311` · Auth: required + owner

BM25 + PageRank document ranking that decides which documents the graph stage reads in full.

**Request**

| Field | Type | Required | Default |
| --- | --- | --- | --- |
| `k` | number | no | `HEURISTIC_K` env, default `2` |

`k` is passed through `parseInt`, so the string `"3"` **is** accepted here (unlike `force`
elsewhere). It must end up a positive integer.

**Success — `200`**

```json
{ "k": 6, "topK": [ { "docId": "a3f1c0d9...", "filename": "paper.pdf", "cluster": 0,
                      "finalScore": 0.81, "bm25Score": 0.7, "bm25Representativeness": 0.6,
                      "bm25Novelty": 0.4, "pagerankScore": 0.02 } ],
  "edges": 143 }
```

`topK` is the full ranking array from `heuristic_output.json`; `edges` is a count.

**Errors**

| Status | Message |
| --- | --- |
| 400 | `"k" must be a positive integer` |

Shared 502 cases apply. The 503 remap is **not** applied here.

**Side effects** — the effective k is `max(k, ceil(documentCount × KG_FULL_TEXT_FRACTION))`
(`pipeline.js:202-204`, `KG_FULL_TEXT_FRACTION` default `0.4`), so the returned `k` is frequently
**larger than what you sent**. That is intentional: the graph stage needs at least that many ranked
documents. Assert `response.k >= requested_k`, not equality.

Results are written to scratch and console-logged but **never persisted to the DB** — which is why
`GET /status` reports `heuristic: null` forever.

---

#### 19. `POST /api/collections/:collectionId/pipeline/build-graph`

`pipeline.js:319` · Auth: required + owner · **No request body is read**

The knowledge-graph stage, deliberately separate from `/run` because it is a long LLM job that
nothing else depends on and that you re-run on its own when `KG_MODEL` changes.

**Success — `200`**

```json
{
  "model": "ministral-3:3b-instruct-2512-q4_K_M",
  "entities": 1420, "edges": 2310, "relations": 2310,
  "docs": 13, "fullTextDocs": 6, "summaryDocs": 7,
  "chunks": 812, "calls": 96, "callsFailed": 3
}
```

| Field | Meaning |
| --- | --- |
| `docs` | `sourceDocIds.length` |
| `fullTextDocs` | Documents sent every chunk (falls back to `docs` on older graphs) |
| `summaryDocs` | Documents sent only title + abstract + conclusion (falls back to `0`) |
| `calls` / `callsFailed` | **LLM calls, not chunks.** Chunks are packed many-per-call, so one failed call drops every chunk batched into it |

**Errors** — shared 502 cases, plus `503` for missing inputs (run `embed` first).

**Side effects and timing — read before benchmarking this.**

- Spawns `backend/extraction/kg_graph.py`. **Runtime is minutes to hours** on a local model.
- The Python process prints `@@KG_GRAPH_SAVED@@` after each per-call atomic flush of `graph.json`;
  the Node side watches for that marker and **incrementally ingests the partial graph** into
  `Collection.knowledgeGraph`. A crashed or killed run therefore leaves a usable partial graph with
  `complete: false`.
- After the subprocess exits, a final authoritative ingest writes both
  `Collection.knowledgeGraph` and `Collection.knowledgeGraphHtml`.
- Post-processing passes (entity resolution, abbreviation merging, predicate normalization, author
  pruning) run inside the Python stage before the final flush.

Because progress is only observable through the DB, a harness that wants live progress should poll
`GET …/corpus/graph` on a second connection and watch `callsCompleted` / `complete`.

Do not issue concurrent `build-graph` calls for the same collection — both will write
`Collection.knowledgeGraph` and the last writer wins.

---

#### 20. `POST /api/collections/:collectionId/pipeline/run`

`pipeline.js:326` · Auth: required + owner

Runs the four **indexing** stages in order: `extract` → `embed` → `categorize` → `heuristic`.
`graph` is deliberately excluded — call `/build-graph` separately.

**Request** — all optional.

| Field | Type | Default | Passed to |
| --- | --- | --- | --- |
| `threshold` | number in `(0, 1]` | `CATEGORIES_SIMILARITY` env, default `0.75` | `categorize` |
| `k` | number | `HEURISTIC_K` env, default `2` | `heuristic` |
| `force` | boolean | `false` | `extract` and `embed` |

Only `threshold` is validated at the route (`400 "threshold" must be a number in (0, 1]`). Note the
validation here omits the `isNaN` check that `/categorize` has, so an explicit `NaN` slips past
this guard and fails inside the stage instead.

**Success — `200`, always**

```json
{ "stages": {
  "extract":    { "ok": true, "extracted": 3, "skipped": 10, "errors": [] },
  "embed":      { "ok": true, "chunks": 812, "docs": 13, "model": "...", "dimensions": 384 },
  "categorize": { "ok": true, "threshold": 0.75, "categories": 4, "docs": 13 },
  "heuristic":  { "ok": true, "k": 6, "topK": [], "edges": 143 }
} }
```

**This endpoint returns HTTP 200 even when a stage fails.** A failing stage is recorded as
`{ "ok": false, "error": "<message>" }` and **breaks the loop**, so every later stage key is
*absent* from the object:

```json
{ "stages": {
  "extract": { "ok": true, "extracted": 3, "skipped": 10, "errors": [] },
  "embed":   { "ok": false, "error": "embeddings.json not found" }
} }
```

This is the single most important assertion detail in the API. A harness must:

```python
r = requests.post(f"{BASE}/api/collections/{cid}/pipeline/run", headers=H, json={}, timeout=None)
r.raise_for_status()                      # necessary but nowhere near sufficient
stages = r.json()["stages"]
assert set(stages) == {"extract", "embed", "categorize", "heuristic"}, f"stopped early: {stages}"
assert all(s["ok"] for s in stages.values()), stages
```

**Side effects** — the collection row is **re-read from the DB before each stage**
(`pipeline.js:348`), because earlier stages write fields later ones export. Individual stage side
effects are as documented for endpoints 15–18.

---

### 3.6 `/api/collections/:collectionId/corpus` — 4 endpoints

Source: `backend/routes/corpus.js` (`collectionCorpusRouter`).
Auth: required + collection ownership on all four. All are read-only.

---

#### 21. `GET /api/collections/:collectionId/corpus/embedding-map`

`corpus.js:188` · Auth: required + owner

Document vectors projected to 3D and 2D via UMAP, plus mutual-kNN edges with cosine similarities.

**Success — `200`**

```json
{
  "generatedAt": "2026-08-06T12:41:00.000Z",
  "mutualK": 10,
  "defaultThreshold": 0.75,
  "points": [ { "docId": "a3f1c0d9...", "filename": "paper.pdf", "title": "Some Paper",
                "p3": [0.412, -0.887, 0.031], "p2": [0.55, -0.21] } ],
  "edges": [ { "i": 0, "j": 4, "sim": 0.8123 } ]
}
```

| Field | Notes |
| --- | --- |
| `mutualK` | `CATEGORIES_MUTUAL_K`, default `10` |
| `defaultThreshold` | `CATEGORIES_SIMILARITY`, default `0.75` |
| `points[].title` | Falls back to `filename` when the document has no title |
| `points[].p3` / `p2` | Per-axis min–max scaled to `[-1, 1]`, rounded to 3 decimals |
| `edges[].i` / `j` | **Indices into `points`**, not docIds. Always `i < j` |
| `edges[].sim` | Cosine similarity, rounded to 4 decimals |

**Errors** — `404 No document vectors — run the categorize stage first`.

**Determinism and cost.** The UMAP layout is seeded with a fixed PRNG (`mulberry32(1337)`), so
coordinates are **reproducible across requests and restarts** for the same input vectors — you can
assert on them. Degenerate cases are special-cased: 1 document returns the origin, 2 documents
return `[-1,0,0]` / `[1,0,0]`.

The projection is CPU-heavy and memoized in a **process-level `Map` keyed by collection id**,
invalidated when `docVectors.generatedAt` changes. Expect a slow first call after each
`categorize`, then near-instant responses. The cache does not survive a server restart.

Edge construction is O(n²) in document count.

---

#### 22. `GET /api/collections/:collectionId/corpus/graph`

`corpus.js:216` · Auth: required + owner

**Success — `200`** — `Collection.knowledgeGraph` passed through verbatim:

```json
{
  "createdAt": "2026-08-06T14:10:00.000Z",
  "model": "ministral-3:3b-instruct-2512-q4_K_M",
  "contextTokens": 8192, "callMaxChars": 12000,
  "sourceDocIds": ["a3f1c0d9..."], "fullTextDocIds": [], "summaryDocIds": [],
  "chunksProcessed": 812,
  "calls": 96, "callsFailed": 3, "callsCompleted": 93, "complete": true,
  "promptTokens": 0, "outputTokens": 0, "totalTokens": 0, "modelRequests": 96,
  "tolerantParse": true, "droppedElements": 0, "salvagedFields": 0, "failedParses": 3,
  "entities": ["..."],
  "edges": ["..."],
  "relations": [["subject", "predicate", "object"]],
  "relationDocIds": [["a3f1c0d9..."]],
  "authorNodesDropped": 12,
  "entityClusters": {}, "predicateClusters": {}
}
```

`relations` is an array of `[subject, predicate, object]` triples. `relationDocIds` is
**index-aligned** with `relations` — `relationDocIds[i]` lists the docIds supporting
`relations[i]`.

**Errors** — `404 No knowledge graph — run the build-graph stage first`.

**Schema is unversioned.** Graphs written by older builds lack `relationDocIds`,
`authorNodesDropped`, `entityClusters`, and `predicateClusters`. Use `.get()`-style access with
defaults rather than assuming presence. `complete: false` means the graph is a partial flush from
an in-progress or aborted run.

---

#### 23. `GET /api/collections/:collectionId/corpus/graph/view`

`corpus.js:225` · Auth: required + owner

**Success — `200`**, `Content-Type: text/html` — the raw kg-gen standalone interactive page
(roughly 60 kB) as a string body, not JSON.

**Errors** — `404 {"error":"No knowledge graph — run the build-graph stage first"}` — a **JSON**
body even though the success path is HTML. Branch on status, not content type.

The HTML column is deliberately omitted from the collection row every other route loads, so this
endpoint re-queries it directly. There is little for a benchmark to assert here beyond a 200 and a
non-empty body.

---

#### 24. `GET /api/collections/:collectionId/corpus/chunks/:chunkId`

`corpus.js:237` · Auth: required + owner

**Path params** — `chunkId`, of the form `<docId>_<chunkIndex>` (e.g. `a3f1c0d92e4b7a15_7`).

**Success — `200`**

```json
{
  "id": "a3f1c0d92e4b7a15_7",
  "docId": "a3f1c0d92e4b7a15",
  "filename": "paper.pdf",
  "pages": [3, 4],
  "prefixLen": 0,
  "chunkIndex": 7,
  "heading": "3. Methods",
  "sectionIndex": 2,
  "chunkType": "body",
  "text": "..."
}
```

`pages` is a 1-based `[first, last]` pair (or `null`). `prefixLen` is the character offset of the
chunk body within `text`, used by the PDF viewer for highlight alignment. Note the response key is
`id`, not `chunkId`.

**Errors** — `404 Unknown chunk "<chunkId>"`.

This is the only endpoint that returns full chunk text — `POST …/chat` strips it from `sources`.
Use it to resolve a citation back to its source text.

---

### 3.7 `/api/chats` — 5 endpoints

Source: `backend/routes/chats.js`. Auth: required. A chat belongs to a collection; ownership is
checked through the collection's owner.

Chat summary shape (`chats.js:29-42`):

```json
{ "id": 4, "title": "New chat",
  "collection": { "id": 7, "name": "My corpus", "color": "#199e70", "crawler": "sapphire" },
  "createdAt": "...", "updatedAt": "..." }
```

---

#### 25. `GET /api/chats`

`chats.js:46` · Auth: required

**Success — `200`**

```json
{ "chats": [ { "id": 4, "title": "...", "collection": { }, "createdAt": "...", "updatedAt": "..." } ] }
```

Ordered `updatedAt` descending (most recent activity first), filtered to chats whose collection
belongs to the caller. `conversation` is **not** included — fetch a single chat for that.

---

#### 26. `POST /api/chats`

`chats.js:55` · Auth: required

**Request**

| Field | Type | Required | Default | Validation |
| --- | --- | --- | --- | --- |
| `collectionId` | integer | yes | — | `Number.isInteger` — **the string `"7"` is rejected** |
| `title` | string | no | `"New chat"` | trimmed; falls back to the default when blank |

**Success — `201`**

```json
{ "chat": { "id": 4, "title": "New chat", "collection": { }, "createdAt": "...", "updatedAt": "..." } }
```

**Errors**

| Status | Message |
| --- | --- |
| 400 | `"collectionId" is required` |
| 404 | `No collection <id>` |

The 400 message says "is required" but also fires for a present-but-non-integer value — including
a numeric string. Send a real JSON number.

The 404 also covers a collection owned by another user.

---

#### 27. `GET /api/chats/:chatId`

`chats.js:88` · Auth: required + ownership

**Success — `200`** — the summary plus the full conversation:

```json
{ "chat": { "id": 4, "title": "...", "collection": { },
  "createdAt": "...", "updatedAt": "...",
  "conversation": [
    { "role": "user", "content": "What is X?" },
    { "role": "assistant", "content": "X is ... [1]", "sources": [ ] }
  ] } }
```

`conversation` defaults to `[]`. Assistant messages carry a `sources` array (same shape as the
chat response); user messages do not.

**Errors** — `400 chatId must be an integer` · `404 No chat <id>`.

---

#### 28. `PATCH /api/chats/:chatId`

`chats.js:92` · Auth: required + ownership

**Request** — at least one field required.

| Field | Type | Validation |
| --- | --- | --- |
| `title` | string | non-empty after `.trim()` |
| `conversation` | array | every element must have string `role` **and** string `content`. Extra keys (e.g. `sources`) pass through untouched |

**Success — `200`**

```json
{ "chat": { "id": 4, "title": "Renamed", "collection": { }, "createdAt": "...", "updatedAt": "..." } }
```

The response is the **summary only** — it does not echo `conversation` back. Re-fetch with
`GET /api/chats/:chatId` to verify a conversation write.

**Errors**

| Status | Message |
| --- | --- |
| 400 | `"title" must be a non-empty string` |
| 400 | `"conversation" must be an array of {role, content} messages` |
| 400 | `Nothing to update` |

`Nothing to update` fires when neither field is present (both `undefined`).

**Side effects** — `conversation` is **overwritten wholesale**, not merged or appended. This is how
the UI deletes individual Q/A pairs: read, filter, write back. A benchmark using this to seed
history must send the complete array.

---

#### 29. `DELETE /api/chats/:chatId`

`chats.js:116` · Auth: required + ownership

**Success — `200`**

```json
{ "ok": true, "id": 4 }
```

**Errors** — `400 chatId must be an integer` · `404 No chat <id>`.

**Side effects** — deletes the `Chat` row and its conversation. The collection, its documents,
chunks, and graph are untouched.

---

### 3.8 `/api/chats/:chatId/chat` — 1 endpoint

Source: `backend/routes/chat.js`. Auth: required + chat ownership.

---

#### 30. `POST /api/chats/:chatId/chat`

`chat.js:68` · Auth: required + owner

The RAG endpoint: hybrid vector + BM25 retrieval over the collection's chunks, plus knowledge-graph
fact retrieval, then answer synthesis by `REASONING_MODEL`.

**Request**

| Field | Type | Required | Validation |
| --- | --- | --- | --- |
| `content` | string | yes | non-empty after `.trim()` |
| `queryEmbedding` | number[] | yes | array where **every** element is `typeof === 'number'`; length must equal the corpus dimensionality |

`queryEmbedding` is computed **client-side** — the browser embeds the question with the same MiniLM
model as the corpus (`Xenova/all-MiniLM-L12-v2`, 384 dims). The server does not embed queries.
A benchmark harness must produce this vector itself, using the same model, and L2-normalize it
(the retriever computes cosine as a plain dot product and assumes unit vectors).

Get the required dimensionality from the `dimensions` field of the `embed` stage response rather
than hardcoding it.

**Success — `200`**

```json
{
  "reply": "X is a method for ... [1] and it improves Y [2].",
  "model": "gemini/gemini-3.1-flash-lite",
  "sources": [ {
    "chunkId": "a3f1c0d92e4b7a15_7",
    "docId": "a3f1c0d92e4b7a15",
    "filename": "paper.pdf",
    "heading": "3. Methods",
    "pages": [3, 4],
    "sim": 0.7412,
    "boost": 1.05,
    "lex": 0.6231,
    "score": 0.9494,
    "quotes": ["verbatim span from the reply that this source grounds"]
  } ]
}
```

| Field | Meaning |
| --- | --- |
| `sim` | Cosine similarity between query and chunk, 4 dp |
| `boost` | Category keyword boost, `1` when none applies, 4 dp |
| `lex` | BM25 score normalized to the query's best chunk, 4 dp |
| `score` | `sim × boost + LEXICAL_WEIGHT × lex`, 4 dp — the ranking key |
| `quotes` | Verbatim spans in `reply` this source grounds; the viewer highlights exactly these |

`sources` is ordered by descending `score`. `sources[n-1]` is the target of a `[n]` citation marker
in `reply`. The chunk's full `text` and `embedding` are stripped server-side (`chat.js:96`) — use
[endpoint 24](#24-get-apicollectionscollectionidcorpuschunkschunkid) to fetch the text.

Retrieval keeps at most `RETRIEVER_TOP_K` chunks (default **8**) and then drops anything scoring
below `RETRIEVER_SCORE_FLOOR` × the top score, so `len(sources)` is variable and frequently less
than the cap. The top chunk always survives.

**Citation markers in `reply`**

| Marker | Meaning |
| --- | --- |
| `[n]` | Grounded in `sources[n-1]` |
| `[n!]` | Attributed to `sources[n-1]` but the grounding check failed |
| `[!]` | Ungrounded claim |
| `[G]` | Sourced from the knowledge graph, not a chunk |

**Errors**

| Status | Message |
| --- | --- |
| 400 | `"content" must be a non-empty string` |
| 400 | `"queryEmbedding" must be a number array (computed in the browser)` |
| 400 | `query embedding has <N> dims; corpus uses <M>` |
| 429 | `<label> rate limit hit for <model> — raise its quota or retrieve fewer chunks` |
| 502 | `Ollama unreachable at <url> (<reason>)` |
| 502 | `Ollama HTTP <status>: <detail>` |
| 502 | `Ollama at <url> sent nothing for <n>s` |
| 502 | `Ollama stalled — no output for <n>s` |
| 502 | `Ollama stream failed: <reason>` |
| 503 | `This collection has no indexed chunks — upload PDFs and run the pipeline first` |
| 503 | `REASONING_MODEL is not set — pick one in the Models tab` |
| 503 | `GEMINI_API_KEY is not set — <model> cannot be reached` |
| 503 | `<model> needs MICROSOFT_AZURE_PROJECT_ENDPOINT, MICROSOFT_AZURE_API_KEY and AZURE_DEPLOYMENT_NAME` |

The `429` is propagated from the hosted provider — a benchmark hitting Gemini in a loop will see it.

**Side effects**

- Builds LLM history from the stored `conversation` (roles and text only, `sources` dropped) plus
  the new question — **the whole conversation is resent to the model each turn**, so cost grows
  with chat length.
- Appends `{role:"user", content}` and `{role:"assistant", content: reply, sources}` to
  `Chat.conversation`, bumping `updatedAt`.
- **Auto-titles the chat** from the first question (`content.trim().slice(0, 60)`) when the
  conversation was empty *and* the title was still exactly `"New chat"`. A benchmark asserting on
  chat titles must account for this.
- Provider routing is decided by the `REASONING_MODEL` id: `gemini/*` → Google,
  a value matching `AZURE_DEPLOYMENT_NAME` → Azure AI Foundry, anything else → Ollama.
- Admin only: appends a prompt/context/response trio to `chat_log.txt` at the repo root. Fire and
  forget — a logging failure only warns.

The corpus (chunks + graph) is cached in-process, keyed on `Collection.corpusUpdatedAt`, so the
first chat after a pipeline run pays a load cost.

---

### 3.9 Static mounts (unauthenticated)

Two non-`/api` mounts, both conditional on the directory existing at server start:

| Path | Serves | Condition |
| --- | --- | --- |
| `GET /models/*` | `models/` — the vendored browser MiniLM + ONNX runtime wasm | `models/` exists (`npm run fetch:model`) |
| `GET /*` (non-`/api/`) | `frontend/dist/index.html` — SPA fallback | `frontend/dist` exists (`npm run build:web`) |

`server.js:48-60`. Both are **unauthenticated**.

The SPA fallback is registered as `app.get(/^\/(?!api\/).*/, …)` — the negative lookahead means
**anything under `/api/` is excluded**. So the boundary is:

| Request | Result (with `frontend/dist` present) |
| --- | --- |
| `GET /api/anything/unmatched` | `404` JSON `{"error":"No route GET /api/..."}` |
| `GET /collections` (missing the `/api` prefix) | `200` `text/html` — the SPA shell |

A typo'd path that keeps the `/api/` prefix fails loudly as JSON; one that drops it silently
returns HTML with status 200. Assert on `Content-Type` when testing 404 behavior.

---

## 4. End-to-end benchmark walkthrough

Full pipeline exercise, in dependency order. Each step lists what it unblocks.

| # | Call | Unblocks |
| --- | --- | --- |
| 1 | `POST /api/auth/login` → `token` | everything |
| 2 | `POST /api/collections` `{name}` → `collection.id` | all collection-scoped routes |
| 3 | `POST …/documents` (multipart `files`) | extract |
| 4 | `POST …/pipeline/run` `{threshold, k, force}` | embed → categorize → heuristic; chat |
| 5 | `GET …/pipeline/status` | confirms doclings / embeddings / categories timestamps |
| 6 | `GET …/corpus/embedding-map` | assert clustering geometry (needs categorize) |
| 7 | `POST …/pipeline/build-graph` | graph facts in chat; `…/corpus/graph` |
| 8 | `GET …/corpus/graph` | assert entity/relation counts |
| 9 | `POST /api/chats` `{collectionId}` → `chat.id` | chat |
| 10 | `POST /api/chats/:chatId/chat` `{content, queryEmbedding}` | assert reply + sources |
| 11 | `GET …/corpus/chunks/:chunkId` | resolve a cited source back to its text |
| 12 | `DELETE /api/collections/:id` | resets everything for the next run |

```python
import requests

BASE = "http://localhost:3000"

# 1. auth
token = requests.post(f"{BASE}/api/auth/login",
                      json={"email": "demo@gmail.com", "password": "demo123"}).json()["token"]
H = {"Authorization": f"Bearer {token}"}

# 2. collection
cid = requests.post(f"{BASE}/api/collections", headers=H,
                    json={"name": "bench", "crawler": "sapphire"}).json()["collection"]["id"]

# 3. upload
with open("paper.pdf", "rb") as fh:
    r = requests.post(f"{BASE}/api/collections/{cid}/documents", headers=H,
                      files=[("files", ("paper.pdf", fh, "application/pdf"))])
assert r.status_code == 201 and all(x["ok"] for x in r.json()["results"]), r.json()

# 4. index — blocking, minutes. timeout=None is deliberate.
r = requests.post(f"{BASE}/api/collections/{cid}/pipeline/run", headers=H,
                  json={"threshold": 0.75, "k": 2, "force": False}, timeout=None)
r.raise_for_status()
stages = r.json()["stages"]
assert set(stages) == {"extract", "embed", "categorize", "heuristic"}, f"stopped early: {stages}"
assert all(s["ok"] for s in stages.values()), stages
dims = stages["embed"]["dimensions"]        # 384 — what queryEmbedding must match

# 7. graph — blocking, can take hours
requests.post(f"{BASE}/api/collections/{cid}/pipeline/build-graph",
              headers=H, timeout=None).raise_for_status()

# 9-10. chat.  query_vector must be a unit-norm MiniLM embedding of `question`.
chat_id = requests.post(f"{BASE}/api/chats", headers=H,
                        json={"collectionId": cid}).json()["chat"]["id"]
r = requests.post(f"{BASE}/api/chats/{chat_id}/chat", headers=H,
                  json={"content": question, "queryEmbedding": query_vector}, timeout=None)
r.raise_for_status()
reply, sources = r.json()["reply"], r.json()["sources"]

# 12. teardown
requests.delete(f"{BASE}/api/collections/{cid}", headers=H).raise_for_status()
```

### Isolation notes

- **Per-collection state is fully isolated** — chunks, categories, vectors, and graph all hang off
  the collection. Concurrent benchmarks on *different* collections are safe.
- **Two caches are process-level and survive between runs**: the embedding-map memo (keyed on
  collection id + `docVectors.generatedAt`) and the retrieval corpus cache (keyed on
  `corpusUpdatedAt`). Both invalidate correctly on new pipeline output, but neither survives a
  server restart — first-call latency after a restart is not representative.
- **`data/enhanced/` is global**, keyed by content hash, not per collection. Enhancement output is
  shared across collections containing the same PDF.
- Deleting a collection removes its DB rows, `uploads/<id>/`, and `data/collections/<id>/`, but
  leaves `data/enhanced/` alone.

---

## 5. Environment reference

Read from the repo-root `.env` via `dotenv`. Values shown are the **code defaults** — the local
`.env` may override any of them, and `POST /api/corpus/settings` rewrites the five model roles.

### Server and auth

| Variable | Default | Affects |
| --- | --- | --- |
| `PORT` | `3000` | Base URL |
| `DATABASE_URL` | *(required, no default)* | Everything. Postgres; `docker-compose.yml` maps host **5433** → container 5432 |
| `JWT_SECRET` | `opencrawl-local-dev-secret` | Token signing/verification |
| `JWT_TTL` | `7d` | Token expiry |

### Documents

| Variable | Default | Affects |
| --- | --- | --- |
| `UPLOADS_DIR` | `uploads` | Endpoints 8, 10–12 |
| `MAX_PDF_SIZE_MB` | `50` | Endpoint 10 per-file limit |

### Pipeline

| Variable | Default | Affects |
| --- | --- | --- |
| `PYTHON` | `python` | Endpoints 15, 18, 19 — interpreter used to spawn stages |
| `DATA_DIR` | *(set per-spawn)* | Overridden by the route to `data/collections/<id>` |
| `ENHANCED_DIR` | `data/enhanced` | Endpoint 14 |
| `GROBID_URL` | `http://localhost:8070` | Endpoint 15 — **required** |
| `CROSSREF_MAILTO` | *(unset)* | Endpoint 15 — Crossref polite pool |
| `CHUNK_SIZE` | `180` | Endpoint 16 |
| `CHUNK_OVERLAP` | `0` | Endpoint 16 |
| `EMBED_BATCH_SIZE` | `32` | Endpoint 16 |
| `SAPPHIRE_EMBEDDING_MODEL` | `Xenova/all-MiniLM-L12-v2` | Endpoint 16 — corpus embeddings |
| `CLIENT_EMBEDDING_MODEL` | `Xenova/all-MiniLM-L12-v2` | Endpoint 30 — must match the corpus model |
| `CATEGORIES_SIMILARITY` | `0.75` | Endpoints 20, 21 — default threshold |
| `CATEGORIES_MUTUAL_K` | `10` | Endpoint 21 — `mutualK` |
| `HEURISTIC_K` | `2` | Endpoints 18, 20 |
| `KG_FULL_TEXT_FRACTION` | `0.4` | Endpoint 18 — raises effective `k` |

### Models

| Variable | Default | Affects |
| --- | --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` | Endpoints 4, 30 |
| `METADATA_MODEL` | *(unset)* | Endpoint 15 |
| `EXTRACTION_MODEL` | *(unset)* | Endpoint 15 |
| `QUERY_CLASSIFIER_MODEL` | *(unset)* | Query classification (written but not wired) |
| `KG_MODEL` | *(unset)* | Endpoint 19 |
| `REASONING_MODEL` | *(unset)* | Endpoint 30 — 503 if empty |
| `CHAT_REASONING_EFFORT` | `low` | Endpoint 30 — Gemini thinking budget |

### Retrieval tuning

| Variable | Default | Affects |
| --- | --- | --- |
| `RETRIEVER_TOP_K` | `8` | Endpoint 30 — max `sources` length |
| `RETRIEVER_SCORE_FLOOR` | `0.5` | Endpoint 30 — drops chunks scoring below this fraction of the top score |
| `RETRIEVER_LEXICAL_WEIGHT` | `0.3` | Endpoint 30 — BM25 blend weight in `score` |
| `RETRIEVER_KEYWORD_BOOST` | `1.05` | Endpoint 30 — multiplicative, per matched category keyword |
| `RETRIEVER_BM25_K1` | `1.5` | Endpoint 30 — term-frequency saturation |
| `RETRIEVER_BM25_B` | `0.75` | Endpoint 30 — length normalization strength |
| `RETRIEVER_FREQUENCY_PENALTY` | `0.3` | Endpoint 30 — anti token-loop penalty |
| `OLLAMA_IDLE_TIMEOUT_MS` | `60000` | Endpoint 30 — inactivity timeout, reset on every token (not a wall-clock deadline) |

### Provider credentials — redacted

These carry live secrets in the local `.env`. Values are **not** reproduced here.

| Variable | Value | Affects |
| --- | --- | --- |
| `GEMINI_API_KEY` | `<redacted — see local .env>` | Endpoint 30 when `REASONING_MODEL` starts with `gemini/` |
| `GEMINI_BASE_URL` | defaults to Google's OpenAI-compatible endpoint | Endpoint 30 |
| `MICROSOFT_AZURE_PROJECT_ENDPOINT` | `<redacted — see local .env>` | Endpoint 30 (Azure route) |
| `MICROSOFT_AZURE_API_KEY` | `<redacted — see local .env>` | Endpoint 30 (Azure route) |
| `AZURE_DEPLOYMENT_NAME` | `<redacted — see local .env>` | Endpoints 4 and 30 — its exact value is how the chat router detects the Azure provider |
| `CORE_API_KEY` | `<redacted — see local .env>` | Reference fetching scripts (not reachable over HTTP) |

Do not copy the real values into the benchmark repo — `.env` is gitignored here and has never been
committed, and the benchmark repo has no such guarantee.

---

## 6. Known gaps and benchmark gotchas

Ranked by how likely each is to cost you a debugging session.

1. **`POST …/pipeline/run` returns 200 on stage failure.** Assert on `stages[*].ok` and on the
   presence of all four keys. `raise_for_status()` alone will pass a run that died at `extract`.

2. **Long blocking requests with no streaming.** `build-graph` can run for hours on one HTTP
   connection. Any client with a default timeout (httpx: 5 s) will abort it. Use `timeout=None`.
   The server keeps working after your client disconnects, and partial graph progress is still
   persisted.

3. **`GET …/pipeline/status` always reports `heuristic: null`.** The stage is never persisted.
   `null` there is not evidence it did not run.

4. **`POST …/pipeline/heuristic` returns a `k` larger than you sent.** Effective `k` is
   `max(k, ceil(docCount × KG_FULL_TEXT_FRACTION))`. Assert `>=`, not `==`.

5. **Multer limit breaches are 500, not 413.** `MulterError` carries no `.status`, so the global
   handler defaults to 500 with `File too large` / `Too many files`.

6. **Upload partial success is a 201.** Inspect every `results[].ok`.

7. **`POST /api/chats` rejects a string `collectionId`.** `Number.isInteger("7")` is false. Send a
   JSON number. The error message ("is required") does not hint at this.

8. **`force` on extract/embed is `=== true`.** The string `"true"` is silently treated as false.
   But `k` on `/heuristic` *is* `parseInt`ed, so `"3"` works there. The strictness is inconsistent.

9. **Another user's resource is a 404, not a 403.** Do not write authorization tests expecting 403.

10. **No `GET /api/collections/:collectionId`.** Only `DELETE` is registered on that path. A `GET`
    returns `404 No collection <id>` for an unknown id but `404 No route GET
    /api/collections/<id>` for one you own — two different messages for the same request shape.
    Read collection metadata from `GET /api/collections`.

11. **`POST /api/corpus/settings` rewrites the developer's `.env`.** Snapshot and restore, or avoid
    it entirely.

12. **`GET …/corpus/graph/view` returns HTML on success but JSON on 404.** Branch on status.

13. **Graph JSON is unversioned.** Older graphs lack `relationDocIds`, `authorNodesDropped`,
    `entityClusters`, `predicateClusters`. `complete: false` means a partial flush.

14. **Chat auto-titles itself** on the first message when the title is still `"New chat"`
    (first 60 characters of the question).

15. **`PATCH /api/chats/:chatId` overwrites `conversation` wholesale** and does not echo it back in
    the response.

16. **`enhance` is a retired stage**, still routed but excluded from `/run` and depended on by
    nothing.

17. **Only the `sapphire` crawler is implemented.** `ruby` and `topaz` are accepted by
    `POST /api/collections` but their stage implementations are stubs.

18. **`QUERY_CLASSIFIER_MODEL` is configurable but unwired** — setting it changes nothing at
    runtime.

19. **`KG_MODEL` accepts the Azure deployment but the graph stage will fail with it.** The graph
    stage reaches models through dspy/litellm, which has no Azure route configured here
    (`corpus.js:59-62`). Ollama and Gemini both work.

20. **Concurrent pipeline calls on one collection are not guarded.** No locking, no job table.
    Serialize per collection in your harness.

21. **`npm run dev` restarts on any `backend/` file change**, killing in-flight pipeline runs. Use
    `npm start` for benchmarking.
