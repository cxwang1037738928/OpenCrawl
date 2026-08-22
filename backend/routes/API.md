# API reference

All routes are mounted in [server.js](../server.js) under `/api`. Everything
except `/api/auth` requires `Authorization: Bearer <token>` (`requireAuth`).
Errors are always `{ error: string }` with the status on the response.

Ownership is enforced once per subtree, not per route: `loadOwnedCollection`
(collections.js) and `loadOwnedChat` (chats.js) resolve `:collectionId` /
`:chatId` for the calling user and 404 if it isn't theirs.

## `/api/auth` — [auth.js](auth.js)

| Method | Path | Body → Response |
| --- | --- | --- |
| POST | `/register` | `{email, password}` → `201 {token, user}` · 409 if taken |
| POST | `/login` | `{email, password}` → `{token, user}` · 401 on bad credentials |
| GET | `/me` | → `{user}` — `{user: null}` for guests/expired tokens, never 401 |

Passwords are bcrypt-hashed; min 6 chars. No email verification or resets.

## `/api/corpus` — [corpus.js](corpus.js) (`modelsRouter`, global)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/models` | Installed Ollama tags, KG model catalog, per-role choices + pick-lists |
| POST | `/settings` | `{ROLE: model, …}` → persists to `.env` **and** `process.env` |

Roles: `METADATA_MODEL`, `EXTRACTION_MODEL`, `QUERY_CLASSIFIER_MODEL`,
`KG_MODEL`, `REASONING_MODEL`.

## `/api/collections` — [collections.js](collections.js)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | The owner's collections (+ document counts) |
| POST | `/` | `{name, crawler?}` — `sapphire`\|`ruby`\|`topaz`; orb color auto-assigned |
| DELETE | `/:collectionId` | Deletes the collection and everything under it (DB cascade + uploaded PDFs + scratch dir) |

### `/:collectionId/documents` — [documents.js](documents.js)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | List the collection's documents |
| POST | `/` | Upload PDFs (multipart field `files`, ≤20) → `{uploaded, results[]}`; per-file failures are reported, not thrown |
| GET | `/:docId/pdf` | Stream the original PDF |
| DELETE | `/:docId` | Delete the row + file (chunks cascade) |

Uploads are validated on magic bytes, size (`MAX_PDF_SIZE_MB`, default 50),
page count, and per-collection duplicate hash. `docId` is a sha256 prefix, so
the same PDF gets the same id in every collection.

### `/:collectionId/pipeline` — [pipeline.js](pipeline.js)

| Method | Path | Body | Purpose |
| --- | --- | --- | --- |
| GET | `/status` | — | Last-run timestamp per stage (`null` = never ran) |
| POST | `/enhance` | `{docId, dpi?}` | Per-document page enhancement; run before extract |
| POST | `/extract` | `{force?, forceOcr?}` | docling PDF → `Document.docling` |
| POST | `/citations` | — | **sapphire only** — reference lists → `Collection.citationGraph`; 400 on other crawlers |
| POST | `/embed` | `{force?}` | Chunk + MiniLM encode → `Chunk` rows |
| POST | `/categorize` | `{threshold}` | Required, in `(0, 1]` — clusters → `categories`/`docVectors` |
| POST | `/heuristic` | `{k?}` | BM25 (+ PageRank when a citation graph exists); console only, not persisted |
| POST | `/build-graph` | — | kg-gen entity/relation graph → `knowledgeGraph` + `knowledgeGraphHtml` |
| POST | `/run` | `{threshold?, k?, force?, forceOcr?}` | The **indexing** stages in order → `{stages}`; a failing stage stops the run |

`/run` deliberately excludes `/build-graph`: indexing is minutes and everything
depends on it, while the graph is a long LLM run nothing else depends on. Which
stages `/run` covers depends on the collection's crawler (`indexStagesFor`).

### `/:collectionId/corpus` — [corpus.js](corpus.js) (`collectionCorpusRouter`)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/embedding-map` | Doc points projected to 3D + 2D (UMAP) with mutual-kNN edges; memoized on `docVectors.generatedAt` |
| GET | `/graph` | `Collection.knowledgeGraph` passthrough |
| GET | `/graph/view` | kg-gen's standalone interactive page, as HTML text |
| GET | `/chunks/:chunkId` | One indexed chunk (text, pages, `prefixLen`) for PDF highlighting |

The first three 404 until the matching stage has run.

## `/api/chats` — [chats.js](chats.js)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | The owner's chats, newest activity first, each with its collection's `{id, name, color, crawler}` |
| POST | `/` | `{collectionId, title?}` → `201 {chat}` |
| GET | `/:chatId` | One chat, including `conversation` |
| PATCH | `/:chatId` | `{title?}` rename · `{conversation?}` rewrite (used to persist deleted Q/A pairs) |
| DELETE | `/:chatId` | Delete the chat; its collection stays |

### `POST /:chatId/chat` — [chat.js](chat.js)

RAG over the chat's collection. The **browser** embeds the question (same MiniLM
model as the corpus) and sends the vector along; retrieval and answering run
server-side.

```
→ { content, queryEmbedding: number[], docIds?: string[] }
← { reply, model, sources: [{chunkId, docId, filename, heading, pages, sim, boost, score, quotes}] }
  400 bad payload · 502 Ollama failure · 503 missing corpus/model
```

- `sources[n-1]` is what a `[n]` citation marker in `reply` refers to; `quotes`
  are the verbatim spans that source grounds, which the PDF viewer highlights.
- `docIds`, when present, is a **hard** retrieval scope. Omitted, the retriever
  infers document names from the question and only boosts them.
- Both messages are appended to `Chat.conversation`; the first exchange titles
  the chat. `chat_log.txt` is written for the admin user only.
