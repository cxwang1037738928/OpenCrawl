# OpenCrawl

**Ask questions about a library of research papers and get answers you can check.**

Upload PDFs — scanned or digital — and OpenCrawl extracts them, maps how they cite each other,
builds a knowledge graph over the whole corpus, and answers questions in chat. Every citation is
verified against the source text before you see it, and clicking one opens the PDF with the
supporting sentences highlighted.

The design principle is verifiability over fluency: an answer that cannot be traced to a source
is treated as a defect, not a feature.

---

## Features

**Grounded answers with sentence-level citations**
Each claim links to the passage that supports it. Click a citation and the source PDF opens to
the page with the exact sentences highlighted — not the whole paragraph.

**Automatic hallucination flagging**
Every citation the model writes is re-checked against the retrieved text. Quotes must match
verbatim; paraphrases must clear a lexical or semantic bar. Claims that fail are visibly marked
rather than silently accepted or quietly deleted.

**Handles scanned documents**
Digitally-born PDFs take a fast text path; scans are routed through OCR automatically. A
1990s scan and a clean arXiv preprint end up equally searchable.

**Automatic citation graph**
Reads each paper's own bibliography and links it to the other papers in your collection, then
uses PageRank over those links to surface which documents matter most — no external citation
index needed.

**Corpus-wide knowledge graph**
An LLM reads the collection and extracts entities and relationships, then cleanup passes merge
acronyms with their definitions, collapse duplicate concepts, unify synonymous relationships, and
strip author names.

**Graph-augmented retrieval**
Questions are answered from both retrieved passages and facts traversed from the knowledge
graph, so answers can draw on connections stated across different papers.

**Visual corpus exploration**
A 3D embedding map shows how documents cluster by topic, with an adjustable similarity threshold,
alongside an interactive view of the knowledge graph itself.

**Local or hosted models, per task**
Each role — extraction, graph building, answering — picks its own model independently. Run
everything locally through Ollama, or point the expensive stages at Gemini or Azure AI Foundry.

---

## How It Works

### Ingestion

```mermaid
flowchart TD
    A["User uploads PDF"] --> B["Validate and hash. docId is the first 16 hex of SHA-256"]
    B --> DB[("PostgreSQL")]

    DB --> D["Stage 1 - extract"]
    D --> D1{"Text layer in first 3 pages?"}
    D1 -->|"yes"| D2["Docling with OCR disabled"]
    D1 -->|"no"| D3["Docling with Tesseract OCR"]
    D --> D4["GROBID header and references, in parallel over HTTP"]
    D2 --> D5["Per-field metadata merge"]
    D3 --> D5
    D4 --> D5
    D5 --> D6["DOI regex over the first 5000 chars"]
    D6 --> D7["Crossref enrichment, non-fatal"]

    D7 --> E["Stage 2 - embed"]
    E --> E1["Structure-aware chunking, 180 words, section-aligned"]
    E1 --> E2["MiniLM-L12-v2, 384-d, L2-normalized"]

    E2 --> F["Stage 3 - categorize"]
    F --> F1["Mutual-kNN gated clustering plus TF-IDF keywords per cluster"]

    F1 --> G["Stage 4 - heuristic"]
    G --> G1["BM25 representativeness"]
    G --> G2["Vocabulary novelty from IDF"]
    G --> G3["PageRank over the citation graph"]
    G1 --> H["Blended score, ranked documents"]
    G2 --> H
    G3 --> H
    H --> DB
```

- OCR is decided per document by probing the first pages for a text layer — about 15 ms, versus
  minutes wasted running OCR on a document that never needed it.
- Bibliographic parsing runs concurrently with text extraction, so the network wait overlaps the
  local work instead of adding to it.
- Metadata falls back field by field across four sources, so a partial result still gets its gaps
  filled.
- Chunks follow the document's own section structure rather than a fixed window, and tables stay
  intact as single units.
- Documents are content-addressed by hash, so re-uploading the same paper is idempotent.

### Citation Graph

```mermaid
flowchart TD
    A["Parsed references per document"] --> B["Phase 1 - exact DOI match"]
    A --> C["Phase 2 - fuzzy title match"]
    C --> C1["Inverted token index, at least 2 shared informative tokens"]
    C1 --> C2["Bidirectional containment, contained side at least 15 chars"]
    C2 --> C3{"Both sides have authors?"}
    C3 -->|"yes"| C4["Require at least 1 shared surname"]
    C3 -->|"no"| C5["Accept the title match alone"]
    B --> D["Union of both phases. Edge points citing to cited"]
    C4 --> D
    C5 --> D
    D --> E["PageRank, damping 0.85"]
```

- Two passes: exact DOI matches give certain edges, fuzzy title matching covers the rest.
- Titles match by containment rather than equality, because reference parsers routinely return
  mangled titles — first-page license boilerplate, or a journal name in place of the article.
- Shared author surnames disambiguate near-matches; an inverted index keeps candidate generation
  from going quadratic.
- Publication dates are deliberately ignored — a scanned PDF's date is when it was scanned, and
  arXiv re-stamps its files, so date filtering deletes correct links.

### Knowledge Graph

```mermaid
flowchart TD
    A["Ranked documents from stage 4"] --> B{"In the top 40 percent?"}
    B -->|"yes"| C["Full text - every body chunk"]
    B -->|"no"| D["Summary - title, abstract, conclusion"]
    C --> E["Pack into calls"]
    D --> E
    E --> E1["One document per call. Chunks never split. Budget derived from the model context window"]
    E1 --> F["LLM entity and relation extraction"]
    F --> G{"Parse result"}
    G -->|"clean"| H["Merge triples into the graph"]
    G -->|"bad element"| G1["Tolerant parser drops that element, keeps the rest"]
    G -->|"call failed"| G2["Halve the batch and recurse, then the temperature ladder"]
    G1 --> H
    G2 --> F
    H --> I["Atomic flush plus stdout marker"]
    I --> J[("PostgreSQL - partial graph ingested")]
    I --> K{"More calls?"}
    K -->|"yes"| E
    K -->|"no"| L["graph.raw.json snapshot"]
    L --> M["Cleanup passes"]
```

- The ranking decides depth: important papers are read in full, the rest by title, abstract, and
  conclusion — full coverage without full cost.
- Batches are sized from the model's actual context window, and never mix two documents, so every
  extracted fact keeps exact provenance.
- Three layers of failure handling — a tolerant parser that discards only the malformed item,
  batch halving, then a temperature ladder — kept 1,145 calls running with zero failures.
- Progress is saved after every call, so a multi-hour build interrupted at hour three keeps
  everything up to hour three.

### Graph Cleanup

```mermaid
flowchart LR
    A["graph.raw.json"] --> B["1 - Entity resolution"]
    B --> C["2 - Abbreviation merging"]
    C --> D["3 - Predicate normalization"]
    D --> E["4 - Author pruning"]
    E --> F["graph.json plus interactive kg_view.html"]
```

- **Entity resolution** merges spelling and punctuation variants, while protecting case-sensitive
  distinctions like CO versus Co that naive folding would destroy.
- **Abbreviation merging** reunites acronyms with their definitions using the definitions authors
  actually wrote in the text — embedding similarity scores DFT against "density functional
  theory" below unrelated pairs, so similarity is the wrong instrument here.
- **Predicate normalization** collapses synonymous relationships, using a natural-language
  inference model to decide, since similarity alone would merge "derived from" with "designed for".
- **Author pruning** removes the paper authors that extraction pulls in from bibliographies —
  1,195 nodes on the reference corpus.

### Retrieval

```mermaid
flowchart TD
    Q["User question"] --> B["Browser embeds the query with MiniLM"]
    B --> S["POST to the chat endpoint with text plus vector"]
    S --> R1["Chunk retrieval"]
    S --> R2["Graph retrieval"]

    R1 --> L["BM25 with squared IDF and plural fallback"]
    R1 --> V["Dense cosine over chunk vectors"]
    L --> BL["cosine times keyword boost, plus weight times normalized BM25"]
    V --> BL
    BL --> FL["Score floor - drop anything below half the best chunk"]

    R2 --> G1["Gazetteer match over entities and merged surface forms"]
    G1 --> G2["Seed filter - document frequency at least 2"]
    G2 --> G3["Two-hop BFS. Hubs are valid seeds but never stepping stones"]
    G3 --> G4["score is seed specificity times corroboration"]
    G4 --> G5["Top 25 facts"]

    FL --> P["Prompt assembly - numbered excerpts plus an unnumbered fact block"]
    G5 --> P
    P --> M["LLM - Ollama, Gemini, or Azure AI Foundry"]
    M --> RC["Citation verification"]
```

- Keyword and semantic search are blended, so exact terminology and paraphrased questions both
  work.
- Results are cut by a relative score floor instead of a fixed count — a question with one real
  answer returns one, not seven near-misses the model has to explain away.
- Graph lookup matches entities and every surface form merged into them, so asking about "DFT"
  finds facts stored under the full name.
- Highly connected concepts can start a traversal but never be traversed through, which keeps
  two-hop expansion from reaching half the corpus.
- Queries are embedded in the browser, keeping model inference off the request path.

### Answer Verification

```mermaid
flowchart TD
    A["LLM reply with numeric citation markers"] --> B["Normalize marker syntax and drop out-of-range numbers"]
    B --> C["For each marker, locate the claim it belongs to"]
    C --> D{"Is the claim quoted?"}
    D -->|"yes"| E["Verbatim check - an 8-word run inside the whitespace-free chunk text"]
    D -->|"no"| F["Paraphrase check - lexical overlap at least 0.5 OR cosine at least 0.6"]
    E --> G{"Grounded?"}
    F --> G
    G -->|"yes"| H["Renumber to the chunk that actually supports it"]
    G -->|"no, number usable"| I["Flag as unverified but keep it clickable"]
    G -->|"no chunk at all"| J["Replace with a bare unsupported flag"]
    H --> K["Client renders the citation"]
    I --> K
    J --> K
    K --> L["Click opens the source PDF at the recorded page"]
    L --> M["Locate the chunk - whitespace-free match, then word-window anchors"]
    M --> N["Score sentences in the browser - cosine plus keyword bonus"]
    N --> O["Highlight overlay on the page"]
```

| Marker | Meaning |
|---|---|
| `[n]` | Verified, renumbered to the passage that actually supports it |
| `[n!]` | Plausible but unverified — kept clickable so you can judge it |
| `[!]` | No retrieved passage supports this claim |
| `[G]` | Drawn from a knowledge-graph fact |

- Citation numbers from the model are treated as a guess, not an index, and are corrected against
  what the text actually says.
- Quoted material is held to a stricter standard than paraphrase: it must appear word for word.
- Matching ignores whitespace, so PDF line-break hyphenation does not break a valid quote.
- Highlighting narrows to the specific sentences behind the claim, tightening or widening
  depending on whether you clicked a precise citation or a general source link.

---

## By the Numbers

Reference build — a 192-document machine-learning and materials-science corpus.

| | |
|---|---|
| Documents indexed | 192 PDFs → 6,628 searchable chunks |
| Knowledge graph | 20,301 entities · 22,260 relations |
| Cleanup effect | distinct relationship types 3,824 → 2,581 (−32%) |
| Author noise | 1,195 nodes removed|

---

## Stack

**Frontend** React 18 · Vite · Three.js · pdf.js · Transformers.js
**Backend** Node 22 · Express · PostgreSQL 16 · Prisma · JWT
**Processing** Python 3.11 · Docling · Tesseract · GROBID · NetworkX · kg-gen · DSPy
**Models** MiniLM-L12-v2 embeddings · NLI cross-encoder · Ollama, Gemini, or Azure AI Foundry

---

## Architecture

Implementation detail — stage-by-stage design, the reasoning behind each threshold, and the
measurements that motivated them — lives in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.
