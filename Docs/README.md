# Day 5 Hands-on Lab: RAG Chatbot & Tool-Using AI Agent

> **Purpose of this file:** This README is a build specification. Hand it to Claude Code (or any coding agent) and ask it to generate the full project exactly as described below. It is also the student-facing guide for the lab.

**Audience:** Second-year B.E./B.Tech CSE students who have completed Days 1–4 (neural networks, CNNs, RNN/LSTMs, Transformers, Hugging Face, FastAPI basics).
**Duration:** ~3 hours (one lab session).
**Goal:** Build a "Ask My Documents" assistant in five progressive steps: semantic search → RAG → agent from scratch → agent with LangGraph → full-stack web app.

---

## 1. Learning Outcomes

By the end of the lab, students will be able to:

1. Convert text into embeddings and run semantic similarity search.
2. Build a RAG pipeline: load → chunk → embed → store → retrieve → generate with citations.
3. Explain and implement the ReAct (Thought → Action → Observation) loop in plain Python.
4. Rebuild the same agent with LangGraph using tool nodes and conditional routing.
5. Trace and evaluate agent behaviour (tool choice, faithfulness to sources).
6. Expose the agent through a FastAPI `/chat` endpoint and use it from a web UI.

---

## 2. Tech Stack (keep it exactly this simple)

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.10+ | |
| LLM (default) | **Ollama** running locally | Model name configurable via `.env` (default `llama3.2:3b`; must support tool calling) |
| LLM (fallback) | Any OpenAI-compatible hosted API (e.g., Groq free tier) | Selected via `LLM_PROVIDER=openai_compatible` in `.env` |
| Embeddings | `sentence-transformers` with `all-MiniLM-L6-v2` | Runs on CPU, no API key |
| Vector store | `chromadb` (persistent, local folder) | |
| Document loading | `pypdf` for PDFs, plain read for `.md` / `.txt` | |
| Agent framework | `langgraph` + `langchain-core` + `langchain-ollama` / `langchain-openai` | Used only in Step 4 |
| External tool | `wikipedia` Python package | Gracefully degrade if offline |
| Backend | `fastapi` + `uvicorn` | Same style as Day 1 |
| Frontend | `index.html` + `explorer.html`, Tailwind (CDN) + vanilla JavaScript | No build step, no Node.js |
| Charts (Explorer only) | Chart.js from cdnjs | Histogram, scatter map, similarity bars |
| Config | `python-dotenv` | |
| E2E testing | `pytest` + `pytest-playwright` (Python) + `requests` | Chromium only; no Node.js needed |

Do **not** add other frameworks (no LlamaIndex, no CrewAI, no React). Simplicity matters more than features for this audience.

---

## 3. Project Structure

```
day5-agentic-rag-lab/
├── README.md                  # Student-facing run guide (generate a shorter version of this file)
├── requirements.txt
├── .env.example
├── config.py                  # Loads .env, exposes settings (provider, model names, paths, top_k, chunk sizes)
├── llm.py                     # Single function get_llm_response(messages) + get_chat_model() for LangGraph
├── sample_docs/               # Knowledge base for the demo (see §5)
│   ├── hr_leave_policy.md     # PROVIDED — copy in as-is, do not regenerate
│   └── hr_leave_policy.pdf    # PROVIDED — same content as a 6-page PDF, for the upload demo
├── data/
│   └── chroma_db/             # Created at runtime; add to .gitignore
├── steps/
│   ├── step1_embeddings.py    # Semantic search demo
│   ├── step2_rag.py           # Ingestion + RAG Q&A, with vs without retrieval
│   ├── step3_agent_scratch.py # ReAct loop in plain Python
│   ├── step4_agent_langgraph.py
│   └── step5_evaluate.py      # Tracing + simple faithfulness check
├── rag/
│   ├── __init__.py
│   ├── ingest.py              # load_documents, chunk_text, build_index
│   ├── chunkers.py            # 3 chunking strategies, all returning character offsets (see §7A)
│   ├── projection.py          # PCA (NumPy SVD) to project 384-dim vectors to 2D
│   └── retriever.py           # retrieve(query, k) -> list of {text, source, score}
├── agent/
│   ├── __init__.py
│   ├── tools.py               # calculator, search_documents, wikipedia_lookup (+ TOOL_REGISTRY)
│   ├── react_agent.py         # From-scratch ReAct agent
│   └── graph_agent.py         # LangGraph agent
├── app/
│   ├── main.py                # FastAPI app (includes the explorer router)
│   ├── explorer.py            # /explorer/* API endpoints (see §7A)
│   └── static/
│       ├── index.html         # Chat UI
│       └── explorer.html      # Chunking & Vector DB Explorer UI
├── eval/
│   └── questions.json         # 8–10 test questions with expected tool + expected source file
├── tests/
│   ├── conftest.py            # Starts uvicorn on a free port, shared fixtures
│   ├── fixtures/
│   │   └── wfh_policy.md      # Small extra doc used by the upload test
│   ├── test_api.py            # API-level E2E tests (requests)
│   ├── test_ui.py             # Browser E2E tests (Playwright)
│   └── test_explorer.py       # Explorer API + UI tests
└── pytest.ini                 # Registers markers: ui, llm
```

---

## 4. Configuration

`.env.example`:

```
LLM_PROVIDER=ollama               # ollama | openai_compatible | mock (mock is for tests only, see §8)
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

# Only used if LLM_PROVIDER=openai_compatible
OPENAI_COMPAT_BASE_URL=
OPENAI_COMPAT_API_KEY=
OPENAI_COMPAT_MODEL=

EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=data/chroma_db
COLLECTION_NAME=course_notes
CHUNK_SIZE=500                    # characters
CHUNK_OVERLAP=100
TOP_K=4
MAX_AGENT_STEPS=6
```

`llm.py` must hide the provider difference so every other file just calls `get_llm_response(messages: list[dict]) -> str`. Print a clear, friendly error if Ollama is not running or the model is not pulled (e.g., "Run: `ollama pull llama3.2:3b`").

---

## 5. Sample Document (`sample_docs/hr_leave_policy.md`)

The knowledge base is a **fictional HR leave policy** for "Nexora Technologies Pvt. Ltd.", provided alongside this README in two formats: `hr_leave_policy.md` and `hr_leave_policy.pdf` (6 pages, same content). Copy both into `sample_docs/` unchanged. By default `build_index()` indexes only the `.md` copy so the chat answers aren't duplicated; the PDF is used for the upload demo in the Explorer (§7A). It was chosen because:

- Every student understands leave rules, so no domain explanation is needed.
- It is full of company-specific numbers (12 CL, 18 EL, 1.5 EL/month, 30-day carry forward, 60-day Comp Off expiry) that a general LLM **cannot** know, which makes the "with vs without RAG" comparison obvious.
- It mixes headings, bullet lists, tables and an FAQ section, which is ideal for showing how chunk size and overlap affect retrieval (e.g., a table split across two chunks).
- Its numbers combine naturally with the calculator tool (e.g., EL accrual and encashment questions).

Students may later drop their own PDFs into this folder.

**Chunking demo requirement:** `rag/ingest.py` must include a `preview_chunks(n=5)` helper, and `step2_rag.py` must accept `--show-chunks` to print the first 5 chunks with their boundaries marked (`---- chunk 3 [hr_leave_policy.md] (487 chars) ----`). Also accept `--chunk-size` and `--chunk-overlap` CLI overrides so the instructor can re-index live with, say, 200 vs 800 characters and show how answers change.

---

## 6. Step-by-Step Lab Specification

Each `steps/stepN_*.py` must be runnable on its own (`python steps/step1_embeddings.py`), print clearly labelled output with section headers, and contain generous comments written for students. Keep each file under ~150 lines.

### Step 1 — Embeddings & Semantic Search (`step1_embeddings.py`)

- Load `all-MiniLM-L6-v2`; embed ~8 hard-coded sentences (mix of HR policy lines like "Employees get 5 working days of Marriage Leave" and unrelated ones like cricket, cooking).
- Print the embedding shape (should be 384 dims) and the first 8 values of one vector.
- Take a user query from `input()` (default if empty: "How many days off can I take for my wedding?" — this matches "Marriage Leave" despite sharing almost no keywords).
- Rank sentences by cosine similarity (compute manually with NumPy, show the formula in a comment).
- Also run a naive keyword-overlap score side by side to show semantic search finds matches with no shared words.

### Step 2 — RAG Pipeline (`step2_rag.py`, using `rag/ingest.py` and `rag/retriever.py`)

**`rag/ingest.py`:**
- `load_documents(folder)` → list of `{text, source}`; supports `.md`, `.txt`, `.pdf`.
- `chunk_text(text, size, overlap)` → character-based sliding window; try to break on paragraph/sentence boundaries when possible.
- `build_index(rebuild=False)` → embeds chunks and stores them in Chroma with metadata `{source, chunk_id}`. Skip rebuilding if the collection already has data unless `rebuild=True`.
- Print stats: number of files, chunks, average chunk length.

**`rag/retriever.py`:**
- `retrieve(query, k=TOP_K)` → list of `{text, source, score}` sorted by relevance.

**`step2_rag.py` flow:**
1. Build index.
2. Loop: read a question; show the top-k retrieved chunks (source + first 150 chars + score).
3. Build a grounded prompt: system instruction "Answer only from the context. If the answer is not in the context, say you don't know. Cite sources as [filename]."
4. Print **two answers side by side**: (a) LLM without context, (b) LLM with RAG context. This comparison is the teaching moment.
5. Type `exit` to quit.

### Step 3 — Agent from Scratch (`step3_agent_scratch.py`, `agent/tools.py`, `agent/react_agent.py`)

**`agent/tools.py`** — three tools, each a plain function with a docstring, plus a `TOOL_REGISTRY` dict `{name: {"fn": ..., "description": ...}}`:

| Tool | Behaviour |
|---|---|
| `calculator(expression: str)` | Safely evaluate arithmetic using Python's `ast` module (allow + − × ÷ ** % parentheses and numbers only). **Never use `eval()`.** Return an error string on invalid input. |
| `search_documents(query: str)` | Calls `retrieve()`; returns top chunks formatted as `[source] text`. |
| `wikipedia_lookup(topic: str)` | Returns a 3-sentence summary; on network failure or disambiguation, return a helpful message instead of crashing. |

**`agent/react_agent.py`** — implement ReAct **without any framework** and **without native tool calling** (text-based, so it works with any model):

- System prompt lists the tools and enforces this exact format:
  ```
  Thought: <reasoning>
  Action: <tool_name>
  Action Input: <input>
  ```
  or, when done:
  ```
  Thought: <reasoning>
  Final Answer: <answer>
  ```
- Loop up to `MAX_AGENT_STEPS`: call LLM → parse with regex → run tool → append `Observation: <result>` → repeat.
- Use a stop-sequence or truncate the LLM output at "Observation:" so the model can't invent observations.
- Handle: unknown tool name, unparseable output (send a correction message back to the model), and hitting the step limit (return best effort + a warning).
- Return `{"answer": str, "trace": [ {step, thought, action, action_input, observation} ]}`.

**`step3_agent_scratch.py`:** run the agent on these demo questions and print the trace with colour (use `rich` if installed, otherwise plain prints):
1. "What is 18% of 4520, plus 75?" → calculator
2. "How many days of Sick Leave can be carried forward?" → search_documents
3. "What is the Maternity Benefit Act in India?" → wikipedia_lookup
4. "I joined in March. How many Earned Leave days will I have accrued by the end of September, and can I encash them?" → search_documents then calculator (7 × 1.5 = 10.5)

Then allow free-form questions from `input()`.

### Step 4 — Same Agent with LangGraph (`step4_agent_langgraph.py`, `agent/graph_agent.py`)

- Wrap the same three functions from `tools.py` as LangChain tools (`@tool` decorator) — do not duplicate logic.
- Build a `StateGraph` with `MessagesState`: an `agent` node (chat model with `.bind_tools()`), a `tools` node (`ToolNode`), and a conditional edge (`tools_condition`) looping back to `agent`.
- Set a recursion limit equal to `MAX_AGENT_STEPS * 2`.
- Expose `run_graph_agent(question) -> {"answer", "trace"}` using the **same trace shape** as Step 3 so the UI can display either.
- `step4_agent_langgraph.py` runs the same 4 demo questions, prints the trace, and prints an ASCII/Mermaid drawing of the graph (`graph.get_graph().draw_mermaid()`).
- Add a comment block at the top comparing Step 3 and Step 4 line-by-line ("what the framework is doing for you").

### Step 5 — Tracing & Evaluation (`step5_evaluate.py`, `eval/questions.json`)

- `eval/questions.json`: 8–10 items like `{"question": ..., "expected_tool": ..., "expected_source": "hr_leave_policy.md" | null, "expected_keywords": ["24"]}`. Include at least one question the policy does **not** answer (e.g., "What is the cafeteria menu?") where the correct behaviour is "I don't know".
- For each question, run the chosen agent (`--agent scratch|graph`, default `scratch`) and record:
  - tools actually used, number of steps, latency in seconds
  - **tool accuracy:** did it call `expected_tool`?
  - **source hit:** did retrieved chunks include `expected_source`?
  - **faithfulness (simple LLM-as-judge):** ask the LLM "Is every claim in the answer supported by these sources? Reply YES or NO with one reason."
- Print a summary table and save `eval/results.json`.

---

## 7. Capstone: FastAPI Backend + Web UI (`app/`)

### Backend (`app/main.py`)

| Method | Endpoint | Request | Response |
|---|---|---|---|
| GET | `/health` | — | `{status, llm_provider, model, indexed_chunks}` |
| POST | `/ingest` | `{rebuild: bool}` | `{files, chunks}` |
| POST | `/upload` | multipart PDF/MD/TXT file | Saves to `sample_docs/`, re-indexes, returns `{filename, chunks}` |
| POST | `/chat` | `{message: str, mode: "rag" \| "agent_scratch" \| "agent_graph", history: list}` | `{answer, sources: [..], trace: [..], latency_ms}` |

- Serve `app/static/index.html` at `/`.
- Build the index on startup if empty.
- Enable CORS for local testing.
- Return friendly JSON errors (e.g., LLM not reachable) instead of stack traces.

### Frontend (`app/static/index.html`)

Single file, Tailwind via CDN, vanilla JS, clean and modern:

- Header "Ask My Documents — Day 5 Agentic AI Lab" with a health indicator dot (green/red from `/health`).
- Mode selector: **RAG only** / **Agent (from scratch)** / **Agent (LangGraph)**.
- Chat window with user/assistant bubbles; a "thinking…" indicator while waiting.
- Under each answer: collapsible **"Show reasoning trace"** panel listing each step (Thought / Action / Input / Observation) with tool-name badges, and a **Sources** list.
- File upload button for adding a PDF to the knowledge base.
- Show latency per answer.
- Mobile-friendly layout.
- Send button disabled while the input is empty or a request is in flight; Enter key sends.
- Show a red error bubble (not a blank screen) when `/chat` returns an error.
- Header link **"Chunking Explorer"** → `/explorer` (see §7A).

**Stable test selectors (required for §8):** add these `data-testid` attributes exactly:

| Element | `data-testid` |
|---|---|
| Health dot | `health-indicator` (plus `data-status="ok"` or `"error"`) |
| Mode selector | `mode-select` (option values `rag`, `agent_scratch`, `agent_graph`) |
| Message input | `chat-input` |
| Send button | `send-button` |
| Thinking indicator | `thinking-indicator` |
| Each user bubble | `user-message` |
| Each assistant bubble | `assistant-message` |
| Error bubble | `error-message` |
| Trace toggle (per answer) | `trace-toggle` |
| Trace panel | `trace-panel` |
| Each trace step | `trace-step` |
| Tool badge in a step | `tool-badge` (text = tool name) |
| Sources list | `sources-list` |
| Latency label | `latency` |
| File input | `upload-input` |
| Upload status text | `upload-status` |
| Indexed chunk count in header | `chunk-count` |
| Link to the Explorer | `explorer-link` |

---

## 7A. Chunking & Vector DB Explorer (`/explorer`)

A second page whose only job is to **show students what happens inside a RAG pipeline**: upload a PDF → see the raw extracted text → see exactly where it is cut into chunks → see the vectors stored in ChromaDB → see a query find its nearest chunks. The instructor projects this page during Step 2.

The Explorer uses its **own Chroma collections** (named `explorer_<config_hash>`) so it never disturbs the index used by the chat assistant.

### 7A.1 Chunking strategies (`rag/chunkers.py`)

Every strategy returns a list of `Chunk` dicts with **character offsets into the original text**, which is what makes the highlighted view possible:

```python
{"id": "c007", "text": "...", "start": 3120, "end": 3604, "length": 484,
 "page": 2, "section": "6. Earned Leave (EL)", "overlap_with_prev": 100}
```

| Strategy | Behaviour | Teaching point |
|---|---|---|
| `fixed` | Exactly `size` characters, step `size - overlap`, cuts anywhere | Fast, but cuts mid-word and mid-sentence |
| `recursive` | Try to split on `\n\n`, then `\n`, then `. `, then space, keeping chunks ≤ `size`, then add overlap | The common default (same idea as LangChain's RecursiveCharacterTextSplitter, but written by hand in ~40 lines) |
| `heading` | Split at Markdown headings / numbered section titles (`## `, `6. `, `6.1 `), then sub-split any section larger than `size` using `recursive` | Chunks follow the document's meaning; each chunk knows its section |

`rag/ingest.py` must use `chunkers.py` (default strategy `recursive`), so chat and Explorer share one implementation.

`page` comes from tracking cumulative character offsets of each PDF page during extraction. `section` is the nearest preceding heading.

**Warnings** to compute per chunk (shown as yellow badges in the UI):
- `mid_sentence` — chunk ends without `.`, `?`, `!` or a newline
- `split_table` — chunk boundary falls between two consecutive lines that both look like table rows
- `tiny` — chunk shorter than 25% of `size`

### 7A.2 Explorer API (`app/explorer.py`, mounted in `main.py`)

| Method | Endpoint | Request | Response |
|---|---|---|---|
| POST | `/explorer/parse` | multipart file (PDF/MD/TXT) | `{doc_id, filename, pages: [{page, text, chars}], total_chars, full_text}` — store parsed text in memory keyed by `doc_id` |
| POST | `/explorer/chunk` | `{doc_id, strategy, size, overlap}` | `{chunks: [Chunk], stats: {count, avg, min, max, lengths: [...]}, warnings_count}` — **no embedding**, so this is instant for live slider changes |
| POST | `/explorer/embed` | `{doc_id, strategy, size, overlap}` | `{collection, count, dims, embed_ms, records: [{id, document, metadata, embedding_preview (first 16 values), norm}], points: [{id, x, y, section}]}` |
| GET | `/explorer/collection/{name}?limit=20&offset=0` | — | Raw rows exactly as Chroma stores them (`ids`, `documents`, `metadatas`, `embeddings` truncated to 16 values) |
| POST | `/explorer/query` | `{collection, query, k}` | `{query_embedding_preview, query_point: {x, y}, results: [{id, rank, score, distance, text}], all_scores: [{id, score}], prompt_preview}` |
| POST | `/explorer/compare` | `{doc_id, query, k, configs: [{strategy, size, overlap}, {…}]}` | Two result sets side by side (embeds both configs if needed) |
| DELETE | `/explorer/collection/{name}` | — | `{deleted: true}` |
| GET | `/explorer/sample` | — | Returns the parse result for `sample_docs/hr_leave_policy.pdf` so the demo works with one click |

`rag/projection.py`: fit PCA on the chunk embeddings with NumPy SVD (no scikit-learn), store the mean and the top-2 components per collection, and project the query with the **same** components so it lands in the same 2D space.

`prompt_preview` is the exact grounded prompt that would be sent to the LLM (system instruction + numbered retrieved chunks + question), so students see what "augmented" means in Retrieval-*Augmented* Generation.

### 7A.3 Explorer UI (`app/static/explorer.html`)

Single file, Tailwind (CDN), vanilla JS, **Chart.js from cdnjs** for the charts. Layout is a vertical "pipeline" of five numbered panels, with an arrow between each, so the page itself reads like the RAG diagram from Session 1. Every panel has a collapsible **"ℹ️ What's happening here?"** box with 2–3 plain-English sentences for students.

**Panel ① — Upload & Parse**
- Drag-and-drop zone + "Use sample HR policy PDF" button.
- Shows filename, page count, total characters, and the extracted text in a scrollable box with page separators (`── Page 2 ──`).
- Teaching callout: point out that the PDF's **table became one cell per line** after extraction — "the computer doesn't see the table, only text".

**Panel ② — Chunking (live, no embedding yet)**
- Controls: strategy radio buttons (`fixed` / `recursive` / `heading`), **chunk size slider** (100–1500, step 50) and **overlap slider** (0–300, step 10). Debounce 250 ms and call `/explorer/chunk` on every change.
- **Highlighted document view:** the full text rendered once, with each chunk's span given a background colour from a rotating 8-colour palette. **Overlap regions** (text belonging to two chunks) get a striped/hatched background. A small chunk number tag sits at the start of each chunk. Hovering a chunk highlights its card in the list and vice versa.
- **Stats bar:** chunk count, average/min/max length, number of warnings.
- **Length histogram** (Chart.js bar chart) with a vertical line at the target size.
- **Chunk cards list:** id, page, section, char range (`3120–3604`), length, warning badges, full text.
- "Before vs after" toggle that shows two strategies' highlighted views side by side (e.g., `fixed` vs `heading`).

**Panel ③ — Embed & Store in ChromaDB**
- Button "Embed & store N chunks" → progress spinner → shows `embed_ms`.
- **Vector strip:** for each chunk, a row of 16 small coloured squares (blue = negative, white = 0, red = positive) for the first 16 of 384 dimensions, with "… 368 more" at the end. Clicking a row expands all 384 values as a 24 × 16 heat grid.
- **"What Chroma actually stores" table:** columns `id | document (first 80 chars) | metadata (JSON) | embedding (first 6 values…)`, paginated, read from `/explorer/collection/{name}` so students see the real database rows, not a mock.

**Panel ④ — Vector Space Map**
- Chart.js scatter plot of all chunks in 2D (PCA), points coloured by `section`, with a legend. Hover shows chunk id + first 100 chars.
- Info box: "384 dimensions squashed to 2 — nearby points mean similar meaning, but the picture is only approximate."

**Panel ⑤ — Query Playground**
- Query input with example chips: "How many days off for my wedding?", "Can I carry forward sick leave?", "Who approves a 10-day leave?", "What is the cafeteria menu?" (shows low scores everywhere).
- `k` selector (1–8).
- On search:
  - Query appears on the map as a **star**, with lines drawn to the top-k chunks.
  - **Ranked similarity bar chart** of *all* chunks (top-k highlighted), so students see the score drop-off.
  - Top-k chunks are re-highlighted in Panel ②'s document view with a thick border and rank number.
  - Result cards: rank, score (cosine similarity, 3 decimals), section, text.
  - **"Prompt sent to LLM"** box showing `prompt_preview` in a monospace block.
- **Compare mode** checkbox: pick a second chunk size/strategy, run the same query against both, and show the two top-k lists side by side. Suggested demo: size 200 vs 800 on "How many Earned Leave days can I carry forward and encash?".

**Header:** link back to the chat (`/`), and the current collection name and chunk count.

**Stable test selectors** (`data-testid`):

| Element | `data-testid` |
|---|---|
| File input | `ex-upload-input` |
| Sample PDF button | `ex-sample-button` |
| Page count / char count | `ex-page-count`, `ex-char-count` |
| Extracted text box | `ex-raw-text` |
| Strategy radios | `ex-strategy-fixed`, `ex-strategy-recursive`, `ex-strategy-heading` |
| Size / overlap sliders | `ex-size-slider`, `ex-overlap-slider` |
| Highlighted chunk span | `ex-chunk-span` (with `data-chunk-id`) |
| Overlap span | `ex-overlap-span` |
| Chunk count stat | `ex-chunk-count` |
| Chunk card | `ex-chunk-card` |
| Warning badge | `ex-warning` |
| Histogram canvas | `ex-histogram` |
| Embed button | `ex-embed-button` |
| Vector strip row | `ex-vector-row` |
| Chroma rows table | `ex-db-table` (rows `ex-db-row`) |
| Scatter canvas | `ex-scatter` |
| Query input / search button | `ex-query-input`, `ex-search-button` |
| Result card | `ex-result-card` (with `data-rank`, `data-score`) |
| Prompt preview | `ex-prompt-preview` |
| Compare checkbox / panels | `ex-compare-toggle`, `ex-compare-left`, `ex-compare-right` |

### 7A.4 Explorer requirements

- Must handle the 6-page sample PDF in under 2 s for parse + chunk, and under 10 s for embed on a CPU laptop.
- Reject files over 10 MB and non-PDF/MD/TXT files with a friendly message.
- Scanned (image-only) PDFs: detect near-zero extracted text and show "This PDF has no text layer — it would need OCR".
- All rendering of document text must be HTML-escaped (the document is user-supplied).

---

## 8. End-to-End Tests (Playwright + pytest)

### 8.1 Two test modes

LLM output is slow and non-deterministic, so tests run in two modes:

| Marker | LLM | Purpose | Command |
|---|---|---|---|
| `ui` (default) | `LLM_PROVIDER=mock` | Fast, deterministic checks of the whole pipeline: UI, API, retrieval, tool routing, trace rendering | `pytest -m ui` |
| `llm` | Real Ollama | Smoke tests that real answers contain the right facts | `pytest -m llm` (auto-skipped if Ollama is unreachable) |

**Mock LLM (`llm.py`, provider `mock`)** — deterministic, no network:
- **RAG mode:** returns `"Based on the policy: " + <first sentence of the top retrieved chunk> + " [<source>]"`. If the top retrieval score is below a threshold (configurable, e.g. 0.3), returns `"I don't know based on the provided documents."`
- **ReAct (scratch) agent:** if the question contains an arithmetic expression (regex for digits and operators), emit `Action: calculator`; otherwise emit `Action: search_documents` with the question as input. On the next call (after an Observation), emit `Final Answer:` containing the observation text.
- **LangGraph agent:** a fake chat model (subclass of LangChain's `BaseChatModel` or `GenericFakeChatModel`) that applies the same routing rule by returning an `AIMessage` with `tool_calls`, then a final `AIMessage`.
- Retrieval and tools are **real** in mock mode — only the LLM is faked — so the tests genuinely exercise chunking, embeddings, Chroma and tool execution.

### 8.2 `tests/conftest.py`

- Session fixture `server_url`: copy `sample_docs/` to a temp folder, set `CHROMA_PATH` to a temp folder (tests must never touch the real index), set `LLM_PROVIDER=mock` for `ui` runs, start `uvicorn app.main:app` on a free port in a subprocess, poll `/health` until ready (timeout 90 s for the first embedding-model download), and terminate it on teardown.
- Override pytest-playwright's `base_url` to `server_url`.
- Fixture `chat_page`: opens `/`, waits for `health-indicator` to have `data-status="ok"`.
- Helper `ask(page, question, mode)` that selects the mode, types, clicks send, and waits for `thinking-indicator` to disappear and a new `assistant-message` to appear.
- Default timeouts: 15 s for `ui`, 120 s for `llm`.
- Save screenshot + Playwright trace on failure (`--tracing retain-on-failure`, `--screenshot only-on-failure` in `pytest.ini`).

### 8.3 API tests (`tests/test_api.py`, marker `ui`)

| ID | Test | Assertion |
|---|---|---|
| API-01 | `GET /health` | 200; `status == "ok"`; `indexed_chunks > 0` |
| API-02 | `POST /ingest {rebuild: true}` | `files >= 1`, `chunks >= 5`; chunk count equals `/health` value afterwards |
| API-03 | `/chat` RAG: "How many days of Casual Leave per year?" | `sources` includes `hr_leave_policy.md`; at least one source chunk text contains `12` |
| API-04 | `/chat` RAG off-topic: "What is the cafeteria menu?" | answer contains "don't know" |
| API-05 | `/chat` `agent_scratch`: "What is 18% of 4520 plus 75?" | trace contains a step with `action == "calculator"`; answer contains `888.6` |
| API-06 | `/chat` `agent_scratch`: "How long is Paternity Leave?" | trace contains `search_documents` |
| API-07 | `/chat` `agent_graph`: same as API-05 and API-06 | same assertions (proves both agents share the trace shape) |
| API-08 | `/chat` with empty message | 422 or 400 with a JSON error, not a 500 |
| API-09 | `/chat` with invalid mode | 422 |
| API-10 | `POST /upload` `tests/fixtures/wfh_policy.md` | 200; `/health` chunk count increases |
| API-11 | Calculator safety: `/chat` agent with "Calculate __import__('os').system('ls')" | calculator observation is an error string; server still healthy |
| API-12 | `latency_ms` present and > 0 on every `/chat` response | — |

### 8.4 UI tests (`tests/test_ui.py`, marker `ui`)

| ID | Test | Steps / Assertion |
|---|---|---|
| UI-01 | Page loads | Title contains "Ask My Documents"; `health-indicator` status ok; `chunk-count` shows a number > 0 |
| UI-02 | Mode selector | `mode-select` has exactly 3 options with the expected values |
| UI-03 | Send button state | Disabled when `chat-input` is empty; enabled after typing |
| UI-04 | RAG answer + sources | Ask "How many days of Casual Leave per year?" in `rag` mode → one `user-message`, one `assistant-message`; `sources-list` contains `hr_leave_policy.md`; `latency` visible |
| UI-05 | Off-topic question | Ask "What is the cafeteria menu?" → assistant message contains "don't know" |
| UI-06 | Scratch agent uses calculator | `agent_scratch` mode, ask "What is 18% of 4520 plus 75?" → click `trace-toggle` → `trace-panel` visible; a `tool-badge` with text `calculator` exists |
| UI-07 | Scratch agent uses retriever | Ask "Who approves a 10-day leave request?" → `tool-badge` `search_documents` present |
| UI-08 | LangGraph agent | Repeat UI-06 in `agent_graph` mode |
| UI-09 | Trace toggle collapses | Click `trace-toggle` twice → `trace-panel` hidden again |
| UI-10 | Multi-turn chat | Ask two questions → 2 user and 2 assistant bubbles, in order |
| UI-11 | Enter key sends | Type and press Enter → message sent; input cleared |
| UI-12 | Thinking indicator | Use `page.route("**/chat", ...)` to delay the response by 2 s → `thinking-indicator` visible during the wait, hidden after |
| UI-13 | Backend error handling | Use `page.route("**/chat", ...)` to return HTTP 500 → `error-message` visible; page still usable (next question works after removing the route) |
| UI-14 | Upload flow | `set_input_files` on `upload-input` with `tests/fixtures/wfh_policy.md` → `upload-status` shows success; `chunk-count` increases; ask "How many WFH days per week are allowed?" in `rag` mode → sources include `wfh_policy.md` |
| UI-15 | Mobile layout | Viewport 375×812 → `document.documentElement.scrollWidth <= 375`; input and send button visible |

`tests/fixtures/wfh_policy.md`: generate a short (~150 words) fictional Nexora Work-From-Home policy stating employees may work from home up to **2 days per week** with manager approval.

### 8.4A Explorer tests (`tests/test_explorer.py`, marker `ui`)

The Explorer never calls the LLM, so all of these run with the mock provider.

| ID | Test | Assertion |
|---|---|---|
| EX-01 | `POST /explorer/parse` with `sample_docs/hr_leave_policy.pdf` | `pages` length is 6; `total_chars > 10000`; text contains "Casual Leave" |
| EX-02 | `/explorer/chunk` `fixed`, size 500, overlap 0 | every chunk `length <= 500`; `start`/`end` are contiguous (`chunks[i].start == chunks[i-1].end`) |
| EX-03 | `/explorer/chunk` with overlap 100 | `chunks[i].start == chunks[i-1].end - 100` for `fixed`; `overlap_with_prev > 0` |
| EX-04 | Offsets are honest | for every chunk, `full_text[start:end] == text` |
| EX-05 | Smaller size → more chunks | count at size 200 > count at size 800 |
| EX-06 | `heading` strategy | every chunk has a non-empty `section`; at least one chunk's section starts with "6." |
| EX-07 | `/explorer/embed` | `dims == 384`; `count` equals chunk count; `points` length equals `count` |
| EX-08 | `/explorer/collection/{name}` | returns `ids`, `documents`, `metadatas`, `embeddings`; metadata includes `page` and `section` |
| EX-09 | `/explorer/query` "How many days off for my wedding?" | top-1 result text contains "Marriage Leave" |
| EX-10 | `/explorer/query` "What is the cafeteria menu?" | top-1 score is lower than EX-09's top-1 score |
| EX-11 | Explorer collections are isolated | after EX-07, `/health` `indexed_chunks` is unchanged |
| EX-12 | Upload a 1×1 PNG renamed `.pdf` / an empty PDF | 400 with a friendly message; not 500 |
| UI-EX-01 | Open `/explorer`, click `ex-sample-button` | `ex-page-count` shows 6; `ex-raw-text` visible and contains "Nexora" |
| UI-EX-02 | Chunk view renders | `ex-chunk-span` count equals the number in `ex-chunk-count` and equals `ex-chunk-card` count |
| UI-EX-03 | Slider updates live | set `ex-size-slider` to 200 → `ex-chunk-count` increases; set to 1200 → decreases |
| UI-EX-04 | Overlap highlighting | set `ex-overlap-slider` to 0 → no `ex-overlap-span`; set to 100 → at least one `ex-overlap-span` |
| UI-EX-05 | Strategy switch | click `ex-strategy-fixed` → at least one `ex-warning` with text containing "mid"; click `ex-strategy-heading` → fewer warnings than `fixed` |
| UI-EX-06 | Embed & store | click `ex-embed-button` → `ex-vector-row` count equals chunk count; `ex-db-table` has ≥ 1 `ex-db-row`; `ex-scatter` visible |
| UI-EX-07 | Query playground | type "How many days off for my wedding?" → click `ex-search-button` → first `ex-result-card` (rank 1) contains "Marriage"; `ex-prompt-preview` contains the question and "Marriage Leave" |
| UI-EX-08 | Compare mode | enable `ex-compare-toggle`, sizes 200 vs 800 → both `ex-compare-left` and `ex-compare-right` show k result cards |
| UI-EX-09 | Upload via file input | `set_input_files` on `ex-upload-input` with `tests/fixtures/wfh_policy.md` → `ex-raw-text` contains "Work-From-Home" |
| UI-EX-10 | XSS safety | upload a `.md` fixture containing `<img src=x onerror="window.__xss=1">` → after render, `page.evaluate("window.__xss")` is `None` |
| UI-EX-11 | Navigation | click `explorer-link` on `/` → URL ends with `/explorer`; the back link returns to `/` |

### 8.5 Real-LLM smoke tests (marker `llm`)

Run against real Ollama with `LLM_PROVIDER=ollama`. Assert on facts that appear in the policy, using case-insensitive keyword checks rather than exact strings:

| ID | Question (mode) | Answer must contain |
|---|---|---|
| LLM-01 | "How many days of Sick Leave can I carry forward?" (`rag`) | `24` |
| LLM-02 | "How long is Paternity Leave?" (`rag`) | `10` |
| LLM-03 | "Within how many days must Comp Off be used?" (`agent_scratch`) | `60` |
| LLM-04 | "If I earn 1.5 days of Earned Leave per month, how many do I have after 8 months?" (`agent_graph`) | `12` |
| LLM-05 | "What is the cafeteria menu?" (`rag`) | one of: "don't know", "not mentioned", "not available" |

### 8.6 Requirements for the code generator

- Add `pytest`, `pytest-playwright` and `requests` to a separate `requirements-dev.txt`.
- `pytest.ini`: register `ui` and `llm` markers; default `addopts = -m "not llm" --browser chromium`.
- Tests must be independent (no ordering assumptions) and clean up temp folders.
- After generating, run `playwright install chromium` and `pytest -m ui`, and fix failures until all `ui` tests pass.

---

## 9. Mini Challenge (for students, after the lab)

Add your own tool to `agent/tools.py` and register it — for example:
- `leave_balance(employee_id: str)` — look up a small hard-coded JSON of employee leave balances
- `cgpa_calculator(grades: str)` — compute CGPA from "A,B+,O,..." style input
- `unit_converter(query: str)`

The tool must appear automatically in both agents and in the UI trace with no other code changes. Generate **one** example (`leave_balance`, with 3 fake employees) commented out in `tools.py` as a template. Students can then ask "How many EL days does EMP002 have, and how many can they encash?" — combining the new tool, retrieval and the calculator.

---

## 10. Non-Functional Requirements (for the code generator)

- Everything must run on a CPU-only laptop with 8 GB RAM.
- No API keys required in the default (Ollama) configuration.
- Every module has a top-of-file docstring explaining *what concept it teaches*.
- Use type hints and short functions; prefer readability over cleverness.
- No hard-coded absolute paths; all paths relative to project root via `config.py`.
- `requirements.txt` with pinned major versions only (e.g., `fastapi>=0.110,<1`).
- Add a `.gitignore` (`data/`, `.env`, `__pycache__/`, `.venv/`).
- Before finishing, run each step script once (with Ollama available) and fix any errors.

---

## 11. Student Run Guide (include in the generated project's README)

```bash
# 1. Install Ollama from ollama.com, then:
ollama pull llama3.2:3b

# 2. Set up Python
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 3. Run the lab steps in order
python steps/step1_embeddings.py
python steps/step2_rag.py
python steps/step3_agent_scratch.py
python steps/step4_agent_langgraph.py
python steps/step5_evaluate.py

# 4. Launch the web app
uvicorn app.main:app --reload
# open http://localhost:8000           -> chat assistant
# open http://localhost:8000/explorer  -> chunking & vector DB explorer

# 5. Run the end-to-end tests
pip install -r requirements-dev.txt
playwright install chromium
pytest -m ui                       # fast, uses the mock LLM
pytest -m llm                      # optional, needs Ollama running
pytest -m ui --headed --slowmo 500 # watch the browser (great for the demo)
```

**Troubleshooting section to include:** Ollama not running, model not pulled, slow first run (embedding model download), Wikipedia tool failing offline, Windows path issues, and what to do if a small model produces badly formatted ReAct output (switch to a larger model or the hosted fallback).

---

## 12. Suggested Lab Timing (instructor reference)

| Time | Activity |
|---|---|
| 0:00–0:15 | Setup check (Ollama + venv) |
| 0:15–0:35 | Step 1 — Embeddings |
| 0:35–0:55 | Chunking Explorer demo — upload the HR PDF, move the sliders, embed, query |
| 0:55–1:10 | Step 2 — RAG, with vs without retrieval |
| 1:10–1:50 | Step 3 — ReAct agent from scratch |
| 1:50–2:15 | Step 4 — LangGraph version |
| 2:15–2:30 | Step 5 — Evaluation |
| 2:30–3:00 | Capstone web app demo + mini challenge |
