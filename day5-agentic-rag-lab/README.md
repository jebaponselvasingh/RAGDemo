# Day 5 Hands-on Lab: RAG Chatbot & Tool-Using AI Agent

Build an **"Ask My Documents"** assistant step by step:

**semantic search → RAG → agent from scratch → agent with LangGraph → evaluation → agent memory → full-stack web app**

The knowledge base is a fictional HR leave policy for *Nexora Technologies Pvt. Ltd.* (`sample_docs/`).
It is full of company-specific numbers (12 CL, 18 EL, 1.5 EL/month, 30-day carry forward, 60-day Comp Off…)
that a general LLM **cannot** know, which makes the "with vs without RAG" comparison obvious.

- **Audience:** second-year B.E./B.Tech CSE students who know neural networks, Transformers, Hugging Face and FastAPI basics
- **Duration:** about 3 hours
- **Hardware:** a CPU-only laptop with 8 GB RAM; no API keys needed

---

## Contents

1. [What you will learn](#1-what-you-will-learn)
2. [Requirements](#2-requirements)
3. [Setup](#3-setup)
4. [The lab steps](#4-the-lab-steps)
5. [Agent memory (Step 6)](#5-agent-memory-step-6)
6. [The web app](#6-the-web-app)
7. [API reference](#7-api-reference)
8. [Configuration (`.env`)](#8-configuration-env)
9. [Tests](#9-tests)
10. [Measured results](#10-measured-results)
11. [Project structure](#11-project-structure)
12. [Mini challenge](#12-mini-challenge)
13. [Known limitations](#13-known-limitations)
14. [Troubleshooting](#14-troubleshooting)
15. [Suggested lab timing (instructors)](#15-suggested-lab-timing-instructors)

---

## 1. What you will learn

1. Turn text into **embeddings** and run semantic similarity search.
2. Build a **RAG pipeline**: load → chunk → embed → store → retrieve → generate with citations.
3. See how **chunking** (strategy, size, overlap) changes what the retriever can find.
4. Implement the **ReAct loop** (Thought → Action → Observation) in plain Python.
5. Rebuild the same agent with **LangGraph** (tool nodes, conditional routing).
6. **Trace and evaluate** an agent: tool choice, sources, key facts, faithfulness (LLM-as-judge).
7. Give agents **short-term memory**, both by hand and with a framework checkpointer.
8. Expose everything through **FastAPI** and use it from a web UI.

## 2. Requirements

| | |
|---|---|
| Python | 3.10 – 3.12 (developed and tested on **3.12**; 3.13+ may not have wheels for every dependency yet) |
| LLM | [Ollama](https://ollama.com) running locally with **`llama3.2:3b`** (default) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, CPU, downloaded automatically, ~90 MB) |
| Vector DB | ChromaDB, stored in `data/chroma_db/` (created at runtime) |
| Frameworks | `langgraph` + `langchain-core` (Step 4 only), FastAPI + uvicorn |
| Frontend | Two HTML files with Tailwind (CDN), vanilla JavaScript and Chart.js (cdnjs). No Node.js, no build step. |

Optional fallback: any **OpenAI-compatible hosted API** (e.g. Groq's free tier). See [Configuration](#8-configuration-env).

## 3. Setup

```bash
# 1. Install Ollama from https://ollama.com, then download the model (~2 GB):
ollama pull llama3.2:3b

# 2. Set up Python
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

The chat index is built automatically the first time a step script or the web app needs it.

---

## 4. The lab steps

Each script runs on its own, prints labelled sections and is heavily commented. Run them in order:

```bash
python steps/step1_embeddings.py        # sentences -> vectors; cosine similarity vs keyword search
python steps/step2_rag.py               # the same LLM with and without retrieved context
python steps/step3_agent_scratch.py     # a ReAct agent in plain Python
python steps/step4_agent_langgraph.py   # the same agent as a LangGraph graph
python steps/step5_evaluate.py          # score the agent   (--agent graph for Step 4's agent)
python steps/step6_memory.py            # give the agents short-term memory, two different ways
```

| Step | What happens | What to look for |
|---|---|---|
| **1. Embeddings** | Embeds 8 sentences (HR lines mixed with cricket and cooking) and ranks them for your query, by cosine similarity and by naive keyword overlap. Press Enter to use the default query. | *"How many days off can I take for my wedding?"* matches **Marriage Leave** although they share almost no words; keyword search picks the wrong sentence. |
| **2. RAG** | Builds the index, shows the top-k chunks with scores, then asks the LLM twice: **(a) without** and **(b) with** the retrieved context. Type `exit` to quit. | Without context the model guesses or talks about other countries' laws; with RAG it answers *"24 days [hr_leave_policy.md]"*. Off-topic questions get "I don't know". |
| **3. ReAct from scratch** | Runs 4 demo questions through a hand-written Thought → Action → Observation loop, printing a coloured trace, then takes your own questions. | The calculator answers 888.6, the search finds 24 days, Wikipedia explains the Maternity Benefit Act. |
| **4. LangGraph** | The same 4 questions with a `StateGraph` (agent node ⇄ ToolNode). It prints the graph as Mermaid (paste into <https://mermaid.live>). | The comment block at the top compares Step 3 and Step 4 line by line: what the framework does for you. |
| **5. Evaluation** | Runs `eval/questions.json` (10 questions) and scores each answer; saves `eval/results.json`. | Columns: tool accuracy, source hit, key facts, faithfulness (LLM-as-judge), steps, latency. |
| **6. Memory** | The follow-up *"Can it be split into two parts?"* asked without memory, with memory by hand, and with LangGraph's checkpointer. | Only with memory does the agent know "it" = Paternity Leave → *"up to two blocks (5 + 5 days)"*. |

**Step 2 flags:**

```bash
python steps/step2_rag.py --show-chunks                          # print the first 5 chunks with their boundaries
python steps/step2_rag.py --chunk-size 200 --chunk-overlap 50    # re-index live with other settings
```

The index re-builds automatically whenever the chunk settings change, and returns to the defaults the next
time you run without the flags.

**The three tools** (`agent/tools.py`) are plain Python functions:

| Tool | What it does |
|---|---|
| `calculator(expression)` | Safe arithmetic with Python's `ast` module (numbers, `+ - * / ** %`, parentheses). **Never uses `eval()`.** Returns the full equation, e.g. `18/100*4520 + 75 = 888.6`, or an error string. |
| `search_documents(query)` | Our RAG retriever: the top-k chunks formatted as `[source] text`. |
| `wikipedia_lookup(topic)` | A 3-sentence Wikipedia summary. Times out after 10 s and returns a helpful message when offline. |

---

## 5. Agent memory (Step 6)

An LLM remembers nothing between calls. A follow-up like *"Can it be split into two parts?"* only works if the
earlier conversation is sent again. The two agents deliberately show two ways to do that:

| Agent | Memory style | Where it lives |
|---|---|---|
| Step 3, from scratch | **By hand**: we pass the earlier questions and answers in as `history` | A Python list. In the web app, the browser keeps it and sends it with each request. |
| Step 4, LangGraph | **Framework**: a checkpointer saves every conversation under a `thread_id` | `InMemorySaver` on the server. In the web app, the page's session id is the `thread_id`. |

- **What the model sees:** only the last `MEMORY_MAX_MESSAGES` earlier messages (default 10), and only
  **questions and final answers**, not old tool results. The LangGraph checkpointer stores *everything*;
  `model_view()` in `agent/graph_agent.py` chooses what the model sees, and the current turn is always kept in full.
- **Why limit it:** every remembered message costs tokens on every call and can distract a small model.
- **Persistence:** memory is in RAM. Restarting the program or server forgets it. (LangGraph's `SqliteSaver`
  would make it survive restarts; it isn't used here, to avoid an extra dependency.)
- **RAG mode** also uses the browser's history: the last 6 messages go into the prompt, and the previous question
  is added to the search query so follow-ups find the right chunks.

---

## 6. The web app

```bash
uvicorn app.main:app --reload
```

### Chat assistant — <http://localhost:8000>

- **Modes:** *RAG only*, *Agent (from scratch)*, *Agent (LangGraph)*.
- **Header:** a health dot (green = LLM reachable; hover it for details), the number of indexed chunks, and a link to the Explorer.
- **Each answer shows:**
  - its **sources** (file, chunk id, score; click one to read the chunk)
  - the **latency**
  - for agents, a collapsible **reasoning trace** with tool-name badges and each Thought / Input / Observation
  - for follow-ups, a **🧠 remembers N earlier messages** label
- **📄 Add document:** upload a PDF, `.md` or `.txt` file (max 10 MB). It is saved to `sample_docs/` and indexed
  immediately. A PDF whose `.md`/`.txt` twin is already indexed is saved but not indexed twice.
- **＋ New chat:** forgets the conversation (clears the history and starts a new session id).
- **Keys:** Enter sends, Shift+Enter adds a new line. The Send button is disabled while empty or busy. Errors appear as a red bubble.
- **Layout:** works on a phone-sized screen.

### Chunking & Vector DB Explorer — <http://localhost:8000/explorer>

A page that opens up the RAG pipeline, designed to be projected during Step 2. It reads top to bottom like the RAG diagram:

| Panel | What you see |
|---|---|
| **① Upload & Parse** | Drag-and-drop or **Use sample HR policy PDF**. Shows the page count, character count and the extracted text page by page. Notice that the policy table became one cell per line. |
| **② Chunking** (live, no embedding) | Strategy (`fixed` / `recursive` / `heading`), a size slider (100–1500) and an overlap slider (0–300). The whole document is coloured chunk by chunk; **striped** text is overlap. Also: chunk-length histogram with the target size, chunk cards with warnings (`mid-sentence cut`, `split table`, `tiny chunk`), and a *Before vs after* view of two strategies side by side. |
| **③ Embed & Store** | Embeds the chunks into ChromaDB. Each vector is shown as a strip of 16 coloured cells (blue −, white 0, red +); click one for all 384 values as a heat grid. Below it, the **real rows Chroma stores** (id, document, metadata, embedding). |
| **④ Vector Space Map** | All chunks in 2D (PCA with NumPy), coloured by section. |
| **⑤ Query Playground** | Your query appears as a ★ on the map with lines to its nearest chunks. Also: a similarity bar chart of every chunk, the top-k result cards, top-k highlighted back in panel ②, and the exact **prompt sent to the LLM**. **Compare mode** runs the same query on a second chunk config (e.g. size 800 vs 200). |

- The Explorer uses its **own** Chroma collections (`explorer_<hash>`), so it never changes the chat's index.
- It opens with **`heading` / 800 / 100**, which keeps each policy section whole. Try `recursive` / 500 with
  *"How many days off for my wedding?"* to watch the *Marriage Leave* heading get cut off from its text.
- Uploads are checked: PDF/MD/TXT only, max 10 MB, and scanned PDFs are detected ("no text layer — it would need OCR").
  All document text is HTML-escaped.

---

## 7. API reference

Interactive docs: <http://localhost:8000/docs>. All errors are JSON (`{"error": ...}` or `{"detail": ...}`), never stack traces.

### Chat

| Method | Endpoint | Request | Response |
|---|---|---|---|
| GET | `/health` | — | `{status, message, llm_provider, model, indexed_chunks}`. `status` is `"error"` when the LLM can't be reached. |
| POST | `/ingest` | `{rebuild: bool}` | `{files, chunks}` |
| POST | `/upload` | multipart `file` (PDF/MD/TXT, ≤ 10 MB) | `{filename, chunks, indexed_chunks, skipped, message?}`. `skipped: true` means a PDF whose `.md`/`.txt` twin is already indexed. 400 with a friendly message for bad files. |
| POST | `/chat` | `{message, mode: "rag" \| "agent_scratch" \| "agent_graph", history: [{role, content}], session_id?}` | `{answer, sources: [{source, chunk_id, score, text}], trace: [{step, thought, action, action_input, observation}], latency_ms, mode, memory_messages}` |

- An empty message → 422.
- An unknown mode → 422.
- LLM unreachable → 503 with an explanation.

### Explorer

| Method | Endpoint | Request | Response |
|---|---|---|---|
| POST | `/explorer/parse` | multipart `file` | `{doc_id, filename, pages: [{page, text, chars, start}], total_chars, full_text}` |
| GET | `/explorer/sample` | — | Same as `/parse`, for `sample_docs/hr_leave_policy.pdf` |
| POST | `/explorer/chunk` | `{doc_id, strategy, size, overlap}` | `{chunks: [{id, text, start, end, length, page, section, overlap_with_prev, warnings}], stats: {count, avg, min, max, lengths}, warnings_count, collection}`. No embedding, so it's instant. |
| POST | `/explorer/embed` | `{doc_id, strategy, size, overlap}` | `{collection, count, dims, embed_ms, records: [{id, document, metadata, embedding_preview, norm}], points: [{id, x, y, section, text}]}` |
| GET | `/explorer/collection/{name}?limit=20&offset=0` | — | Raw Chroma rows: `ids, documents, metadatas, embeddings` (first 16 values each) |
| GET | `/explorer/vector/{name}/{chunk_id}` | — | All 384 values of one embedding (for the heat grid) |
| POST | `/explorer/query` | `{collection, query, k}` | `{query_embedding_preview, query_point, results: [{id, rank, score, distance, text, section, page, source}], all_scores, prompt_preview}` |
| POST | `/explorer/compare` | `{doc_id, query, k, configs: [cfg, cfg]}` | `{query, left: {config, collection, count, results}, right: {...}}` |
| DELETE | `/explorer/collection/{name}` | — | `{deleted: true}`. Only `explorer_*` collections can be deleted. |

Every chunk records exact character offsets, so `full_text[start:end] == text` always holds.
This is what makes the highlighted view possible.

---

## 8. Configuration (`.env`)

`.env.example` contains the core settings. Every setting can also be overridden by an environment variable.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai_compatible`, or `mock` (deterministic fake LLM, for tests) |
| `OLLAMA_MODEL` | `llama3.2:3b` | Any Ollama model you have pulled (`ollama list`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is running |
| `OPENAI_COMPAT_BASE_URL` / `_API_KEY` / `_MODEL` | — | Only for `LLM_PROVIDER=openai_compatible`, e.g. Groq: `https://api.groq.com/openai/v1` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHROMA_PATH` | `data/chroma_db` | Vector database folder |
| `COLLECTION_NAME` | `course_notes` | Chroma collection used by the chat |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | Chat index chunking, in characters |
| `TOP_K` | `4` | Chunks retrieved per question |
| `MAX_AGENT_STEPS` | `6` | Step limit for both agents |
| *Extra settings (not in `.env.example`)* | | |
| `CHUNK_STRATEGY` | `recursive` | Chat index strategy: `fixed`, `recursive` or `heading` |
| `DOCS_PATH` | `sample_docs` | Folder of documents to index (the tests point it at a temp copy) |
| `MEMORY_MAX_MESSAGES` | `10` | Earlier messages an agent may see (Step 6) |
| `MAX_OUTPUT_TOKENS` | `768` | Cap on one LLM reply, so a small model stuck in a loop can't run forever |
| `MOCK_SCORE_THRESHOLD` | `0.3` | Mock LLM only: below this retrieval score it answers "I don't know" |

Paths are relative to the project folder, so the lab works from any directory.

- **Chat index, the "text twin" rule:** when a PDF and a `.md`/`.txt` file share a name, only the text version
  is indexed, so answers don't cite the same content twice. That's why only `hr_leave_policy.md` is indexed by
  default; the PDF copy is used by the Explorer. Uploads follow the same rule:
  - uploading a PDF whose text twin exists saves the file but doesn't index it
  - uploading a `.md`/`.txt` file replaces any indexed PDF with the same name
- **Other models:** for answers from a different model, set `OLLAMA_MODEL`, or use the hosted fallback.

---

## 9. Tests

```bash
pip install -r requirements-dev.txt
playwright install chromium
pytest                                # same as: pytest -m "not llm"
pytest -m ui                          # 66 fast tests (~30 s), mock LLM, no Ollama needed
pytest -m llm                         # 7 smoke tests against real Ollama (auto-skipped if it isn't running)
pytest -m ui --headed --slowmo 500    # watch the browser (great for a demo)
```

| File | Tests | Covers |
|---|---|---|
| `tests/test_api.py` | 21 | API-01 – API-17: health, ingest, RAG answers and sources, "I don't know", both agents' tool routing and trace shape, input validation, uploads (including the text-twin rule), calculator safety, latency, **memory** (history, session threads, isolation) |
| `tests/test_ui.py` | 19 | UI-01 – UI-18: page load, mode selector, send-button state, answers and sources, traces, multi-turn, Enter key, thinking indicator (held request), error bubble (mocked 500), upload flow (including a PDF twin), mobile layout, **memory label and New chat** |
| `tests/test_explorer.py` | 26 | EX-01 – EX-12 (parse, chunk offsets and overlaps, strategies, embed, Chroma rows, queries, isolation, bad PDFs) and UI-EX-01 – UI-EX-11 (sliders, overlap highlighting, warnings, embedding, query, compare mode, upload, **XSS safety**, navigation) |
| `tests/test_llm.py` | 7 | LLM-01 – LLM-06 with the real model: Sick Leave carry-forward 24, Paternity Leave 10, Comp Off 60, 1.5 × 8 = 12 (LangGraph), cafeteria → "don't know", and the memory follow-up in both agents |

How the tests work:

- **Isolation:** each session starts the real app with uvicorn on a free port, using a **temporary copy** of
  `sample_docs/` and a **temporary Chroma folder**, so your real data is never touched. Tests that upload files clean up after themselves.
- **Mock mode:** only the LLM is faked. Retrieval, embeddings, Chroma and tools are all real.
  - The mock routes arithmetic to the calculator and everything else to the search.
  - It answers RAG questions from the top chunk, or says "I don't know" below `MOCK_SCORE_THRESHOLD`.
  - It resolves follow-ups by searching with the previous question too.
- **Order:** tests are independent and pass in any order.
- **On failure:** pytest saves a screenshot and a Playwright trace in `test-results/`.

---

## 10. Measured results

These were measured with **`llama3.2:3b`** on an Apple-silicon laptop (CPU and GPU via Ollama), with temperature 0.

**Step 5 evaluation (10 questions):**

| Agent | Tool accuracy | Source hit | Key facts | Faithful (judge) | Avg latency |
|---|---|---|---|---|---|
| From scratch (Step 3) | 10/10 | 6/6 | 9/10 | 9/10 | ~3 s |
| LangGraph (Step 4) | 10/10 | 6/6 | 8/10 | 8/10 | ~2–3.5 s |

**Demo questions in Steps 3 and 4:**

| Demo | Result |
|---|---|
| 18% of 4520 + 75 | 888.6 ✓ |
| Sick Leave carry forward | 24 ✓ |
| Maternity Benefit Act | Wikipedia ✓ |
| Accrual March→September + encashment | ✗ (see [Known limitations](#13-known-limitations)) |

**Explorer performance:** embedding the 6-page sample PDF takes about 0.1–0.2 s. Parsing and chunking are instant.

---

## 11. Project structure

```
day5-agentic-rag-lab/
├── README.md                  # this file
├── requirements.txt           # runtime dependencies (major versions pinned)
├── requirements-dev.txt       # + pytest, pytest-playwright, requests
├── .env.example               # copy to .env
├── pytest.ini                 # markers ui / llm, chromium, traces + screenshots on failure
├── config.py                  # every setting, read from .env; paths relative to the project root
├── llm.py                     # get_llm_response(), get_chat_model(), check_llm(); ollama / openai_compatible / mock
├── sample_docs/
│   ├── hr_leave_policy.md     # the knowledge base (indexed for the chat)
│   └── hr_leave_policy.pdf    # the same content as a 6-page PDF (used by the Explorer)
├── data/chroma_db/            # created at runtime (git-ignored)
├── steps/
│   ├── step1_embeddings.py    # semantic search vs keyword search
│   ├── step2_rag.py           # ingestion + RAG, with vs without retrieval
│   ├── step3_agent_scratch.py # ReAct loop in plain Python
│   ├── step4_agent_langgraph.py
│   ├── step5_evaluate.py      # tracing + evaluation (LLM-as-judge)
│   └── step6_memory.py        # short-term memory: by hand vs checkpointer
├── rag/
│   ├── chunkers.py            # fixed / recursive / heading, with exact character offsets and warnings
│   ├── ingest.py              # load (PDF page tracking), chunk, embed, store; upload validation
│   ├── retriever.py           # retrieve(), grounded prompt, rag_answer()
│   └── projection.py          # PCA via NumPy SVD (384 -> 2 dimensions)
├── agent/
│   ├── __init__.py            # print_trace() shared by the step scripts
│   ├── tools.py               # calculator, search_documents, wikipedia_lookup, TOOL_REGISTRY (+ leave_balance template)
│   ├── react_agent.py         # from-scratch ReAct agent (memory via `history`)
│   └── graph_agent.py         # LangGraph agent (memory via checkpointer + thread_id)
├── app/
│   ├── main.py                # FastAPI: /health /ingest /upload /chat, serves the pages
│   ├── explorer.py            # /explorer/* API
│   └── static/
│       ├── index.html         # chat UI
│       └── explorer.html      # Chunking & Vector DB Explorer
├── eval/
│   ├── questions.json         # 10 questions with expected tool, source and key facts
│   └── results.json           # written by Step 5 (git-ignored)
└── tests/
    ├── conftest.py            # starts the server on temp data; fixtures and helpers
    ├── fixtures/              # wfh_policy.md (upload test), xss.md (XSS test)
    ├── test_api.py  test_ui.py  test_explorer.py  test_llm.py
```

**How a question flows:**

```
browser ──POST /chat──► app/main.py
                          ├─ rag ──────────► rag/retriever.py ─► Chroma (top-k) ─► llm.py ─► answer + sources
                          ├─ agent_scratch ► agent/react_agent.py ─┐
                          └─ agent_graph ──► agent/graph_agent.py ─┴─► agent/tools.py ─► calculator / search / Wikipedia
```

---

## 12. Mini challenge

Add your own tool to `agent/tools.py` and register it in `TOOL_REGISTRY`. It appears in both agents and in the UI
trace with no other changes. A ready-made template, `leave_balance` (3 fake employees), is commented out near the
bottom of the file:

1. Un-comment `_LEAVE_BALANCES` and `leave_balance`.
2. Add it to the registry: `for fn in [calculator, search_documents, wikipedia_lookup, leave_balance]`.
3. Ask: *"How many EL days does EMP002 have, and how many can they encash?"*

> `llama3.2:3b` often keeps calling `search_documents` instead of the new tool. Add a rule to the agent prompts
> (e.g. *"questions that mention an employee id: use leave_balance"*) or try a larger model, then compare the traces.
> Other ideas: `cgpa_calculator(grades)`, `unit_converter(query)`.

---

## 13. Known limitations

- **Small-model reasoning.** With `llama3.2:3b`, the multi-step demo (*"I joined in March… accrued by September,
  and can I encash them?"*, expected 7 × 1.5 = 10.5) usually fails in both agents. A larger model such as
  `mistral-small` solves it (search → `1.5 * 7` → 10.5).
- **The LLM judge is weak.** The same 3B model judges faithfulness. It can wave through a wrong answer, which is why
  Step 5 also checks key facts.
- **Memory is in RAM.** It is lost on restart. It is also per mode: the LangGraph agent remembers only the turns it
  answered itself, while RAG mode and the from-scratch agent use the browser's history of all turns.
- **Not tested:**
  - the `openai_compatible` provider (no API key was available during development)
  - Windows
  - Python 3.10/3.11 and the oldest dependency versions allowed by `requirements.txt`
- **Scanned PDFs** are rejected; OCR is out of scope.

---

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| **"Cannot reach Ollama"** / red dot in the chat header | Start Ollama (open the app, or run `ollama serve`). Hover the dot to see the exact message. |
| **"Model 'llama3.2:3b' is not pulled"** | `ollama pull llama3.2:3b`, or set `OLLAMA_MODEL` in `.env` to a model you have (`ollama list`). |
| **First run is slow** | The embedding model (~90 MB) downloads once; later runs are fast. Ignore the "unauthenticated requests to the HF Hub" warning. |
| **Wikipedia tool says it is unavailable** | You are offline or Wikipedia is blocked. The agent keeps working; avoid general-knowledge questions. |
| **Windows path or activation problems** | Run commands from the project folder, use `.venv\Scripts\activate`, and use `copy` instead of `cp`. All paths in the code are relative. |
| **Badly formatted ReAct output, loops, or wrong multi-step maths** | Small (3B) models struggle with long multi-step questions. The agent asks the model to fix its format, skips repeated identical tool calls, and stops after `MAX_AGENT_STEPS`. For better results use a larger model (`OLLAMA_MODEL=qwen2.5:7b`, `mistral-small`, …) or the hosted fallback: set `LLM_PROVIDER=openai_compatible` plus `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`, `OPENAI_COMPAT_MODEL`. |
| **Step 5 "faithful" column looks wrong** | The judge is the same small model and can be too strict or too lenient. Compare with the "facts" column; a larger judge model gives better verdicts. |
| **Answers stop mid-sentence** | Replies are capped at `MAX_OUTPUT_TOKENS` (default 768) so a looping model can't run forever. Raise it in `.env`. |
| **The agent "forgot" the conversation** | Memory is kept in RAM only: **＋ New chat**, reloading the page (new session) or restarting the server clears it. |
| **"ℹ … is already indexed as text" after uploading a PDF** | Expected: that PDF has a `.md`/`.txt` twin with the same name, which is already indexed. The PDF is saved but not indexed, so answers don't cite the same text twice. |
| **Chat answers changed after Step 2 with custom flags** | Step 2's `--chunk-size/--chunk-overlap` re-index the chat. Run `python steps/step2_rag.py` once without flags to restore the defaults. |
| **"This PDF has no text layer"** | The PDF is a scanned image. It would need OCR, which this lab doesn't include. |
| **Port 8000 is already in use** | `uvicorn app.main:app --reload --port 8001` |
| **Playwright: "Executable doesn't exist"** | Run `playwright install chromium` inside the activated virtual environment. |
| **Explorer charts or styles missing** | The pages load Tailwind and Chart.js from CDNs, so the browser needs internet access. |

---

## 15. Suggested lab timing (instructors)

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

Step 6 (memory, ~15 min) is an add-on. Run it after Step 5 if time allows, or set it as homework.

---

*Nexora Technologies is fictional; the HR policy exists only for training and demonstration.*
