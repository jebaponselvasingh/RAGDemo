"""
app/main.py — the capstone: expose RAG and both agents through a web API.

Concept taught: an AI feature becomes a product when it sits behind a normal
HTTP endpoint. The browser (app/static/index.html) only ever talks to /chat;
it doesn't know or care which LLM, vector DB or agent framework is behind it.

Run:  uvicorn app.main:app --reload     then open http://localhost:8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

import config
from llm import LLMError, check_llm
from rag.ingest import build_index, index_file, indexed_chunk_count, parse_upload
from rag.retriever import rag_answer, retrieve

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()  # builds only if the collection is empty (or chunk settings changed)
    yield


app = FastAPI(title="Ask My Documents — Day 5 Agentic AI Lab", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from app.explorer import router as explorer_router  # noqa: E402  (imported after `app` on purpose)

app.include_router(explorer_router)


# ---------------------------------------------------------------------------
# Friendly errors: always JSON, never a stack trace in the browser
# ---------------------------------------------------------------------------
@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    message = f"{type(exc).__name__}: {exc}"
    if "connect" in message.lower():  # e.g. LangChain can't reach Ollama
        message = f"Cannot reach the LLM ({config.LLM_PROVIDER}). Is it running? Details: {message}"
    return JSONResponse(status_code=500, content={"error": message})


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    rebuild: bool = False


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    mode: Literal["rag", "agent_scratch", "agent_graph"] = "rag"
    history: list[dict] = []                     # earlier turns, kept by the browser (RAG + scratch agent)
    session_id: str | None = Field(None, max_length=100)  # conversation id for the LangGraph checkpointer

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty")
        return value.strip()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def chat_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/explorer", include_in_schema=False)
def explorer_page():
    return FileResponse(STATIC_DIR / "explorer.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    llm_ok, message = check_llm()
    return {
        "status": "ok" if llm_ok else "error",
        "message": message,
        "llm_provider": config.LLM_PROVIDER,
        "model": config.active_model_name(),
        "indexed_chunks": indexed_chunk_count(),
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    return build_index(rebuild=request.rebuild, verbose=False)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        parse_upload(file.filename, data)  # validates type, size and text content
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    target = config.DOCS_PATH / Path(file.filename).name
    target.write_bytes(data)
    result = index_file(target)
    response = {"filename": target.name, "chunks": result["chunks"], "indexed_chunks": indexed_chunk_count(),
                "skipped": result["skipped_for"] is not None}
    if response["skipped"]:
        response["message"] = (f"Saved, but not indexed: its text version {result['skipped_for']} is already "
                               "in the index, and indexing both would make every answer cite it twice.")
    return response


def _sources_from_hits(hits: list[dict]) -> list[dict]:
    seen, sources = set(), []
    for h in hits:
        key = (h["source"], h.get("chunk_id"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": h["source"], "chunk_id": h.get("chunk_id"),
                            "score": h["score"], "text": h["text"]})
    return sources


def _agent_sources(trace: list[dict]) -> list[dict]:
    """Which document chunks did the agent's search_documents calls retrieve?"""
    hits = []
    for step in trace:
        if step["action"] == "search_documents" and step["action_input"]:
            hits.extend(retrieve(step["action_input"]))
    return _sources_from_hits(hits)


@app.post("/chat")
def chat(request: ChatRequest):
    started = time.perf_counter()
    memory_messages = 0
    if request.mode == "rag":
        result = rag_answer(request.message, history=request.history)
        answer, trace, sources = result["answer"], [], _sources_from_hits(result["hits"])
    else:
        # Two styles of short-term memory (Step 6):
        #   agent_scratch: the browser sends the history and we pass it in by hand
        #   agent_graph  : the server keeps the conversation in LangGraph's checkpointer, by session_id
        if request.mode == "agent_scratch":
            from agent.react_agent import run_react_agent

            result = run_react_agent(request.message, history=request.history)
        else:
            from agent.graph_agent import run_graph_agent

            result = run_graph_agent(request.message, thread_id=request.session_id)
        answer, trace, sources = result["answer"], result["trace"], _agent_sources(result["trace"])
        memory_messages = result.get("memory_messages", 0)
    latency_ms = max(1, round((time.perf_counter() - started) * 1000))
    return {"answer": answer, "sources": sources, "trace": trace, "latency_ms": latency_ms, "mode": request.mode,
            "memory_messages": memory_messages}
