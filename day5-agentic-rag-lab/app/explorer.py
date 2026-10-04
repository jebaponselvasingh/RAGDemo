"""
app/explorer.py — API behind the Chunking & Vector DB Explorer (/explorer).

Concept taught: open up the RAG pipeline and LOOK inside every stage:
    parse  -> raw text extracted from the file (what the computer actually "sees")
    chunk  -> where the text is cut (instant, no embedding — perfect for sliders)
    embed  -> the vectors, and the exact rows ChromaDB stores
    query  -> which chunks are nearest to a question, and the prompt RAG would build

The Explorer uses its OWN Chroma collections (explorer_<config hash>), so playing
here never changes the index used by the chat assistant.
"""
from __future__ import annotations

import hashlib
import time
from typing import Literal

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

import config
from rag import projection
from rag.chunkers import chunk_document, chunk_stats
from rag.ingest import embed, get_client, parse_file, parse_upload
from rag.retriever import prompt_preview

router = APIRouter(prefix="/explorer", tags=["explorer"])

PREVIEW_DIMS = 16
SAMPLE_PDF = "hr_leave_policy.pdf"

# Parsed documents live in memory: doc_id -> {filename, full_text, pages}
_DOCS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ChunkConfig(BaseModel):
    # Explorer defaults: heading / 800 / 100 keeps each policy section whole, so a query like
    # "days off for my wedding?" finds the chunk titled "10. Marriage Leave". (The chat index
    # uses recursive / 500 / 100 from .env — try that here to watch the heading get cut off.)
    strategy: Literal["fixed", "recursive", "heading"] = "heading"
    size: int = Field(800, ge=50, le=5000)
    overlap: int = Field(100, ge=0, le=2000)

    @model_validator(mode="after")
    def overlap_smaller_than_size(self):
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        return self


class ChunkRequest(ChunkConfig):
    doc_id: str


class QueryRequest(BaseModel):
    collection: str
    query: str = Field(..., min_length=1)
    k: int = Field(4, ge=1, le=20)


class CompareRequest(BaseModel):
    doc_id: str
    query: str = Field(..., min_length=1)
    k: int = Field(4, ge=1, le=20)
    configs: list[ChunkConfig] = Field(..., min_length=2, max_length=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _store(doc: dict) -> dict:
    """Keep a parsed document in memory and return the /parse response."""
    doc_id = hashlib.sha1(doc["text"].encode("utf-8")).hexdigest()[:12]
    _DOCS[doc_id] = {"filename": doc["source"], "full_text": doc["text"], "pages": doc["pages"]}
    return {
        "doc_id": doc_id,
        "filename": doc["source"],
        "pages": [{"page": p["page"], "text": p["text"], "chars": p["chars"], "start": p["start"]} for p in doc["pages"]],
        "total_chars": len(doc["text"]),
        "full_text": doc["text"],
    }


def _doc(doc_id: str) -> dict:
    if doc_id not in _DOCS:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Parse the document again (the server may have restarted).")
    return _DOCS[doc_id]


def _chunks(doc: dict, cfg: ChunkConfig) -> list[dict]:
    page_starts = [p["start"] for p in doc["pages"]]
    return [c for c in chunk_document(doc["full_text"], cfg.strategy, cfg.size, cfg.overlap, page_starts)
            if c["text"].strip()]


def _collection_name(doc_id: str, cfg: ChunkConfig) -> str:
    key = f"{doc_id}|{cfg.strategy}|{cfg.size}|{cfg.overlap}"
    return "explorer_" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _get_collection(name: str):
    try:
        return get_client().get_collection(name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found. Embed the chunks first.")


def _all_rows(collection) -> dict:
    return collection.get(include=["documents", "metadatas", "embeddings"])


def _ensure_projection(name: str, collection) -> None:
    """PCA components live in memory; re-fit them if the server restarted."""
    if not projection.has_model(name):
        rows = _all_rows(collection)
        projection.fit(name, rows["embeddings"])


def _embed_and_store(doc_id: str, cfg: ChunkConfig) -> dict:
    """Embed the chunks for one config into its own collection (reused if it already exists)."""
    doc = _doc(doc_id)
    chunks = _chunks(doc, cfg)
    name = _collection_name(doc_id, cfg)
    collection = get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    embed_ms = 0
    if collection.count() != len(chunks):
        if collection.count():
            get_client().delete_collection(name)
            collection = get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        started = time.perf_counter()
        vectors = embed([c["text"] for c in chunks])
        embed_ms = round((time.perf_counter() - started) * 1000)
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=vectors,
            metadatas=[{"source": doc["filename"], "chunk_id": c["id"], "page": c["page"], "section": c["section"],
                        "start": c["start"], "end": c["end"], "length": c["length"], "strategy": cfg.strategy,
                        "size": cfg.size, "overlap": cfg.overlap} for c in chunks],
        )
    rows = _all_rows(collection)
    points = projection.fit(name, rows["embeddings"])
    return {"name": name, "rows": rows, "points": points, "embed_ms": embed_ms}


def _run_query(name: str, query: str, k: int) -> dict:
    collection = _get_collection(name)
    _ensure_projection(name, collection)
    q_vec = embed([query])[0]
    rows = _all_rows(collection)

    # Vectors are normalised, so cosine similarity is just a dot product.
    scores = np.asarray(rows["embeddings"]) @ np.asarray(q_vec)
    order = np.argsort(-scores)
    results = []
    for rank, i in enumerate(order[:k], start=1):
        meta = rows["metadatas"][i]
        results.append({"id": rows["ids"][i], "rank": rank, "score": round(float(scores[i]), 4),
                        "distance": round(1 - float(scores[i]), 4), "text": rows["documents"][i],
                        "section": meta.get("section"), "page": meta.get("page"), "source": meta.get("source")})
    qx, qy = projection.project(name, [q_vec])[0]
    hits = [{"text": r["text"], "source": r["source"] or "document"} for r in results]
    return {
        "collection": name,
        "query_embedding_preview": [round(v, 4) for v in q_vec[:PREVIEW_DIMS]],
        "query_point": {"x": qx, "y": qy},
        "results": results,
        "all_scores": [{"id": rows["ids"][i], "score": round(float(scores[i]), 4)} for i in order],
        "prompt_preview": prompt_preview(query, hits),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/parse")
async def parse(file: UploadFile = File(...)):
    data = await file.read()
    try:
        doc = parse_upload(file.filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _store(doc)


@router.get("/sample")
def sample():
    for folder in (config.DOCS_PATH, config.PROJECT_ROOT / "sample_docs"):
        path = folder / SAMPLE_PDF
        if path.exists():
            return _store(parse_file(path))
    raise HTTPException(status_code=404, detail=f"{SAMPLE_PDF} not found in sample_docs/.")


@router.post("/chunk")
def chunk(request: ChunkRequest):
    chunks = _chunks(_doc(request.doc_id), request)
    return {
        "chunks": chunks,
        "stats": chunk_stats(chunks),
        "warnings_count": sum(len(c["warnings"]) for c in chunks),
        "collection": _collection_name(request.doc_id, request),
    }


@router.post("/embed")
def embed_chunks(request: ChunkRequest):
    out = _embed_and_store(request.doc_id, request)
    rows = out["rows"]
    records = []
    for cid, doc, meta, vec in zip(rows["ids"], rows["documents"], rows["metadatas"], rows["embeddings"]):
        records.append({"id": cid, "document": doc, "metadata": meta,
                        "embedding_preview": [round(float(v), 4) for v in vec[:PREVIEW_DIMS]],
                        "norm": round(float(np.linalg.norm(vec)), 4)})
    points = [{"id": cid, "x": x, "y": y, "section": meta.get("section"), "text": doc[:100]}
              for cid, (x, y), meta, doc in zip(rows["ids"], out["points"], rows["metadatas"], rows["documents"])]
    dims = len(rows["embeddings"][0]) if len(rows["embeddings"]) else 0
    return {"collection": out["name"], "count": len(records), "dims": dims, "embed_ms": out["embed_ms"],
            "records": records, "points": points}


@router.get("/collection/{name}")
def collection_rows(name: str, limit: int = 20, offset: int = 0):
    """Rows exactly as Chroma stores them (embeddings truncated so the response stays small)."""
    collection = _get_collection(name)
    limit = max(1, min(limit, 200))
    rows = collection.get(limit=limit, offset=max(0, offset), include=["documents", "metadatas", "embeddings"])
    return {
        "name": name,
        "count": collection.count(),
        "limit": limit,
        "offset": offset,
        "ids": rows["ids"],
        "documents": rows["documents"],
        "metadatas": rows["metadatas"],
        "embeddings": [[round(float(v), 4) for v in vec[:PREVIEW_DIMS]] for vec in rows["embeddings"]],
    }


@router.get("/vector/{name}/{chunk_id}")
def full_vector(name: str, chunk_id: str):
    """All 384 numbers of one stored embedding (for the heat-grid view)."""
    rows = _get_collection(name).get(ids=[chunk_id], include=["embeddings"])
    if not rows["ids"]:
        raise HTTPException(status_code=404, detail=f"No chunk '{chunk_id}' in '{name}'.")
    return {"id": chunk_id, "embedding": [round(float(v), 4) for v in rows["embeddings"][0]]}


@router.post("/query")
def query(request: QueryRequest):
    return _run_query(request.collection, request.query, request.k)


@router.post("/compare")
def compare(request: CompareRequest):
    sides = []
    for cfg in request.configs:
        name = _embed_and_store(request.doc_id, cfg)["name"]
        result = _run_query(name, request.query, request.k)
        sides.append({"config": cfg.model_dump(), "collection": name, "count": _get_collection(name).count(),
                      "results": result["results"]})
    return {"query": request.query, "left": sides[0], "right": sides[1]}


@router.delete("/collection/{name}")
def delete_collection(name: str):
    if not name.startswith("explorer_"):
        raise HTTPException(status_code=400, detail="Only Explorer collections (explorer_*) can be deleted here.")
    _get_collection(name)
    get_client().delete_collection(name)
    projection.forget(name)
    return {"deleted": True}
