"""
rag/retriever.py — the "retrieval + generation" half of RAG.

Concept taught: to answer a question we (1) embed the question, (2) ask the
vector database for the nearest chunks, and (3) paste those chunks into the
prompt — that pasting is the "Augmented" in Retrieval-Augmented Generation.
"""
from __future__ import annotations

import config
from rag.ingest import embed, get_collection

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer only from the context. "
    "If the answer is not in the context, say you don't know. "
    "Cite sources as [filename]: end every answer with the file name of the chunk you used "
    "in square brackets, for example: Casual Leave is 12 days per year [policy.md]."
)


def retrieve(query: str, k: int = config.TOP_K) -> list[dict]:
    """Return the k most similar chunks as [{text, source, score, chunk_id}], best first.

    Chroma returns cosine *distance*; similarity = 1 - distance (1.0 = same meaning).
    """
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(query_embeddings=embed([query]), n_results=min(k, collection.count()))
    hits = []
    for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": text, "source": meta["source"], "score": round(1 - dist, 4),
                     "chunk_id": meta.get("chunk_id"), "section": meta.get("section")})
    return hits


def build_context(hits: list[dict]) -> str:
    """Number the retrieved chunks so the model (and students) can see each one."""
    return "\n\n".join(f"--- Chunk {i} (filename: {h['source']}) ---\n{h['text'].strip()}" for i, h in enumerate(hits, start=1))


def build_rag_messages(question: str, hits: list[dict], history: list[dict] | None = None) -> list[dict]:
    """The exact grounded prompt we send to the LLM.

    `history` (earlier chat turns) sits between the system prompt and the new question,
    so follow-ups like "and for Sick Leave?" make sense. Only the last 6 turns are kept.
    """
    user = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
    from llm import clean_history

    past = clean_history(history, limit=6)
    return [{"role": "system", "content": SYSTEM_PROMPT}, *past, {"role": "user", "content": user}]


def prompt_preview(question: str, hits: list[dict]) -> str:
    """The grounded prompt as one readable string (shown in the Explorer)."""
    return "\n\n".join(f"### {m['role'].upper()}\n{m['content']}" for m in build_rag_messages(question, hits))


def rag_answer(question: str, k: int = config.TOP_K, history: list[dict] | None = None) -> dict:
    """Retrieve, then generate. Returns {answer, sources, hits}."""
    from llm import get_llm_response

    # Follow-ups like "Can it be split?" don't name their topic, so we search with the
    # previous user question too. (Real systems often ask the LLM to rewrite the query.)
    previous = [m["content"] for m in (history or []) if m.get("role") == "user"]
    search_query = f"{previous[-1]} {question}" if previous else question
    hits = retrieve(search_query, k)
    answer = get_llm_response(build_rag_messages(question, hits, history))
    sources = list(dict.fromkeys(h["source"] for h in hits))
    return {"answer": answer, "sources": sources, "hits": hits}
