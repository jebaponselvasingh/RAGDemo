"""
rag/ingest.py — the "indexing" half of RAG:  load → chunk → embed → store.

Concept taught: before an LLM can answer from your documents, the documents
must be turned into small pieces (chunks), each converted into a vector
(embedding) and saved in a vector database (ChromaDB) together with metadata
saying where it came from.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

if __package__ in (None, ""):  # allow `python rag/ingest.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.chunkers import chunk_document

SUPPORTED = {".md", ".txt", ".pdf"}


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def _clean_pdf_text(text: str) -> str:
    """PDF bullet glyphs often come out as control characters (e.g. '\x7f'). Show them as '•'.
    One character is swapped for one character, so text length and offsets don't change."""
    text = "".join("•" if (ord(ch) < 32 and ch not in "\n\t") or ord(ch) == 127 else ch for ch in text)
    return text.replace("•\n", "• ")  # pypdf puts each bullet on its own line; rejoin it with its text


def parse_file(path: Path) -> dict:
    """Read one file. Returns {text, source, pages: [{page, text, chars, start}]}.

    For PDFs we remember where each page starts (a character offset) so every
    chunk can later report which page it came from.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        page_texts = [_clean_pdf_text(p.extract_text() or "") for p in PdfReader(str(path)).pages]
    else:
        page_texts = [path.read_text(encoding="utf-8", errors="replace")]

    pages, parts, offset = [], [], 0
    for number, page_text in enumerate(page_texts, start=1):
        pages.append({"page": number, "text": page_text, "chars": len(page_text), "start": offset})
        parts.append(page_text)
        offset += len(page_text) + 2  # +2 for the "\n\n" joining pages
    return {"text": "\n\n".join(parts), "source": path.name, "pages": pages}


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def parse_upload(filename: str, data: bytes) -> dict:
    """Validate and parse an uploaded file. Raises ValueError with a friendly message.

    Used by both /upload (chat) and /explorer/parse, so users get the same rules everywhere.
    """
    import tempfile

    name = Path(filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{suffix or name}'. Please upload a PDF, .md or .txt file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File is larger than 10 MB. Please upload a smaller file.")
    if not data.strip():
        raise ValueError("The file is empty.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(data)
        try:
            doc = parse_file(path)
        except Exception as exc:
            raise ValueError(f"Could not read '{name}'. Is it a valid {suffix[1:].upper()} file? ({type(exc).__name__})")
    if suffix == ".pdf" and len(doc["text"].strip()) < 20 * max(1, len(doc["pages"])):
        raise ValueError("This PDF has no text layer — it would need OCR (e.g. a scanned document).")
    if not doc["text"].strip():
        raise ValueError("No text could be extracted from this file.")
    return doc


def load_documents(folder: Path | str = config.DOCS_PATH, skip_duplicate_pdfs: bool = True) -> list[dict]:
    """Load every .md / .txt / .pdf in `folder` as {text, source, pages}.

    A PDF is skipped when a .md/.txt with the same name exists (our sample policy
    ships in both formats and we don't want every answer twice).
    """
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
    text_stems = {p.stem for p in files if p.suffix.lower() in {".md", ".txt"}}
    docs = []
    for path in files:
        if skip_duplicate_pdfs and path.suffix.lower() == ".pdf" and path.stem in text_stems:
            continue
        doc = parse_file(path)
        if doc["text"].strip():
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# 2. CHUNK
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP,
               strategy: str = config.CHUNK_STRATEGY, page_starts: list[int] | None = None) -> list[dict]:
    """Sliding-window chunking that prefers paragraph/sentence boundaries (see rag/chunkers.py)."""
    return chunk_document(text, strategy=strategy, size=size, overlap=overlap, page_starts=page_starts)


# ---------------------------------------------------------------------------
# 3. EMBED
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embedder():
    """Load the sentence-transformers model once (first run downloads ~90 MB)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")


def embed(texts: list[str]) -> list[list[float]]:
    """Turn texts into 384-dim vectors. normalize=True makes cosine similarity a simple dot product."""
    vectors = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ---------------------------------------------------------------------------
# 4. STORE (ChromaDB)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def get_client(path: str = str(config.CHROMA_PATH)):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))


def get_collection(name: str = config.COLLECTION_NAME):
    """Open (or create) a collection that compares vectors with cosine distance."""
    return get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def indexed_chunk_count() -> int:
    return get_collection().count()


def _records(doc: dict, size: int, overlap: int, strategy: str) -> tuple[list, list, list]:
    """Chunk one document into the (ids, texts, metadatas) lists that Chroma stores."""
    chunk_config = f"{strategy}/{size}/{overlap}"
    ids, texts, metadatas = [], [], []
    page_starts = [p["start"] for p in doc["pages"]]
    for chunk in chunk_text(doc["text"], size, overlap, strategy, page_starts):
        if not chunk["text"].strip():
            continue
        ids.append(f"{doc['source']}::{chunk['id']}")
        texts.append(chunk["text"])
        metadatas.append({"source": doc["source"], "chunk_id": chunk["id"], "page": chunk["page"],
                          "section": chunk["section"], "chunk_config": chunk_config})
    return ids, texts, metadatas


def text_twin(path: Path) -> Path | None:
    """For a PDF, the .md/.txt file with the same name in the same folder (if any).

    The same rule as load_documents(): when both exist, only the text version is indexed,
    so answers don't cite the same content twice.
    """
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return None
    for suffix in (".md", ".txt"):
        twin = path.with_suffix(suffix)
        if twin.exists():
            return twin
    return None


def index_file(path: Path, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP,
               strategy: str = config.CHUNK_STRATEGY) -> dict:
    """Add (or replace) ONE file in the index without re-embedding everything.

    Returns {"chunks": n, "skipped_for": twin filename or None}.
    """
    path = Path(path)
    collection = get_collection()
    twin = text_twin(path)
    if twin is not None:  # e.g. hr_leave_policy.pdf while hr_leave_policy.md is indexed
        collection.delete(where={"source": path.name})
        return {"chunks": 0, "skipped_for": twin.name}
    if path.suffix.lower() in (".md", ".txt"):
        # A text version replaces its PDF twin, exactly as a full rebuild would.
        collection.delete(where={"source": path.with_suffix(".pdf").name})

    doc = parse_file(path)
    collection.delete(where={"source": doc["source"]})  # replace an older version of the same file
    ids, texts, metadatas = _records(doc, size, overlap, strategy)
    if texts:
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embed(texts))
    return {"chunks": len(texts), "skipped_for": None}


def build_index(rebuild: bool = False, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP,
                strategy: str = config.CHUNK_STRATEGY, verbose: bool = True) -> dict:
    """Load, chunk, embed and store every document. Returns {files, chunks}.

    If the collection already has data built with the SAME chunk settings we skip
    the (slow) work unless rebuild=True. Different settings trigger a re-index.
    """
    collection = get_collection()
    chunk_config = f"{strategy}/{size}/{overlap}"
    if collection.count() > 0 and not rebuild:
        stored = collection.get(limit=1, include=["metadatas"])["metadatas"][0].get("chunk_config")
        if stored != chunk_config:
            if verbose:
                print(f"Chunk settings changed ({stored} -> {chunk_config}); re-indexing...")
            rebuild = True
    if collection.count() > 0 and not rebuild:
        if verbose:
            print(f"Index already has {collection.count()} chunks (use rebuild=True to re-index).")
        return {"files": len({m["source"] for m in collection.get(include=["metadatas"])["metadatas"]}),
                "chunks": collection.count()}

    if rebuild:
        get_client().delete_collection(config.COLLECTION_NAME)
        collection = get_collection()

    docs = load_documents()
    ids, texts, metadatas = [], [], []
    for doc in docs:
        doc_ids, doc_texts, doc_metas = _records(doc, size, overlap, strategy)
        ids += doc_ids
        texts += doc_texts
        metadatas += doc_metas

    if texts:
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embed(texts))

    if verbose:
        avg = sum(len(t) for t in texts) / max(1, len(texts))
        print(f"Indexed {len(docs)} file(s) -> {len(texts)} chunks (avg {avg:.0f} chars, "
              f"strategy={strategy}, size={size}, overlap={overlap})")
    return {"files": len(docs), "chunks": len(texts)}


def preview_chunks(n: int = 5, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP,
                   strategy: str = config.CHUNK_STRATEGY) -> None:
    """Print the first `n` chunks with their boundaries marked, to SEE what chunking does."""
    shown = 0
    for doc in load_documents():
        for i, chunk in enumerate(chunk_text(doc["text"], size, overlap, strategy)):
            if shown >= n:
                return
            print(f"---- chunk {i + 1} [{doc['source']}] ({chunk['length']} chars) ----")
            print(chunk["text"].strip())
            shown += 1
    print("---- end of preview ----")


if __name__ == "__main__":
    build_index(rebuild=True)
    preview_chunks()
