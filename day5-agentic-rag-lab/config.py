"""
config.py — one place for every setting in the lab.

Concept taught: keep configuration (model names, paths, chunk sizes) OUT of
your code. Values come from the `.env` file, and real environment variables
win over `.env`, which is how the tests point the app at a temporary database.

All paths are resolved relative to the project root, so the lab works no
matter which folder you run a script from (and on Windows too).
"""
from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _path(value: str) -> Path:
    """Turn a relative path from .env into an absolute path under the project root."""
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --- LLM -------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

OPENAI_COMPAT_BASE_URL: str = os.getenv("OPENAI_COMPAT_BASE_URL", "").rstrip("/")
OPENAI_COMPAT_API_KEY: str = os.getenv("OPENAI_COMPAT_API_KEY", "")
OPENAI_COMPAT_MODEL: str = os.getenv("OPENAI_COMPAT_MODEL", "")

# Upper limit on the length of one LLM reply (stops runaway generations on small models).
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "768"))

# Mock LLM only: below this retrieval score the mock answers "I don't know".
MOCK_SCORE_THRESHOLD: float = float(os.getenv("MOCK_SCORE_THRESHOLD", "0.3"))

# --- Embeddings & vector store ---------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH: Path = _path(os.getenv("CHROMA_PATH", "data/chroma_db"))
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "course_notes")
DOCS_PATH: Path = _path(os.getenv("DOCS_PATH", "sample_docs"))

# --- Chunking & retrieval --------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "recursive")
TOP_K: int = int(os.getenv("TOP_K", "4"))

# --- Agents ----------------------------------------------------------------
MAX_AGENT_STEPS: int = int(os.getenv("MAX_AGENT_STEPS", "6"))
# Short-term memory: how many earlier chat messages an agent may see (Step 6).
MEMORY_MAX_MESSAGES: int = int(os.getenv("MEMORY_MAX_MESSAGES", "10"))


def active_model_name() -> str:
    """Human-readable name of the model currently in use (shown in /health)."""
    if LLM_PROVIDER == "ollama":
        return OLLAMA_MODEL
    if LLM_PROVIDER == "openai_compatible":
        return OPENAI_COMPAT_MODEL
    return "mock"
