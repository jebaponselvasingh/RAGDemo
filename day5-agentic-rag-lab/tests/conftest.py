"""
tests/conftest.py — shared fixtures for the end-to-end tests.

Each test session starts the real FastAPI app with uvicorn in a subprocess,
pointed at a TEMPORARY copy of sample_docs/ and a TEMPORARY Chroma folder, so
the tests never touch your real index or documents.

  server       mock LLM (deterministic)   used by tests marked `ui`
  llm_server   real Ollama                used by tests marked `llm` (skipped if Ollama is down)
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(PROJECT_ROOT))

UI_TIMEOUT_MS = 15_000
LLM_TIMEOUT_S = 120


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """A uvicorn subprocess running app.main:app on a temporary copy of the data."""

    def __init__(self, provider: str):
        self.tmp = Path(tempfile.mkdtemp(prefix="day5-lab-tests-"))
        self.docs_dir = self.tmp / "docs"
        shutil.copytree(PROJECT_ROOT / "sample_docs", self.docs_dir)
        self.original_docs = {p.name for p in self.docs_dir.iterdir()}
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = {**os.environ, "LLM_PROVIDER": provider, "CHROMA_PATH": str(self.tmp / "chroma"),
               "DOCS_PATH": str(self.docs_dir), "TOKENIZERS_PARALLELISM": "false"}
        self.log = open(self.tmp / "server.log", "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=PROJECT_ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT,
        )

    def wait_until_ready(self, timeout: float = 90) -> None:
        """Poll /health. The first run may download the embedding model, hence 90 s."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                if requests.get(f"{self.url}/health", timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"Server did not start. Log:\n{(self.tmp / 'server.log').read_text()[-3000:]}")

    def reset_documents(self) -> None:
        """Remove any uploaded files and rebuild the index (keeps tests independent)."""
        removed = False
        for path in self.docs_dir.iterdir():
            if path.name not in self.original_docs:
                path.unlink()
                removed = True
        if removed:
            requests.post(f"{self.url}/ingest", json={"rebuild": True}, timeout=120).raise_for_status()

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Servers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def server():
    srv = Server(provider="mock")
    srv.wait_until_ready()
    yield srv
    srv.stop()


@pytest.fixture(scope="session")
def server_url(server) -> str:
    return server.url


@pytest.fixture(scope="session")
def base_url(server_url) -> str:
    """Override pytest-playwright's base_url so page.goto("/") hits our test server."""
    return server_url


@pytest.fixture
def clean_docs(server):
    """Use in tests that upload files: removes the uploads afterwards."""
    yield
    server.reset_documents()


def _ollama_ready() -> tuple[bool, str]:
    import config

    try:
        tags = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).json()
    except requests.RequestException:
        return False, f"Ollama is not reachable at {config.OLLAMA_BASE_URL}"
    names = {m["name"] for m in tags.get("models", [])}
    if config.OLLAMA_MODEL not in names and f"{config.OLLAMA_MODEL}:latest" not in names:
        return False, f"Model {config.OLLAMA_MODEL} is not pulled (ollama pull {config.OLLAMA_MODEL})"
    return True, ""


@pytest.fixture(scope="session")
def llm_server():
    ok, reason = _ollama_ready()
    if not ok:
        pytest.skip(reason)
    srv = Server(provider="ollama")
    srv.wait_until_ready()
    yield srv
    srv.stop()


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------
def tid(name: str) -> str:
    """CSS selector for a data-testid."""
    return f'[data-testid="{name}"]'


@pytest.fixture(autouse=True)
def _ui_timeouts(request):
    """15 s default timeout for every browser test."""
    if "page" in request.fixturenames:
        request.getfixturevalue("page").set_default_timeout(UI_TIMEOUT_MS)


@pytest.fixture
def chat_page(page):
    page.goto("/")
    page.wait_for_selector(f'{tid("health-indicator")}[data-status="ok"]')
    return page


def ask(page, question: str, mode: str = "rag") -> None:
    """Select the mode, type, send, and wait for the new answer (or error) to appear."""
    page.select_option(tid("mode-select"), mode)
    before = page.locator(tid("assistant-message")).count()
    errors_before = page.locator(tid("error-message")).count()
    page.fill(tid("chat-input"), question)
    page.click(tid("send-button"))
    page.wait_for_selector(tid("thinking-indicator"), state="detached")
    page.wait_for_function(
        "([n, e]) => document.querySelectorAll('[data-testid=assistant-message]').length > n"
        " || document.querySelectorAll('[data-testid=error-message]').length > e",
        arg=[before, errors_before],
    )
