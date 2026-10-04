"""
API-level end-to-end tests (README §8.3). The LLM is mocked; retrieval, embeddings,
Chroma and the tools are all real.
"""
import pytest
import requests

from conftest import FIXTURES, PROJECT_ROOT

pytestmark = pytest.mark.ui


def chat(url, message, mode="rag", **extra):
    return requests.post(f"{url}/chat", json={"message": message, "mode": mode, **extra}, timeout=60)


def actions(body):
    return [step["action"] for step in body["trace"]]


def test_api_01_health(server_url):
    r = requests.get(f"{server_url}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 0


def test_api_02_ingest_rebuild(server_url):
    body = requests.post(f"{server_url}/ingest", json={"rebuild": True}, timeout=120).json()
    assert body["files"] >= 1
    assert body["chunks"] >= 5
    assert requests.get(f"{server_url}/health").json()["indexed_chunks"] == body["chunks"]


def test_api_03_rag_answer_with_sources(server_url):
    body = chat(server_url, "How many days of Casual Leave per year?").json()
    assert "hr_leave_policy.md" in [s["source"] for s in body["sources"]]
    assert any("12" in s["text"] for s in body["sources"])


def test_api_04_rag_off_topic(server_url):
    body = chat(server_url, "What is the cafeteria menu?").json()
    assert "don't know" in body["answer"]


def test_api_05_scratch_agent_calculator(server_url):
    body = chat(server_url, "What is 18% of 4520 plus 75?", "agent_scratch").json()
    assert "calculator" in actions(body)
    assert "888.6" in body["answer"]


def test_api_06_scratch_agent_search(server_url):
    body = chat(server_url, "How long is Paternity Leave?", "agent_scratch").json()
    assert "search_documents" in actions(body)


def test_api_07_graph_agent_same_trace_shape(server_url):
    calc = chat(server_url, "What is 18% of 4520 plus 75?", "agent_graph").json()
    assert "calculator" in actions(calc)
    assert "888.6" in calc["answer"]
    search = chat(server_url, "How long is Paternity Leave?", "agent_graph").json()
    assert "search_documents" in actions(search)
    scratch = chat(server_url, "How long is Paternity Leave?", "agent_scratch").json()
    assert set(search["trace"][0]) == set(scratch["trace"][0]) == {"step", "thought", "action", "action_input", "observation"}


def test_api_08_empty_message(server_url):
    r = chat(server_url, "   ")
    assert r.status_code in (400, 422)
    assert r.headers["content-type"].startswith("application/json")


def test_api_09_invalid_mode(server_url):
    assert chat(server_url, "hello", "not_a_mode").status_code == 422


def test_api_10_upload_increases_chunks(server_url, clean_docs):
    before = requests.get(f"{server_url}/health").json()["indexed_chunks"]
    with open(FIXTURES / "wfh_policy.md", "rb") as f:
        r = requests.post(f"{server_url}/upload", files={"file": ("wfh_policy.md", f, "text/markdown")}, timeout=60)
    assert r.status_code == 200
    assert r.json()["filename"] == "wfh_policy.md"
    assert requests.get(f"{server_url}/health").json()["indexed_chunks"] > before


def test_api_11_calculator_is_safe(server_url):
    body = chat(server_url, "Calculate __import__('os').system('ls')", "agent_scratch").json()
    calc_steps = [s for s in body["trace"] if s["action"] == "calculator"]
    assert calc_steps, "the calculator should have been called"
    assert calc_steps[0]["observation"].startswith("Error")
    assert requests.get(f"{server_url}/health").json()["status"] == "ok"


@pytest.mark.parametrize("mode", ["rag", "agent_scratch", "agent_graph"])
def test_api_12_latency_reported(server_url, mode):
    body = chat(server_url, "How long is Paternity Leave?", mode).json()
    assert body["latency_ms"] > 0


def test_upload_rejects_unsupported_file(server_url):
    r = requests.post(f"{server_url}/upload", files={"file": ("notes.docx", b"hello", "application/octet-stream")})
    assert r.status_code == 400
    assert "Unsupported" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Short-term memory (Step 6)
# ---------------------------------------------------------------------------
FIRST, FOLLOW_UP = "How long is Paternity Leave?", "Can it be split into two parts?"


def search_inputs(body):
    return [s["action_input"] for s in body["trace"] if s["action"] == "search_documents"]


def test_api_13_scratch_agent_memory_by_history(server_url):
    history = [{"role": "user", "content": FIRST}, {"role": "assistant", "content": "10 working days."}]
    with_memory = chat(server_url, FOLLOW_UP, "agent_scratch", history=history).json()
    assert any("Paternity" in q for q in search_inputs(with_memory))
    without = chat(server_url, FOLLOW_UP, "agent_scratch").json()
    assert not any("Paternity" in q for q in search_inputs(without))


def test_api_14_graph_agent_memory_by_session(server_url):
    import uuid

    session = f"test-{uuid.uuid4().hex}"
    first = chat(server_url, FIRST, "agent_graph", session_id=session).json()
    follow = chat(server_url, FOLLOW_UP, "agent_graph", session_id=session).json()
    assert any("Paternity" in q for q in search_inputs(follow))
    # The trace shows the current turn only, not the whole stored conversation.
    assert len(follow["trace"]) == len(first["trace"]) == 1
    # A different session is a different conversation.
    other = chat(server_url, FOLLOW_UP, "agent_graph", session_id=f"test-{uuid.uuid4().hex}").json()
    assert not any("Paternity" in q for q in search_inputs(other))


@pytest.mark.parametrize("mode", ["agent_scratch", "agent_graph"])
def test_api_15_memory_messages_reported(server_url, mode):
    import uuid

    session = f"test-{uuid.uuid4().hex}"
    first = chat(server_url, FIRST, mode, session_id=session).json()
    history = [{"role": "user", "content": FIRST}, {"role": "assistant", "content": first["answer"]}]
    follow = chat(server_url, FOLLOW_UP, mode, session_id=session, history=history).json()
    assert first["memory_messages"] == 0
    assert follow["memory_messages"] > 0


# ---------------------------------------------------------------------------
# Uploads follow the same "text twin wins" rule as a full rebuild
# ---------------------------------------------------------------------------
def health_chunks(url):
    return requests.get(f"{url}/health").json()["indexed_chunks"]


def upload(url, name, data):
    return requests.post(f"{url}/upload", files={"file": (name, data)}, timeout=120)


def test_api_16_pdf_twin_of_indexed_md_is_not_indexed_twice(server_url, clean_docs):
    before = health_chunks(server_url)
    r = upload(server_url, "hr_leave_policy.pdf", (PROJECT_ROOT / "sample_docs" / "hr_leave_policy.pdf").read_bytes())
    assert r.status_code == 200
    body = r.json()
    assert body["skipped"] is True and body["chunks"] == 0
    assert "hr_leave_policy.md" in body["message"]
    assert health_chunks(server_url) == before
    sources = chat(server_url, "How many days of Casual Leave per year?").json()["sources"]
    assert {s["source"] for s in sources} == {"hr_leave_policy.md"}


def test_api_17_text_upload_replaces_its_pdf_twin(server_url, clean_docs):
    before = health_chunks(server_url)
    pdf = upload(server_url, "policy_copy.pdf", (PROJECT_ROOT / "sample_docs" / "hr_leave_policy.pdf").read_bytes()).json()
    assert pdf["skipped"] is False and pdf["chunks"] > 0
    assert health_chunks(server_url) == before + pdf["chunks"]
    md = upload(server_url, "policy_copy.md", (PROJECT_ROOT / "sample_docs" / "hr_leave_policy.md").read_bytes()).json()
    assert md["skipped"] is False
    assert health_chunks(server_url) == before + md["chunks"]  # the PDF's chunks were removed
