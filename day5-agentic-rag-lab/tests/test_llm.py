"""
Real-LLM smoke tests (README §8.5) — run with:  pytest -m llm
They start a second server with LLM_PROVIDER=ollama and are skipped automatically
if Ollama isn't running or the model isn't pulled. Answers vary, so we only check
that the key fact appears (case-insensitive), not the exact wording.
"""
import pytest
import requests

from conftest import LLM_TIMEOUT_S

pytestmark = pytest.mark.llm


@pytest.mark.parametrize("test_id,question,mode,any_of", [
    pytest.param("LLM-01", "How many days of Sick Leave can I carry forward?", "rag", ["24"], id="LLM-01"),
    pytest.param("LLM-02", "How long is Paternity Leave?", "rag", ["10"], id="LLM-02"),
    pytest.param("LLM-03", "Within how many days must Comp Off be used?", "agent_scratch", ["60"], id="LLM-03"),
    pytest.param("LLM-04", "If I earn 1.5 days of Earned Leave per month, how many do I have after 8 months?",
                 "agent_graph", ["12"], id="LLM-04"),
    pytest.param("LLM-05", "What is the cafeteria menu?", "rag", ["don't know", "not mentioned", "not available"], id="LLM-05"),
])
def test_real_llm_answers(llm_server, test_id, question, mode, any_of):
    r = requests.post(f"{llm_server.url}/chat", json={"message": question, "mode": mode}, timeout=LLM_TIMEOUT_S)
    assert r.status_code == 200, r.text
    answer = r.json()["answer"].lower()
    assert any(phrase.lower() in answer for phrase in any_of), f"{test_id}: expected one of {any_of} in: {answer!r}"


@pytest.mark.parametrize("mode", ["agent_scratch", "agent_graph"])
def test_llm_06_agent_remembers_follow_up(llm_server, mode):
    """LLM-06: the follow-up only makes sense if the agent remembers the first question."""
    import uuid

    session, history = f"llm-{uuid.uuid4().hex}", []
    for question in ("How long is Paternity Leave?", "Can it be split into two parts?"):
        r = requests.post(f"{llm_server.url}/chat", timeout=LLM_TIMEOUT_S,
                          json={"message": question, "mode": mode, "history": history, "session_id": session})
        assert r.status_code == 200, r.text
        answer = r.json()["answer"]
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    assert any(word in answer.lower() for word in ("two blocks", "two parts", "5 days", "block")), answer
