"""
llm.py — the ONLY file that knows which LLM provider we are talking to.

Concept taught: hide provider differences behind one small function.
Every other file just calls:

    get_llm_response(messages: list[dict]) -> str        # plain text in/out
    get_chat_model()                                      # LangChain chat model (Step 4)

Providers (set LLM_PROVIDER in .env):
  * ollama             — a local model served by Ollama (default, no API key)
  * openai_compatible  — any hosted OpenAI-style API (e.g. Groq free tier)
  * mock               — a fake, deterministic "LLM" used ONLY by the tests.
                         Retrieval and tools stay real; only the LLM is faked.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

import requests

import config


class LLMError(RuntimeError):
    """Raised with a friendly, student-readable message when the LLM can't be reached."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_llm_response(messages: list[dict], stop: list[str] | None = None) -> str:
    """Send chat messages [{role, content}, ...] to the configured LLM and return its text reply."""
    provider = config.LLM_PROVIDER
    if provider == "ollama":
        return _ollama_chat(messages, stop)
    if provider == "openai_compatible":
        return _openai_compat_chat(messages, stop)
    if provider == "mock":
        return _mock_chat(messages)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use ollama, openai_compatible or mock.")


def get_chat_model():
    """Return a LangChain chat model for the LangGraph agent (Step 4)."""
    provider = config.LLM_PROVIDER
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0,
                          num_predict=config.MAX_OUTPUT_TOKENS, num_ctx=8192)
    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        _require_openai_compat_settings()
        return ChatOpenAI(
            model=config.OPENAI_COMPAT_MODEL,
            base_url=config.OPENAI_COMPAT_BASE_URL,
            api_key=config.OPENAI_COMPAT_API_KEY,
            temperature=0,
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )
    if provider == "mock":
        return FakeToolCallingChatModel()
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'.")


def check_llm() -> tuple[bool, str]:
    """Quick reachability check used by /health and the step scripts."""
    if config.LLM_PROVIDER != "ollama":
        return True, "ok"
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        names = [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return False, _ollama_not_running_msg()
    if not any(n == config.OLLAMA_MODEL or n == f"{config.OLLAMA_MODEL}:latest" for n in names):
        return False, _model_missing_msg()
    return True, "ok"


def clean_history(history: list[dict] | None, limit: int = config.MEMORY_MAX_MESSAGES) -> list[dict]:
    """Keep the last `limit` user/assistant messages from a chat history (short-term memory).

    Anything else (empty messages, unknown roles) is dropped, so a malformed history
    from the browser can never break the prompt.
    """
    turns = [{"role": m["role"], "content": str(m["content"])} for m in (history or [])
             if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")]
    return turns[-limit:] if limit > 0 else []


# ---------------------------------------------------------------------------
# Real providers
# ---------------------------------------------------------------------------
def _ollama_not_running_msg() -> str:
    return (
        f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. "
        "Start it (open the Ollama app or run: ollama serve) and try again."
    )


def _model_missing_msg() -> str:
    return f"Model '{config.OLLAMA_MODEL}' is not pulled. Run: ollama pull {config.OLLAMA_MODEL}"


def _ollama_chat(messages: list[dict], stop: list[str] | None) -> str:
    payload: dict[str, Any] = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        # num_predict caps the reply length so a small model stuck in a repetition loop can't run forever.
        "options": {"temperature": 0, "num_predict": config.MAX_OUTPUT_TOKENS, "num_ctx": 8192, **({"stop": stop} if stop else {})},
    }
    try:
        r = requests.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
    except requests.RequestException as exc:
        raise LLMError(_ollama_not_running_msg()) from exc
    if r.status_code == 404:
        raise LLMError(_model_missing_msg())
    if r.status_code != 200:
        raise LLMError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["message"]["content"]


def _require_openai_compat_settings() -> None:
    if not (config.OPENAI_COMPAT_BASE_URL and config.OPENAI_COMPAT_MODEL):
        raise LLMError("Set OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_API_KEY and OPENAI_COMPAT_MODEL in .env.")


def _openai_compat_chat(messages: list[dict], stop: list[str] | None) -> str:
    _require_openai_compat_settings()
    payload: dict[str, Any] = {"model": config.OPENAI_COMPAT_MODEL, "messages": messages, "temperature": 0,
                               "max_tokens": config.MAX_OUTPUT_TOKENS}
    if stop:
        payload["stop"] = stop
    headers = {"Authorization": f"Bearer {config.OPENAI_COMPAT_API_KEY}"}
    try:
        r = requests.post(f"{config.OPENAI_COMPAT_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=120)
    except requests.RequestException as exc:
        raise LLMError(f"Cannot reach {config.OPENAI_COMPAT_BASE_URL}: {exc}") from exc
    if r.status_code != 200:
        raise LLMError(f"Hosted LLM returned HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Mock provider (tests only) — deterministic routing rules from README §8.1
# ---------------------------------------------------------------------------
_NUM = r"\d+(?:\.\d+)?"
_WORD_OPS = [
    (r"\bmultiplied by\b|\btimes\b", "*"),
    (r"\bdivided by\b", "/"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
]
_ARITHMETIC = re.compile(rf"{_NUM}\s*%\s*of\s*{_NUM}|{_NUM}\s*([+\-*/×÷^%]|\*\*)\s*\(?\s*{_NUM}")


def _to_expression(question: str) -> str | None:
    """Turn 'What is 18% of 4520 plus 75?' into '(18/100*4520) + 75'. None if no arithmetic."""
    text = question.lower().replace(",", "")  # "4,520" -> "4520"; "4520, plus 75" -> "4520 plus 75"
    text = re.sub(rf"({_NUM})\s*%\s*of\s*({_NUM})", r"(\1/100*\2)", text)
    for pattern, symbol in _WORD_OPS:
        text = re.sub(pattern, symbol, text)
    text = text.replace("×", "*").replace("÷", "/").replace("^", "**")
    if not _ARITHMETIC.search(text) and "/100*" not in text:
        return None
    spans = re.findall(r"[\d\.\s+\-*/()%]+", text)
    expr = max(spans, key=lambda s: sum(ch.isdigit() for ch in s)).strip(" .")
    return expr or None


def mock_route(question: str, previous: str | None = None) -> tuple[str, str]:
    """Pick (tool_name, tool_input) the way the mock LLM 'decides'.

    `previous` is the earlier user question when the agent has memory. Like a real
    context-aware LLM resolving "it" in a follow-up, the mock then searches with both.
    """
    stripped = question.strip()
    if stripped.lower().startswith("calculate"):
        return "calculator", stripped[len("calculate"):].strip(" :?")
    expr = _to_expression(stripped)
    if expr:
        return "calculator", expr
    if previous:
        return "search_documents", f"{previous.strip()} {stripped}"
    return "search_documents", stripped


def _mock_chat(messages: list[dict]) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    last = messages[-1]["content"]

    if "Reply YES or NO" in last or "Reply YES or NO" in system:  # LLM-as-judge (Step 5)
        return "YES - mock judge always agrees."

    if "Action Input:" in system:  # ReAct agent (Step 3)
        observations = [m["content"] for m in messages if m["role"] == "user" and m["content"].startswith("Observation:")]
        if observations:
            obs = observations[-1][len("Observation:"):].strip()
            return f"Thought: I now have the information I need.\nFinal Answer: {obs}"
        # The current question is the LAST real user message; earlier ones come from memory.
        asked = [m["content"] for m in messages if m["role"] == "user"
                 and not m["content"].startswith(("Observation:", "Your reply did not follow"))]
        previous = asked[-2] if len(asked) > 1 else None
        tool, tool_input = mock_route(asked[-1], previous)
        return f"Thought: I should use {tool}.\nAction: {tool}\nAction Input: {tool_input}"

    if "Answer only from the context" in system:  # grounded RAG prompt
        match = re.search(r"Question:\s*(.+)", last, re.S)
        question = match.group(1).strip() if match else last
        from rag.retriever import retrieve

        hits = retrieve(question, k=1)
        if not hits or hits[0]["score"] < config.MOCK_SCORE_THRESHOLD:
            return "I don't know based on the provided documents."
        top = hits[0]
        first_sentence = re.split(r"(?<=[.!?])\s|\n", top["text"].strip())[0]
        return f"Based on the policy: {first_sentence} [{top['source']}]"

    return "I don't know — the mock LLM has no general knowledge."


def _make_fake_chat_model_class():
    """Built lazily so importing llm.py never requires langchain unless Step 4 runs."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class _FakeToolCallingChatModel(BaseChatModel):
        """Deterministic stand-in for a tool-calling chat model (same rules as the ReAct mock)."""

        @property
        def _llm_type(self) -> str:
            return "fake-tool-calling"

        def bind_tools(self, tools, **kwargs):  # the fake model already "knows" our tools
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            last = messages[-1]
            if isinstance(last, ToolMessage):
                reply = AIMessage(content=str(last.content))
            else:
                asked = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
                tool, tool_input = mock_route(asked[-1], asked[-2] if len(asked) > 1 else None)
                arg = {"calculator": "expression", "search_documents": "query"}[tool]
                reply = AIMessage(
                    content="",
                    tool_calls=[{"name": tool, "args": {arg: tool_input}, "id": f"call_{uuid.uuid4().hex[:8]}"}],
                )
            return ChatResult(generations=[ChatGeneration(message=reply)])

    return _FakeToolCallingChatModel


def FakeToolCallingChatModel():  # noqa: N802 — reads like a class at the call site
    return _make_fake_chat_model_class()()
