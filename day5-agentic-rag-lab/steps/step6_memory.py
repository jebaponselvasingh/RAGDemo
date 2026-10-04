"""
Step 6 — Giving the agents (short-term) memory

Concept taught: an LLM call remembers NOTHING. Every request starts from zero,
so a follow-up like "Can it be split into two parts?" means nothing on its own —
"it" only makes sense if the earlier conversation is sent along again.

We try the same two-turn conversation three ways:
  A. Step 3 agent, no memory          -> the follow-up loses its topic
  B. Step 3 agent, memory BY HAND     -> we pass the earlier turns in ourselves
  C. Step 4 agent, FRAMEWORK memory   -> LangGraph's checkpointer stores each
                                         conversation under a thread_id

Run:  python steps/step6_memory.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agent import print_trace
from agent.graph_agent import build_graph, model_view, run_graph_agent
from agent.react_agent import run_react_agent
from llm import LLMError, check_llm
from rag.ingest import build_index

FIRST = "How long is Paternity Leave?"
FOLLOW_UP = "Can it be split into two parts?"  # "it" = Paternity Leave — only memory can tell


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def show(label: str, result: dict) -> None:
    print(f"\n--- {label}  (agent could see {result.get('memory_messages', 0)} earlier messages)")
    print_trace(result)


def demo_no_memory() -> None:
    header("A. Step 3 agent WITHOUT memory")
    show(f"Q1: {FIRST}", run_react_agent(FIRST))
    show(f"Q2: {FOLLOW_UP}", run_react_agent(FOLLOW_UP))
    print("\n👉 The second search has no idea what 'it' is.")


def demo_memory_by_hand() -> None:
    header("B. Step 3 agent WITH memory passed by hand")
    history: list[dict] = []  # this list IS the memory — we own it
    for question in (FIRST, FOLLOW_UP):
        result = run_react_agent(question, history=history)
        show(f"Q: {question}", result)
        # Remember only the question and the final answer, not the tool results.
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}]
    print(f"\n👉 We kept {len(history)} messages in a plain Python list and sent them with each call.")


def demo_framework_memory() -> None:
    header("C. Step 4 (LangGraph) agent WITH a checkpointer")
    thread = {"configurable": {"thread_id": "step6-demo"}}
    for question in (FIRST, FOLLOW_UP):
        show(f"Q: {question}  [thread_id=step6-demo]", run_graph_agent(question, thread_id="step6-demo"))

    # The checkpointer saved EVERYTHING: questions, tool calls, tool results, answers...
    stored = build_graph().get_state(thread).values["messages"]
    visible, _ = model_view(stored)
    print(f"\n👉 The checkpointer stores {len(stored)} messages for thread 'step6-demo'.")
    print(f"   Before each model call we trim that to the last {config.MEMORY_MAX_MESSAGES} earlier "
          f"questions/answers + the current turn ({len(visible)} messages right now).")

    header("C2. Same follow-up on a NEW thread_id")
    show(f"Q: {FOLLOW_UP}  [thread_id=another-chat]", run_graph_agent(FOLLOW_UP, thread_id="another-chat"))
    print("\n👉 A new thread_id is a new conversation: nothing carried over. (In the web app: '＋ New chat'.)")


def main() -> None:
    header("Preparing the document index")
    build_index()
    ok, message = check_llm()
    if not ok:
        print(f"[LLM not available] {message}")
        return
    try:
        demo_no_memory()
        demo_memory_by_hand()
        demo_framework_memory()
    except LLMError as exc:
        print(f"[LLM error] {exc}")
        return

    header("What memory costs")
    print("* Every earlier message is sent again on EVERY call: more tokens, slower answers, and old\n"
          "  turns can distract a small model. That's why we keep only the last few messages.\n"
          "* This memory lives in RAM: restarting the program (or the server) forgets it.\n"
          "  Saving it to a database (e.g. LangGraph's SqliteSaver) would make it survive restarts.")


if __name__ == "__main__":
    main()
