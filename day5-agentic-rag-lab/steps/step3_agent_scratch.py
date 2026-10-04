"""
Step 3 — A ReAct agent from scratch (plain Python, no framework)

Concept taught: an agent = LLM + tools + a loop. The LLM decides WHICH tool to
use; our code runs it and feeds the result back as an "Observation".
Watch the trace: every Thought / Action / Observation is printed.

Run:  python steps/step3_agent_scratch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import print_trace
from agent.react_agent import run_react_agent
from llm import LLMError, check_llm
from rag.ingest import build_index

DEMO_QUESTIONS = [
    ("What is 18% of 4520, plus 75?", "calculator"),
    ("How many days of Sick Leave can be carried forward?", "search_documents"),
    ("What is the Maternity Benefit Act in India?", "wikipedia_lookup"),
    ("I joined in March. How many Earned Leave days will I have accrued by the end of September, "
     "and can I encash them?", "search_documents, then calculator (7 x 1.5 = 10.5)"),
]


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def ask(question: str) -> None:
    try:
        print_trace(run_react_agent(question))
    except LLMError as exc:
        print(f"[LLM error] {exc}")


def main() -> None:
    header("Preparing the document index for the search_documents tool")
    build_index()

    ok, message = check_llm()
    if not ok:
        print(f"[LLM not available] {message}")
        return

    for i, (question, expected) in enumerate(DEMO_QUESTIONS, start=1):
        header(f"Demo {i}: {question}\n(expected tool: {expected})")
        ask(question)

    header("Your turn — ask anything (type 'exit' to quit)")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            ask(question)


if __name__ == "__main__":
    main()
