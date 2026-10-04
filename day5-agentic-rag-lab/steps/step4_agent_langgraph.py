"""
Step 4 — The same agent, rebuilt with LangGraph

Concept taught: frameworks don't do magic — they package the loop you wrote
in Step 3. Compare the two, line by line ("what the framework is doing for you"):

  Step 3 (agent/react_agent.py)                  Step 4 (agent/graph_agent.py)
  ---------------------------------------------  ----------------------------------------------
  build_system_prompt() lists tools as TEXT      tool(fn) turns each function into a JSON schema;
                                                 .bind_tools() sends it to the model natively
  regex parses "Action: / Action Input:"         the model returns structured `tool_calls`
                                                 (no parsing, no format errors to correct)
  for step in range(max_steps): ...              StateGraph: agent node <-> tools node, looping
  if parsed["final"]: return                     tools_condition edge: no tool call -> END
  run_tool(name, input)                          ToolNode runs the requested tool
  messages.append("Observation: ...")            ToolNode appends a ToolMessage to the state
  stop=["Observation:"]                          not needed: tool results are separate messages
  max_steps -> "Stopped after N steps"           recursion_limit -> GraphRecursionError
  trace list built by hand                       messages_to_trace() reads the message history
  memory: pass `history` in by hand (Step 6)     checkpointer saves the state per thread_id (Step 6)

  Trade-off: LangGraph needs a model that supports native tool calling;
  the Step 3 text format works with ANY model.

Run:  python steps/step4_agent_langgraph.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import print_trace
from agent.graph_agent import build_graph, run_graph_agent
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
        print_trace(run_graph_agent(question))
    except LLMError as exc:
        print(f"[LLM error] {exc}")
    except Exception as exc:  # e.g. Ollama not running -> connection error inside LangChain
        print(f"[Agent error] {type(exc).__name__}: {exc}")


def main() -> None:
    header("Preparing the document index for the search_documents tool")
    build_index()

    ok, message = check_llm()
    if not ok:
        print(f"[LLM not available] {message}")
        return

    header("The agent graph (Mermaid — paste into https://mermaid.live to see it drawn)")
    print(build_graph().get_graph().draw_mermaid())

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
