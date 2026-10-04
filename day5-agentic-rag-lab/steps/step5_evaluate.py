"""
Step 5 — Tracing & Evaluation

Concept taught: "it answered something" is not the same as "it works".
For every test question we record WHAT the agent did (its trace) and score it:

  tool accuracy  did it call the tool we expected?
  source hit     did its searches retrieve the document that holds the answer?
  keywords       does the answer contain the key fact (e.g. "24")?
  faithfulness   LLM-as-judge: is every claim supported by what the tools returned?

Run:  python steps/step5_evaluate.py               (from-scratch agent)
      python steps/step5_evaluate.py --agent graph  (LangGraph agent)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from llm import LLMError, check_llm, get_llm_response
from rag.ingest import build_index
from rag.retriever import retrieve

EVAL_DIR = config.PROJECT_ROOT / "eval"
DONT_KNOW_PHRASES = ["don't know", "do not know", "don't have", "do not have", "not mentioned", "not available",
                     "no information", "not provided", "unable to find"]


def judge_faithfulness(answer: str, evidence: str) -> tuple[bool, str]:
    """LLM-as-judge: a second LLM call that grades the first one."""
    prompt = (f"Sources:\n{evidence[:4000]}\n\nAnswer:\n{answer}\n\n"
              "Is every claim in the answer supported by these sources? Reply YES or NO with one reason.\n"
              "Rules: only check that what the answer SAYS appears in the sources. An answer that leaves out "
              "other details is still YES. A calculator result in the sources is correct; do not redo the maths. "
              "An answer that says it doesn't know makes no claims, so it is YES.")
    verdict = get_llm_response([{"role": "user", "content": prompt}]).strip()
    return verdict.upper().startswith("YES"), verdict.replace("\n", " ")[:120]


def evaluate(item: dict, run_agent) -> dict:
    started = time.perf_counter()
    result = run_agent(item["question"])
    latency = time.perf_counter() - started
    answer, trace = result["answer"], result["trace"]
    tools = [s["action"] for s in trace if s["action"]]

    # Which documents did the agent's searches actually retrieve?
    sources = {h["source"] for s in trace if s["action"] == "search_documents" for h in retrieve(s["action_input"])}
    evidence = "\n\n".join(str(s["observation"]) for s in trace if s["observation"])

    if item.get("expect_dont_know"):
        keywords_ok = any(p in answer.lower() for p in DONT_KNOW_PHRASES)
    else:
        keywords_ok = all(k.lower() in answer.lower() for k in item["expected_keywords"])
    faithful, reason = judge_faithfulness(answer, evidence) if evidence else (False, "no tool results to check against")

    return {
        "question": item["question"],
        "answer": answer,
        "tools_used": tools,
        "steps": len(trace),
        "latency_s": round(latency, 2),
        "tool_accuracy": item["expected_tool"] in tools,
        "source_hit": None if item["expected_source"] is None else item["expected_source"] in sources,
        "keywords_ok": keywords_ok,
        "faithful": faithful,
        "judge_reason": reason,
    }


def mark(value) -> str:
    return "n/a" if value is None else ("✅" if value else "❌")


def print_summary(rows: list[dict]) -> None:
    headers = ["#", "question", "tools", "steps", "sec", "tool", "source", "facts", "faithful"]
    table = [[str(i), r["question"][:42], ",".join(r["tools_used"]) or "-", str(r["steps"]), f"{r['latency_s']:.1f}",
              mark(r["tool_accuracy"]), mark(r["source_hit"]), mark(r["keywords_ok"]), mark(r["faithful"])]
             for i, r in enumerate(rows, start=1)]
    try:
        from rich.console import Console
        from rich.table import Table

        t = Table(*headers, title="Evaluation results")
        for row in table:
            t.add_row(*row)
        Console().print(t)
    except ImportError:
        print(" | ".join(headers))
        for row in table:
            print(" | ".join(row))

    def rate(key):
        scored = [r[key] for r in rows if r[key] is not None]
        return f"{sum(scored)}/{len(scored)}"

    print(f"\nTool accuracy {rate('tool_accuracy')} · Source hit {rate('source_hit')} · "
          f"Facts {rate('keywords_ok')} · Faithful {rate('faithful')} · "
          f"Avg latency {sum(r['latency_s'] for r in rows) / len(rows):.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5 — evaluate an agent")
    parser.add_argument("--agent", choices=["scratch", "graph"], default="scratch")
    args = parser.parse_args()

    build_index(verbose=False)
    ok, message = check_llm()
    if not ok:
        print(f"[LLM not available] {message}")
        return

    if args.agent == "scratch":
        from agent.react_agent import run_react_agent as run_agent
    else:
        from agent.graph_agent import run_graph_agent as run_agent

    questions = json.loads((EVAL_DIR / "questions.json").read_text())
    rows = []
    for i, item in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {item['question']}")
        try:
            rows.append(evaluate(item, run_agent))
        except LLMError as exc:
            print(f"   LLM error: {exc}")
            return

    print_summary(rows)
    out = EVAL_DIR / "results.json"
    out.write_text(json.dumps({"agent": args.agent, "model": config.active_model_name(), "results": rows}, indent=2))
    print(f"Saved {out.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
