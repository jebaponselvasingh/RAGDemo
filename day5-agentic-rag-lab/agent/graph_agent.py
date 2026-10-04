"""
agent/graph_agent.py — the SAME agent as react_agent.py, built with LangGraph.

Concept taught: an agent is a small GRAPH.

    START → [agent] ──(model asked for a tool?)── yes → [tools] ─┐
               ▲                     │                             │
               │                     no → END                      │
               └───────────────────────────────────────────────────┘

  * agent node  : the chat model with .bind_tools() — it uses NATIVE tool calling
                  (structured JSON) instead of our hand-written text format
  * tools node  : LangGraph's prebuilt ToolNode runs whichever tool was requested
  * tools_condition : the conditional edge that decides "tools" or END

The tools are the very same Python functions from agent/tools.py — no logic is duplicated.

Memory (Step 6): the graph is compiled with a CHECKPOINTER. After every node it saves the
state (all messages) under a `thread_id`; calling the graph again with the same thread_id
continues that conversation. That is the framework doing what Step 3 does by hand.
The checkpointer keeps EVERYTHING, so before each model call we choose what the model sees:
the current turn in full + only the last few earlier questions/answers.
"""
from __future__ import annotations

import json
import uuid
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config
from agent.tools import TOOL_REGISTRY
from llm import get_chat_model

SYSTEM_PROMPT = """You are a helpful assistant for employees of Nexora Technologies.
Tools:
- search_documents: Nexora leave policy facts. Use it for every company question; never guess company facts.
- calculator: arithmetic. Pass a plain expression like 3 * 4.
- wikipedia_lookup: general public knowledge.
For questions that need both, search the policy first, then calculate.
When you write the final answer, copy numbers EXACTLY from the tool results and answer every part of the question."""

# Wrap the plain functions as LangChain tools. tool(fn) is exactly what the @tool decorator does;
# it reads the function name, type hints and docstring to describe the tool to the model.
LC_TOOLS = [tool(entry["fn"]) for entry in TOOL_REGISTRY.values()]


def _current_turn_start(messages: list) -> int:
    """Index of the newest HumanMessage: everything from there on is the current turn."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0


def model_view(messages: list) -> tuple[list, int]:
    """What the model actually sees: the current turn in full, plus a trimmed memory of earlier turns.

    Earlier turns keep only questions and final answers (no tool calls/results), and only
    the last MEMORY_MAX_MESSAGES of them. Returns (messages_for_model, memory_message_count).
    """
    start = _current_turn_start(messages)
    earlier = [m for m in messages[:start]
               if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls and m.content)]
    memory = trim_messages(earlier, max_tokens=config.MEMORY_MAX_MESSAGES, token_counter=len,
                           strategy="last", start_on="human") if earlier else []
    return memory + messages[start:], len(memory)


@lru_cache(maxsize=1)
def build_graph():
    """Create and compile the agent graph (done once). The checkpointer gives it memory."""
    model = get_chat_model().bind_tools(LC_TOOLS)

    def agent_node(state: MessagesState) -> dict:
        visible, _ = model_view(state["messages"])
        reply = model.invoke([SystemMessage(SYSTEM_PROMPT)] + visible)
        return {"messages": [reply]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(LC_TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # → "tools" or END
    builder.add_edge("tools", "agent")  # after a tool runs, go back to the model
    return builder.compile(checkpointer=InMemorySaver())  # in RAM: a server restart forgets everything


def _tool_input(args: dict) -> str:
    """Show the tool input like Step 3 does: the bare value when there is one argument."""
    if len(args) == 1:
        return str(next(iter(args.values())))
    return json.dumps(args)


def messages_to_trace(messages: list) -> list[dict]:
    """Convert LangGraph's message list into Step 3's trace shape."""
    results = {m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)}
    trace: list[dict] = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        thought = m.content if isinstance(m.content, str) else str(m.content)
        for call in m.tool_calls:
            trace.append({"step": len(trace) + 1, "thought": thought.strip(), "action": call["name"],
                          "action_input": _tool_input(call["args"]),
                          "observation": results.get(call["id"], "(no result)")})
            thought = ""  # only show the thought once for parallel tool calls
    return trace


def run_graph_agent(question: str, thread_id: str | None = None, max_steps: int = config.MAX_AGENT_STEPS) -> dict:
    """Run the LangGraph agent. Returns {answer, trace, memory_messages} — the same shape as run_react_agent.

    Same `thread_id` again -> the conversation continues (memory). No thread_id -> a one-off
    conversation that is deleted afterwards.
    """
    graph = build_graph()
    temporary = thread_id is None
    thread_id = thread_id or f"tmp-{uuid.uuid4().hex}"
    run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps * 2}
    state_messages: list = [HumanMessage(question)]
    try:
        # stream_mode="values" yields the full state after every node, so we keep the latest one
        # even if the recursion limit stops the run.
        for state in graph.stream({"messages": [HumanMessage(question)]}, run_config, stream_mode="values"):
            state_messages = state["messages"]
        stopped = False
    except GraphRecursionError:
        stopped = True
    finally:
        if temporary:
            graph.checkpointer.delete_thread(thread_id)

    _, memory_count = model_view(state_messages)
    current_turn = state_messages[_current_turn_start(state_messages):]  # the trace shows this turn only
    trace = messages_to_trace(current_turn)
    if stopped:
        last = trace[-1]["observation"] if trace else "I could not find an answer."
        return {"answer": f"⚠️ Stopped after {max_steps} steps without a final answer. "
                          f"Best effort from the last tool result:\n{last}", "trace": trace,
                "memory_messages": memory_count}

    final = state_messages[-1]
    answer = final.content if isinstance(final, AIMessage) and final.content else "(no answer)"
    return {"answer": str(answer), "trace": trace, "memory_messages": memory_count}
