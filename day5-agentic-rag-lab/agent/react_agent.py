"""
agent/react_agent.py — a ReAct agent written in plain Python (no framework).

Concept taught: an "agent" is just a LOOP around an LLM.

    Thought  -> the model reasons about what to do next
    Action   -> it names a tool and its input (as plain text)
    Observation -> OUR code runs the tool and shows the model the result
    ... repeat until the model writes "Final Answer:"

We use plain-text parsing (regex) instead of native tool calling, so this
works with ANY model. Step 4 rebuilds the same thing with LangGraph.

Memory (Step 6): an LLM remembers nothing between calls. To let the agent
understand follow-ups, WE pass the earlier questions and answers back in
(`history`) — memory, by hand.
"""
from __future__ import annotations

import re

import config
from agent.tools import TOOL_REGISTRY, run_tool
from llm import clean_history, get_llm_response


def build_system_prompt() -> str:
    tool_lines = "\n".join(f"- {name}: {t['description']}" for name, t in TOOL_REGISTRY.items())
    return f"""You are a helpful assistant for employees of Nexora Technologies.
You can use these tools:
{tool_lines}

To use a tool, reply in EXACTLY this format and then stop:
Thought: <your reasoning>
Action: <one of: {", ".join(TOOL_REGISTRY)}>
Action Input: <the input for the tool>

You will then receive "Observation: <tool result>". You may use tools several times.
When you know the answer, reply in EXACTLY this format:
Thought: <your reasoning>
Final Answer: <your answer to the user>

Rules:
- Questions about Nexora policies or leave rules: use search_documents. Never guess company numbers.
- Search using the key words of the question (the leave type and what is asked about).
- If a question needs a policy fact AND arithmetic: FIRST search_documents to find the fact,
  THEN use calculator with it.
- Any arithmetic: use calculator with a plain expression such as 7 * 1.5 (not words).
- General public knowledge (laws, people, places): use wikipedia_lookup.
- Use the earlier conversation to understand follow-up questions (e.g. what "it" refers to).
- Never write "Observation:" yourself. Answer every part of the question.
"""


# Regexes that read the model's text reply
_THOUGHT = re.compile(r"Thought:\s*(.*?)(?=\n\s*(?:Action:|Final Answer:)|\Z)", re.S)
_ACTION = re.compile(r"Action:\s*`?([A-Za-z_]+)`?")
_ACTION_INPUT = re.compile(r"Action Input:\s*(.*?)\s*\Z", re.S)
_FINAL = re.compile(r"Final Answer:\s*(.*)", re.S)

FORMAT_CORRECTION = (
    "Your reply did not follow the required format. Reply with either\n"
    "Thought: ...\nAction: <tool name>\nAction Input: ...\n"
    "or\nThought: ...\nFinal Answer: ..."
)


def parse_reply(text: str) -> dict:
    """Turn the model's reply into {thought, action, action_input, final}. Missing parts are None."""
    thought = _THOUGHT.search(text)
    action = _ACTION.search(text)
    final = _FINAL.search(text)
    parsed = {"thought": thought.group(1).strip() if thought else "", "action": None,
              "action_input": None, "final": None}
    # If both appear, whichever comes FIRST wins (a model may hallucinate past its action).
    if action and (not final or action.start() < final.start()):
        parsed["action"] = action.group(1).strip()
        after_action = text[action.end():]
        tool_input = _ACTION_INPUT.search(after_action)
        parsed["action_input"] = tool_input.group(1).strip().strip("'\"").strip() if tool_input else ""
    elif final:
        parsed["final"] = final.group(1).strip()
    return parsed


def run_react_agent(question: str, history: list[dict] | None = None,
                    max_steps: int = config.MAX_AGENT_STEPS) -> dict:
    """Run the Thought -> Action -> Observation loop. Returns {answer, trace, memory_messages}.

    `history` = earlier [{role, content}] turns. Only questions and final answers are kept
    (not old tool results) and only the last MEMORY_MAX_MESSAGES, so the prompt stays short.
    """
    past = clean_history(history)
    messages = [{"role": "system", "content": build_system_prompt()}, *past, {"role": "user", "content": question}]
    trace: list[dict] = []
    last_observation = ""
    seen_calls: set[tuple[str, str]] = set()

    for step in range(1, max_steps + 1):
        # 1. Ask the LLM. The stop sequence stops it from inventing its own Observation.
        reply = get_llm_response(messages, stop=["Observation:"])
        reply = reply.split("Observation:")[0].strip()  # belt and braces, for providers ignoring `stop`
        parsed = parse_reply(reply)
        messages.append({"role": "assistant", "content": reply})

        # 2a. Done?
        if parsed["final"] is not None:
            trace.append({"step": step, "thought": parsed["thought"], "action": None,
                          "action_input": None, "observation": None})
            return {"answer": parsed["final"], "trace": trace, "memory_messages": len(past)}

        # 2b. Couldn't understand the reply -> ask the model to fix its format.
        if parsed["action"] is None:
            trace.append({"step": step, "thought": parsed["thought"] or reply[:300], "action": None,
                          "action_input": None, "observation": "(reply not in ReAct format; asked the model to retry)"})
            messages.append({"role": "user", "content": FORMAT_CORRECTION})
            continue

        # 3. Run the tool (unknown tool names come back as an error observation).
        #    Small models sometimes repeat the exact same call; nudge them forward instead.
        call = (parsed["action"], parsed["action_input"])
        if call in seen_calls:
            observation = ("You already ran this exact action and input; its result is above. "
                           "Use it, try different keywords, or give the Final Answer.")
        else:
            observation = run_tool(*call)
            last_observation = observation
        seen_calls.add(call)
        trace.append({"step": step, "thought": parsed["thought"], "action": parsed["action"],
                      "action_input": parsed["action_input"], "observation": observation})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # 4. Out of steps: return our best effort plus a warning.
    best_effort = last_observation or "I could not find an answer."
    return {"answer": f"⚠️ Stopped after {max_steps} steps without a final answer. "
                      f"Best effort from the last tool result:\n{best_effort}", "trace": trace,
            "memory_messages": len(past)}
