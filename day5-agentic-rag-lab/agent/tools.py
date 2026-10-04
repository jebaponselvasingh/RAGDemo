"""
agent/tools.py — the tools our agents can use.

Concept taught: a "tool" is just a normal Python function with a clear name
and docstring. The LLM never runs code itself — it only *asks* for a tool by
name, our code runs the function, and the result goes back to the LLM as an
"Observation".

Both agents (Step 3 from scratch, Step 4 LangGraph) read TOOL_REGISTRY, so a
new tool added here appears in both agents and in the UI automatically.
"""
from __future__ import annotations

import ast
import operator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout


# ---------------------------------------------------------------------------
# Tool 1: calculator  (safe — never uses eval())
# ---------------------------------------------------------------------------
_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    """Walk the expression tree, allowing ONLY numbers and + - * / ** % ( )."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"'{type(node).__name__}' is not allowed")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '18/100*4520 + 75' or '7 * 1.5'.
    Only numbers, + - * / ** % and parentheses are allowed."""
    cleaned = expression.strip().strip("`'\"").replace("×", "*").replace("÷", "/").replace("^", "**")
    try:
        result = _evaluate(ast.parse(cleaned, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero."
    except (SyntaxError, ValueError, TypeError) as exc:
        return f"Error: invalid expression ({exc}). Use numbers and + - * / ** % ( ) only."
    result = round(result, 6)
    value = str(int(result)) if float(result).is_integer() else str(result)
    return f"{cleaned} = {value}"  # the full equation is easier for a small model to quote correctly


# ---------------------------------------------------------------------------
# Tool 2: search_documents  (our RAG retriever)
# ---------------------------------------------------------------------------
def search_documents(query: str) -> str:
    """Search the company documents (HR leave policy etc.) and return the most relevant passages.
    Use this for any question about Nexora's policies, leave rules, numbers or procedures.
    Every document is about Nexora, so do NOT put the company name in the query; use the
    specific topic words instead, e.g. 'Bereavement Leave documents'."""
    from rag.retriever import retrieve

    hits = retrieve(query)
    if not hits:
        return "No documents are indexed yet."
    return "\n\n".join(f"[{h['source']}] {h['text'].strip()}" for h in hits)


# ---------------------------------------------------------------------------
# Tool 3: wikipedia_lookup  (an external, general-knowledge tool)
# ---------------------------------------------------------------------------
def _wiki_summary(topic: str) -> str:
    import wikipedia

    try:
        return wikipedia.summary(topic, sentences=3, auto_suggest=False)
    except wikipedia.DisambiguationError as exc:
        return f"'{topic}' is ambiguous. Try one of: {', '.join(exc.options[:5])}"
    except wikipedia.PageError:
        results = wikipedia.search(topic, results=1)
        if results:
            return wikipedia.summary(results[0], sentences=3, auto_suggest=False)
        return f"No Wikipedia page found for '{topic}'."


def wikipedia_lookup(topic: str) -> str:
    """Look up a general-knowledge topic on Wikipedia and return a 3-sentence summary.
    Use this for public facts that are NOT in the company documents (laws, people, places)."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(_wiki_summary, topic.strip()).result(timeout=10)
    except FutureTimeout:
        return "Wikipedia did not respond in time (are you offline?). Try again later."
    except Exception as exc:  # network errors, JSON errors, ...
        return f"Wikipedia is unavailable right now ({type(exc).__name__}). Are you offline?"
    finally:
        pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Mini challenge (README §9): un-comment this tool and add it to TOOL_REGISTRY.
# It then shows up in BOTH agents and in the UI trace with no other changes.
# Try: "How many EL days does EMP002 have, and how many can they encash?"
# ---------------------------------------------------------------------------
# _LEAVE_BALANCES = {
#     "EMP001": {"name": "Asha", "CL": 7, "SL": 15, "EL": 22.5},
#     "EMP002": {"name": "Ravi", "CL": 3, "SL": 24, "EL": 31.5},
#     "EMP003": {"name": "Meena", "CL": 12, "SL": 9, "EL": 6},
# }
#
#
# def leave_balance(employee_id: str) -> str:
#     """Return the current leave balances (CL, SL, EL) for an employee id like 'EMP002'.
#     Use this whenever the question mentions an employee id (EMP001, EMP002, ...)."""
#     record = _LEAVE_BALANCES.get(employee_id.strip().upper())
#     if record is None:
#         return f"No employee with id '{employee_id}'. Known ids: {', '.join(_LEAVE_BALANCES)}"
#     return f"{record['name']} ({employee_id.upper()}): CL={record['CL']}, SL={record['SL']}, EL={record['EL']} days"


# ---------------------------------------------------------------------------
# Registry: name -> function + description (the description is what the LLM reads)
# ---------------------------------------------------------------------------
def _describe(fn) -> str:
    return " ".join(fn.__doc__.split())


TOOL_REGISTRY: dict[str, dict] = {
    fn.__name__: {"fn": fn, "description": _describe(fn)}
    for fn in [calculator, search_documents, wikipedia_lookup]  # + [leave_balance]
}


def run_tool(name: str, tool_input: str) -> str:
    """Run a tool by name; unknown names come back as an error message for the LLM to read."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOL_REGISTRY)}."
    try:
        return entry["fn"](tool_input)
    except Exception as exc:  # a tool must never crash the agent
        return f"Error while running {name}: {exc}"
