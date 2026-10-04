"""
agent/ — two agents with the SAME output shape:

    {"answer": str, "trace": [{step, thought, action, action_input, observation}, ...]}

react_agent.py builds the loop by hand; graph_agent.py lets LangGraph run it.
print_trace() below is shared by the step scripts to show what the agent did.
"""
try:
    from rich.console import Console
    from rich.markup import escape

    _console = Console()
except ImportError:  # rich is optional
    _console = None


def _short(text, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def print_trace(result: dict) -> None:
    """Pretty-print an agent result: each Thought / Action / Observation, then the answer."""
    for s in result["trace"]:
        lines = [
            ("Thought", s["thought"], "cyan"),
            ("Action", s["action"], "yellow"),
            ("Action Input", s["action_input"], "yellow"),
            ("Observation", s["observation"], "magenta"),
        ]
        header = f"--- step {s['step']} ---"
        if _console:
            _console.print(f"[bold]{header}[/bold]")
        else:
            print(header)
        for label, value, colour in lines:
            if value in (None, ""):
                continue
            if _console:
                _console.print(f"[{colour}]{label}:[/{colour}] {escape(_short(value))}", highlight=False)
            else:
                print(f"{label}: {_short(value)}")
    if _console:
        _console.print(f"[bold green]Final Answer:[/bold green] {escape(result['answer'])}", highlight=False)
    else:
        print(f"Final Answer: {result['answer']}")
