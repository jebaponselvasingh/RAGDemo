"""
Step 2 — RAG Pipeline: load → chunk → embed → store → retrieve → generate

Concept taught: a general LLM has never seen Nexora's leave policy, so it
guesses. If we RETRIEVE the relevant chunks and paste them into the prompt,
the same model answers correctly and cites its source.

Run:
  python steps/step2_rag.py
  python steps/step2_rag.py --show-chunks                  # see where the text is cut
  python steps/step2_rag.py --chunk-size 200 --chunk-overlap 50   # re-index live
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from llm import LLMError, check_llm, get_llm_response
from rag.ingest import build_index, preview_chunks
from rag.retriever import build_rag_messages, retrieve

try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
except ImportError:  # rich is optional
    console = None


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def show_side_by_side(without_rag: str, with_rag: str) -> None:
    """The teaching moment: same model, same question, with and without retrieved context."""
    if console:
        console.print(Columns([
            Panel(without_rag, title="(a) LLM WITHOUT context", border_style="red", width=60),
            Panel(with_rag, title="(b) LLM WITH RAG context", border_style="green", width=60),
        ]))
    else:
        print("\n--- (a) LLM WITHOUT context ---\n" + without_rag)
        print("\n--- (b) LLM WITH RAG context ---\n" + with_rag)


def answer_question(question: str) -> None:
    # 1. RETRIEVE: embed the question and find the nearest chunks.
    hits = retrieve(question, config.TOP_K)
    header(f"Top-{len(hits)} retrieved chunks")
    for i, h in enumerate(hits, start=1):
        snippet = h["text"].strip().replace("\n", " ")[:150]
        print(f"{i}. [{h['source']}] score={h['score']:.3f}  {snippet}...")

    # 2. GENERATE twice: once with no context, once with the grounded prompt.
    plain = [{"role": "user", "content": question}]
    grounded = build_rag_messages(question, hits)
    print("\nAsking the LLM twice (this can take a few seconds on a laptop)...")
    show_side_by_side(get_llm_response(plain), get_llm_response(grounded))


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2 — RAG with vs without retrieval")
    parser.add_argument("--show-chunks", action="store_true", help="print the first 5 chunks and their boundaries")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    header("1. Build the index (load → chunk → embed → store)")
    # build_index re-indexes automatically when the chunk settings differ from what's on disk.
    build_index(size=args.chunk_size, overlap=args.chunk_overlap)

    if args.show_chunks:
        header("Chunk preview")
        preview_chunks(5, size=args.chunk_size, overlap=args.chunk_overlap)

    ok, message = check_llm()
    if not ok:
        print(f"\n[LLM not available] {message}")
        return

    header("2. Ask questions (type 'exit' to quit)")
    print("Try: How many days of Sick Leave can be carried forward?")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            answer_question(question)
        except LLMError as exc:
            print(f"[LLM error] {exc}")


if __name__ == "__main__":
    main()
