"""
Step 1 — Embeddings & Semantic Search

Concept taught: an embedding model turns a sentence into a list of numbers
(a vector) so that sentences with SIMILAR MEANING get vectors that point in a
similar direction — even when they share no words at all.

Run:  python steps/step1_embeddings.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import config` works

import numpy as np
from sentence_transformers import SentenceTransformer

import config

DEFAULT_QUERY = "How many days off can I take for my wedding?"

# A tiny "knowledge base": some HR policy lines mixed with unrelated sentences.
SENTENCES = [
    "Employees get 5 working days of Marriage Leave.",
    "Sick Leave of more than 2 consecutive days requires a medical certificate.",
    "Employees earn 1.5 days of Earned Leave for every completed month.",
    "Comp Off must be used within 60 days of the date it was earned.",
    "Virat Kohli scored a brilliant century in the test match.",
    "India won the cricket world cup final by six wickets.",
    "Add the onions to the pan and fry them until golden brown.",
    "Biryani tastes best when the rice is cooked with whole spices.",
]


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # cos(θ) = (a · b) / (‖a‖ × ‖b‖)
    #   1.0  -> same direction (same meaning)
    #   0.0  -> unrelated
    #  -1.0  -> opposite
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def keyword_overlap(query: str, sentence: str) -> float:
    """Naive 'search engine': fraction of query words that also appear in the sentence."""
    words = lambda s: set(re.findall(r"[a-z]+", s.lower()))  # noqa: E731
    q = words(query)
    return len(q & words(sentence)) / max(1, len(q))


def main() -> None:
    header("1. Load the embedding model")
    print(f"Model: {config.EMBEDDING_MODEL} (runs on CPU, first run downloads ~90 MB)")
    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

    header("2. Embed the sentences")
    vectors = model.encode(SENTENCES)
    print(f"Embedding matrix shape: {vectors.shape}  -> {len(SENTENCES)} sentences x {vectors.shape[1]} dimensions")
    print(f"First 8 values of sentence 1's vector:\n  {np.round(vectors[0][:8], 4)}")

    header("3. Ask a question")
    try:
        query = input(f"Your query (Enter for default: '{DEFAULT_QUERY}'): ").strip() or DEFAULT_QUERY
    except EOFError:
        query = DEFAULT_QUERY
    print(f"Query: {query}")
    query_vector = model.encode(query)

    header("4. Semantic search (cosine similarity) vs keyword overlap")
    rows = [(cosine_similarity(query_vector, v), keyword_overlap(query, s), s) for v, s in zip(vectors, SENTENCES)]
    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"{'rank':<5}{'cosine':>8}{'keyword':>9}   sentence")
    for rank, (cos, kw, sentence) in enumerate(rows, start=1):
        print(f"{rank:<5}{cos:>8.3f}{kw:>9.2f}   {sentence}")

    best_semantic = rows[0][2]
    best_keyword = max(rows, key=lambda r: r[1])
    header("5. What did we learn?")
    print(f"Semantic search's best match : {best_semantic}")
    if best_keyword[1] == 0:
        print("Keyword search found NO sentence sharing a word with the query!")
    else:
        print(f"Keyword search's best match  : {best_keyword[2]}  (overlap {best_keyword[1]:.2f})")
    print("Embeddings match on MEANING ('wedding' ~ 'Marriage Leave'), not on exact words.")


if __name__ == "__main__":
    main()
