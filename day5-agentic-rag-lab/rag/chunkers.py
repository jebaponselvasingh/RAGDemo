"""
rag/chunkers.py — three ways to cut a document into chunks.

Concept taught: HOW you chunk decides WHAT the retriever can find.

  fixed      exactly `size` characters, step `size - overlap`; cuts anywhere (even mid-word)
  recursive  split on "\n\n", then "\n", then ". ", then " " so chunks end at natural
             boundaries; same idea as LangChain's RecursiveCharacterTextSplitter
  heading    one chunk per document section (Markdown "## " or numbered titles like
             "6. " / "6.1 "); big sections are sub-split with `recursive`

Every chunk records CHARACTER OFFSETS into the original text, so this is always true:
    full_text[chunk["start"]:chunk["end"]] == chunk["text"]
That is what lets the Explorer page highlight exactly where each chunk came from.
"""
from __future__ import annotations

import bisect
import re

STRATEGIES = ("fixed", "recursive", "heading")
SEPARATORS = ["\n\n", "\n", ". ", " "]

# "## 6. Earned Leave (EL)"  or a short numbered title line like "6.1 Accrual"
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_NUM_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][^\n]*$")


# ---------------------------------------------------------------------------
# Headings and pages (used to label every chunk)
# ---------------------------------------------------------------------------
def find_headings(text: str) -> list[tuple[int, str]]:
    """Return [(char_offset, heading_title), ...] in document order."""
    headings = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        md = _MD_HEADING.match(stripped)
        if md:
            headings.append((offset, md.group(1).strip()))
        # Numbered titles are short and don't end like a sentence ("1. Log in ... ." is a list item).
        elif _NUM_HEADING.match(stripped) and len(stripped) <= 60 and not stripped.endswith((".", ",", ":")):
            headings.append((offset, stripped))
        offset += len(line)
    return headings


def _section_at(pos: int, headings: list[tuple[int, str]]) -> str:
    """Nearest heading at or before `pos`."""
    starts = [h[0] for h in headings]
    i = bisect.bisect_right(starts, pos) - 1
    return headings[i][1] if i >= 0 else "(document start)"


def _page_at(pos: int, page_starts: list[int] | None) -> int:
    """Page number (1-based) containing `pos`. Non-PDF documents are a single page."""
    if not page_starts:
        return 1
    return max(1, bisect.bisect_right(page_starts, pos))


# ---------------------------------------------------------------------------
# Strategy 1: fixed
# ---------------------------------------------------------------------------
def _fixed_spans(text: str, size: int, overlap: int, base: int = 0) -> list[tuple[int, int]]:
    step = max(1, size - overlap)
    spans = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        spans.append((base + start, base + end))
        if end == len(text):
            break
        start += step
    return spans


# ---------------------------------------------------------------------------
# Strategy 2: recursive
# ---------------------------------------------------------------------------
def _split_pieces(text: str, start: int, end: int, size: int, seps: list[str]) -> list[tuple[int, int]]:
    """Break text[start:end] into contiguous spans of at most `size`, trying separators in order."""
    if end - start <= size:
        return [(start, end)]
    if not seps:  # no separator left: hard cut
        return [(s, min(s + size, end)) for s in range(start, end, size)]
    sep, rest = seps[0], seps[1:]
    pieces, cursor = [], start
    while cursor < end:
        hit = text.find(sep, cursor, end)
        piece_end = end if hit == -1 else hit + len(sep)  # keep the separator on the left piece
        if piece_end - cursor <= size:
            pieces.append((cursor, piece_end))
        else:
            pieces.extend(_split_pieces(text, cursor, piece_end, size, rest))
        cursor = piece_end
    return pieces


def _recursive_spans(text: str, size: int, overlap: int, base: int = 0) -> list[tuple[int, int]]:
    # Leave room for the overlap so that body + overlap never exceeds `size`.
    body = max(1, size - overlap)
    merged: list[list[int]] = []
    for s, e in _split_pieces(text, 0, len(text), body, SEPARATORS):
        if merged and e - merged[-1][0] <= body:
            merged[-1][1] = e  # grow the current chunk
        else:
            merged.append([s, e])
    spans = []
    for i, (s, e) in enumerate(merged):
        if i > 0 and overlap > 0:
            s2 = max(0, s - overlap)
            space = text.find(" ", s2, s)  # start the overlap on a word boundary
            s = space + 1 if space != -1 else s2
        spans.append((base + s, base + e))
    return spans


# ---------------------------------------------------------------------------
# Strategy 3: heading
# ---------------------------------------------------------------------------
def _heading_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    cut_points = sorted({0, *[h[0] for h in find_headings(text)], len(text)})
    spans = []
    for s, e in zip(cut_points, cut_points[1:]):
        if not text[s:e].strip():
            continue
        if e - s <= size:
            spans.append((s, e))
        else:  # big section: sub-split it, still inside the section
            spans.extend(_recursive_spans(text[s:e], size, overlap, base=s))
    return spans


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
def _is_table_row(line: str) -> bool:
    return line.count("|") >= 2


def _line_around(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start: end if end != -1 else len(text)]


def _warnings(text: str, start: int, end: int, size: int) -> list[str]:
    chunk = text[start:end]
    warnings = []
    if not chunk.rstrip(" \t").endswith((".", "?", "!", "\n")):
        warnings.append("mid_sentence")
    if 0 < end < len(text) and _is_table_row(_line_around(text, end - 1)) and _is_table_row(_line_around(text, end)):
        warnings.append("split_table")
    if len(chunk) < 0.25 * size:
        warnings.append("tiny")
    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def chunk_document(
    text: str,
    strategy: str = "recursive",
    size: int = 500,
    overlap: int = 100,
    page_starts: list[int] | None = None,
) -> list[dict]:
    """Split `text` and return Chunk dicts with offsets, page, section, overlap and warnings."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose one of {STRATEGIES}.")
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))

    if strategy == "fixed":
        spans = _fixed_spans(text, size, overlap)
    elif strategy == "recursive":
        spans = _recursive_spans(text, size, overlap)
    else:
        spans = _heading_spans(text, size, overlap)

    headings = find_headings(text)
    chunks, prev_end = [], None
    for i, (s, e) in enumerate(spans):
        chunks.append({
            "id": f"c{i:03d}",
            "text": text[s:e],
            "start": s,
            "end": e,
            "length": e - s,
            "page": _page_at(s, page_starts),
            "section": _section_at(s, headings),
            "overlap_with_prev": max(0, prev_end - s) if prev_end is not None else 0,
            "warnings": _warnings(text, s, e, size),
        })
        prev_end = e
    return chunks


def chunk_stats(chunks: list[dict]) -> dict:
    """Count / average / min / max length, plus every length (for the histogram)."""
    lengths = [c["length"] for c in chunks]
    if not lengths:
        return {"count": 0, "avg": 0, "min": 0, "max": 0, "lengths": []}
    return {
        "count": len(lengths),
        "avg": round(sum(lengths) / len(lengths), 1),
        "min": min(lengths),
        "max": max(lengths),
        "lengths": lengths,
    }
