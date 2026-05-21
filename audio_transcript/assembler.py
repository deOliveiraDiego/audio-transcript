"""Combine transcribed chunk texts into a paragraph-formatted document."""
from __future__ import annotations

from collections.abc import Iterable


def combine(chunks: Iterable[str]) -> str:
    """Join chunk texts with blank lines between them.

    Each chunk becomes one paragraph in the output. Empty or whitespace-only
    chunks are dropped. Per-chunk leading/trailing whitespace is trimmed.
    """
    cleaned = [c.strip() for c in chunks]
    paragraphs = [c for c in cleaned if c]
    return "\n\n".join(paragraphs)
