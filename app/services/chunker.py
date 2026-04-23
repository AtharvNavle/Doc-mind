"""Text chunking utilities for retrieval."""

from __future__ import annotations

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """Split text into overlapping word-based chunks."""
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    tokens = text.split()
    if not tokens:
        return []

    chunks: list[str] = []
    step = size - overlap

    for start in range(0, len(tokens), step):
        end = start + size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            continue
        chunks.append(" ".join(chunk_tokens))
        if end >= len(tokens):
            break

    logger.info(
        "Chunked text into %d chunks (chunk_size=%d, chunk_overlap=%d)",
        len(chunks),
        size,
        overlap,
    )
    return chunks
