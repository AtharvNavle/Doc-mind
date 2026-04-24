"""Ollama embedding generation service."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
client = AsyncOpenAI(
    api_key="ollama",
    base_url=settings.ollama_base_url,
)


async def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single text input."""
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    embedding = response.data[0].embedding
    logger.info("Generated embedding for text (%d chars)", len(text))
    return embedding


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for multiple text inputs in a batch."""
    if not texts:
        return []

    embeddings: list[list[float]] = []
    for text in texts:
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        embeddings.append(response.data[0].embedding)

    logger.info("Generated %d embeddings", len(embeddings))
    return embeddings
