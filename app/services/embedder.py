"""Gemini embedding generation service."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI, NotFoundError

from app.config import get_settings

logger = logging.getLogger(__name__)
EMBEDDING_FALLBACK_MODEL = "gemini-embedding-001"

settings = get_settings()
client = AsyncOpenAI(
    api_key=settings.gemini_api_key,
    base_url=settings.gemini_base_url,
)


async def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single text input."""
    try:
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
    except NotFoundError:
        logger.warning(
            "Embedding model '%s' unavailable, retrying with '%s'.",
            settings.embedding_model,
            EMBEDDING_FALLBACK_MODEL,
        )
        response = await client.embeddings.create(
            model=EMBEDDING_FALLBACK_MODEL,
            input=text,
        )
    embedding = response.data[0].embedding
    logger.info("Generated embedding for text (%d chars)", len(text))
    return embedding


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for multiple text inputs."""
    if not texts:
        return []

    try:
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
    except NotFoundError:
        logger.warning(
            "Embedding model '%s' unavailable, retrying with '%s'.",
            settings.embedding_model,
            EMBEDDING_FALLBACK_MODEL,
        )
        response = await client.embeddings.create(
            model=EMBEDDING_FALLBACK_MODEL,
            input=texts,
        )
    embeddings = [item.embedding for item in response.data]
    logger.info("Generated %d embeddings", len(embeddings))
    return embeddings
