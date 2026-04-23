"""ChromaDB-backed vector store operations."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
client = chromadb.PersistentClient(path=settings.chroma_persist_dir)


def _collection_name(document_id: str) -> str:
    """Build collection name for a document."""
    return document_id


def get_or_create_document_collection(document_id: str) -> Collection:
    """Get or create a Chroma collection for a specific document."""
    return client.get_or_create_collection(name=_collection_name(document_id))


def add_document_chunks(document_id: str, chunks: list[str], embeddings: list[list[float]]) -> None:
    """Persist document chunks and embeddings in ChromaDB."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings lengths must match.")

    collection = get_or_create_document_collection(document_id)
    ids = [f"{document_id}-{idx}" for idx in range(len(chunks))]
    metadatas = [{"chunk_index": idx} for idx in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("Stored %d chunks in ChromaDB for document '%s'", len(chunks), document_id)


def search_document_chunks(document_id: str, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    """Search top-k most similar chunks for a document."""
    collection = get_or_create_document_collection(document_id)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches: list[dict[str, Any]] = []
    for idx, chunk_text in enumerate(documents):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        distance = distances[idx] if idx < len(distances) else 1.0
        similarity_score = 1.0 / (1.0 + float(distance))
        matches.append(
            {
                "chunk_text": chunk_text,
                "chunk_index": int(metadata.get("chunk_index", idx)),
                "similarity_score": round(similarity_score, 4),
            }
        )

    logger.info("Retrieved %d chunks from ChromaDB for document '%s'", len(matches), document_id)
    return matches


def delete_document_collection(document_id: str) -> None:
    """Delete a document collection from ChromaDB if present."""
    name = _collection_name(document_id)
    existing = {collection.name for collection in client.list_collections()}
    if name in existing:
        client.delete_collection(name=name)
        logger.info("Deleted ChromaDB collection '%s'", name)
