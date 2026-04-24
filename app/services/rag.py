"""RAG pipeline orchestration service using Ollama models."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.document import QueryResponse, QuerySource
from app.services.embedder import embed_text
from app.services.vector_store import search_document_chunks

logger = logging.getLogger(__name__)

settings = get_settings()
client = AsyncOpenAI(
    api_key="ollama",
    base_url=settings.ollama_base_url,
)


def _build_context(chunks: list[dict[str, object]]) -> str:
    """Build a prompt context block from retrieved chunks."""
    return "\n\n".join(
        f"[Chunk {item['chunk_index']}] {item['chunk_text']}" for item in chunks
    )


async def run_rag_query(document_id: str, question: str) -> QueryResponse:
    """Run the end-to-end RAG pipeline for a single question."""
    logger.info("Running RAG query for document '%s'", document_id)

    question_embedding = await embed_text(question)
    top_chunks = search_document_chunks(
        document_id=document_id,
        query_embedding=question_embedding,
        top_k=settings.top_k_results,
    )

    context = _build_context(top_chunks)
    system_prompt = (
        "You are a helpful assistant. Answer ONLY using the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    completion = await client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    answer = completion.choices[0].message.content or "I don't know."
    sources = [QuerySource(**item) for item in top_chunks]
    logger.info("RAG query completed for document '%s'", document_id)
    return QueryResponse(answer=answer, sources=sources)
