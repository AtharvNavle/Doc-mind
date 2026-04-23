"""Query API endpoints for RAG question answering."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.document import Document, DocumentStatus
from app.schemas.document import QueryRequest, QueryResponse
from app.services.rag import run_rag_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
async def query_document(payload: QueryRequest, db: AsyncSession = Depends(get_db)) -> QueryResponse:
    """Run RAG query for a ready document."""
    result = await db.execute(select(Document).where(Document.id == payload.document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if document.status != DocumentStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document is not ready for querying.",
        )

    logger.info("Received query for document '%s'", payload.document_id)
    return await run_rag_query(document_id=str(payload.document_id), question=payload.question)
