"""Pydantic schemas for document and query APIs."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentStatus


class DocumentCreate(BaseModel):
    """Schema for internal document creation payloads."""

    filename: str
    original_name: str
    file_type: str
    file_size: int

    model_config = ConfigDict(from_attributes=True)


class DocumentRead(BaseModel):
    """Schema returned for document metadata."""

    id: uuid.UUID
    filename: str
    original_name: str
    file_type: str
    file_size: int
    status: DocumentStatus
    chunk_count: int | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class QueryRequest(BaseModel):
    """Request schema for the RAG query endpoint."""

    document_id: uuid.UUID
    question: str = Field(min_length=1)

    model_config = ConfigDict(from_attributes=True)


class QuerySource(BaseModel):
    """Source chunk metadata returned alongside an answer."""

    chunk_text: str
    chunk_index: int
    similarity_score: float

    model_config = ConfigDict(from_attributes=True)


class QueryResponse(BaseModel):
    """Response schema for RAG answers."""

    answer: str
    sources: list[QuerySource]

    model_config = ConfigDict(from_attributes=True)
