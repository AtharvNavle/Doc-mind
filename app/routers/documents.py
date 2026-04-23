"""Document management API endpoints."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal, get_db
from app.models.document import Document, DocumentStatus
from app.schemas.document import DocumentRead
from app.services.chunker import chunk_text
from app.services.embedder import embed_texts
from app.services.parser import extract_text_from_file
from app.services.vector_store import add_document_chunks, delete_document_collection
from app.utils.file_handler import save_upload_file, validate_upload_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


async def process_document_background(document_id: str) -> None:
    """Run document parsing, chunking, embedding, and vector persistence."""
    settings = get_settings()
    async with AsyncSessionLocal() as session:
        statement = select(Document).where(Document.id == uuid.UUID(document_id))
        result = await session.execute(statement)
        document = result.scalar_one_or_none()
        if not document:
            logger.warning("Background task skipped: document '%s' not found", document_id)
            return

        try:
            file_path = str(Path(settings.upload_dir) / document.filename)
            text = await extract_text_from_file(file_path=file_path, file_type=document.file_type)
            chunks = chunk_text(text=text)
            embeddings = await embed_texts(chunks)
            add_document_chunks(
                document_id=str(document.id),
                chunks=chunks,
                embeddings=embeddings,
            )

            document.status = DocumentStatus.READY
            document.chunk_count = len(chunks)
            await session.commit()
            logger.info(
                "Document '%s' processing completed with %d chunks",
                document_id,
                len(chunks),
            )
        except Exception as exc:  # noqa: BLE001
            document.status = DocumentStatus.FAILED
            document.chunk_count = None
            await session.commit()
            logger.exception("Document '%s' processing failed: %s", document_id, exc)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Upload a document and schedule background processing."""
    file_type, file_size = await validate_upload_file(file)
    stored_filename, _ = await save_upload_file(file)

    document = Document(
        filename=stored_filename,
        original_name=file.filename or stored_filename,
        file_type=file_type,
        file_size=file_size,
        status=DocumentStatus.PROCESSING,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    background_tasks.add_task(process_document_background, str(document.id))
    logger.info("Document '%s' uploaded and queued for processing", document.id)
    return {"document_id": str(document.id), "status": document.status.value}


@router.get("/", response_model=list[DocumentRead])
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[DocumentRead]:
    """List all uploaded documents."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    documents = result.scalars().all()
    return [DocumentRead.model_validate(document) for document in documents]


@router.get("/{doc_id}", response_model=DocumentRead)
async def get_document(doc_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> DocumentRead:
    """Get one document by id."""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentRead.model_validate(document)


@router.delete("/{doc_id}")
async def delete_document(doc_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """Delete document metadata, vectors, and persisted file."""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    settings = get_settings()
    file_path = Path(settings.upload_dir) / document.filename
    if file_path.exists():
        file_path.unlink()

    delete_document_collection(str(doc_id))

    await db.delete(document)
    await db.commit()

    logger.info("Deleted document '%s'", doc_id)
    return {"detail": "Document deleted successfully."}
