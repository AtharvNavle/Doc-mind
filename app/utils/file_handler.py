"""Utilities for validating and persisting uploaded files."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.config import get_settings

logger = logging.getLogger(__name__)

ALLOWED_FILE_TYPES: set[str] = {"pdf", "txt"}


def _get_file_extension(filename: str) -> str:
    """Extract and normalize a file extension from a filename."""
    return Path(filename).suffix.lower().lstrip(".")


async def validate_upload_file(upload_file: UploadFile) -> tuple[str, int]:
    """Validate uploaded file type and size, returning file type and byte size."""
    if not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file must include a filename.",
        )

    file_type = _get_file_extension(upload_file.filename)
    if file_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF and TXT files are supported.",
        )

    content = await upload_file.read()
    file_size = len(content)
    await upload_file.seek(0)

    settings = get_settings()
    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb} MB.",
        )

    logger.info(
        "Validated upload file '%s' (type=%s, size=%d bytes)",
        upload_file.filename,
        file_type,
        file_size,
    )
    return file_type, file_size


async def save_upload_file(upload_file: UploadFile) -> tuple[str, str]:
    """Persist an uploaded file to disk and return stored filename and path."""
    if not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file must include a filename.",
        )

    file_type = _get_file_extension(upload_file.filename)
    stored_filename = f"{uuid.uuid4()}.{file_type}"

    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / stored_filename
    content = await upload_file.read()
    await upload_file.seek(0)

    file_path.write_bytes(content)

    logger.info("Saved upload file '%s' to '%s'", upload_file.filename, file_path.as_posix())
    return stored_filename, file_path.as_posix()
