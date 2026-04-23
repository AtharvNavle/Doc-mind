"""Document parsing utilities for PDF and text files."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


def _extract_text_from_pdf(file_path: Path) -> str:
    """Extract plain text from a PDF file using PyMuPDF."""
    text_parts: list[str] = []
    with fitz.open(file_path) as document:
        for page in document:
            text_parts.append(page.get_text())
    return "\n".join(text_parts).strip()


def _extract_text_from_txt(file_path: Path) -> str:
    """Extract plain text from a UTF-8 text file."""
    return file_path.read_text(encoding="utf-8").strip()


async def extract_text_from_file(file_path: str, file_type: str) -> str:
    """Extract raw text from a persisted document file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at '{file_path}'.")

    logger.info("Extracting text from '%s' (type=%s)", file_path, file_type)
    if file_type == "pdf":
        text = _extract_text_from_pdf(path)
    elif file_type == "txt":
        text = _extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type '{file_type}'.")

    if not text:
        raise ValueError("No extractable text found in the document.")

    logger.info("Extracted %d characters from '%s'", len(text), file_path)
    return text
