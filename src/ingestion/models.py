"""Common data models for document ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class DocumentType(str, Enum):
    """Enumerated document types detected during ingestion."""

    TEXT_PDF = "text_pdf"
    SCANNED_PDF = "scanned_pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class ParsedDocument:
    """Structured representation of parsed content."""

    metadata: Dict[str, Any]
    content: Dict[str, Any]
