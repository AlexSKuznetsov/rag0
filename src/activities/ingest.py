"""Activities responsible for document ingestion preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from temporalio import activity

from ..ingestion.detector import detect_document_type


@activity.defn
async def detect_document_type_activity(source_path: str) -> Dict[str, Any]:
    """Detect the document type for the provided source file."""

    file_path = Path(source_path)
    detected_type = detect_document_type(file_path, {})

    return {
        "document_type": detected_type.value,
    }


__all__ = ["detect_document_type_activity"]
