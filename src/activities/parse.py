"""Activities for parsing documents into normalized structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from temporalio import activity

from ..ingestion.chunking import ChunkingConfig
from ..ingestion.doc_parser import DocParser
from ..ingestion.models import DocumentType
from ..ingestion.vision_parser import VisionParser


@activity.defn
async def parse_document_activity(
    source_path: str,
    document_type: str,
    chunk_size: int = 700,
    chunk_overlap: int = 150,
    chunk_merge_threshold: int = 60,
) -> Dict[str, Any]:
    """Parse the document based on the detected document type."""

    try:
        doc_type = DocumentType(document_type)
    except ValueError:
        doc_type = DocumentType.UNKNOWN

    file_path = Path(source_path)
    chunk_config = ChunkingConfig(
        chunk_size_tokens=max(chunk_size, 1),
        chunk_overlap_tokens=max(min(chunk_overlap, max(chunk_size - 1, 0)), 0),
        merge_threshold_tokens=max(chunk_merge_threshold, 0),
    )
    doc_parser = DocParser(chunk_config)
    vision_parser = VisionParser(chunking_config=chunk_config)

    metadata = {"source_path": str(file_path), "file_name": file_path.name}
    metadata["document_type"] = doc_type.value

    if doc_type is DocumentType.TEXT_PDF:
        parsed = doc_parser.parse(file_path, metadata)
    elif doc_type in {DocumentType.SCANNED_PDF, DocumentType.IMAGE}:
        if vision_parser.is_available():
            parsed = vision_parser.parse(file_path, metadata)
        else:
            parsed = doc_parser.parse(file_path, metadata)
    else:
        parsed = doc_parser.parse(file_path, metadata)

    return {
        "document_type": doc_type.value,
        "metadata": parsed.metadata,
        "content": parsed.content,
    }


__all__ = ["parse_document_activity"]
