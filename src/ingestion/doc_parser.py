"""Simplified document parser using Docling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from docling.document_converter import DocumentConverter

from .chunking import ChunkingConfig, generate_chunks
from .models import ParsedDocument

logger = logging.getLogger(__name__)


class DocParser:
    """Parses documents using Docling DocumentConverter."""

    def __init__(self, chunking_config: Optional[ChunkingConfig] = None) -> None:
        self._converter = DocumentConverter()
        self._chunking_config = chunking_config or ChunkingConfig()

    def parse(self, file_path: Path, metadata: Dict[str, Any]) -> ParsedDocument:
        """Parse a document and return structured content."""

        result = self._converter.convert(str(file_path))
        markdown = result.document.export_to_markdown()
        text_blocks = [segment.strip() for segment in markdown.split("\n\n") if segment.strip()]
        paragraphs = [{"text": block} for block in text_blocks]

        warnings = [
            getattr(error, "error_message", "")
            for error in getattr(result, "errors", []) or []
            if getattr(error, "error_message", "")
        ]

        merged_metadata = dict(metadata)
        input_doc = getattr(result, "input", None)
        file_name = getattr(input_doc, "filename", None)
        if isinstance(file_name, str) and file_name.strip():
            merged_metadata.setdefault("file_name", file_name.strip())

        chunks = generate_chunks(paragraphs, merged_metadata, self._chunking_config)
        merged_metadata["chunk_descriptors"] = [
            {
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "paragraph_start": chunk["paragraph_start"],
                "paragraph_end": chunk["paragraph_end"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "token_count": chunk["token_count"],
            }
            for chunk in chunks
        ]

        return ParsedDocument(
            metadata=merged_metadata,
            content={
                "paragraphs": paragraphs,
                "markdown": markdown,
                "warnings": warnings,
                "chunks": chunks,
            },
        )
