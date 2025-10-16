"""Activities handling vector index updates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from temporalio import activity

from ..ingestion.storage import load_parsed_markdown
from ..ingestion.vector_store import VectorStoreManager


@activity.defn
async def update_index_activity(
    parsed_markdown_path: str,
    index_dir: str = "storage/index",
    metadata_path: str | None = None,
) -> Dict[str, Any]:
    """Load parsed Markdown content and upsert it into the vector index."""

    manager = VectorStoreManager(storage_dir=Path(index_dir))
    parsed_payload = load_parsed_markdown(
        Path(parsed_markdown_path),
        Path(metadata_path) if metadata_path else None,
    )

    metadata = parsed_payload.get("metadata", {})
    content = parsed_payload.get("content", {})
    paragraphs: List[Dict[str, Any]] = content.get("paragraphs", [])
    chunks: List[Dict[str, Any]] = content.get("chunks", [])
    markdown_body: str = content.get("markdown", "")
    warnings: List[Any] = parsed_payload.get("warnings", [])

    text_blocks: List[Dict[str, Any]] = []
    if chunks:
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue
            chunk_metadata = {
                **metadata,
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "paragraph_start": chunk.get("paragraph_start"),
                "paragraph_end": chunk.get("paragraph_end"),
            }
            if metadata_path:
                chunk_metadata["parsed_metadata_path"] = metadata_path
            text_blocks.append(
                {
                    "text": text,
                    "metadata": chunk_metadata,
                }
            )
    else:
        for paragraph in paragraphs:
            text = paragraph.get("text", "")
            if not text:
                continue
            block_metadata = {
                **metadata,
                "page": paragraph.get("page"),
            }
            text_blocks.append(
                {
                    "text": text,
                    "metadata": block_metadata,
                }
            )

    if text_blocks:
        manager.upsert_documents(text_blocks)

    return {
        "status": "indexed",
        "documents": str(len(text_blocks)),
        "markdown_preview": markdown_body[:500],
        "warnings": warnings,
    }


__all__ = ["update_index_activity"]
