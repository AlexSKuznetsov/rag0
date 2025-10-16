"""Activities for persisting parsed document artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from temporalio import activity

from ..ingestion.storage import store_parsed_markdown


@activity.defn
async def store_parsed_document_activity(
    metadata: Dict[str, Any],
    content: Dict[str, Any],
    parsed_dir: str = "parsed",
) -> Dict[str, Any]:
    """Persist parsed Markdown content to disk."""

    paths = store_parsed_markdown(metadata, content, Path(parsed_dir))
    markdown_path = paths["markdown_path"]
    metadata_path = paths["metadata_path"]
    return {
        "parsed_path": str(markdown_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
        "content": content,
    }


__all__ = ["store_parsed_document_activity"]
