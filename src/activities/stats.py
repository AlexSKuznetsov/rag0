"""Activity that reports high-level vector index statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from temporalio import activity

from ..ingestion.vector_store import VectorStoreManager


@activity.defn
async def index_stats_activity(index_dir: str = "storage/index") -> Dict[str, Any]:
    """Return basic statistics about the backing vector store."""

    stats = VectorStoreManager.get_stats(Path(index_dir))
    stats["status"] = "ok"
    return stats


__all__ = ["index_stats_activity"]
