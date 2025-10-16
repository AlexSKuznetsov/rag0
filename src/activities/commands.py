"""Activities that validate and prepare workflow command payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from temporalio import activity

from ..ingestion.vector_store import VectorStoreManager


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


@activity.defn
async def ingest_command_activity(
    arguments: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate ingest command arguments and construct the workflow payload."""

    source_path_value = arguments.get("source_path")
    if not source_path_value:
        raise ValueError("source_path is required for ingest command")

    source_path = Path(source_path_value)
    if not source_path.exists():
        raise ValueError(f"File not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Not a file: {source_path}")

    parsed_dir = config.get("parsed_dir", "parsed")
    index_dir = config.get("index_dir", "storage/index")
    chunk_size = max(int(arguments.get("chunk_size", config.get("chunk_size", 700))), 1)
    raw_chunk_overlap = int(arguments.get("chunk_overlap", config.get("chunk_overlap", 150)))
    chunk_overlap = max(min(raw_chunk_overlap, chunk_size - 1), 0) if chunk_size > 1 else 0
    chunk_merge_threshold = max(
        int(arguments.get("chunk_merge_threshold", config.get("chunk_merge_threshold", 60))),
        0,
    )

    return {
        "parsed_dir": parsed_dir,
        "index_dir": index_dir,
        "source_path": str(source_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_merge_threshold": chunk_merge_threshold,
    }


@activity.defn
async def question_command_activity(
    arguments: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize question command arguments for the downstream workflow."""

    question = (arguments.get("question") or "").strip()
    if not question:
        raise ValueError("question is required for ask command")

    ask_top_k = int(arguments.get("top_k", config.get("ask_top_k", 6)))
    temperature = float(arguments.get("temperature", config.get("temperature", 0.0)))
    max_subquestions = int(arguments.get("max_subquestions", config.get("max_subquestions", 3)))
    neighbor_span = int(arguments.get("neighbor_span", config.get("neighbor_span", 1)))
    reflection_enabled = _coerce_bool(
        arguments.get("reflection_enabled", config.get("reflection_enabled", True))
    )
    max_reflections = int(arguments.get("max_reflections", config.get("max_reflections", 2)))
    min_citations = int(arguments.get("min_citations", config.get("min_citations", 1)))

    return {
        "question": question,
        "index_dir": config.get("index_dir", "storage/index"),
        "top_k": ask_top_k or config.get("ask_top_k", 6),
        "ollama_model": arguments.get("ollama_model", config.get("ollama_model", "")),
        "ollama_base_url": arguments.get("ollama_base_url", config.get("ollama_base_url", "")),
        "temperature": temperature,
        "max_subquestions": max_subquestions,
        "neighbor_span": max(neighbor_span, 0),
        "reflection_enabled": reflection_enabled,
        "max_reflections": max(max_reflections, 0),
        "min_citations": max(min_citations, 0),
    }


@activity.defn
async def stats_command_activity(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return vector index statistics for the stats command."""

    stats = VectorStoreManager.get_stats(Path(config.get("index_dir", "storage/index")))
    stats["status"] = "ok"
    return stats


@activity.defn
async def quit_command_activity() -> Dict[str, Any]:
    """Return the terminal payload used when the user quits the session."""

    return {
        "status": "quit",
        "command": "quit",
        "message": "Session terminated by user",
    }


__all__ = [
    "ingest_command_activity",
    "question_command_activity",
    "quit_command_activity",
    "stats_command_activity",
]
