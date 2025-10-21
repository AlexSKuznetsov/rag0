"""Application configuration dataclasses."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pydantic.dataclasses import dataclass

DEFAULT_PARSED_DIR = "parsed"
DEFAULT_INDEX_DIR = "storage/index"
DEFAULT_TEMPORAL_ADDRESS = "127.0.0.1:7233"
DEFAULT_TEMPORAL_NAMESPACE = "default"
DEFAULT_TEMPORAL_TASK_QUEUE = "rag0"
DEFAULT_WORKFLOW_ID_PREFIX: Optional[str] = None
DEFAULT_OLLAMA_MODEL = os.environ.get("RAG0_OLLAMA_MODEL", "qwen3:4b")
DEFAULT_OLLAMA_BASE_URL = os.environ.get("RAG0_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_ASK_TOP_K = int(os.environ.get("RAG0_ASK_TOP_K", "6"))
DEFAULT_ASK_MAX_SUBQUESTIONS = int(os.environ.get("RAG0_ASK_MAX_SUBQUESTIONS", "3"))
DEFAULT_ASK_NEIGHBOR_SPAN = int(os.environ.get("RAG0_ASK_NEIGHBOR_SPAN", "1"))
DEFAULT_ASK_REFLECTION_ENABLED = os.environ.get("RAG0_ASK_REFLECTION_ENABLED", "1").lower() not in {
    "0",
    "false",
    "no",
}
DEFAULT_ASK_MAX_REFLECTIONS = int(os.environ.get("RAG0_ASK_MAX_REFLECTIONS", "2"))
DEFAULT_ASK_MIN_CITATIONS = int(os.environ.get("RAG0_ASK_MIN_CITATIONS", "1"))
DEFAULT_ASK_TEMPERATURE = float(os.environ.get("RAG0_ASK_TEMPERATURE", "0.0"))
DEFAULT_CHUNK_SIZE = int(os.environ.get("RAG0_CHUNK_SIZE", "700"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("RAG0_CHUNK_OVERLAP", "150"))
DEFAULT_CHUNK_MERGE_THRESHOLD = int(os.environ.get("RAG0_CHUNK_MERGE_THRESHOLD", "60"))


@dataclass
class WorkflowConfig:
    """Typed configuration for launching the interactive workflow."""

    parsed_dir: str = DEFAULT_PARSED_DIR
    index_dir: str = DEFAULT_INDEX_DIR
    address: str = DEFAULT_TEMPORAL_ADDRESS
    namespace: str = DEFAULT_TEMPORAL_NAMESPACE
    task_queue: str = DEFAULT_TEMPORAL_TASK_QUEUE
    workflow_id_prefix: Optional[str] = DEFAULT_WORKFLOW_ID_PREFIX
    ask_top_k: int = DEFAULT_ASK_TOP_K
    ask_max_subquestions: int = DEFAULT_ASK_MAX_SUBQUESTIONS
    ask_neighbor_span: int = DEFAULT_ASK_NEIGHBOR_SPAN
    ask_reflection_enabled: bool = DEFAULT_ASK_REFLECTION_ENABLED
    ask_max_reflections: int = DEFAULT_ASK_MAX_REFLECTIONS
    ask_min_citations: int = DEFAULT_ASK_MIN_CITATIONS
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ask_temperature: float = DEFAULT_ASK_TEMPERATURE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    chunk_merge_threshold: int = DEFAULT_CHUNK_MERGE_THRESHOLD

    def to_activity_payload(self) -> Dict[str, Any]:
        """Return a dict compatible with activity execution."""

        return {
            "parsed_dir": self.parsed_dir,
            "index_dir": self.index_dir,
            "ask_top_k": self.ask_top_k,
            "ollama_model": self.ollama_model,
            "ollama_base_url": self.ollama_base_url,
            "temperature": self.ask_temperature,
            "max_subquestions": self.ask_max_subquestions,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_merge_threshold": self.chunk_merge_threshold,
            "neighbor_span": self.ask_neighbor_span,
            "reflection_enabled": self.ask_reflection_enabled,
            "max_reflections": self.ask_max_reflections,
            "min_citations": self.ask_min_citations,
        }

    def workflow_id_prefix_value(self) -> str:
        """Return a sanitized workflow ID prefix."""

        prefix = (self.workflow_id_prefix or "main").strip() or "main"
        return prefix

    def copy(self, **updates: Any) -> "WorkflowConfig":
        """Return a shallow copy with optional overrides."""

        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values.update(updates)
        return WorkflowConfig(**values)
