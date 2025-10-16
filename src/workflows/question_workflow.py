"""Temporal workflow that routes a user question to the vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from temporalio import workflow

ANSWER_QUERY_ACTIVITY = "answer_query_activity"


@dataclass
class QuestionWorkflowInput:
    """Input payload for the Q&A workflow."""

    question: str
    index_dir: str = "storage/index"
    top_k: int = 6
    ollama_model: str = "qwen3:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    max_subquestions: int = 3
    neighbor_span: int = 1
    reflection_enabled: bool = True
    max_reflections: int = 2
    min_citations: int = 1

    def to_activity_args(self) -> tuple[Any, ...]:
        return (
            self.question,
            self.index_dir,
            self.top_k,
            self.ollama_model,
            self.ollama_base_url,
            self.temperature,
            self.max_subquestions,
            self.neighbor_span,
            self.reflection_enabled,
            self.max_reflections,
            self.min_citations,
        )


@workflow.defn
class QuestionWorkflow:
    """Workflow that routes a question to the vector store."""

    @workflow.run
    async def run(self, payload: QuestionWorkflowInput) -> Dict[str, Any]:
        info = workflow.info()
        parent = info.parent
        parent_workflow_id = parent.workflow_id if parent else None
        parent_run_id = parent.run_id if parent else None

        activity_args = payload.to_activity_args() + (parent_workflow_id, parent_run_id)
        response = await workflow.execute_activity(
            ANSWER_QUERY_ACTIVITY,
            args=activity_args,
            schedule_to_close_timeout=workflow.timedelta(minutes=10),
        )
        return response


__all__ = ["QuestionWorkflow", "QuestionWorkflowInput"]
