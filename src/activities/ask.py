"""Activities that power question answering against the vector index."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from temporalio import activity

from ..agents.ask import AskAgentConfig, AskAgentState, LangGraphAskAgent

ASK_AGENT = LangGraphAskAgent()
logger = logging.getLogger(__name__)


@activity.defn
async def answer_query_activity(
    question: str,
    index_dir: str = "storage/index",
    top_k: int = AskAgentConfig().top_k,
    ollama_model: str = AskAgentConfig().ollama_model,
    ollama_base_url: str = AskAgentConfig().ollama_base_url,
    temperature: float = AskAgentConfig().temperature,
    max_subquestions: int = AskAgentConfig().max_subquestions,
    neighbor_span: int = AskAgentConfig().neighbor_span,
    reflection_enabled: bool = AskAgentConfig().reflection_enabled,
    max_reflections: int = AskAgentConfig().max_reflections,
    min_citations: int = AskAgentConfig().min_citations,
    parent_workflow_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the LangGraph ask agent to produce an answer."""

    config = AskAgentConfig(
        index_dir=index_dir,
        top_k=top_k,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        temperature=temperature,
        max_subquestions=max_subquestions,
        neighbor_span=neighbor_span,
        reflection_enabled=reflection_enabled,
        max_reflections=max_reflections,
        min_citations=min_citations,
    )

    progress_events: List[Dict[str, Any]] = []
    loop = asyncio.get_running_loop()

    parent_handle = None
    if parent_workflow_id:
        try:
            client = activity.client()
            parent_handle = client.get_workflow_handle(
                parent_workflow_id,
                run_id=parent_run_id,
            )
        except Exception as exc:  # pragma: no cover - best-effort signal hookup
            logger.debug("Unable to acquire parent workflow handle: %s", exc)
            parent_handle = None

    def _record_progress(event: Dict[str, Any]) -> None:
        payload = {
            "label": str(event.get("label", "")),
            "detail": str(event.get("detail", "")),
            "metadata": dict(event.get("metadata") or {}),
        }
        progress_events.append(payload)
        loop.call_soon_threadsafe(activity.heartbeat, payload)

        if parent_handle is not None:

            def _schedule_signal(event_payload: Dict[str, Any]) -> None:
                async def _send() -> None:
                    try:
                        await parent_handle.signal(
                            "push_progress",
                            {
                                "command": "ask",
                                "event": event_payload,
                            },
                        )
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.debug("Failed to signal parent workflow progress: %s", exc)

                loop.create_task(_send())

            loop.call_soon_threadsafe(_schedule_signal, payload)

    def _on_step(label: str, agent_state: AskAgentState) -> None:
        try:
            last_step = agent_state.reasoning[-1] if agent_state.reasoning else None
            detail = (last_step.detail if last_step else "").strip()
            metadata = dict(last_step.metadata) if last_step else {}
        except Exception:  # pragma: no cover - defensive guard against unexpected state
            detail = ""
            metadata = {}

        event = {
            "label": label,
            "detail": detail,
            "metadata": metadata,
        }
        _record_progress(event)

    state = await asyncio.to_thread(ASK_AGENT.run, question, config, _on_step)

    formatted_documents: List[Dict[str, Any]] = [
        {
            "text": doc.text,
            "score": doc.score,
            "metadata": doc.metadata,
        }
        for doc in state.retrieved_documents
    ]
    return {
        "answer": state.answer,
        "citations": state.citations,
        "documents": formatted_documents,
        "reasoning": [step.model_dump() for step in state.reasoning],
        "conversation": [turn.model_dump() for turn in state.conversation],
        "progress": progress_events,
    }


__all__ = ["answer_query_activity"]
