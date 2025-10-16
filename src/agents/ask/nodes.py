"""LangGraph node implementations for the ask workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Set

from src.ingestion.vector_store import VectorStoreManager

from .state import (
    AskAgentConfig,
    AskAgentState,
    AskGraphState,
    ConversationTurn,
    ReasoningStep,
    RetrievedDocument,
)

MAX_CONTEXT_CHARS = 1200
MIN_CONTEXT_RESULTS = 2


def _normalize_question(text: str) -> str:
    return " ".join(text.strip().split())


def _generate_subquestions(question: str, max_subquestions: int) -> List[str]:
    """Split a composite question into analyzable chunks."""

    question = question.strip()
    if not question:
        return []

    candidates = [_normalize_question(segment) for segment in re.split(r"[?!.]", question) if segment.strip()]
    if not candidates:
        return [question]

    refined: List[str] = []
    for item in candidates:
        if not item.endswith("?"):
            refined.append(f"{item}?")
        else:
            refined.append(item)
        if len(refined) >= max_subquestions:
            break
    if not refined:
        refined.append(question if question.endswith("?") else f"{question}?")
    return refined


@dataclass
class NodeDependencies:
    """Factories shared across nodes."""

    vector_store_factory: Callable[[AskAgentConfig], VectorStoreManager]
    llm_callable: Callable[[AskAgentState, str], str]
    on_step: Optional[Callable[[str, AskAgentState], None]] = None


def _notify_step(deps: NodeDependencies, label: str, agent_state: AskAgentState) -> None:
    if deps.on_step is not None:
        deps.on_step(label, agent_state)


def build_question_analyzer(deps: NodeDependencies) -> Callable[[AskGraphState], AskGraphState]:
    def _node(state: AskGraphState) -> AskGraphState:
        agent_state = AskAgentState.from_graph_state(state)
        question = _normalize_question(agent_state.question)
        agent_state.question = question

        subquestions = _generate_subquestions(question, agent_state.config.max_subquestions)
        agent_state.sub_questions = subquestions
        agent_state.conversation.append(
            ConversationTurn(role="user", content=question, metadata={"type": "question"})
        )
        agent_state.reasoning.append(
            ReasoningStep(
                label="analysis",
                detail=f"Generated {len(subquestions)} sub-question(s) for retrieval.",
                metadata={"sub_questions": subquestions},
            )
        )
        _notify_step(deps, "analysis", agent_state)
        return agent_state.to_graph_state()

    return _node


def build_retriever(deps: NodeDependencies) -> Callable[[AskGraphState], AskGraphState]:
    def _node(state: AskGraphState) -> AskGraphState:
        agent_state = AskAgentState.from_graph_state(state)
        manager = deps.vector_store_factory(agent_state.config)
        prompts = agent_state.sub_questions or [agent_state.question]
        try:
            retrieved = manager.multi_query(
                prompts,
                top_k=agent_state.config.top_k,
                deduplicate=True,
                neighbor_span=max(agent_state.config.neighbor_span, 0),
                dedupe_fields=agent_state.config.dedupe_fields,
            )
        except Exception as exc:  # pragma: no cover - defensive against backend failure
            agent_state.error = str(exc)
            agent_state.reasoning.append(
                ReasoningStep(
                    label="retrieval_error",
                    detail="Vector store query failed.",
                    metadata={"exception": str(exc)},
                )
            )
            return agent_state.to_graph_state()

        reranked = VectorStoreManager.rerank_documents(
            retrieved,
            max_per_source=max(agent_state.config.neighbor_span + 1, 2),
        )
        merged = VectorStoreManager.merge_adjacent_documents(reranked)

        documents: List[RetrievedDocument] = []
        for item in merged:
            raw_text = item.get("text", "") or ""
            trimmed = raw_text[:MAX_CONTEXT_CHARS]
            if len(raw_text) > MAX_CONTEXT_CHARS:
                trimmed = trimmed.rstrip() + "..."
            documents.append(
                RetrievedDocument(
                    text=trimmed,
                    score=float(item.get("score") or 0.0),
                    metadata=item.get("metadata") or {},
                )
            )
        agent_state.retrieved_documents = documents
        unique_sources: Set[str] = {
            doc.metadata.get("source_path") or doc.metadata.get("file_name") or "unknown" for doc in documents
        }
        agent_state.reasoning.append(
            ReasoningStep(
                label="retrieval",
                detail=f"Retrieved {len(documents)} document chunk(s) for reasoning.",
                metadata={
                    "top_k": agent_state.config.top_k,
                    "neighbor_span": agent_state.config.neighbor_span,
                    "sources": len(unique_sources),
                },
            )
        )
        _notify_step(deps, "retrieval", agent_state)
        return agent_state.to_graph_state()

    return _node


def build_answer_grader(deps: NodeDependencies) -> Callable[[AskGraphState], AskGraphState]:
    def _node(state: AskGraphState) -> AskGraphState:
        agent_state = AskAgentState.from_graph_state(state)
        answer = (agent_state.answer or "").strip()
        citations = set(agent_state.citations)

        needs_revision = False
        needs_context = not answer

        if (
            answer
            and agent_state.config.min_citations > 0
            and len(citations) < agent_state.config.min_citations
        ):
            needs_revision = True

        agent_state.needs_more_context = agent_state.needs_more_context or needs_context
        agent_state.needs_answer_revision = needs_revision
        agent_state.reasoning.append(
            ReasoningStep(
                label="grade_answer",
                detail="Evaluated draft answer for completeness and citations.",
                metadata={
                    "has_answer": bool(answer),
                    "citation_count": len(citations),
                    "needs_more_context": needs_context,
                    "needs_answer_revision": needs_revision,
                },
            )
        )
        _notify_step(deps, "grade_answer", agent_state)
        return agent_state.to_graph_state()

    return _node


def build_document_grader(deps: NodeDependencies) -> Callable[[AskGraphState], AskGraphState]:
    def _node(state: AskGraphState) -> AskGraphState:
        agent_state = AskAgentState.from_graph_state(state)
        documents = agent_state.retrieved_documents
        unique_sources = {
            doc.metadata.get("source_path") or doc.metadata.get("file_name") or "unknown" for doc in documents
        }

        if len(documents) < max(MIN_CONTEXT_RESULTS, agent_state.config.min_citations):
            agent_state.needs_more_context = True

        agent_state.reasoning.append(
            ReasoningStep(
                label="grade_documents",
                detail="Checked retrieved context diversity.",
                metadata={
                    "document_count": len(documents),
                    "unique_sources": len(unique_sources),
                    "needs_more_context": agent_state.needs_more_context,
                },
            )
        )
        _notify_step(deps, "grade_documents", agent_state)
        return agent_state.to_graph_state()

    return _node


def build_reasoner(deps: NodeDependencies) -> Callable[[AskGraphState], AskGraphState]:
    def _node(state: AskGraphState) -> AskGraphState:
        agent_state = AskAgentState.from_graph_state(state)
        context = VectorStoreManager.format_documents_for_llm(
            [doc.model_dump() for doc in agent_state.retrieved_documents]
        )
        if agent_state.error:
            return agent_state.to_graph_state()

        if not context.strip():
            agent_state.answer = "I could not locate relevant context in the vector store."
            agent_state.reasoning.append(
                ReasoningStep(
                    label="reasoner",
                    detail="No context available; returned fallback answer.",
                    metadata={},
                )
            )
            return agent_state.to_graph_state()

        answer = deps.llm_callable(agent_state, context)

        agent_state.answer = answer.strip()
        agent_state.reasoning.append(
            ReasoningStep(
                label="reasoner",
                detail="Generated a draft response using the Ollama model.",
                metadata={
                    "context_tokens": len(context.split()),
                    "answer_tokens": len(answer.split()),
                },
            )
        )
        _notify_step(deps, "reasoner", agent_state)
        return agent_state.to_graph_state()

    return _node


def _extract_citations(answer: str) -> List[str]:
    pattern = re.compile(r"\[(\d+)\]")
    return sorted(set(pattern.findall(answer)))


def build_response_generator(deps: NodeDependencies) -> Callable[[AskGraphState], AskGraphState]:
    def _node(state: AskGraphState) -> AskGraphState:
        agent_state = AskAgentState.from_graph_state(state)
        answer = agent_state.answer or ""
        citations = _extract_citations(answer)
        agent_state.citations = citations
        agent_state.conversation.append(
            ConversationTurn(
                role="assistant",
                content=answer,
                metadata={"citations": citations},
            )
        )
        agent_state.reasoning.append(
            ReasoningStep(
                label="response",
                detail="Formatted final response with extracted citations.",
                metadata={"citations": citations},
            )
        )
        _notify_step(deps, "response", agent_state)
        return agent_state.to_graph_state()

    return _node


def build_query_rewriter(deps: NodeDependencies) -> Callable[[AskGraphState], AskGraphState]:
    def _node(state: AskGraphState) -> AskGraphState:
        agent_state = AskAgentState.from_graph_state(state)
        if not agent_state.config.reflection_enabled:
            agent_state.needs_more_context = False
            agent_state.needs_answer_revision = False
            return agent_state.to_graph_state()

        if agent_state.reflection_count >= agent_state.config.max_reflections:
            agent_state.needs_more_context = False
            agent_state.needs_answer_revision = False
            agent_state.reasoning.append(
                ReasoningStep(
                    label="rewrite_query",
                    detail="Reached reflection limit; proceeding with current context.",
                    metadata={"reflection_count": agent_state.reflection_count},
                )
            )
            _notify_step(deps, "rewrite_query", agent_state)
            return agent_state.to_graph_state()

        followups: List[str] = []
        if agent_state.needs_more_context:
            followups.append(
                f"Provide additional background and supporting facts for: {agent_state.question}"
            )
        if agent_state.needs_answer_revision:
            followups.append(f"List the key evidence with citations for: {agent_state.question}")
        if not followups:
            followups.append(agent_state.question)

        agent_state.reflection_count += 1
        agent_state.sub_questions = [agent_state.question] + followups
        agent_state.needs_more_context = False
        agent_state.needs_answer_revision = False
        agent_state.reflection_notes.extend(followups)
        if len(agent_state.reflection_notes) > 5:
            agent_state.reflection_notes = agent_state.reflection_notes[-5:]
        agent_state.reasoning.append(
            ReasoningStep(
                label="rewrite_query",
                detail="Generated follow-up queries for additional retrieval.",
                metadata={
                    "reflection_count": agent_state.reflection_count,
                    "followups": followups,
                },
            )
        )
        _notify_step(deps, "rewrite_query", agent_state)
        return agent_state.to_graph_state()

    return _node


MAX_CONTEXT_CHARS = 1200
