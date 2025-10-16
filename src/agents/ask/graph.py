"""LangGraph orchestration helpers for the ask workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from src.ingestion.vector_store import VectorStoreManager

from .nodes import (
    NodeDependencies,
    build_answer_grader,
    build_document_grader,
    build_query_rewriter,
    build_question_analyzer,
    build_reasoner,
    build_response_generator,
    build_retriever,
)
from .state import AskAgentConfig, AskAgentState, AskGraphState

logger = logging.getLogger(__name__)


class OllamaResponder:
    """Wrapper around a LangChain Ollama chat model with graceful fallback."""

    def __init__(self, config: AskAgentConfig) -> None:
        self._config = config
        self._chain = None
        self._setup_chain()

    def _setup_chain(self) -> None:
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_ollama import ChatOllama
        except ImportError:
            logger.debug("LangChain modules not available; using fallback responder.")
            return

        system_prompt = (
            "You are the RAG0 local assistant. "
            "Answer with concise paragraphs and cite supporting chunks using [n] "
            "where n references the numbered context blocks."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("history"),
                (
                    "human",
                    "Question: {question}\nContext:\n{context}\nInstructions: {instructions}",
                ),
            ]
        )
        llm = ChatOllama(
            model=self._config.ollama_model,
            base_url=self._config.ollama_base_url,
            temperature=self._config.temperature,
        )
        parser = StrOutputParser()
        self._chain = prompt | llm | parser

    def __call__(self, state: AskAgentState, context: str) -> str:
        if self._chain is None:
            return self._fallback_response(state, context)

        history = [{"role": turn.role, "content": turn.content} for turn in state.conversation]
        instructions = "Use numbered citations and keep answers under 200 words."
        if state.reflection_notes:
            focus = "; ".join(state.reflection_notes[-3:])
            instructions += f" Address the following focus areas: {focus}."
        try:
            result = self._chain.invoke(
                {
                    "question": state.question,
                    "context": context,
                    "instructions": instructions,
                    "history": history,
                }
            )
        except Exception as exc:  # pragma: no cover - network/model failures
            logger.warning("Ollama responder failed; using fallback: %s", exc)
            return self._fallback_response(state, context)

        if isinstance(result, str):
            return result
        if isinstance(result, Dict):
            return result.get("content") or ""
        return str(result)

    def _fallback_response(self, state: AskAgentState, context: str) -> str:
        if not state.retrieved_documents:
            return "I do not have enough information to answer that yet."

        fragments: List[str] = []
        for idx, doc in enumerate(state.retrieved_documents[:3], start=1):
            snippet_lines = [line.strip() for line in doc.text.splitlines() if line.strip()]
            if not snippet_lines:
                continue
            metadata = doc.metadata
            page_start = metadata.get("page_start") or metadata.get("page")
            page_end = metadata.get("page_end")
            if page_start is not None and page_end is not None and page_start != page_end:
                location = f"pages {page_start}-{page_end}"
            elif page_start is not None:
                location = f"page {page_start}"
            else:
                location = "context"
            fragments.append(f"[{idx}] {snippet_lines[0]} ({location})")

        if not fragments:
            return "I do not have enough information to answer that yet."

        joined = " ".join(fragments)
        return (
            f"Key context snippets: {joined}\n"
            "This is a fallback response generated without calling Ollama."
        )


def _default_vector_store_factory(config: AskAgentConfig) -> VectorStoreManager:
    storage_dir = Path(config.index_dir)
    return VectorStoreManager(storage_dir=storage_dir, collection_name="rag0")


class LangGraphAskAgent:
    """Encapsulates LangGraph execution for the ask workflow."""

    def __init__(
        self,
        vector_store_factory: Callable[[AskAgentConfig], VectorStoreManager] = _default_vector_store_factory,
        responder_factory: Callable[[AskAgentConfig], OllamaResponder] = OllamaResponder,
    ) -> None:
        self._vector_store_factory = vector_store_factory
        self._responder_factory = responder_factory

    def _build_graph(
        self, config: AskAgentConfig, *, on_step: Optional[Callable[[str, AskAgentState], None]] = None
    ):
        vector_manager_cache: Dict[str, VectorStoreManager] = {}

        def vector_factory(cfg: AskAgentConfig) -> VectorStoreManager:
            cache_key = cfg.index_dir
            if cache_key not in vector_manager_cache:
                vector_manager_cache[cache_key] = self._vector_store_factory(cfg)
            return vector_manager_cache[cache_key]

        responder = self._responder_factory(config)
        deps = NodeDependencies(
            vector_store_factory=vector_factory,
            llm_callable=responder,
            on_step=on_step,
        )
        graph_builder = StateGraph(AskGraphState)
        graph_builder.add_node("analysis", build_question_analyzer(deps))
        graph_builder.add_node("retrieval", build_retriever(deps))
        graph_builder.add_node("reasoner", build_reasoner(deps))
        graph_builder.add_node("grade_answer", build_answer_grader(deps))
        graph_builder.add_node("grade_documents", build_document_grader(deps))
        graph_builder.add_node("rewrite_query", build_query_rewriter(deps))
        graph_builder.add_node("response", build_response_generator(deps))
        graph_builder.add_edge(START, "analysis")
        graph_builder.add_edge("analysis", "retrieval")
        graph_builder.add_edge("retrieval", "reasoner")
        graph_builder.add_edge("reasoner", "grade_answer")
        graph_builder.add_edge("grade_answer", "grade_documents")

        def _route(state: AskGraphState) -> str:
            agent_state = AskAgentState.from_graph_state(state)
            if (
                not agent_state.config.reflection_enabled
                or agent_state.reflection_count >= agent_state.config.max_reflections
                or not (agent_state.needs_more_context or agent_state.needs_answer_revision)
            ):
                return "response"
            return "rewrite_query"

        graph_builder.add_conditional_edges(
            "grade_documents",
            _route,
            {
                "rewrite_query": "rewrite_query",
                "response": "response",
            },
        )
        graph_builder.add_edge("rewrite_query", "retrieval")
        graph_builder.add_edge("response", END)
        return graph_builder.compile()

    def run(
        self,
        question: str,
        config: Optional[AskAgentConfig] = None,
        on_step: Optional[Callable[[str, AskAgentState], None]] = None,
    ) -> AskAgentState:
        agent_config = config or AskAgentConfig()
        initial_state = AskAgentState(question=question, config=agent_config)

        graph = self._build_graph(agent_config, on_step=on_step)
        result_state = graph.invoke(initial_state.to_graph_state())
        return AskAgentState.from_graph_state(result_state)
