from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, cast

from llama_index.core.schema import QueryBundle
from src.agents.ask import AskAgentConfig, AskAgentState, LangGraphAskAgent
from src.ingestion.vector_store import VectorStoreManager
from src.workflows import QuestionWorkflowInput


class StubVectorStore:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = responses
        self.requested: List[
            Tuple[Tuple[str, ...], int, bool, int, Tuple[str, ...]]
        ] = []

    def multi_query(
        self,
        prompts: Sequence[str],
        top_k: int,
        deduplicate: bool,
        neighbor_span: int = 0,
        dedupe_fields: Sequence[str] | None = None,
    ):
        dedupe_tuple = tuple(dedupe_fields or [])
        self.requested.append((tuple(prompts), top_k, deduplicate, neighbor_span, dedupe_tuple))
        return self.responses


class StubResponder:
    def __init__(self) -> None:
        self.prompts: List[Tuple[str, str]] = []

    def __call__(self, state: AskAgentState, context: str) -> str:
        self.prompts.append((state.question, context))
        return "Stub answer with cite [1]"


class ReflectiveResponder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state: AskAgentState, context: str) -> str:
        self.calls += 1
        if state.reflection_count == 0:
            return ""
        return "Improved answer with cite [1]"


def test_langgraph_agent_uses_stubs(monkeypatch):
    responses = [
        {
            "text": "RAG0 is a retrieval augmented generation prototype.",
            "metadata": {"source_path": "doc.md", "chunk_id": "doc-1", "chunk_index": 0},
        },
    ]
    stub_store = StubVectorStore(responses)
    stub_responder = StubResponder()

    def vector_factory(_config: AskAgentConfig):
        return stub_store

    def responder_factory(_config: AskAgentConfig):
        return stub_responder

    agent = LangGraphAskAgent(
        vector_store_factory=vector_factory,
        responder_factory=responder_factory,
    )
    config = AskAgentConfig(top_k=2, max_subquestions=2, reflection_enabled=False)
    result = agent.run("Explain RAG0 goals.", config)

    assert result.answer == "Stub answer with cite [1]"
    assert result.citations == ["1"]
    assert stub_store.requested, "Expected vector store to receive query."
    assert stub_responder.prompts, "Responder should be invoked."


def test_state_round_trip():
    config = AskAgentConfig(ollama_model="mistral")
    state = AskAgentState(
        question="What is LangGraph?",
        sub_questions=["What is LangGraph?"],
        config=config,
    )
    graph_state = state.to_graph_state()
    restored = AskAgentState.from_graph_state(graph_state)
    assert restored.question == state.question
    assert restored.config.ollama_model == "mistral"


def test_format_documents_for_llm():
    docs = [
        {
            "text": "Context snippet",
            "metadata": {"file_name": "doc.md", "page": 2},
        }
    ]
    formatted = VectorStoreManager.format_documents_for_llm(docs)
    assert "[1] doc.md (page 2)" in formatted
    assert "Context snippet" in formatted


def test_question_workflow_input_to_args():
    payload = QuestionWorkflowInput(
        question="What is LangGraph?",
        index_dir="storage/index",
        top_k=4,
        ollama_model="mistral",
        temperature=0.3,
        max_subquestions=5,
        neighbor_span=2,
        reflection_enabled=False,
        max_reflections=3,
        min_citations=2,
    )
    args = payload.to_activity_args()
    assert args[0] == "What is LangGraph?"
    assert args[2] == 4
    assert args[3] == "mistral"
    assert args[4] == payload.ollama_base_url
    assert args[5] == 0.3
    assert args[6] == 5
    assert args[7] == 2
    assert args[8] is False
    assert args[9] == 3
    assert args[10] == 2


def test_reflection_loop_triggers_additional_retrieval():
    responses = [
        {
            "text": "RAG0 overview chunk.",
            "metadata": {"source_path": "doc.md", "chunk_id": "doc-1", "chunk_index": 0},
            "score": 0.1,
        }
    ]
    stub_store = StubVectorStore(responses)
    responder = ReflectiveResponder()

    def vector_factory(_config: AskAgentConfig):
        return stub_store

    def responder_factory(_config: AskAgentConfig):
        return responder

    agent = LangGraphAskAgent(
        vector_store_factory=vector_factory,
        responder_factory=responder_factory,
    )
    config = AskAgentConfig(top_k=2, max_subquestions=2, max_reflections=2, min_citations=1)
    result = agent.run("Explain RAG0 goals.", config)

    assert responder.calls >= 2
    assert result.reflection_count == config.max_reflections
    assert len(stub_store.requested) >= 2
    assert (result.answer or "").strip(), "Expected a non-empty answer after reflection"


def test_vector_store_retriever_input_with_embedding(monkeypatch):
    manager = object.__new__(VectorStoreManager)

    def fake_create_query_bundle(prompt: str) -> QueryBundle:
        return QueryBundle(prompt, embedding=[0.1, 0.2])

    cast(Any, manager)._create_query_bundle = fake_create_query_bundle
    bundle = VectorStoreManager._retriever_input(cast(VectorStoreManager, manager), "question?")
    assert isinstance(bundle, QueryBundle)
