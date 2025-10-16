"""Pydantic models and helpers for LangGraph-based ask workflow state."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single conversational exchange."""

    role: Literal["user", "assistant", "tool"]
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    """Representation of a retrieved vector store document."""

    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReasoningStep(BaseModel):
    """Represents intermediate reasoning or a tool invocation."""

    label: str
    detail: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AskAgentConfig(BaseModel):
    """Configuration for the ask agent graph."""

    index_dir: str = "storage/index"
    top_k: int = 6
    neighbor_span: int = 1
    dedupe_fields: List[str] = Field(default_factory=lambda: ["source_path", "chunk_id"])
    ollama_model: str = "qwen3:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    reasoning_steps: int = 1
    max_subquestions: int = 3
    reflection_enabled: bool = True
    max_reflections: int = 2
    min_citations: int = 1


class AskAgentState(BaseModel):
    """State shared across LangGraph nodes."""

    question: str
    sub_questions: List[str] = Field(default_factory=list)
    conversation: List[ConversationTurn] = Field(default_factory=list)
    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)
    reasoning: List[ReasoningStep] = Field(default_factory=list)
    answer: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    reflection_count: int = 0
    needs_more_context: bool = False
    needs_answer_revision: bool = False
    reflection_notes: List[str] = Field(default_factory=list)
    config: AskAgentConfig = Field(default_factory=AskAgentConfig)

    def to_graph_state(self) -> "AskGraphState":
        """Return a LangGraph compatible dictionary."""

        return {
            "question": self.question,
            "sub_questions": list(self.sub_questions),
            "conversation": [turn.model_dump() for turn in self.conversation],
            "retrieved_documents": [doc.model_dump() for doc in self.retrieved_documents],
            "reasoning": [step.model_dump() for step in self.reasoning],
            "answer": self.answer,
            "citations": list(self.citations),
            "error": self.error,
            "reflection_count": self.reflection_count,
            "needs_more_context": self.needs_more_context,
            "needs_answer_revision": self.needs_answer_revision,
            "reflection_notes": list(self.reflection_notes),
            "config": self.config.model_dump(),
        }

    @classmethod
    def from_graph_state(cls, state: "AskGraphState") -> "AskAgentState":
        """Instantiate from a LangGraph state payload."""

        return cls(
            question=state.get("question", ""),
            sub_questions=list(state.get("sub_questions", [])),
            conversation=[ConversationTurn(**turn) for turn in state.get("conversation", [])],
            retrieved_documents=[RetrievedDocument(**doc) for doc in state.get("retrieved_documents", [])],
            reasoning=[ReasoningStep(**step) for step in state.get("reasoning", [])],
            answer=state.get("answer"),
            citations=list(state.get("citations", [])),
            error=state.get("error"),
            reflection_count=int(state.get("reflection_count", 0)),
            needs_more_context=bool(state.get("needs_more_context", False)),
            needs_answer_revision=bool(state.get("needs_answer_revision", False)),
            reflection_notes=list(state.get("reflection_notes", [])),
            config=AskAgentConfig(**state.get("config", {})),
        )


class AskGraphState(TypedDict, total=False):
    """TypedDict representation consumed by LangGraph."""

    question: str
    sub_questions: List[str]
    conversation: List[Dict[str, Any]]
    retrieved_documents: List[Dict[str, Any]]
    reasoning: List[Dict[str, Any]]
    answer: Optional[str]
    citations: List[str]
    error: Optional[str]
    reflection_count: int
    needs_more_context: bool
    needs_answer_revision: bool
    reflection_notes: List[str]
    config: Dict[str, Any]
