from __future__ import annotations

from typing import Dict, List

from src.ingestion.vector_store import VectorStoreManager


class StubVectorStoreManager(VectorStoreManager):
    def __init__(self, responses: Dict[str, List[Dict[str, object]]]) -> None:  # type: ignore[call-arg]
        self.responses = responses
        self._chunk_sidecar_cache = {}

    def query(self, prompt: str, top_k: int = 3):
        return iter(self.responses.get(prompt, []))


def test_multi_query_deduplicates_chunk_ids() -> None:
    manager = StubVectorStoreManager(
        {
            "q": [
                {"text": "A", "metadata": {"source_path": "doc", "chunk_id": "doc-1"}, "score": 0.1},
                {"text": "A dup", "metadata": {"source_path": "doc", "chunk_id": "doc-1"}, "score": 0.2},
                {"text": "B", "metadata": {"source_path": "doc", "chunk_id": "doc-2"}, "score": 0.3},
            ]
        }
    )
    results = manager.multi_query(["q"], top_k=3, deduplicate=True)
    chunk_ids = [item["metadata"]["chunk_id"] for item in results]
    assert chunk_ids == ["doc-1", "doc-2"]


def test_multi_query_expands_neighbors(monkeypatch) -> None:
    base_match = {"text": "A", "metadata": {"source_path": "doc", "chunk_id": "doc-1"}, "score": 0.1}
    manager = StubVectorStoreManager({"q": [base_match]})

    def fake_expand(matches, neighbor_span):
        assert neighbor_span == 1
        return [{"text": "B", "metadata": {"source_path": "doc", "chunk_id": "doc-2"}, "score": 0.2}]

    monkeypatch.setattr(manager, "_expand_neighbors", fake_expand)
    results = manager.multi_query(["q"], top_k=1, deduplicate=True, neighbor_span=1)
    chunk_ids = [item["metadata"]["chunk_id"] for item in results]
    assert chunk_ids == ["doc-1", "doc-2"]


def test_merge_adjacent_documents_merges_contiguous_chunks() -> None:
    docs = [
        {
            "text": "First chunk.",
            "metadata": {
                "source_path": "doc",
                "chunk_index": 0,
                "chunk_id": "doc-1",
                "page_start": 1,
                "page_end": 1,
            },
            "score": 0.1,
        },
        {
            "text": "Second chunk.",
            "metadata": {
                "source_path": "doc",
                "chunk_index": 1,
                "chunk_id": "doc-2",
                "page_start": 1,
                "page_end": 1,
            },
            "score": 0.2,
        },
        {
            "text": "Other document.",
            "metadata": {
                "source_path": "other",
                "chunk_index": 0,
                "chunk_id": "other-1",
                "page_start": 2,
                "page_end": 2,
            },
            "score": 0.3,
        },
    ]
    merged = VectorStoreManager.merge_adjacent_documents(docs)
    assert len(merged) == 2
    assert "First chunk." in merged[0]["text"]
    assert "Second chunk." in merged[0]["text"]
