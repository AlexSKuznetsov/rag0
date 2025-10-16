import pytest
from src.utils.cli import (
    RICH_AVAILABLE,
    ResponseView,
    build_response_view,
    create_cli_layout,
)

rich = pytest.importorskip("rich")


@pytest.mark.skipif(not RICH_AVAILABLE, reason="Rich console not available")
def test_create_cli_layout_regions() -> None:
    layout = create_cli_layout()
    region_names = {child.name for child in layout.children}
    assert {"header", "body", "footer"} <= region_names

    body_sections = {child.name for child in layout["body"].children}
    assert {"chat", "response"} <= body_sections


@pytest.mark.skipif(not RICH_AVAILABLE, reason="Rich console not available")
def test_build_response_view_for_ask_success() -> None:
    result = {
        "command": "ask",
        "status": "success",
        "result": {
            "answer": "Test answer referencing [1].",
            "citations": ["[1] Example citation"],
            "documents": [
                {
                    "text": "Example snippet from the source document.",
                    "score": 0.987,
                    "metadata": {"source_path": "doc1.pdf"},
                }
            ],
            "reasoning": [
                {"label": "retrieve", "detail": "Fetched relevant documents.", "metadata": {"count": 3}},
                {"label": "answer", "detail": "Drafted final answer.", "metadata": {}},
            ],
        },
    }

    view = build_response_view(result)

    assert isinstance(view, ResponseView)
    assert view.command == "ask"
    assert view.status == "success"
    assert view.metadata["document_count"] == 1
    assert view.metadata["reasoning_count"] == 2


@pytest.mark.skipif(not RICH_AVAILABLE, reason="Rich console not available")
def test_build_response_view_warns_without_citations() -> None:
    result = {
        "command": "ask",
        "status": "success",
        "result": {
            "answer": "",
            "citations": [],
            "documents": [],
            "reasoning": [],
        },
    }

    view = build_response_view(result)

    assert view.status == "warn"
    assert view.metadata.get("message") == "No answer returned."
