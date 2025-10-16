from __future__ import annotations

from pathlib import Path

from src.ingestion.storage import load_parsed_markdown, store_parsed_markdown


def test_store_and_load_preserves_chunk_metadata(tmp_path: Path) -> None:
    metadata = {
        "file_name": "example.md",
        "chunk_descriptors": [
            {
                "chunk_id": "example-chunk-0001",
                "chunk_index": 0,
                "paragraph_start": 0,
                "paragraph_end": 1,
                "page_start": None,
                "page_end": None,
                "token_count": 42,
            }
        ],
    }
    content = {
        "markdown": "First paragraph.\n\nSecond paragraph.",
        "paragraphs": [
            {"text": "First paragraph."},
            {"text": "Second paragraph."},
        ],
        "chunks": [
            {
                "chunk_id": "example-chunk-0001",
                "text": "First paragraph.\n\nSecond paragraph.",
                "token_count": 42,
                "chunk_index": 0,
                "paragraph_start": 0,
                "paragraph_end": 1,
                "page_start": None,
                "page_end": None,
            }
        ],
        "warnings": [],
    }

    paths = store_parsed_markdown(metadata, content, tmp_path)

    payload = load_parsed_markdown(paths["markdown_path"], paths["metadata_path"])

    assert payload["metadata"]["chunk_descriptors"][0]["chunk_id"] == "example-chunk-0001"
    assert payload["content"]["chunks"][0]["chunk_id"] == "example-chunk-0001"
    assert payload["content"]["markdown"] == content["markdown"]
