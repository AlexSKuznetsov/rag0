from __future__ import annotations

from src.ingestion.chunking import ChunkingConfig, generate_chunks


def test_generate_chunks_respects_overlap() -> None:
    paragraphs = [
        {"text": "one two three four five", "page": 1},
        {"text": "six seven eight nine ten", "page": 1},
        {"text": "eleven twelve thirteen fourteen fifteen", "page": 2},
    ]
    base_metadata = {"file_name": "sample.md"}
    config = ChunkingConfig(chunk_size_tokens=8, chunk_overlap_tokens=3, merge_threshold_tokens=2)

    chunks = generate_chunks(paragraphs, base_metadata, config)

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "sample.md-chunk-0001"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["paragraph_start"] == 0
    assert chunks[0]["paragraph_end"] == 1
    assert "\n\n" in chunks[0]["text"]

    assert chunks[1]["chunk_id"] == "sample.md-chunk-0002"
    assert chunks[1]["chunk_index"] == 1
    assert chunks[1]["paragraph_start"] == 1
    assert chunks[1]["paragraph_end"] == 2
    assert chunks[1]["page_end"] == 2
    assert chunks[1]["metadata"]["chunk_id"] == "sample.md-chunk-0002"
