"""Utilities for generating document chunks with configurable overlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ChunkingConfig:
    """Configuration controlling chunk size and overlap."""

    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 150
    merge_threshold_tokens: int = 60

    def clamp(self) -> "ChunkingConfig":
        """Return a sanitized copy with non-negative values."""

        size = max(self.chunk_size_tokens, 1)
        overlap = min(max(self.chunk_overlap_tokens, 0), size - 1)
        threshold = max(self.merge_threshold_tokens, 0)
        return ChunkingConfig(
            chunk_size_tokens=size,
            chunk_overlap_tokens=overlap,
            merge_threshold_tokens=threshold,
        )


def _tokenize(text: str) -> List[str]:
    return [token for token in text.strip().split() if token]


def _merge_short_paragraphs(
    paragraphs: Sequence[Dict[str, Any]],
    paragraph_tokens: Sequence[List[str]],
    threshold: int,
) -> List[Tuple[Dict[str, Any], List[int], List[str]]]:
    """Merge adjacent paragraphs whose token counts fall below the threshold."""

    merged: List[Tuple[Dict[str, Any], List[int], List[str]]] = []
    buffer: Optional[Dict[str, Any]] = None
    buffer_indices: List[int] = []
    buffer_tokens: List[str] = []

    for index, (paragraph, tokens) in enumerate(zip(paragraphs, paragraph_tokens, strict=False)):
        text = paragraph.get("text", "").strip()
        if not text:
            continue

        if buffer is None:
            buffer = dict(paragraph)
            buffer_indices = [index]
            buffer_tokens = list(tokens)
            continue

        if len(buffer_tokens) < threshold and len(tokens) < threshold:
            buffer["text"] = f"{buffer['text'].rstrip()}\n\n{text}"
            page_candidates = [
                int(value)
                for value in (
                    buffer.get("page"),
                    buffer.get("page_end"),
                    paragraph.get("page"),
                    paragraph.get("page_end"),
                )
                if isinstance(value, (int, float))
            ]
            if page_candidates:
                buffer["page"] = min(page_candidates)
                buffer["page_end"] = max(page_candidates)
            buffer_indices.append(index)
            buffer_tokens.extend(tokens)
            continue

        merged.append((buffer, buffer_indices.copy(), list(buffer_tokens)))
        buffer = dict(paragraph)
        buffer_indices = [index]
        buffer_tokens = list(tokens)

    if buffer is not None:
        merged.append((buffer, buffer_indices.copy(), list(buffer_tokens)))

    return merged


def generate_chunks(
    paragraphs: Sequence[Dict[str, Any]],
    base_metadata: Dict[str, Any],
    config: Optional[ChunkingConfig] = None,
) -> List[Dict[str, Any]]:
    """Return chunk dictionaries derived from paragraphs and config."""

    if not paragraphs:
        return []

    cfg = (config or ChunkingConfig()).clamp()
    paragraph_tokens = [_tokenize(paragraph.get("text", "")) for paragraph in paragraphs]
    merged_segments = _merge_short_paragraphs(paragraphs, paragraph_tokens, cfg.merge_threshold_tokens)

    segments: List[Dict[str, Any]] = []
    for segment, indices, tokens in merged_segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        if not tokens:
            tokens = _tokenize(text)
        if not tokens:
            continue
        page_start = segment.get("page")
        page_end = segment.get("page_end", page_start)
        segments.append(
            {
                "text": text,
                "tokens": tokens,
                "indices": indices,
                "page_start": page_start,
                "page_end": page_end if page_end is not None else page_start,
            }
        )

    if not segments:
        return []

    chunks: List[Dict[str, Any]] = []
    segment_count = len(segments)
    start = 0
    chunk_index = 0

    while start < segment_count:
        token_total = 0
        end = start
        included_segments: List[Dict[str, Any]] = []

        while end < segment_count and (token_total < cfg.chunk_size_tokens or not included_segments):
            segment = segments[end]
            included_segments.append(segment)
            token_total += len(segment["tokens"])
            end += 1
            if token_total >= cfg.chunk_size_tokens and end - start > 1:
                break

        text = "\n\n".join(segment["text"] for segment in included_segments).strip()
        if not text:
            start = max(end, start + 1)
            continue

        chunk_index += 1
        ordinal = chunk_index - 1
        chunk_id = f"{base_metadata.get('file_name', 'document')}-chunk-{chunk_index:04d}"
        paragraph_indices = sorted({idx for segment in included_segments for idx in segment["indices"]})
        page_values = [
            segment["page_start"] for segment in included_segments if segment["page_start"] is not None
        ]
        page_values_end = [
            segment["page_end"] for segment in included_segments if segment["page_end"] is not None
        ]
        page_start = page_values[0] if page_values else None
        page_end = page_values_end[-1] if page_values_end else page_start

        chunk_metadata = {
            "chunk_id": chunk_id,
            "text": text,
            "token_count": token_total,
            "chunk_index": ordinal,
            "paragraph_start": paragraph_indices[0] if paragraph_indices else None,
            "paragraph_end": paragraph_indices[-1] if paragraph_indices else None,
            "page_start": page_start,
            "page_end": page_end,
            "metadata": {
                **base_metadata,
                "chunk_id": chunk_id,
                "chunk_index": ordinal,
                "page_start": page_start,
                "page_end": page_end,
                "paragraph_start": paragraph_indices[0] if paragraph_indices else None,
                "paragraph_end": paragraph_indices[-1] if paragraph_indices else None,
            },
        }
        chunks.append(chunk_metadata)

        if end >= segment_count:
            break

        if cfg.chunk_overlap_tokens <= 0:
            start = end
            continue

        remaining_overlap = cfg.chunk_overlap_tokens
        back_index = end - 1
        while back_index > start and remaining_overlap > 0:
            remaining_overlap -= len(segments[back_index]["tokens"])
            back_index -= 1
        start = max(back_index + 1, 0)
        if start >= end:
            start = end

    return chunks
