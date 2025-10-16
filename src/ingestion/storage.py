"""Simple storage for parsed markdown documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def store_parsed_markdown(
    metadata: Dict[str, Any],
    content: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Store the parsed markdown content and metadata sidecar."""

    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = metadata.get("file_name", "document")
    stem = Path(file_name).stem or "document"
    target_path = output_dir / f"{stem}.md"
    metadata_path = output_dir / f"{stem}.metadata.json"

    markdown_body = content.get("markdown") or ""

    target_path.write_text(markdown_body, encoding="utf-8")

    payload = {
        "metadata": metadata,
        "content": content,
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_real = target_path.resolve()
    metadata_real = metadata_path.resolve()
    return {
        "markdown_path": markdown_real,
        "metadata_path": metadata_real,
    }


def load_parsed_markdown(markdown_path: Path, metadata_path: Path | None = None) -> Dict[str, Any]:
    """Load the stored markdown and accompanying metadata."""

    markdown_body = markdown_path.read_text(encoding="utf-8")

    sidecar_path = metadata_path or markdown_path.with_suffix(".metadata.json")
    if sidecar_path.exists():
        try:
            sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sidecar_payload = {}
    else:
        sidecar_payload = {}

    metadata = sidecar_payload.get("metadata")
    content = sidecar_payload.get("content")

    if not metadata or not content:
        text_blocks = [segment.strip() for segment in markdown_body.split("\n\n") if segment.strip()]
        paragraphs = [{"text": block} for block in text_blocks]
        metadata = metadata or {"file_name": markdown_path.stem}
        content = {
            "paragraphs": paragraphs,
            "markdown": markdown_body,
            "chunks": [],
            "warnings": [],
        }

    content.setdefault("markdown", markdown_body)
    content.setdefault("paragraphs", [])
    content.setdefault("chunks", [])
    content.setdefault("warnings", [])

    return {
        "metadata": metadata,
        "content": content,
        "warnings": content.get("warnings", []),
    }
