"""Document type detection utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, cast

from .models import DocumentType

logger = logging.getLogger(__name__)

try:
    import filetype
except ImportError:  # pragma: no cover - optional dependency
    filetype = cast(Any, None)

_PdfReader: Optional[Any] = None
try:
    from pypdf import PdfReader as _PdfReader
except ImportError:  # pragma: no cover - optional dependency
    _PdfReader = None

PdfReader = cast(Optional[Any], _PdfReader)


def _detect_mime(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return None

    if filetype is not None:
        try:
            kind = filetype.guess(str(file_path))
            if kind and kind.mime:
                return kind.mime.lower()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("filetype.guess failed for %s: %s", file_path, exc)

    # Fallback to simple signature checks
    try:
        with file_path.open("rb") as fh:
            header = fh.read(8)
    except OSError:  # pragma: no cover - defensive
        return None

    if header.startswith(b"%PDF"):
        return "application/pdf"

    signatures = {
        b"\x89PNG\r\n\x1a\n": "image/png",
        b"\xff\xd8\xff": "image/jpeg",
        b"II*\x00": "image/tiff",
        b"MM\x00*": "image/tiff",
        b"BM": "image/bmp",
        b"GIF87a": "image/gif",
        b"GIF89a": "image/gif",
    }
    for sig, mime in signatures.items():
        if header.startswith(sig):
            return mime

    return None


def _pdf_has_text(file_path: Path, max_pages: int = 3) -> bool:
    if PdfReader is None:
        return False

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("PdfReader failed for %s: %s", file_path, exc)
        return False

    pages = min(len(reader.pages), max_pages)
    for index in range(pages):
        try:
            page = reader.pages[index]
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to extract text from page %s of %s: %s", index, file_path, exc)
            continue

        if text.strip():
            return True

    return False


def detect_document_type(file_path: Path, metadata: Dict[str, str] | None = None) -> DocumentType:
    """Detect the document type using magic bytes, mime detection, and PDF heuristics."""

    if metadata and (override := metadata.get("detected_type")):
        try:
            return DocumentType(override)
        except ValueError:
            logger.debug("Invalid detected_type override: %s", override)

    mime = _detect_mime(file_path)
    if mime == "application/pdf":
        if _pdf_has_text(file_path):
            return DocumentType.TEXT_PDF

        file_size = file_path.stat().st_size if file_path.exists() else 0
        if file_size <= 0:
            return DocumentType.UNKNOWN

        return DocumentType.SCANNED_PDF

    if mime and mime.startswith("image/"):
        return DocumentType.IMAGE

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        if _pdf_has_text(file_path):
            return DocumentType.TEXT_PDF
        return DocumentType.SCANNED_PDF

    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}:
        return DocumentType.IMAGE

    return DocumentType.UNKNOWN
