"""Agent implementations used by Temporal workflows."""

from __future__ import annotations

import warnings

try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning
except ImportError:  # pragma: no cover - pydantic compat guard
    pass
else:
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

from .ask import LangGraphAskAgent

__all__ = ["LangGraphAskAgent"]
