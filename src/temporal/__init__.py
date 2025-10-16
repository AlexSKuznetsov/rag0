"""Temporal-specific entry points and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["build_parser", "main", "_run_worker"]

if TYPE_CHECKING:
    from .worker import _run_worker, build_parser, main


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import worker

        return getattr(worker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
