"""Temporal activities supporting the interactive CLI."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict

from temporalio import activity


class CommandParseError(ValueError):
    """Raised when a raw CLI command cannot be parsed."""


_MENU_TEXT = (
    "\nCommands:\n"
    "  /ingest <path-to-document> ingest a document into the index\n"
    "  /ask [question text] query the indexed documents\n"
    "  /stat show vector index statistics\n"
    "  /quit quit the session\n"
)


@activity.defn
async def render_cli_menu_activity() -> Dict[str, str]:
    """Return the interactive menu text shown to CLI users."""

    return {"prompt": _MENU_TEXT}


@activity.defn
async def parse_cli_command_activity(raw_input: str) -> Dict[str, Any]:
    """Parse the raw CLI string into a command payload."""

    if not raw_input.strip():
        raise CommandParseError("Enter a command to continue.")

    try:
        tokens = shlex.split(raw_input)
    except ValueError as exc:
        raise CommandParseError(str(exc)) from exc

    if not tokens:
        raise CommandParseError("Enter a command to continue.")

    command_token = tokens[0].lower()
    if command_token.startswith("/"):
        command_token = command_token[1:]

    if command_token not in {"ingest", "ask", "stat", "quit"}:
        raise CommandParseError(f"Unknown command: {tokens[0]}")

    if command_token == "ingest":
        if len(tokens) < 2:
            raise CommandParseError("Usage: /ingest <path-to-document>")
        source_path = Path(tokens[1]).expanduser()
        if not source_path.exists():
            raise CommandParseError(f"File not found: {source_path}")
        if not source_path.is_file():
            raise CommandParseError(f"Not a file: {source_path}")
        return {
            "command": "ingest",
            "arguments": {"source_path": str(source_path.resolve())},
        }

    if command_token == "ask":
        question = " ".join(tokens[1:]).strip()
        args: Dict[str, Any] = {}
        if question:
            args["question"] = question
        return {"command": "ask", "arguments": args}

    if command_token == "stat":
        return {"command": "stat", "arguments": {}}

    return {"command": "quit", "arguments": {}}


__all__ = [
    "CommandParseError",
    "parse_cli_command_activity",
    "render_cli_menu_activity",
]
