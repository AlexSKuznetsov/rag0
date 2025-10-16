from pathlib import Path

import pytest
from src.activities.cli import CommandParseError, parse_cli_command_activity


@pytest.mark.asyncio
async def test_parse_ingest_valid(tmp_path: Path) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("hello world", encoding="utf-8")

    command = await parse_cli_command_activity(f"/ingest {doc}")

    assert command["command"] == "ingest"
    assert Path(command["arguments"]["source_path"]).resolve() == doc.resolve()


@pytest.mark.asyncio
async def test_parse_ingest_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"

    with pytest.raises(CommandParseError):
        await parse_cli_command_activity(f"/ingest {missing}")


@pytest.mark.asyncio
async def test_parse_ask_inline_question() -> None:
    command = await parse_cli_command_activity("/ask What is RAG0?")
    assert command["command"] == "ask"
    assert command["arguments"]["question"] == "What is RAG0?"


@pytest.mark.asyncio
async def test_parse_quit_command() -> None:
    command = await parse_cli_command_activity("/quit")
    assert command["command"] == "quit"
    assert command["arguments"] == {}


@pytest.mark.asyncio
async def test_parse_unknown_command() -> None:
    with pytest.raises(CommandParseError):
        await parse_cli_command_activity("/unknown")
