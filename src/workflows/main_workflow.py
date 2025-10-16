"""Main Temporal workflow orchestrating CLI-driven commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from temporalio import exceptions as temporal_exceptions
from temporalio import workflow

from .ingestion_workflow import IngestionWorkflow
from .question_workflow import QuestionWorkflow, QuestionWorkflowInput

RENDER_CLI_MENU_ACTIVITY = "render_cli_menu_activity"
PARSE_CLI_COMMAND_ACTIVITY = "parse_cli_command_activity"
INGEST_COMMAND_ACTIVITY = "ingest_command_activity"
QUESTION_COMMAND_ACTIVITY = "question_command_activity"
STATS_COMMAND_ACTIVITY = "stats_command_activity"
QUIT_COMMAND_ACTIVITY = "quit_command_activity"


@dataclass
class CommandPayload:
    """Payload parsed from CLI input."""

    command: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MainWorkflowConfig:
    """Configuration parameters provided to the main workflow."""

    parsed_dir: str = "parsed"
    index_dir: str = "storage/index"
    ask_top_k: int = 6
    ollama_model: str = "qwen3:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    max_subquestions: int = 3
    chunk_size: int = 700
    chunk_overlap: int = 150
    chunk_merge_threshold: int = 60
    neighbor_span: int = 1
    reflection_enabled: bool = True
    max_reflections: int = 2
    min_citations: int = 1

    def to_activity_payload(self) -> Dict[str, Any]:
        """Return a dict compatible with activity execution."""

        return {
            "parsed_dir": self.parsed_dir,
            "index_dir": self.index_dir,
            "ask_top_k": self.ask_top_k,
            "ollama_model": self.ollama_model,
            "ollama_base_url": self.ollama_base_url,
            "temperature": self.temperature,
            "max_subquestions": self.max_subquestions,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_merge_threshold": self.chunk_merge_threshold,
            "neighbor_span": self.neighbor_span,
            "reflection_enabled": self.reflection_enabled,
            "max_reflections": self.max_reflections,
            "min_citations": self.min_citations,
        }


@workflow.defn
class MainWorkflow:
    """Entry workflow coordinating CLI commands and downstream workflows."""

    def __init__(self) -> None:
        self._pending_input: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._result_revision = 0
        self._next_prompt: Dict[str, Any] = {"prompt": "", "revision": 0}
        self._prompt_revision = 0
        self._config: Optional[MainWorkflowConfig] = None
        self._active_command: Optional[str] = None
        self._active_progress: List[Dict[str, Any]] = []

    @workflow.signal
    def submit_input(self, raw_input: str) -> None:
        self._pending_input = raw_input

    @workflow.query
    def get_last_result(self) -> Optional[Dict[str, Any]]:
        return self._last_result

    @workflow.query
    def get_next_prompt(self) -> Dict[str, Any]:
        return self._next_prompt

    @workflow.signal
    def push_progress(self, payload: Dict[str, Any]) -> None:
        """Receive incremental progress updates from downstream executions."""

        if self._active_command != "ask":
            return

        event = payload.get("event") if isinstance(payload, dict) else None
        if not isinstance(event, dict):
            return

        copied = {
            "label": str(event.get("label", "")),
            "detail": str(event.get("detail", "")),
            "metadata": dict(event.get("metadata") or {}),
        }
        self._active_progress.append(copied)
        interim = {
            "status": "running",
            "command": "ask",
            "result": {
                "progress": list(self._active_progress),
            },
        }
        self._store_result(interim)

    @workflow.run
    async def run(self, config: Optional[MainWorkflowConfig] = None) -> Dict[str, Any]:
        self._config = config or MainWorkflowConfig()
        await self._refresh_prompt()

        while True:
            await workflow.wait_condition(lambda: self._pending_input is not None)
            raw_input = self._pending_input or ""
            self._pending_input = None

            if not raw_input.strip():
                await self._refresh_prompt()
                continue

            command_payload = await self._parse_command(raw_input)
            if command_payload is None:
                await self._refresh_prompt()
                continue

            result = await self._dispatch_command(command_payload)
            stored_result = self._store_result(result)
            await self._refresh_prompt()

            if stored_result.get("status") == "quit":
                return stored_result

    async def _refresh_prompt(self) -> None:
        try:
            prompt_payload = await workflow.execute_activity(
                RENDER_CLI_MENU_ACTIVITY,
                schedule_to_close_timeout=workflow.timedelta(seconds=5),
            )
        except temporal_exceptions.ActivityError as exc:
            prompt_text = f"[error] Unable to render menu: {self._activity_error_message(exc)}"
            prompt_payload = {"prompt": prompt_text}

        self._prompt_revision += 1
        payload = dict(prompt_payload or {})
        payload.setdefault("prompt", "")
        payload["revision"] = self._prompt_revision
        self._next_prompt = payload

    async def _parse_command(self, raw_input: str) -> Optional[CommandPayload]:
        try:
            parsed = await workflow.execute_activity(
                PARSE_CLI_COMMAND_ACTIVITY,
                args=(raw_input,),
                schedule_to_close_timeout=workflow.timedelta(seconds=10),
            )
            return CommandPayload(
                command=parsed.get("command", ""),
                arguments=parsed.get("arguments", {}) or {},
            )
        except temporal_exceptions.ActivityError as exc:
            self._store_result(
                {
                    "status": "error",
                    "command": "parse",
                    "message": self._activity_error_message(exc),
                }
            )
            return None

    async def _dispatch_command(self, command: CommandPayload) -> Dict[str, Any]:
        config_payload = self._config.to_activity_payload() if self._config else {}
        try:
            if command.command == "ingest":
                ingest_payload = await workflow.execute_activity(
                    INGEST_COMMAND_ACTIVITY,
                    args=(command.arguments, config_payload),
                    schedule_to_close_timeout=workflow.timedelta(seconds=10),
                )
                result = await workflow.execute_child_workflow(
                    IngestionWorkflow.run,
                    ingest_payload,
                )
                return {
                    "status": "ok",
                    "command": "ingest",
                    "result": result,
                }

            if command.command == "ask":
                self._active_command = "ask"
                self._active_progress = []
                try:
                    question_payload = await workflow.execute_activity(
                        QUESTION_COMMAND_ACTIVITY,
                        args=(command.arguments, config_payload),
                        schedule_to_close_timeout=workflow.timedelta(seconds=10),
                    )
                    question_input = QuestionWorkflowInput(**question_payload)
                    result = await workflow.execute_child_workflow(
                        QuestionWorkflow.run,
                        question_input,
                    )
                finally:
                    self._active_command = None
                    self._active_progress = []
                return {
                    "status": "ok",
                    "command": "ask",
                    "result": result,
                }

            if command.command == "stat":
                stats_result = await workflow.execute_activity(
                    STATS_COMMAND_ACTIVITY,
                    args=(config_payload,),
                    schedule_to_close_timeout=workflow.timedelta(seconds=30),
                )
                return {
                    "status": "ok",
                    "command": "stat",
                    "result": stats_result,
                }

            if command.command == "quit":
                quit_payload = await workflow.execute_activity(
                    QUIT_COMMAND_ACTIVITY,
                    schedule_to_close_timeout=workflow.timedelta(seconds=5),
                )
                return quit_payload

            return {
                "status": "error",
                "command": command.command,
                "message": "Unknown command",
            }
        except temporal_exceptions.ActivityError as exc:
            return {
                "status": "error",
                "command": command.command,
                "message": self._activity_error_message(exc),
            }
        except Exception as exc:  # pragma: no cover - defensive guardrail
            return {
                "status": "error",
                "command": command.command,
                "message": str(exc),
            }

    @staticmethod
    def _activity_error_message(error: temporal_exceptions.ActivityError) -> str:
        return str(error.cause or error)

    def _store_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        self._result_revision += 1
        payload = dict(result)
        payload["revision"] = self._result_revision
        self._last_result = payload
        return payload


__all__ = ["CommandPayload", "MainWorkflow", "MainWorkflowConfig"]
