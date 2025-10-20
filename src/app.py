"""Command-line interface for the RAG0 interactive workflow."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import nullcontext, suppress
from typing import Callable, Dict, Optional, Tuple

from dotenv import load_dotenv
from temporalio.client import Client, WorkflowHandle

from .config import WorkflowConfig
from .utils.cli import (
    DEFAULT_THEME,
    RICH_AVAILABLE,
    build_main_cli_parser,
    build_response_view,
    emit_plain_result,
    get_console,
    print_session_banner,
    progress_label_style,
    temporal_ui_url,
    workflow_history_url,
)
from .workflows import MainWorkflow

load_dotenv()


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(stripped)
        except ValueError:
            return default
    return default


async def _query_prompt(
    handle: WorkflowHandle,
    last_revision: int,
) -> Tuple[int, Optional[str], Optional[str]]:
    """Fetch the latest prompt payload and return revision, prompt, and error."""

    try:
        payload_raw: Optional[Dict[str, object]] = await handle.query(MainWorkflow.get_next_prompt)
    except Exception as exc:  # pragma: no cover - defensive network guard
        return last_revision, None, f"Unable to fetch workflow prompt: {exc}"

    payload = payload_raw if isinstance(payload_raw, dict) else {}
    prompt_text = str(payload.get("prompt") or "")
    revision = _safe_int(payload.get("revision"), last_revision)

    if revision <= last_revision or not prompt_text:
        return revision, None, None

    return revision, prompt_text, None


async def _await_prompt(
    handle: WorkflowHandle,
    last_revision: int,
    *,
    on_prompt: Optional[Callable[[str], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    timeout_seconds: Optional[float] = None,
) -> int:
    """Poll for a prompt refresh before prompting the user for input."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - fallback for non-async contexts
        loop = asyncio.new_event_loop()

    deadline = loop.time() + timeout_seconds if timeout_seconds is not None else None
    new_revision = last_revision

    while True:
        revision, prompt_text, error = await _query_prompt(handle, last_revision)

        if error and on_error:
            on_error(error)

        if prompt_text and on_prompt:
            on_prompt(prompt_text)

        if revision > last_revision:
            return revision

        if revision > new_revision:
            new_revision = revision

        if deadline is not None and loop.time() >= deadline:
            return new_revision

        await asyncio.sleep(0.05)


async def _poll_for_result(
    handle: WorkflowHandle,
    last_revision: int,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """Poll the workflow until a newer result revision is available."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - fallback for non-async contexts
        loop = asyncio.new_event_loop()

    deadline = loop.time() + 30.0  # generous upper bound for long-running commands
    while True:
        try:
            result_raw = await handle.query(MainWorkflow.get_last_result)
        except Exception as exc:  # pragma: no cover - defensive network guard
            return None, f"Unable to query workflow result: {exc}"

        if isinstance(result_raw, dict):
            revision = _safe_int(result_raw.get("revision"), last_revision)
            if revision > last_revision:
                return result_raw, None

        if loop.time() >= deadline:
            return None, "Timed out waiting for workflow result."

        await asyncio.sleep(0.05)


async def _start_workflow(cfg: WorkflowConfig) -> Tuple[WorkflowHandle, str, Optional[str], Optional[str]]:
    """Create workflow config, start the run, and return identifiers."""

    ui_url = temporal_ui_url(cfg.address, cfg.namespace)

    client = await Client.connect(cfg.address, namespace=cfg.namespace)
    workflow_config = cfg.copy()

    prefix = cfg.workflow_id_prefix_value()
    wf_id = f"{prefix}-{uuid.uuid4().hex}"

    handle = await client.start_workflow(
        MainWorkflow.run,
        workflow_config,
        id=wf_id,
        task_queue=cfg.task_queue,
    )

    workflow_link = workflow_history_url(ui_url, wf_id) if ui_url else None
    return handle, wf_id, ui_url, workflow_link


async def _run_workflow_plain(
    handle: WorkflowHandle,
    wf_id: str,
    temporal_ui: Optional[str],
    workflow_link: Optional[str],
) -> None:
    """Fallback interactive loop without Rich UI enhancements."""

    console = get_console() if RICH_AVAILABLE else None
    use_rich = bool(
        console is not None
        and getattr(console, "is_terminal", False)
        and getattr(console, "is_interactive", False)
    )

    print_session_banner(
        console=console,
        use_rich=use_rich,
        workflow_id=wf_id,
        workflow_link=workflow_link,
        temporal_ui=temporal_ui,
    )

    last_prompt_revision = 0
    last_result_revision = 0

    def handle_prompt(prompt: str) -> None:
        if use_rich and console is not None:
            from rich.panel import Panel

            message = prompt.strip() or "Workflow awaiting input."
            console.print(Panel(message, title="[bold magenta]Workflow[/]", border_style="magenta"))
        else:
            print(
                prompt,
                end="" if prompt.endswith("\n") else "\n",
                flush=True,
            )

    def handle_error(message: str) -> None:
        if use_rich and console is not None:
            console.print(f"[bold red]Error:[/] {message}")
        else:
            print(f"[error] {message}")

    def _sanitize_command(raw_command: str) -> str:
        if not raw_command:
            return ""

        normalized = raw_command.encode("utf-8", "ignore").decode("utf-8")
        stripped_leading = normalized.lstrip()
        if not stripped_leading:
            return ""

        start_index = 0
        for idx, char in enumerate(stripped_leading):
            if char.isalnum() or char in {"/"}:
                start_index = idx
                break
        cleaned = stripped_leading[start_index:]
        return cleaned.rstrip()

    try:
        while True:
            prompt_wait = (
                console.status("[cyan]Waiting for workflow prompt…[/]", spinner="dots")
                if use_rich and console is not None
                else nullcontext()
            )
            with prompt_wait:
                last_prompt_revision = await _await_prompt(
                    handle,
                    last_prompt_revision,
                    on_prompt=handle_prompt,
                    on_error=handle_error,
                )

            try:
                if use_rich and console is not None:
                    raw_command = console.input("[bold cyan]rag0> [/]")
                else:
                    raw_command = input("rag0> ")
            except EOFError:
                if use_rich and console is not None:
                    console.print("\n[bold yellow]Input closed. Exiting session.[/]")
                else:
                    print("\nInput closed. Exiting session.")
                break

            sanitized_command = _sanitize_command(raw_command)

            await handle.signal(MainWorkflow.submit_input, sanitized_command)

            if not sanitized_command.strip():
                continue

            progress_seen = 0
            final_result: Optional[Dict[str, object]] = None
            final_error: Optional[str] = None

            result_wait = (
                console.status("[cyan]Processing command…[/]", spinner="dots")
                if use_rich and console is not None
                else nullcontext()
            )
            with result_wait as status:
                while True:
                    result, error = await _poll_for_result(handle, last_result_revision)
                    if result is None:
                        final_error = error or "No response received from workflow."
                        break

                    last_result_revision = _safe_int(result.get("revision"), last_result_revision)
                    status_value = str(result.get("status") or "").lower()
                    command = str(result.get("command") or "").lower()
                    payload_obj = result.get("result")
                    payload = payload_obj if isinstance(payload_obj, dict) else {}

                    if status_value == "running" and command == "ask":
                        progress_obj = payload.get("progress") or []
                        progress = progress_obj if isinstance(progress_obj, list) else []
                        if progress:
                            new_events = progress[progress_seen:]
                            if new_events:
                                start_index = progress_seen + 1
                                progress_seen = len(progress)
                                if use_rich and console is not None and status is not None:
                                    last_event = new_events[-1]
                                    raw_label = str(last_event.get("label", "") or "step")
                                    label_text = raw_label.replace("_", " ").title()
                                    color = progress_label_style(raw_label, DEFAULT_THEME)
                                    status.update(f"[cyan]Processing command…[/] [{color}]{label_text}[/]")
                                    for idx, event in enumerate(new_events, start=start_index):
                                        event_label_raw = str(event.get("label", "") or "step")
                                        detail = str(event.get("detail", "")).strip() or "Step completed."
                                        event_label = event_label_raw.replace("_", " ").title()
                                        event_color = progress_label_style(event_label_raw, DEFAULT_THEME)
                                        console.print(f"[bold {event_color}]{idx}. {event_label}[/] {detail}")
                                else:
                                    for idx, event in enumerate(new_events, start=start_index):
                                        if not isinstance(event, dict):
                                            continue
                                        event_label_raw = str(event.get("label", "") or "step")
                                        event_label = event_label_raw.replace("_", " ").title()
                                        detail = str(event.get("detail", "")).strip() or "Step completed."
                                        print(f"[progress] {idx}. {event_label}: {detail}")
                        continue

                    final_result = result
                    break

            if final_result is None:
                warning = final_error or "No response received from workflow."
                if use_rich and console is not None:
                    console.print(f"[bold yellow]Warning:[/] {warning}")
                else:
                    print(f"[warn] {warning}")
                continue

            result = final_result

            if use_rich and console is not None:
                view = build_response_view(result)
                console.print(view.body)
                alert = view.metadata.get("message")
                if alert and view.status in {"warn", "error"}:
                    style = "yellow" if view.status == "warn" else "red"
                    console.print(f"[bold {style}]{alert}[/]")
            else:
                emit_plain_result(result)

            if result.get("status") == "quit":
                break
    except KeyboardInterrupt:
        if use_rich and console is not None:
            console.print("\n[bold yellow]Interrupted. Exiting session.[/]")
        else:
            print("\nInterrupted. Exiting session.")


async def _run_workflow_interactive(cfg: WorkflowConfig) -> None:
    handle, wf_id, ui_url, workflow_link = await _start_workflow(cfg)

    try:
        await _run_workflow_plain(handle, wf_id, ui_url, workflow_link)
    finally:
        with suppress(Exception):
            await handle.result()


def main() -> None:
    parser = build_main_cli_parser()
    args = parser.parse_args()

    config = WorkflowConfig(**vars(args))
    asyncio.run(_run_workflow_interactive(config))


if __name__ == "__main__":  # pragma: no cover
    main()
