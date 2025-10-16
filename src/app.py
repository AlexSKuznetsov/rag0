"""Command-line interface for the RAG0 interactive workflow."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import nullcontext, suppress
from typing import Callable, Dict, Optional, Tuple

from dotenv import load_dotenv
from temporalio.client import Client, WorkflowHandle

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
from .workflows import MainWorkflow, MainWorkflowConfig

load_dotenv()


async def _query_prompt(
    handle: WorkflowHandle,
    last_revision: int,
) -> Tuple[int, Optional[str], Optional[str]]:
    """Fetch the latest prompt payload and return revision, prompt, and error."""

    try:
        payload: Dict[str, object] = await handle.query(MainWorkflow.get_next_prompt)
    except Exception as exc:  # pragma: no cover - defensive network guard
        return last_revision, None, f"Unable to fetch workflow prompt: {exc}"

    prompt_text = str((payload or {}).get("prompt") or "")
    revision = int((payload or {}).get("revision", 0))

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
            result = await handle.query(MainWorkflow.get_last_result)
        except Exception as exc:  # pragma: no cover - defensive network guard
            return None, f"Unable to query workflow result: {exc}"

        if result is not None:
            revision = int(result.get("revision", 0))
            if revision > last_revision:
                return result, None

        if loop.time() >= deadline:
            return None, "Timed out waiting for workflow result."

        await asyncio.sleep(0.05)


async def _start_workflow(
    parsed_dir: str,
    index_dir: str,
    address: str,
    namespace: str,
    task_queue: str,
    workflow_id_prefix: Optional[str],
    ask_top_k: int,
    ask_max_subquestions: int,
    ask_neighbor_span: int,
    ask_reflection_enabled: bool,
    ask_max_reflections: int,
    ask_min_citations: int,
    ollama_model: str,
    ollama_base_url: str,
    ask_temperature: float,
    chunk_size: int,
    chunk_overlap: int,
    chunk_merge_threshold: int,
) -> Tuple[WorkflowHandle, str, Optional[str], Optional[str]]:
    """Create workflow config, start the run, and return identifiers."""

    ui_url = temporal_ui_url(address, namespace)

    client = await Client.connect(address, namespace=namespace)
    config = MainWorkflowConfig(
        parsed_dir=parsed_dir,
        index_dir=index_dir,
        ask_top_k=ask_top_k,
        max_subquestions=ask_max_subquestions,
        neighbor_span=ask_neighbor_span,
        reflection_enabled=ask_reflection_enabled,
        max_reflections=ask_max_reflections,
        min_citations=ask_min_citations,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        temperature=ask_temperature,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_merge_threshold=chunk_merge_threshold,
    )

    prefix = (workflow_id_prefix or "main").strip() or "main"
    wf_id = f"{prefix}-{uuid.uuid4().hex}"

    handle = await client.start_workflow(
        MainWorkflow.run,
        config,
        id=wf_id,
        task_queue=task_queue,
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
    use_rich = bool(console and console.is_terminal and console.is_interactive)

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
        if use_rich:
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
        if use_rich:
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
                if use_rich
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
                if use_rich:
                    raw_command = console.input("[bold cyan]rag0> [/]")
                else:
                    raw_command = input("rag0> ")
            except EOFError:
                if use_rich:
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
                console.status("[cyan]Processing command…[/]", spinner="dots") if use_rich else nullcontext()
            )
            with result_wait as status:
                while True:
                    result, error = await _poll_for_result(handle, last_result_revision)
                    if result is None:
                        final_error = error or "No response received from workflow."
                        break

                    last_result_revision = int(result.get("revision", last_result_revision))
                    status_value = str(result.get("status") or "").lower()
                    command = str(result.get("command") or "").lower()
                    payload = result.get("result") or {}

                    if status_value == "running" and command == "ask":
                        progress = payload.get("progress") or []
                        if progress:
                            new_events = progress[progress_seen:]
                            if new_events:
                                start_index = progress_seen + 1
                                progress_seen = len(progress)
                                if use_rich:
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
                                        event_label_raw = str(event.get("label", "") or "step")
                                        event_label = event_label_raw.replace("_", " ").title()
                                        detail = str(event.get("detail", "")).strip() or "Step completed."
                                        print(f"[progress] {idx}. {event_label}: {detail}")
                        continue

                    final_result = result
                    break

            if final_result is None:
                warning = final_error or "No response received from workflow."
                if use_rich:
                    console.print(f"[bold yellow]Warning:[/] {warning}")
                else:
                    print(f"[warn] {warning}")
                continue

            result = final_result

            if use_rich:
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
        if use_rich:
            console.print("\n[bold yellow]Interrupted. Exiting session.[/]")
        else:
            print("\nInterrupted. Exiting session.")


async def _run_workflow_interactive(
    parsed_dir: str,
    index_dir: str,
    address: str,
    namespace: str,
    task_queue: str,
    workflow_id_prefix: Optional[str],
    ask_top_k: int,
    ask_max_subquestions: int,
    ask_neighbor_span: int,
    ask_reflection_enabled: bool,
    ask_max_reflections: int,
    ask_min_citations: int,
    ollama_model: str,
    ollama_base_url: str,
    ask_temperature: float,
    chunk_size: int,
    chunk_overlap: int,
    chunk_merge_threshold: int,
) -> None:
    handle, wf_id, ui_url, workflow_link = await _start_workflow(
        parsed_dir=parsed_dir,
        index_dir=index_dir,
        address=address,
        namespace=namespace,
        task_queue=task_queue,
        workflow_id_prefix=workflow_id_prefix,
        ask_top_k=ask_top_k,
        ask_max_subquestions=ask_max_subquestions,
        ask_neighbor_span=ask_neighbor_span,
        ask_reflection_enabled=ask_reflection_enabled,
        ask_max_reflections=ask_max_reflections,
        ask_min_citations=ask_min_citations,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        ask_temperature=ask_temperature,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_merge_threshold=chunk_merge_threshold,
    )

    try:
        await _run_workflow_plain(handle, wf_id, ui_url, workflow_link)
    finally:
        with suppress(Exception):
            await handle.result()


def main() -> None:
    parser = build_main_cli_parser()
    args = parser.parse_args()

    asyncio.run(
        _run_workflow_interactive(
            parsed_dir=args.parsed_dir,
            index_dir=args.index_dir,
            address=args.address,
            namespace=args.namespace,
            task_queue=args.task_queue,
            workflow_id_prefix=args.workflow_id_prefix,
            ask_top_k=args.ask_top_k,
            ask_max_subquestions=args.ask_max_subquestions,
            ask_neighbor_span=args.ask_neighbor_span,
            ask_reflection_enabled=args.ask_reflection_enabled,
            ask_max_reflections=args.ask_max_reflections,
            ask_min_citations=args.ask_min_citations,
            ollama_model=args.ollama_model,
            ollama_base_url=args.ollama_base_url,
            ask_temperature=args.ask_temperature,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunk_merge_threshold=args.chunk_merge_threshold,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
