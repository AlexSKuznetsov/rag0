"""CLI helper utilities shared across the RAG0 commands."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Dict, Iterable, List, Optional, cast
from urllib.parse import urlparse

if TYPE_CHECKING:
    from rich.align import Align as AlignType
    from rich.console import Console as ConsoleType
    from rich.console import Group as GroupType
    from rich.console import RenderableType as RenderableTypeType
    from rich.layout import Layout as LayoutType
    from rich.live import Live as LiveType
    from rich.markdown import Markdown as MarkdownType
    from rich.panel import Panel as PanelType
    from rich.spinner import Spinner as SpinnerType
    from rich.table import Table as TableType
    from rich.text import Text as TextType
else:
    AlignType = Any
    ConsoleType = Any
    GroupType = Any
    RenderableTypeType = Any
    LayoutType = Any
    LiveType = Any
    MarkdownType = Any
    PanelType = Any
    SpinnerType = Any
    TableType = Any
    TextType = Any

_AlignRuntime: Optional[AlignType] = None
_ConsoleRuntime: Optional[ConsoleType] = None
_GroupRuntime: Optional[GroupType] = None
_RenderableRuntime: Optional[RenderableTypeType] = None
_LayoutRuntime: Optional[LayoutType] = None
_LiveRuntime: Optional[LiveType] = None
_MarkdownRuntime: Optional[MarkdownType] = None
_PanelRuntime: Optional[PanelType] = None
_SpinnerRuntime: Optional[SpinnerType] = None
_TableRuntime: Optional[TableType] = None
_TextRuntime: Optional[TextType] = None

try:  # pragma: no cover - exercised indirectly in CLI runtime
    from rich import box
    from rich.align import Align as _AlignCls
    from rich.console import Console as _ConsoleCls
    from rich.console import Group as _GroupCls
    from rich.console import RenderableType as _RenderableCls
    from rich.layout import Layout as _LayoutCls
    from rich.live import Live as _LiveCls
    from rich.markdown import Markdown as _MarkdownCls
    from rich.panel import Panel as _PanelCls
    from rich.spinner import Spinner as _SpinnerCls
    from rich.table import Table as _TableCls
    from rich.text import Text as _TextCls

    _AlignRuntime = cast(Optional[AlignType], _AlignCls)
    _ConsoleRuntime = cast(Optional[ConsoleType], _ConsoleCls)
    _GroupRuntime = cast(Optional[GroupType], _GroupCls)
    _RenderableRuntime = cast(Optional[RenderableTypeType], _RenderableCls)
    _LayoutRuntime = cast(Optional[LayoutType], _LayoutCls)
    _LiveRuntime = cast(Optional[LiveType], _LiveCls)
    _MarkdownRuntime = cast(Optional[MarkdownType], _MarkdownCls)
    _PanelRuntime = cast(Optional[PanelType], _PanelCls)
    _SpinnerRuntime = cast(Optional[SpinnerType], _SpinnerCls)
    _TableRuntime = cast(Optional[TableType], _TableCls)
    _TextRuntime = cast(Optional[TextType], _TextCls)
except Exception:  # pragma: no cover - fallback if Rich missing
    box = cast(Any, None)
    _AlignRuntime = None
    _ConsoleRuntime = None
    _GroupRuntime = None
    _RenderableRuntime = None
    _LayoutRuntime = None
    _LiveRuntime = None
    _MarkdownRuntime = None
    _PanelRuntime = None
    _SpinnerRuntime = None
    _TableRuntime = None
    _TextRuntime = None

Align = cast(Any, _AlignRuntime)
Console = cast(Any, _ConsoleRuntime)
Group = cast(Any, _GroupRuntime)
Layout = cast(Any, _LayoutRuntime)
Live = cast(Any, _LiveRuntime)
Markdown = cast(Any, _MarkdownRuntime)
Panel = cast(Any, _PanelRuntime)
Spinner = cast(Any, _SpinnerRuntime)
Table = cast(Any, _TableRuntime)
Text = cast(Any, _TextRuntime)
RenderableRuntime = _RenderableRuntime if _RenderableRuntime is not None else Any

RICH_AVAILABLE = Console is not None

_CONSOLE: Optional[ConsoleType] = None


@dataclass(frozen=True)
class CLITheme:
    """Palette and glyph configuration for the dark-mode Rich layout."""

    header_bg: str = "#1f2933"
    header_fg: str = "#e6edf3"
    chat_bg: str = "#111827"
    chat_fg: str = "#e5e7eb"
    response_bg: str = "#0f172a"
    response_fg: str = "#f9fafb"
    footer_bg: str = "#111827"
    footer_fg: str = "#9ca3af"
    success: str = "#22c55e"
    warning: str = "#facc15"
    error: str = "#f87171"
    accent: str = "#38bdf8"
    highlight: str = "#a855f7"


DEFAULT_THEME = CLITheme()


def _render_text(message: str, style: str) -> RenderableTypeType:
    if Text is None:
        return message
    return cast(RenderableTypeType, Text(message, style=style))


def get_console() -> Optional[ConsoleType]:
    """Return a singleton Rich Console configured for the CLI."""

    global _CONSOLE

    if not RICH_AVAILABLE:
        return None

    if _CONSOLE is None:
        _CONSOLE = Console(
            log_time=False,
            log_path=False,
            highlight=False,
            color_system="truecolor",
            soft_wrap=True,
        )

    return _CONSOLE


def print_session_banner(
    *,
    console: Optional[ConsoleType],
    use_rich: bool,
    workflow_id: str,
    workflow_link: Optional[str],
    temporal_ui: Optional[str],
) -> None:
    if use_rich:
        if console is None or Panel is None or Text is None:
            return

        console.rule("[bold cyan]RAG0 Interactive Session[/]")
        details = Text(f"Workflow ID: {workflow_id}", style="bold white")
        if workflow_link:
            details.append(f"\nTrack progress: {workflow_link}", style="bold cyan")
        if temporal_ui:
            details.append(f"\nTemporal UI: {temporal_ui}", style="bold cyan")
        console.print(Panel(details, border_style="cyan", title="Session"))
        console.print(
            "Type commands like [bold]ingest <path>[/] or [bold]ask <question>[/]. Press Ctrl+C to exit.\n"
        )
        return

    if temporal_ui:
        print(f"\nTemporal UI: {temporal_ui}")
    print(f"\nWorkflow ID     : {workflow_id}")
    if workflow_link:
        print(f"Track progress  : {workflow_link}")


def is_rich_live_supported(console: Optional[ConsoleType] = None) -> bool:
    """Return whether the current environment can render Rich Live layouts."""

    if not RICH_AVAILABLE:
        return False

    if console is None:
        console = get_console()

    if console is None:
        return False

    if not console.is_terminal or not console.is_interactive:
        return False

    term = os.environ.get("TERM", "").lower()
    if term in {"", "dumb", "unknown"}:
        return False

    if os.environ.get("RAG0_FORCE_PLAIN", "").lower() in {"1", "true", "yes"}:
        return False

    return True


@dataclass
class ChatTurn:
    """Represents a single chat entry in the interactive CLI."""

    role: str
    content: str


@dataclass
class ResponseView:
    """Renderable details for the response pane."""

    status: str
    command: str
    title: str
    body: RenderableTypeType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CLIState:
    """Current UI state for the interactive CLI session."""

    theme: CLITheme = DEFAULT_THEME
    workflow_id: str = ""
    temporal_link: Optional[str] = None
    status_message: str = ""
    hint_message: str = "Type commands like 'ingest <path>' or 'ask <question>'. Press Ctrl+C to exit."
    chat_history: Deque[ChatTurn] = field(default_factory=lambda: deque(maxlen=50))
    response_view: Optional[ResponseView] = None
    loading_label: Optional[str] = None
    error_message: Optional[str] = None
    prompt_message: Optional[str] = None
    last_prompt: Optional[str] = None

    def append_chat(self, content: str, role: str = "You") -> None:
        entry = content.strip()
        if not entry:
            return
        self.chat_history.append(ChatTurn(role=role, content=entry))


def create_cli_layout(theme: CLITheme = DEFAULT_THEME) -> LayoutType:
    """Build the Rich layout for the interactive session."""

    if not RICH_AVAILABLE or Layout is None:
        raise RuntimeError("Rich layout is unavailable; install the 'rich' package.")

    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="chat", ratio=1),
        Layout(name="response", ratio=2),
    )
    return layout


def _header_table(state: CLIState) -> TableType:
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="left")
    theme = state.theme
    title = Text("RAG0 Interactive Session", style=f"bold {theme.accent}")
    table.add_row(title)

    wf_line = f"Workflow ID: {state.workflow_id or 'starting...'}"
    table.add_row(Text(wf_line, style=theme.header_fg))

    if state.temporal_link:
        link_text = Text(f"Temporal UI: {state.temporal_link}", style=theme.highlight)
        table.add_row(link_text)

    return table


def render_header(state: CLIState) -> RenderableTypeType:
    """Return the header panel renderable."""

    theme = state.theme
    return Panel(
        _header_table(state),
        style=f"on {theme.header_bg}",
        border_style=theme.accent,
    )


def render_chat_panel(state: CLIState) -> RenderableTypeType:
    """Render the chat history panel."""

    theme = state.theme
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style=f"bold {theme.highlight}")
    table.add_column(justify="left", style=theme.chat_fg)

    def role_style(role: str) -> str:
        normalized = role.lower()
        if normalized in {"you", "user"}:
            return theme.highlight
        if normalized in {"workflow", "system"}:
            return theme.accent
        if normalized in {"error", "warn"}:
            return theme.error
        return theme.highlight

    if not state.chat_history:
        table.add_row(
            Text("hint", style=f"italic {theme.highlight}"),
            Text("No commands yet. Type something to get started.", style=f"italic {theme.chat_fg}"),
        )
    else:
        for turn in state.chat_history:
            role_label = Text(turn.role.upper(), style=f"bold {role_style(turn.role)}")
            table.add_row(role_label, Text(turn.content, style=theme.chat_fg))

    panel_title = f"[{theme.accent}]Chat[/]"
    return Panel(
        Align.left(table),
        title=panel_title,
        border_style=theme.accent,
        style=f"on {theme.chat_bg}",
    )


def render_response_panel(state: CLIState) -> RenderableTypeType:
    """Render the response panel for the latest workflow output."""

    theme = state.theme

    if not state.response_view:
        placeholder = Text(
            "Responses will appear here once the workflow returns data.",
            style=f"italic {theme.response_fg}",
        )
        return Panel(
            Align.left(placeholder),
            title=f"[{theme.accent}]Response[/]",
            border_style=theme.accent,
            style=f"on {theme.response_bg}",
        )

    view = state.response_view
    border_color = {
        "error": theme.error,
        "warn": theme.warning,
        "success": theme.success,
    }.get(view.status, theme.accent)

    return Panel(
        view.body,
        title=f"[{border_color}]{view.title}[/]",
        border_style=border_color,
        style=f"on {theme.response_bg}",
    )


def render_footer(state: CLIState) -> RenderableTypeType:
    """Render the footer with status, loading indicators, and hints."""

    theme = state.theme
    rows: List[RenderableTypeType] = []

    if state.error_message:
        rows.append(Text(state.error_message, style=f"bold {theme.error}"))

    if state.loading_label:
        label = Text(state.loading_label, style=theme.accent)
        rows.append(Spinner("dots", text=label))
    elif state.status_message:
        rows.append(Text(state.status_message, style=theme.footer_fg))
    else:
        rows.append(Text("Ready.", style=theme.footer_fg))

    if state.prompt_message:
        rows.append(Text(state.prompt_message, style=f"{theme.highlight}"))

    if state.hint_message:
        rows.append(Text(state.hint_message, style=f"italic {theme.footer_fg}"))

    content: RenderableTypeType
    if len(rows) == 1:
        content = rows[0]
    else:
        content = Group(*rows)

    return Panel(
        content,
        border_style=theme.accent,
        style=f"on {theme.footer_bg}",
    )


def refresh_layout(layout: LayoutType, state: CLIState) -> None:
    """Update the root layout panels with the latest state."""

    if not RICH_AVAILABLE:
        return

    layout["header"].update(render_header(state))
    layout["chat"].update(render_chat_panel(state))
    layout["response"].update(render_response_panel(state))
    layout["footer"].update(render_footer(state))


def _stringify_metadata_value(value: Any, max_chars: int = 80) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float)):
        text = f"{value}"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)

    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _summarize_answer(answer: str, max_chars: int = 160) -> str:
    text = " ".join(line.strip() for line in answer.splitlines() if line.strip())
    if not text:
        return "No answer returned."
    if len(text) <= max_chars:
        return text
    return textwrap.shorten(text, width=max_chars, placeholder="…")


def _answer_renderable(answer: str, theme: CLITheme) -> RenderableTypeType:
    cleaned = answer.strip()
    if not cleaned:
        return Text("I could not find that in the stored documents.", style=f"italic {theme.warning}")
    if Markdown is not None:
        return Markdown(cleaned)
    return Text(cleaned, style=theme.response_fg)


def _build_citations_panel(citations: List[str], theme: CLITheme) -> PanelType:
    if not citations:
        content = Text("No citations returned.", style=f"italic {theme.warning}")
        return Panel(
            Align.left(content),
            title=f"[{theme.warning}]Citations[/]",
            border_style=theme.warning,
            style=f"on {theme.response_bg}",
            expand=True,
        )

    table = Table(
        show_header=False,
        expand=True,
        padding=(0, 1),
        box=box.SIMPLE_HEAD,
    )
    table.add_column(justify="right", width=4, style=f"bold {theme.highlight}")
    table.add_column(justify="left", style=theme.response_fg, ratio=1)

    for idx, citation in enumerate(citations, start=1):
        table.add_row(
            Text(str(idx), style=f"bold {theme.highlight}"),
            Text(citation, style=theme.response_fg),
        )

    return Panel(
        Align.left(table),
        title=f"[{theme.success}]Citations[/]",
        border_style=theme.success,
        style=f"on {theme.response_bg}",
        expand=True,
    )


def _reasoning_label_style(label: str, theme: CLITheme) -> str:
    label_upper = label.upper()
    if "REFLECTION" in label_upper or "GRADE" in label_upper:
        return theme.highlight
    if "RETRIEVE" in label_upper or "DOCUMENT" in label_upper:
        return theme.accent
    if "ANSWER" in label_upper or "FINAL" in label_upper:
        return theme.success
    if "WARNING" in label_upper or "ERROR" in label_upper:
        return theme.error
    return theme.highlight


def _format_metadata(metadata: Dict[str, Any], theme: CLITheme) -> TextType:
    meta_text = Text()
    for index, (key, value) in enumerate(metadata.items()):
        if index > 0:
            meta_text.append("\n")
        meta_text.append(f"{key}: ", style=f"bold {theme.accent}")
        meta_text.append(_stringify_metadata_value(value), style=theme.footer_fg)
    return meta_text


def _build_reasoning_panel(reasoning: Iterable[Dict[str, Any]], theme: CLITheme) -> Optional[PanelType]:
    steps = list(reasoning)
    if not steps:
        return None

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=18, style=f"bold {theme.highlight}")
    table.add_column(justify="left", style=theme.response_fg, ratio=1)

    for step in steps:
        label = str(step.get("label", "step"))
        detail = str(step.get("detail", "")).strip()
        metadata = step.get("metadata") or {}

        detail_text = Text(detail or "—", style=theme.response_fg if detail else f"italic {theme.footer_fg}")
        if metadata:
            if detail:
                detail_text.append("\n")
            detail_text.append(_format_metadata(metadata, theme))

        table.add_row(
            Text(label.upper(), style=f"bold {_reasoning_label_style(label, theme)}"),
            detail_text,
        )

    return Panel(
        Align.left(table),
        title=f"[{theme.highlight}]Reasoning[/]",
        border_style=theme.highlight,
        style=f"on {theme.response_bg}",
        expand=True,
    )


def _build_documents_panel(
    documents: Iterable[Dict[str, Any]], theme: CLITheme, limit: int = 3
) -> Optional[PanelType]:
    docs = list(documents)
    if not docs:
        return None

    table = Table(
        show_header=True,
        header_style=f"bold {theme.accent}",
        box=box.SIMPLE_HEAD,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#", justify="right", width=4, style=f"bold {theme.highlight}")
    table.add_column("Score", justify="right", width=8, style=theme.footer_fg)
    table.add_column("Snippet", justify="left", style=theme.response_fg, ratio=2)
    table.add_column("Source", justify="left", style=theme.footer_fg, ratio=1)

    for idx, doc in enumerate(docs[:limit], start=1):
        score = doc.get("score")
        snippet_raw = str(doc.get("text", "")).strip()
        snippet = _stringify_metadata_value(snippet_raw, max_chars=120) if snippet_raw else "—"
        metadata = doc.get("metadata") or {}
        source = metadata.get("source_path") or metadata.get("source") or metadata.get("file_name") or ""
        source_text = _stringify_metadata_value(source, max_chars=60) if source else "—"
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "—"

        table.add_row(
            Text(str(idx), style=f"bold {theme.highlight}"),
            Text(score_text, style=theme.footer_fg),
            Text(snippet, style=theme.response_fg),
            Text(source_text, style=theme.footer_fg),
        )

    return Panel(
        Align.left(table),
        title=f"[{theme.accent}]Context[/]",
        border_style=theme.accent,
        style=f"on {theme.response_bg}",
        expand=True,
    )


def _progress_label_style(label: str, theme: CLITheme) -> str:
    mapping = {
        "analysis": theme.accent,
        "retrieval": theme.highlight,
        "reasoner": theme.success,
        "grade_answer": theme.warning,
        "grade_documents": theme.warning,
        "rewrite_query": theme.highlight,
        "response": theme.success,
    }
    return mapping.get(label.lower(), theme.accent)


def _build_progress_panel(
    events: Iterable[Dict[str, Any]],
    theme: CLITheme,
    start_index: int = 1,
) -> Optional[PanelType]:
    items = list(events)
    if not items:
        return None

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=4, style=f"bold {theme.highlight}")
    table.add_column(justify="left", style=theme.response_fg, ratio=1)

    for idx, event in enumerate(items, start=start_index):
        raw_label = str(event.get("label", "")).strip() or "step"
        detail = str(event.get("detail", "")).strip() or "Step completed."
        metadata = event.get("metadata") or {}

        detail_text = Text(detail, style=theme.response_fg, overflow="fold", no_wrap=False)
        if metadata:
            detail_text.append("\n")
            detail_text.append(_format_metadata(metadata, theme))

        color = _progress_label_style(raw_label, theme)
        table.add_row(
            Text(f"{idx}.", style=f"bold {color}"),
            detail_text,
        )

    return Panel(
        Align.left(table),
        title=f"[{theme.accent}]Progress[/]",
        border_style=theme.accent,
        style=f"on {theme.response_bg}",
        expand=True,
    )


def progress_label_style(label: str, theme: CLITheme = DEFAULT_THEME) -> str:
    """Public helper to obtain the color style for a LangGraph progress label."""

    return _progress_label_style(label, theme)


def _build_ask_response_view(payload: Dict[str, Any], theme: CLITheme) -> ResponseView:
    if any(component is None for component in (Text, Panel, Align, Group)):
        raise RuntimeError("Rich components are unavailable; cannot render ask response view.")

    answer = str(payload.get("answer") or "").strip()
    citations = list(payload.get("citations") or [])
    documents = payload.get("documents") or []
    reasoning = payload.get("reasoning") or []

    summary_text = Text(answer.strip() or "No answer returned.", style=f"bold {theme.accent}")

    summary_panel = Panel(
        Align.left(summary_text),
        title=f"[{theme.accent}]Summary[/]",
        border_style=theme.accent,
        style=f"on {theme.response_bg}",
        expand=True,
    )
    answer_panel = Panel(
        Align.left(_answer_renderable(answer, theme)),
        title=f"[{theme.highlight}]Response[/]",
        border_style=theme.highlight,
        style=f"on {theme.response_bg}",
        expand=True,
    )

    documents_panel = _build_documents_panel(documents, theme)
    segments: List[RenderableTypeType] = []
    segments.extend([summary_panel, answer_panel])
    if documents_panel:
        segments.append(documents_panel)

    body = cast(RenderableTypeType, Group(*segments))

    alerts: List[str] = []
    if not answer:
        alerts.append("No answer returned.")
    if answer and not citations:
        alerts.append("No citations were produced for this answer.")

    metadata = {
        "answer": answer,
        "citations": citations,
        "document_count": len(documents),
        "reasoning_count": len(reasoning),
        "message": alerts[0] if alerts else None,
    }

    status = "success"
    if alerts:
        status = "warn"

    return ResponseView(
        status=status,
        command="ask",
        title="Ask Result",
        body=body,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _json_renderable(data: Dict[str, Any], theme: CLITheme) -> RenderableTypeType:
    text = json.dumps(data, indent=2, sort_keys=True)
    if RICH_AVAILABLE and Markdown is not None:
        return cast(RenderableTypeType, Markdown(f"```json\n{text}\n```"))
    if Text is None:
        return text
    return cast(RenderableTypeType, Text(text, style=theme.response_fg))


def build_response_view(result: Dict[str, Any], theme: CLITheme = DEFAULT_THEME) -> ResponseView:
    """Create a ResponseView suitable for the Rich response panel."""

    status = (result.get("status") or "ok").lower()
    command = (result.get("command") or "result").lower()
    payload = result.get("result") or {}

    if status == "error":
        message = result.get("message") or "Workflow returned an error."
        body = _render_text(message, f"bold {theme.error}")
        return ResponseView(
            status="error",
            command=command,
            title="Error",
            body=body,
            metadata={"message": message},
        )

    if status == "quit":
        message = result.get("message", "Session ended.")
        body = _render_text(message, f"bold {theme.warning}")
        return ResponseView(
            status="warn",
            command=command or "quit",
            title="Session Closed",
            body=body,
            metadata={"message": message},
        )

    if command == "ask":
        return _build_ask_response_view(payload, theme)

    if command == "ingest":
        if not payload:
            body = _render_text(
                "Ingestion completed with no reported artifacts.",
                theme.response_fg,
            )
        else:
            body = _json_renderable(payload, theme)
        return ResponseView(
            status="success",
            command=command,
            title="Ingestion Summary",
            body=body,
            metadata=payload,
        )

    if command == "stat":
        if not payload:
            body = _render_text("No stats available.", theme.response_fg)
        else:
            body = _json_renderable(payload, theme)
        return ResponseView(
            status="success",
            command=command,
            title="Workflow Stats",
            body=body,
            metadata=payload,
        )

    # Generic fallback
    if payload:
        body = _json_renderable(payload, theme)
    else:
        message = result.get("message") or "No data returned."
        body = _render_text(message, theme.response_fg)

    return ResponseView(
        status="success",
        command=command or "result",
        title=command.title() or "Result",
        body=body,
        metadata=payload,
    )


def emit_plain_result(result: Dict[str, Any]) -> None:
    """Fallback plain-text rendering when Rich is unavailable."""

    status = result.get("status")
    if status == "error":
        message = result.get("message") or "Workflow returned an error."
        print(f"[error] {message}")
        return

    if status == "quit":
        print(result.get("message", "Session ended."))
        return

    payload = result.get("result") or {}
    command = (result.get("command") or "").lower()

    if command == "ingest":
        print("Ingestion completed.")
        if payload:
            print(json.dumps(payload, indent=2))
        return

    if command == "ask":
        progress = payload.get("progress") or []
        if progress:
            print("Progress:")
            for event in progress:
                label = str(event.get("label", "")).strip() or "step"
                detail = str(event.get("detail", "")).strip()
                print(f"- {label}: {detail or 'Step completed.'}")
                metadata = event.get("metadata") or {}
                for key, value in metadata.items():
                    print(f"    {key}: {value}")
        answer = payload.get("answer")
        citations = payload.get("citations") or []
        if answer:
            print(answer)
            if citations:
                print(f"Citations: {', '.join(citations)}")
        else:
            print("I could not find that in the stored documents.")
        reasoning = payload.get("reasoning") or []
        if reasoning:
            print("\nReasoning trace:")
            for step in reasoning:
                label = step.get("label", "step")
                detail = step.get("detail", "")
                print(f"- {label}: {detail}")
        return

    if command == "stat":
        if not payload:
            print("No stats available.")
            return
        print(json.dumps(payload, indent=2))
        return

    if payload:
        print(json.dumps(payload, indent=2))
    else:
        print(result.get("message", ""))


DEFAULT_PARSED_DIR = Path("parsed")
DEFAULT_TEMPORAL_ADDRESS = "127.0.0.1:7233"
DEFAULT_TEMPORAL_NAMESPACE = "default"
DEFAULT_TEMPORAL_TASK_QUEUE = "rag0"
DEFAULT_OLLAMA_MODEL = os.environ.get("RAG0_OLLAMA_MODEL", "qwen3:4b")
DEFAULT_OLLAMA_BASE_URL = os.environ.get("RAG0_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_ASK_TOP_K = int(os.environ.get("RAG0_ASK_TOP_K", "6"))
DEFAULT_ASK_MAX_SUBQUESTIONS = int(os.environ.get("RAG0_ASK_MAX_SUBQUESTIONS", "3"))
DEFAULT_ASK_TEMPERATURE = float(os.environ.get("RAG0_ASK_TEMPERATURE", "0.0"))
DEFAULT_ASK_NEIGHBOR_SPAN = int(os.environ.get("RAG0_ASK_NEIGHBOR_SPAN", "1"))
DEFAULT_ASK_REFLECTION_ENABLED = os.environ.get("RAG0_ASK_REFLECTION_ENABLED", "1").lower() not in {
    "0",
    "false",
    "no",
}
DEFAULT_ASK_MAX_REFLECTIONS = int(os.environ.get("RAG0_ASK_MAX_REFLECTIONS", "2"))
DEFAULT_ASK_MIN_CITATIONS = int(os.environ.get("RAG0_ASK_MIN_CITATIONS", "1"))
DEFAULT_CHUNK_SIZE = int(os.environ.get("RAG0_CHUNK_SIZE", "700"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("RAG0_CHUNK_OVERLAP", "150"))
DEFAULT_CHUNK_MERGE_THRESHOLD = int(os.environ.get("RAG0_CHUNK_MERGE_THRESHOLD", "60"))


def build_main_cli_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the interactive application."""

    parser = argparse.ArgumentParser(description="RAG0 interactive workflow CLI")
    parser.add_argument("--address", default=DEFAULT_TEMPORAL_ADDRESS)
    parser.add_argument("--namespace", default=DEFAULT_TEMPORAL_NAMESPACE)
    parser.add_argument("--task-queue", default=DEFAULT_TEMPORAL_TASK_QUEUE)
    parser.add_argument("--parsed-dir", default=str(DEFAULT_PARSED_DIR))
    parser.add_argument("--index-dir", default="storage/index")
    parser.add_argument("--workflow-id-prefix", default=None)
    parser.add_argument(
        "--ask-top-k",
        type=int,
        default=DEFAULT_ASK_TOP_K,
        help="Default top_k for ask workflow.",
    )
    parser.add_argument(
        "--ask-max-subquestions",
        type=int,
        default=DEFAULT_ASK_MAX_SUBQUESTIONS,
        help="Maximum number of generated sub-questions per ask invocation.",
    )
    parser.add_argument(
        "--ask-neighbor-span",
        type=int,
        default=DEFAULT_ASK_NEIGHBOR_SPAN,
        help="Number of adjacent chunks to expand around each retrieved result.",
    )
    parser.add_argument(
        "--ask-max-reflections",
        type=int,
        default=DEFAULT_ASK_MAX_REFLECTIONS,
        help="Maximum number of self-reflection loops per ask invocation.",
    )
    parser.add_argument(
        "--ask-min-citations",
        type=int,
        default=DEFAULT_ASK_MIN_CITATIONS,
        help="Minimum number of citations expected in the final answer.",
    )
    parser.add_argument(
        "--ask-reflection-enabled",
        dest="ask_reflection_enabled",
        action="store_true",
        default=DEFAULT_ASK_REFLECTION_ENABLED,
        help="Enable self-reflective grading to improve retrieval quality.",
    )
    parser.add_argument(
        "--ask-reflection-disabled",
        dest="ask_reflection_enabled",
        action="store_false",
        help="Disable self-reflective grading.",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model identifier to use for reasoning.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--ask-temperature",
        type=float,
        default=DEFAULT_ASK_TEMPERATURE,
        help="Temperature used by the Ollama reasoning model.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Token-aligned chunk size used during ingestion.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Token overlap between successive chunks.",
    )
    parser.add_argument(
        "--chunk-merge-threshold",
        type=int,
        default=DEFAULT_CHUNK_MERGE_THRESHOLD,
        help="Token threshold for merging short paragraphs before chunking.",
    )
    return parser



def temporal_ui_url(address: str, namespace: str) -> Optional[str]:
    """Return the Temporal UI base URL for a given address/namespace."""

    if not address:
        return None

    host = address
    if "://" in address:
        parsed = urlparse(address)
        host = parsed.hostname or ""
    else:
        host = address.split(":")[0]

    if not host:
        return None

    return f"http://{host}:8233/namespaces/{namespace}/workflows"


def workflow_history_url(base_url: str, workflow_id: str, run_id: Optional[str] = None) -> str:
    """Compose a Temporal history view URL for the workflow/run pair."""

    url = f"{base_url}/{workflow_id}"
    if run_id:
        url += f"/{run_id}/history"
    return url


def open_browser_url(url: str) -> None:
    """Attempt to open the Temporal UI in a browser if the environment allows."""

    if not _should_attempt_browser_launch():
        return

    try:
        opened = webbrowser.open(url, new=2)
        if opened:
            print(f"Opened Temporal UI: {url}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Could not open browser for {url}: {exc}")


def prompt_for_question(message: str) -> str:
    """Prompt the user until a non-empty question is supplied."""

    while True:
        value = input(message).strip()
        if value:
            print(f"Question received: {value}")
            return value
        print("Please enter a question.")


def _should_attempt_browser_launch() -> bool:
    if os.environ.get("RAG0_DISABLE_BROWSER") == "1":
        return False

    if "WSL_DISTRO_NAME" in os.environ or "WSL_INTEROP" in os.environ:
        return False

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True

    if os.name == "nt":
        return True

    return False
