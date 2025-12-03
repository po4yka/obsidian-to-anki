"""Rich progress reporting for the sync CLI command."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.sync.progress import ProgressTracker, SyncProgress

logger = get_logger(__name__)


@dataclass(slots=True)
class ProgressMeta:
    """Metadata describing the current sync session."""

    mode_label: str
    incremental: bool
    queue_enabled: bool
    langgraph_enabled: bool
    vault_label: str


def summarize_progress(snapshot: SyncProgress) -> dict[str, Any]:
    """Convert ``SyncProgress`` into a serializable summary."""
    total = max(snapshot.total_notes, 0)
    processed = max(snapshot.notes_processed, 0)
    percent = 0.0 if total == 0 else min((processed / total) * 100.0, 100.0)

    elapsed_seconds = 0.0
    if snapshot.updated_at and snapshot.started_at:
        elapsed_seconds = max(
            (snapshot.updated_at - snapshot.started_at).total_seconds(), 0.0
        )

    return {
        "session_id": snapshot.session_id,
        "session_id_short": snapshot.session_id.split("-")[0],
        "phase": snapshot.phase.value,
        "phase_label": snapshot.phase.value.replace("_", " ").title(),
        "percent_complete": round(percent, 1),
        "processed": processed,
        "total": total,
        "remaining": max(total - processed, 0),
        "cards_created": snapshot.cards_created,
        "cards_updated": snapshot.cards_updated,
        "cards_deleted": snapshot.cards_deleted,
        "cards_restored": snapshot.cards_restored,
        "cards_skipped": snapshot.cards_skipped,
        "errors": snapshot.errors,
        "started_at": snapshot.started_at,
        "updated_at": snapshot.updated_at,
        "elapsed_seconds": elapsed_seconds,
    }


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_notes_table(summary: dict[str, Any]) -> Table:
    table = Table(title="Items", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    total = summary["total"]
    processed = summary["processed"]
    percent = summary["percent_complete"]

    table.add_row("Processed", f"{processed}")
    table.add_row("Total", f"{total}" if total else "–")
    table.add_row("Remaining", f"{summary['remaining']}")
    table.add_row("Progress", f"{percent:.1f}%")
    return table


def _build_cards_table(summary: dict[str, Any]) -> Table:
    table = Table(title="Cards", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Created", str(summary["cards_created"]))
    table.add_row("Updated", str(summary["cards_updated"]))
    table.add_row("Deleted", str(summary["cards_deleted"]))
    table.add_row("Restored", str(summary["cards_restored"]))
    table.add_row("Skipped", str(summary["cards_skipped"]))
    table.add_row("Errors", str(summary["errors"]))
    return table


def _build_status_table(summary: dict[str, Any], meta: ProgressMeta) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan", width=14)
    table.add_column(style="white")

    mode_description = meta.mode_label
    if meta.incremental:
        mode_description += " · incremental"
    if meta.queue_enabled:
        mode_description += " · queue"
    if meta.langgraph_enabled:
        mode_description += " · LangGraph"

    updated_at = summary["updated_at"]
    updated_text = (
        updated_at.strftime("%H:%M:%S") if isinstance(updated_at, datetime) else "–"
    )

    table.add_row("Session", summary["session_id_short"])
    table.add_row(
        "Phase", f"{summary['phase_label']} ({summary['percent_complete']:.1f}%)"
    )
    table.add_row("Mode", mode_description)
    table.add_row("Vault", meta.vault_label)
    table.add_row("Elapsed", _format_duration(summary["elapsed_seconds"]))
    table.add_row("Updated", updated_text)
    return table


def render_progress_panel(summary: dict[str, Any], meta: ProgressMeta) -> Panel:
    """Create a renderable progress panel for Rich live display."""
    status_table = _build_status_table(summary, meta)
    notes_table = _build_notes_table(summary)
    cards_table = _build_cards_table(summary)

    group = Group(status_table, notes_table, cards_table)
    return Panel(group, title="Sync Progress", border_style="cyan")


class SyncProgressReporter:
    """Continuously render sync progress while the engine is running."""

    def __init__(
        self,
        progress_tracker: ProgressTracker | None,
        *,
        console: Console,
        refresh_interval: float = 1.0,
        meta: ProgressMeta,
    ) -> None:
        self._tracker = progress_tracker
        self._console = console
        self._refresh_interval = refresh_interval
        self._meta = meta
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._live: Live | None = None
        self._plain_mode = False

    def start(self) -> None:
        if not self._tracker:
            return
        if not self._console.is_terminal:
            self._plain_mode = True
            summary = summarize_progress(self._tracker.get_snapshot())
            self._console.print(render_progress_panel(summary, self._meta))
            return

        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._tracker:
            return
        if self._plain_mode:
            summary = summarize_progress(self._tracker.get_snapshot())
            self._console.print(render_progress_panel(summary, self._meta))
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._live:
            self._live.stop()

    def _run(self) -> None:
        try:
            with Live(
                self._build_renderable(),
                refresh_per_second=max(int(1 / self._refresh_interval), 1),
                console=self._console,
                transient=False,
            ) as live:
                self._live = live
                while not self._stop_event.wait(self._refresh_interval):
                    live.update(self._build_renderable())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("progress_reporter_error", error=str(exc))

    def _build_renderable(self) -> Panel:
        snapshot = self._tracker.get_snapshot()
        summary = summarize_progress(snapshot)
        return render_progress_panel(summary, self._meta)
