"""Progress display utilities for sync operations."""

import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


class ProgressDisplay:
    """Display progress and reflections during sync operations."""

    def __init__(self, show_reflections: bool = True):
        """Initialize progress display.

        Args:
            show_reflections: Whether to show LLM reflections/reasoning
        """
        self.show_reflections = show_reflections
        self.current_operation = "Initializing..."
        self.current_note = ""
        self.reflections: list[str] = []
        self.max_reflections = 5  # Keep last N reflections visible

    def create_progress_bar(self, total: int) -> Progress:
        """Create a Rich Progress bar.

        Args:
            total: Total number of items to process

        Returns:
            Progress instance
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

    def create_status_panel(self) -> Panel:
        """Create a status panel showing current operation and note.

        Returns:
            Panel instance
        """
        status_text = f"[cyan]Operation:[/cyan] {self.current_operation}"
        if self.current_note:
            # Truncate long note names
            note_display = (
                self.current_note[:60] + "..." if len(self.current_note) > 60 else self.current_note
            )
            status_text += f"\n[cyan]Note:[/cyan] {note_display}"

        return Panel(status_text, title="[bold]Status[/bold]", border_style="blue")

    def create_reflections_panel(self) -> Panel | None:
        """Create a panel showing LLM reflections/reasoning.

        Returns:
            Panel instance or None if no reflections
        """
        if not self.show_reflections or not self.reflections:
            return None

        # Show last N reflections
        recent_reflections = self.reflections[-self.max_reflections :]
        reflection_text = "\n".join(
            [f"[dim]•[/dim] {reflection[:200]}..." if len(reflection) > 200 else f"[dim]•[/dim] {reflection}"
             for reflection in recent_reflections]
        )

        return Panel(
            reflection_text,
            title="[bold]LLM Reflections[/bold]",
            border_style="yellow",
            height=min(len(recent_reflections) + 3, 10),
        )

    def update_operation(self, operation: str, note: str = "") -> None:
        """Update current operation status.

        Args:
            operation: Current operation name
            note: Current note being processed
        """
        self.current_operation = operation
        self.current_note = note

    def add_reflection(self, reflection: str) -> None:
        """Add a reflection/reasoning from LLM.

        Args:
            reflection: Reflection text to display
        """
        if not self.show_reflections:
            return

        # Clean up reflection (remove markdown, truncate)
        cleaned = reflection.strip()
        if len(cleaned) > 300:
            cleaned = cleaned[:300] + "..."

        self.reflections.append(cleaned)
        # Keep only last N reflections
        if len(self.reflections) > self.max_reflections * 2:
            self.reflections = self.reflections[-self.max_reflections :]

    def clear_reflections(self) -> None:
        """Clear all reflections."""
        self.reflections.clear()

    def render_live_display(
        self, progress: Progress, task_id: int, stats: dict[str, Any] | None = None
    ) -> Live:
        """Create a live display with progress and status.

        Args:
            progress: Progress instance
            task_id: Task ID from progress
            stats: Optional statistics to display

        Returns:
            Live instance
        """
        from rich.layout import Layout
        from rich.text import Text

        def generate_layout() -> Layout:
            layout = Layout()

            # Top section: Progress bar
            layout.split_column(
                Layout(name="progress", size=3),
                Layout(name="status"),
            )

            # Status section: Operation and reflections
            status_layout = Layout()
            status_layout.split_row(
                Layout(self.create_status_panel(), name="status_panel"),
                Layout(name="reflections"),
            )

            # Add reflections panel if available
            reflections_panel = self.create_reflections_panel()
            if reflections_panel:
                status_layout["reflections"].update(reflections_panel)
            else:
                status_layout["reflections"].update("")

            layout["status"].update(status_layout)

            # Add stats table if provided
            if stats:
                stats_table = self._create_stats_table(stats)
                layout.split_row(Layout(name="main"), Layout(stats_table, size=12))
                layout["main"].update(status_layout)
            else:
                layout["status"].update(status_layout)

            return layout

        return Live(generate_layout(), refresh_per_second=2, console=console)

    def _create_stats_table(self, stats: dict[str, Any]) -> Table:
        """Create a statistics table.

        Args:
            stats: Statistics dictionary

        Returns:
            Table instance
        """
        table = Table(title="[bold]Statistics[/bold]", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        return table


def extract_reasoning_from_response(response: dict[str, Any], model: str) -> str | None:
    """Extract reasoning/reflection from LLM response if available.

    Args:
        response: LLM response dictionary
        model: Model name

    Returns:
        Reasoning text or None
    """
    # Check for reasoning field (thinking models)
    if "reasoning" in response and response["reasoning"]:
        return response["reasoning"]

    # Check for thinking/reasoning in message
    message = response.get("message", {})
    if isinstance(message, dict):
        if "reasoning" in message:
            return message["reasoning"]
        if "thinking" in message:
            return message["thinking"]

    # Check for extraction_notes or similar fields
    if "extraction_notes" in response:
        notes = response["extraction_notes"]
        if notes and len(notes) > 50:  # Only show substantial notes
            return notes

    return None

