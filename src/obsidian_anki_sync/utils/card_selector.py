"""Interactive card selection UI using Rich."""

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()


@dataclass
class CardCandidate:
    """A generated card candidate for selection."""

    index: int
    fields: dict[str, str]
    is_duplicate: bool = False
    duplicate_reason: str = ""
    quality_score: float | None = None
    quality_reason: str = ""

    def display_summary(self) -> str:
        """Get a short summary for display."""
        # Get first few fields for preview
        preview_fields = list(self.fields.items())[:2]
        preview = ", ".join(
            f"{k}: {v[:30]}..." if len(v) > 30 else f"{k}: {v}"
            for k, v in preview_fields
        )
        return preview


def select_cards_interactive(
    candidates: list[CardCandidate],
    title: str = "Select cards to add to Anki",
) -> list[int]:
    """
    Interactive card selection using Rich terminal UI.

    Args:
        candidates: List of card candidates
        title: Title for the selection interface

    Returns:
        List of selected card indices (0-based)
    """
    if not candidates:
        console.print("[yellow]No cards to select.[/yellow]")
        return []

    selected_indices: set[int] = set()

    # Display cards in a table
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Select", style="cyan", width=8)
    table.add_column("Card", style="white", width=50)
    table.add_column("Status", style="yellow", width=20)

    for candidate in candidates:
        # Build selection indicator
        selected = "✓" if candidate.index in selected_indices else " "
        select_text = f"[{selected}]"

        # Build card preview
        card_preview = candidate.display_summary()

        # Build status
        status_parts = []
        if candidate.is_duplicate:
            status_parts.append("[red]Duplicate[/red]")
        if candidate.quality_score is not None:
            if candidate.quality_score < 0.7:
                status_parts.append("[yellow]Low Quality[/yellow]")
            else:
                status_parts.append("[green]Good[/green]")
        status = " | ".join(status_parts) if status_parts else "[green]Ready[/green]"

        table.add_row(select_text, card_preview, status)

    console.print()
    console.print(table)
    console.print()

    # Interactive selection loop
    while True:
        console.print("[cyan]Commands:[/cyan]")
        console.print(
            "  [bold]number[/bold] - Toggle selection of card (e.g., '1', '2')"
        )
        console.print("  [bold]all[/bold] - Select all cards")
        console.print("  [bold]none[/bold] - Deselect all cards")
        console.print("  [bold]info N[/bold] - Show details for card N")
        console.print("  [bold]done[/bold] - Confirm selection and proceed")
        console.print("  [bold]cancel[/bold] - Cancel without selecting")

        choice = (
            Prompt.ask("\n[bold]Your choice[/bold]", default="done").strip().lower()
        )

        if choice == "done":
            break
        elif choice == "cancel":
            return []
        elif choice == "all":
            selected_indices = set(range(len(candidates)))
            console.print(f"[green]Selected all {len(candidates)} cards[/green]")
        elif choice == "none":
            selected_indices = set()
            console.print("[yellow]Deselected all cards[/yellow]")
        elif choice.startswith("info "):
            try:
                card_num = int(choice.split()[1])
                if 1 <= card_num <= len(candidates):
                    _show_card_details(candidates[card_num - 1])
                else:
                    console.print(f"[red]Invalid card number: {card_num}[/red]")
            except (ValueError, IndexError):
                console.print(
                    "[red]Invalid format. Use 'info N' where N is card number[/red]"
                )
        elif choice.isdigit():
            card_index = int(choice) - 1  # Convert to 0-based
            if 0 <= card_index < len(candidates):
                if card_index in selected_indices:
                    selected_indices.remove(card_index)
                    console.print(f"[yellow]Deselected card {choice}[/yellow]")
                else:
                    selected_indices.add(card_index)
                    console.print(f"[green]Selected card {choice}[/green]")
            else:
                console.print(f"[red]Invalid card number: {choice}[/red]")
        else:
            console.print("[red]Invalid command. Try again.[/red]")

        # Redraw table with updated selections
        console.clear()
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Select", style="cyan", width=8)
        table.add_column("Card", style="white", width=50)
        table.add_column("Status", style="yellow", width=20)

        for candidate in candidates:
            selected = "✓" if candidate.index in selected_indices else " "
            select_text = f"[{selected}] {candidate.index + 1}"
            card_preview = candidate.display_summary()

            status_parts = []
            if candidate.is_duplicate:
                status_parts.append("[red]Duplicate[/red]")
            if candidate.quality_score is not None:
                if candidate.quality_score < 0.7:
                    status_parts.append("[yellow]Low Quality[/yellow]")
                else:
                    status_parts.append("[green]Good[/green]")
            status = (
                " | ".join(status_parts) if status_parts else "[green]Ready[/green]"
            )

            table.add_row(select_text, card_preview, status)

        console.print()
        console.print(table)
        console.print()

    return sorted(selected_indices)


def _show_card_details(candidate: CardCandidate) -> None:
    """Show detailed information about a card candidate."""
    console.print()
    panel_content = f"[bold]Card {candidate.index + 1}[/bold]\n\n"

    # Show fields
    panel_content += "[bold cyan]Fields:[/bold cyan]\n"
    for key, value in candidate.fields.items():
        panel_content += f"  {key}: {value[:100]}{'...' if len(value) > 100 else ''}\n"

    # Show duplicate info
    if candidate.is_duplicate:
        panel_content += (
            f"\n[bold red]Duplicate:[/bold red] {candidate.duplicate_reason}\n"
        )

    # Show quality info
    if candidate.quality_score is not None:
        panel_content += f"\n[bold yellow]Quality Score:[/bold yellow] {candidate.quality_score:.2f}\n"
        if candidate.quality_reason:
            panel_content += (
                f"[bold yellow]Reason:[/bold yellow] {candidate.quality_reason}\n"
            )

    panel = Panel(panel_content, title="Card Details", border_style="blue")
    console.print(panel)
    console.print()
