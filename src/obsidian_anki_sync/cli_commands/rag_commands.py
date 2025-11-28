"""CLI commands for RAG operations."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .shared import get_config_and_logger

rag_app = typer.Typer(
    name="rag",
    help="RAG (Retrieval-Augmented Generation) commands for knowledge base.",
)

console = Console()


@rag_app.command("index")
def index_vault(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force re-index (delete existing index)"),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Index the vault for RAG operations.

    Creates vector embeddings for all markdown files in the vault,
    enabling semantic search, duplicate detection, and context enrichment.
    """
    config, logger = get_config_and_logger(config_path, log_level)

    from obsidian_anki_sync.rag import RAGService

    console.print("[bold]Indexing vault for RAG...[/bold]")

    if force:
        console.print("[yellow]Force mode: existing index will be reset[/yellow]")

    try:
        rag_service = RAGService(config)
        result = rag_service.index_vault(force_reindex=force)

        # Display results
        console.print()
        console.print("[green]Indexing complete![/green]")
        console.print(f"  Chunks indexed: {result['chunks_indexed']}")

        stats = result.get("stats", {})
        console.print(f"  Total chunks: {stats.get('total_chunks', 0)}")
        console.print(f"  Unique files: {stats.get('unique_files', 0)}")

        topics = stats.get("topics", [])
        if topics:
            console.print(f"  Topics: {', '.join(topics[:10])}")
            if len(topics) > 10:
                console.print(f"    ...and {len(topics) - 10} more")

        logger.info(
            "rag_index_complete",
            chunks=result["chunks_indexed"],
            files=stats.get("unique_files", 0),
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error("rag_index_failed", error=str(e))
        raise typer.Exit(1)


@rag_app.command("search")
def search(
    query: Annotated[
        str,
        typer.Argument(help="Search query"),
    ],
    limit: Annotated[
        int,
        typer.Option("-n", "--limit", help="Number of results"),
    ] = 5,
    topic: Annotated[
        str | None,
        typer.Option("-t", "--topic", help="Filter by topic"),
    ] = None,
    min_similarity: Annotated[
        float,
        typer.Option("--min-sim", help="Minimum similarity score (0-1)"),
    ] = 0.3,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Search the knowledge base.

    Performs semantic search across all indexed vault content.
    """
    config, logger = get_config_and_logger(config_path, log_level)

    from obsidian_anki_sync.rag import RAGService

    try:
        rag_service = RAGService(config)

        if not rag_service.is_indexed():
            console.print(
                "[yellow]Vault is not indexed. Run 'rag index' first.[/yellow]"
            )
            raise typer.Exit(1)

        # Build filters
        filters = {}
        if topic:
            filters["topic"] = topic

        # Perform search
        results = rag_service.vector_store.search(
            query=query,
            k=limit,
            filter_metadata=filters if filters else None,
            min_similarity=min_similarity,
        )

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        # Display results
        table = Table(title=f"Search Results for: {query}")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Topic", style="green", width=15)
        table.add_column("Source", style="blue", width=30)
        table.add_column("Content", width=50)

        for result in results:
            # Truncate content for display
            content = (
                result.content[:100] + "..."
                if len(result.content) > 100
                else result.content
            )
            content = content.replace("\n", " ")

            source_name = Path(result.source_file).stem

            table.add_row(
                f"{result.similarity:.2f}",
                result.metadata.get("topic", ""),
                source_name,
                content,
            )

        console.print(table)

        logger.info(
            "rag_search_complete",
            query=query,
            results=len(results),
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error("rag_search_failed", error=str(e))
        raise typer.Exit(1)


@rag_app.command("stats")
def stats(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Show RAG index statistics."""
    config, logger = get_config_and_logger(config_path, log_level)

    from obsidian_anki_sync.rag import RAGService

    try:
        rag_service = RAGService(config)
        stats_data = rag_service.get_stats()

        console.print("[bold]RAG Index Statistics[/bold]")
        console.print()

        if not stats_data.get("indexed"):
            console.print(
                "[yellow]Vault is not indexed. Run 'rag index' first.[/yellow]"
            )
            return

        vs_stats = stats_data.get("vector_store_stats", {})

        table = Table(show_header=False)
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Total Chunks", str(vs_stats.get("total_chunks", 0)))
        table.add_row("Unique Files", str(vs_stats.get("unique_files", 0)))
        table.add_row("Embedding Model", vs_stats.get("embedding_model", "N/A"))
        table.add_row("Storage Path", vs_stats.get("persist_directory", "N/A"))

        # Topics
        topics = vs_stats.get("topics", [])
        if topics:
            table.add_row("Topics", ", ".join(topics[:10]))

        # Chunk types
        chunk_types = vs_stats.get("chunk_types", [])
        if chunk_types:
            table.add_row("Chunk Types", ", ".join(chunk_types))

        # Cache stats
        cache_stats = vs_stats.get("embedding_cache_stats", {})
        if cache_stats:
            table.add_row("Cached Embeddings", str(cache_stats.get("entries", 0)))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error("rag_stats_failed", error=str(e))
        raise typer.Exit(1)


@rag_app.command("similar")
def find_similar(
    question: Annotated[
        str,
        typer.Argument(help="Question to check for duplicates"),
    ],
    threshold: Annotated[
        float,
        typer.Option("-t", "--threshold", help="Similarity threshold (0-1)"),
    ] = 0.85,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Find similar content (duplicate detection).

    Checks if similar Q&A content already exists in the knowledge base.
    """
    config, logger = get_config_and_logger(config_path, log_level)

    from obsidian_anki_sync.rag import RAGService

    try:
        rag_service = RAGService(config)

        if not rag_service.is_indexed():
            console.print(
                "[yellow]Vault is not indexed. Run 'rag index' first.[/yellow]"
            )
            raise typer.Exit(1)

        # Run duplicate check
        result = asyncio.run(
            rag_service.check_duplicate(
                question=question,
                answer="",  # Check based on question only
                threshold=threshold,
            )
        )

        console.print("[bold]Duplicate Check Results[/bold]")
        console.print()

        if result.is_duplicate:
            console.print("[red]Potential duplicate found![/red]")
        else:
            console.print("[green]No duplicates found.[/green]")

        console.print(f"Confidence: {result.confidence:.2%}")
        console.print(f"Recommendation: {result.recommendation}")

        if result.similar_items:
            console.print()
            console.print("[bold]Similar items:[/bold]")
            for item in result.similar_items[:5]:
                source_name = Path(item.source_file).stem
                console.print(f"  - [{item.similarity:.2%}] {source_name}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error("rag_similar_failed", error=str(e))
        raise typer.Exit(1)


@rag_app.command("reset")
def reset_index(
    confirm: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Reset the RAG index (delete all indexed data)."""
    config, logger = get_config_and_logger(config_path, log_level)

    from obsidian_anki_sync.rag import RAGService

    if not confirm:
        console.print("[yellow]This will delete all indexed data.[/yellow]")
        response = console.input("Are you sure? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            console.print("Cancelled.")
            return

    try:
        rag_service = RAGService(config)
        success = rag_service.vector_store.reset()

        if success:
            console.print("[green]Index reset successfully.[/green]")
            logger.info("rag_index_reset")
        else:
            console.print("[red]Failed to reset index.[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error("rag_reset_failed", error=str(e))
        raise typer.Exit(1)
