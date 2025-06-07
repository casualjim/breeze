#!/usr/bin/env python3
"""CLI entry point for Breeze MCP server."""

import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import List, Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from collections import deque

from breeze.core import BreezeEngine, BreezeConfig

# Suppress Voyage AI logging and HTTP request logging
logging.getLogger("voyageai").setLevel(logging.ERROR)
logging.getLogger("voyage").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

try:
    import voyageai
    voyageai.log = "error"  # Set default log level for Voyage AI
except ImportError:
    pass  # voyageai not installed yet

try:
    from dotenv import load_dotenv
except ImportError:
    print(
        "Error: python-dotenv is required. Install it with: pip install python-dotenv"
    )
    sys.exit(1)

load_dotenv()  # Load environment variables from .env file if present

# Create Typer app
app = typer.Typer(
    name="breeze",
    help="""Breeze - Semantic Code Search MCP Server

Supports multiple embedding models including Voyage AI with tier-based rate limits:
- Tier 1 (default): 3M tokens/min, 2000 requests/min
- Tier 2: 6M tokens/min, 4000 requests/min  
- Tier 3: 9M tokens/min, 6000 requests/min

Set tier with --voyage-tier or BREEZE_VOYAGE_TIER environment variable.""",
    add_completion=False,
)

# Console for rich output
console = Console()


def setup_logging(verbose: bool = False, use_rich: bool = True):
    """Set up logging configuration with Rich formatting."""
    from rich.logging import RichHandler

    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    level = logging.DEBUG if verbose else logging.INFO

    # Create handlers
    handlers = [logging.FileHandler("/tmp/breeze.log")]

    if use_rich:
        handlers.append(
            RichHandler(
                console=console,
                show_time=True,
                show_path=verbose,
                markup=True,
                rich_tracebacks=True,
                tracebacks_show_locals=verbose,
            )
        )
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format="%(message)s"
        if use_rich
        else "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,  # Force reconfiguration
    )
    
    # Even in verbose mode, suppress these noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("voyageai").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


@app.command()
def index(
    directories: List[Path] = typer.Argument(
        ...,
        help="Directories to index",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-indexing of all files"
    ),
    concurrent_readers: int = typer.Option(
        int(os.getenv("BREEZE_CONCURRENT_READERS", "20")),
        "--concurrent-readers",
        help="Number of concurrent file readers (default: 20)",
        min=1,
        max=100,
    ),
    concurrent_embedders: int = typer.Option(
        int(os.getenv("BREEZE_CONCURRENT_EMBEDDERS", "10")),
        "--concurrent-embedders",
        help="Number of concurrent embedders (default: 10)",
        min=1,
        max=100,
    ),
    model: Optional[str] = typer.Option(
        os.getenv(
            "BREEZE_EMBEDDING_MODEL", "ibm-granite/granite-embedding-125m-english"
        ),
        "--model",
        "-m",
        help="Embedding model to use (e.g., voyage-code-3, ibm-granite/granite-embedding-125m-english)",
    ),
    device: Optional[str] = typer.Option(
        os.getenv("BREEZE_EMBEDDING_DEVICE", "cpu"),
        "--device",
        help="Device for embeddings: cpu, cuda, mps (auto-detects if not specified)",
    ),
    api_key: Optional[str] = typer.Option(
        os.getenv("BREEZE_EMBEDDING_API_KEY", None),
        "--api-key",
        help="API key for cloud embedding providers (Voyage AI, Google Gemini)",
    ),
    voyage_tier: int = typer.Option(
        int(os.getenv("BREEZE_VOYAGE_TIER", "1")),
        "--voyage-tier",
        help="Voyage AI tier (1: 3M tokens/min, 2: 6M tokens/min, 3: 9M tokens/min)",
        min=1,
        max=3,
    ),
    voyage_requests: int = typer.Option(
        int(os.getenv("BREEZE_VOYAGE_CONCURRENT_REQUESTS", "5")),
        "--voyage-requests",
        help="Max concurrent Voyage AI API requests (auto-calculated based on tier if not set)",
        min=1,
        max=100,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Index code directories for semantic search."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    async def run_indexing():
        # Create config with provided options
        config_args = {
            "concurrent_readers": concurrent_readers,
            "concurrent_embedders": concurrent_embedders,
            "concurrent_writers": 1,  # Always 1 writer to avoid concurrency issues
        }
        if model:
            config_args["embedding_model"] = model
        if device:
            config_args["embedding_device"] = device
        if api_key:
            config_args["embedding_api_key"] = api_key
        if model and model.startswith("voyage-"):
            config_args["voyage_tier"] = voyage_tier
            config_args["voyage_concurrent_requests"] = voyage_requests

        config = BreezeConfig(**config_args)
        engine = BreezeEngine(config)
        await engine.initialize()

        # Convert paths to strings
        dir_strings = [str(d) for d in directories]

        console.print("[bold blue]Indexing directories:[/bold blue]")
        for d in dir_strings:
            console.print(f"  • {d}")

        console.print(
            f"[yellow]Using {concurrent_readers} file readers, {concurrent_embedders} embedders, 1 writer[/yellow]"
        )
        
        # Show Voyage tier info if using Voyage model
        if model and model.startswith("voyage-"):
            rate_limits = config.get_voyage_rate_limits()
            console.print(
                f"[cyan]Voyage AI {rate_limits['tier_name']}: "
                f"{rate_limits['tokens_per_minute']:,} tokens/min, "
                f"{rate_limits['requests_per_minute']:,} requests/min, "
                f"{rate_limits['concurrent_requests']} concurrent requests[/cyan]"
            )

        # If verbose, use simple logging without fancy UI
        if verbose:
            console.print("[dim]Verbose mode - showing all logs inline[/dim]\n")

            # Simple progress callback for verbose mode
            async def simple_progress_callback(progress_info):
                # Track actual processed files
                processed = (
                    progress_info.get("files_indexed", 0) +
                    progress_info.get("files_updated", 0) +
                    progress_info.get("files_skipped", 0) +
                    progress_info.get("errors", 0)
                )
                if processed % 100 == 0 and processed > 0:  # Log every 100 processed files
                    logger.info(
                        f"Progress: {processed} files processed "
                        f"(indexed: {progress_info.get('files_indexed', 0)}, "
                        f"updated: {progress_info.get('files_updated', 0)}, "
                        f"skipped: {progress_info.get('files_skipped', 0)})"
                    )

            # Run indexing with simple logging
            stats = await engine.index_directories_sync(
                dir_strings, force, simple_progress_callback
            )

            console.print("\n[green]✓ Indexing complete![/green]")

        else:
            # Use fancy UI for non-verbose mode with proper layout
            # First count total files
            total_files = 0
            with console.status("[cyan]Counting files...[/cyan]"):
                for directory in directories:
                    for item in directory.rglob("*"):
                        if item.is_file():
                            total_files += 1

            console.print(f"[green]Found {total_files} files to scan[/green]\n")

            # Create layout that fills terminal height
            layout = Layout()
            
            # Get terminal height to properly size sections
            terminal_height = console.size.height
            
            # Fixed sizes for stats and progress
            stats_height = 8
            progress_height = 3
            padding_height = 3  # Reduced padding - just for borders
            
            # Calculate remaining height for logs (leave 1-2 lines at bottom for safety)
            logs_height = max(10, terminal_height - stats_height - progress_height - padding_height - 2)
            
            layout.split_column(
                Layout(name="logs", size=logs_height),     # Top section expands to fill
                Layout(name="stats", size=stats_height),   # Fixed size for stats
                Layout(name="progress", size=progress_height)  # Fixed size for progress bar
            )

            # Keep log messages scaled to available display height
            # Each log line typically takes 1-2 lines when wrapped
            max_log_messages = max(10, logs_height // 2)
            log_messages = deque(maxlen=max_log_messages)
            
            # Create progress bar for the bottom section
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
            )
            index_task = progress.add_task(
                "[yellow]Processing files...", total=total_files
            )

            # Stats tracking
            current_stats = {
                "files_indexed": 0,
                "files_updated": 0,
                "files_skipped": 0,
                "errors": 0,
                "processed": 0
            }

            def update_display():
                """Update all sections of the display."""
                # Update logs section
                combined_logs = Text()
                for msg in log_messages:
                    if isinstance(msg, Text):
                        combined_logs.append(msg)
                        combined_logs.append("\n")
                    else:
                        # Fallback for string messages
                        combined_logs.append(str(msg) + "\n")
                layout["logs"].update(Panel(combined_logs, title="Logs", border_style="blue"))

                # Update stats section
                stats_table = Table(show_header=False, box=None)
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green", justify="right")
                
                stats_table.add_row("Files Indexed", str(current_stats["files_indexed"]))
                stats_table.add_row("Files Updated", str(current_stats["files_updated"]))
                stats_table.add_row("Files Skipped", str(current_stats["files_skipped"]))
                if current_stats["errors"] > 0:
                    stats_table.add_row("Errors", f"[red]{current_stats['errors']}[/red]")
                
                layout["stats"].update(Panel(stats_table, title="Live Statistics", border_style="green"))

                # Update progress section
                layout["progress"].update(Panel(progress, border_style="yellow"))

            # Custom logger handler to capture logs
            class RichLayoutHandler(logging.Handler):
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        # Add timestamp and level with proper Text object for rendering
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        level_styles = {
                            "DEBUG": "dim",
                            "INFO": "blue",
                            "WARNING": "yellow",
                            "ERROR": "red",
                            "CRITICAL": "bold red"
                        }
                        style = level_styles.get(record.levelname, "white")
                        
                        # Create a Text object with proper styling
                        log_text = Text()
                        log_text.append(f"[{timestamp}] ", style="dim")
                        log_text.append(msg, style=style)
                        log_messages.append(log_text)
                    except Exception:
                        pass

            # Add our custom handler temporarily
            layout_handler = RichLayoutHandler()
            layout_handler.setFormatter(logging.Formatter('%(message)s'))
            logger = logging.getLogger()
            logger.addHandler(layout_handler)

            try:
                with Live(layout, console=console, refresh_per_second=4, screen=True, vertical_overflow="visible") as live:
                    # Progress callback that updates both stats and progress
                    async def progress_callback_with_layout(progress_info):
                        # Update stats
                        current_stats["files_indexed"] = progress_info.get("files_indexed", 0)
                        current_stats["files_updated"] = progress_info.get("files_updated", 0)
                        current_stats["files_skipped"] = progress_info.get("files_skipped", 0)
                        current_stats["errors"] = progress_info.get("errors", 0)
                        
                        # Track actual processed files
                        current_stats["processed"] = (
                            current_stats["files_indexed"] +
                            current_stats["files_updated"] +
                            current_stats["files_skipped"] +
                            current_stats["errors"]
                        )
                        
                        # Update progress bar
                        progress.update(
                            index_task, completed=current_stats["processed"]
                        )
                        
                        # Refresh the display
                        update_display()

                    # Initial display update
                    update_display()

                    # Run indexing with the layout callback
                    stats = await engine.index_directories_sync(
                        dir_strings, force, progress_callback_with_layout
                    )

                    # Complete the progress bar
                    progress.update(index_task, completed=total_files)
                    update_display()
            finally:
                # Remove our custom handler
                logger.removeHandler(layout_handler)

        # Display results in a nice table
        table = Table(title="Indexing Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("Files Scanned", str(stats.files_scanned))
        table.add_row("Files Indexed", str(stats.files_indexed))
        table.add_row("Files Updated", str(stats.files_updated))
        table.add_row("Files Skipped", str(stats.files_skipped))
        table.add_row("Errors", str(stats.errors))
        table.add_row("Tokens Processed", f"{stats.total_tokens_processed:,}")

        console.print(table)
        
        # Show failed batch info if any
        engine_stats = await engine.get_stats()
        failed_batches = engine_stats.get("failed_batches", {})
        if failed_batches and failed_batches.get("total", 0) > 0:
            console.print("\n[yellow]Failed Batch Retry Queue:[/yellow]")
            console.print(f"  • Pending: {failed_batches.get('pending', 0)}")
            console.print(f"  • Processing: {failed_batches.get('processing', 0)}")
            console.print(f"  • Abandoned: {failed_batches.get('abandoned', 0)}")
            if "next_retry_at" in failed_batches:
                console.print(f"  • Next retry: {failed_batches['next_retry_at']}")
        
        console.print("\n[bold green]✓ Indexing complete![/bold green]")

    asyncio.run(run_indexing())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    min_relevance: float = typer.Option(
        0.0, "--min-relevance", "-r", help="Minimum relevance score (0.0-1.0)"
    ),
    model: Optional[str] = typer.Option(
        os.getenv(
            "BREEZE_EMBEDDING_MODEL", "ibm-granite/granite-embedding-125m-english"
        ),
        "--model",
        "-m",
        help="Embedding model to use (must match the model used for indexing)",
    ),
    device: Optional[str] = typer.Option(
        os.getenv("BREEZE_EMBEDDING_DEVICE", "cpu"),
        "--device",
        help="Device for embeddings: cpu, cuda, mps",
    ),
    api_key: Optional[str] = typer.Option(
        os.getenv("BREEZE_EMBEDDING_API_KEY", None),
        "--api-key",
        help="API key for cloud embedding providers",
    ),
    voyage_tier: int = typer.Option(
        int(os.getenv("BREEZE_VOYAGE_TIER", "1")),
        "--voyage-tier",
        help="Voyage AI tier (1-3)",
        min=1,
        max=3,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Search indexed code semantically."""
    setup_logging(verbose)

    async def run_search():
        # Create config with provided options (same as index command)
        config_args = {}
        if model:
            config_args["embedding_model"] = model
        if device:
            config_args["embedding_device"] = device
        if api_key:
            config_args["embedding_api_key"] = api_key
        if model and model.startswith("voyage-"):
            config_args["voyage_tier"] = voyage_tier

        config = BreezeConfig(**config_args)
        engine = BreezeEngine(config)
        await engine.initialize()

        with console.status(f"[bold blue]Searching for: {query}[/bold blue]"):
            results = await engine.search(query, limit, min_relevance)

        if not results:
            console.print("[red]No results found.[/red]")
            return

        console.print(f"\n[bold green]Found {len(results)} results:[/bold green]\n")

        for i, result in enumerate(results, 1):
            # Create a panel for each result
            console.print(f"[bold cyan]{i}. {result.file_path}[/bold cyan]")
            console.print(f"   [dim]Score: {result.relevance_score:.3f}[/dim]")
            console.print(f"   [dim]Type: {result.file_type}[/dim]")
            console.print()

            # Show snippet with syntax highlighting if possible
            if result.snippet:
                # Truncate long snippets
                snippet_lines = result.snippet.split("\n")[:10]
                snippet = "\n".join(snippet_lines)
                if len(snippet_lines) < result.snippet.count("\n"):
                    snippet += "\n..."

                console.print(snippet, style="dim")

            console.print("─" * 80)

    asyncio.run(run_search())


@app.command()
def serve(
    host: Optional[str] = typer.Option(
        None, "--host", help="Server host (default: from env or 127.0.0.1)"
    ),
    port: Optional[int] = typer.Option(
        None, "--port", help="Server port (default: from env or 9483)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Run as MCP server."""
    setup_logging(verbose)

    # Import here to avoid circular imports
    from breeze.mcp.server import create_app, shutdown_engine
    import uvicorn
    import signal

    # Get host and port from args or environment
    host = host or os.environ.get("BREEZE_HOST", "127.0.0.1")
    port = port or int(os.environ.get("BREEZE_PORT", "9483"))

    console.print("[bold green]Starting Breeze MCP server[/bold green]")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  SSE endpoint: http://{host}:{port}/sse")
    console.print(f"  HTTP endpoint: http://{host}:{port}/mcp")
    console.print(f"  Health check: http://{host}:{port}/health")
    console.print()
    console.print("[dim]Press CTRL+C to stop[/dim]")

    # Set up signal handlers for graceful shutdown
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            console.print("\n[yellow]Shutdown signal received. Shutting down gracefully...[/yellow]")
            # Schedule shutdown in async context
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(shutdown_engine())
            except RuntimeError:
                # No event loop running, will shut down naturally
                pass
        else:
            console.print("\n[red]Force shutdown requested[/red]")
            sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and run the app
    app = create_app()
    try:
        uvicorn.run(app, host=host, port=port, log_config=None)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
    finally:
        # Ensure cleanup happens
        asyncio.run(shutdown_engine())


@app.command()
def stats(
    model: Optional[str] = typer.Option(
        os.getenv(
            "BREEZE_EMBEDDING_MODEL", "ibm-granite/granite-embedding-125m-english"
        ),
        "--model",
        "-m",
        help="Embedding model to use (must match the model used for indexing)",
    ),
    device: Optional[str] = typer.Option(
        os.getenv("BREEZE_EMBEDDING_DEVICE", "cpu"),
        "--device",
        help="Device for embeddings: cpu, cuda, mps",
    ),
    api_key: Optional[str] = typer.Option(
        os.getenv("BREEZE_EMBEDDING_API_KEY", None),
        "--api-key",
        help="API key for cloud embedding providers",
    ),
    voyage_tier: int = typer.Option(
        int(os.getenv("BREEZE_VOYAGE_TIER", "1")),
        "--voyage-tier",
        help="Voyage AI tier (1-3)",
        min=1,
        max=3,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Show index statistics."""
    setup_logging(verbose)

    async def show_stats():
        # Create config with provided options (same as index command)
        config_args = {}
        if model:
            config_args["embedding_model"] = model
        if device:
            config_args["embedding_device"] = device
        if api_key:
            config_args["embedding_api_key"] = api_key
        if model and model.startswith("voyage-"):
            config_args["voyage_tier"] = voyage_tier

        config = BreezeConfig(**config_args)
        engine = BreezeEngine(config)
        await engine.initialize()

        with console.status("[bold blue]Getting index statistics...[/bold blue]"):
            stats = await engine.get_stats()

        table = Table(title="Index Statistics", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Documents", str(stats.get("total_documents", 0)))
        table.add_row("Initialized", "✓" if stats.get("initialized", False) else "✗")
        table.add_row("Embedding Model", stats.get("model", "Unknown"))
        table.add_row("Database Path", stats.get("database_path", "Unknown"))

        # Show failed batch info if any
        failed_batches = stats.get("failed_batches", {})
        if failed_batches and failed_batches.get("total", 0) > 0:
            table.add_row("Failed Batches (Total)", str(failed_batches.get("total", 0)))
            table.add_row("Failed Batches (Pending)", str(failed_batches.get("pending", 0)))
            if "next_retry_at" in failed_batches:
                table.add_row("Next Retry", failed_batches["next_retry_at"])

        console.print(table)

    asyncio.run(show_stats())


@app.command()
def watch(
    project_name: str = typer.Argument(..., help="Name of the project to watch"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Watch a registered project for file changes and auto-index."""
    setup_logging(verbose, use_rich=False)  # Disable rich for continuous output

    async def run_watch():
        config = BreezeConfig()
        engine = BreezeEngine(config)
        await engine.initialize()

        # Find the project
        projects = await engine.list_projects()
        project = None
        for p in projects:
            if p.name.lower() == project_name.lower():
                project = p
                break

        if not project:
            console.print(f"[red]Project '{project_name}' not found.[/red]")
            console.print("\n[yellow]Available projects:[/yellow]")
            for p in projects:
                console.print(f"  • {p.name}")
            return

        console.print(
            f"[bold green]Starting file watcher for project: {project.name}[/bold green]"
        )
        console.print("[dim]Watching paths:[/dim]")
        for path in project.paths:
            console.print(f"  • {path}")
        console.print()
        console.print("[dim]Press CTRL+C to stop watching[/dim]\n")

        # Event callback for file watching
        async def watch_callback(event):
            event_type = event.get("type", "unknown")

            if event_type == "watching_started":
                console.print("[green]✓ File watching started[/green]")
            elif event_type == "indexing_started":
                files = event.get("files", [])
                console.print(
                    f"\n[yellow]⟳ Changes detected in {len(files)} file(s)[/yellow]"
                )
                for file in files[:5]:  # Show first 5 files
                    console.print(f"  • {Path(file).name}")
                if len(files) > 5:
                    console.print(f"  ... and {len(files) - 5} more")
            elif event_type == "indexing_progress":
                # Show progress inline
                progress = event.get("files_scanned", 0)
                total = event.get("total_files", 0)
                if total > 0:
                    percent = (progress / total) * 100
                    console.print(
                        f"  [dim]Progress: {progress}/{total} ({percent:.0f}%)[/dim]",
                        end="\r",
                    )
            elif event_type == "indexing_completed":
                stats = event.get("stats", {})
                console.print("\n[green]✓ Indexing complete:[/green]")
                console.print(f"  • Indexed: {stats.get('files_indexed', 0)}")
                console.print(f"  • Updated: {stats.get('files_updated', 0)}")
                console.print(f"  • Skipped: {stats.get('files_skipped', 0)}")
                if stats.get("errors", 0) > 0:
                    console.print(f"  • [red]Errors: {stats.get('errors', 0)}[/red]")
                console.print()
            elif event_type == "indexing_error":
                error = event.get("error", "Unknown error")
                console.print(f"\n[red]✗ Indexing error: {error}[/red]")

        # Start watching
        success = await engine.start_watching(project.id, watch_callback)
        if not success:
            console.print("[red]Failed to start file watching[/red]")
            return

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Stopping file watcher...[/yellow]")
            await engine.stop_watching(project.id)
            console.print("[green]✓ File watching stopped[/green]")

    try:
        asyncio.run(run_watch())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
