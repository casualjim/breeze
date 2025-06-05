#!/usr/bin/env python3
"""CLI entry point for Breeze MCP server."""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from collections import deque

from breeze.core import BreezeEngine, BreezeConfig


# Create Typer app
app = typer.Typer(
    name="breeze",
    help="Breeze - Semantic Code Search MCP Server",
    add_completion=False,
)

# Console for rich output
console = Console()


def setup_logging(verbose: bool = False, use_rich: bool = True):
    """Set up logging configuration with Rich formatting."""
    from rich.logging import RichHandler
    
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
        format="%(message)s" if use_rich else "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True  # Force reconfiguration
    )


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
        False, 
        "--force", 
        "-f", 
        help="Force re-indexing of all files"
    ),
    workers: int = typer.Option(
        20,
        "--workers",
        "-w",
        help="Number of concurrent workers (default: 20)",
        min=1,
        max=100,
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", 
        "-v", 
        help="Enable verbose logging"
    ),
):
    """Index code directories for semantic search."""
    setup_logging(verbose)
    
    async def run_indexing():
        config = BreezeConfig()
        engine = BreezeEngine(config)
        
        # Convert paths to strings
        dir_strings = [str(d) for d in directories]
        
        console.print(f"[bold blue]Indexing directories:[/bold blue]")
        for d in dir_strings:
            console.print(f"  • {d}")
        
        # First count total files
        total_files = 0
        with console.status("[cyan]Counting files...[/cyan]"):
            for directory in directories:
                for item in directory.rglob("*"):
                    if item.is_file():
                        total_files += 1
        
        console.print(f"[green]Found {total_files} files to scan[/green]")
        console.print(f"[yellow]Using {workers} concurrent workers[/yellow]\n")
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=7),
            Layout(name="logs", size=10),
            Layout(name="stats", size=6),
        )
        
        # Create progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        
        # Create log buffer (last 8 log entries)
        log_entries = deque(maxlen=8)
        
        # Stats tracking
        current_stats = {
            "files_scanned": 0,
            "files_indexed": 0,
            "files_updated": 0,
            "files_skipped": 0,
            "errors": 0,
            "current_file": "",
        }
        
        # Create indexing task
        index_task = progress.add_task(
            "[yellow]Indexing files...", 
            total=total_files
        )
        
        # Progress callback for the engine
        async def progress_callback(progress_info):
            current_stats.update(progress_info)
            progress.update(
                index_task,
                completed=progress_info["files_scanned"],
            )
            
            # Add log entry
            file_name = Path(progress_info['current_file']).name
            if progress_info["files_indexed"] > current_stats.get("last_indexed", 0):
                log_entries.append(f"[green]✓ Indexed: {file_name}[/green]")
                current_stats["last_indexed"] = progress_info["files_indexed"]
            elif progress_info["files_updated"] > current_stats.get("last_updated", 0):
                log_entries.append(f"[yellow]↻ Updated: {file_name}[/yellow]")
                current_stats["last_updated"] = progress_info["files_updated"]
            elif progress_info["files_skipped"] > current_stats.get("last_skipped", 0):
                log_entries.append(f"[dim]- Skipped: {file_name}[/dim]")
                current_stats["last_skipped"] = progress_info["files_skipped"]
            elif progress_info.get("errors", 0) > current_stats.get("last_errors", 0):
                log_entries.append(f"[red]✗ Error: {file_name}[/red]")
                current_stats["last_errors"] = progress_info.get("errors", 0)
        
        def generate_layout():
            # Progress panel
            layout["progress"].update(
                Panel(
                    progress,
                    title="[bold blue]Progress[/bold blue]",
                    border_style="blue",
                )
            )
            
            # Logs panel
            log_text = Text.from_markup("\n".join(log_entries))
            layout["logs"].update(
                Panel(
                    log_text,
                    title="[bold cyan]Activity Log[/bold cyan]",
                    border_style="cyan",
                )
            )
            
            # Stats panel
            stats_table = Table(show_header=False, box=None)
            stats_table.add_column("Metric", style="dim")
            stats_table.add_column("Value", style="bold")
            
            stats_table.add_row("Files Scanned", str(current_stats.get("files_scanned", 0)))
            stats_table.add_row("Files Indexed", f"[green]{current_stats.get('files_indexed', 0)}[/green]")
            stats_table.add_row("Files Updated", f"[yellow]{current_stats.get('files_updated', 0)}[/yellow]")
            stats_table.add_row("Files Skipped", f"[dim]{current_stats.get('files_skipped', 0)}[/dim]")
            stats_table.add_row("Errors", f"[red]{current_stats.get('errors', 0)}[/red]")
            
            layout["stats"].update(
                Panel(
                    stats_table,
                    title="[bold magenta]Statistics[/bold magenta]",
                    border_style="magenta",
                )
            )
            
            return layout
        
        # Run indexing with live display
        # Temporarily set logging to ERROR level during live display
        original_level = logging.root.level
        if not verbose:
            logging.root.setLevel(logging.ERROR)
            
        with Live(generate_layout(), refresh_per_second=2, console=console) as live:
            # Update function that refreshes the display
            async def progress_callback_with_update(progress_info):
                await progress_callback(progress_info)
                live.update(generate_layout())
            
            # Perform indexing
            stats = await engine.index_directories(
                dir_strings, 
                force, 
                progress_callback_with_update,
                num_workers=workers  # Use configured number of workers
            )
            
            # Update progress to completion
            progress.update(index_task, completed=total_files)
            live.update(generate_layout())
        
        # Restore original logging level
        logging.root.setLevel(original_level)
        
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
        console.print("[bold green]✓ Indexing complete![/bold green]")
    
    asyncio.run(run_indexing())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    min_relevance: float = typer.Option(
        0.0, 
        "--min-relevance", 
        "-m", 
        help="Minimum relevance score (0.0-1.0)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Search indexed code semantically."""
    setup_logging(verbose)
    
    async def run_search():
        config = BreezeConfig()
        engine = BreezeEngine(config)
        
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
                snippet_lines = result.snippet.split('\n')[:10]
                snippet = '\n'.join(snippet_lines)
                if len(snippet_lines) < result.snippet.count('\n'):
                    snippet += "\n..."
                
                console.print(snippet, style="dim")
            
            console.print("─" * 80)
    
    asyncio.run(run_search())


@app.command()
def serve(
    host: Optional[str] = typer.Option(
        None, 
        "--host", 
        help="Server host (default: from env or 127.0.0.1)"
    ),
    port: Optional[int] = typer.Option(
        None, 
        "--port", 
        help="Server port (default: from env or 9483)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run as MCP server."""
    setup_logging(verbose)
    
    # Import here to avoid circular imports
    from breeze.mcp.server import create_app
    import uvicorn
    
    # Get host and port from args or environment
    host = host or os.environ.get("BREEZE_HOST", "127.0.0.1")
    port = port or int(os.environ.get("BREEZE_PORT", "9483"))
    
    console.print(f"[bold green]Starting Breeze MCP server[/bold green]")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  SSE endpoint: http://{host}:{port}/sse")
    console.print(f"  HTTP endpoint: http://{host}:{port}/mcp")
    console.print(f"  Health check: http://{host}:{port}/health")
    console.print()
    console.print("[dim]Press CTRL+C to stop[/dim]")
    
    # Create and run the app
    app = create_app()
    uvicorn.run(app, host=host, port=port)


@app.command()
def stats(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Show index statistics."""
    setup_logging(verbose)
    
    async def show_stats():
        config = BreezeConfig()
        engine = BreezeEngine(config)
        
        with console.status("[bold blue]Getting index statistics...[/bold blue]"):
            stats = await engine.get_stats()
        
        table = Table(title="Index Statistics", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Documents", str(stats.get("total_documents", 0)))
        table.add_row("Initialized", "✓" if stats.get("initialized", False) else "✗")
        table.add_row("Embedding Model", stats.get("model", "Unknown"))
        table.add_row("Database Path", stats.get("database_path", "Unknown"))
        
        console.print(table)
    
    asyncio.run(show_stats())


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
