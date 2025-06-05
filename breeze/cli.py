#!/usr/bin/env python3
"""CLI entry point for Breeze MCP server."""

import argparse
import asyncio
import logging
from pathlib import Path

from breeze.core import BreezeEngine, BreezeConfig


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


async def index_command(args):
    """Handle the index command."""
    config = BreezeConfig()
    engine = BreezeEngine(config)

    directories = [str(Path(d).resolve()) for d in args.directories]
    print(f"Indexing directories: {', '.join(directories)}")

    stats = await engine.index_directories(directories, args.force)

    print("\nIndexing complete!")
    print(f"Files scanned: {stats.files_scanned}")
    print(f"Files indexed: {stats.files_indexed}")
    print(f"Files updated: {stats.files_updated}")
    print(f"Files skipped: {stats.files_skipped}")
    print(f"Errors: {stats.errors}")


async def search_command(args):
    """Handle the search command."""
    config = BreezeConfig()
    engine = BreezeEngine(config)

    print(f"Searching for: {args.query}")
    results = await engine.search(args.query, args.limit, args.min_relevance)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.file_path} (score: {result.relevance_score:.3f})")
        print("-" * 80)
        print(result.snippet)


def serve_command(args):
    """Handle the serve command (run MCP server)."""
    # Import here to avoid circular imports
    from breeze.mcp.server import create_app
    import os
    import uvicorn

    # Get host and port from args or environment
    host = args.host or os.environ.get("BREEZE_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("BREEZE_PORT", "9483"))

    print(f"Starting Breeze MCP server on {host}:{port}")
    print("SSE endpoint: /sse")
    print("HTTP endpoint: /")

    # Create and run the app
    app = create_app()
    uvicorn.run(app, host=host, port=port)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Breeze - Semantic Code Search MCP Server"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index code directories")
    index_parser.add_argument("directories", nargs="+", help="Directories to index")
    index_parser.add_argument(
        "-f", "--force", action="store_true", help="Force re-indexing of all files"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed code")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results (default: 10)",
    )
    search_parser.add_argument(
        "-m",
        "--min-relevance",
        type=float,
        default=0.0,
        help="Minimum relevance score (0.0-1.0, default: 0.0)",
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run as MCP server")
    serve_parser.add_argument("--host", help="Server host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, help="Server port (default: 9483)")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    # Handle commands
    if args.command == "index":
        asyncio.run(index_command(args))
    elif args.command == "search":
        asyncio.run(search_command(args))
    elif args.command == "serve":
        serve_command(args)
    else:
        # Default to serve if no command specified
        serve_command(args)


if __name__ == "__main__":
    main()
