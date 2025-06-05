"""MCP server implementation for Breeze code indexing."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from breeze.core import BreezeEngine, BreezeConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create MCP server with SSE support
mcp = FastMCP(
    "breeze",
    version="0.1.0",
    description="Semantic code indexing and search using LanceDB",
)

# Global engine instance
engine: Optional[BreezeEngine] = None
engine_lock = asyncio.Lock()


async def get_engine() -> BreezeEngine:
    """Get or create the global engine instance."""
    global engine
    async with engine_lock:
        if engine is None:
            # Load configuration from environment
            config = BreezeConfig(
                data_root=os.environ.get(
                    "BREEZE_DATA_ROOT"
                ),  # Will use platformdirs default if None
                db_name=os.environ.get("BREEZE_DB_NAME", "code_index"),
                embedding_model=os.environ.get(
                    "BREEZE_EMBEDDING_MODEL", "nomic-ai/CodeRankEmbed"
                ),
                trust_remote_code=True,
            )
            engine = BreezeEngine(config)
            await engine.initialize()
        return engine


@mcp.tool()
async def index_repository(
    directories: List[str], force_reindex: bool = False
) -> Dict[str, Any]:
    """
    Index code files from specified directories into the semantic search database.

    This tool scans the specified directories for code files and indexes their content
    using advanced embedding models for high-quality semantic search.

    Args:
        directories: List of absolute paths to directories to index
        force_reindex: If true, will reindex all files even if they already exist in the index

    Returns:
        Statistics about the indexing operation including files indexed, updated, and skipped
    """
    try:
        logger.info(f"Indexing directories: {directories}")
        engine = await get_engine()

        # Validate directories
        valid_dirs = []
        for directory in directories:
            path = Path(directory).resolve()
            if path.exists() and path.is_dir():
                valid_dirs.append(str(path))
            else:
                logger.warning(f"Skipping invalid directory: {directory}")

        if not valid_dirs:
            return {"status": "error", "message": "No valid directories provided"}

        # Perform indexing
        stats = await engine.index_directories(valid_dirs, force_reindex)

        return {
            "status": "success",
            "message": "Indexing completed successfully",
            "statistics": stats.to_dict(),
            "indexed_directories": valid_dirs,
        }

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def search_code(
    query: str, limit: int = 10, min_relevance: float = 0.0
) -> Dict[str, Any]:
    """
    Search for code snippets semantically similar to the query.

    This performs semantic search over previously indexed code files using
    state-of-the-art code embedding models. Results are ranked by relevance.

    Args:
        query: Search query describing what you're looking for
        limit: Maximum number of results to return (default: 10)
        min_relevance: Minimum relevance score threshold (0.0 to 1.0)

    Returns:
        Search results with relevant code snippets and metadata
    """
    try:
        logger.info(f"Searching for: {query}")
        engine = await get_engine()

        # Get index stats first
        stats = await engine.get_stats()
        if stats["total_documents"] == 0:
            return {
                "status": "error",
                "message": "No code has been indexed yet. Use the index_repository tool first.",
            }

        # Perform search
        results = await engine.search(query, limit, min_relevance)

        return {
            "status": "success",
            "query": query,
            "total_results": len(results),
            "results": [result.to_dict() for result in results],
        }

    except Exception as e:
        logger.error(f"Error during search: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def get_index_stats() -> Dict[str, Any]:
    """
    Get statistics about the current code index.

    Returns information about the number of indexed documents, the embedding model
    being used, and the database location.

    Returns:
        Dictionary containing index statistics
    """
    try:
        engine = await get_engine()
        stats = await engine.get_stats()

        return {"status": "success", **stats}

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def list_directory(directory_path: str) -> Dict[str, Any]:
    """
    List the contents of a directory to help identify what to index.

    Args:
        directory_path: Path to the directory to list

    Returns:
        Directory contents with file types and sizes
    """
    try:
        path = Path(directory_path).resolve()

        if not path.exists():
            return {
                "status": "error",
                "message": f"Directory does not exist: {directory_path}",
            }

        if not path.is_dir():
            return {
                "status": "error",
                "message": f"Path is not a directory: {directory_path}",
            }

        contents = []
        for item in sorted(path.iterdir()):
            if item.is_file():
                contents.append(
                    {
                        "name": item.name,
                        "type": "file",
                        "size": item.stat().st_size,
                        "extension": item.suffix,
                    }
                )
            elif item.is_dir():
                # Count items in subdirectory
                try:
                    item_count = len(list(item.iterdir()))
                except PermissionError:
                    item_count = -1

                contents.append(
                    {"name": item.name, "type": "directory", "items": item_count}
                )

        return {
            "status": "success",
            "path": str(path),
            "total_items": len(contents),
            "contents": contents,
        }

    except Exception as e:
        logger.error(f"Error listing directory: {e}")
        return {"status": "error", "message": str(e)}


# Create ASGI app with both transports
def create_app():
    """Create Starlette app with both SSE and HTTP transports."""
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.responses import JSONResponse

    async def health_endpoint(request):
        """Health check endpoint."""
        stats = {"status": "healthy", "server": "breeze-mcp", "version": "0.1.0"}

        try:
            engine = await get_engine()
            db_stats = await engine.get_stats()
            stats["database"] = {
                "initialized": db_stats.get("initialized", False),
                "total_documents": db_stats.get("total_documents", 0),
            }
        except Exception as e:
            stats["database"] = {"error": str(e)}

        return JSONResponse(stats)

    # Create ASGI apps for both transports
    sse_app = mcp.http_app(path="/", transport="sse")
    http_app = mcp.http_app(path="/", transport="streamable-http")

    # Create Starlette app that mounts both
    app = Starlette(
        routes=[
            Mount("/sse", app=sse_app),
            Mount("/mcp", app=http_app),
            Route("/health", endpoint=health_endpoint, methods=["GET"]),
        ],
        lifespan=http_app.lifespan,
    )

    return app


# Run the server
if __name__ == "__main__":
    import uvicorn

    # Get host and port from environment
    host = os.environ.get("BREEZE_HOST", "127.0.0.1")
    port = int(os.environ.get("BREEZE_PORT", "9483"))

    logger.info(f"Starting Breeze MCP server on {host}:{port}")
    logger.info("SSE endpoint: /sse")
    logger.info("HTTP endpoint: /mcp")

    # Create and run the app
    app = create_app()
    uvicorn.run(app, host=host, port=port)
