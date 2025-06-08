"""MCP server implementation for Breeze code indexing."""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from fastmcp import FastMCP, Context
from fastmcp.utilities.logging import get_logger

from breeze.core import BreezeEngine, BreezeConfig
from breeze.core.tokenizer_utils import load_tokenizer_for_model
from breeze.core.models import IndexingTask

# Use FastMCP's logging utility for consistent formatting
logger = get_logger("breeze.server")

# Create MCP server with SSE support
mcp = FastMCP(
    "breeze",
    version="0.1.0",
    description="Semantic code indexing and search using LanceDB",
)

# Global engine instance
engine: Optional[BreezeEngine] = None
engine_lock = asyncio.Lock()


async def shutdown_engine():
    """Shutdown the global engine gracefully."""
    global engine
    if engine:
        logger.info("Shutting down engine...")
        try:
            await engine.shutdown()
        except Exception as e:
            logger.error(f"Error during engine shutdown: {e}")
        finally:
            engine = None
        logger.info("Engine shutdown complete")


async def get_engine() -> BreezeEngine:
    """Get or create the global engine instance."""
    global engine
    async with engine_lock:
        if engine is None:
            logger.info("Creating new BreezeEngine instance...")
            # Load configuration from environment
            config = BreezeConfig(
                data_root=os.environ.get(
                    "BREEZE_DATA_ROOT"
                ),  # Will use platformdirs default if None
                db_name=os.environ.get("BREEZE_DB_NAME", "code_index"),
                embedding_model=os.environ.get(
                    "BREEZE_EMBEDDING_MODEL",
                    "ibm-granite/granite-embedding-125m-english",
                ),
                embedding_device=os.environ.get(
                    "BREEZE_EMBEDDING_DEVICE", "cpu"
                ),  # Will auto-detect if cpu
                trust_remote_code=True,
                embedding_api_key=os.environ.get("BREEZE_EMBEDDING_API_KEY"),
                concurrent_readers=int(
                    os.environ.get("BREEZE_CONCURRENT_READERS", "20")
                ),
                concurrent_embedders=int(
                    os.environ.get("BREEZE_CONCURRENT_EMBEDDERS", "10")
                ),
                # concurrent_writers is always 1 (hardcoded in BreezeConfig)
                voyage_concurrent_requests=int(
                    os.environ.get("BREEZE_VOYAGE_CONCURRENT_REQUESTS", "5")
                ),
            )
            
            # Load tokenizer once for the model
            tokenizer = load_tokenizer_for_model(
                config.embedding_model, 
                trust_remote_code=config.trust_remote_code
            )
            
            engine = BreezeEngine(config, tokenizer=tokenizer)
            try:
                await engine.initialize()
                logger.info("BreezeEngine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize BreezeEngine: {e}", exc_info=True)
                engine = None
                raise
        return engine


@mcp.tool()
async def index_repository(
    directories: List[str], force_reindex: bool = False, ctx: Context = None
) -> Dict[str, Any]:
    """
    Index code files from specified directories into the semantic search database.

    This tool queues an indexing task that scans the specified directories for code files
    and indexes their content using advanced embedding models for high-quality semantic search.
    The task runs asynchronously in the background and returns immediately with a task ID.

    Args:
        directories: List of absolute paths to directories to index
        force_reindex: If true, will reindex all files even if they already exist in the index

    Returns:
        Task information including task_id and queue position
    """
    try:
        if ctx:
            await ctx.info(f"Queueing indexing for directories: {directories}")
        logger.info(f"Queueing indexing for directories: {directories}")
        engine = await get_engine()

        # Validate directories
        valid_dirs = []
        for directory in directories:
            path = Path(directory).resolve()
            if path.exists() and path.is_dir():
                valid_dirs.append(str(path))
            else:
                if ctx:
                    await ctx.warning(f"Skipping invalid directory: {directory}")
                logger.warning(f"Skipping invalid directory: {directory}")

        if not valid_dirs:
            return {"status": "error", "message": "No valid directories provided"}

        # Create indexing task
        task = IndexingTask(paths=valid_dirs, force_reindex=force_reindex)

        # Define progress callback
        async def progress_callback(stats):
            # Progress is now exposed via resources
            pass

        # Add task to queue
        queue_position = await engine._indexing_queue.add_task(task, progress_callback)

        if ctx:
            await ctx.info(
                f"Indexing task {task.task_id} queued at position {queue_position}"
            )
        logger.info(f"Indexing task {task.task_id} queued at position {queue_position}")

        return {
            "status": "success",
            "message": "Indexing task queued successfully",
            "task_id": task.task_id,
            "queue_position": queue_position,
            "indexed_directories": valid_dirs,
        }

    except Exception as e:
        if ctx:
            await ctx.error(f"Error queueing indexing task: {e}")
        logger.error(f"Error queueing indexing task: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def search_code(
    query: str, limit: int = 10, min_relevance: float = 0.0, ctx: Context = None
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
        if ctx:
            await ctx.info(f"Searching for: {query}")
        logger.info(f"Searching for: {query}")
        engine = await get_engine()

        # Get index stats first
        stats = await engine.get_stats()
        if stats.total_documents == 0:
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
            "results": [result.model_dump() for result in results],
        }

    except Exception as e:
        if ctx:
            await ctx.error(f"Error during search: {e}")
        logger.error(f"Error during search: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def get_index_stats(ctx: Context = None) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the code index and indexing queue.

    Returns information about:
    - Number of indexed documents
    - Embedding model being used
    - Database location
    - Failed batch statistics
    - Indexing queue status (queue size, current task, queued tasks)

    Returns:
        Dictionary containing index and queue statistics
    """
    try:
        engine = await get_engine()
        stats = await engine.get_stats()

        # Convert dataclass to dict for JSON response
        return {
            "status": "success",
            "total_documents": stats.total_documents,
            "initialized": stats.initialized,
            "model": stats.model,
            "database_path": stats.database_path,
            "failed_batches": asdict(stats.failed_batches) if stats.failed_batches else None,
            "indexing_queue": asdict(stats.indexing_queue) if stats.indexing_queue else None,
        }

    except Exception as e:
        if ctx:
            await ctx.error(f"Error getting stats: {e}")
        logger.error(f"Error getting stats: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def list_directory(directory_path: str, ctx: Context = None) -> Dict[str, Any]:
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
        if ctx:
            await ctx.error(f"Error listing directory: {e}")
        logger.error(f"Error listing directory: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def register_project(
    name: str, paths: List[str], auto_index: bool = True, ctx: Context = None
) -> Dict[str, Any]:
    """
    Register a new project and start watching it for changes.

    This will:
    1. Register the project in the database
    2. Optionally perform initial indexing
    3. Start watching for file changes

    The system automatically detects code files using content analysis rather than
    file extensions, ensuring all relevant code is indexed.

    Args:
        name: Name of the project
        paths: List of directory paths to track
        auto_index: Whether to perform initial indexing (default: True)

    Returns:
        Project registration details including project ID
    """
    try:
        engine = await get_engine()

        # Register the project
        project = await engine.add_project(
            name=name, paths=paths, auto_index=auto_index
        )

        # Define event callback for file watching
        async def watch_callback(event):
            # Events are now exposed via resources
            pass

        # Start watching the project
        watch_success = await engine.start_watching(project.id, watch_callback)

        result = {
            "status": "success",
            "project_id": project.id,
            "name": project.name,
            "paths": project.paths,
            "watching": watch_success,
            "message": f"Project '{name}' registered and watching started",
        }

        # Optionally perform initial indexing
        if auto_index:
            # Create indexing task with progress notifications
            task = await engine.create_indexing_task(
                paths=project.paths,
                project_id=project.id,
                progress_callback=None,  # Progress is exposed via resources
            )
            result["indexing_task_id"] = task.task_id
            result["message"] += f". Initial indexing started (task: {task.task_id})"

        return result

    except Exception as e:
        if ctx:
            await ctx.error(f"Error registering project: {e}")
        logger.error(f"Error registering project: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def unregister_project(project_id: str, ctx: Context = None) -> Dict[str, Any]:
    """
    Unregister a project, stop watching it, and remove it from tracking.

    This will stop file watching and remove the project from the database.
    The indexed code will remain searchable.

    Args:
        project_id: ID of the project to unregister

    Returns:
        Confirmation of project removal
    """
    try:
        engine = await get_engine()

        # Get project info before removing
        project = await engine.get_project(project_id)
        if not project:
            return {"status": "error", "message": f"Project not found: {project_id}"}

        # Remove the project (this also stops watching)
        success = await engine.remove_project(project_id)

        if success:
            return {
                "status": "success",
                "message": f"Project '{project.name}' unregistered and watching stopped",
                "project_id": project_id,
                "name": project.name,
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to unregister project: {project_id}",
            }

    except Exception as e:
        if ctx:
            await ctx.error(f"Error unregistering project: {e}")
        logger.error(f"Error unregistering project: {e}")
        return {"status": "error", "message": str(e)}




@mcp.tool()
async def list_projects(ctx: Context = None) -> Dict[str, Any]:
    """
    List all registered projects with their current status.

    Returns:
        List of all projects with their details including watch status
    """
    try:
        engine = await get_engine()
        projects = await engine.list_projects()

        # Get active indexing tasks
        tasks = await engine.list_indexing_tasks()
        active_tasks_by_project = {}
        for task in tasks:
            if task.project_id and task.status == "running":
                active_tasks_by_project[task.project_id] = {
                    "task_id": task.task_id,
                    "progress": task.progress,
                    "files_processed": task.processed_files,
                    "total_files": task.total_files,
                }

        return {
            "status": "success",
            "total_projects": len(projects),
            "projects": [
                {
                    "id": p.id,
                    "name": p.name,
                    "paths": p.paths,
                    "is_watching": p.is_watching,
                    "last_indexed": p.last_indexed.isoformat()
                    if p.last_indexed
                    else None,
                    "created_at": p.created_at.isoformat(),
                    "updated_at": p.updated_at.isoformat(),
                    "active_indexing": active_tasks_by_project.get(p.id),
                }
                for p in projects
            ],
        }

    except Exception as e:
        if ctx:
            await ctx.error(f"Error listing projects: {e}")
        logger.error(f"Error listing projects: {e}")
        return {"status": "error", "message": str(e)}


# Resources for exposing indexing tasks and projects
@mcp.resource("indexing://tasks")
async def get_all_indexing_tasks() -> Dict[str, Any]:
    """Get all indexing tasks with their current status."""
    try:
        engine = await get_engine()
        if not engine._indexing_queue:
            return {"tasks": []}

        tasks = []

        # Get queue status
        queue_status = await engine._indexing_queue.get_queue_status()

        # Add current task if any
        if queue_status.current_task:
            tasks.append(
                {
                    "task_id": queue_status.current_task,
                    "status": "running",
                    "progress": queue_status.current_task_progress,
                }
            )

        # Add queued tasks
        for task_info in queue_status.queued_tasks:
            tasks.append(
                {
                    "task_id": task_info.task_id,
                    "status": "queued",
                    "queue_position": task_info.queue_position,
                    "paths": task_info.paths,
                    "created_at": task_info.created_at,
                }
            )

        return {"total_tasks": len(tasks), "tasks": tasks}
    except Exception as e:
        logger.error(f"Error getting indexing tasks: {e}")
        return {"error": str(e)}


@mcp.resource("indexing://tasks/{task_id}")
async def get_indexing_task(task_id: str) -> Dict[str, Any]:
    """Get details for a specific indexing task."""
    try:
        engine = await get_engine()
        if not engine._indexing_queue:
            return {"error": "Task not found"}

        # Get queue status
        queue_status = await engine._indexing_queue.get_queue_status()

        # Check if it's the current task
        if queue_status.current_task == task_id:
            # Get the task details from database
            tasks = await engine.list_indexing_tasks()
            for task in tasks:
                if task.task_id == task_id:
                    return {
                        "task_id": task.task_id,
                        "status": "running",
                        "paths": task.paths,
                        "created_at": task.created_at.isoformat(),
                        "progress": queue_status.current_task_progress,
                        "stats": getattr(task, "stats", None),
                    }
            return {"error": "Task not found"}

        # Check queued tasks
        for task_info in queue_status.queued_tasks:
            if task_info.task_id == task_id:
                return {
                    "task_id": task_info.task_id,
                    "status": "queued",
                    "queue_position": task_info.queue_position,
                    "paths": task_info.paths,
                    "created_at": task_info.created_at,
                }

        return {"error": "Task not found"}
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {e}")
        return {"error": str(e)}


@mcp.resource("projects://list")
async def get_all_projects_resource() -> Dict[str, Any]:
    """Get all registered projects as a resource."""
    try:
        engine = await get_engine()
        projects = await engine.list_projects()

        return {
            "total_projects": len(projects),
            "projects": [
                {
                    "id": p.id,
                    "name": p.name,
                    "paths": p.paths,
                    "is_watching": p.is_watching,
                    "last_indexed": p.last_indexed.isoformat()
                    if p.last_indexed
                    else None,
                    "created_at": p.created_at.isoformat(),
                    "updated_at": p.updated_at.isoformat(),
                }
                for p in projects
            ],
        }
    except Exception as e:
        logger.error(f"Error getting projects: {e}")
        return {"error": str(e)}


@mcp.resource("projects://{project_id}")
async def get_project_resource(project_id: str) -> Dict[str, Any]:
    """Get details for a specific project."""
    try:
        engine = await get_engine()
        project = await engine.get_project(project_id)

        if not project:
            return {"error": f"Project not found: {project_id}"}

        return {
            "id": project.id,
            "name": project.name,
            "paths": project.paths,
            "is_watching": project.is_watching,
            "last_indexed": project.last_indexed.isoformat()
            if project.last_indexed
            else None,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting project {project_id}: {e}")
        return {"error": str(e)}


@mcp.resource("projects://{project_id}/files")
async def get_project_files(project_id: str) -> Dict[str, Any]:
    """Get list of files tracked for a specific project."""
    try:
        engine = await get_engine()
        project = await engine.get_project(project_id)

        if not project:
            return {"error": f"Project not found: {project_id}"}

        # Get indexed files for this project
        # This would need to be implemented in the engine
        # For now, return project paths
        return {
            "project_id": project_id,
            "project_name": project.name,
            "tracked_paths": project.paths,
            "is_watching": project.is_watching,
        }
    except Exception as e:
        logger.error(f"Error getting project files for {project_id}: {e}")
        return {"error": str(e)}


# Create ASGI app with both transports
def create_app():
    """Create Starlette app with both SSE and HTTP transports."""
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.responses import JSONResponse
    from contextlib import asynccontextmanager

    async def health_endpoint(request):
        """Health check endpoint."""
        stats = {"status": "healthy", "server": "breeze-mcp", "version": "0.1.0"}

        try:
            engine = await get_engine()
            db_stats = await engine.get_stats()
            stats["database"] = {
                "initialized": db_stats.initialized,
                "total_documents": db_stats.total_documents,
            }
        except Exception as e:
            stats["database"] = {"error": str(e)}

        return JSONResponse(stats)

    # Create ASGI apps for both transports
    sse_app = mcp.http_app(path="/", transport="sse")
    http_app = mcp.http_app(path="/", transport="streamable-http")

    @asynccontextmanager
    async def combined_lifespan(app):
        """Combined lifespan that initializes both MCP apps."""
        # Initialize our engine
        logger.info("Starting Breeze MCP server...")
        await get_engine()

        # Initialize both MCP apps
        async with sse_app.lifespan(app):
            async with http_app.lifespan(app):
                yield

        logger.info("Shutting down Breeze MCP server...")
        await shutdown_engine()

    # Create Starlette app with combined lifespan
    app = Starlette(
        routes=[
            Mount("/sse", app=sse_app),
            Mount("/mcp", app=http_app),
            Route("/health", endpoint=health_endpoint, methods=["GET"]),
        ],
        lifespan=combined_lifespan,
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
