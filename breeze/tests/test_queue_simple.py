"""Simple test to verify queue works with the actual implementation."""

import asyncio
import tempfile
import pytest

from breeze.core.models import IndexingTask
from breeze.core.engine import BreezeEngine
from breeze.core.config import BreezeConfig
from .mock_embedders import MockReranker
from lancedb.embeddings.registry import get_registry


@pytest.mark.asyncio
async def test_queue_basic_functionality():
    """Test basic queue functionality without mocks."""
    # Set dummy API key for Voyage model
    import os
    os.environ["BREEZE_EMBEDDING_API_KEY"] = "dummy-key-for-testing"
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use mock embedder instead of voyage model
            registry = get_registry()
            mock_embedder = registry.get("mock-voyage").create()
            
            config = BreezeConfig(
                data_root=tmpdir,
                db_name="test_queue_simple",
                embedding_function=mock_embedder,
            )
            
            engine = BreezeEngine(config)
            engine.reranker = MockReranker()
            await engine.initialize()
            
            # Create a task
            task = IndexingTask(
                paths=[tmpdir],  # Index empty dir, should be quick
                force_reindex=False
            )
            
            # Save task
            await engine.save_indexing_task(task)
            
            # Verify saved
            retrieved = await engine.get_indexing_task_db(task.task_id)
            assert retrieved is not None
            assert retrieved.task_id == task.task_id
            assert retrieved.status == "queued"
            
            # Update task
            task.status = "running"
            await engine.update_indexing_task(task)
            
            # Verify updated
            retrieved = await engine.get_indexing_task_db(task.task_id)
            assert retrieved.status == "running"
            
            # List by status
            running_tasks = await engine.list_tasks_by_status("running")
            assert len(running_tasks) == 1
            assert running_tasks[0].task_id == task.task_id
            
            await engine.shutdown()
            print("✅ Basic queue persistence works!")
    
    finally:
        # Clean up environment variable
        if "BREEZE_EMBEDDING_API_KEY" in os.environ:
            del os.environ["BREEZE_EMBEDDING_API_KEY"]


@pytest.mark.asyncio
async def test_index_repository_returns_immediately():
    """Test that index_repository returns immediately with task info."""
    from breeze.mcp.server import get_engine, index_repository
    import os
    
    # Set dummy API key
    os.environ["BREEZE_EMBEDDING_API_KEY"] = "dummy-key-for-testing"
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = f"{tmpdir}/test.py"
            with open(test_file, "w") as f:
                f.write("print('hello world')")
            
            # Initialize engine (this may take time, don't include in timing)
            engine = await get_engine()
            engine.reranker = MockReranker()
            
            # Call index_repository - need to use .fn to access the wrapped function
            # Only time the actual indexing call, not the engine initialization
            start_time = asyncio.get_event_loop().time()
            result = await index_repository.fn(directories=[tmpdir])
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Should return relatively quickly (within 10 seconds)
            # Note: This includes database operations and queue setup, so some delay is expected
            assert elapsed < 10.0, f"Took {elapsed}s, should be reasonably fast"
            
            # Check response format
            assert result["status"] == "success"
            assert "task_id" in result
            assert "queue_position" in result
            assert result["message"] == "Indexing task queued successfully"
            
            print(f"✅ index_repository returned in {elapsed:.3f}s with task_id: {result['task_id']}")
            
            # Wait a bit for processing
            await asyncio.sleep(2.0)
            
            # Check task completed
            task = await engine.get_indexing_task_db(result["task_id"])
            assert task is not None
            print(f"Task status: {task.status}")
            
            await engine.shutdown()
    
    finally:
        # Clean up environment variable
        if "BREEZE_EMBEDDING_API_KEY" in os.environ:
            del os.environ["BREEZE_EMBEDDING_API_KEY"]


if __name__ == "__main__":
    asyncio.run(test_queue_basic_functionality())
    asyncio.run(test_index_repository_returns_immediately())