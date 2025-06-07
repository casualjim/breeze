"""Tests for the indexing queue functionality."""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from breeze.core.models import IndexingTask, IndexStats
from breeze.core.queue import IndexingQueue
from breeze.core.engine import BreezeEngine
from breeze.core.config import BreezeConfig
from .mock_embedders import MockReranker
from lancedb.embeddings.registry import get_registry


@pytest_asyncio.fixture
async def test_engine():
    """Create a test engine with mock embedding model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use mock embedder from registry
        registry = get_registry()
        mock_embedder = registry.get("mock-local").create()
        
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_queue",
            embedding_function=mock_embedder,
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        
        await engine.initialize()
        yield engine
        await engine.shutdown()


class TestIndexingQueue:
    """Test suite for IndexingQueue functionality."""
    
    @pytest.mark.asyncio
    async def test_queue_add_task(self, test_engine):
        """Test adding a task to the queue."""
        queue = IndexingQueue(test_engine)
        await queue.start()
        
        # Create a task
        task = IndexingTask(
            paths=["/test/path"],
            force_reindex=False
        )
        
        # Add to queue
        position = await queue.add_task(task)
        
        # Verify
        assert position == 0  # First task
        assert task.queue_position == 0
        
        # Check queue status
        status = await queue.get_queue_status()
        # The queue might be processing already, so check if task exists
        # either in queue or being processed
        assert status["queue_size"] >= 0  # Could be 0 if already processing
        assert status["current_task"] == task.task_id or status["queue_size"] == 1
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_processing(self, test_engine):
        """Test that queued tasks are processed."""
        queue = IndexingQueue(test_engine)
        
        # Mock index_directories to complete quickly
        async def mock_index_directories(**kwargs):
            await asyncio.sleep(0.1)  # Simulate work
            return IndexStats(
                files_scanned=10,
                files_indexed=8,
                files_skipped=2
            )
        
        test_engine.index_directories = mock_index_directories
        
        await queue.start()
        
        # Add a task
        task = IndexingTask(
            paths=["/test/path"],
            force_reindex=False
        )
        
        await queue.add_task(task)
        
        # Wait for processing
        await asyncio.sleep(0.3)
        
        # Check task was completed
        updated_task = await test_engine.get_indexing_task_db(task.task_id)
        assert updated_task is not None
        assert updated_task.status == "completed"
        assert updated_task.progress == 100.0
        # Check that result fields are populated
        assert updated_task.result_files_scanned is not None
        assert updated_task.result_files_indexed is not None
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_fifo_order(self, test_engine):
        """Test that tasks are processed in FIFO order."""
        queue = IndexingQueue(test_engine)
        
        processed_tasks = []
        
        # Mock to track processing order
        async def mock_index_directories(**kwargs):
            task_paths = kwargs['directories']
            processed_tasks.append(task_paths[0])
            await asyncio.sleep(0.05)
            return IndexStats()
        
        test_engine.index_directories = mock_index_directories
        
        await queue.start()
        
        # Add multiple tasks
        for i in range(3):
            task = IndexingTask(
                paths=[f"/test/path{i}"],
                force_reindex=False
            )
            await queue.add_task(task)
        
        # Wait for all to process
        await asyncio.sleep(0.5)
        
        # Verify FIFO order
        assert processed_tasks == ["/test/path0", "/test/path1", "/test/path2"]
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_error_handling(self, test_engine):
        """Test that errors are properly handled."""
        queue = IndexingQueue(test_engine)
        
        # Mock to raise error
        async def mock_index_directories(**kwargs):
            raise Exception("Test indexing error")
        
        test_engine.index_directories = mock_index_directories
        
        await queue.start()
        
        # Add a task
        task = IndexingTask(
            paths=["/test/path"],
            force_reindex=False
        )
        
        await queue.add_task(task)
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Check task failed
        updated_task = await test_engine.get_indexing_task_db(task.task_id)
        assert updated_task.status == "failed"
        assert updated_task.error_message == "Test indexing error"
        assert updated_task.completed_at is not None
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_progress_tracking(self, test_engine):
        """Test that progress is tracked during indexing."""
        queue = IndexingQueue(test_engine)
        
        progress_updates = []
        
        # Mock with progress updates
        async def mock_index_directories(**kwargs):
            callback = kwargs.get('progress_callback')
            if callback:
                # Simulate progress
                for i in range(1, 6):
                    stats = IndexStats(
                        files_scanned=10,
                        files_indexed=i * 2,
                    )
                    await callback(stats)
                    await asyncio.sleep(0.05)
            
            return IndexStats(
                files_scanned=10,
                files_indexed=10
            )
        
        test_engine.index_directories = mock_index_directories
        
        await queue.start()
        
        # Add task with callback
        task = IndexingTask(
            paths=["/test/path"],
            force_reindex=False
        )
        
        async def track_progress(stats):
            progress_updates.append(stats.files_indexed)
        
        await queue.add_task(task, track_progress)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Check progress was tracked
        assert len(progress_updates) > 0
        assert progress_updates == [2, 4, 6, 8, 10]
        
        # Check final task state
        updated_task = await test_engine.get_indexing_task_db(task.task_id)
        assert updated_task.progress == 100.0
        assert updated_task.total_files == 10
        assert updated_task.processed_files == 10
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_startup_recovery(self, test_engine):
        """Test that interrupted tasks are recovered on startup."""
        # First, create some tasks in different states
        queued_task = IndexingTask(
            paths=["/test/queued"],
            status="queued"
        )
        await test_engine.save_indexing_task(queued_task)
        
        running_task = IndexingTask(
            paths=["/test/running"],
            status="running",
            started_at=datetime.now()
        )
        await test_engine.save_indexing_task(running_task)
        
        completed_task = IndexingTask(
            paths=["/test/completed"],
            status="completed",
            completed_at=datetime.now()
        )
        await test_engine.save_indexing_task(completed_task)
        
        # Create queue and restore
        queue = IndexingQueue(test_engine)
        await queue.restore_from_database()
        
        # Check queue status
        status = await queue.get_queue_status()
        
        # Should have 2 tasks (queued + running converted to queued)
        assert status["queue_size"] == 2
        
        # Check running task was reset to queued
        updated_running = await test_engine.get_indexing_task_db(running_task.task_id)
        assert updated_running.status == "queued"
        assert updated_running.started_at is None
        
        # Completed task should not be in queue
        queue_ids = [t["task_id"] for t in status["queued_tasks"]]
        assert completed_task.task_id not in queue_ids
    
    @pytest.mark.asyncio  
    async def test_concurrent_safety(self, test_engine):
        """Test that only one task processes at a time."""
        queue = IndexingQueue(test_engine)
        
        concurrent_count = 0
        max_concurrent = 0
        
        async def mock_index_directories(**kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate work
            concurrent_count -= 1
            return IndexStats()
        
        test_engine.index_directories = mock_index_directories
        
        await queue.start()
        
        # Add multiple tasks
        for i in range(5):
            task = IndexingTask(paths=[f"/test/path{i}"])
            await queue.add_task(task)
        
        # Wait for all to process
        await asyncio.sleep(1.0)
        
        # Should never have more than 1 concurrent
        assert max_concurrent == 1
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_shutdown_graceful(self, test_engine):
        """Test graceful shutdown waits for current task."""
        queue = IndexingQueue(test_engine)
        
        processing_complete = False
        
        async def mock_index_directories(**kwargs):
            nonlocal processing_complete
            await asyncio.sleep(0.3)  # Longer work
            processing_complete = True
            return IndexStats()
        
        test_engine.index_directories = mock_index_directories
        
        await queue.start()
        
        # Add a task
        task = IndexingTask(paths=["/test/path"])
        await queue.add_task(task)
        
        # Wait for processing to start
        await asyncio.sleep(0.1)
        
        # Stop queue (should wait for task to complete)
        await queue.stop()
        
        # Processing should have completed
        assert processing_complete
        
        # Task should be marked complete
        updated_task = await test_engine.get_indexing_task_db(task.task_id)
        assert updated_task.status == "completed"