"""Unit tests for IndexingQueue operations and task state transitions."""

import asyncio
import tempfile
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from breeze.core.models import IndexingTask, IndexStats
from breeze.core.queue import IndexingQueue
from breeze.core.engine import BreezeEngine
from breeze.core.config import BreezeConfig
from .mock_embedders import MockReranker
from lancedb.embeddings.registry import get_registry


@pytest_asyncio.fixture
async def mock_engine():
    """Create a mock engine with necessary methods."""
    engine = Mock(spec=BreezeEngine)
    engine.save_indexing_task = AsyncMock()
    engine.update_indexing_task = AsyncMock()
    engine.get_indexing_task_db = AsyncMock()
    engine.list_tasks_by_status = AsyncMock(return_value=[])
    engine.index_directories = AsyncMock(return_value=IndexStats(
        files_scanned=10,
        files_indexed=8,
        files_updated=1,
        files_skipped=1,
        errors=0,
        total_tokens_processed=1000
    ))
    return engine


@pytest_asyncio.fixture
async def queue(mock_engine):
    """Create an IndexingQueue with mock engine."""
    queue = IndexingQueue(mock_engine)
    await queue.start()
    yield queue
    await queue.stop()


@pytest.mark.asyncio
async def test_queue_initialization(mock_engine):
    """Test queue initializes correctly."""
    queue = IndexingQueue(mock_engine)

    assert queue.engine is mock_engine
    assert queue._current_task is None
    assert queue._worker_task is None
    assert not queue._shutdown_event.is_set()
    assert len(queue._progress_callbacks) == 0


@pytest.mark.asyncio
async def test_queue_start_stop(mock_engine):
    """Test starting and stopping the queue."""
    queue = IndexingQueue(mock_engine)

    # Start queue
    await queue.start()
    assert queue._worker_task is not None
    assert not queue._worker_task.done()

    # Stop queue
    await queue.stop()
    assert queue._shutdown_event.is_set()
    assert queue._worker_task.done()


@pytest.mark.asyncio
async def test_add_task(queue, mock_engine):
    """Test adding a task to the queue."""
    task = IndexingTask(
        paths=["/test/path"],
        force_reindex=False
    )

    # Add task with progress callback
    callback = AsyncMock()
    position = await queue.add_task(task, callback)

    # Verify task was saved
    mock_engine.save_indexing_task.assert_called_once_with(task)

    # Verify task was updated with queue position
    assert task.queue_position == 0
    mock_engine.update_indexing_task.assert_called_once_with(task)

    # Verify callback was stored
    assert queue._progress_callbacks[task.task_id] == callback

    # Verify position returned
    assert position == 0


@pytest.mark.asyncio
async def test_add_multiple_tasks(queue, mock_engine):
    """Test adding multiple tasks maintains FIFO order."""
    tasks = []
    for i in range(3):
        task = IndexingTask(
            paths=[f"/test/path{i}"],
            force_reindex=False
        )
        position = await queue.add_task(task)
        assert position == i
        tasks.append(task)

    # Verify all tasks have correct queue positions
    for i, task in enumerate(tasks):
        assert task.queue_position == i


@pytest.mark.asyncio
async def test_get_queue_status(queue, mock_engine):
    """Test getting queue status."""
    # Mock some queued tasks
    queued_tasks = [
        IndexingTask(
            task_id="task1",
            paths=["/path1"],
            created_at=datetime.now() - timedelta(minutes=5)
        ),
        IndexingTask(
            task_id="task2",
            paths=["/path2"],
            created_at=datetime.now() - timedelta(minutes=3)
        )
    ]
    mock_engine.list_tasks_by_status.return_value = queued_tasks

    # Get status
    status = await queue.get_queue_status()

    assert status["queue_size"] == 0  # No tasks actually in memory queue
    assert status["current_task"] is None
    assert status["current_task_progress"] is None
    assert len(status["queued_tasks"]) == 2

    # Verify tasks are ordered by task_id (chronological for UUID v7)
    assert status["queued_tasks"][0]["task_id"] == "task1"
    assert status["queued_tasks"][1]["task_id"] == "task2"


@pytest.mark.asyncio
async def test_process_task_success(mock_engine):
    """Test successful task processing."""
    queue = IndexingQueue(mock_engine)

    task = IndexingTask(
        paths=["/test/path"],
        force_reindex=False
    )

    # Add progress callback
    progress_callback = AsyncMock()
    queue._progress_callbacks[task.task_id] = progress_callback

    # Process task
    await queue._process_task(task)

    # Verify task status updates
    assert task.status == "completed"
    assert task.started_at is not None
    assert task.completed_at is not None
    assert task.attempt_count == 1
    assert task.progress == 100.0
    assert task.result_files_scanned == 10
    assert task.result_files_indexed == 8
    assert task.result_files_updated == 1
    assert task.result_files_skipped == 1
    assert task.result_errors == 0
    assert task.result_total_tokens_processed == 1000

    # Verify engine methods called
    assert mock_engine.update_indexing_task.call_count >= 2  # At start and end
    mock_engine.index_directories.assert_called_once()

    # Verify callback was cleaned up
    assert task.task_id not in queue._progress_callbacks


@pytest.mark.asyncio
async def test_process_task_failure(mock_engine):
    """Test task processing with failure."""
    queue = IndexingQueue(mock_engine)

    task = IndexingTask(
        paths=["/test/path"],
        force_reindex=False
    )

    # Make index_directories fail
    mock_engine.index_directories.side_effect = Exception("Test error")

    # Process task
    await queue._process_task(task)

    # Verify task marked as failed
    assert task.status == "failed"
    assert task.started_at is not None
    assert task.completed_at is not None
    assert task.error_message == "Test error"
    assert task.attempt_count == 1

    # Verify engine update called
    assert mock_engine.update_indexing_task.call_count >= 2


@pytest.mark.asyncio
async def test_progress_callback_integration(mock_engine):
    """Test progress callback is called during indexing."""
    queue = IndexingQueue(mock_engine)

    task = IndexingTask(
        paths=["/test/path"],
        force_reindex=False
    )

    # Track progress updates
    progress_updates = []
    async def track_progress(stats):
        progress_updates.append(stats)

    queue._progress_callbacks[task.task_id] = track_progress

    # Mock index_directories to call progress callback
    async def mock_index_with_progress(*args, **kwargs):
        callback = kwargs.get('progress_callback')
        if callback:
            # Simulate progress updates
            await callback(IndexStats(files_scanned=5, files_indexed=2))
            await callback(IndexStats(files_scanned=10, files_indexed=8))
        return IndexStats(
            files_scanned=10,
            files_indexed=8,
            files_updated=1,
            files_skipped=1,
            errors=0
        )

    mock_engine.index_directories.side_effect = mock_index_with_progress

    # Process task
    await queue._process_task(task)

    # Verify progress updates
    assert len(progress_updates) == 2
    assert progress_updates[0].files_scanned == 5
    assert progress_updates[1].files_scanned == 10

    # Verify task progress was updated
    assert task.progress == 100.0



@pytest.mark.asyncio
async def test_update_queue_positions(queue, mock_engine):
    """Test updating queue positions after task completion."""
    # Mock queued tasks
    tasks = [
        IndexingTask(task_id=f"task{i}", paths=[f"/path{i}"], queue_position=i+1)
        for i in range(3)
    ]
    mock_engine.list_tasks_by_status.return_value = tasks

    # Update positions
    await queue._update_queue_positions()

    # Verify positions updated to 0, 1, 2
    for i, task in enumerate(tasks):
        assert task.queue_position == i
        mock_engine.update_indexing_task.assert_any_call(task)


@pytest.mark.asyncio
async def test_worker_processes_tasks_sequentially(mock_engine):
    """Test worker processes tasks one at a time in FIFO order."""
    queue = IndexingQueue(mock_engine)

    # Track processing order
    processed_tasks = []

    async def track_processing(*args, **kwargs):
        # Extract task path from the call
        dirs = args[0] if args else kwargs.get('directories', [])
        processed_tasks.append(dirs[0] if dirs else None)
        return IndexStats(files_scanned=1, files_indexed=1)

    mock_engine.index_directories.side_effect = track_processing

    # Start queue
    await queue.start()

    # Add multiple tasks
    tasks = []
    for i in range(3):
        task = IndexingTask(paths=[f"/path{i}"])
        await queue.add_task(task)
        tasks.append(task)

    # Wait for processing
    await asyncio.sleep(0.5)

    # Stop queue
    await queue.stop()

    # Verify tasks processed in order
    assert processed_tasks == ["/path0", "/path1", "/path2"]


@pytest.mark.asyncio
async def test_queue_shutdown_waits_for_current_task():
    """Test queue shutdown waits for current task to complete."""
    # Create real engine for this test with mock embedder
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = get_registry()
        mock_embedder = registry.get("mock-local").create()

        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_queue_shutdown",
            embedding_function=mock_embedder,
        )

        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()

        queue = engine._indexing_queue

        # Ensure queue is started
        if not queue._worker_task or queue._worker_task.done():
            await queue.start()

        # Create a slow task
        task = IndexingTask(paths=[tmpdir])

        # Mock index_directories to be slow
        original_index = engine.index_directories
        async def slow_index(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate slow indexing
            return await original_index(*args, **kwargs)

        engine.index_directories = slow_index

        # Add task and let it start
        await queue.add_task(task)

        # Wait for worker to pick up task with timeout
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < 2.0:
            if queue._current_task is not None:
                break
            await asyncio.sleep(0.1)

        # Verify task is running
        assert queue._current_task is not None

        # Stop queue (should wait for task)
        stop_task = asyncio.create_task(queue.stop())
        await asyncio.sleep(0.2)  # Partial wait

        # Task should still be running
        assert queue._current_task is not None
        assert not stop_task.done()

        # Wait for stop to complete
        await stop_task

        # Now task should be done
        assert queue._current_task is None

        await engine.shutdown()


@pytest.mark.asyncio
async def test_concurrent_task_additions(queue, mock_engine):
    """Test adding tasks concurrently maintains queue integrity."""
    # Add many tasks concurrently
    async def add_task_batch(start_idx):
        tasks = []
        for i in range(10):
            task = IndexingTask(paths=[f"/path{start_idx + i}"])
            await queue.add_task(task)
            tasks.append(task)
        return tasks

    # Run multiple batches concurrently
    results = await asyncio.gather(
        add_task_batch(0),
        add_task_batch(10),
        add_task_batch(20)
    )

    # Verify all tasks added
    all_tasks = [task for batch in results for task in batch]
    assert len(all_tasks) == 30

    # Verify queue positions are unique and sequential
    positions = [task.queue_position for task in all_tasks]
    assert len(set(positions)) == 30  # All unique
    assert min(positions) == 0
    assert max(positions) == 29


if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
