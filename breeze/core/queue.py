"""Indexing queue management for Breeze."""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Dict, Callable
from datetime import datetime

from breeze.core.models import IndexingTask, IndexStats, QueueStatus, QueuedTaskInfo

if TYPE_CHECKING:
    from breeze.core.engine import BreezeEngine

logger = logging.getLogger(__name__)


class IndexingQueue:
    """FIFO queue for managing indexing tasks with single worker execution."""
    
    def __init__(self, engine: "BreezeEngine"):
        self.engine = engine
        self._queue: asyncio.Queue[IndexingTask] = asyncio.Queue()
        self._current_task: Optional[IndexingTask] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._progress_callbacks: Dict[str, Callable] = {}
        
    async def start(self):
        """Start the queue worker."""
        if self._worker_task is None or self._worker_task.done():
            # Ensure shutdown event is clear
            self._shutdown_event.clear()
            self._worker_task = asyncio.create_task(self._worker())
            logger.info(f"IndexingQueue worker started. Queue size: {self._queue.qsize()}")
        else:
            logger.info(f"Worker already running. Queue size: {self._queue.qsize()}")
    
    async def stop(self):
        """Stop the queue worker gracefully."""
        logger.info("Stopping IndexingQueue worker...")
        self._shutdown_event.set()
        
        # Wait for current task to complete
        if self._worker_task:
            try:
                # Use shorter timeout in test environment
                import os
                timeout = 5 if os.environ.get("PYTEST_CURRENT_TEST") else 300
                await asyncio.wait_for(self._worker_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Worker task timeout during shutdown")
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
        
        logger.info("IndexingQueue worker stopped")
    
    async def add_task(self, task: IndexingTask, progress_callback: Optional[Callable] = None) -> int:
        """Add a task to the queue and return its queue position."""
        # Save task to database
        await self.engine.save_indexing_task(task)
        
        # Update queue position
        task.queue_position = self._queue.qsize()
        await self.engine.update_indexing_task(task)
        
        # Store progress callback if provided
        if progress_callback:
            self._progress_callbacks[task.task_id] = progress_callback
        
        # Add to queue
        await self._queue.put(task)
        logger.info(f"Task {task.task_id} added to queue at position {task.queue_position}")
        
        return task.queue_position
    
    async def get_queue_status(self) -> QueueStatus:
        """Get current queue status."""
        queued_tasks = await self.engine.list_tasks_by_status("queued")
        
        # UUID v7 is already time-ordered, so sorting by task_id gives us chronological order
        queued_tasks.sort(key=lambda t: t.task_id)
        
        return QueueStatus(
            queue_size=self._queue.qsize(),
            current_task=self._current_task.task_id if self._current_task else None,
            current_task_progress=self._current_task.progress if self._current_task else None,
            queued_tasks=[
                QueuedTaskInfo(
                    task_id=task.task_id,
                    paths=task.paths,
                    queue_position=idx,
                    created_at=task.created_at.isoformat()
                )
                for idx, task in enumerate(queued_tasks)
            ]
        )
    
    async def _worker(self):
        """Background worker that processes tasks from the queue."""
        logger.info(f"IndexingQueue worker starting. Initial queue size: {self._queue.qsize()}")
        
        task_count = 0
        while not self._shutdown_event.is_set():
            try:
                # Wait for task with timeout to check shutdown periodically
                task = await asyncio.wait_for(
                    self._queue.get(), 
                    timeout=1.0
                )
                
                task_count += 1
                logger.info(f"Worker processing task #{task_count}: {task.task_id}")
                await self._process_task(task)
                logger.info(f"Worker completed task #{task_count}: {task.task_id}")
                
            except asyncio.TimeoutError:
                # No task available, continue to check shutdown
                if task_count == 0 and self._queue.qsize() > 0:
                    logger.debug(f"Worker timeout but queue has {self._queue.qsize()} tasks")
                continue
            except Exception as e:
                if "Event loop is closed" in str(e) or isinstance(e, RuntimeError):
                    # Event loop closing, exit gracefully
                    logger.debug("Event loop closing, worker exiting")
                    break
                logger.error(f"Worker error: {e}", exc_info=True)
                try:
                    await asyncio.sleep(1)  # Brief pause before continuing
                except asyncio.CancelledError:
                    # Can't sleep, probably shutting down
                    break
        
        logger.info(f"IndexingQueue worker exiting. Processed {task_count} tasks")
    
    async def _process_task(self, task: IndexingTask):
        """Process a single indexing task."""
        self._current_task = task
        logger.info(f"Processing task {task.task_id}")
        
        try:
            # Update task status to running
            task.status = "running"
            task.started_at = datetime.now()
            task.attempt_count += 1
            await self.engine.update_indexing_task(task)
            
            # Get progress callback
            progress_callback = self._progress_callbacks.get(task.task_id)
            
            # Create a wrapper callback that updates the task
            async def update_progress(stats: IndexStats):
                if stats.files_scanned > 0:
                    task.progress = (stats.files_indexed / stats.files_scanned) * 100
                    task.processed_files = stats.files_indexed
                    task.total_files = stats.files_scanned
                    await self.engine.update_indexing_task(task)
                
                # Call original callback if provided
                if progress_callback:
                    await progress_callback(stats)
            
            # Run the actual indexing
            stats = await self.engine.index_directories(
                directories=task.paths,
                force_reindex=task.force_reindex,
                progress_callback=update_progress
            )
            
            # Update task as completed
            task.status = "completed"
            task.completed_at = datetime.now()
            task.progress = 100.0
            # Store IndexStats fields directly
            task.result_files_scanned = stats.files_scanned
            task.result_files_indexed = stats.files_indexed
            task.result_files_updated = stats.files_updated
            task.result_files_skipped = stats.files_skipped
            task.result_errors = stats.errors
            task.result_total_tokens_processed = stats.total_tokens_processed
            await self.engine.update_indexing_task(task)
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Update task as failed
            task.status = "failed"
            task.completed_at = datetime.now()
            task.error_message = str(e)
            await self.engine.update_indexing_task(task)
            
            logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)
        
        finally:
            # Clean up
            self._current_task = None
            if task.task_id in self._progress_callbacks:
                del self._progress_callbacks[task.task_id]
            
            # Update queue positions for remaining tasks
            await self._update_queue_positions()
    
    async def _update_queue_positions(self):
        """Update queue positions for all queued tasks."""
        try:
            queued_tasks = await self.engine.list_tasks_by_status("queued")
            
            # UUID v7 is time-ordered, so sort by task_id maintains FIFO order
            queued_tasks.sort(key=lambda t: t.task_id)
            
            # Update positions
            for idx, task in enumerate(queued_tasks):
                if task.queue_position != idx:
                    task.queue_position = idx
                    await self.engine.update_indexing_task(task)
        
        except Exception as e:
            logger.error(f"Error updating queue positions: {e}")
