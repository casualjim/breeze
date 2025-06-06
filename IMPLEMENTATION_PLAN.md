# Breeze MCP Server - Implementation Plan for Long-Running Operations

## Problem Statement

The MCP server currently has issues with long-running indexing operations:

1. **Timeout Issue**: The `index_repository` tool runs synchronously within the request-response cycle, causing timeouts for large repositories when accessed via MCP Inspector or other clients
2. **Missing Notifications**: While notification code exists, it's not properly integrated with FastMCP's SSE/streaming capabilities
3. **No Queue Management**: Multiple indexing requests can overwhelm the embedding model (rate limits for APIs, resource constraints for local models)
4. **No Persistence**: Server restarts lose all in-progress indexing work, leaving codebases partially indexed

## Design Decisions Made

### 1. Task Persistence is Required

**Decision**: All indexing tasks MUST be persisted to LanceDB.

**Rationale**:

- Users express clear intent when requesting codebase indexing
- Partial indexing due to server restarts violates user expectations
- This is more like a database migration tool than a stateless API
- The existing codebase already uses LanceDB for persistence (failed_batches_table, projects_table)

**Rejected Alternative**: In-memory only tasks

- Would lose work on restart
- Goes against the tool's purpose

### 2. Simple FIFO Queue (No Priority System)

**Decision**: Implement a flat, first-in-first-out queue without priority levels.

**Rationale**:

- Cannot predict job size accurately upfront
- All indexing jobs are important to their requesters
- Simpler to implement and reason about
- More predictable for users

**Rejected Alternative**: Priority queue based on estimated job size

- Would require guessing at directory sizes
- Small directories might not actually be faster (dense code vs sparse)
- Adds complexity without clear benefit

### 3. Single Concurrent Indexing Task

**Decision**: Limit to one indexing operation at a time.

**Rationale**:

- Embedding models have hard limits:
  - Local models (sentence-transformers): GPU/CPU can only handle one operation efficiently
  - API models (Voyage, Gemini): Rate limits make concurrent operations problematic
- Prevents resource exhaustion
- Makes progress tracking clearer

**Considered Alternative**: Dynamic concurrency based on model

- Too complex for initial implementation
- Can be added later if needed

### 4. Background Processing via asyncio

**Decision**: Use Python's built-in `asyncio.create_task()` for background work.

**Rationale**:

- MCP servers are long-running processes
- No need for external dependencies (Celery, RQ)
- Integrates naturally with FastMCP's async architecture
- Can access the same engine instance

**Rejected Alternatives**:

- External queue (Celery): Overkill, adds complexity
- Threading: Not compatible with async-first MCP design
- Subprocess: Would lose access to engine state

## Implementation Plan

### Phase 1: Add Task Persistence

1. **Create `indexing_tasks` table in LanceDB**

   ```python
   class IndexingTask(BaseModel):
       task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
       paths: List[str]
       force_reindex: bool = False
       status: str = "queued"  # queued, running, completed, failed
       
       # Progress tracking
       progress: float = 0.0
       total_files: int = 0
       processed_files: int = 0
       
       # Timestamps
       created_at: datetime = Field(default_factory=datetime.now)
       started_at: Optional[datetime] = None
       completed_at: Optional[datetime] = None
       
       # Results
       result_stats: Optional[Dict] = None
       error_message: Optional[str] = None
       
       # Queue management
       queue_position: Optional[int] = None
       attempt_count: int = 0
   ```

2. **Add persistence methods to BreezeEngine**
   - `init_indexing_tasks_table()`
   - `save_indexing_task()`
   - `get_indexing_task()`
   - `update_indexing_task()`
   - `list_tasks_by_status()`

### Phase 2: Implement Queue System

1. **Create `IndexingQueue` class**
   - FIFO queue with single worker
   - Automatic task persistence
   - Progress tracking integration
   - Graceful shutdown handling

2. **Modify `index_repository` to return immediately**
   - Create task record
   - Add to queue
   - Return task ID and queue position
   - No more synchronous indexing

3. **Add queue status tool**

   ```python
   @mcp.tool()
   async def get_queue_status() -> Dict[str, Any]:
       """Get current indexing queue status."""
   ```

### Phase 3: Server Startup Recovery

1. **On server startup**:
   - Query all tasks with status in ["queued", "running"]
   - Reset "running" tasks to "queued" (they were interrupted)
   - Re-populate queue in original order
   - Start queue worker

2. **Ensure queue integrity**:
   - Periodic validation that in-memory queue matches DB
   - Automatic recovery from inconsistencies

### Phase 4: Fix SSE Notifications

1. **Research FastMCP's notification system**
   - Confirm proper SSE integration approach
   - Test with MCP Inspector

2. **Wire up existing notification calls**:
   - Task queued
   - Indexing started
   - Progress updates
   - Task completed/failed
   - File change events (from watchers)

3. **Add notification support to tools**:

   ```python
   # Ensure mcp instance supports notifications
   if hasattr(mcp, 'send_notification'):
       await mcp.send_notification(...)
   ```

### Phase 5: Update Model Detection

1. **Support new Gemini naming**:
   - Check for both `models/` and `gemini-` prefixes
   - Update rate limit configurations

2. **Ensure queue respects model limits**:
   - Voyage AI: Handle rate limits properly
   - Gemini: Configure appropriate limits
   - Local models: Prevent resource exhaustion

### Phase 6: Comprehensive Testing

1. **Unit tests**:
   - Queue operations (enqueue, dequeue, persistence)
   - Task state transitions
   - Model detection logic

2. **Integration tests**:
   - Full indexing flow with queue
   - Server restart recovery
   - SSE notification delivery
   - Concurrent request handling

3. **Stress tests**:
   - Queue many large indexing jobs
   - Restart server mid-indexing
   - Verify recovery and completion

## What NOT to Re-Consider

These decisions have been thoroughly discussed and should not be revisited:

1. **Don't add priority queuing** - We established job size is unpredictable
2. **Don't use external queues** - asyncio tasks are sufficient for our needs
3. **Don't allow concurrent indexing** - Model constraints make this problematic
4. **Don't make persistence optional** - User intent must be durable
5. **Don't complicate model detection** - Simple prefix matching is enough

## Success Criteria

1. Large repository indexing doesn't timeout in MCP Inspector
2. Users can see queue position and progress
3. All queued tasks complete even if server restarts
4. SSE notifications work in Claude Desktop and MCP Inspector
5. No rate limit errors or resource exhaustion
6. Clear task status visibility at all times

## CLI Compatibility

The `breeze index` CLI command needs to work seamlessly with the queue system while presenting as synchronous to users:

1. **Simple Synchronous Wrapper**:
   - Add `index_directories_sync()` method to BreezeEngine
   - This method creates a task, queues it, and polls until completion
   - Returns the same IndexStats as before

2. **Implementation**:

   ```python
   async def index_directories_sync(self, directories, force_reindex, progress_callback):
       # Create and queue task
       task = await self.create_indexing_task(paths=directories, force_reindex=force_reindex)
       
       # Poll until complete
       while task.status not in ["completed", "failed"]:
           task = await self.get_indexing_task(task.task_id)
           if progress_callback and task.progress_info:
               await progress_callback(task.progress_info)
           await asyncio.sleep(0.1)
       
       # Return results or raise error
       if task.status == "failed":
           raise Exception(task.error_message)
       return task.result_stats
   ```

3. **Minimal CLI Changes**:
   - Change one line: call `index_directories_sync()` instead of `index_directories()`
   - All existing progress UI, Rich display, and logging continues to work
   - Users see no difference in behavior

This is the simplest approach - the CLI blocks and shows progress exactly as before, while the backend properly queues and manages tasks.

## Next Steps

Begin with Phase 1 (Task Persistence) as it's foundational for everything else. The implementation should follow the patterns already established in the codebase (see failed_batches_table for reference).
