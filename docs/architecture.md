# Architecture Overview

This document describes the technical architecture and design decisions of Breeze.

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐
│   MCP Clients   │────▶│    MCP Server    │
│ (Claude, etc.)  │     │   (FastMCP/SSE)  │
└─────────────────┘     └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Breeze Engine   │
                        │  (Core Logic)    │
                        └────────┬─────────┘
                                 │
                ┌────────────────┴──────────────────┐
                │                                   │
       ┌────────▼─────────┐              ┌─────────▼────────┐
       │  Indexing Queue  │              │  File Watcher    │
       │  (Async Tasks)   │              │  (Watchdog)      │
       └────────┬─────────┘              └─────────┬────────┘
                │                                   │
                └────────────┬──────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │    LanceDB       │
                    │ (Vector Storage) │
                    └──────────────────┘
```

## Core Components

### 1. MCP Server (`breeze/mcp/server.py`)

- Built with FastMCP for async MCP implementation
- Supports both SSE and HTTP transports
- Exposes tools and resources to MCP clients
- Handles request routing and response formatting

**Key Features:**

- Async request handling
- Context-based logging
- Resource endpoints for state monitoring
- Tool parameter validation

### 2. Breeze Engine (`breeze/core/engine.py`)

The core indexing and search logic:

- **Initialization**: Sets up LanceDB connection and embedding models
- **Indexing**: Processes files, generates embeddings, stores in database
- **Search**: Performs semantic similarity search
- **Project Management**: Tracks projects and file changes

**Key Methods:**

- `index_directories()`: Main indexing entry point
- `search()`: Semantic search implementation
- `add_project()` / `remove_project()`: Project lifecycle
- `start_watching()`: File change monitoring

### 3. Indexing Queue (`breeze/core/queue.py`)

Manages asynchronous indexing tasks:

- **FIFO Queue**: Tasks processed in order received
- **Single Worker**: Prevents concurrent write conflicts
- **Progress Tracking**: Real-time progress updates
- **Error Handling**: Retry logic and failure tracking
- **Persistence**: Queue state saved to database

**Queue Lifecycle:**

1. Task created with unique UUID v7 (time-ordered)
2. Task added to queue with position tracking
3. Worker processes task asynchronously
4. Progress updates sent via callbacks
5. Task marked complete/failed in database

### 4. Data Models (`breeze/core/models.py`)

Pydantic v2 models for type safety:

- **BreezeDocument**: Indexed document with metadata
- **SearchResult**: Search response with relevance
- **IndexStats**: Indexing statistics
- **Project**: Project configuration
- **IndexingTask**: Queue task with progress

### 5. Embeddings (`breeze/core/embeddings.py`)

Handles embedding generation with rate limiting:

- **Multi-provider Support**: Voyage AI, Google, local models
- **Rate Limiting**: Token bucket implementation
- **Batch Processing**: Efficient API usage
- **Error Recovery**: Automatic retries with backoff

### 6. Rate Limiter (`breeze/core/rate_limiter.py`)

Token bucket rate limiter for API calls:

- **Dual Limits**: Request and token rate limiting
- **Concurrent Operations**: Tracks in-flight requests
- **Context Manager**: Clean resource management
- **Adaptive Waiting**: Calculates optimal wait times

## Database Schema

### LanceDB Tables

**documents**

```
- id: String (UUID)
- file_path: String
- file_name: String  
- content: String
- chunk_index: Int32
- chunk_count: Int32
- file_hash: String
- file_size: Int64
- last_modified: Timestamp
- indexed_at: Timestamp
- project_id: String (nullable)
- vector: FixedSizeList[Float32] (embedding)
```

**projects**

```
- id: String (UUID)
- name: String
- paths: List[String]
- is_watching: Boolean
- last_indexed: Timestamp (nullable)
- created_at: Timestamp
- updated_at: Timestamp
```

**indexing_tasks**

```
- task_id: String (UUID v7)
- paths: List[String]
- project_id: String (nullable)
- status: String (enum)
- progress: Float
- created_at: Timestamp
- started_at: Timestamp (nullable)
- completed_at: Timestamp (nullable)
- error_message: String (nullable)
- stats: JSON
```

## Data Flow

### Indexing Flow

1. **File Discovery**
   - Recursively scan directories
   - Filter by content detection (not extensions)
   - Check file size limits

2. **Content Processing**
   - Read file content asynchronously
   - Split into chunks with overlap
   - Generate metadata (hash, size, modified time)

3. **Embedding Generation**
   - Batch documents for efficiency
   - Apply rate limiting for API models
   - Handle retries on failure

4. **Storage**
   - Upsert documents to LanceDB
   - Update file hash for deduplication
   - Track indexing statistics

### Search Flow

1. **Query Processing**
   - Receive natural language query
   - Generate query embedding

2. **Vector Search**
   - Perform similarity search in LanceDB
   - Apply relevance threshold filtering
   - Limit results

3. **Result Enhancement**
   - Create code snippets
   - Add file metadata
   - Sort by relevance score

## Concurrency Model

### Async Architecture

- **Event Loop**: Single event loop per process
- **Coroutines**: All I/O operations are async
- **Semaphores**: Control concurrent operations
- **Queues**: Async queues for task management

### Rate Limiting

- **Token Bucket**: Refills at configured rate
- **Request Tracking**: Monitor active operations
- **Backpressure**: Wait when limits exceeded

### File Operations

- **Concurrent Reads**: Multiple files read in parallel
- **Batch Processing**: Documents processed in batches
- **Memory Efficient**: Streaming for large files

## Performance Optimizations

### 1. Incremental Indexing

- File hash comparison to skip unchanged files
- Timestamp tracking for quick change detection
- Project-level indexing state

### 2. Efficient Embeddings

- Batch API calls to reduce overhead
- Local model caching
- Optimal chunk sizes for models

### 3. Database Optimization

- Arrow format for zero-copy reads
- Columnar storage for analytics
- Vector indexing for fast search

### 4. Memory Management

- Async generators for file processing
- Bounded queues to prevent memory overflow
- Explicit garbage collection for large operations

## Security Considerations

### API Key Management

- Environment variable storage
- No keys in logs or error messages
- Secure transmission to providers

### File Access

- Path validation and normalization
- Size limits to prevent DoS
- Binary file detection and skipping

### Error Handling

- Sanitized error messages
- No sensitive path exposure
- Graceful degradation

## Extension Points

### 1. Embedding Providers

Add new providers by:

1. Implementing provider in `embeddings.py`
2. Adding to LanceDB registry
3. Configuring in `BreezeConfig`

### 2. File Type Support

Extend content detection:

1. Update `_should_index_file()` logic
2. Add specialized chunking if needed
3. Configure type-specific settings

### 3. Storage Backends

While LanceDB is the primary backend:

1. Abstract storage interface possible
2. Alternative vector stores could be added
3. Hybrid storage for metadata

## Monitoring and Debugging

### Logging

- Structured logging throughout
- Log levels: DEBUG, INFO, WARNING, ERROR
- Context-based logging via MCP Context parameter

### Health Checks

- `/health` endpoint for server status
- Database connection verification
- Model initialization checks
