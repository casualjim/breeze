# Concurrency Guide

This guide explains important limitations and best practices for using Breeze with multiple processes.

## LanceDB Concurrency Model

Breeze uses LanceDB as its vector database, which has specific concurrency characteristics:

### ✅ Supported: Concurrent Reads

- Multiple processes can read from the same database simultaneously
- No locking required for read operations
- MVCC (Multi-Version Concurrency Control) ensures consistent snapshots

### ⚠️ Limited: Concurrent Writes

- LanceDB does **NOT** support concurrent writes from multiple processes without external coordination
- Attempting concurrent writes will result in commit conflicts and errors
- Only one process should write to the database at a time

## Practical Implications

### Running the MCP Server and CLI Commands

**Important**: When the MCP server is running, you should stop it before running indexing commands from the CLI.

#### Option 1: Stop Server Before Indexing (Recommended)

```bash
# Stop the MCP server (if running as a service)
launchctl stop com.breeze-mcp.server

# Run indexing
uv run python -m breeze index /path/to/repo

# Restart the server
launchctl start com.breeze-mcp.server
```

#### Option 2: Use MCP Tools for Indexing

Instead of CLI commands, use the MCP server's tools while it's running:

```python
# Through Claude or another MCP client
index_repository(directories=["/path/to/repo"])
```

The MCP server handles indexing through an internal queue, ensuring no concurrent write conflicts.

### Multiple Breeze Instances

If you need multiple Breeze instances:

1. **Different Databases**: Each instance should use a different `BREEZE_DATA_ROOT`:

   ```bash
   # Instance 1
   BREEZE_DATA_ROOT=/data/breeze1 uv run python -m breeze serve --port 9483
   
   # Instance 2
   BREEZE_DATA_ROOT=/data/breeze2 uv run python -m breeze serve --port 9484
   ```

2. **Read-Only Instances**: Configure additional instances for search only:

   ```python
   # In custom code - read-only connection
   db = await lancedb.connect_async(db_path, read_only=True)
   ```

## Best Practices

### 1. Single Writer Pattern

Designate one process as the writer:

- MCP server handles all writes through its queue
- CLI commands should only be used when server is stopped
- Or implement a dedicated indexing service

### 2. Queue-Based Architecture

The MCP server uses an internal queue for indexing:

- All indexing requests are serialized
- No concurrent write conflicts
- Progress tracking and error handling

### 3. Deployment Strategies

**Development**:

- Stop server before CLI indexing
- Use MCP tools when server is running

**Production**:

- Run MCP server continuously
- Use MCP tools exclusively for indexing
- Monitor queue status with `get_index_stats`

**CI/CD**:

- Index during deployment (server stopped)
- Start server after indexing completes

## Error Messages

If you see these errors, you have a concurrency issue:

```text
CommitConflict: Another writer has already written to this table
```

**Solution**: Ensure only one process is writing at a time.

## Future Improvements

We're exploring options for better multi-process support:

1. **External Lock Manager**: Redis or DynamoDB-based locking
2. **Write Proxy Service**: Dedicated service to coordinate writes
3. **Built-in Coordination**: Automatic write serialization

## FAQ

**Q: Can I run multiple search operations simultaneously?**
A: Yes, searches are read operations and can run concurrently.

**Q: Can I index while searching?**
A: Yes, but only through the MCP server's queue system, not via CLI.

**Q: What happens if I accidentally run concurrent writes?**
A: One write will succeed, others will fail with CommitConflict errors. No data corruption occurs.

**Q: Can I use Breeze in a distributed system?**
A: Yes, but you need external coordination for writes (Redis lock, DynamoDB, etc.)

**Q: Is this a Breeze limitation or LanceDB limitation?**
A: This is a LanceDB design choice. LanceDB prioritizes simplicity and performance over distributed write coordination.
