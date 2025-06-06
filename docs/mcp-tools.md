# MCP Tools Reference

This document describes all tools exposed by the Breeze MCP server.

## Tools

### index_repository

Queue an asynchronous indexing task for specified directories.

**Parameters:**
- `directories` (List[str], required): List of absolute paths to directories to index
- `force_reindex` (bool, optional): If true, re-index all files even if already indexed (default: false)

**Returns:**
```json
{
  "status": "success",
  "message": "Indexing task queued successfully", 
  "task_id": "01234567-89ab-cdef-0123-456789abcdef",
  "queue_position": 0,
  "indexed_directories": ["/path/to/repo"]
}
```

**Notes:**
- Indexing runs asynchronously in the background
- Use `get_index_stats` to monitor progress
- Large repositories may take several minutes

### search_code

Search for code snippets semantically similar to a query.

**Parameters:**
- `query` (str, required): Natural language description of what you're looking for
- `limit` (int, optional): Maximum number of results (default: 10, max: 100)
- `min_relevance` (float, optional): Minimum relevance score 0.0-1.0 (default: 0.0)

**Returns:**
```json
{
  "status": "success",
  "query": "authentication logic",
  "total_results": 5,
  "results": [
    {
      "file_path": "/repo/src/auth.py",
      "file_name": "auth.py",
      "snippet": "def authenticate_user(username, password):\n    ...",
      "relevance_score": 0.892,
      "file_size": 2048,
      "last_modified": "2024-01-15T10:30:00Z"
    }
  ]
}
```

**Search Tips:**
- Use descriptive queries: "user authentication" instead of "auth"
- Describe functionality: "function that validates email addresses"
- Include context: "React component for file upload"

### get_index_stats

Get comprehensive statistics about the index and queue.

**Parameters:** None

**Returns:**
```json
{
  "status": "success",
  "total_documents": 1523,
  "total_chunks": 4892,
  "embedding_model": "voyage-code-3",
  "embedding_dimensions": 1024,
  "database_path": "/home/user/.breeze/data/code_index",
  "initialized": true,
  "indexing_queue": {
    "queue_size": 2,
    "current_task": "task-123",
    "current_task_progress": {
      "files_scanned": 150,
      "files_indexed": 120,
      "files_updated": 30,
      "files_skipped": 0,
      "errors": 0,
      "progress_percentage": 80.0
    }
  },
  "failed_batches": {
    "count": 0,
    "total_files": 0
  }
}
```

### list_directory

List contents of a directory to identify what to index.

**Parameters:**
- `directory_path` (str, required): Path to the directory to list

**Returns:**
```json
{
  "status": "success",
  "path": "/absolute/path/to/directory",
  "total_items": 25,
  "contents": [
    {
      "name": "src",
      "type": "directory", 
      "items": 15
    },
    {
      "name": "README.md",
      "type": "file",
      "size": 2048,
      "extension": ".md"
    }
  ]
}
```

### register_project

Register a project for automatic file watching and re-indexing.

**Parameters:**
- `name` (str, required): Human-readable project name
- `paths` (List[str], required): List of directory paths to watch
- `auto_index` (bool, optional): Perform initial indexing (default: true)

**Returns:**
```json
{
  "status": "success",
  "project_id": "proj_01234567",
  "name": "My Project",
  "paths": ["/path/to/project"],
  "watching": true,
  "message": "Project 'My Project' registered and watching started",
  "indexing_task_id": "task_789"
}
```

**File Watching:**
- Automatically detects new, modified, and deleted files
- Re-indexes changed files immediately
- Uses content detection to identify code files

### unregister_project

Stop watching and unregister a project.

**Parameters:**
- `project_id` (str, required): ID of the project to unregister

**Returns:**
```json
{
  "status": "success",
  "message": "Project 'My Project' unregistered and watching stopped",
  "project_id": "proj_01234567",
  "name": "My Project"
}
```

### list_projects

List all registered projects with their status.

**Parameters:** None

**Returns:**
```json
{
  "status": "success",
  "total_projects": 2,
  "projects": [
    {
      "id": "proj_01234567",
      "name": "Backend API",
      "paths": ["/home/user/backend"],
      "is_watching": true,
      "last_indexed": "2024-01-15T10:30:00Z",
      "created_at": "2024-01-10T08:00:00Z",
      "updated_at": "2024-01-15T10:30:00Z",
      "active_indexing": {
        "task_id": "task_123",
        "progress": 45.5,
        "files_processed": 120,
        "total_files": 264
      }
    }
  ]
}
```

## Resources

The MCP server also exposes read-only resources that can be monitored:

### Indexing Tasks

- `indexing://tasks` - List all indexing tasks
- `indexing://tasks/{task_id}` - Get specific task details

### Projects

- `projects://list` - List all projects
- `projects://{project_id}` - Get project details
- `projects://{project_id}/files` - Get tracked files for a project

## Error Handling

All tools return consistent error responses:

```json
{
  "status": "error",
  "message": "Detailed error description"
}
```

Common errors:
- `"No valid directories provided"` - Invalid or non-existent paths
- `"No code has been indexed yet"` - Attempting search before indexing
- `"Project not found"` - Invalid project ID

## Usage Examples

### Python
```python
# Using MCP client library
async with mcp_client.connect("http://localhost:9483/sse") as client:
    # Index a repository
    result = await client.call_tool(
        "index_repository",
        directories=["/path/to/repo"],
        force_reindex=False
    )
    
    # Search for code
    results = await client.call_tool(
        "search_code",
        query="error handling middleware",
        limit=5
    )
```

### Claude Desktop

When configured in Claude Desktop, use natural language:

```
"Search for authentication logic in the codebase"
"Index the /home/user/projects/api directory"
"Show me the indexing queue status"
"Register my-project at /path/to/project for watching"
```

## Best Practices

1. **Index Before Searching**: Always ensure directories are indexed
2. **Use Semantic Queries**: Describe what the code does, not exact names
3. **Monitor Queue**: Check `get_index_stats` for large indexing jobs
4. **Register Projects**: Use project registration for codebases you actively develop
5. **Appropriate Limits**: Use reasonable result limits (10-20 for most cases)