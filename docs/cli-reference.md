# CLI Reference

Breeze provides a comprehensive command-line interface for indexing and searching code.

## Commands

### `breeze serve`

Start the MCP server.

```bash
uv run uv run python -m breeze serve [OPTIONS]
```

**Options:**

- `--host TEXT`: Host to bind to (default: 127.0.0.1)
- `--port INTEGER`: Port to bind to (default: 9483)

**Examples:**

```bash
# Start with default settings
uv run python -m breeze serve

# Custom host and port
uv run python -m breeze serve --host 0.0.0.0 --port 8080

# Using uvx
uvx --from git+https://github.com/casualjim/breeze.git uv run python -m breeze serve
```

### `breeze index`

Index code repositories for semantic search.

```bash
uv run python -m breeze index PATH [PATH ...] [OPTIONS]
```

**Arguments:**

- `PATH`: One or more directory paths to index

**Options:**

- `--force`: Force re-indexing of all files, even if already indexed
- `--model TEXT`: Embedding model to use (default: ibm-granite/granite-embedding-125m-english)
- `--api-key TEXT`: API key for cloud embedding providers
- `--voyage-tier INTEGER`: Voyage AI tier level (1-3) for rate limiting
- `--voyage-requests INTEGER`: Override concurrent requests for Voyage AI
- `--data-root PATH`: Custom data storage location

**Examples:**

```bash
# Index a single repository
uv run python -m breeze index /path/to/repo

# Index multiple directories
uv run python -m breeze index /project1 /project2 /project3

# Force re-indexing with a specific model
uv run python -m breeze index /path/to/repo --force --model voyage-code-3

# Use Voyage AI with tier 2 rate limits
export VOYAGE_API_KEY="your-api-key"
uv run python -m breeze index /path/to/repo --model voyage-code-3 --voyage-tier 2
```

### `breeze search`

Search indexed code semantically.

```bash
uv run python -m breeze search QUERY [OPTIONS]
```

**Arguments:**

- `QUERY`: Search query describing what you're looking for

**Options:**

- `--limit INTEGER`: Maximum number of results (default: 10)
- `--min-relevance FLOAT`: Minimum relevance score 0.0-1.0 (default: 0.0)
- `--data-root PATH`: Custom data storage location

**Examples:**

```bash
# Basic search
uv run python -m breeze search "authentication logic"

# Search with more results
uv run python -m breeze search "database connection" --limit 20

# Filter by relevance
uv run python -m breeze search "error handling" --min-relevance 0.7
```

### `breeze stats`

Display index statistics.

```bash
uv run python -m breeze stats [OPTIONS]
```

**Options:**

- `--data-root PATH`: Custom data storage location

**Output includes:**

- Total indexed documents
- Index size
- Embedding model information
- Failed batch statistics
- Queue status (if server is running)

### `breeze add-project`

Register a project for automatic file watching and indexing.

```bash
uv run python -m breeze add-project NAME PATH [PATH ...] [OPTIONS]
```

**Arguments:**

- `NAME`: Project name
- `PATH`: One or more directory paths to watch

**Options:**

- `--no-index`: Skip initial indexing
- `--data-root PATH`: Custom data storage location

**Examples:**

```bash
# Register and index a project
uv run python -m breeze add-project myapp /path/to/myapp

# Register without initial indexing
uv run python -m breeze add-project myapp /path/to/myapp --no-index

# Register with multiple paths
uv run python -m breeze add-project monorepo /apps /packages /services
```

### `breeze remove-project`

Unregister a project and stop file watching.

```bash
uv run python -m breeze remove-project PROJECT_ID [OPTIONS]
```

**Arguments:**

- `PROJECT_ID`: ID of the project to remove

**Options:**

- `--data-root PATH`: Custom data storage location

### `breeze list-projects`

List all registered projects.

```bash
uv run python -m breeze list-projects [OPTIONS]
```

**Options:**

- `--data-root PATH`: Custom data storage location

**Output includes:**

- Project ID
- Project name
- Watched paths
- Watch status
- Last indexed timestamp

## Environment Variables

All commands respect environment variables for configuration:

```bash
# Set data storage location
export BREEZE_DATA_ROOT=/custom/path

# Configure embedding model
export BREEZE_EMBEDDING_MODEL=voyage-code-3
export BREEZE_EMBEDDING_API_KEY=your-api-key

# Configure Voyage AI tier
export BREEZE_VOYAGE_TIER=2

# Run commands with environment config
uv run python -m breeze index /path/to/repo
uv run python -m breeze search "query"
```

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `3`: Database error
- `4`: API error (for cloud models)

## Performance Tips

1. **Use appropriate models**:
   - Local models for offline/privacy
   - Voyage AI for best code understanding
   - Google Gemini for general text

2. **Batch operations**: Index multiple directories in one command

3. **Incremental indexing**: Breeze automatically skips unchanged files

4. **Project registration**: Use `add-project` for repositories you work on frequently
