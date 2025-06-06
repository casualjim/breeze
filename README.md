# Breeze MCP Server

A high-performance MCP (Model Context Protocol) server for semantic code search and indexing, powered by LanceDB and optimized code embedding models.

## Features

- 🚀 **Fast semantic search** - Uses LanceDB for efficient vector similarity search
- 🧠 **Code-optimized embeddings** - Supports multiple embedding providers including Voyage AI
- 📁 **Incremental indexing** - Only re-indexes changed files
- 🔄 **Async architecture** - Built with async/await for optimal performance
- 🗄️ **Efficient data processing** - Uses Polars DataFrames and Arrow format
- 🌐 **Cloud & Local Models** - Support for Voyage AI, Google Gemini, and local models
- 🔍 **Intelligent Content Detection** - Automatically identifies code files using content analysis, not just extensions
- 📂 **Project Management** - Register projects for automatic file watching and re-indexing on changes
- ⏳ **Async Queue System** - Long-running indexing operations are queued and processed in the background

## Quick Start

### Using uvx (recommended)

```bash
# Run as MCP server
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze serve

# Index a repository
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze index /path/to/repo

# Search for code
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze search "factorial function"
```

### Using Voyage AI (Recommended for Code)

Voyage AI's `voyage-code-3` model provides state-of-the-art code embeddings with tier-based rate limits:

```bash
# Set your Voyage AI API key
export VOYAGE_API_KEY="your-api-key"

# Index with Voyage AI (Tier 1 - default)
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze index /path/to/repo \
  --model voyage-code-3 \
  --voyage-tier 1

# Use higher tiers for faster indexing
# Tier 2: 2x the rate limits (6M tokens/min, 4000 requests/min)
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze index /path/to/repo \
  --model voyage-code-3 \
  --voyage-tier 2

# Tier 3: 3x the rate limits (9M tokens/min, 6000 requests/min)
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze index /path/to/repo \
  --model voyage-code-3 \
  --voyage-tier 3

# Or configure via environment
export BREEZE_EMBEDDING_MODEL="voyage-code-3"
export BREEZE_EMBEDDING_API_KEY="your-api-key"
export BREEZE_VOYAGE_TIER="2"  # Use tier 2
```

**Voyage AI Tier Rate Limits:**

- **Tier 1** (default): 3M tokens/minute, 2000 requests/minute
- **Tier 2**: 6M tokens/minute, 4000 requests/minute (2x base)
- **Tier 3**: 9M tokens/minute, 6000 requests/minute (3x base)

The concurrent requests are automatically calculated based on your tier, but can be overridden with `--voyage-requests`.

### Using Other Embedding Providers

```bash
# Google Gemini
export GOOGLE_API_KEY="your-api-key"
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze index /path/to/repo \
  --model models/text-embedding-004

# Local models (default)
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze index /path/to/repo \
  --model ibm-granite/granite-embedding-125m-english
```

### Claude Desktop Configuration

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "breeze": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/casualjim/breeze.git", "python", "-m", "breeze", "serve"],
      "env": {
        "BREEZE_DATA_ROOT": "/path/to/index/storage"
      }
    }
  }
}
```

## Installation

### As MCP Server

```bash
# Run directly with uvx
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze serve

# Or clone and run locally
git clone https://github.com/casualjim/breeze.git
cd breeze
uv run python -m breeze serve
```

### As Python Library

```python
import asyncio
from breeze.core import BreezeEngine, BreezeConfig

async def main():
    # Configure engine
    config = BreezeConfig(
        data_root="/path/to/index/storage",
        embedding_model="nomic-ai/CodeRankEmbed"
    )
    
    # Create engine
    engine = BreezeEngine(config)
    
    # Index a codebase
    stats = await engine.index_directories(["/path/to/codebase"])
    print(f"Indexed {stats.files_indexed} files")
    
    # Search for code
    results = await engine.search("factorial function", limit=5)
    for result in results:
        print(f"{result.file_path} (score: {result.relevance_score:.3f})")
        print(result.snippet)
        print("---")

asyncio.run(main())
```

### Running the MCP Server

```bash
# Using uvx (no installation needed)
uvx --from . python -m breeze serve

# With custom settings
BREEZE_DATA_ROOT=/custom/path uvx --from . python -m breeze serve --port 8080

# Or run directly with uv
uv run python -m breeze serve
```

## MCP Tools

The server exposes the following tools:

### `index_repository`

Index code files from specified directories. This tool now queues an indexing task that runs asynchronously in the background, preventing timeouts for large repositories.

**Parameters:**

- `directories` (List of strings): List of absolute paths to index
- `force_reindex` (bool): Force re-indexing of all files

### `search_code`

Search for code semantically similar to a query.

**Parameters:**

- `query` (str): Search query
- `limit` (int): Maximum results (default: 10)
- `min_relevance` (float): Minimum relevance score (0.0-1.0)

### `get_index_stats`

Get comprehensive statistics about the index and indexing queue.

### `list_directory`

List contents of a directory to help identify what to index.

**Parameters:**

- `directory_path` (str): Path to list

### `register_project`

Register a new project for automatic file watching and indexing. The system uses intelligent content detection to identify code files, not just file extensions.

**Parameters:**

- `name` (str): Project name
- `paths` (List of strings): Directory paths to track
- `auto_index` (bool): Whether to perform initial indexing (default: True)

### `unregister_project`

Stop watching and unregister a project.

**Parameters:**

- `project_id` (str): ID of the project to unregister

### `list_projects`

List all registered projects with their current status.

## Configuration

### Environment Variables

- `BREEZE_DATA_ROOT`: Directory for storing index data (default: `~/.breeze/data`)
- `BREEZE_DB_NAME`: Database name (default: `code_index`)
- `BREEZE_EMBEDDING_MODEL`: Embedding model to use (default: `nomic-ai/CodeRankEmbed`)
- `BREEZE_EMBEDDING_API_KEY`: API key for cloud embedding providers
- `BREEZE_HOST`: Server host (default: `0.0.0.0`)
- `BREEZE_PORT`: Server port (default: `9483`)
- `BREEZE_CONCURRENT_READERS`: Concurrent file readers (default: `20`)
- `BREEZE_CONCURRENT_EMBEDDERS`: Concurrent embedders (default: `10`)
- `BREEZE_CONCURRENT_WRITERS`: Concurrent DB writers (default: `10`)
- `BREEZE_VOYAGE_CONCURRENT_REQUESTS`: Max concurrent Voyage AI requests (default: `5`)

### MCP Configuration Examples

#### VS Code

```json
{
  "mcp.servers": {
    "breeze": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/casualjim/breeze.git", "python", "-m", "breeze", "serve"],
      "env": {
        "BREEZE_DATA_ROOT": "${workspaceFolder}/.breeze"
      }
    }
  }
}
```

#### Cursor

```json
{
  "mcpServers": {
    "breeze": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/casualjim/breeze.git", "breeze", "serve"]
    }
  }
}
```

## Testing

Run the test script to verify functionality:

```bash
uv run python test_breeze.py
```

## Deployment

### Native macOS with MPS Acceleration (Recommended for Apple Silicon)

For best performance on Apple Silicon Macs, run natively to access MPS hardware acceleration:

#### Option 1: LaunchAgent (Auto-start)

```bash
# Configure your settings
cp .env.example .env
# Edit .env with your settings (especially API keys for cloud models)

# Install as a LaunchAgent
python install-launchd.py

# Check status
launchctl list | grep breeze

# View logs
tail -f /usr/local/var/log/breeze-mcp.log
```

#### Option 2: Direct Execution

```bash
# Run directly
uv run python -m breeze serve

# Or with custom settings
BREEZE_PORT=8080 uv run python -m breeze serve
```

### Docker Deployment (CPU-only)

Docker runs in a Linux VM on macOS and cannot access MPS. Use for Linux servers or CI/CD:

```bash
# Using docker-compose
docker-compose up -d

# Or using docker directly
docker build -t breeze-mcp .
docker run -d \
  -p 9483:9483 \
  -v breeze-data:/data \
  -e BREEZE_HOST=0.0.0.0 \
  breeze-mcp
```

### Performance Comparison

| Method | Hardware Acceleration | Embedding Speed |
|--------|----------------------|----------------|
| Native macOS | MPS (Metal) | ~10x faster on Apple Silicon |
| Docker | CPU only | Baseline speed |

## Architecture

- **Core Engine** (`breeze/core/engine.py`): Main indexing and search logic
- **Models** (`breeze/core/models.py`): Pydantic v2 data models
- **MCP Server** (`breeze/mcp/server.py`): FastMCP server implementation
- **Single LanceDB connection**: Efficient resource usage
- **Async generators**: Memory-efficient file processing

## Improvements over windtools-mcp

- Uses LanceDB instead of ChromaDB for better performance and Arrow integration
- Proper async/await throughout with async generators
- Single database connection shared across operations
- Polars for efficient data processing
- Pydantic v2 models with proper type annotations
- CodeRankEmbed model specifically designed for code search

## CLAUDE.md Example

To encourage Claude to use Breeze for finding code instead of built-in tools, add this to your project's CLAUDE.md:

```markdown
# Code Search Instructions

This project has a Breeze MCP server configured for fast semantic code search. 

## When searching for code:

1. **Use Breeze first**: Always use the `search_code` tool from the Breeze MCP server before using other search methods
2. **Semantic queries work best**: Instead of searching for exact function names, describe what the code does
3. **Check index status**: Use `get_index_stats` to see how many files are indexed
4. **Register projects**: For ongoing work, use `register_project` to enable automatic re-indexing on file changes

## Examples:

- Instead of: "find handleClick function"
- Use: "search_code" with query "click event handler"

- Instead of: grep or file searching
- Use: "search_code" with descriptive queries like "authentication logic" or "database connection setup"

## Project Registration:

If working on this codebase long-term:
```
register_project(
    name="MyProject",
    paths=["/path/to/project"],
    auto_index=true
)
```

This enables automatic re-indexing when files change, keeping search results current.
```

## License

MIT
