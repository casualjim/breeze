# Breeze MCP Server

A high-performance MCP (Model Context Protocol) server for semantic code search and indexing, powered by LanceDB and optimized code embedding models.

## Features

- 🚀 **Fast semantic search** - Uses LanceDB for efficient vector similarity search
- 🧠 **Code-optimized embeddings** - Powered by nomic-ai/CodeRankEmbed model
- 📁 **Incremental indexing** - Only re-indexes changed files
- 🔄 **Async architecture** - Built with async/await for optimal performance
- 🗄️ **Efficient data processing** - Uses Polars DataFrames and Arrow format

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

Index code files from specified directories.

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

Get statistics about the current index.

### `list_directory`

List contents of a directory to help identify what to index.

**Parameters:**

- `directory_path` (str): Path to list

## Configuration

### Environment Variables

- `BREEZE_DATA_ROOT`: Directory for storing index data (default: `~/.breeze/data`)
- `BREEZE_DB_NAME`: Database name (default: `code_index`)
- `BREEZE_EMBEDDING_MODEL`: Embedding model to use (default: `nomic-ai/CodeRankEmbed`)
- `BREEZE_HOST`: Server host (default: `0.0.0.0`)
- `BREEZE_PORT`: Server port (default: `9483`)

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

### Quick Start (Native - Recommended for Apple Silicon)

```bash
# Install and run as a LaunchAgent (auto-starts on login)
./install-launchd.sh
```

### Docker (CPU-only)

```bash
docker-compose up -d
```

See [README-DEPLOYMENT.md](README-DEPLOYMENT.md) for detailed deployment options and performance considerations.

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

## License

MIT
