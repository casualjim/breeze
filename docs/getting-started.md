# Getting Started

This guide will help you get up and running with Breeze quickly.

## Prerequisites

- Python 3.10 or higher
- (Optional) [uv](https://github.com/astral-sh/uv) for package management
- (Optional) API keys for cloud embedding providers

## Installation

### Quick Start with uvx

The fastest way to use Breeze is with `uvx` (no installation required):

```bash
# Start the MCP server
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze serve

# Index a codebase
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze index /path/to/code

# Search for code
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze search "authentication logic"
```

### Local Installation

For development or frequent use:

```bash
# Clone the repository
git clone https://github.com/casualjim/breeze.git
cd breeze

# Install with uv
uv sync

# Run commands
uv run python -m breeze serve
```

### Install with pip

```bash
pip install git+https://github.com/casualjim/breeze.git
```

## Basic Usage

### 1. Index Your Code

First, index a codebase to make it searchable:

```bash
# Index a single project
uv run python -m breeze index /path/to/your/project

# Index multiple directories
uv run python -m breeze index /project1 /project2 /shared/libraries
```

Breeze automatically:

- Detects code files using content analysis
- Skips binary files and non-code content
- Shows progress during indexing

### 2. Search Your Code

Once indexed, search semantically:

```bash
# Find authentication logic
uv run python -m breeze search "user authentication and password validation"

# Find error handling
uv run python -m breeze search "error handling for network requests"

# Get more results
uv run python -m breeze search "database connection" --limit 20
```

### 3. Check Index Status

View statistics about your index:

```bash
uv run python -m breeze stats
```

## Using with Claude Desktop

### 1. Configure Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "breeze": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/casualjim/breeze.git",
        "python",
        "-m", 
        "breeze",
        "serve"
      ],
      "env": {
        "BREEZE_DATA_ROOT": "/path/to/store/indexes"
      }
    }
  }
}
```

### 2. Restart Claude Desktop

After updating the configuration, restart Claude Desktop.

### 3. Use in Conversations

Now you can ask Claude to search your code:

```text
"Search for the user authentication logic"
"Find where database connections are configured"
"Show me error handling examples in the codebase"
```

## Choosing an Embedding Model

### Local Models (Default)

Best for privacy and offline use:

```bash
# Default model (works offline)
uv run python -m breeze index /path/to/code

# Specific local model
uv run python -m breeze index /path/to/code --model ibm-granite/granite-embedding-125m-english
```

### Voyage AI (Recommended for Code)

Superior code understanding with API:

```bash
# Set API key
export VOYAGE_API_KEY="your-api-key"

# Use voyage-code-3
uv run python -m breeze index /path/to/code --model voyage-code-3

# With higher rate limits (tier 2)
uv run python -m breeze index /path/to/code --model voyage-code-3 --voyage-tier 2
```

### Google Gemini

For general-purpose embeddings:

```bash
# Set API key
export GOOGLE_API_KEY="your-api-key"

# Use Gemini embeddings
uv run python -m breeze index /path/to/code --model models/text-embedding-004
```

## Project Management

For codebases you actively develop:

### Register a Project

```bash
# Register and watch for changes
uv run python -m breeze add-project myapp /path/to/myapp

# Register multiple paths
uv run python -m breeze add-project monorepo /apps /packages /libs
```

Benefits:

- Automatic re-indexing when files change
- Tracks project metadata
- Efficient incremental updates

### List Projects

```bash
uv run python -m breeze list-projects
```

### Remove a Project

```bash
uv run python -m breeze remove-project proj_12345
```

## Important: Concurrency Limitations

**Warning**: Breeze's database (LanceDB) does not support concurrent writes from multiple processes.

If running the MCP server, you must either:

1. Stop the server before using CLI commands for indexing
2. Use the MCP tools instead of CLI commands

See the [Concurrency Guide](concurrency.md) for details.

## Next Steps

- Read the [CLI Reference](cli-reference.md) for all command options
- Explore [MCP Tools](mcp-tools.md) for integration
- Check [Configuration](configuration.md) for advanced settings
- Review [Best Practices](#best-practices) below

## Best Practices

1. **Index Regularly**: Re-index after major code changes
2. **Use Semantic Queries**: Describe what the code does, not exact names
3. **Register Active Projects**: Use project management for codebases you modify
4. **Choose the Right Model**:
   - Local for privacy/offline
   - Voyage AI for best code search
   - Gemini for mixed content
5. **Monitor Performance**: Use `stats` command to check index health

## Troubleshooting

### "No code has been indexed yet"

Run indexing first:

```bash
python -m breeze index /path/to/code
```

### Slow Indexing

- Use cloud models with higher tiers for large codebases
- Ensure good network connectivity for API-based models
- Check available disk space for local models

### API Key Errors

Set the appropriate environment variable:

```bash
export VOYAGE_API_KEY="your-key"  # For Voyage AI
export GOOGLE_API_KEY="your-key"  # For Gemini
```

### Concurrent Write Errors

Stop the MCP server before running CLI indexing commands. See [Concurrency Guide](concurrency.md).
