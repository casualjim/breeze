# Configuration Guide

Breeze can be configured through environment variables, command-line options, or programmatically.

## Environment Variables

### Core Settings

| Variable           | Description                      | Default                              |
| ------------------ | -------------------------------- | ------------------------------------ |
| `BREEZE_DATA_ROOT` | Directory for storing index data | `~/.breeze/data` (platform-specific) |
| `BREEZE_DB_NAME`   | Database name                    | `code_index`                         |
| `BREEZE_HOST`      | Server host binding              | `127.0.0.1`                          |
| `BREEZE_PORT`      | Server port                      | `9483`                               |

### Embedding Model Configuration

| Variable                   | Description                 | Default                                      |
| -------------------------- | --------------------------- | -------------------------------------------- |
| `BREEZE_EMBEDDING_MODEL`   | Model to use for embeddings | `ibm-granite/granite-embedding-125m-english` |
| `BREEZE_EMBEDDING_API_KEY` | API key for cloud models    | None                                         |
| `BREEZE_EMBEDDING_DEVICE`  | Device for local models     | `cpu` (auto-detects GPU)                     |

### Performance Tuning

| Variable                      | Description                     | Default |
| ----------------------------- | ------------------------------- | ------- |
| `BREEZE_CONCURRENT_READERS`   | Concurrent file readers         | `20`    |
| `BREEZE_CONCURRENT_EMBEDDERS` | Concurrent embedding operations | `10`    |
| `BREEZE_BATCH_SIZE`           | Documents per batch             | `100`   |


### Voyage AI Specific

| Variable                            | Description                 | Default |
| ----------------------------------- | --------------------------- | ------- |
| `VOYAGE_API_KEY`                    | Voyage AI API key           | None    |
| `BREEZE_VOYAGE_CONCURRENT_REQUESTS` | Max concurrent API requests | `5`     |
| `BREEZE_VOYAGE_TIER`                | Rate limit tier (1-3)       | `1`     |

### Google Gemini Specific

| Variable         | Description               | Default |
| ---------------- | ------------------------- | ------- |
| `GOOGLE_API_KEY` | Google API key for Gemini | None    |

## Configuration Precedence

Configuration is applied in this order (later overrides earlier):

1. Default values
2. Environment variables
3. Command-line arguments
4. Programmatic configuration

## Model Configuration

### Local Models

Available local models:

- `ibm-granite/granite-embedding-125m-english`
- `sentence-transformers/all-MiniLM-L6-v2`
- Any HuggingFace sentence-transformer model

Configuration example:

```bash
export BREEZE_EMBEDDING_MODEL="ibm-granite/granite-embedding-125m-english"
export BREEZE_EMBEDDING_DEVICE="cuda"  # Use GPU if available
```

### Voyage AI Models

Available models:

- `voyage-code-3` (recommended for code)
- `voyage-code-2`
- `voyage-3`
- `voyage-3-lite`

Configuration example:

```bash
export BREEZE_EMBEDDING_MODEL="voyage-code-3"
export VOYAGE_API_KEY="your-api-key"
export BREEZE_VOYAGE_TIER="2"  # Use tier 2 rate limits
```

Rate limit tiers:

- Tier 1: 3M tokens/min, 2000 requests/min
- Tier 2: 6M tokens/min, 4000 requests/min
- Tier 3: 9M tokens/min, 6000 requests/min

### Google Gemini Models

Available models:

- `models/text-embedding-004` (latest)
- `models/embedding-001`

Configuration example:

```bash
export BREEZE_EMBEDDING_MODEL="models/text-embedding-004"
export GOOGLE_API_KEY="your-api-key"
```

## Performance Configuration

### Concurrency Settings

Adjust based on your hardware:

```bash
# For high-end machines
export BREEZE_CONCURRENT_READERS=50
export BREEZE_CONCURRENT_EMBEDDERS=20
export BREEZE_CONCURRENT_WRITERS=20

# For low-end machines or CI/CD
export BREEZE_CONCURRENT_READERS=5
export BREEZE_CONCURRENT_EMBEDDERS=2
export BREEZE_CONCURRENT_WRITERS=2
```

### Memory Management

Control batch sizes to manage memory:

```bash
# Smaller batches for limited memory
export BREEZE_BATCH_SIZE=50

# Larger batches for high memory systems
export BREEZE_BATCH_SIZE=200
```

## Storage Configuration

### Data Location

Control where indexes are stored:

```bash
# Project-specific storage
export BREEZE_DATA_ROOT="/path/to/project/.breeze"

# Shared storage
export BREEZE_DATA_ROOT="/var/lib/breeze"

# Temporary storage
export BREEZE_DATA_ROOT="/tmp/breeze"
```

### Multiple Indexes

Use different database names for separate indexes:

```bash
# Development index
export BREEZE_DB_NAME="dev_code_index"

# Production index
export BREEZE_DB_NAME="prod_code_index"
```

## MCP Server Configuration

### Claude Desktop

```json
{
  "mcpServers": {
    "breeze": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/casualjim/breeze.git", "python", "-m", "breeze", "serve"],
      "env": {
        "BREEZE_DATA_ROOT": "/path/to/indexes",
        "BREEZE_EMBEDDING_MODEL": "voyage-code-3",
        "VOYAGE_API_KEY": "your-api-key",
        "BREEZE_PORT": "9483"
      }
    }
  }
}
```

### VS Code Extension

```json
{
  "mcp.servers": {
    "breeze": {
      "command": "python",
      "args": ["-m", "breeze", "serve"],
      "env": {
        "BREEZE_DATA_ROOT": "${workspaceFolder}/.breeze",
        "BREEZE_EMBEDDING_MODEL": "ibm-granite/granite-embedding-125m-english"
      }
    }
  }
}
```

## Programmatic Configuration

### Python API

```python
from breeze.core import BreezeConfig, BreezeEngine

# Create custom configuration
config = BreezeConfig(
    data_root="/custom/path",
    db_name="my_index",
    embedding_model="voyage-code-3",
    embedding_api_key="your-key",
    concurrent_readers=30,
    concurrent_embedders=15,
    concurrent_writers=15,
    batch_size=150
)

# Create engine with config
engine = BreezeEngine(config)
await engine.initialize()
```

### Configuration Object

All configuration options:

```python
@dataclass
class BreezeConfig:
    # Storage
    data_root: Optional[str] = None
    db_name: str = "code_index"
    
    # Embedding model
    embedding_model: str = "ibm-granite/granite-embedding-125m-english"
    embedding_device: Optional[str] = None
    embedding_api_key: Optional[str] = None
    trust_remote_code: bool = True
    
    # Performance
    concurrent_readers: int = 20
    concurrent_embedders: int = 10
    concurrent_writers: int = 10
    batch_size: int = 100
    
    # Voyage AI specific
    voyage_concurrent_requests: int = 5
    
    # File handling
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    chunk_size: int = 1500
    chunk_overlap: int = 200
```


## Configuration Examples

### High-Performance Setup

```bash
# Use Voyage AI with maximum concurrency
export BREEZE_EMBEDDING_MODEL="voyage-code-3"
export VOYAGE_API_KEY="your-api-key"
export BREEZE_VOYAGE_TIER="3"
export BREEZE_CONCURRENT_READERS=100
export BREEZE_CONCURRENT_EMBEDDERS=50
export BREEZE_CONCURRENT_WRITERS=30
export BREEZE_BATCH_SIZE=200
```

### Privacy-Focused Setup

```bash
# Use local model, no external APIs
export BREEZE_EMBEDDING_MODEL="ibm-granite/granite-embedding-125m-english"
export BREEZE_EMBEDDING_DEVICE="cuda"
export BREEZE_DATA_ROOT="/encrypted/volume/breeze"
```

### CI/CD Setup

```bash
# Conservative settings for CI environments
export BREEZE_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export BREEZE_CONCURRENT_READERS=5
export BREEZE_CONCURRENT_EMBEDDERS=2
export BREEZE_CONCURRENT_WRITERS=2
export BREEZE_BATCH_SIZE=50
export BREEZE_DATA_ROOT="${CI_PROJECT_DIR}/.breeze"
```

### Development Setup

```bash
# Fast iteration with local model
export BREEZE_EMBEDDING_MODEL="ibm-granite/granite-embedding-125m-english"
export BREEZE_DATA_ROOT="${HOME}/dev/.breeze"
export BREEZE_DB_NAME="dev_index"
export BREEZE_PORT=9484  # Different port to avoid conflicts
```

## Debugging Configuration

Enable debug logging:

```bash
# Set Python logging level
export PYTHONUNBUFFERED=1
export LOG_LEVEL=DEBUG

# Run with verbose output
uv run python -m breeze serve --verbose
```

Check current configuration:

```bash
# Display effective configuration
uv run python -m breeze config

# Validate settings
uv run python -m breeze validate-config
```
