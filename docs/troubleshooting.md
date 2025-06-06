# Troubleshooting Guide

This guide helps diagnose and resolve common issues with Breeze.

## Common Issues

### Installation Issues

#### "Command not found: breeze"

**Problem**: Breeze is not installed or not in PATH.

**Solutions**:
```bash
# Use with uvx (no installation needed)
uvx --from git+https://github.com/casualjim/breeze.git python -m breeze

# Or install locally
git clone https://github.com/casualjim/breeze.git
cd breeze
uv sync
uv run python -m breeze serve
```

#### "No module named 'breeze'"

**Problem**: Python can't find the breeze module.

**Solutions**:
```bash
# Ensure you're in the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Reinstall
uv sync
```

### Indexing Issues

#### "No code has been indexed yet"

**Problem**: Trying to search before indexing.

**Solution**:
```bash
# Index your codebase first
uv run python -m breeze index /path/to/your/code

# Then search
uv run python -m breeze search "your query"
```

#### Indexing is Very Slow

**Possible Causes**:
1. Large codebase
2. Slow network (for API models)
3. Insufficient resources

**Solutions**:
```bash
# Use a faster model tier (Voyage AI)
export BREEZE_VOYAGE_TIER=3
uv run python -m breeze index /path/to/code --model voyage-code-3

# Increase concurrency
export BREEZE_CONCURRENT_READERS=50
export BREEZE_CONCURRENT_EMBEDDERS=20

# Use local model for offline speed
uv run python -m breeze index /path/to/code --model ibm-granite/granite-embedding-125m-english
```

#### "Rate limit exceeded"

**Problem**: API rate limits hit.

**Solutions**:
```bash
# For Voyage AI, use a higher tier
export BREEZE_VOYAGE_TIER=2  # or 3

# Reduce concurrent requests
export BREEZE_VOYAGE_CONCURRENT_REQUESTS=2

# Add retry logic (automatic in Breeze)
```

### Search Issues

#### No Search Results

**Possible Causes**:
1. Query too specific
2. Index is empty
3. Relevance threshold too high

**Solutions**:
```bash
# Check index status
uv run python -m breeze stats

# Use more general queries
uv run python -m breeze search "authentication" --limit 20

# Lower relevance threshold
uv run python -m breeze search "auth logic" --min-relevance 0.0
```

#### Poor Search Quality

**Problem**: Results aren't relevant.

**Solutions**:
1. Use better embedding models:
   ```bash
   # Re-index with Voyage AI
   export VOYAGE_API_KEY="your-key"
   uv run python -m breeze index /path/to/code --model voyage-code-3 --force
   ```

2. Use semantic descriptions:
   - ❌ "handleClick"
   - ✅ "function that handles button click events"

3. Include context:
   - ❌ "parse"
   - ✅ "XML parsing logic for configuration files"

### Database Issues

#### "Failed to connect to database"

**Problem**: LanceDB connection issues.

**Solutions**:
```bash
# Check data directory permissions
ls -la ~/.breeze/data

# Fix permissions
chmod -R 755 ~/.breeze/data

# Use different data root
export BREEZE_DATA_ROOT=/tmp/breeze-data
uv run python -m breeze serve
```

#### "CommitConflict: Another writer has already written"

**Problem**: Multiple processes writing to the database.

**Solutions**:
1. Stop the MCP server before CLI indexing:
   ```bash
   # Stop server
   launchctl stop com.breeze-mcp.server  # macOS
   # or
   systemctl stop breeze-mcp  # Linux
   
   # Run indexing
   uv run python -m breeze index /path/to/code
   
   # Restart server
   launchctl start com.breeze-mcp.server
   ```

2. Use MCP tools instead of CLI when server is running

See [Concurrency Guide](concurrency.md) for details.

### API Key Issues

#### "API key required for Voyage models"

**Problem**: Missing API key for cloud models.

**Solutions**:
```bash
# Set Voyage AI key
export VOYAGE_API_KEY="your-api-key"
# or
export BREEZE_EMBEDDING_API_KEY="your-api-key"

# Set Google key
export GOOGLE_API_KEY="your-api-key"
```

#### "Invalid API key"

**Problem**: Incorrect or expired API key.

**Solutions**:
1. Verify key is correct
2. Check API dashboard for key status
3. Regenerate key if needed
4. Ensure no extra spaces in environment variable

### MCP Server Issues

#### Server Won't Start

**Common Causes**:
1. Port already in use
2. Permission issues
3. Missing dependencies

**Debug Steps**:
```bash
# Check if port is in use
lsof -i :9483

# Kill existing process
kill -9 <PID>

# Try different port
uv run python -m breeze serve --port 9484

# Check logs
tail -f /usr/local/var/log/breeze-mcp.log  # macOS LaunchAgent
journalctl -u breeze-mcp -f  # Linux systemd
```

#### "Connection refused" from Claude

**Problem**: MCP server not accessible.

**Solutions**:
1. Verify server is running:
   ```bash
   curl http://localhost:9483/health
   ```

2. Check Claude Desktop config:
   ```json
   {
     "mcpServers": {
       "breeze": {
         "command": "uvx",
         "args": ["--from", "git+https://github.com/casualjim/breeze.git", "python", "-m", "breeze", "serve"]
       }
     }
   }
   ```

3. Restart Claude Desktop after config changes

### Performance Issues

#### High Memory Usage

**Problem**: Breeze consuming too much RAM.

**Solutions**:
```bash
# Reduce batch sizes
export BREEZE_BATCH_SIZE=50

# Limit concurrent operations
export BREEZE_CONCURRENT_READERS=10
export BREEZE_CONCURRENT_EMBEDDERS=5

# Use streaming for large files
```

#### High CPU Usage

**Problem**: CPU at 100% during indexing.

**Solutions**:
1. Expected during embedding generation
2. Reduce concurrent embedders for CPU models
3. Use GPU if available:
   ```bash
   export BREEZE_EMBEDDING_DEVICE=cuda
   ```

### File Watching Issues

#### Changes Not Detected

**Problem**: File watcher not picking up changes.

**Solutions**:
1. Check project is registered:
   ```bash
   uv run python -m breeze list-projects
   ```

2. Verify watching is active:
   - Look for `"is_watching": true` in project list

3. Check file types:
   - Breeze uses content detection, not extensions
   - Binary files are skipped

4. Manual re-index:
   ```bash
   uv run python -m breeze index /path/to/project --force
   ```

## Debugging Tools

### Enable Debug Logging

```bash
# Set debug level
export LOG_LEVEL=DEBUG

# Run with verbose output
uv run python -m breeze serve --verbose
```

### Check System State

```python
# Python script to check Breeze state
import asyncio
from breeze.core import BreezeEngine, BreezeConfig

async def check_system():
    config = BreezeConfig()
    engine = BreezeEngine(config)
    await engine.initialize()
    
    # Get stats
    stats = await engine.get_stats()
    print(f"Documents: {stats['total_documents']}")
    print(f"Model: {stats['embedding_model']}")
    print(f"Database: {stats['database_path']}")
    
    # Test search
    results = await engine.search("test query", limit=1)
    print(f"Search works: {len(results) > 0}")

asyncio.run(check_system())
```

### Profile Performance

```python
# Profile indexing performance
import cProfile
import asyncio
from breeze.cli import index_command

def profile_indexing():
    asyncio.run(index_command(["/path/to/code"]))

cProfile.run('profile_indexing()', 'indexing.prof')

# Analyze results
import pstats
stats = pstats.Stats('indexing.prof')
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Getting Help

### Gather Information

Before reporting issues, collect:

1. **Version Information**:
   ```bash
   uv run python -m breeze --version
   python --version
   uv --version
   ```

2. **Configuration**:
   ```bash
   env | grep BREEZE
   ```

3. **Error Logs**:
   ```bash
   # Full error with traceback
   uv run python -m breeze serve 2>&1 | tee error.log
   ```

4. **System Info**:
   ```bash
   # macOS
   sw_vers
   
   # Linux
   lsb_release -a
   uname -a
   ```

### Report Issues

Create a GitHub issue with:
1. Clear description of the problem
2. Steps to reproduce
3. Expected vs actual behavior
4. System information
5. Relevant logs

### Community Support

- GitHub Issues: [github.com/casualjim/breeze/issues](https://github.com/casualjim/breeze/issues)
- Discussions: [github.com/casualjim/breeze/discussions](https://github.com/casualjim/breeze/discussions)

## Advanced Diagnostics

### Database Inspection

```python
# Inspect LanceDB directly
import lancedb
import asyncio

async def inspect_db():
    db = await lancedb.connect_async("~/.breeze/data/code_index")
    
    # List tables
    tables = await db.table_names()
    print(f"Tables: {tables}")
    
    # Check document count
    if "documents" in tables:
        docs = await db.open_table("documents")
        count = await docs.count()
        print(f"Document count: {count}")

asyncio.run(inspect_db())
```

### Network Diagnostics

```bash
# Test API connectivity (Voyage AI)
curl -X POST https://api.voyageai.com/v1/embeddings \
  -H "Authorization: Bearer $VOYAGE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "voyage-code-3", "input": ["test"]}'

# Test API connectivity (Google)
curl -X POST https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent \
  -H "X-Goog-Api-Key: $GOOGLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": {"parts":[{"text": "test"}]}}'
```

### File System Checks

```bash
# Check file permissions
find /path/to/code -type f -name "*.py" ! -readable

# Check disk space
df -h ~/.breeze

# Check inode usage
df -i ~/.breeze
```