# Deployment Options for Breeze MCP Server

## Native macOS with MPS Acceleration (Recommended for Apple Silicon)

For best performance on Apple Silicon Macs, run natively to access MPS hardware acceleration:

### Option 1: LaunchAgent (Auto-start)
```bash
# Install as a LaunchAgent
./install-launchd.sh

# Check status
launchctl list | grep breeze

# View logs
tail -f /usr/local/var/log/breeze-mcp.log

# Stop/Start
launchctl unload ~/Library/LaunchAgents/com.breeze-mcp.server.plist
launchctl load ~/Library/LaunchAgents/com.breeze-mcp.server.plist
```

### Option 2: Direct Execution
```bash
# Install dependencies
uv sync

# Run directly
uv run python -m breeze serve

# Or with custom settings
BREEZE_PORT=8080 uv run python -m breeze serve
```

## Docker Deployment (CPU-only)

Docker runs in a Linux VM on macOS and cannot access MPS. Use this for:
- Linux servers
- CI/CD environments
- When MPS acceleration is not needed

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

## Performance Comparison

| Method | Hardware Acceleration | Embedding Speed |
|--------|----------------------|----------------|
| Native macOS | MPS (Metal) | ~10x faster on Apple Silicon |
| Docker | CPU only | Baseline speed |

## Choosing the Right Method

- **Apple Silicon Mac (M1/M2/M3)**: Use native execution for MPS acceleration
- **Intel Mac**: Either method works, no MPS available
- **Linux Server**: Use Docker
- **Development**: Use native execution with `uv run`
- **Production on Mac**: Use LaunchAgent for auto-start and MPS access