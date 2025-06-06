# Deployment Guide

This guide covers various deployment options for Breeze.

## Deployment Options

### 1. Direct Execution with uv (Recommended for Development)

Best for development and testing:

```bash
# Clone repository
git clone https://github.com/casualjim/breeze
cd breeze

# Install dependencies
uv sync

# Run directly
uv run python -m breeze serve
```

### 2. LaunchAgent (macOS Auto-start)

For automatic startup on macOS:

```bash
# Configure settings
cp .env.example .env
# Edit .env with your configuration

# Install LaunchAgent
uv run python install-launchd.py

# Verify installation
launchctl list | grep breeze

# View logs
tail -f /usr/local/var/log/breeze-mcp.log
```

**LaunchAgent Management:**

```bash
# Stop service
launchctl stop com.breeze-mcp.server

# Start service
launchctl start com.breeze-mcp.server

# Uninstall
launchctl unload ~/Library/LaunchAgents/com.breeze-mcp.server.plist
rm ~/Library/LaunchAgents/com.breeze-mcp.server.plist
```

### 3. Systemd Service (Linux)

#### Setup User and Environment

First, create a dedicated user and set up the environment:

```bash
# Create system user for breeze
sudo useradd -r -m -d /var/lib/breeze -s /bin/bash breeze

# Create application directory
sudo mkdir -p /opt/breeze
sudo chown breeze:breeze /opt/breeze

# Switch to breeze user
sudo -u breeze -i

# Clone the repository
cd /opt/breeze
git clone https://github.com/casualjim/breeze.git .

# Install uv for the breeze user
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create virtual environment and install dependencies
uv sync

# Exit back to your regular user
exit
```

#### Create Systemd Service

Create `/etc/systemd/system/breeze-mcp.service`:

```ini
[Unit]
Description=Breeze MCP Server
After=network.target

[Service]
Type=simple
User=breeze
Group=breeze
WorkingDirectory=/opt/breeze
Environment="PATH=/var/lib/breeze/.local/bin:/usr/bin:/bin"
Environment="BREEZE_DATA_ROOT=/var/lib/breeze/data"
Environment="BREEZE_HOST=0.0.0.0"
Environment="BREEZE_PORT=9483"
ExecStart=/var/lib/breeze/.local/bin/uv run python -m breeze serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Service Management:**

```bash
# Enable and start
sudo systemctl enable breeze-mcp
sudo systemctl start breeze-mcp

# Check status
sudo systemctl status breeze-mcp

# View logs
sudo journalctl -u breeze-mcp -f
```

### 4. Docker Deployment

**Note**: Docker runs CPU-only on macOS (no MPS acceleration).

Build and run using the included Dockerfile:

```bash
# Build the image
docker build -t breeze-mcp .

# Run with volume for data persistence
docker run -d \
  --name breeze \
  -p 9483:9483 \
  -v breeze-data:/data \
  -e BREEZE_EMBEDDING_MODEL="voyage-code-3" \
  -e VOYAGE_API_KEY="your-api-key" \
  breeze-mcp

# Check logs
docker logs -f breeze

# Health check
curl http://localhost:9483/health
```

**Docker Compose** (`docker-compose.yml`):

```yaml

services:
  breeze:
    build: .
    ports:
      - "9483:9483"
    volumes:
      - breeze-data:/data
    environment:
      - BREEZE_HOST=0.0.0.0
      - BREEZE_PORT=9483
      - BREEZE_EMBEDDING_MODEL=${BREEZE_EMBEDDING_MODEL:-ibm-granite/granite-embedding-125m-english}
      - BREEZE_EMBEDDING_API_KEY=${BREEZE_EMBEDDING_API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9483/health"]
      interval: 30s
      timeout: 3s
      retries: 3

volumes:
  breeze-data:
```

Run with docker-compose:

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## Environment Configuration

Create a `.env` file for your deployment:

```bash
# Embedding configuration
BREEZE_EMBEDDING_MODEL=voyage-code-3
VOYAGE_API_KEY=your-voyage-api-key

# Or use Google
# BREEZE_EMBEDDING_MODEL=models/text-embedding-004
# GOOGLE_API_KEY=your-google-api-key

# Performance tuning
BREEZE_CONCURRENT_READERS=20
BREEZE_CONCURRENT_EMBEDDERS=10
BREEZE_CONCURRENT_WRITERS=10

# Data storage
BREEZE_DATA_ROOT=/var/lib/breeze/data
```

## Important Considerations

### Concurrency Limitations

⚠️ **LanceDB does not support concurrent writes from multiple processes.** This means:

- Only run one instance of Breeze at a time
- Stop the server before running CLI indexing commands
- Each instance must use a different `BREEZE_DATA_ROOT`

See the [Concurrency Guide](concurrency.md) for details.

### Resource Requirements

- **Memory**: 2GB minimum, 4GB+ recommended
- **Disk**: Depends on codebase size (roughly 2-3x the size of indexed code)
- **CPU**: Benefits from multiple cores for concurrent processing

### Security

- Always use environment variables for API keys
- Restrict access to the data directory
- Use firewall rules to limit access to the MCP port
- Consider using a reverse proxy for HTTPS

## Monitoring

### Health Check

The server exposes a health endpoint:

```bash
curl http://localhost:9483/health
```

### Logs

- **Systemd**: `journalctl -u breeze-mcp -f`
- **Docker**: `docker logs -f breeze`
- **LaunchAgent**: `tail -f /usr/local/var/log/breeze-mcp.log`

### Check Index Status

```bash
# Using the CLI
uv run python -m breeze stats

# Or via MCP tools
# Use get_index_stats tool through an MCP client
```
