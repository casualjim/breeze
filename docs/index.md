# Breeze Documentation

Welcome to the Breeze documentation. Breeze is a high-performance MCP (Model Context Protocol) server for semantic code search and indexing, powered by LanceDB and optimized code embedding models.

## Quick Links

- [Getting Started](getting-started.md) - Installation and basic usage
- [CLI Reference](cli-reference.md) - Complete command-line interface documentation
- [MCP Tools](mcp-tools.md) - MCP server tools and API reference
- [Configuration](configuration.md) - Environment variables and settings
- [Architecture](architecture.md) - Technical design and implementation details
- [Concurrency Guide](concurrency.md) - Important information about multi-process usage
- [Deployment](deployment.md) - Production deployment options

## Key Features

- **Fast Semantic Search**: Uses LanceDB for efficient vector similarity search
- **Code-Optimized Embeddings**: Supports multiple embedding providers including Voyage AI
- **Incremental Indexing**: Only re-indexes changed files
- **Async Architecture**: Built with async/await for optimal performance
- **Intelligent Content Detection**: Automatically identifies code files using content analysis
- **Project Management**: Register projects for automatic file watching and re-indexing

## Why Breeze?

Breeze is designed specifically for code search, offering:

1. **Better Context Understanding**: Semantic search understands the meaning of your query, not just keywords
2. **Language Agnostic**: Works with any programming language or text format
3. **Fast Performance**: Optimized for large codebases with efficient indexing
4. **MCP Integration**: Seamlessly integrates with Claude and other AI assistants

## Getting Help

- [Troubleshooting Guide](troubleshooting.md)
- [GitHub Issues](https://github.com/casualjim/breeze/issues)
