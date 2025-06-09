# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Breeze is an MCP (Model Context Protocol) server that supports streamable HTTP transport for semantically indexing codebases. It uses LanceDB for vector storage and FastMCP for the server implementation.

## Development Setup

- Python version: 3.12
- Virtual environment: `.venv` directory
- Dependencies: Listed in `pyproject.toml` (lancedb, fastmcp)

## Common Commands

### Install dependencies

```bash
uv sync
```

### Run the application

```bash
uv run python -m breeze serve
```

### Run tests

```bash
uv run pytest
```

## Project Structure

- `main.py` - Entry point for the application
- `pyproject.toml` - Project configuration
- `breeze/core/` - Core functionality modules
  - `content_detection.py` - Language detection using identify & python-magic
  - `text_chunker.py` - Semantic text chunking for embeddings
  - `tree_sitter_queries.py` - Tree-sitter queries for code analysis
  - `embeddings.py` - Embedding generation with chunking support
  - `engine.py` - Main indexing and search engine

## Architecture & Design Decisions

### Content Detection

- **Single source of truth**: All language detection and normalization happens in `content_detection.py`
- Uses `identify` and `python-magic` for accurate language detection
- The `detect_language()` method returns normalized language names for tree-sitter
- Language aliases (e.g., 'py' → 'python', 'js' → 'javascript') are centralized here
- **DO NOT** duplicate language mappings elsewhere in the codebase

### Chunking Strategy

- **Current issues**: ModelAwareChunker has hardcoded values and truncates instead of chunking
- **Solution**: Replace with TextChunker which:
  - Looks for natural boundaries (newlines, spaces, periods, commas)
  - Properly handles overlapping chunks
  - Uses 16k token chunks (sweet spot for code files)
  - Combines chunks using weighted_average by token count
- **DO NOT** use character-based chunking for code

### Tree-sitter Integration

- Queries are defined in `tree_sitter_queries.py` for semantic code understanding
- Supports extensibility through QueryManager class
- Can load custom queries from JSON files
- Includes support for Zig, shell scripts, and other languages
- Used for identifying semantic boundaries (functions, classes, etc.)

### Rate Limiting

- Voyage API tiers properly configured with safety margins
- Uses RateLimiterV2 with token bucket algorithm
- Returns partial results instead of failing completely

## Key Technologies

- **LanceDB**: Vector database for semantic indexing
- **FastMCP**: Framework for building MCP servers with streamable HTTP transport
- **tree-sitter**: For semantic code analysis and chunking
- **identify**: For file type detection
- **python-magic**: For MIME type detection fallback

## Current Issues to Fix

1. **Replace ModelAwareChunker with TextChunker** - Priority HIGH
2. **Fix failing tests**:
   - `test_local_embedder_chunking.py`
   - `test_queue.py::test_startup_recovery`
3. **Refactor engine.py** - Split into separate modules for search, project, and task management

## Notes

- This project is in early development stage
- Based on the windtools-mcp implementation but exploring LanceDB as an alternative approach
- LanceDB only supports single writer (`concurrent_writers: int = 1`)

## Development Workflow

- Use uv commands in this project
- Use the uv venv in this project
- Run linting/typechecking before considering tasks complete

## Guidelines for Modification

- DO NOT ADD FILES, MODIFY THE EXISTING ONES
- Do not make up requirements or features
- KISS: Keep it Simple, stupid! the simplest possible implementation
- YAGNI: You are not going to need it
- DRY: Don't repeat yourself - centralize common logic

## Technology Preferences

- Use polars instead of pandas
- Use identify/python-magic for language detection (not hardcoded mappings)
- Use tree-sitter for semantic code analysis

## Collaboration Guidelines

- DO NOT MAKE STUFF UP / DO NOT MAKE ASSUMPTIONS - instead start a dialog with the user, we're a collaborative unit