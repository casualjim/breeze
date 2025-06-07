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

## Key Technologies

- **LanceDB**: Vector database for semantic indexing
- **FastMCP**: Framework for building MCP servers with streamable HTTP transport

## Notes

- This project is in early development stage
- Based on the windtools-mcp implementation but exploring LanceDB as an alternative approach

## Development Workflow

- Use uv commands in this project
- Use the uv venv in this project

## Guidelines for Modification

- DO NOT ADD FILES, MODIFY THE EXISTING ONES
- Do not make up requirements or features.
- KISS: Keep it Simple, stupid! the simplest possible implementation
- YAGNI: You are not going to need it

## Technology Preferences

- Use polars instead of pandas
