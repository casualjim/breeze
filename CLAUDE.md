# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Breeze is an MCP (Model Context Protocol) server that supports streamable HTTP transport for semantically indexing codebases. It uses LanceDB for vector storage and FastMCP for the server implementation.

## Development Setup

- Python version: 3.13
- Virtual environment: `.venv` directory
- Dependencies: Listed in `requirements.txt` (lancedb, fastmcp)

## Common Commands

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python main.py
```

### Run tests

```bash
pytest
```

## Project Structure

- `main.py` - Entry point for the application
- `pyproject.toml` - Project configuration
- `requirements.txt` - Python dependencies

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

DO NOT ADD FILES, MODIFY THE EXISTING ONES