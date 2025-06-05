#!/usr/bin/env python3
"""Main entry point for Breeze MCP server."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breeze.mcp.server import mcp

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()