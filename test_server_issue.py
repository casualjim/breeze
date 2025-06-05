#!/usr/bin/env python3
"""Test script to reproduce the server issue using FastMCP client.

First, start the server in another terminal:
    uv run python -m breeze serve

Then run this test:
    uv run python test_server_issue.py
"""

import asyncio
import logging
from pathlib import Path
import httpx

from fastmcp import Client

logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG to see more details
logger = logging.getLogger(__name__)


async def test_raw_http():
    """Test with raw HTTP to see what's happening."""
    logger.info("Testing raw HTTP request to server...")
    
    async with httpx.AsyncClient() as client:
        # First test the health endpoint
        try:
            response = await client.get("http://localhost:9483/health")
            logger.info(f"Health check response: {response.status_code}")
            logger.info(f"Health data: {response.json()}")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        # Test the MCP endpoint with a simple request
        try:
            response = await client.post(
                "http://localhost:9483/mcp/",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "0.1.0",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "test-client",
                            "version": "1.0.0"
                        }
                    }
                },
                headers={"Content-Type": "application/json"}
            )
            logger.info(f"MCP initialize response: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            if response.status_code != 200:
                logger.error(f"Response body: {response.text}")
        except Exception as e:
            logger.error(f"MCP request failed: {e}")


async def test_server_connection():
    """Test connecting to the Breeze MCP server using FastMCP client."""
    logger.info("Testing Breeze MCP server connection...")
    
    # Test with streamable HTTP transport
    try:
        logger.info("Testing HTTP endpoint at http://localhost:9483/mcp/")
        async with Client("http://localhost:9483/mcp/") as client:
            logger.info("Connected successfully!")
            
            # Get server info
            logger.info(f"Server info: {client.server}")
            
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[tool.name for tool in tools]}")
            
            # Test get_index_stats
            logger.info("Testing get_index_stats...")
            result = await client.call_tool(
                "get_index_stats",
                arguments={}
            )
            logger.info(f"Index stats: {result}")
            
            # Test list_directory
            test_dir = Path.home() / "github" / "lancedb" / "lancedb"
            if test_dir.exists():
                logger.info(f"Testing list_directory with {test_dir}...")
                result = await client.call_tool(
                    "list_directory",
                    arguments={"directory_path": str(test_dir)}
                )
                logger.info(f"Directory listing: {result}")
                
                # Test index_repository
                logger.info(f"Testing index_repository with {test_dir}...")
                result = await client.call_tool(
                    "index_repository",
                    arguments={
                        "directories": [str(test_dir)],
                        "force_reindex": False
                    }
                )
                logger.info(f"Indexing result: {result}")
                
                # Test search after indexing
                logger.info("Testing search_code...")
                result = await client.call_tool(
                    "search_code",
                    arguments={
                        "query": "database connection",
                        "limit": 5
                    }
                )
                logger.info(f"Search results: {result}")
            
    except Exception as e:
        logger.error(f"HTTP client test failed: {e}", exc_info=True)
        return False
    
    # Test with SSE transport
    try:
        logger.info("\nTesting SSE endpoint at http://localhost:9483/sse/")
        async with Client("http://localhost:9483/sse/") as client:
            logger.info("Connected successfully!")
            
            # Get server info
            logger.info(f"Server info: {client.server}")
            
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[tool.name for tool in tools]}")
            
            # Test a simple tool
            result = await client.call_tool(
                "get_index_stats",
                arguments={}
            )
            logger.info(f"Index stats: {result}")
            
    except Exception as e:
        logger.error(f"SSE client test failed: {e}", exc_info=True)
        return False
    
    return True


async def main():
    """Run the test."""
    logger.info("Starting Breeze MCP server tests...")
    
    # First test raw HTTP to see what's happening
    await test_raw_http()
    
    # Then test with FastMCP client
    success = await test_server_connection()
    
    if success:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed!")
    
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)