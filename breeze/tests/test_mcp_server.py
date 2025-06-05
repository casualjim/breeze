"""Integration tests for Breeze MCP server."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from fastmcp import FastMCP
from mcp.client import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.http import http_client
from starlette.testclient import TestClient

from breeze.core import BreezeConfig, BreezeEngine
from breeze.mcp.server import create_app, get_engine, mcp


@pytest.fixture
async def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
async def test_code_dir(temp_data_dir):
    """Create test code files in a temporary directory."""
    test_dir = Path(temp_data_dir) / "test_code"
    test_dir.mkdir()
    
    # Create test files
    (test_dir / "hello.py").write_text('''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")
    return "Hello"

def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"
''')
    
    (test_dir / "math_utils.py").write_text('''
def factorial(n: int) -> int:
    """Calculate the factorial of n."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
''')
    
    return test_dir


@pytest.fixture
async def test_engine(temp_data_dir):
    """Create a test engine with a temporary database."""
    config = BreezeConfig(
        data_root=temp_data_dir,
        db_name="test_index",
        embedding_model="nomic-ai/CodeRankEmbed",
        trust_remote_code=True,
    )
    engine = BreezeEngine(config)
    await engine.initialize()
    
    # Reset the global engine to use our test engine
    import breeze.mcp.server
    breeze.mcp.server.engine = engine
    
    yield engine
    
    # Clean up
    breeze.mcp.server.engine = None


@pytest.fixture
def test_app(test_engine):
    """Create a test app instance."""
    return create_app()


@pytest.fixture
def test_client(test_app):
    """Create a test client for the app."""
    return TestClient(test_app)


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_check(self, test_client):
        """Test that health check returns correct status."""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["server"] == "breeze-mcp"
        assert data["version"] == "0.1.0"
        assert "database" in data
        assert data["database"]["initialized"] is True


class TestSSEEndpoint:
    """Test the SSE (Server-Sent Events) endpoint."""
    
    @pytest.mark.asyncio
    async def test_sse_client_connection(self, test_engine):
        """Test SSE client connection and initialization."""
        # Create SSE client URL
        base_url = "http://localhost:9483"
        
        async with sse_client(f"{base_url}/sse/") as (read, write):
            # Create client session
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                assert len(tools) == 4  # We have 4 tools defined
                
                tool_names = {tool.name for tool in tools}
                assert "index_repository" in tool_names
                assert "search_code" in tool_names
                assert "get_index_stats" in tool_names
                assert "list_directory" in tool_names
    
    @pytest.mark.asyncio
    async def test_sse_tool_execution(self, test_engine, test_code_dir):
        """Test executing tools via SSE client."""
        base_url = "http://localhost:9483"
        
        async with sse_client(f"{base_url}/sse/") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Test get_index_stats tool
                result = await session.call_tool(
                    "get_index_stats",
                    arguments={}
                )
                assert result[0].content[0].text["status"] == "success"
                assert result[0].content[0].text["total_documents"] == 0
                
                # Test index_repository tool
                result = await session.call_tool(
                    "index_repository",
                    arguments={
                        "directories": [str(test_code_dir)],
                        "force_reindex": False
                    }
                )
                assert result[0].content[0].text["status"] == "success"
                
                # Test search_code tool
                result = await session.call_tool(
                    "search_code",
                    arguments={
                        "query": "factorial",
                        "limit": 5
                    }
                )
                assert result[0].content[0].text["status"] == "success"
                assert result[0].content[0].text["total_results"] > 0


class TestStreamableHTTPEndpoint:
    """Test the streamable HTTP endpoint."""
    
    @pytest.mark.asyncio
    async def test_http_client_connection(self, test_engine):
        """Test HTTP client connection and initialization."""
        base_url = "http://localhost:9483"
        
        async with http_client(f"{base_url}/mcp/") as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # Get server info
                server_info = session.server
                assert server_info.name == "breeze"
                assert server_info.version == "0.1.0"
    
    @pytest.mark.asyncio
    async def test_http_tool_execution(self, test_engine, test_code_dir):
        """Test executing tools via HTTP client."""
        base_url = "http://localhost:9483"
        
        async with http_client(f"{base_url}/mcp/") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Test list_directory tool
                result = await session.call_tool(
                    "list_directory",
                    arguments={"directory_path": str(test_code_dir)}
                )
                assert result[0].content[0].text["status"] == "success"
                assert result[0].content[0].text["total_items"] == 2
                
                # Test index_repository with progress tracking
                result = await session.call_tool(
                    "index_repository",
                    arguments={
                        "directories": [str(test_code_dir)],
                        "force_reindex": True
                    }
                )
                assert result[0].content[0].text["status"] == "success"
                stats = result[0].content[0].text["statistics"]
                assert stats["files_indexed"] == 2


class TestMCPTools:
    """Test MCP tool functionality."""
    
    @pytest.mark.asyncio
    async def test_index_repository_tool(self, test_engine, test_code_dir):
        """Test the index_repository tool."""
        from breeze.mcp.server import index_repository
        
        result = await index_repository(
            directories=[str(test_code_dir)],
            force_reindex=False
        )
        
        assert result["status"] == "success"
        assert "statistics" in result
        stats = result["statistics"]
        assert stats["files_indexed"] == 2  # hello.py and math_utils.py
        assert stats["files_scanned"] == 2
        assert stats["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_search_code_tool(self, test_engine, test_code_dir):
        """Test the search_code tool."""
        from breeze.mcp.server import index_repository, search_code
        
        # First index the repository
        await index_repository(
            directories=[str(test_code_dir)],
            force_reindex=False
        )
        
        # Then search for factorial
        result = await search_code(
            query="factorial function",
            limit=5,
            min_relevance=0.0
        )
        
        assert result["status"] == "success"
        assert result["total_results"] > 0
        assert len(result["results"]) > 0
        
        # Check that we found the factorial function
        found_factorial = False
        for res in result["results"]:
            if "factorial" in res["snippet"]:
                found_factorial = True
                break
        assert found_factorial
    
    @pytest.mark.asyncio
    async def test_get_index_stats_tool(self, test_engine, test_code_dir):
        """Test the get_index_stats tool."""
        from breeze.mcp.server import index_repository, get_index_stats
        
        # Get stats before indexing
        stats_before = await get_index_stats()
        assert stats_before["status"] == "success"
        assert stats_before["total_documents"] == 0
        
        # Index repository
        await index_repository(
            directories=[str(test_code_dir)],
            force_reindex=False
        )
        
        # Get stats after indexing
        stats_after = await get_index_stats()
        assert stats_after["status"] == "success"
        assert stats_after["total_documents"] == 2
        assert stats_after["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_list_directory_tool(self, test_code_dir):
        """Test the list_directory tool."""
        from breeze.mcp.server import list_directory
        
        result = await list_directory(directory_path=str(test_code_dir))
        
        assert result["status"] == "success"
        assert result["total_items"] == 2
        
        # Check that both files are listed
        file_names = {item["name"] for item in result["contents"]}
        assert "hello.py" in file_names
        assert "math_utils.py" in file_names


class TestAsyncStreamHandling:
    """Test async stream handling to reproduce and fix the reported error."""
    
    @pytest.mark.asyncio
    async def test_concurrent_sse_requests(self, test_engine):
        """Test handling multiple concurrent SSE requests."""
        base_url = "http://localhost:9483"
        
        # Create multiple concurrent SSE connections
        async def run_client(client_id: int):
            async with sse_client(f"{base_url}/sse/") as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Each client makes multiple requests
                    for i in range(3):
                        result = await session.call_tool(
                            "get_index_stats",
                            arguments={}
                        )
                        assert result[0].content[0].text["status"] == "success"
                        await asyncio.sleep(0.01)  # Small delay
        
        # Run multiple clients concurrently
        tasks = [run_client(i) for i in range(5)]
        await asyncio.gather(*tasks)
    
    @pytest.mark.asyncio
    async def test_long_running_operation(self, test_engine, test_code_dir):
        """Test handling long-running operations that might cause timeouts."""
        base_url = "http://localhost:9483"
        
        # Create a larger test directory with more files
        large_test_dir = Path(test_code_dir).parent / "large_test"
        large_test_dir.mkdir(exist_ok=True)
        
        # Create 50 test files
        for i in range(50):
            (large_test_dir / f"file_{i}.py").write_text(f'''
# File {i}
def function_{i}(x):
    """Function {i} documentation."""
    return x * {i}

class Class_{i}:
    """Class {i} documentation."""
    def method(self):
        return "result_{i}"
''')
        
        async with http_client(f"{base_url}/mcp/") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Index the large directory
                result = await session.call_tool(
                    "index_repository",
                    arguments={
                        "directories": [str(large_test_dir)],
                        "force_reindex": True
                    }
                )
                
                assert result[0].content[0].text["status"] == "success"
                stats = result[0].content[0].text["statistics"]
                assert stats["files_indexed"] == 50


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_directory_indexing(self, test_engine):
        """Test indexing with invalid directory."""
        from breeze.mcp.server import index_repository
        
        result = await index_repository(
            directories=["/nonexistent/directory/path"],
            force_reindex=False
        )
        
        assert result["status"] == "error"
        assert "No valid directories provided" in result["message"]
    
    @pytest.mark.asyncio
    async def test_search_without_index(self, test_engine):
        """Test searching when no documents are indexed."""
        from breeze.mcp.server import search_code
        
        result = await search_code(
            query="test query",
            limit=10,
            min_relevance=0.0
        )
        
        assert result["status"] == "error"
        assert "No code has been indexed yet" in result["message"]
    
    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self):
        """Test listing a non-existent directory."""
        from breeze.mcp.server import list_directory
        
        result = await list_directory(
            directory_path="/this/directory/does/not/exist"
        )
        
        assert result["status"] == "error"
        assert "does not exist" in result["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])