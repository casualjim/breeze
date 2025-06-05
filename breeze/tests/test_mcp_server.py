"""Tests for Breeze MCP server."""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from starlette.testclient import TestClient

from breeze.core import BreezeConfig, BreezeEngine
from breeze.mcp.server import create_app


@pytest_asyncio.fixture
async def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest_asyncio.fixture
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


@pytest_asyncio.fixture
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


class TestMCPTools:
    """Test MCP tool functionality directly."""
    
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
    
    @pytest.mark.asyncio
    async def test_project_management_tools(self, test_engine, test_code_dir):
        """Test project registration, listing, and unregistration."""
        from breeze.mcp.server import register_project, list_projects, unregister_project
        
        # Register a project
        result = await register_project(
            name="Test Project",
            paths=[str(test_code_dir)],
            auto_index=False  # Don't auto-index to speed up test
        )
        
        assert result["status"] == "success"
        assert "project_id" in result
        project_id = result["project_id"]
        
        # List projects
        result = await list_projects()
        assert result["status"] == "success"
        assert result["total_projects"] == 1
        assert result["projects"][0]["name"] == "Test Project"
        
        # Unregister the project
        result = await unregister_project(project_id)
        assert result["status"] == "success"
        
        # Verify it's gone
        result = await list_projects()
        assert result["total_projects"] == 0


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