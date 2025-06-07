"""Tests for project management functionality in BreezeEngine."""

import asyncio
import tempfile
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from breeze.core.engine import BreezeEngine
from breeze.core.config import BreezeConfig
from breeze.core.models import Project
from lancedb.embeddings.registry import get_registry
from .mock_embedders import MockReranker


@pytest.mark.asyncio
async def test_project_table_initialization():
    """Test that project table is created correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Table should be created during engine initialization
        assert engine.projects_table is None  # Not created until first use
        
        # Initialize project table explicitly
        await engine.init_project_table()
        assert engine.projects_table is not None
        
        # Should be idempotent
        await engine.init_project_table()
        assert engine.projects_table is not None
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_add_project():
    """Test adding a new project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Create test directories
        project_dir1 = Path(tmpdir) / "project1"
        project_dir2 = Path(tmpdir) / "project2"
        project_dir1.mkdir()
        project_dir2.mkdir()
        
        # Add a project
        project = await engine.add_project(
            name="Test Project",
            paths=[str(project_dir1), str(project_dir2)],
            file_extensions=[".py", ".js"],
            exclude_patterns=["node_modules", "__pycache__"],
            auto_index=False
        )
        
        assert project.name == "Test Project"
        assert len(project.paths) == 2
        # Use resolved paths for comparison (macOS has symlinks)
        resolved_paths = [str(Path(p).resolve()) for p in project.paths]
        assert str(project_dir1.resolve()) in resolved_paths
        assert str(project_dir2.resolve()) in resolved_paths
        # file_extensions is deprecated but kept if explicitly provided
        assert project.file_extensions == [".py", ".js"]
        assert project.exclude_patterns == ["node_modules", "__pycache__"]
        assert project.auto_index is False
        assert project.is_watching is False
        assert project.last_indexed is None
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_add_project_invalid_path():
    """Test adding a project with invalid paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Try to add project with non-existent path
        with pytest.raises(ValueError, match="Path does not exist"):
            await engine.add_project(
                name="Invalid Project",
                paths=["/non/existent/path"]
            )
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_add_project_duplicate_path():
    """Test that duplicate paths across projects are prevented."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Create test directory
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        # Add first project
        await engine.add_project(
            name="Project 1",
            paths=[str(project_dir)]
        )
        
        # Try to add second project with same path
        with pytest.raises(ValueError, match="already tracked by project"):
            await engine.add_project(
                name="Project 2",
                paths=[str(project_dir)]
            )
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_list_projects():
    """Test listing all projects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Initially empty
        projects = await engine.list_projects()
        assert len(projects) == 0
        
        # Add some projects
        for i in range(3):
            project_dir = Path(tmpdir) / f"project{i}"
            project_dir.mkdir()
            await engine.add_project(
                name=f"Project {i}",
                paths=[str(project_dir)]
            )
        
        # List projects
        projects = await engine.list_projects()
        assert len(projects) == 3
        
        # Verify all projects are returned
        names = {p.name for p in projects}
        assert names == {"Project 0", "Project 1", "Project 2"}
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_get_project():
    """Test getting a project by ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Create and add a project
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        added_project = await engine.add_project(
            name="Test Project",
            paths=[str(project_dir)]
        )
        
        # Get project by ID
        retrieved_project = await engine.get_project(added_project.id)
        assert retrieved_project is not None
        assert retrieved_project.id == added_project.id
        assert retrieved_project.name == "Test Project"
        assert retrieved_project.paths == added_project.paths
        
        # Try to get non-existent project
        non_existent = await engine.get_project("non-existent-id")
        assert non_existent is None
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_remove_project():
    """Test removing a project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Create and add a project
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        project = await engine.add_project(
            name="Test Project",
            paths=[str(project_dir)]
        )
        
        # Verify project exists
        assert await engine.get_project(project.id) is not None
        
        # Remove project
        success = await engine.remove_project(project.id)
        assert success is True
        
        # Verify project is gone
        assert await engine.get_project(project.id) is None
        
        # Try to remove non-existent project
        success = await engine.remove_project("non-existent-id")
        assert success is False
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_update_project_indexed_time():
    """Test updating project's last indexed time."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Create and add a project
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        project = await engine.add_project(
            name="Test Project",
            paths=[str(project_dir)]
        )
        
        # Initially no last_indexed time
        assert project.last_indexed is None
        original_updated_at = project.updated_at
        
        # Wait a bit to ensure time difference
        await asyncio.sleep(0.1)
        
        # Update indexed time
        await engine.update_project_indexed_time(project.id)
        
        # Retrieve and verify
        updated_project = await engine.get_project(project.id)
        assert updated_project.last_indexed is not None
        assert updated_project.updated_at > original_updated_at
        assert (datetime.now() - updated_project.last_indexed) < timedelta(seconds=1)
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_file_watching_lifecycle():
    """Test starting and stopping file watching for a project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Create and add a project
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        project = await engine.add_project(
            name="Test Project",
            paths=[str(project_dir)]
        )
        
        # Track events
        events_received = []
        
        async def event_callback(event):
            events_received.append(event)
        
        # Start watching
        success = await engine.start_watching(project.id, event_callback)
        assert success is True
        assert project.id in engine._watchers
        assert project.id in engine._observers
        
        # Should receive watching_started event
        await asyncio.sleep(0.1)  # Give time for event
        assert len(events_received) == 1
        assert events_received[0]["type"] == "watching_started"
        assert events_received[0]["project_id"] == project.id
        
        # Verify project is marked as watching
        updated_project = await engine.get_project(project.id)
        assert updated_project.is_watching is True
        
        # Try to start watching again (should succeed but warn)
        success = await engine.start_watching(project.id)
        assert success is True
        
        # Stop watching
        success = await engine.stop_watching(project.id)
        assert success is True
        assert project.id not in engine._watchers
        assert project.id not in engine._observers
        
        # Verify project is no longer watching
        updated_project = await engine.get_project(project.id)
        assert updated_project.is_watching is False
        
        # Try to stop watching again (should fail)
        success = await engine.stop_watching(project.id)
        assert success is False
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_get_watching_projects():
    """Test getting all projects currently being watched."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Create multiple projects
        projects = []
        for i in range(3):
            project_dir = Path(tmpdir) / f"project{i}"
            project_dir.mkdir()
            
            project = await engine.add_project(
                name=f"Project {i}",
                paths=[str(project_dir)]
            )
            projects.append(project)
        
        # Initially no projects are being watched
        watching = await engine.get_watching_projects()
        assert len(watching) == 0
        
        # Start watching first two projects
        await engine.start_watching(projects[0].id)
        await engine.start_watching(projects[1].id)
        
        # Verify watching projects
        watching = await engine.get_watching_projects()
        assert len(watching) == 2
        watching_ids = {p.id for p in watching}
        assert watching_ids == {projects[0].id, projects[1].id}
        
        # Stop watching one project
        await engine.stop_watching(projects[0].id)
        
        # Verify updated watching list
        watching = await engine.get_watching_projects()
        assert len(watching) == 1
        assert watching[0].id == projects[1].id
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_project_with_file_watching_and_remove():
    """Test that removing a project stops its file watching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Create and add a project
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        project = await engine.add_project(
            name="Test Project",
            paths=[str(project_dir)]
        )
        
        # Start watching
        await engine.start_watching(project.id)
        assert project.id in engine._watchers
        
        # Remove project (should stop watching)
        success = await engine.remove_project(project.id)
        assert success is True
        assert project.id not in engine._watchers
        assert project.id not in engine._observers
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_project_default_values():
    """Test that projects get sensible default values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_projects",
            embedding_function=get_registry().get("mock-local").create(),  # Use fast mock embedder
            code_extensions=[".py", ".js", ".ts"],
            exclude_patterns=["node_modules", "__pycache__", ".git"]
        )
        
        engine = BreezeEngine(config)
        engine.reranker = MockReranker()
        await engine.initialize()
        
        # Create project with minimal args
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        project = await engine.add_project(
            name="Test Project",
            paths=[str(project_dir)]
            # Not specifying file_extensions or exclude_patterns
        )
        
        # file_extensions is deprecated and should be None when not specified
        assert project.file_extensions is None
        # exclude_patterns should use config defaults
        assert project.exclude_patterns == ["node_modules", "__pycache__", ".git"]
        assert project.auto_index is True  # Default value
        
        await engine.shutdown()


if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))