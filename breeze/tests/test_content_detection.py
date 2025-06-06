"""Tests for content-based file detection in file watching."""

import asyncio
import tempfile
import pytest
from pathlib import Path

from breeze.core.engine import BreezeEngine
from breeze.core.config import BreezeConfig


@pytest.mark.asyncio
async def test_file_watcher_content_detection():
    """Test that file watcher uses content detection instead of file extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_content_detection",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        engine = BreezeEngine(config)
        await engine.initialize()
        
        # Create test directory
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        # Add project without specifying file extensions
        project = await engine.add_project(
            name="Test Project",
            paths=[str(project_dir)]
        )
        
        # Track events
        events_received = []
        
        async def event_callback(event):
            events_received.append(event)
        
        # Start watching
        await engine.start_watching(project.id, event_callback)
        
        # Create various test files
        test_files = [
            # Standard extensions
            ("test.py", "def hello():\n    print('hello')\n"),
            ("test.js", "console.log('hello');\n"),
            # No extension but has code content
            ("Makefile", "all:\n\techo 'hello'\n"),
            ("Dockerfile", "FROM python:3.9\nRUN echo 'hello'\n"),
            # Unusual extension but code content
            ("script.sh", "#!/bin/bash\necho 'hello'\n"),
            ("config.yaml", "name: test\nvalue: 123\n"),
            # Binary file (should be skipped)
            ("test.bin", b"\x00\x01\x02\x03"),
            # Text file without code
            ("readme.txt", "This is just plain text\n"),
        ]
        
        for filename, content in test_files:
            file_path = project_dir / filename
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content)
        
        # Wait for file watcher to process
        await asyncio.sleep(3)
        
        # Check that an indexing event was triggered
        indexing_events = [e for e in events_received if e["type"] == "indexing_started"]
        assert len(indexing_events) > 0
        
        # The indexing should include code files regardless of extension
        indexed_files = []
        for event in events_received:
            if event["type"] == "indexing_started" and "files" in event:
                indexed_files.extend(event["files"])
        
        # Convert to just filenames for easier checking
        indexed_filenames = [Path(f).name for f in indexed_files]
        
        # These files should be indexed (they have code content)
        expected_code_files = ["test.py", "test.js", "Makefile", "Dockerfile", "script.sh", "config.yaml"]
        for expected in expected_code_files:
            assert expected in indexed_filenames, f"{expected} should be indexed"
        
        # Binary file should not be indexed
        assert "test.bin" not in indexed_filenames
        
        await engine.shutdown()


@pytest.mark.asyncio
async def test_direct_indexing_content_detection():
    """Test that direct indexing uses content detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BreezeConfig(
            data_root=tmpdir,
            db_name="test_content_detection",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        engine = BreezeEngine(config)
        await engine.initialize()
        
        # Create test files
        test_files = {
            "script": "#!/usr/bin/env python\nprint('hello')\n",  # No extension
            "config.json": '{"name": "test"}\n',  # JSON
            "README": "# Project\nThis is a code project\n",  # Markdown without .md
            "binary.dat": b"\x00\x01\x02\x03",  # Binary
        }
        
        for filename, content in test_files.items():
            file_path = Path(tmpdir) / filename
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content)
        
        # Index the directory
        stats = await engine.index_directories([tmpdir])
        
        # Should index text files but not binary
        assert stats.files_scanned >= 3  # At least the text files
        assert stats.files_indexed >= 3  # Should index script, config.json, and README
        
        # Search for content to verify what was indexed
        results = await engine.search("hello")
        assert len(results) > 0  # Should find the script file
        
        await engine.shutdown()


if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))