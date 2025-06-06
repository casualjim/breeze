"""Test to reproduce the rate limiting issue where indexing stops after hitting a rate limit."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from breeze.core.engine import BreezeEngine
from breeze.core.config import BreezeConfig


@pytest.mark.asyncio
async def test_indexing_continues_after_rate_limit():
    """Test that indexing continues after hitting rate limit instead of stopping."""
    import tempfile
    import os
    
    # Create test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create many files to ensure we hit rate limits
        num_files = 100
        for i in range(num_files):
            file_path = os.path.join(tmpdir, f"test_{i}.py")
            with open(file_path, "w") as f:
                # Write substantial content to each file
                f.write(f"# File {i}\n" + "x" * 1000 + "\n")
        
        # Create engine with mocked embedding model
        config = BreezeConfig(
            data_root="/tmp",
            db_name="test_rate_limit_issue",
            embedding_model="voyage-code-3",
            voyage_tier=1,  # Lowest tier for strictest limits
        )
        
        engine = BreezeEngine(config)
        
        # Track embedding calls
        embedding_calls = []
        rate_limit_count = 0
        
        def mock_compute_embeddings(texts):
            nonlocal rate_limit_count
            embedding_calls.append(len(texts))
            
            # Simulate rate limit after processing some files
            if len(embedding_calls) == 5 and rate_limit_count == 0:
                rate_limit_count += 1
                raise Exception("429 Rate Limit Exceeded")
            
            return np.random.rand(len(texts), 768)
        
        # Mock the Voyage model
        mock_model = MagicMock()
        mock_model.compute_source_embeddings = mock_compute_embeddings
        engine.embedding_model = mock_model
        engine.is_voyage_model = True
        
        # Initialize tables
        await engine.init_tables()
        
        # Run indexing
        stats = await engine.index_directory(
            tmpdir,
            force_reindex=True,
            concurrent_readers=2,
            concurrent_embedders=2,
            concurrent_writers=1,
        )
        
        # Verify results
        print(f"Files scanned: {stats.files_scanned}")
        print(f"Files indexed: {stats.files_indexed}")
        print(f"Files updated: {stats.files_updated}")
        print(f"Errors: {stats.errors}")
        print(f"Total embedding calls: {len(embedding_calls)}")
        print(f"Rate limit hit count: {rate_limit_count}")
        
        # The issue: When rate limit is hit, indexing stops prematurely
        # Expected: All files should be processed (with retries)
        # Actual: Only files before rate limit are processed
        
        # This assertion will likely fail, demonstrating the issue
        assert stats.files_indexed + stats.files_updated >= num_files - 20, \
            f"Expected most files to be indexed, but only {stats.files_indexed + stats.files_updated} out of {num_files} were processed"


@pytest.mark.asyncio 
async def test_already_indexed_files_skip_embedding():
    """Test that already indexed files skip embedding generation."""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_files = []
        for i in range(5):
            file_path = os.path.join(tmpdir, f"test_{i}.py")
            with open(file_path, "w") as f:
                f.write(f"# Test file {i}\nprint('hello')\n")
            test_files.append(file_path)
        
        # Create engine
        config = BreezeConfig(
            data_root="/tmp",
            db_name="test_skip_embedding",
            embedding_model="test-model",
        )
        
        engine = BreezeEngine(config)
        
        # Track embedding calls
        embedding_calls = []
        
        def track_embeddings(texts):
            embedding_calls.append(len(texts))
            return np.random.rand(len(texts), 768)
        
        mock_model = MagicMock()
        mock_model.compute_source_embeddings = track_embeddings
        engine.embedding_model = mock_model
        
        # First indexing
        await engine.init_tables()
        stats1 = await engine.index_directory(tmpdir, force_reindex=False)
        
        print(f"\nFirst indexing:")
        print(f"Files indexed: {stats1.files_indexed}")
        print(f"Embedding calls: {len(embedding_calls)}")
        
        # Reset tracking
        embedding_calls.clear()
        
        # Second indexing without changes
        stats2 = await engine.index_directory(tmpdir, force_reindex=False)
        
        print(f"\nSecond indexing (no changes):")
        print(f"Files skipped: {stats2.files_skipped}")
        print(f"Embedding calls: {len(embedding_calls)}")
        
        # The issue: Files might be re-embedded even when unchanged
        assert len(embedding_calls) == 0, \
            f"Expected no embedding calls for unchanged files, but got {len(embedding_calls)}"
        assert stats2.files_skipped == 5, \
            f"Expected all 5 files to be skipped, but only {stats2.files_skipped} were skipped"


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_indexing_continues_after_rate_limit())
    asyncio.run(test_already_indexed_files_skip_embedding())