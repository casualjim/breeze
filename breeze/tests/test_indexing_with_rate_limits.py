"""Integration test to verify indexing continues after rate limits."""

import asyncio
import tempfile
import os
from unittest.mock import MagicMock, patch
import numpy as np

import pytest

from breeze.core.engine import BreezeEngine
from breeze.core.config import BreezeConfig


@pytest.mark.asyncio
async def test_indexing_continues_with_rate_limits():
    """Test that indexing continues processing files even when some batches hit rate limits."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        num_files = 300  # Enough to create multiple embedding batches
        for i in range(num_files):
            file_path = os.path.join(tmpdir, f"test_{i}.py")
            with open(file_path, "w") as f:
                f.write(f"# Test file {i}\n" + "x" * 100)
        
        # Configure engine
        config = BreezeConfig(
            data_root="/tmp",
            db_name="test_rate_limit_indexing",
            embedding_model="voyage-code-3",
            voyage_tier=1,
            voyage_concurrent_requests=3,
            voyage_max_retries=2,
            voyage_retry_base_delay=0.01,
        )
        
        engine = BreezeEngine(config)
        
        # Mock the Voyage embedding function
        original_get_embeddings = None
        
        # Patch the get_voyage_embeddings_with_limits function
        with patch("breeze.core.engine.get_voyage_embeddings_with_limits") as mock_get_embeddings:
            call_count = 0
            
            async def mock_embeddings_func(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                texts = args[0]
                num_texts = len(texts)
                
                # Simulate some batches failing due to rate limits
                if call_count in [2, 5]:  # Fail batches 2 and 5
                    # Return partial failure
                    return {
                        'embeddings': np.array([]),  # No embeddings for failed batch
                        'successful_batches': [],
                        'failed_batches': [0],  # Single batch failed
                        'texts': texts,
                        'safe_batches': [texts]
                    }
                else:
                    # Success
                    return {
                        'embeddings': np.random.rand(num_texts, 768),
                        'successful_batches': [0],
                        'failed_batches': [],
                        'texts': texts,
                        'safe_batches': [texts]
                    }
            
            mock_get_embeddings.side_effect = mock_embeddings_func
            
            # Mock standard embedding model
            mock_model = MagicMock()
            mock_model.compute_source_embeddings = lambda texts: np.random.rand(len(texts), 768)
            engine.embedding_model = mock_model
            engine.is_voyage_model = True
            
            # Initialize and run indexing
            await engine.initialize()
            
            # Run indexing
            stats = await engine.index_directories(
                [tmpdir],
                force_reindex=True,
                concurrent_readers=5,
                concurrent_embedders=2,
                concurrent_writers=2,
            )
            
            print(f"\nIndexing Statistics:")
            print(f"Files scanned: {stats.files_scanned}")
            print(f"Files indexed: {stats.files_indexed}")
            print(f"Files updated: {stats.files_updated}")
            print(f"Files skipped: {stats.files_skipped}")
            print(f"Errors: {stats.errors}")
            print(f"Total API calls: {call_count}")
            
            # Verify indexing continued despite rate limits
            # Some files should have been indexed successfully
            assert stats.files_indexed > 0
            
            # Not all files were indexed due to rate limits
            assert stats.files_indexed < num_files
            
            # Check that failed batches were stored for retry
            failed_batches_table = await engine.db.open_table("failed_batches")
            failed_count = await failed_batches_table.count_rows()
            assert failed_count > 0
            
            print(f"Failed batches stored for retry: {failed_count}")


if __name__ == "__main__":
    asyncio.run(test_indexing_continues_with_rate_limits())