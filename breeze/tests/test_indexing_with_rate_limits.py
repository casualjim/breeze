"""Integration test to verify indexing continues after rate limits."""

import asyncio
import tempfile
import os
from unittest.mock import MagicMock, patch
import numpy as np

import pytest

from breeze.core.engine import BreezeEngine
from breeze.core.config import BreezeConfig
from .mock_embedders import MockReranker
from lancedb.embeddings.registry import get_registry


@pytest.mark.asyncio
async def test_indexing_continues_with_rate_limits():
    """Test that indexing continues processing files even when some batches hit rate limits.
    
    This test verifies that:
    1. The engine correctly uses get_voyage_embeddings_with_limits for Voyage models
    2. When some embedding batches fail due to rate limits, the successful batches are still indexed
    3. Failed batches are stored for later retry
    4. The indexing process continues and doesn't fail completely due to partial failures
    """

    # Set dummy API key for Voyage model
    os.environ["BREEZE_EMBEDDING_API_KEY"] = "dummy-key-for-testing"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files - use enough to trigger multiple batches
            # Each file will be a separate document, and we'll create content that's
            # substantial enough to test the batching logic
            num_files = 150  # Should create multiple batches
            for i in range(num_files):
                file_path = os.path.join(tmpdir, f"test_{i}.py")
                with open(file_path, "w") as f:
                    # Create substantial content to make batching more realistic
                    content = f"# Test file {i}\n"
                    content += f"def function_{i}():\n"
                    content += f"    '''Function {i} for testing.'''\n"
                    content += f"    return {i}\n\n"
                    content += f"class Class_{i}:\n"
                    content += f"    def method_{i}(self):\n"
                    content += f"        return 'method_{i}'\n\n"
                    content += "# " + "x" * 200  # Add some bulk to the content
                    f.write(content)

            # Configure engine with mock Voyage embedder that supports rate limiting
            from .mock_embedders import MockVoyageEmbedder
            mock_embedder = MockVoyageEmbedder()
            
            config = BreezeConfig(
                data_root="/tmp",
                db_name="test_rate_limit_indexing",
                voyage_tier=1,
                voyage_concurrent_requests=3,
                voyage_max_retries=2,
                voyage_retry_base_delay=0.01,
                embedding_function=mock_embedder,
            )

            engine = BreezeEngine(config)
            engine.reranker = MockReranker()

            # Patch the get_voyage_embeddings_with_limits function
            with patch(
                "breeze.core.engine.get_voyage_embeddings_with_limits"
            ) as mock_get_embeddings:
                call_count = 0

                async def mock_embeddings_func(*args, **kwargs):  # noqa: ARG001
                    nonlocal call_count
                    call_count += 1
                    print(f"Mock embeddings function called! Call count: {call_count}")

                    texts = args[0]

                    # Simulate some batches failing due to rate limits
                    if call_count in [2, 4]:  # Fail batches 2 and 4 to show partial success
                        print(f"Simulating failure for call {call_count}")
                        # Return partial failure
                        return {
                            "embeddings": np.array(
                                []
                            ),  # No embeddings for failed batch
                            "successful_batches": [],
                            "failed_batches": [0],  # Single batch failed
                            "texts": texts,
                            "safe_batches": [texts],
                        }
                    else:
                        print(f"Simulating success for call {call_count}")
                        # Success - use len(texts) directly
                        # voyage-code-3 uses 1024 dimensions
                        return {
                            "embeddings": np.random.rand(len(texts), 1024),
                            "successful_batches": [0],
                            "failed_batches": [],
                            "texts": texts,
                            "safe_batches": [texts],
                        }

                mock_get_embeddings.side_effect = mock_embeddings_func

                # Configure the mock to fail at specific calls
                mock_embedder.set_rate_limit_behavior([2, 4])
                
                # Initialize the engine (embedding model set via config)
                await engine.initialize()
                
                # Mark as voyage model after initialization
                engine.is_voyage_model = True

                # Run indexing
                stats = await engine.index_directories(
                    [tmpdir],
                    force_reindex=True,
                )

                print("\nIndexing Statistics:")
                print(f"Files scanned: {stats.files_scanned}")
                print(f"Files indexed: {stats.files_indexed}")
                print(f"Files updated: {stats.files_updated}")
                print(f"Files skipped: {stats.files_skipped}")
                print(f"Errors: {stats.errors}")
                print(f"Total API calls: {call_count}")

                # Verify that our mock was called multiple times (showing batching)
                assert call_count > 1, f"Expected multiple API calls for {num_files} files, got {call_count}"
                
                # Verify that some files were indexed successfully
                assert stats.files_scanned > 0, "Should have scanned files"
                assert stats.files_indexed > 0, "Should have indexed some files successfully"
                
                # Verify that some files failed and were marked for retry
                assert stats.files_skipped > 0, "Should have some files marked for retry"
                
                # Not all files should be indexed due to some batch failures
                assert stats.files_indexed < num_files, "Some files should have failed due to rate limits"
                
                # Check that failed batches were stored for retry
                failed_batches_table = await engine.db.open_table("failed_batches")
                failed_count = await failed_batches_table.count_rows()
                assert failed_count > 0, "Failed batches should be stored for retry"
                
                print(f"Successfully indexed: {stats.files_indexed}/{num_files} files")
                print(f"Failed batches stored for retry: {failed_count}")
                print("✓ Test passed: Rate limiting with partial success works correctly")

    finally:
        # Clean up environment variable
        if "BREEZE_EMBEDDING_API_KEY" in os.environ:
            del os.environ["BREEZE_EMBEDDING_API_KEY"]


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

        # Create engine with mock embedder
        registry = get_registry()
        mock_embedder = registry.get("mock-local").create()
        
        config = BreezeConfig(
            data_root="/tmp",
            db_name="test_skip_embedding",
            embedding_function=mock_embedder,
        )

        engine = BreezeEngine(config)
        engine.reranker = MockReranker()

        # Track embedding calls
        embedding_calls = []

        def track_embeddings(texts):
            embedding_calls.append(len(texts))
            return np.random.rand(len(texts), 768)

        mock_model = MagicMock()
        mock_model.compute_source_embeddings = track_embeddings
        engine.embedding_model = mock_model

        # First indexing
        await engine.initialize()
        stats1 = await engine.index_directories(
            directories=[tmpdir], force_reindex=False
        )

        print("\nFirst indexing:")
        print(f"Files indexed: {stats1.files_indexed}")
        print(f"Embedding calls: {len(embedding_calls)}")

        # Reset tracking
        embedding_calls.clear()

        # Second indexing without changes
        stats2 = await engine.index_directories(
            directories=[tmpdir], force_reindex=False
        )

        print("\nSecond indexing (no changes):")
        print(f"Files skipped: {stats2.files_skipped}")
        print(f"Embedding calls: {len(embedding_calls)}")

        # Verify files are skipped on second run
        assert len(embedding_calls) == 0, (
            f"Expected no embedding calls for unchanged files, but got {len(embedding_calls)}"
        )
        assert stats2.files_skipped == 5, (
            f"Expected all 5 files to be skipped, but only {stats2.files_skipped} were skipped"
        )


if __name__ == "__main__":
    asyncio.run(test_indexing_continues_with_rate_limits())
    asyncio.run(test_already_indexed_files_skip_embedding())
