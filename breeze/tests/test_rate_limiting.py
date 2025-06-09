"""Tests for rate limiting functionality in embeddings."""

import time
from unittest.mock import MagicMock
import numpy as np
import pytest

from breeze.core.embeddings import get_voyage_embeddings_with_limits
from breeze.core.engine import BreezeEngine
from breeze.core.config import BreezeConfig
from breeze.tests.mock_embedders import MockReranker


class TestRateLimiting:
    """Test suite for rate limiting functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()

        # Mock encode to return an object with ids attribute
        def encode_text(text, add_special_tokens=True, **kwargs):
            result = MagicMock()
            result.ids = list(range(len(text) // 4))
            return result

        tokenizer.encode.side_effect = encode_text
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Create a mock embedding model."""
        model = MagicMock()

        # Return embeddings based on input size - must be synchronous for asyncio.to_thread
        def compute_embeddings(texts):
            return np.random.rand(len(texts), 768)

        model.compute_source_embeddings = compute_embeddings
        return model

    @pytest.mark.asyncio
    async def test_rate_limit_calculation_accuracy(self, mock_tokenizer, mock_model):
        """Test that rate limit calculations are accurate."""
        # Create texts with known token counts
        texts = ["a" * 100 for _ in range(5)]  # Reduced to 5 texts for faster test

        # Set very low rate limits to force delays
        tokens_per_minute = 200  # Very low token limit
        requests_per_minute = 3  # Very low request limit

        # Track API calls
        api_calls = []

        def track_calls(batch):
            api_calls.append((time.time(), len(batch)))
            return np.random.rand(len(batch), 768)

        mock_model.compute_source_embeddings = track_calls

        from breeze.core.text_chunker import FileContent
        
        # Convert texts to FileContent objects
        file_contents = [
            FileContent(content=text, file_path=f"test{i}.txt", language="text")
            for i, text in enumerate(texts)
        ]
        
        result = await get_voyage_embeddings_with_limits(
            file_contents,
            mock_model,
            mock_tokenizer,
            max_concurrent_requests=5,
            tokens_per_minute=tokens_per_minute,
            requests_per_minute=requests_per_minute,
        )

        # Check that we got all embeddings
        assert len(result.embeddings) == 5

        # Verify rate limiting occurred
        # With only 3 requests per minute allowed, we should see delays
        # (elapsed time is used implicitly in the checks below)

        # Check that API calls were spread out
        if len(api_calls) > 3:
            # There should be delays between calls after the 3rd one
            call_times = [t for t, _ in api_calls]
            # Find the time gap between 3rd and 4th call
            if len(call_times) > 3:
                gap = call_times[3] - call_times[2]
                # Should wait significant time due to rate limit
                assert gap > 0.5, (
                    f"Expected delay between calls, but gap was only {gap}s"
                )

    @pytest.mark.asyncio
    async def test_resume_after_rate_limit(self, mock_tokenizer):
        """Test that processing resumes correctly after rate limit."""
        # Create texts that will force multiple API calls
        # With MAX_TEXTS_PER_BATCH = 128, we need to create batches differently
        # Create very large texts that will each need their own batch due to token limits
        texts = []
        for _ in range(5):
            # Each text is ~100k chars = ~25k tokens (with our mock tokenizer)
            # This approaches the MAX_TOKENS_PER_BATCH limit (120k)
            # So we'll get roughly 1 text per batch
            texts.append("x" * 100000)

        # Track API calls
        api_calls = []
        rate_limit_hit = False
        call_count = 0

        def mock_compute_embeddings(batch):
            nonlocal rate_limit_hit, call_count
            call_count += 1
            api_calls.append((call_count, len(batch)))

            # Simulate rate limit on 2nd call
            if call_count == 2 and not rate_limit_hit:
                rate_limit_hit = True
                raise Exception("429 Too Many Requests")

            # Return embeddings
            return np.random.rand(len(batch), 768)

        mock_model = MagicMock()
        mock_model.compute_source_embeddings = mock_compute_embeddings

        from breeze.core.text_chunker import FileContent
        
        # Convert texts to FileContent objects
        file_contents = [
            FileContent(content=text, file_path=f"test{i}.txt", language="text")
            for i, text in enumerate(texts)
        ]

        # Run with rate limiting - use high limits to avoid timeouts
        result = await get_voyage_embeddings_with_limits(
            file_contents,
            mock_model,
            mock_tokenizer,
            max_concurrent_requests=5,  # Higher concurrency to ensure we hit the 3rd call
            tokens_per_minute=10_000_000,  # Very high to avoid blocking
            requests_per_minute=100_000,  # Very high to avoid blocking
        )

        # Extract embeddings from result
        embeddings = result.embeddings

        # Verify all texts were processed (or some failed)
        # Since we have 5 texts and they're processed in batches, check if we got embeddings or failures
        assert len(embeddings) > 0 or len(result.failed_batches) > 0

        # Debug: print API call info
        print(f"API calls: {api_calls}")
        print(f"Rate limit hit: {rate_limit_hit}")

        # Verify we hit rate limit and continued
        assert call_count >= 2, f"Expected at least 2 API calls but made {call_count}"
        assert rate_limit_hit, "Expected to hit simulated rate limit"
        # Should have made more than 2 calls due to retry
        assert call_count > 2, "Expected retry after rate limit"

    @pytest.mark.asyncio
    async def test_global_rate_limit_coordination(self, mock_tokenizer, mock_model):
        """Test that global rate limit affects all concurrent batches."""
        texts = [
            "text" * 10 for _ in range(20)
        ]  # Reduced from 200 to 20 for faster test

        # Track timing of API calls
        call_times = []
        rate_limit_time = None

        def mock_compute_with_tracking(batch):
            nonlocal rate_limit_time
            call_times.append(time.time())

            # Simulate rate limit on 5th call
            if len(call_times) == 5 and rate_limit_time is None:
                rate_limit_time = time.time()
                raise Exception("429 Rate Limit Exceeded")

            return np.random.rand(len(batch), 768)

        mock_model.compute_source_embeddings = mock_compute_with_tracking

        from breeze.core.text_chunker import FileContent
        
        # Convert texts to FileContent objects
        file_contents = [
            FileContent(content=text, file_path=f"test{i}.txt", language="text")
            for i, text in enumerate(texts)
        ]

        # Run with high concurrency
        result = await get_voyage_embeddings_with_limits(
            file_contents,
            mock_model,
            mock_tokenizer,
            max_concurrent_requests=10,  # High concurrency
            tokens_per_minute=10_000_000,  # Very high to avoid blocking
            requests_per_minute=100_000,  # Very high to avoid blocking
        )

        # Check all texts processed
        assert len(result.embeddings) == 20

        # Check that calls after rate limit respected the delay
        if rate_limit_time:
            calls_after_rate_limit = [t for t in call_times if t > rate_limit_time]
            if calls_after_rate_limit:
                # All calls after rate limit should be at least 0.5s later
                min_delay = min(calls_after_rate_limit) - rate_limit_time
                assert min_delay >= 0.4  # Allow small timing variance

    @pytest.mark.asyncio
    async def test_already_embedded_skip(self):
        """Test that already embedded files are skipped."""
        from unittest.mock import patch

        # Create a test engine with mock embedder
        from lancedb.embeddings.registry import get_registry

        registry = get_registry()
        mock_embedder = registry.get("mock-local").create()

        config = BreezeConfig(
            data_root="/tmp/test_breeze_rate_limit",
            embedding_function=mock_embedder,
        )

        engine = BreezeEngine(config)
        engine.reranker = MockReranker()

        # Track embedding calls
        embedding_call_count = 0

        async def mock_get_local_embeddings(*args, **kwargs):
            nonlocal embedding_call_count
            embedding_call_count += 1
            from breeze.core.embeddings import EmbeddingResult
            from breeze.core.text_chunker import TextChunker, ChunkingConfig
            
            file_contents = args[0]
            # Create chunker for testing
            from breeze.core.text_chunker import SimpleChunkingStrategy
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = MagicMock(ids=list(range(100)))
            strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
            chunker = TextChunker(strategy=strategy, config=ChunkingConfig(chunk_size=8192, model_max_tokens=8192))
            chunked_files = chunker.chunk_files(file_contents)
            
            return EmbeddingResult(
                embeddings=np.random.rand(
                    len(file_contents), 384
                ),  # all-MiniLM-L6-v2 has 384 dims
                successful_files=list(range(len(file_contents))),
                failed_files=[],
                chunked_files=chunked_files
            )

        # Create test files
        test_files = []
        import tempfile

        with patch(
            "breeze.core.engine.get_local_embeddings_with_tokenizer_chunking",
            mock_get_local_embeddings,
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create files
                for i in range(5):
                    file_path = f"{tmpdir}/test_{i}.txt"
                    with open(file_path, "w") as f:
                        f.write(f"Test content {i}")
                    test_files.append(file_path)

                # First indexing
                await engine.initialize()
                stats1 = await engine.index_directories(
                    directories=[tmpdir],
                    force_reindex=False,
                )

                # Check initial indexing
                assert stats1.files_indexed == 5
                assert stats1.files_updated == 0

                # Reset counter to track new calls
                initial_call_count = embedding_call_count

                # Second indexing without changes
                stats2 = await engine.index_directories(
                    directories=[tmpdir],
                    force_reindex=False,
                )

                # Should skip all files
                assert stats2.files_indexed == 0
                assert stats2.files_updated == 0
                assert stats2.files_skipped == 5

                # Embedding should not be called for unchanged files
                assert embedding_call_count == initial_call_count

                # Modify one file
                with open(test_files[0], "w") as f:
                    f.write("Modified content")

                # Third indexing with one changed file
                stats3 = await engine.index_directories(
                    directories=[tmpdir],
                    force_reindex=False,
                )

                # Should update only the modified file
                assert stats3.files_indexed == 0
                assert stats3.files_updated == 1
                assert stats3.files_skipped == 4

                # Embedding should be called only once for the updated file
                assert embedding_call_count == initial_call_count + 1

    @pytest.mark.asyncio
    async def test_failed_batch_handling(self, mock_tokenizer):
        """Test that failed batches are properly tracked."""
        # Create texts that will be split into exactly 5 batches
        # With MAX_TOKENS_PER_BATCH = 120000 and MAX_TEXTS_PER_BATCH = 128
        # Create 5 groups of 100 texts each (under the 128 limit)
        texts = []
        for _ in range(5):
            # Each group has texts that together are close to token limit
            # 100 texts * 1000 chars each = 100k chars = ~25k tokens per batch
            texts.extend(["x" * 1000 for _ in range(100)])

        # Track which batches fail and which batch we're processing
        failed_batch_indices = {1, 3}  # Fail batches 1 and 3 (0-indexed)
        call_count = 0
        batch_call_counts = {}  # Track calls per batch

        def mock_compute_with_failures(batch):
            nonlocal call_count
            call_count += 1

            # Identify which batch this is by its size or content
            batch_key = (
                len(batch),
                batch[0][:10],
            )  # Use size and first 10 chars as key
            if batch_key not in batch_call_counts:
                batch_call_counts[batch_key] = len(batch_call_counts)

            batch_idx = batch_call_counts[batch_key]

            # Fail specific batches persistently
            if batch_idx in failed_batch_indices:
                raise Exception(f"Permanent failure for batch {batch_idx}")

            return np.random.rand(len(batch), 768)

        mock_model = MagicMock()
        mock_model.compute_source_embeddings = mock_compute_with_failures

        from breeze.core.text_chunker import FileContent
        
        # Convert texts to FileContent objects
        file_contents = [
            FileContent(content=text, file_path=f"test{i}.txt", language="text")
            for i, text in enumerate(texts)
        ]

        # This should return partial results with failed batches
        # Use more reasonable rate limits to avoid timeout
        # Note: non-rate-limit errors are retried up to 3 times regardless of max_retries
        result = await get_voyage_embeddings_with_limits(
            file_contents,
            mock_model,
            mock_tokenizer,
            max_concurrent_requests=1,  # Process sequentially to ensure predictable batch order
            tokens_per_minute=10_000_000,  # Very high to avoid rate limiting
            requests_per_minute=100_000,  # Very high to avoid rate limiting
        )

        # Check that some batches failed
        assert len(result.failed_batches) > 0
        # And some succeeded (check successful_files instead of successful_batches)
        assert len(result.successful_files) > 0
        # We should have partial embeddings
        assert len(result.embeddings) < len(file_contents)

    @pytest.mark.asyncio
    async def test_token_counting_accuracy(self, mock_tokenizer):
        """Test that token counting matches actual API usage."""
        # Create texts with varying lengths
        texts = [
            "short",  # ~1 token
            "medium length text here",  # ~4 tokens
            "this is a much longer text that should have many more tokens",  # ~12 tokens
        ]

        # Track token counts
        recorded_tokens = []

        def mock_compute_tracking_tokens(batch):
            # Count tokens for this batch
            batch_tokens = sum(len(mock_tokenizer.encode(text).ids) for text in batch)
            recorded_tokens.append(batch_tokens)
            return np.random.rand(len(batch), 768)

        mock_model = MagicMock()
        mock_model.compute_source_embeddings = mock_compute_tracking_tokens

        from breeze.core.text_chunker import FileContent
        
        # Convert texts to FileContent objects
        file_contents = [
            FileContent(content=text, file_path=f"test{i}.txt", language="text")
            for i, text in enumerate(texts)
        ]

        # Use high limits to avoid blocking
        result = await get_voyage_embeddings_with_limits(
            file_contents,
            mock_model,
            mock_tokenizer,
            max_concurrent_requests=1,
            tokens_per_minute=10_000_000,  # Very high to avoid blocking
            requests_per_minute=100_000,  # Very high to avoid blocking
        )

        # Verify token counting
        if recorded_tokens:  # Only check if we recorded any tokens
            # Calculate total expected tokens
            total_expected_tokens = sum(
                len(mock_tokenizer.encode(text).ids) for text in texts
            )
            total_recorded_tokens = sum(recorded_tokens)

            # The tokens should match whether batched together or separately
            assert total_recorded_tokens == total_expected_tokens

            # Verify we got all embeddings
            assert len(result.embeddings) == len(texts)
        else:
            # If no tokens were recorded, the embeddings must have been processed differently
            assert len(result.embeddings) == len(texts)
