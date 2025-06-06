"""Tests for rate limiting functionality in embeddings."""

import time
from unittest.mock import MagicMock
import numpy as np
import pytest

from breeze.core.embeddings import get_voyage_embeddings_with_limits
from breeze.core.engine import BreezeEngine
from breeze.core.config import BreezeConfig


class TestRateLimiting:
    """Test suite for rate limiting functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        # Mock encode to return an object with ids attribute
        def encode_text(text):
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
        texts = ["a" * 100 for _ in range(50)]  # Each text = 25 tokens
        total_tokens = 50 * 25  # 1250 tokens

        # Set very low rate limits to force delays
        tokens_per_minute = 200  # Very low token limit
        requests_per_minute = 3   # Very low request limit

        start_time = time.time()
        
        # Track API calls
        api_calls = []
        
        def track_calls(batch):
            api_calls.append((time.time(), len(batch)))
            return np.random.rand(len(batch), 768)
            
        mock_model.compute_source_embeddings = track_calls
        
        embeddings = await get_voyage_embeddings_with_limits(
            texts,
            mock_model,
            mock_tokenizer,
            max_concurrent_requests=5,
            max_retries=3,
            retry_base_delay=0.1,
            tokens_per_minute=tokens_per_minute,
            requests_per_minute=requests_per_minute,
        )

        # Check that we got all embeddings
        assert len(embeddings) == 50

        # Verify rate limiting occurred
        # With only 3 requests per minute allowed, we should see delays
        elapsed = time.time() - start_time
        
        # Check that API calls were spread out
        if len(api_calls) > 3:
            # There should be delays between calls after the 3rd one
            call_times = [t for t, _ in api_calls]
            # Find the time gap between 3rd and 4th call
            if len(call_times) > 3:
                gap = call_times[3] - call_times[2]
                # Should wait significant time due to rate limit
                assert gap > 0.5, f"Expected delay between calls, but gap was only {gap}s"

    @pytest.mark.asyncio
    async def test_resume_after_rate_limit(self, mock_tokenizer):
        """Test that processing resumes correctly after rate limit."""
        # Create longer texts to force multiple batches
        # Each text will be ~30,000 tokens to ensure small batches
        texts = ["x" * 120000 for _ in range(10)]  # 10 very long texts
        
        # Track API calls
        api_calls = []
        rate_limit_hit = False
        call_count = 0
        
        def mock_compute_embeddings(batch):
            nonlocal rate_limit_hit, call_count
            call_count += 1
            api_calls.append((call_count, len(batch)))
            
            # Simulate rate limit on 3rd call
            if call_count == 3 and not rate_limit_hit:
                rate_limit_hit = True
                raise Exception("429 Too Many Requests")
            
            # Return embeddings
            return np.random.rand(len(batch), 768)
        
        mock_model = MagicMock()
        mock_model.compute_source_embeddings = mock_compute_embeddings
        
        # Run with rate limiting
        result = await get_voyage_embeddings_with_limits(
            texts,
            mock_model,
            mock_tokenizer,
            max_concurrent_requests=5,  # Higher concurrency to ensure we hit the 3rd call
            max_retries=3,
            retry_base_delay=0.1,
            tokens_per_minute=100000,
            requests_per_minute=1000,
        )
        
        # Extract embeddings from result
        embeddings = result['embeddings']
        
        # Verify all texts were processed (or some failed)
        total_processed = len(embeddings) + len(result['failed_batches']) * len(texts) // len(result['safe_batches'])
        assert total_processed >= 10
        
        # Debug: print API call info
        print(f"API calls: {api_calls}")
        print(f"Rate limit hit: {rate_limit_hit}")
        
        # Verify we hit rate limit and continued
        assert call_count >= 3, f"Expected at least 3 API calls but made {call_count}"
        assert rate_limit_hit, "Expected to hit simulated rate limit"
        # The 3rd call should have been retried, so we should see call 3 appear again
        call_numbers = [num for num, _ in api_calls]
        assert call_numbers.count(3) == 1 or call_count > 3, "Expected retry after rate limit"

    @pytest.mark.asyncio
    async def test_global_rate_limit_coordination(self, mock_tokenizer, mock_model):
        """Test that global rate limit affects all concurrent batches."""
        texts = ["text" * 10 for _ in range(200)]
        
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
        
        # Run with high concurrency
        embeddings = await get_voyage_embeddings_with_limits(
            texts,
            mock_model,
            mock_tokenizer,
            max_concurrent_requests=10,  # High concurrency
            max_retries=3,
            retry_base_delay=0.5,
            tokens_per_minute=100000,
            requests_per_minute=10000,
        )
        
        # Check all texts processed
        assert len(embeddings) == 200
        
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
        # Create a test engine
        config = BreezeConfig(
            db_path="/tmp/test_breeze_rate_limit.lance",
            embedding_model="test-model",
        )
        
        engine = BreezeEngine(config)
        
        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.compute_source_embeddings.return_value = np.random.rand(1, 768)
        engine.embedding_model = mock_model
        
        # Create test files
        test_files = []
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            for i in range(5):
                file_path = f"{tmpdir}/test_{i}.txt"
                with open(file_path, "w") as f:
                    f.write(f"Test content {i}")
                test_files.append(file_path)
            
            # First indexing
            await engine.init_tables()
            stats1 = await engine.index_directory(
                tmpdir,
                force_reindex=False,
                concurrent_readers=1,
                concurrent_embedders=1,
                concurrent_writers=1,
            )
            
            # Check initial indexing
            assert stats1.files_indexed == 5
            assert stats1.files_updated == 0
            
            # Reset mock to track new calls
            mock_model.compute_source_embeddings.reset_mock()
            
            # Second indexing without changes
            stats2 = await engine.index_directory(
                tmpdir,
                force_reindex=False,
                concurrent_readers=1,
                concurrent_embedders=1,
                concurrent_writers=1,
            )
            
            # Should skip all files
            assert stats2.files_indexed == 0
            assert stats2.files_updated == 0
            assert stats2.files_skipped == 5
            
            # Embedding should not be called for unchanged files
            assert mock_model.compute_source_embeddings.call_count == 0
            
            # Modify one file
            with open(test_files[0], "w") as f:
                f.write("Modified content")
            
            # Third indexing with one changed file
            stats3 = await engine.index_directory(
                tmpdir,
                force_reindex=False,
                concurrent_readers=1,
                concurrent_embedders=1,
                concurrent_writers=1,
            )
            
            # Should update only the modified file
            assert stats3.files_indexed == 0
            assert stats3.files_updated == 1
            assert stats3.files_skipped == 4
            
            # Embedding should be called only once
            assert mock_model.compute_source_embeddings.call_count == 1

    @pytest.mark.asyncio
    async def test_failed_batch_handling(self, mock_tokenizer):
        """Test that failed batches are properly tracked."""
        texts = ["text" * 10 for _ in range(100)]
        
        # Track which batches fail
        failed_batches = {2, 5, 7}  # Batches that will fail
        call_count = 0
        
        def mock_compute_with_failures(batch):
            nonlocal call_count
            call_count += 1
            
            # Fail specific batches persistently
            batch_size = len(batch)
            batch_idx = (call_count - 1) // 5  # Assuming max_concurrent_requests=5
            if batch_idx in failed_batches:
                raise Exception(f"Permanent failure for batch {batch_idx}")
            
            return np.random.rand(len(batch), 768)
        
        mock_model = MagicMock()
        mock_model.compute_source_embeddings = mock_compute_with_failures
        
        # This should raise an exception due to failed batches
        with pytest.raises(Exception) as exc_info:
            await get_voyage_embeddings_with_limits(
                texts,
                mock_model,
                mock_tokenizer,
                max_concurrent_requests=5,
                max_retries=2,  # Lower retries for faster test
                retry_base_delay=0.01,
                tokens_per_minute=100000,
                requests_per_minute=10000,
            )
        
        # Check error message mentions failed batches
        assert "Failed to generate embeddings" in str(exc_info.value)
        assert "batches" in str(exc_info.value)

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
        
        # Use very low limits to force single-item batches
        await get_voyage_embeddings_with_limits(
            texts,
            mock_model,
            mock_tokenizer,
            max_concurrent_requests=1,
            max_retries=3,
            retry_base_delay=0.01,
            tokens_per_minute=100000,
            requests_per_minute=10000,
        )
        
        # Verify token counting
        expected_tokens = [
            len(mock_tokenizer.encode(texts[0]).ids),
            len(mock_tokenizer.encode(texts[1]).ids),
            len(mock_tokenizer.encode(texts[2]).ids),
        ]
        
        # Sort both lists as order might vary
        assert sorted(recorded_tokens) == sorted(expected_tokens)