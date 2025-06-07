"""Test that Voyage embedder properly chunks long texts using TextChunker."""

import pytest
import numpy as np
from unittest.mock import Mock
from breeze.core.embeddings import get_voyage_embeddings_with_limits
from breeze.core.text_chunker import FileContent
from breeze.core.config import BreezeConfig


class TestVoyageChunking:
    """Test that Voyage embedder uses TextChunker for proper chunking."""

    @pytest.mark.asyncio
    async def test_voyage_chunks_long_text_properly(self):
        """Test that long texts are chunked using TextChunker with 16k token chunks."""
        # Create a mock embedding model
        mock_model = Mock()
        embedding_dim = 1024  # Voyage models have 1024 dimensions
        
        # Create a mock that returns embeddings based on the number of texts in the batch
        def mock_compute_embeddings(texts):
            return [np.random.rand(embedding_dim) for _ in texts]
        
        mock_model.compute_source_embeddings = Mock(side_effect=mock_compute_embeddings)
        
        # Create a long text that requires chunking (>16k tokens)
        # ~200k chars = ~50k tokens at 4 chars/token
        long_text = "def example_function():\n    " + "x = 1\n    " * 30000
        
        # Mock tokenizer for Voyage
        mock_tokenizer = Mock()
        def mock_encode(text, add_special_tokens=True, **kwargs):
            token_count = len(text) // 4  # ~4 chars per token
            return Mock(ids=list(range(token_count)))
        
        mock_tokenizer.encode = Mock(side_effect=mock_encode)
        mock_tokenizer.decode = Mock(return_value="decoded_chunk")
        
        # Configure rate limits for tier 1
        config = BreezeConfig(voyage_tier=1)
        rate_limits = config.get_voyage_rate_limits()
        
        # Call the function with FileContent
        file_content = FileContent(
            content=long_text,
            file_path="test.py",
            language="python"
        )
        result = await get_voyage_embeddings_with_limits(
            file_contents=[file_content],
            model=mock_model,
            tokenizer=mock_tokenizer,
            max_concurrent_requests=5,
            tokens_per_minute=rate_limits['tokens_per_minute'],
            requests_per_minute=rate_limits['requests_per_minute']
        )
        
        # Verify results
        embeddings = result.embeddings
        
        # Should get 1 embedding (combined from chunks)
        assert len(embeddings) == 1, f"Expected 1 embedding, got {len(embeddings)}"
        
        # Model should have been called for processing chunks
        assert mock_model.compute_source_embeddings.call_count >= 1, \
            "Model should be called at least once"
        
        # Check that chunking happened with 16k token chunks
        chunked_files = result.chunked_files
        assert len(chunked_files) == 1, "Should have 1 chunked file"
        
        chunks = chunked_files[0].chunks
        assert len(chunks) > 1, f"Long text should be split into multiple chunks, got {len(chunks)}"
        
        # Each chunk should be around 16k tokens (with some tolerance)
        for chunk in chunks:
            assert 10000 < chunk.estimated_tokens <= 16500, \
                f"Chunk has {chunk.estimated_tokens} tokens, expected ~16k"
        
        # Verify embedding shape
        assert embeddings[0].shape == (embedding_dim,), \
            f"Expected shape ({embedding_dim},), got {embeddings[0].shape}"

    @pytest.mark.asyncio
    async def test_voyage_respects_rate_limits(self):
        """Test that Voyage respects configured rate limits for different tiers."""
        mock_model = Mock()
        mock_model.compute_source_embeddings = Mock(
            return_value=[np.random.rand(1024)]
        )
        
        # Test each tier
        for tier in [1, 2, 3]:
            config = BreezeConfig(voyage_tier=tier)
            rate_limits = config.get_voyage_rate_limits()
            
            # Expected values
            expected_tokens = 3_000_000 * tier
            expected_requests = 2000 * tier
            
            assert rate_limits['tokens_per_minute'] == expected_tokens, \
                f"Tier {tier} should have {expected_tokens} tokens/min"
            assert rate_limits['requests_per_minute'] == expected_requests, \
                f"Tier {tier} should have {expected_requests} requests/min"

    @pytest.mark.asyncio
    async def test_voyage_batching_respects_limits(self):
        """Test that Voyage batching respects 120k token and 128 text limits."""
        mock_model = Mock()
        embedding_dim = 1024
        
        # Return embeddings for each text in batch
        def mock_compute_embeddings(texts):
            return [np.random.rand(embedding_dim) for _ in texts]
        
        mock_model.compute_source_embeddings = Mock(side_effect=mock_compute_embeddings)
        
        # Create multiple files with ~16k tokens each
        # With 120k token limit, we should fit ~7 chunks per batch
        file_contents = []
        for i in range(10):
            # Each file is ~16k tokens (64k chars)
            content = f"def function_{i}():\n    " + "x = 1\n    " * 10000
            file_contents.append(FileContent(
                content=content,
                file_path=f"test{i}.py",
                language="python"
            ))
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs:
            Mock(ids=list(range(len(text) // 4))))
        mock_tokenizer.decode = Mock(return_value="decoded_chunk")
        
        result = await get_voyage_embeddings_with_limits(
            file_contents=file_contents,
            model=mock_model,
            tokenizer=mock_tokenizer,
            max_concurrent_requests=5,
            tokens_per_minute=3_000_000,
            requests_per_minute=2000
        )
        
        embeddings = result.embeddings
        
        # Should get 10 embeddings (one per file)
        assert len(embeddings) == 10, f"Expected 10 embeddings, got {len(embeddings)}"
        
        # Check the API was called multiple times due to batching
        call_count = mock_model.compute_source_embeddings.call_count
        assert call_count >= 2, \
            f"Should have multiple API calls due to 120k token limit, got {call_count}"
        
        # Verify each batch respects limits
        for call in mock_model.compute_source_embeddings.call_args_list:
            batch_texts = call[0][0]
            assert len(batch_texts) <= 128, \
                f"Batch has {len(batch_texts)} texts, exceeds 128 limit"

    @pytest.mark.asyncio
    async def test_voyage_chunking_combines_embeddings(self):
        """Test that chunk embeddings are properly combined."""
        mock_model = Mock()
        embedding_dim = 1024
        
        # Create function that returns embeddings for each chunk call
        def mock_compute_embeddings(texts):
            # Return embeddings based on number of texts in batch
            embeddings = []
            for i, _ in enumerate(texts):
                # Create distinct values for testing weighted average
                value = (i % 3) + 1  # Will give 1, 2, 3, 1, 2, 3...
                embeddings.append(np.array([float(value)] * embedding_dim))
            return embeddings
        
        mock_model.compute_source_embeddings = Mock(side_effect=mock_compute_embeddings)
        
        # Text that will be chunked into 3 parts
        text = "x" * 200000  
        
        # Use mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs: 
            Mock(ids=list(range(len(text) // 4))))
        mock_tokenizer.decode = Mock(return_value="decoded_chunk")
        
        file_content = FileContent(
            content=text,
            file_path="test.py",
            language="python"
        )
        result = await get_voyage_embeddings_with_limits(
            file_contents=[file_content],
            model=mock_model,
            tokenizer=mock_tokenizer,
            max_concurrent_requests=1,
            tokens_per_minute=3_000_000,
            requests_per_minute=2000
        )
        
        embeddings = result.embeddings
        assert len(embeddings) == 1
        
        # The combined embedding should be weighted average
        # We should have valid embedding dimensions
        combined = embeddings[0]
        assert combined.shape == (embedding_dim,), \
            f"Expected shape ({embedding_dim},), got {combined.shape}"
        
        # Check that values are reasonable (not NaN or inf)
        assert np.all(np.isfinite(combined)), "Embedding contains non-finite values"