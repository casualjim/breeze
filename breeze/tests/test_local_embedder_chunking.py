"""Test that local embedders properly chunk long texts instead of truncating them."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from breeze.core.embeddings import get_local_embeddings_with_tokenizer_chunking


class TestLocalEmbedderChunking:
    """Test that local embedders chunk text instead of truncating."""

    @pytest.mark.asyncio
    async def test_local_embedder_chunks_long_text_not_truncates(self):
        """Test that long texts are chunked into multiple embeddings, not truncated."""
        # Create a mock embedding model with a low max sequence length
        mock_model = Mock()
        mock_model.compute_source_embeddings = Mock()
        
        # Mock embeddings - return different embeddings for each chunk
        # This simulates the model being called multiple times with different chunks
        embedding_dim = 384
        chunk1_embedding = np.random.rand(embedding_dim)
        chunk2_embedding = np.random.rand(embedding_dim)
        chunk3_embedding = np.random.rand(embedding_dim)
        
        # Set up the mock to return different embeddings for each call
        mock_model.compute_source_embeddings.side_effect = [
            [chunk1_embedding],  # First chunk
            [chunk2_embedding],  # Second chunk  
            [chunk3_embedding],  # Third chunk
        ]
        
        # Create a long text that would require chunking
        # 15,153 tokens is approximately 60,000 characters (using ~4 chars per token)
        long_text = "def example_function():\n    " + "x = 1\n    " * 10000  # ~60,000 chars
        
        # Mock tokenizer with a max length of 8192
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 8192
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs: 
            Mock(ids=list(range(len(text) // 4))))  # Simulate ~4 chars per token
        mock_tokenizer.decode = Mock(side_effect=lambda tokens, **kwargs: 
            "decoded_text" * len(tokens))
        
        with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            # Call the function
            result = await get_local_embeddings_with_tokenizer_chunking(
                texts=[long_text],
                model=mock_model,
                model_name="test-model",
                max_concurrent_requests=1,
                max_sequence_length=8192
            )
            
            # EXPECTED BEHAVIOR: The text should be chunked, not truncated
            # We should get multiple embeddings that we need to combine/average
            embeddings = result['embeddings']
            
            # We should have gotten 1 embedding (even though it was processed in chunks)
            assert len(embeddings) == 1, "Should return one embedding per input text"
            
            # The model should have been called multiple times for chunks
            assert mock_model.compute_source_embeddings.call_count > 1, \
                f"Model was only called {mock_model.compute_source_embeddings.call_count} time(s), but text requires chunking"
            
            # Verify that the text was not truncated by checking the calls
            all_chunks = []
            for call in mock_model.compute_source_embeddings.call_args_list:
                chunks = call[0][0]  # First positional argument is the list of texts
                all_chunks.extend(chunks)
            
            # The chunks should cover the whole text, not just truncate it
            total_chunk_length = sum(len(chunk) for chunk in all_chunks)
            
            # We expect chunking to preserve most of the content
            # Allow some overlap/padding, but should be close to original length
            assert total_chunk_length > len(long_text) * 0.8, \
                f"Chunks only cover {total_chunk_length} chars but original was {len(long_text)} chars - text was truncated, not chunked!"
            
            # The embedding should be some combination of chunk embeddings
            # (average, weighted average, or concatenation)
            assert embeddings[0].shape == (embedding_dim,), \
                f"Expected embedding of shape ({embedding_dim},), got {embeddings[0].shape}"

    @pytest.mark.asyncio
    async def test_local_embedder_handles_multiple_long_texts(self):
        """Test that multiple long texts are each properly chunked."""
        # Create mock model
        mock_model = Mock()
        embedding_dim = 384
        
        # Generate unique embeddings for each chunk
        all_embeddings = [np.random.rand(embedding_dim) for _ in range(6)]
        mock_model.compute_source_embeddings = Mock(side_effect=[
            [all_embeddings[0], all_embeddings[1]],  # First batch: 2 chunks from 2 texts
            [all_embeddings[2], all_embeddings[3]],  # Second batch: 2 more chunks
            [all_embeddings[4], all_embeddings[5]],  # Third batch: final chunks
        ])
        
        # Create two long texts
        long_text1 = "class MyClass:\n    " + "def method(self): pass\n    " * 5000
        long_text2 = "function example() {\n    " + "console.log('hello');\n    " * 5000
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 8192
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs: 
            Mock(ids=list(range(len(text) // 4))))
        mock_tokenizer.decode = Mock(return_value="decoded_chunk")
        
        with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            result = await get_local_embeddings_with_tokenizer_chunking(
                texts=[long_text1, long_text2],
                model=mock_model,
                model_name="test-model",
                max_concurrent_requests=2
            )
            
            embeddings = result['embeddings']
            
            # Should get 2 embeddings (one per text)
            assert len(embeddings) == 2, "Should return one embedding per input text"
            
            # Model should be called multiple times for chunking
            assert mock_model.compute_source_embeddings.call_count >= 2, \
                "Model should be called multiple times for chunking"
            
            # Each embedding should have the correct shape
            for i, emb in enumerate(embeddings):
                assert emb.shape == (embedding_dim,), \
                    f"Text {i} embedding has wrong shape: {emb.shape}"

    @pytest.mark.asyncio 
    async def test_truncation_warning_indicates_bug(self):
        """Test that seeing 'Truncated text from X to Y tokens' in logs indicates a bug."""
        import logging
        
        # Set up logging capture
        with patch('breeze.core.embeddings.logger') as mock_logger:
            mock_model = Mock()
            mock_model.compute_source_embeddings = Mock(return_value=[np.random.rand(384)])
            
            # Text that would trigger truncation
            long_text = "x" * 100000  # Very long text
            
            # Mock tokenizer that will cause truncation
            mock_tokenizer = Mock()
            mock_tokenizer.model_max_length = 8192
            # Simulate encoding that returns more tokens than max length
            def mock_encode(text, add_special_tokens=True, truncation=False, max_length=None, return_tensors=None):
                if truncation and max_length:
                    # When truncation is requested, return exactly max_length tokens
                    return list(range(max_length))
                else:
                    # Without truncation, return actual token count (more than max)
                    return list(range(15153))
            
            mock_tokenizer.encode = Mock(side_effect=mock_encode)
            mock_tokenizer.decode = Mock(return_value="truncated_text")
            
            with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
                mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                
                result = await get_local_embeddings_with_tokenizer_chunking(
                    texts=[long_text],
                    model=mock_model,
                    model_name="test-model"
                )
                
                # Check if truncation message was logged (this indicates the bug)
                truncation_logged = any(
                    "Truncated text from" in str(call) 
                    for call in mock_logger.debug.call_args_list
                )
                
                # This assertion will FAIL with current implementation, proving the bug
                assert not truncation_logged, \
                    "Text was truncated instead of chunked! This is a bug - embeddings will lose information."


# Expected behavior:
# 1. Long texts should be split into overlapping chunks that fit within the model's token limit
# 2. Each chunk should be embedded separately
# 3. The final embedding should be created by combining chunk embeddings (e.g., weighted average)
# 4. No content should be lost due to truncation
#
# Current behavior (BUG):
# - Text is truncated to fit the model's max sequence length
# - Content beyond the token limit is lost
# - Only a single embedding is generated from the truncated text
#
# This is particularly problematic for code files which can be very long and where
# the most important parts (e.g., main function, key classes) might be anywhere in the file.