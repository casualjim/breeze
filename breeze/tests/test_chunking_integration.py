"""Integration test showing how ModelAwareChunker fixes the truncation issue."""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from breeze.core.chunking import ModelAwareChunker


class TestChunkingIntegration:
    """Test that demonstrates the fix for truncation issue."""
    
    @pytest.mark.asyncio
    async def test_chunking_replaces_truncation(self):
        """Show how the chunker properly handles long texts instead of truncating."""
        # Create a mock embedding model
        mock_model = Mock()
        mock_model.compute_source_embeddings = Mock()
        
        # Set up mock to return different embeddings for each chunk
        embedding_dim = 384
        mock_model.compute_source_embeddings.side_effect = [
            # First call - 3 chunks from the long text
            [np.random.rand(embedding_dim) for _ in range(3)],
            # Second call - remaining chunk
            [np.random.rand(embedding_dim)]
        ]
        
        # Create the chunker
        chunker = ModelAwareChunker("test-model")
        chunker.strategy.max_tokens = 2000  # Force chunking for demo
        
        # Create a long text that would be truncated at 8192 tokens (~32k chars)
        # This simulates the "Truncated text from 15153 to 8192 tokens" issue
        long_text = "def complex_function():\n    " + "x = process_data()\n    " * 5000
        
        # Chunk the text
        chunks = chunker.chunk_single_text(long_text)
        
        # This should create multiple chunks, NOT truncate
        assert len(chunks) > 1, "Text should be chunked, not truncated"
        
        # Process chunks through the embedding model
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Simulate batching (as would happen in real usage)
        batches = chunker.prepare_batches([chunks], max_batch_size=32)
        
        all_embeddings = []
        for batch in batches:
            batch_texts = [chunk.text for _, chunk in batch]
            embeddings = mock_model.compute_source_embeddings(batch_texts)
            all_embeddings.extend(embeddings)
            
        # Combine the embeddings
        final_embedding = chunker.combine_embeddings(all_embeddings, chunks)
        
        # Verify results
        assert final_embedding.shape == (embedding_dim,)
        assert mock_model.compute_source_embeddings.call_count >= 1
        
        # Calculate total text coverage
        total_chunk_chars = sum(len(chunk.text) for chunk in chunks)
        original_length = len(long_text)
        
        # With overlap, we should cover MORE than the original
        # (overlapping chunks mean some text appears in multiple chunks)
        coverage_ratio = total_chunk_chars / original_length
        assert coverage_ratio >= 0.95, f"Chunks should cover most of the text, got {coverage_ratio:.2%}"
        
        print(f"\nChunking results:")
        print(f"- Original text: {original_length} chars (~{original_length//4} tokens)")
        print(f"- Created {len(chunks)} chunks")
        print(f"- Total coverage: {coverage_ratio:.2%}")
        print(f"- Each chunk: ~{chunks[0].token_count} tokens")
        
    @pytest.mark.asyncio
    async def test_voyage_specific_chunking(self):
        """Test chunking for Voyage models specifically."""
        with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
            # Mock the Voyage tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.model_max_length = 16000  # Voyage code-3 context length
            
            # Simple token estimation: ~3.5 chars per token
            mock_tokenizer.encode.side_effect = lambda text, **kwargs: list(range(len(text) // 3))
            mock_tokenizer.decode.side_effect = lambda tokens, **kwargs: "x" * (len(tokens) * 3)
            
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            # Create chunker for Voyage
            chunker = ModelAwareChunker("voyage-code-3")
            
            # Create text that exceeds single batch limit (120k tokens = ~420k chars)
            huge_text = "x" * 500000  # ~143k tokens
            
            chunks = chunker.chunk_single_text(huge_text)
            
            # Should create multiple chunks even for Voyage's large context
            assert len(chunks) > 1
            
            # Each chunk should be within Voyage's limits
            for chunk in chunks:
                assert chunk.token_count <= 16000
                
            print(f"\nVoyage chunking:")
            print(f"- Text size: ~{len(huge_text)//3} tokens")  
            print(f"- Created {len(chunks)} chunks")
            print(f"- Chunk sizes: {[c.token_count for c in chunks]}")
            
    def test_no_truncation_message(self):
        """Verify that using the chunker eliminates truncation messages."""
        import logging
        
        # Set up logger capture
        with patch('breeze.core.chunking.logger') as mock_logger:
            chunker = ModelAwareChunker("test-model")
            chunker.strategy.max_tokens = 100
            
            # Process long text
            long_text = "x" * 1000
            chunks = chunker.chunk_single_text(long_text)
            
            # Check that NO truncation message was logged
            for call in mock_logger.debug.call_args_list:
                assert "Truncated text from" not in str(call)
                
            # But we should see chunking happening
            assert len(chunks) > 1