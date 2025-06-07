"""Tests for ModelAwareChunker."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from breeze.core.chunking import ModelAwareChunker, ChunkingStrategy, TextChunk


class TestModelAwareChunker:
    """Test the ModelAwareChunker functionality."""
    
    def test_model_type_inference(self):
        """Test that model types are correctly inferred."""
        # Voyage models
        chunker = ModelAwareChunker("voyage-code-3")
        assert chunker.model_type == "voyage"
        
        # Gemini models  
        chunker = ModelAwareChunker("models/text-embedding-004")
        assert chunker.model_type == "gemini"
        
        # Sentence transformer models
        chunker = ModelAwareChunker("sentence-transformers/all-MiniLM-L6-v2")
        assert chunker.model_type == "sentence-transformers"
        
    def test_default_strategies(self):
        """Test default chunking strategies for different model types."""
        # Voyage
        chunker = ModelAwareChunker("voyage-code-3")
        assert chunker.strategy.max_tokens == 16000
        assert chunker.strategy.overlap_ratio == 0.1
        
        # Local models
        chunker = ModelAwareChunker("BAAI/bge-m3")
        assert chunker.strategy.max_tokens == 8192
        
    def test_single_short_text_no_chunking(self):
        """Test that short texts are not chunked."""
        chunker = ModelAwareChunker("test-model")
        
        short_text = "def hello(): return 'world'"
        chunks = chunker.chunk_single_text(short_text)
        
        assert len(chunks) == 1
        assert chunks[0].text == short_text
        assert chunks[0].chunk_idx == 0
        assert chunks[0].total_chunks == 1
        
    def test_long_text_chunking(self):
        """Test that long texts are properly chunked."""
        chunker = ModelAwareChunker("test-model")
        chunker.strategy.max_tokens = 100  # Force chunking
        
        # Create long text (400+ tokens)
        long_text = "def example():\n    pass\n" * 100
        chunks = chunker.chunk_single_text(long_text)
        
        assert len(chunks) > 1
        assert all(chunk.total_chunks == len(chunks) for chunk in chunks)
        
        # Verify overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            assert chunks[i].end_idx > chunks[i + 1].start_idx  # Overlap exists
            
    @patch('transformers.AutoTokenizer')
    def test_voyage_tokenizer_chunking(self, mock_auto_tokenizer):
        """Test chunking with Voyage tokenizer."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(1000))  # 1000 tokens
        mock_tokenizer.decode.return_value = "decoded text"
        
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        chunker = ModelAwareChunker("voyage-code-3")
        chunker.strategy.max_tokens = 300  # Force multiple chunks
        
        text = "x" * 4000  # Long text
        chunks = chunker.chunk_single_text(text)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        assert mock_tokenizer.encode.called
        assert mock_tokenizer.decode.called
        
    @patch('transformers.AutoTokenizer')  
    def test_sentence_transformer_tokenizer_chunking(self, mock_auto_tokenizer):
        """Test chunking with sentence transformer tokenizer."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 512
        mock_tokenizer.encode.return_value = list(range(800))  # 800 tokens
        mock_tokenizer.decode.return_value = "decoded chunk"
        
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        chunker = ModelAwareChunker("sentence-transformers/all-MiniLM-L6-v2")
        
        # Strategy should be updated from tokenizer
        assert chunker.strategy.max_tokens == 512
        
        text = "x" * 3000
        chunks = chunker.chunk_single_text(text)
        
        assert len(chunks) > 1
        assert mock_tokenizer.encode.called
        
    def test_multiple_texts_chunking(self):
        """Test chunking multiple texts."""
        chunker = ModelAwareChunker("test-model")
        chunker.strategy.max_tokens = 50
        
        texts = [
            "short text",
            "x" * 500,  # Long text needing chunks
            "another short one",
            "y" * 600   # Another long text
        ]
        
        chunks_per_text = chunker.chunk_texts(texts)
        
        assert len(chunks_per_text) == 4
        assert len(chunks_per_text[0]) == 1  # Short text
        assert len(chunks_per_text[1]) > 1   # Long text chunked
        assert len(chunks_per_text[2]) == 1  # Short text
        assert len(chunks_per_text[3]) > 1   # Long text chunked
        
    def test_combine_embeddings_average(self):
        """Test average combination of embeddings."""
        chunker = ModelAwareChunker("test-model")
        chunker.strategy.combine_method = "average"
        
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]
        
        chunks = [Mock(token_count=10) for _ in range(3)]
        
        combined = chunker.combine_embeddings(embeddings, chunks)
        expected = np.array([4.0, 5.0, 6.0])  # Average
        
        np.testing.assert_array_almost_equal(combined, expected)
        
    def test_combine_embeddings_weighted(self):
        """Test weighted combination of embeddings."""
        chunker = ModelAwareChunker("test-model")
        chunker.strategy.combine_method = "weighted_average"
        
        embeddings = [
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
        ]
        
        chunks = [
            Mock(token_count=30),  # 3x weight
            Mock(token_count=10),  # 1x weight
        ]
        
        combined = chunker.combine_embeddings(embeddings, chunks)
        expected = np.array([1.25, 1.25])  # (3*1 + 1*2) / 4
        
        np.testing.assert_array_almost_equal(combined, expected)
        
    def test_prepare_batches(self):
        """Test batch preparation from chunks."""
        chunker = ModelAwareChunker("test-model")
        
        # Create chunks for 3 texts
        chunks_per_text = [
            [Mock(text=f"t0_c{i}") for i in range(2)],  # 2 chunks
            [Mock(text=f"t1_c{i}") for i in range(3)],  # 3 chunks  
            [Mock(text=f"t2_c{i}") for i in range(1)],  # 1 chunk
        ]
        
        batches = chunker.prepare_batches(chunks_per_text, max_batch_size=2)
        
        # Should create 3 batches: [2], [2], [2]
        assert len(batches) == 3
        assert all(len(batch) <= 2 for batch in batches)
        
        # Verify all chunks are included
        total_chunks = sum(len(batch) for batch in batches)
        expected_total = sum(len(chunks) for chunks in chunks_per_text)
        assert total_chunks == expected_total
        
    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        chunker = ModelAwareChunker("test-model")
        
        chunks = chunker.chunk_single_text("")
        assert chunks == []
        
        chunks_list = chunker.chunk_texts(["", "text", ""])
        assert len(chunks_list) == 3
        assert chunks_list[0] == []
        assert len(chunks_list[1]) == 1
        assert chunks_list[2] == []
        
    def test_token_estimation_fallbacks(self):
        """Test token estimation when tokenizer is not available."""
        chunker = ModelAwareChunker("voyage-code-3")
        chunker.tokenizer = None  # Force fallback
        
        # Voyage estimation (3.5 chars/token)
        tokens = chunker.estimate_tokens("x" * 350)
        assert tokens == 100
        
        chunker = ModelAwareChunker("some-local-model")
        chunker.tokenizer = None
        
        # Other models (4 chars/token)
        tokens = chunker.estimate_tokens("x" * 400)
        assert tokens == 100