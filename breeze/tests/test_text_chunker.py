"""Tests for the TextChunker class."""

import pytest
import numpy as np
from unittest.mock import Mock
from breeze.core.text_chunker import (
    TextChunker, ChunkingConfig, TextChunk, FileContent, ChunkedFile, create_batches_from_chunked_files
)


class TestTextChunker:
    """Test the TextChunker functionality."""
    
    def test_short_text_single_chunk(self):
        """Test that short texts return a single chunk."""
        chunker = TextChunker(config=ChunkingConfig(max_tokens=1000))
        
        short_text = "def hello(): return 'world'"
        file_content = FileContent(content=short_text, file_path="test.py", language="python")
        result = chunker.chunk_file(file_content)
        
        assert len(result.chunks) == 1
        assert result.chunks[0].text == short_text
        assert result.chunks[0].chunk_index == 0
        assert result.chunks[0].total_chunks == 1
        
    def test_long_text_multiple_chunks(self):
        """Test that long texts are split into multiple overlapping chunks."""
        chunker = TextChunker(config=ChunkingConfig(
            max_tokens=100,
            overlap_tokens=20,
            stride_ratio=0.8
        ))
        
        # Create text that's definitely longer than 100 tokens (~400 chars)
        long_text = "x" * 1000
        file_content = FileContent(content=long_text, file_path="test.txt", language="text")
        result = chunker.chunk_file(file_content)
        
        assert len(result.chunks) > 1
        assert all(chunk.total_chunks == len(result.chunks) for chunk in result.chunks)
        
        # Check overlap exists
        for i in range(len(result.chunks) - 1):
            # There should be some overlap between consecutive chunks
            chunk1_end = result.chunks[i].text[-50:]  # Last 50 chars
            chunk2_start = result.chunks[i + 1].text[:50]  # First 50 chars
            # At least some characters should match
            assert any(c in chunk2_start for c in chunk1_end[-10:])
    
    def test_chunking_with_tokenizer(self):
        """Test chunking with a real tokenizer."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        
        # Mock encode to return object with ids attribute
        def mock_encode(text, add_special_tokens=True):
            result = Mock()
            # Simulate ~4 chars per token
            result.ids = list(range(len(text) // 4))
            return result
        
        # Mock decode to return a portion of original text
        def mock_decode(token_ids, skip_special_tokens=True):
            # Simulate decoding back to text
            return "decoded_" + str(len(token_ids))
        
        mock_tokenizer.encode = Mock(side_effect=mock_encode)
        mock_tokenizer.decode = Mock(side_effect=mock_decode)
        
        chunker = TextChunker(
            tokenizer=mock_tokenizer,
            config=ChunkingConfig(max_tokens=100, stride_ratio=0.75)
        )
        
        # Text with ~500 tokens
        long_text = "x" * 2000
        file_content = FileContent(content=long_text, file_path="test.txt", language="text")
        result = chunker.chunk_file(file_content)
        
        # Should create multiple chunks
        assert len(result.chunks) > 1
        
        # Tokenizer should be called
        assert mock_tokenizer.encode.called
        assert mock_tokenizer.decode.called
        
    def test_natural_boundary_breaking(self):
        """Test that character-based chunking tries to break at natural boundaries."""
        chunker = TextChunker(config=ChunkingConfig(max_tokens=20))  # Lower limit
        
        # Text with clear sentence boundaries
        text = "This is the first sentence with some words. This is the second sentence with more words. And here is the third sentence."
        file_content = FileContent(content=text, file_path="test.txt", language="text")
        result = chunker.chunk_file(file_content)
        
        # Should have multiple chunks given the token limit
        assert len(result.chunks) >= 2
        
        # Verify chunks don't cut words in half (basic check)
        for chunk in result.chunks:
            # Check that chunk starts and ends with complete words (not mid-word)
            if chunk.chunk_index > 0:  # Not first chunk
                assert chunk.text[0].isspace() or chunk.text[0].isalpha()
            if chunk.chunk_index < chunk.total_chunks - 1:  # Not last chunk
                assert chunk.text[-1].isspace() or chunk.text[-1] in '.!?,'
    
    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = TextChunker()
        file_content = FileContent(content="", file_path="empty.txt", language="text")
        result = chunker.chunk_file(file_content)
        assert result.chunks == []
        
    def test_combine_embeddings_average(self):
        """Test combining embeddings with average method."""
        chunker = TextChunker()
        
        # Create fake embeddings
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([4.0, 5.0, 6.0])
        
        # Create fake chunks
        chunks = [
            TextChunk("chunk1", 0, 10, 0, 2, 10),
            TextChunk("chunk2", 10, 20, 1, 2, 10)
        ]
        
        # Test average
        combined = chunker.combine_chunk_embeddings([emb1, emb2], chunks, method="average")
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(combined, expected)
        
    def test_combine_embeddings_weighted(self):
        """Test combining embeddings with weighted average."""
        chunker = TextChunker()
        
        # Create fake embeddings
        emb1 = np.array([1.0, 1.0, 1.0])
        emb2 = np.array([2.0, 2.0, 2.0])
        
        # Create chunks with different token counts
        chunks = [
            TextChunk("chunk1", 0, 10, 0, 2, 30),  # 30 tokens
            TextChunk("chunk2", 10, 20, 1, 2, 10)  # 10 tokens
        ]
        
        # Test weighted average (30:10 ratio = 3:1)
        combined = chunker.combine_chunk_embeddings([emb1, emb2], chunks, method="weighted_average")
        expected = np.array([1.25, 1.25, 1.25])  # (3*1 + 1*2) / 4
        np.testing.assert_array_almost_equal(combined, expected)
        
    def test_combine_embeddings_single(self):
        """Test that single embedding is returned as-is."""
        chunker = TextChunker()
        
        emb = np.array([1.0, 2.0, 3.0])
        chunks = [TextChunk("chunk", 0, 10, 0, 1, 10)]
        
        combined = chunker.combine_chunk_embeddings([emb], chunks)
        np.testing.assert_array_equal(combined, emb)
        
    def test_create_batches_from_chunks(self):
        """Test batch creation from chunks."""
        # Create chunked files
        chunked_files = [
            ChunkedFile(
                source=FileContent("text0", "file0.txt", "text"),
                chunks=[TextChunk(f"text0_chunk{i}", 0, 10, i, 2, 10) for i in range(2)]
            ),
            ChunkedFile(
                source=FileContent("text1", "file1.txt", "text"),
                chunks=[TextChunk(f"text1_chunk{i}", 0, 10, i, 3, 10) for i in range(3)]
            ),
            ChunkedFile(
                source=FileContent("text2", "file2.txt", "text"),
                chunks=[TextChunk(f"text2_chunk{i}", 0, 10, i, 1, 10) for i in range(1)]
            ),
        ]
        
        # Create batches with size 2
        batches = create_batches_from_chunked_files(chunked_files, batch_size=2)
        
        # Should have 3 batches: [2 chunks], [2 chunks], [2 chunks]
        assert len(batches) == 3
        
        # Check first batch - each item is (file_idx, file_content, chunk)
        assert len(batches[0]) == 2
        assert batches[0][0][0] == 0  # From file 0
        assert batches[0][1][0] == 0  # From file 0
        
        # Check that all chunks are included
        total_chunks = sum(len(batch) for batch in batches)
        expected_total = sum(len(cf.chunks) for cf in chunked_files)
        assert total_chunks == expected_total
        
    def test_chunking_config_defaults(self):
        """Test ChunkingConfig default values."""
        config = ChunkingConfig(max_tokens=1000)
        
        assert config.max_tokens == 1000
        assert config.overlap_tokens == 128
        assert config.reserved_tokens == 10
        assert config.stride_ratio == 0.75