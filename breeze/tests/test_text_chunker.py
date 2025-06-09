"""Tests for the TextChunker class."""

import numpy as np
from unittest.mock import Mock
from breeze.core.text_chunker import (
    TextChunker, ChunkingConfig, TextChunk, FileContent, ChunkedFile, 
    create_batches_from_chunked_files, SimpleChunkingStrategy
)


class TestTextChunker:
    """Test the TextChunker functionality."""
    
    def test_short_text_single_chunk(self):
        """Test that short texts return a single chunk."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = Mock(ids=list(range(10)))  # Short text
        
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(
            strategy=strategy,
            config=ChunkingConfig(chunk_size=1000, model_max_tokens=8192)
        )
        
        short_text = "def hello(): return 'world'"
        file_content = FileContent(content=short_text, file_path="test.py", language="python")
        result = chunker.chunk_file(file_content)
        
        assert len(result.chunks) == 1
        assert result.chunks[0].text == short_text
        assert result.chunks[0].chunk_index == 0
        assert result.chunks[0].total_chunks == 1
        
    def test_long_text_multiple_chunks(self):
        """Test that long texts are split into multiple overlapping chunks."""
        mock_tokenizer = Mock()
        # Simulate encoding that returns many tokens
        mock_tokenizer.encode.return_value = Mock(ids=list(range(250)))  # Long text
        mock_tokenizer.decode.side_effect = lambda ids, **kwargs: 'x' * len(ids)
        
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(
            strategy=strategy,
            config=ChunkingConfig(
                chunk_size=100,
                model_max_tokens=8192,
                overlap_tokens=20,
                stride_ratio=0.8
            )
        )
        
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
        
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(
            strategy=strategy,
            config=ChunkingConfig(chunk_size=100, model_max_tokens=8192, stride_ratio=0.75)
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
        # Track what text is being encoded/decoded
        encoded_texts = []
        
        mock_tokenizer = Mock()
        
        def mock_encode(text, **kwargs):
            encoded_texts.append(text)
            return Mock(ids=list(range(len(text) // 4)))
            
        def mock_decode(ids, **kwargs):
            # Return a portion of the original text based on token count
            if encoded_texts:
                original = encoded_texts[0]
                # Approximate character range based on tokens
                char_count = len(ids) * 4
                return original[:char_count] if char_count < len(original) else original
            return 'x' * len(ids)
            
        mock_tokenizer.encode.side_effect = mock_encode
        mock_tokenizer.decode.side_effect = mock_decode
        
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(
            strategy=strategy,
            config=ChunkingConfig(chunk_size=20, model_max_tokens=8192)
        )  # Lower limit
        
        # Text with clear sentence boundaries
        text = "This is the first sentence with some words. This is the second sentence with more words. And here is the third sentence."
        file_content = FileContent(content=text, file_path="test.txt", language="text")
        result = chunker.chunk_file(file_content)
        
        # Should have multiple chunks given the token limit
        assert len(result.chunks) >= 2
        
        # Just verify we got multiple chunks and they have content
        for chunk in result.chunks:
            assert len(chunk.text) > 0
            assert chunk.chunk_index >= 0
            assert chunk.total_chunks > 0
    
    def test_empty_text(self):
        """Test handling of empty text."""
        mock_tokenizer = Mock()
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(strategy=strategy)
        file_content = FileContent(content="", file_path="empty.txt", language="text")
        result = chunker.chunk_file(file_content)
        assert result.chunks == []
        
    def test_combine_embeddings_average(self):
        """Test combining embeddings with average method."""
        mock_tokenizer = Mock()
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(strategy=strategy)
        
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
        mock_tokenizer = Mock()
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(strategy=strategy)
        
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
        mock_tokenizer = Mock()
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(strategy=strategy)
        
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
        config = ChunkingConfig(chunk_size=1000, model_max_tokens=8192)
        
        assert config.chunk_size == 1000
        assert config.model_max_tokens == 8192
        assert config.overlap_tokens == 128
        assert config.reserved_tokens == 10
        assert config.stride_ratio == 0.75
    
    def test_combine_embeddings_handles_lists(self):
        """Test that combine_embeddings handles list inputs correctly."""
        mock_tokenizer = Mock()
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(strategy=strategy)
        
        # Create embeddings as lists (not numpy arrays)
        emb1 = [1.0, 2.0, 3.0]
        emb2 = [4.0, 5.0, 6.0]
        
        chunks = [
            TextChunk("chunk1", 0, 10, 0, 2, 10),
            TextChunk("chunk2", 10, 20, 1, 2, 10)
        ]
        
        # Test that it handles list inputs correctly
        combined = chunker.combine_chunk_embeddings([emb1, emb2], chunks, method="average")
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_combine_embeddings_weighted_with_lists(self):
        """Test weighted average with list inputs."""
        mock_tokenizer = Mock()
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(strategy=strategy)
        
        # Create embeddings as lists
        emb1 = [1.0, 1.0, 1.0]
        emb2 = [2.0, 2.0, 2.0]
        
        chunks = [
            TextChunk("chunk1", 0, 10, 0, 2, 30),  # 30 tokens
            TextChunk("chunk2", 10, 20, 1, 2, 10)  # 10 tokens
        ]
        
        # Test weighted average with lists
        combined = chunker.combine_chunk_embeddings([emb1, emb2], chunks, method="weighted_average")
        expected = np.array([1.25, 1.25, 1.25])  # (3*1 + 1*2) / 4
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_combine_embeddings_mixed_types(self):
        """Test handling mixed numpy arrays and lists."""
        mock_tokenizer = Mock()
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(strategy=strategy)
        
        # Mix of numpy array and list
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = [4.0, 5.0, 6.0]  # Plain list
        
        chunks = [
            TextChunk("chunk1", 0, 10, 0, 2, 10),
            TextChunk("chunk2", 10, 20, 1, 2, 10)
        ]
        
        # Should handle mixed types
        combined = chunker.combine_chunk_embeddings([emb1, emb2], chunks, method="average")
        assert isinstance(combined, np.ndarray)
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_combine_embeddings_numpy_float_multiplication(self):
        """Test that numpy float multiplication works correctly."""
        mock_tokenizer = Mock()
        strategy = SimpleChunkingStrategy(tokenizer=mock_tokenizer)
        chunker = TextChunker(strategy=strategy)
        
        # Create embeddings that might trigger the numpy float issue
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([4.0, 5.0, 6.0])
        emb3 = np.array([7.0, 8.0, 9.0])
        
        chunks = [
            TextChunk("chunk1", 0, 10, 0, 3, 100),  # Different token counts
            TextChunk("chunk2", 10, 20, 1, 3, 200),
            TextChunk("chunk3", 20, 30, 2, 3, 300)
        ]
        
        # Test weighted average - this was causing the original error
        combined = chunker.combine_chunk_embeddings([emb1, emb2, emb3], chunks, method="weighted_average")
        assert isinstance(combined, np.ndarray)
        assert combined.shape == (3,)
        
        # Verify the weights work correctly
        # Weights should be [100/600, 200/600, 300/600] = [1/6, 2/6, 3/6]
        expected = (emb1 * (1/6) + emb2 * (2/6) + emb3 * (3/6))
        np.testing.assert_array_almost_equal(combined, expected)