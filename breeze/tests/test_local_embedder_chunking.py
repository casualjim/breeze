"""Test that local embedders properly chunk long texts instead of truncating them."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from breeze.core.embeddings import get_local_embeddings_with_tokenizer_chunking
from breeze.core.text_chunker import FileContent


class TestLocalEmbedderChunking:
    """Test that local embedders chunk text instead of truncating."""

    @pytest.mark.asyncio
    async def test_local_embedder_chunks_long_text_properly(self):
        """Test that long texts are chunked and combined properly."""
        # Create a mock embedding model
        mock_model = Mock()
        embedding_dim = 384
        
        # Create a mock that returns embeddings based on the number of texts in the batch
        def mock_compute_embeddings(texts):
            return [np.random.rand(embedding_dim) for _ in texts]
        
        mock_model.compute_source_embeddings = Mock(side_effect=mock_compute_embeddings)
        
        # Create a long text that requires chunking (>8192 tokens)
        long_text = "def example_function():\n    " + "x = 1\n    " * 10000  # ~60,000 chars
        
        # Mock tokenizer with proper behavior
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 8192
        
        # Proper encode mock that returns object with ids attribute
        def mock_encode(text, add_special_tokens=True, **kwargs):
            token_count = len(text) // 4  # ~4 chars per token
            return Mock(ids=list(range(token_count)))
        
        mock_tokenizer.encode = Mock(side_effect=mock_encode)
        mock_tokenizer.decode = Mock(return_value="decoded_chunk")
        
        with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            # Call the function with FileContent
            file_content = FileContent(
                content=long_text,
                file_path="test.py", 
                language="python"
            )
            
            result = await get_local_embeddings_with_tokenizer_chunking(
                file_contents=[file_content],
                model=mock_model,
                model_name="test-model",
                max_concurrent_requests=1,
                max_sequence_length=8192
            )
            
            # Verify results
            embeddings = result.embeddings
            
            # Should get 1 embedding (combined from chunks)
            assert len(embeddings) == 1, f"Expected 1 embedding, got {len(embeddings)}"
            
            # Model should have been called for processing chunks
            assert mock_model.compute_source_embeddings.call_count >= 1, \
                "Model should be called at least once"
            
            # Check that chunking happened
            chunked_files = result.chunked_files
            assert len(chunked_files) == 1, "Should have 1 chunked file"
            
            chunks = chunked_files[0].chunks
            assert len(chunks) > 1, f"Long text should be split into multiple chunks, got {len(chunks)}"
            
            # Verify embedding shape
            assert embeddings[0].shape == (embedding_dim,), \
                f"Expected shape ({embedding_dim},), got {embeddings[0].shape}"

    @pytest.mark.asyncio
    async def test_multiple_files_chunked_independently(self):
        """Test that multiple files are chunked independently."""
        mock_model = Mock()
        embedding_dim = 384
        
        # Return embeddings for each text in batch
        def mock_compute_embeddings(texts):
            return [np.random.rand(embedding_dim) for _ in texts]
        
        mock_model.compute_source_embeddings = Mock(side_effect=mock_compute_embeddings)
        
        # Create two long texts
        long_text1 = "class MyClass:\n    " + "def method(self): pass\n    " * 5000
        long_text2 = "function example() {\n    " + "console.log('hello');\n    " * 5000
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 8192
        
        def mock_encode(text, add_special_tokens=True, **kwargs):
            return Mock(ids=list(range(len(text) // 4)))
        
        mock_tokenizer.encode = Mock(side_effect=mock_encode)
        mock_tokenizer.decode = Mock(return_value="decoded_chunk")
        
        with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            file_contents = [
                FileContent(content=long_text1, file_path="test1.py", language="python"),
                FileContent(content=long_text2, file_path="test2.js", language="javascript")
            ]
            
            result = await get_local_embeddings_with_tokenizer_chunking(
                file_contents=file_contents,
                model=mock_model,
                model_name="test-model",
                max_concurrent_requests=2
            )
            
            embeddings = result.embeddings
            
            # Should get 2 embeddings (one per file)
            assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
            
            # Each should have correct shape
            for emb in embeddings:
                assert emb.shape == (embedding_dim,)
            
            # Check chunking info
            chunked_files = result.chunked_files
            assert len(chunked_files) == 2, "Should have 2 chunked files"
            
            # Both files should be chunked
            for cf in chunked_files:
                assert len(cf.chunks) > 1, "Each long file should have multiple chunks"

    @pytest.mark.asyncio
    async def test_short_text_not_chunked(self):
        """Test that short texts are not unnecessarily chunked."""
        mock_model = Mock()
        embedding_dim = 384
        
        # Single embedding for single chunk
        mock_model.compute_source_embeddings = Mock(
            return_value=[np.random.rand(embedding_dim)]
        )
        
        # Short text that fits in one chunk
        short_text = "def hello():\n    return 'world'"
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 8192
        mock_tokenizer.encode = Mock(return_value=Mock(ids=list(range(10))))  # Only 10 tokens
        mock_tokenizer.decode = Mock(return_value=short_text)
        
        with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            file_content = FileContent(
                content=short_text,
                file_path="test.py",
                language="python"
            )
            
            result = await get_local_embeddings_with_tokenizer_chunking(
                file_contents=[file_content],
                model=mock_model,
                model_name="test-model"
            )
            
            embeddings = result.embeddings
            assert len(embeddings) == 1
            
            # Should only call model once
            assert mock_model.compute_source_embeddings.call_count == 1
            
            # Should have only one chunk
            chunked_files = result.chunked_files
            assert len(chunked_files[0].chunks) == 1, "Short text should not be chunked"

    @pytest.mark.asyncio
    async def test_no_truncation_occurs(self):
        """Test that text is never truncated, only chunked."""
        
        # Set up logging capture
        with patch('breeze.core.embeddings.logger') as mock_logger:
            mock_model = Mock()
            embedding_dim = 384
            
            # Return embeddings for chunks
            def mock_compute_embeddings(texts):
                return [np.random.rand(embedding_dim) for _ in texts]
            
            mock_model.compute_source_embeddings = Mock(side_effect=mock_compute_embeddings)
            
            # Very long text
            long_text = "x" * 100000  # 100k chars
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.model_max_length = 8192
            
            def mock_encode(text, add_special_tokens=True, **kwargs):
                # Return proper Mock with ids attribute
                return Mock(ids=list(range(len(text) // 4)))
            
            mock_tokenizer.encode = Mock(side_effect=mock_encode)
            mock_tokenizer.decode = Mock(return_value="decoded_chunk")
            
            with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
                mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                
                file_content = FileContent(
                    content=long_text,
                    file_path="test.txt",
                    language="text"
                )
                
                result = await get_local_embeddings_with_tokenizer_chunking(
                    file_contents=[file_content],
                    model=mock_model,
                    model_name="test-model"
                )
                
                # Check that we got a result
                assert len(result.embeddings) == 1
                
                # Check if any truncation message was logged
                truncation_logged = any(
                    "Truncated" in str(call_arg)
                    for call_arg in mock_logger.debug.call_args_list
                )
                
                # With proper chunking, there should be NO truncation
                assert not truncation_logged, \
                    "Text was truncated! The new implementation should chunk, not truncate."
                
                # Verify chunking occurred
                assert len(result.chunked_files[0].chunks) > 1, \
                    "Long text should be split into multiple chunks"