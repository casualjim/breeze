"""Integration tests to verify LangChain text splitters are properly integrated."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from breeze.core.embeddings import get_local_embeddings_with_tokenizer_chunking
from breeze.core.text_chunker import FileContent
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class TestLangChainIntegration:
    """Test that LangChain splitters are actually being used."""

    @pytest.mark.asyncio
    async def test_langchain_language_aware_splitting(self):
        """Test that language-aware splitting is used for known languages."""
        mock_model = Mock()
        embedding_dim = 384
        
        # Track what texts are passed to the model
        texts_processed = []
        
        def mock_compute_embeddings(texts):
            texts_processed.extend(texts)
            return [np.random.rand(embedding_dim) for _ in texts]
        
        mock_model.compute_source_embeddings = Mock(side_effect=mock_compute_embeddings)
        
        # Python code with specific structure that LangChain should respect
        python_code = '''
def function_one():
    """This is a complete function."""
    x = 1
    y = 2
    return x + y

def function_two():
    """This is another complete function."""
    a = 3
    b = 4
    return a * b

class MyClass:
    """This is a complete class."""
    def method_one(self):
        return "method one"
    
    def method_two(self):
        return "method two"
''' * 100  # Repeat to ensure chunking
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 512  # Small to force chunking
        
        def mock_encode(text, add_special_tokens=True, **kwargs):
            # Approximate token count
            return Mock(ids=list(range(len(text) // 4)))
        
        mock_tokenizer.encode = Mock(side_effect=mock_encode)
        
        with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            file_content = FileContent(
                content=python_code,
                file_path="test.py",
                language="python"
            )
            
            result = await get_local_embeddings_with_tokenizer_chunking(
                file_contents=[file_content],
                model=mock_model,
                model_name="test-model",
                max_concurrent_requests=1,
                max_sequence_length=512
            )
            
            # Verify chunking occurred
            assert len(result.chunked_files[0].chunks) > 1
            
            # Verify that chunks respect Python structure
            # LangChain's Python splitter should try to keep functions/classes intact
            chunks_text = [chunk.text for chunk in result.chunked_files[0].chunks]
            
            # At least some chunks should start with 'def' or 'class'
            structural_starts = sum(
                1 for chunk in chunks_text 
                if chunk.strip().startswith(('def ', 'class '))
            )
            assert structural_starts > 0, "LangChain should preserve Python structure"

    @pytest.mark.asyncio
    async def test_langchain_splitter_behavior_matches_expected(self):
        """Test that LangChain splitter behavior matches what we expect."""
        # Create a direct LangChain splitter to compare
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=100,
            chunk_overlap=10,
            length_function=len
        )
        
        test_code = '''
def hello():
    return "world"

def goodbye():
    return "universe"
'''
        
        # Get chunks from LangChain directly
        langchain_chunks = splitter.split_text(test_code)
        
        # Now test our integration
        mock_model = Mock()
        mock_model.compute_source_embeddings = Mock(
            return_value=[np.random.rand(384) for _ in range(len(langchain_chunks))]
        )
        
        # Mock tokenizer that approximates character count
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 100
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs: Mock(ids=list(range(len(text)))))
        
        with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            file_content = FileContent(
                content=test_code,
                file_path="test.py",
                language="python"
            )
            
            result = await get_local_embeddings_with_tokenizer_chunking(
                file_contents=[file_content],
                model=mock_model,
                model_name="test-model",
                max_sequence_length=100
            )
            
            # Our chunks should match LangChain's behavior
            our_chunks = [chunk.text for chunk in result.chunked_files[0].chunks]
            
            # They might not be exactly the same due to tokenizer differences,
            # but the count should be similar
            assert abs(len(our_chunks) - len(langchain_chunks)) <= 1

    @pytest.mark.asyncio
    async def test_unsupported_language_falls_back(self):
        """Test that unsupported languages fall back to generic splitting."""
        mock_model = Mock()
        embedding_dim = 384
        
        mock_model.compute_source_embeddings = Mock(
            return_value=[np.random.rand(embedding_dim)]
        )
        
        # Use a language that's not in the map
        file_content = FileContent(
            content="Some content in an unsupported language" * 100,
            file_path="test.xyz",
            language="xyz"  # Not in language map
        )
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 100
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs: Mock(ids=list(range(len(text) // 4))))
        
        with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            # This should not raise an error
            result = await get_local_embeddings_with_tokenizer_chunking(
                file_contents=[file_content],
                model=mock_model,
                model_name="test-model",
                max_sequence_length=100
            )
            
            # Should still get results
            assert len(result.embeddings) == 1
            assert len(result.chunked_files) == 1

    @pytest.mark.asyncio
    async def test_langchain_preserves_code_structure(self):
        """Test that LangChain preserves code structure better than naive splitting."""
        mock_model = Mock()
        embedding_dim = 384
        
        chunks_seen = []
        
        def track_chunks(texts):
            chunks_seen.extend(texts)
            return [np.random.rand(embedding_dim) for _ in texts]
        
        mock_model.compute_source_embeddings = Mock(side_effect=track_chunks)
        
        # JavaScript code with clear structure
        js_code = '''
function processData(data) {
    // This is a complete function that should ideally stay together
    const result = data.map(item => {
        return {
            id: item.id,
            value: item.value * 2,
            processed: true
        };
    });
    
    return result.filter(item => item.value > 10);
}

class DataProcessor {
    constructor() {
        this.data = [];
    }
    
    add(item) {
        this.data.push(item);
    }
    
    process() {
        return processData(this.data);
    }
}
''' * 50  # Repeat to force chunking
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 1000
        mock_tokenizer.encode = Mock(
            side_effect=lambda text, **kwargs: Mock(ids=list(range(len(text) // 4)))
        )
        
        with patch('transformers.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            file_content = FileContent(
                content=js_code,
                file_path="test.js",
                language="javascript"
            )
            
            result = await get_local_embeddings_with_tokenizer_chunking(
                file_contents=[file_content],
                model=mock_model,
                model_name="test-model",
                max_sequence_length=1000
            )
            
            # Check that functions are preserved in chunks
            function_chunks = [
                chunk for chunk in chunks_seen 
                if 'function processData' in chunk or 'class DataProcessor' in chunk
            ]
            
            # Should have preserved some complete structures
            assert len(function_chunks) > 0, "LangChain should preserve JS functions/classes"

    def test_all_langchain_languages_mapped(self):
        """Verify all LangChain languages are properly mapped or documented."""
        from langchain_text_splitters import Language
        
        # Get all LangChain language values
        langchain_languages = {lang.value for lang in Language}
        
        # These are the languages we expect to support
        expected_languages = {
            'cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'php', 'proto',
            'python', 'rst', 'ruby', 'rust', 'scala', 'swift', 'markdown',
            'latex', 'html', 'sol', 'csharp', 'cobol', 'c', 'lua', 'perl',
            'haskell', 'elixir', 'powershell'
        }
        
        # Verify we have all expected languages
        assert langchain_languages == expected_languages, \
            f"Language mismatch. Missing: {expected_languages - langchain_languages}, Extra: {langchain_languages - expected_languages}"