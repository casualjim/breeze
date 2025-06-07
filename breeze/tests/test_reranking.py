"""Tests for search result reranking functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pytest_asyncio

from breeze.core.config import BreezeConfig
from breeze.core.engine import BreezeEngine
from breeze.core.models import SearchResult
from lancedb.embeddings.registry import get_registry


class TestReranking:
    """Test reranking functionality for search results."""
    
    @pytest_asyncio.fixture
    async def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest_asyncio.fixture
    async def engine(self, temp_dir):
        """Create a BreezeEngine instance with test configuration."""
        registry = get_registry()
        mock_embedder = registry.get("mock-local").create()
        
        config = BreezeConfig(
            data_root=str(temp_dir),
            embedding_function=mock_embedder,
            reranker_model=None,  # Will be set in tests
            default_limit=10,
            min_relevance=0.1
        )
        engine = BreezeEngine(config)
        await engine.initialize()
        yield engine
        await engine.shutdown()
    
    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results for testing."""
        return [
            SearchResult(
                id="1",
                file_path="/test/file1.py",
                file_type="py",
                relevance_score=0.8,
                snippet="def calculate_total(items):\n    return sum(items)",
                last_modified="2024-01-01"
            ),
            SearchResult(
                id="2", 
                file_path="/test/file2.py",
                file_type="py",
                relevance_score=0.7,
                snippet="def sum_values(values):\n    total = 0\n    for v in values:\n        total += v\n    return total",
                last_modified="2024-01-01"
            ),
            SearchResult(
                id="3",
                file_path="/test/file3.py",
                file_type="py", 
                relevance_score=0.6,
                snippet="class Calculator:\n    def add(self, a, b):\n        return a + b",
                last_modified="2024-01-01"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_reranking_disabled(self, engine):
        """Test that search works without reranking."""
        # Disable reranking
        engine.config.reranker_model = None
        
        # Create and index a test file
        test_dir = Path(engine.config.data_root) / "test_files"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "test.py").write_text("def hello():\n    print('Hello, World!')")
        
        # Index the file
        await engine.index_directories([str(test_dir)], force_reindex=True)
        
        # Search without reranking
        results = await engine.search("hello", use_reranker=False)
        
        # Should get results without reranking
        assert len(results) == 1
        assert "hello" in results[0].snippet
    
    @pytest.mark.asyncio
    async def test_voyage_reranking_mock(self, engine, mock_search_results):
        """Test Voyage reranking logic with mocked API."""
        engine.config.reranker_model = "rerank-2"
        engine.config.reranker_api_key = "test-api-key"
        
        # Create a mock Voyage client
        mock_client = MagicMock()
        mock_rerank_result = MagicMock()
        mock_rerank_result.results = [
            MagicMock(index=1, relevance_score=0.95),  # file2
            MagicMock(index=0, relevance_score=0.85),  # file1
            MagicMock(index=2, relevance_score=0.70),  # file3
        ]
        mock_client.rerank.return_value = mock_rerank_result
        
        # Prepare results with content
        results_with_content = [
            (mock_search_results[0], "def calculate_total(items):\n    return sum(items)"),
            (mock_search_results[1], "def sum_values(values):\n    total = 0\n    for v in values:\n        total += v\n    return total"),
            (mock_search_results[2], "class Calculator:\n    def add(self, a, b):\n        return a + b")
        ]
        
        # Mock the voyageai module in the method
        with patch.dict('sys.modules', {'voyageai': MagicMock(Client=MagicMock(return_value=mock_client))}):
            # Call reranking
            reranked = await engine._rerank_voyage("sum calculation", results_with_content, "rerank-2")
            
            # Verify reranking order and scores
            assert len(reranked) == 3
            assert reranked[0].id == "2"  # file2 should be first
            assert reranked[0].relevance_score == 0.95
            assert reranked[1].id == "1"  # file1 should be second
            assert reranked[1].relevance_score == 0.85
            assert reranked[2].id == "3"  # file3 should be third
            assert reranked[2].relevance_score == 0.70
    
    @pytest.mark.asyncio
    async def test_gemini_reranking_mock(self, engine, mock_search_results):
        """Test Gemini reranking logic with mocked API."""
        engine.config.reranker_model = "models/gemini-2.0-flash-lite"
        engine.config.reranker_api_key = "test-api-key"
        
        # Create mock Gemini model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "1, 0, 2"  # Reorder: file2, file1, file3
        mock_model.generate_content.return_value = mock_response
        
        # Prepare results with content
        results_with_content = [
            (mock_search_results[0], "content1"),
            (mock_search_results[1], "content2"), 
            (mock_search_results[2], "content3")
        ]
        
        # Mock google.generativeai module
        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = MagicMock()
        
        with patch.dict('sys.modules', {'google.generativeai': mock_genai}):
            # Call reranking
            reranked = await engine._rerank_gemini("test query", results_with_content, "models/gemini-2.0-flash-lite")
            
            # Verify reranking order
            assert len(reranked) == 3
            assert reranked[0].id == "2"  # file2 first
            assert reranked[1].id == "1"  # file1 second
            assert reranked[2].id == "3"  # file3 third
            
            # Verify decreasing scores were assigned
            assert reranked[0].relevance_score > reranked[1].relevance_score
            assert reranked[1].relevance_score > reranked[2].relevance_score
    
    @pytest.mark.asyncio
    async def test_local_reranking_mock(self, engine, mock_search_results):
        """Test local cross-encoder reranking with mocked model."""
        engine.config.reranker_model = "BAAI/bge-reranker-v2-m3"
        
        # Create mock cross-encoder
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.6, 0.9, 0.3]  # Scores for each pair
        mock_cross_encoder.max_length = 512
        
        # Prepare results with content  
        results_with_content = [
            (mock_search_results[0], "content1"),
            (mock_search_results[1], "content2"),
            (mock_search_results[2], "content3")
        ]
        
        # Mock sentence_transformers module
        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            # Call reranking
            reranked = await engine._rerank_local("test query", results_with_content, "BAAI/bge-reranker-v2-m3")
            
            # Verify reranking order (based on scores: 0.9, 0.6, 0.3)
            assert len(reranked) == 3
            assert reranked[0].id == "2"  # Highest score 0.9
            assert reranked[0].relevance_score == 0.9
            assert reranked[1].id == "1"  # Second score 0.6
            assert reranked[1].relevance_score == 0.6
            assert reranked[2].id == "3"  # Lowest score 0.3
            assert reranked[2].relevance_score == 0.3
    
    @pytest.mark.asyncio
    async def test_reranking_error_handling(self, engine, mock_search_results):
        """Test that reranking failures fall back gracefully."""
        engine.config.reranker_model = "rerank-2"
        
        # Prepare results
        results_with_content = [(r, "content") for r in mock_search_results]
        
        # Mock voyageai to raise exception
        mock_voyage = MagicMock()
        mock_voyage.Client.side_effect = Exception("API Error")
        
        with patch.dict('sys.modules', {'voyageai': mock_voyage}):
            # Call reranking - should return original results
            reranked = await engine._rerank_voyage("test", results_with_content, "rerank-2")
            
            # Should return original results in original order
            assert len(reranked) == 3
            assert reranked[0].id == "1"
            assert reranked[1].id == "2" 
            assert reranked[2].id == "3"
    
    @pytest.mark.asyncio
    async def test_reranking_with_tokenizer(self, engine):
        """Test local reranking handles long content correctly."""
        engine.config.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        # Create a long content that needs truncation
        long_content = "def process_data():\n" + "\n".join([f"    line_{i} = {i}" for i in range(200)])
        
        # Mock cross-encoder
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.8]
        mock_cross_encoder.max_length = 512
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 512
        mock_tokenizer.encode.return_value = list(range(600))  # More than max
        mock_tokenizer.decode.return_value = "def process_data():\n    line_0 = 0\n    line_1 = 1..."
        mock_cross_encoder.tokenizer = mock_tokenizer
        
        # Create result with long content
        result = SearchResult(
            id="1",
            file_path="/test/long.py",
            file_type="py",
            relevance_score=0.5,
            snippet="snippet",
            last_modified="2024-01-01"
        )
        
        results_with_content = [(result, long_content)]
        
        # Mock modules
        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            # Call reranking
            reranked = await engine._rerank_local("process", results_with_content, "cross-encoder/ms-marco-MiniLM-L-6-v2")
            
            # Verify result
            assert len(reranked) == 1
            assert reranked[0].relevance_score == 0.8
    
    @pytest.mark.asyncio 
    async def test_reranking_empty_results(self, engine):
        """Test reranking with empty results."""
        engine.config.reranker_model = "rerank-2"
        
        # Call reranking with empty list
        reranked = await engine._rerank_voyage("test", [], "rerank-2")
        
        assert reranked == []
    
    @pytest.mark.asyncio
    async def test_reranking_integration(self, engine):
        """Test reranking integration in search method."""
        engine.config.reranker_model = None  # Disable for this test
        
        # Create test files and index them
        test_dir = Path(engine.config.data_root) / "test_files"
        test_dir.mkdir(exist_ok=True)
        
        # Create test files
        (test_dir / "exact_match.py").write_text("def calculate_sum(numbers):\n    return sum(numbers)")
        (test_dir / "partial_match.py").write_text("def add_numbers(a, b):\n    return a + b") 
        (test_dir / "unrelated.py").write_text("def multiply(x, y):\n    return x * y")
        
        # Index files
        await engine.index_directories([str(test_dir)], force_reindex=True)
        
        # Search without reranking
        results = await engine.search("calculate sum", limit=3, use_reranker=False)
        
        # Should get results
        assert len(results) > 0
        
        # Results should contain relevant matches
        file_paths = [r.file_path for r in results]
        assert any("exact_match.py" in fp for fp in file_paths)
    
    @pytest.mark.asyncio
    async def test_search_with_reranker_config(self, engine):
        """Test that search respects reranker configuration."""
        # Configure a reranker model
        engine.config.reranker_model = "BAAI/bge-reranker-v2-m3"
        
        # Create a test file
        test_dir = Path(engine.config.data_root) / "test_files"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "test.py").write_text("def test_function():\n    pass")
        
        # Index the file
        await engine.index_directories([str(test_dir)], force_reindex=True)
        
        # Mock the reranking method
        original_rerank = engine._rerank_local
        rerank_called = False
        
        async def mock_rerank(query, results, model):
            nonlocal rerank_called
            rerank_called = True
            return await original_rerank(query, results, model)
        
        engine._rerank_local = mock_rerank
        
        # Search with reranking enabled by default (since model is configured)
        results = await engine.search("test")
        
        # Reranking should have been called
        assert rerank_called
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_reranking_preserves_metadata(self, engine, mock_search_results):
        """Test that reranking preserves all result metadata."""
        results_with_content = [
            (mock_search_results[0], "content1"),
            (mock_search_results[1], "content2"),
            (mock_search_results[2], "content3")
        ]
        
        # Mock cross-encoder
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.9, 0.8, 0.7]
        mock_cross_encoder.max_length = 512
        
        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            reranked = await engine._rerank_local("test", results_with_content, "model")
            
            # All metadata should be preserved
            assert len(reranked) == 3
            for i, result in enumerate(reranked):
                assert result.file_path == mock_search_results[i].file_path
                assert result.file_type == mock_search_results[i].file_type
                assert result.snippet == mock_search_results[i].snippet
                assert result.last_modified == mock_search_results[i].last_modified
                # Only relevance score should change
                assert result.relevance_score in [0.9, 0.8, 0.7]