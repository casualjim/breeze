"""Shared pytest configuration and fixtures for Breeze tests."""

import os
# Disable tokenizer parallelism to avoid fork warnings/deadlocks in tests
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytest
import pytest_asyncio
import numpy as np
from lancedb.embeddings import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from typing import List, Union
import pyarrow as pa

# Import mock embedders
from breeze.tests.mock_embedders import MockVoyageEmbedderAsync, MockReranker


# Create a test embedding function that avoids real model loading
@register("test-embedder")
class TestEmbeddingFunction(TextEmbeddingFunction):
    """Fast test embedding function that returns random embeddings."""

    def __init__(self, ndims=768):
        self._ndims = ndims
        self.name = "test-embedder"

    def ndims(self):
        return self._ndims

    def generate_embeddings(
        self, texts: Union[List[str], np.ndarray]
    ) -> List[np.array]:
        """Generate random embeddings for testing."""
        embeddings = []
        for _ in texts:
            embeddings.append(np.random.rand(self._ndims))
        return embeddings

    def compute_source_embeddings(self, texts):
        """Compute source embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pa.Array):
            texts = texts.to_pylist()
        elif isinstance(texts, pa.ChunkedArray):
            texts = texts.combine_chunks().to_pylist()
        return self.generate_embeddings(texts)

    def compute_query_embeddings(self, query):
        """Compute query embeddings."""
        return self.generate_embeddings([query])


@pytest.fixture(scope="session")
def fast_embedder_768():
    """Fast 768-dimensional embedder for tests."""
    return TestEmbeddingFunction(768)


@pytest.fixture(scope="session")
def fast_embedder_384():
    """Fast 384-dimensional embedder for tests (all-MiniLM-L6-v2 compatible)."""
    return TestEmbeddingFunction(384)


@pytest.fixture(scope="session")
def fast_embedder_1024():
    """Fast 1024-dimensional embedder for tests (voyage compatible)."""
    return TestEmbeddingFunction(1024)


@pytest_asyncio.fixture
async def clean_engine_shutdown(request):
    """Ensure engines are properly shut down after tests."""
    engines = []

    def register_engine(engine):
        engines.append(engine)

    yield register_engine

    # Cleanup
    for engine in engines:
        try:
            await engine.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup


@pytest_asyncio.fixture(autouse=True)
async def auto_cleanup_pending_tasks():
    """Automatically cleanup any pending asyncio tasks after each test."""
    yield
    
    # Cancel any pending tasks after test completion
    import asyncio
    
    # Get the current event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, nothing to clean up
        return
    
    # Get all tasks for the current loop
    pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
    
    # Don't cancel the current task (the cleanup task itself)
    current_task = asyncio.current_task()
    pending_tasks = [task for task in pending_tasks if task != current_task]
    
    if pending_tasks:
        # Give tasks a chance to complete gracefully
        for task in pending_tasks:
            if not task.cancelled():
                task.cancel()
        
        # Wait briefly for cancellation to complete
        if pending_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True), 
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                pass  # Some tasks may not respond to cancellation


@pytest_asyncio.fixture
async def fast_engine(tmp_path, clean_engine_shutdown):
    """Create a BreezeEngine with fast test embedder for general tests."""
    from breeze.core.engine import BreezeEngine
    from breeze.core.config import BreezeConfig
    from lancedb.embeddings.registry import get_registry

    # Use registered MockLocalEmbedder for 384 dimensions
    registry = get_registry()
    test_embedder = registry.get("mock-local").create()

    config = BreezeConfig(
        data_root=str(tmp_path),
        db_name="test_db",
        embedding_function=test_embedder,
    )

    engine = BreezeEngine(config)
    # Use mock reranker to avoid slowdown
    engine.reranker = MockReranker()
    await engine.initialize()
    clean_engine_shutdown(engine)

    yield engine


@pytest_asyncio.fixture
async def fast_engine_voyage(tmp_path, clean_engine_shutdown):
    """Create a BreezeEngine with fast test embedder for voyage tests."""
    from breeze.core.engine import BreezeEngine
    from breeze.core.config import BreezeConfig
    from lancedb.embeddings.registry import get_registry

    # Use registered MockVoyageEmbedder for 1024 dimensions
    registry = get_registry()
    test_embedder = registry.get("mock-voyage").create()

    config = BreezeConfig(
        data_root=str(tmp_path),
        db_name="test_db",
        embedding_function=test_embedder,
        voyage_tier=1,
        voyage_concurrent_requests=3,
    )

    engine = BreezeEngine(config)
    # Use mock reranker to avoid slowdown
    engine.reranker = MockReranker()
    await engine.initialize()
    clean_engine_shutdown(engine)

    yield engine


@pytest.fixture
def mock_voyage_embedder():
    """Mock Voyage embedder for rate limit testing."""
    from lancedb.embeddings.registry import get_registry

    registry = get_registry()
    embedder = registry.get("mock-voyage").create()
    embedder.set_rate_limit_behavior([2, 5])  # Simulate rate limits at calls 2 and 5
    return embedder


@pytest.fixture
def mock_voyage_embedder_async():
    """Async mock Voyage embedder for rate limit testing."""
    return MockVoyageEmbedderAsync(ndims=1024)


@pytest.fixture
def mock_local_embedder():
    """Mock local embedder for fast testing."""
    from lancedb.embeddings.registry import get_registry

    registry = get_registry()
    return registry.get("mock-local").create()
