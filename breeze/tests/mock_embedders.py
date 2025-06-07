"""Mock embedding functions that simulate API behavior for tests."""

import numpy as np
from typing import List, Union, Dict, Any
import pyarrow as pa
from lancedb.embeddings import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from pydantic import PrivateAttr
import asyncio


class MockVoyageEmbedder(TextEmbeddingFunction):
    """Mock Voyage embedder that simulates rate limiting and chunking behavior."""

    # Private attributes using Pydantic
    _ndims: int = PrivateAttr(default=1024)
    _model_name: str = PrivateAttr(default="voyage-code-3")
    _simulate_rate_limits: bool = PrivateAttr(default=False)
    _call_count: int = PrivateAttr(default=0)
    _tokens_used: int = PrivateAttr(default=0)
    _rate_limit_at_calls: List[int] = PrivateAttr(default_factory=list)

    def __init__(
        self, ndims=1024, model_name="voyage-code-3", simulate_rate_limits=False
    ):
        super().__init__()
        self._ndims = ndims
        self._model_name = model_name
        self._simulate_rate_limits = simulate_rate_limits
        self._call_count = 0
        self._tokens_used = 0
        self._rate_limit_at_calls = []
        # Store original args for LanceDB compatibility
        self._original_args = {
            "ndims": ndims,
            "model_name": model_name,
            "simulate_rate_limits": simulate_rate_limits,
        }

    @property
    def name(self):
        return self._model_name

    def ndims(self):
        return self._ndims

    def set_rate_limit_behavior(self, fail_at_calls: List[int]):
        """Configure which calls should simulate rate limit errors."""
        self._rate_limit_at_calls = fail_at_calls

    def generate_embeddings(
        self, texts: Union[List[str], np.ndarray]
    ) -> List[np.array]:
        """Generate embeddings with optional rate limit simulation."""
        self._call_count += 1

        # Simulate rate limiting at specific calls
        if self._simulate_rate_limits and self._call_count in self._rate_limit_at_calls:
            raise Exception("rate_limit_exceeded")

        # Estimate tokens (simple approximation)
        if isinstance(texts, (list, tuple)):
            self._tokens_used += sum(len(str(t).split()) * 2 for t in texts)

        # Return deterministic embeddings based on text for consistency
        embeddings = []
        for i, text in enumerate(texts):
            # Create a deterministic embedding based on text hash
            hash_val = hash(str(text)) % 1000000
            np.random.seed(hash_val)
            embedding = np.random.rand(self._ndims)
            embeddings.append(embedding)

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

    def SourceField(self, **kwargs):
        """
        Creates a pydantic Field that can automatically annotate
        the source column for this embedding function
        """
        from pydantic import Field

        return Field(json_schema_extra={"source_column_for": self}, **kwargs)

    def VectorField(self, **kwargs):
        """
        Creates a pydantic Field that can automatically annotate
        the target vector column for this embedding function
        """
        from pydantic import Field

        return Field(json_schema_extra={"vector_column_for": self}, **kwargs)


class MockVoyageEmbedderAsync:
    """Async version that works with get_voyage_embeddings_with_limits."""

    def __init__(self, ndims=1024, model_name="voyage-code-3"):
        self._ndims = ndims
        self.name = model_name
        self.model_name = model_name
        self.call_count = 0
        self.tokens_used = 0
        self.rate_limit_at_calls = []

    def ndims(self):
        return self._ndims

    def set_rate_limit_behavior(self, fail_at_calls: List[int]):
        """Configure which calls should simulate rate limit errors."""
        self.rate_limit_at_calls = fail_at_calls

    async def embed(
        self, texts: List[str], model: str = None, input_type: str = "document"  # noqa: ARG002
    ) -> Dict[str, Any]:
        """Async embed method that mimics Voyage API response."""
        self.call_count += 1

        # Simulate rate limiting at specific calls
        if self.call_count in self.rate_limit_at_calls:
            # Return a rate limit error response
            raise Exception("rate_limit_exceeded")

        # Simulate small processing delay
        await asyncio.sleep(0.001)

        # Generate embeddings
        embeddings = []
        for text in texts:
            hash_val = hash(str(text)) % 1000000
            np.random.seed(hash_val)
            embedding = np.random.rand(self._ndims).tolist()
            embeddings.append(embedding)

        # Return Voyage-style response
        return {
            "embeddings": embeddings,
            "model": model or self.model_name,
            "usage": {
                "prompt_tokens": sum(len(str(t).split()) * 2 for t in texts),
                "total_tokens": sum(len(str(t).split()) * 2 for t in texts),
            },
        }

    def compute_source_embeddings(self, texts):
        """Sync method for LanceDB compatibility."""
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pa.Array):
            texts = texts.to_pylist()
        elif isinstance(texts, pa.ChunkedArray):
            texts = texts.combine_chunks().to_pylist()

        # Run async method in sync context
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.embed(texts))
            return [np.array(emb) for emb in result["embeddings"]]
        finally:
            loop.close()

    def compute_query_embeddings(self, query):
        """Compute query embeddings."""
        return self.compute_source_embeddings([query])


class MockLocalEmbedder(TextEmbeddingFunction):
    """Mock local embedder that's extremely fast."""

    # Public fields that can be configured
    max_seq_length: int = 512

    # Private attributes using Pydantic
    _ndims: int = PrivateAttr(default=384)
    _model_name: str = PrivateAttr(default="sentence-transformers/all-MiniLM-L6-v2")

    def __init__(self, ndims=384, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self._ndims = ndims
        self._model_name = model_name
        # Store original args for LanceDB compatibility
        self._original_args = {"ndims": ndims, "model_name": model_name}

    @property
    def name(self):
        return self._model_name

    def ndims(self):
        return self._ndims

    def generate_embeddings(
        self, texts: Union[List[str], np.ndarray]
    ) -> List[np.array]:
        """Generate embeddings instantly."""
        embeddings = []
        for text in texts:
            # Create deterministic embedding
            hash_val = hash(str(text)) % 1000000
            np.random.seed(hash_val)
            embedding = np.random.rand(self._ndims)
            # Normalize like sentence-transformers does
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
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

    def SourceField(self, **kwargs):
        """
        Creates a pydantic Field that can automatically annotate
        the source column for this embedding function
        """
        from pydantic import Field

        return Field(json_schema_extra={"source_column_for": self}, **kwargs)

    def VectorField(self, **kwargs):
        """
        Creates a pydantic Field that can automatically annotate
        the target vector column for this embedding function
        """
        from pydantic import Field

        return Field(json_schema_extra={"vector_column_for": self}, **kwargs)

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:  # noqa: ARG002
        """Mimic sentence-transformers encode method."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.generate_embeddings(texts)
        return np.array(embeddings)


# Register mock embedders for easy access
@register("mock-voyage")
class RegisteredMockVoyageEmbedder(MockVoyageEmbedder):
    """Registered version of mock Voyage embedder."""

    pass


@register("mock-local")
class RegisteredMockLocalEmbedder(MockLocalEmbedder):
    """Registered version of mock local embedder."""

    pass


class MockReranker:
    """Mock reranker for LanceDB compatibility."""
    
    def __init__(self, model_name: str = "mock-reranker", column: str = "content"):
        self.model_name = model_name
        self.column = column
        self.max_length = 512
    
    def _rerank(self, result_set, query: str):
        """Internal rerank method that scores based on simple text similarity."""
        import pyarrow as pa
        
        if len(result_set) == 0:
            # Add empty _relevance_score column for empty results
            return result_set.append_column(
                "_relevance_score", pa.array([], type=pa.float32())
            )
        
        # Get text content from the specified column
        passages = result_set[self.column].to_pylist()
        
        # Calculate simple similarity scores
        scores = []
        for text in passages:
            # Simple similarity based on common words
            query_words = set(query.lower().split())
            text_words = set(str(text).lower().split())
            overlap = len(query_words.intersection(text_words))
            total_words = len(query_words.union(text_words))
            # Score between 0 and 1
            score = overlap / max(total_words, 1)
            scores.append(score)
        
        # Add relevance scores to the result set
        result_set = result_set.append_column(
            "_relevance_score", pa.array(scores, type=pa.float32())
        )
        
        return result_set
    
    def rerank_hybrid(self, query: str, vector_results, fts_results):
        """Rerank hybrid search results."""
        import pyarrow as pa
        
        # Merge vector and FTS results (simple concatenation)
        if len(vector_results) > 0 and len(fts_results) > 0:
            combined_results = pa.concat_tables([vector_results, fts_results])
        elif len(vector_results) > 0:
            combined_results = vector_results
        elif len(fts_results) > 0:
            combined_results = fts_results
        else:
            # Create empty table with _relevance_score column
            return pa.Table.from_arrays([], names=[]).append_column(
                "_relevance_score", pa.array([], type=pa.float32())
            )
        
        # Rerank the combined results
        combined_results = self._rerank(combined_results, query)
        
        # Sort by relevance score (descending)
        combined_results = combined_results.sort_by([("_relevance_score", "descending")])
        
        return combined_results
    
    def rerank_vector(self, query: str, vector_results):
        """Rerank vector search results."""
        vector_results = self._rerank(vector_results, query)
        vector_results = vector_results.sort_by([("_relevance_score", "descending")])
        return vector_results
    
    def rerank_fts(self, query: str, fts_results):
        """Rerank FTS search results."""
        fts_results = self._rerank(fts_results, query)
        fts_results = fts_results.sort_by([("_relevance_score", "descending")])
        return fts_results
    
    def predict(self, pairs):
        """Sentence-transformers style predict method for backward compatibility."""
        scores = []
        for query, text in pairs:
            # Simple similarity based on common words
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            overlap = len(query_words.intersection(text_words))
            total_words = len(query_words.union(text_words))
            # Score between 0 and 1
            score = overlap / max(total_words, 1)
            scores.append(score)
        return scores
