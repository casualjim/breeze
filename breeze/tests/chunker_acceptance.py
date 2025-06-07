"""Acceptance tests for TextChunker.

These tests define the expected behavior and requirements that must be met
for the text chunker to be considered working correctly.
"""

import pytest
from typing import List
import numpy as np

from breeze.core.text_chunker import (
    TextChunker,
    ChunkingConfig,
    FileContent,
    TextChunk,
    ChunkedFile,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs):
        # Simple mock: ~4 chars per token
        tokens = text.split()
        # Return mock token IDs
        return list(range(len(tokens)))

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs
    ) -> str:
        # Simple mock implementation
        return " ".join([f"token_{i}" for i in token_ids])


class TestSemanticChunkingBehavior:
    """Tests for semantic chunking using tree-sitter."""

    def test_chunks_at_function_boundaries(self):
        """Chunks should align with function boundaries when possible."""
        code = """
def hello():
    print("Hello, world!")
    return True

def goodbye():
    print("Goodbye!")
    return False

class Greeter:
    def __init__(self):
        self.name = "Greeter"
    
    def greet(self):
        print(f"Hello from {self.name}")
"""
        chunker = TextChunker(config=ChunkingConfig(max_tokens=50))
        file_content = FileContent(content=code, file_path="test.py", language="python")
        result = chunker.chunk_file(file_content)

        # Should create chunks that align with semantic boundaries
        assert len(result.chunks) >= 3  # At least one per top-level definition

        # Each chunk should contain complete semantic units
        for chunk in result.chunks:
            # Chunks should not split function definitions
            assert (
                chunk.text.count("def ") == chunk.text.count("return ")
                or "class " in chunk.text
            )

    def test_preserves_semantic_unit_metadata(self):
        """Chunks should preserve metadata about the semantic unit type."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b
"""
        chunker = TextChunker(config=ChunkingConfig(max_tokens=100))
        file_content = FileContent(content=code, file_path="calc.py", language="python")
        result = chunker.chunk_file(file_content)

        # Chunks should have semantic metadata
        for chunk in result.chunks:
            # This test will fail with current implementation - chunks need metadata
            assert hasattr(chunk, "node_type") or hasattr(chunk, "semantic_type")
            if hasattr(chunk, "node_type"):
                assert chunk.node_type in [
                    "class_definition",
                    "function_definition",
                    "method_definition",
                ]

    def test_handles_nested_structures(self):
        """Should handle nested functions and classes correctly."""
        code = """
class Outer:
    class Inner:
        def inner_method(self):
            def nested_function():
                return 42
            return nested_function()
    
    def outer_method(self):
        return self.Inner()
"""
        chunker = TextChunker(config=ChunkingConfig(max_tokens=80))
        file_content = FileContent(
            content=code, file_path="nested.py", language="python"
        )
        result = chunker.chunk_file(file_content)

        # Should maintain hierarchical structure
        assert len(result.chunks) >= 2

        # Chunks should preserve nesting context
        for chunk in result.chunks:
            if "nested_function" in chunk.text:
                # The nested function should be within its parent context
                assert "inner_method" in chunk.text or "Inner" in chunk.text


class TestChunkOverlapAndContinuity:
    """Tests for chunk overlap and context continuity."""

    def test_semantic_chunks_have_overlap(self):
        """Semantic chunks should respect overlap configuration."""
        code = """
def first_function():
    # Long function that might need splitting
    x = 1
    y = 2
    z = 3
    result = x + y + z
    return result

def second_function():
    # Another function
    return 42
"""
        config = ChunkingConfig(max_tokens=30, overlap_tokens=10)
        chunker = TextChunker(config=config)
        file_content = FileContent(content=code, file_path="test.py", language="python")
        result = chunker.chunk_file(file_content)

        # Adjacent chunks should have overlap
        if len(result.chunks) > 1:
            for i in range(len(result.chunks) - 1):
                chunk1 = result.chunks[i]
                chunk2 = result.chunks[i + 1]
                # There should be some overlap in character positions
                assert (
                    chunk1.end_char > chunk2.start_char
                    or chunk2.text[:20] in chunk1.text
                )  # Some content overlap

    def test_character_chunking_respects_overlap(self):
        """Character-based chunking should implement proper overlap."""
        # Use plain text to force character chunking
        text = "A" * 1000  # Long text that requires multiple chunks

        config = ChunkingConfig(max_tokens=50, overlap_tokens=10, stride_ratio=0.8)
        chunker = TextChunker(config=config)
        file_content = FileContent(content=text, file_path="test.txt", language="text")
        result = chunker.chunk_file(file_content)

        assert len(result.chunks) > 1

        # Verify overlap exists
        for i in range(len(result.chunks) - 1):
            chunk1 = result.chunks[i]
            chunk2 = result.chunks[i + 1]
            # With stride_ratio=0.8, there should be 20% overlap
            overlap_start = chunk2.start_char
            overlap_end = chunk1.end_char
            if overlap_end > overlap_start:
                overlap_size = overlap_end - overlap_start
                chunk1_size = chunk1.end_char - chunk1.start_char
                overlap_ratio = overlap_size / chunk1_size
                assert overlap_ratio >= 0.15  # Allow some tolerance


class TestLargeSemanticUnits:
    """Tests for handling semantic units that exceed token limits."""

    def test_splits_large_functions_intelligently(self):
        """Large functions should be split while preserving context."""
        # Create a large function
        code = '''
def process_data(data):
    """Process data with many steps."""
    # Step 1: Validate
    if not data:
        raise ValueError("No data")
    
    # Step 2: Transform
    transformed = []
    for item in data:
        if isinstance(item, dict):
            transformed.append(item)
    
    # Step 3: Filter
    filtered = []
    for item in transformed:
        if item.get('active'):
            filtered.append(item)
    
    # Step 4: Sort
    sorted_data = sorted(filtered, key=lambda x: x.get('priority', 0))
    
    # Step 5: Format
    result = []
    for item in sorted_data:
        formatted = {
            'id': item.get('id'),
            'name': item.get('name'),
            'value': item.get('value')
        }
        result.append(formatted)
    
    return result
'''
        config = ChunkingConfig(max_tokens=50)  # Force splitting
        chunker = TextChunker(config=config)
        file_content = FileContent(
            content=code, file_path="large.py", language="python"
        )
        result = chunker.chunk_file(file_content)

        # Should create multiple chunks
        assert len(result.chunks) > 1

        # Each chunk should maintain context about being part of the same function
        for chunk in result.chunks:
            # This will fail with current implementation - need parent context
            assert hasattr(chunk, "parent_unit") or hasattr(chunk, "original_unit_type")

    def test_preserves_semantic_boundaries_when_splitting(self):
        """When splitting large units, should try to split at logical boundaries."""
        code = """
def complex_function():
    # Part 1
    x = calculate_x()
    y = calculate_y()
    
    # Part 2
    if x > y:
        result = process_greater(x, y)
    else:
        result = process_lesser(x, y)
    
    # Part 3
    formatted = format_result(result)
    validated = validate_result(formatted)
    
    return validated
"""
        config = ChunkingConfig(max_tokens=40)
        chunker = TextChunker(config=config)
        file_content = FileContent(content=code, file_path="test.py", language="python")
        result = chunker.chunk_file(file_content)

        # Chunks should split at comment boundaries when possible
        for chunk in result.chunks:
            # Each chunk should contain complete statements
            # (no splits in the middle of if/else blocks)
            if "if x > y:" in chunk.text:
                assert "else:" in chunk.text  # Should keep if/else together


class TestPolyglotSupport:
    """Tests for multi-language support."""

    def test_handles_multiple_languages(self):
        """Should correctly chunk files in different languages."""
        test_cases = [
            ("test.py", "python", "def hello(): return 'world'"),
            ("test.js", "javascript", "function hello() { return 'world'; }"),
            (
                "test.java",
                "java",
                'public class Test { public String hello() { return "world"; } }',
            ),
            ("test.go", "go", 'func hello() string { return "world" }'),
            ("test.rs", "rust", 'fn hello() -> &\'static str { "world" }'),
        ]

        chunker = TextChunker(config=ChunkingConfig(max_tokens=100))

        for filename, language, code in test_cases:
            file_content = FileContent(
                content=code, file_path=filename, language=language
            )
            result = chunker.chunk_file(file_content)

            assert len(result.chunks) >= 1
            assert result.chunks[0].text.strip() == code.strip()

            # Should include language metadata
            for chunk in result.chunks:
                # This will fail - chunks need language info
                assert hasattr(chunk, "language") and chunk.language == language

    def test_token_estimation_varies_by_language(self):
        """Token estimation should account for language differences."""
        # Chinese text typically has different token density
        test_cases = [
            ("english.txt", "text", "Hello world this is a test", 6),  # ~6 tokens
            (
                "chinese.txt",
                "text",
                "你好世界这是一个测试",
                8,
            ),  # More tokens for Chinese
            ("code.py", "python", "def x(): return 1", 5),  # Code tokens
        ]

        chunker = TextChunker()

        for filename, language, text, expected_min_tokens in test_cases:
            file_content = FileContent(
                content=text, file_path=filename, language=language
            )
            estimated = chunker.estimate_tokens(text)
            # Should have language-aware estimation
            assert estimated >= expected_min_tokens // 2  # Allow some variance


class TestChunkMetadataAndRelationships:
    """Tests for chunk metadata and relationships."""

    def test_chunks_maintain_file_relationship(self):
        """Chunks should maintain relationship to source file."""
        code = "def test(): pass"
        chunker = TextChunker()
        file_content = FileContent(
            content=code, file_path="/src/module/test.py", language="python"
        )
        result = chunker.chunk_file(file_content)

        for chunk in result.chunks:
            # Chunks should know their source
            assert (
                hasattr(chunk, "source_file")
                or result.source.file_path == "/src/module/test.py"
            )

    def test_chunks_have_unique_identifiers(self):
        """Each chunk should have a unique identifier within the file."""
        code = """
def first(): pass
def second(): pass
def third(): pass
"""
        chunker = TextChunker(config=ChunkingConfig(max_tokens=20))
        file_content = FileContent(content=code, file_path="test.py", language="python")
        result = chunker.chunk_file(file_content)

        # Check chunk indices are unique and sequential
        indices = [chunk.chunk_index for chunk in result.chunks]
        assert indices == list(range(len(result.chunks)))

        # All chunks should know total chunk count
        for chunk in result.chunks:
            assert chunk.total_chunks == len(result.chunks)

    def test_adjacent_chunks_are_marked(self):
        """Chunks should know about their neighbors for context."""
        code = "x = 1\n" * 100  # Force multiple chunks
        chunker = TextChunker(config=ChunkingConfig(max_tokens=20))
        file_content = FileContent(content=code, file_path="test.py", language="python")
        result = chunker.chunk_file(file_content)

        assert len(result.chunks) > 2

        for i, chunk in enumerate(result.chunks):
            # This will fail - chunks need neighbor info
            if i > 0:
                assert hasattr(chunk, "has_previous") and chunk.has_previous
            if i < len(result.chunks) - 1:
                assert hasattr(chunk, "has_next") and chunk.has_next


class TestErrorHandlingAndRobustness:
    """Tests for error handling and edge cases."""

    def test_handles_empty_files(self):
        """Should handle empty files gracefully."""
        chunker = TextChunker()
        file_content = FileContent(content="", file_path="empty.py", language="python")
        result = chunker.chunk_file(file_content)

        assert len(result.chunks) == 0
        assert result.total_tokens == 0

    def test_handles_whitespace_only_files(self):
        """Should handle files with only whitespace."""
        chunker = TextChunker()
        file_content = FileContent(
            content="   \n\t\n   ", file_path="ws.py", language="python"
        )
        result = chunker.chunk_file(file_content)

        assert len(result.chunks) == 0

    def test_handles_unsupported_languages(self):
        """Should fall back gracefully for unsupported languages."""
        chunker = TextChunker()
        file_content = FileContent(
            content="Some content in unknown language",
            file_path="test.xyz",
            language="unknown_lang",
        )
        result = chunker.chunk_file(file_content)

        # Should still produce chunks using character-based method
        assert len(result.chunks) >= 1
        assert result.chunks[0].text == "Some content in unknown language"

        # Should indicate fallback method was used
        for chunk in result.chunks:
            # This will fail - need to track chunking method
            assert (
                hasattr(chunk, "chunking_method")
                and chunk.chunking_method == "character"
            )

    def test_handles_malformed_code(self):
        """Should handle syntactically invalid code."""
        code = """
def broken_function(
    # Missing closing parenthesis and body
class AlsoBroken
    # Missing colon and body
"""
        chunker = TextChunker()
        file_content = FileContent(
            content=code, file_path="broken.py", language="python"
        )
        result = chunker.chunk_file(file_content)

        # Should still produce chunks even if parsing fails
        assert len(result.chunks) >= 1
        assert "broken_function" in result.chunks[0].text

    def test_handles_binary_content_gracefully(self):
        """Should handle files with binary/non-UTF8 content."""
        # Simulate binary content with null bytes
        content = "def test():\n\x00\x01\x02\npass"
        chunker = TextChunker()
        file_content = FileContent(
            content=content, file_path="mixed.py", language="python"
        )

        # Should not crash
        result = chunker.chunk_file(file_content)
        assert len(result.chunks) >= 1


class TestEmbeddingCombination:
    """Tests for embedding combination methods."""

    def test_combines_embeddings_with_all_methods(self):
        """Should support all documented combination methods."""
        chunker = TextChunker()

        # Create mock embeddings
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
        ]

        # Create mock chunks
        chunks = [
            TextChunk("chunk1", 0, 10, 0, 3, estimated_tokens=10),
            TextChunk("chunk2", 10, 20, 1, 3, estimated_tokens=20),
            TextChunk("chunk3", 20, 30, 2, 3, estimated_tokens=15),
        ]

        # Test all methods
        methods = ["average", "weighted_average", "first", "last"]
        for method in methods:
            result = chunker.combine_chunk_embeddings(embeddings, chunks, method)
            assert result.shape == (3,)

        # Test weighted average specifically
        weighted = chunker.combine_chunk_embeddings(
            embeddings, chunks, "weighted_average"
        )
        # Should be weighted by token count (10, 20, 15)
        expected = (embeddings[0] * 10 + embeddings[1] * 20 + embeddings[2] * 15) / 45
        np.testing.assert_allclose(weighted, expected)

    def test_structure_aware_combination(self):
        """Should support structure-aware embedding combination."""
        chunker = TextChunker()

        # Create chunks with semantic metadata
        chunks = [
            TextChunk("class X:", 0, 10, 0, 3, estimated_tokens=10),
            TextChunk("def method():", 10, 25, 1, 3, estimated_tokens=15),
            TextChunk("# comment", 25, 35, 2, 3, estimated_tokens=5),
        ]

        # Add semantic metadata (this will fail without implementation)
        chunks[0].node_type = "class_definition"
        chunks[1].node_type = "function_definition"
        chunks[2].node_type = "comment"

        embeddings = [np.ones(3) for _ in chunks]

        # Should support structure-aware combination
        result = chunker.combine_chunk_embeddings(
            embeddings, chunks, method="structure_aware"
        )
        assert result.shape == (3,)


class TestBatchProcessing:
    """Tests for batch processing functionality."""

    def test_creates_batches_correctly(self):
        """Should create batches with proper metadata."""
        from breeze.core.text_chunker import create_batches_from_chunked_files

        # Create test data
        chunked_files = []
        for i in range(3):
            file_content = FileContent(
                content=f"file {i} content", file_path=f"file{i}.py", language="python"
            )
            chunks = [
                TextChunk(f"chunk {j}", j * 10, (j + 1) * 10, j, 2, 10)
                for j in range(2)
            ]
            chunked_files.append(ChunkedFile(file_content, chunks))

        # Create batches
        batches = create_batches_from_chunked_files(chunked_files, batch_size=3)

        assert len(batches) == 2  # 6 chunks total, batch size 3
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3

        # Check batch structure
        for batch in batches:
            for file_idx, file_content, chunk in batch:
                assert isinstance(file_idx, int)
                assert isinstance(file_content, FileContent)
                assert isinstance(chunk, TextChunk)


class TestIncrementalAndCaching:
    """Tests for incremental processing and caching."""

    def test_supports_incremental_chunking(self):
        """Should support chunking only changed files."""
        # This is a future feature test
        chunker = TextChunker()

        # Should be able to check if rechunking is needed
        file_content = FileContent(
            content="def test(): pass", file_path="test.py", language="python"
        )

        # This will fail without implementation
        assert hasattr(chunker, "needs_rechunking") or hasattr(
            chunker, "chunk_if_changed"
        )

    def test_caches_tree_sitter_parsers(self):
        """Should cache parsers for performance."""
        chunker = TextChunker()

        # Parse Python file
        file1 = FileContent("def test(): pass", "test1.py", "python")
        result1 = chunker.chunk_file(file1)

        # Parse another Python file - should reuse parser
        file2 = FileContent("def another(): pass", "test2.py", "python")
        result2 = chunker.chunk_file(file2)

        # Check parser was cached
        assert len(chunker._parsers) >= 1
        assert "python" in chunker._parsers


class TestAdvancedPoolingSupport:
    """Tests for advanced pooling method support."""

    def test_supports_norm_based_pooling(self):
        """Should support norm-based weighted pooling."""
        chunker = TextChunker()
        embeddings = [
            np.array([1.0, 0.0, 0.0]),  # norm = 1
            np.array([3.0, 4.0, 0.0]),  # norm = 5
            np.array([0.0, 0.0, 2.0]),  # norm = 2
        ]
        chunks = [
            TextChunk(f"chunk{i}", i * 10, (i + 1) * 10, i, 3, 10) for i in range(3)
        ]

        result = chunker.combine_chunk_embeddings(
            embeddings, chunks, method="norm_weighted"
        )
        assert result.shape == (3,)
        # Higher norm embeddings should have more weight

    def test_supports_cross_lingual_pooling(self):
        """Should support cross-lingual pooling for polyglot codebases."""
        chunker = TextChunker()

        # Create chunks from different languages
        chunks = [
            TextChunk("def python():", 0, 15, 0, 4, 10),
            TextChunk("function js() {}", 15, 30, 1, 4, 12),
            TextChunk("def more_python():", 30, 50, 2, 4, 15),
            TextChunk("func go() {}", 50, 65, 3, 4, 10),
        ]

        # Add language metadata
        chunks[0].language = "python"
        chunks[1].language = "javascript"
        chunks[2].language = "python"
        chunks[3].language = "go"

        embeddings = [np.ones(3) * (i + 1) for i in range(4)]

        result = chunker.combine_chunk_embeddings(
            embeddings, chunks, method="cross_lingual"
        )
        assert result.shape == (3,)

    def test_supports_entropy_based_pooling(self):
        """Should support entropy-based pooling."""
        chunker = TextChunker()

        # Create chunks with different complexity
        chunks = [
            TextChunk("x=1; y=2; z=3;", 0, 15, 0, 3, 10),  # Low entropy
            TextChunk("if x: func(a,b,c)", 15, 35, 1, 3, 15),  # Medium entropy
            TextChunk("# TODO: fix", 35, 50, 2, 3, 5),  # Low entropy
        ]

        embeddings = [np.ones(3) * (i + 1) for i in range(3)]

        result = chunker.combine_chunk_embeddings(
            embeddings, chunks, method="entropy_weighted"
        )
        assert result.shape == (3,)


class TestQueryConfiguration:
    """Tests for query-aware chunking configuration."""

    def test_configurable_semantic_granularity(self):
        """Should support configurable chunking granularity."""
        code = """
class MyClass:
    def method1(self):
        x = 1
        return x
    
    def method2(self):
        y = 2
        return y
"""
        # Chunk at class level
        config = ChunkingConfig(max_tokens=1000, semantic_level="class")
        chunker = TextChunker(config=config)
        file_content = FileContent(content=code, file_path="test.py", language="python")
        result = chunker.chunk_file(file_content)

        assert len(result.chunks) == 1  # Entire class in one chunk

        # Chunk at method level
        config = ChunkingConfig(max_tokens=100, semantic_level="function")
        chunker = TextChunker(config=config)
        result = chunker.chunk_file(file_content)

        assert len(result.chunks) >= 2  # Methods in separate chunks

    def test_semantic_unit_filtering(self):
        """Should support filtering certain semantic units."""
        code = '''
import os
import sys

# This is a comment
def process():
    """Docstring"""
    return 42

class Handler:
    pass
'''
        config = ChunkingConfig(max_tokens=100, exclude_units=["import", "comment"])
        chunker = TextChunker(config=config)
        file_content = FileContent(content=code, file_path="test.py", language="python")
        result = chunker.chunk_file(file_content)

        # Should not include imports or comments
        full_text = " ".join(chunk.text for chunk in result.chunks)
        assert "import os" not in full_text
        assert "# This is a comment" not in full_text
        assert "def process" in full_text


class TestChunkMerging:
    """Tests for intelligent chunk merging."""

    def test_merges_small_adjacent_chunks(self):
        """Should merge small adjacent chunks of the same type."""
        code = """
x = 1
y = 2

def tiny1():
    return 1

def tiny2():
    return 2

class Small:
    pass
"""
        config = ChunkingConfig(max_tokens=100, min_chunk_tokens=20, merge_similar=True)
        chunker = TextChunker(config=config)
        file_content = FileContent(content=code, file_path="test.py", language="python")
        result = chunker.chunk_file(file_content)

        # Small functions should be merged
        chunk_texts = [chunk.text for chunk in result.chunks]
        merged_functions = any(
            "tiny1" in text and "tiny2" in text for text in chunk_texts
        )
        assert merged_functions


class TestPerformanceOptimizations:
    """Tests for performance-related features."""

    def test_query_caching(self):
        """Should cache tree-sitter queries per language."""
        chunker = TextChunker()

        # Process multiple files of same language
        for i in range(3):
            code = f"def func{i}(): pass"
            file_content = FileContent(
                content=code, file_path=f"test{i}.py", language="python"
            )
            chunker.chunk_file(file_content)

        # Should have cached the query
        assert hasattr(chunker, "_query_cache") or hasattr(chunker, "_queries")

    def test_parallel_chunking_support(self):
        """Should support parallel processing of multiple files."""
        files = [
            FileContent(f"def func{i}(): pass", f"test{i}.py", "python")
            for i in range(10)
        ]

        chunker = TextChunker()

        # Should have method for parallel processing
        assert hasattr(chunker, "chunk_files_parallel") or hasattr(
            chunker, "chunk_files"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
