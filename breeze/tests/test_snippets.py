"""Tests for the snippet extraction module."""

import pytest
from breeze.core.snippets import TreeSitterSnippetExtractor, SnippetConfig


class TestTreeSitterSnippetExtractor:
    """Test the TreeSitterSnippetExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a snippet extractor instance."""
        config = SnippetConfig(
            max_snippet_length=500,
            context_lines=2,
            max_complete_lines=20,
            head_lines=5,
            tail_lines=3
        )
        return TreeSitterSnippetExtractor(config)
    
    def test_simple_extraction_fallback(self, extractor):
        """Test simple extraction when tree-sitter is not available."""
        content = """def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
    
def main():
    hello()
    goodbye()
"""
        
        # Extract snippet around "goodbye"
        snippet = extractor.extract_snippet(content, "goodbye")
        
        # Should include context around the goodbye function
        assert "def goodbye():" in snippet
        assert "print(\"Goodbye!\")" in snippet
        # Should have some context (with context_lines=2, should see 2 lines before and after)
        assert "print(\"Hello, World!\")" in snippet  # 2 lines before goodbye def
    
    def test_extract_with_language_hint(self, extractor):
        """Test extraction with language hint."""
        # Skip if tree-sitter-languages not installed
        extractor.initialize()
        if not extractor._initialized:
            pytest.skip("tree-sitter-languages not installed")
        
        content = """class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        '''Add two numbers'''
        return x + y
    
    def multiply(self, x, y):
        '''Multiply two numbers'''
        return x * y
    
    def divide(self, x, y):
        '''Divide two numbers'''
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
"""
        
        # Extract snippet for "divide"
        snippet = extractor.extract_snippet(content, "divide", language="python")
        
        # Should get the complete divide method
        assert "def divide(self, x, y):" in snippet
        assert "Cannot divide by zero" in snippet
        assert "return x / y" in snippet
    
    def test_large_function_truncation(self, extractor):
        """Test smart truncation of large functions."""
        # Create a large function
        lines = ["def large_function():"]
        lines.append("    '''A very large function'''")
        
        # Add many lines
        for i in range(50):
            lines.append(f"    line_{i} = {i}  # Line {i}")
        
        # Add the target line in the middle
        lines.insert(25, "    target_variable = 'FIND_ME'")
        
        lines.append("    return result")
        
        content = "\n".join(lines)
        
        snippet = extractor.extract_snippet(content, "target_variable", language="python")
        
        # Should include the target
        assert "target_variable = 'FIND_ME'" in snippet
        
        # Should be truncated (not include all 50+ lines)
        assert len(snippet.split('\n')) < 30
        
        # Should have truncation markers
        assert "lines omitted" in snippet
    
    def test_context_extraction(self, extractor):
        """Test extraction maintains context."""
        content = """def process_data(data):
    # Validate input
    if not data:
        raise ValueError("No data provided")
    
    # Process each item
    results = []
    for item in data:
        processed = transform(item)
        results.append(processed)
    
    # Return results
    return results
"""
        
        snippet = extractor.extract_snippet(content, "transform")
        
        # Should include the transform line and context
        assert "transform(item)" in snippet
        assert "for item in data:" in snippet
        assert "results.append(processed)" in snippet