#!/usr/bin/env python3
"""Simple test script to verify Breeze functionality without MCP."""

import asyncio
import tempfile
from pathlib import Path
import pytest

from breeze.core import BreezeEngine, BreezeConfig


@pytest.mark.asyncio
async def test_breeze():
    """Test basic Breeze functionality."""
    # Create a temporary directory for the test database
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure Breeze to use the temp directory
        config = BreezeConfig(
            data_root=temp_dir,
            db_name="test_index"
        )
        
        # Create engine
        engine = BreezeEngine(config)
        
        # Create a test directory with some Python files
        test_dir = Path(temp_dir) / "test_code"
        test_dir.mkdir()
        
        # Create test files
        (test_dir / "hello.py").write_text('''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")
    return "Hello"

def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"
''')
        
        (test_dir / "math_utils.py").write_text('''
def factorial(n: int) -> int:
    """Calculate the factorial of n."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
''')
        
        (test_dir / "data_processor.py").write_text('''
import json
from typing import List, Dict, Any

class DataProcessor:
    """Process various data formats."""
    
    def __init__(self):
        self.data = []
    
    def load_json(self, filepath: str) -> List[Dict[str, Any]]:
        """Load data from a JSON file."""
        with open(filepath, 'r') as f:
            self.data = json.load(f)
        return self.data
    
    def process_data(self) -> Dict[str, Any]:
        """Process the loaded data."""
        if not self.data:
            return {"error": "No data loaded"}
        
        return {
            "count": len(self.data),
            "items": self.data
        }
''')
        
        print("Test files created successfully")
        
        # Test indexing
        print("\n1. Testing indexing...")
        stats = await engine.index_directories([str(test_dir)])
        print(f"Indexing stats: {stats.to_dict()}")
        
        # Test search
        print("\n2. Testing search...")
        
        # Search for factorial
        print("\nSearching for 'factorial':")
        results = await engine.search("factorial", limit=5)
        for result in results:
            print(f"  - {result.file_path} (score: {result.relevance_score:.3f})")
            print(f"    Snippet: {result.snippet[:100]}...")
        
        # Search for hello functions
        print("\nSearching for 'hello function':")
        results = await engine.search("hello function", limit=5)
        for result in results:
            print(f"  - {result.file_path} (score: {result.relevance_score:.3f})")
        
        # Search for JSON processing
        print("\nSearching for 'json data processing':")
        results = await engine.search("json data processing", limit=5)
        for result in results:
            print(f"  - {result.file_path} (score: {result.relevance_score:.3f})")
        
        # Get stats
        print("\n3. Getting index statistics...")
        stats = await engine.get_stats()
        print(f"Index stats: {stats}")
        
        print("\nAll tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_breeze())