"""Simple test to verify the rate limit fix."""

import numpy as np
from breeze.core.embeddings import get_voyage_embeddings_with_limits
from unittest.mock import MagicMock


def test_rate_limit_returns_partial_results():
    """Test that function returns partial results instead of raising exception."""
    
    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.encode.return_value.ids = [0] * 100  # 100 tokens per text
    
    # Mock model that fails sometimes
    model = MagicMock()
    call_count = 0
    
    def mock_compute(texts):
        nonlocal call_count
        call_count += 1
        # Fail on second batch persistently
        if call_count in [2, 3]:  # Will fail with retries
            raise Exception("429 Too Many Requests")
        return np.random.rand(len(texts), 768)
    
    model.compute_source_embeddings = mock_compute
    
    # Create enough texts to ensure multiple batches
    # With 100 tokens each and 120k limit, ~1200 texts per batch
    # But there's also a 128 text limit per batch
    texts = ["text"] * 200  # Should create at least 2 batches
    
    # Run synchronously for simplicity
    import asyncio
    result = asyncio.run(get_voyage_embeddings_with_limits(
        texts,
        model,
        tokenizer,
        max_concurrent_requests=1,  # Sequential for predictability
        max_retries=2,  # Low retries
        retry_base_delay=0.001,
        tokens_per_minute=1000000,
        requests_per_minute=1000,
    ))
    
    print(f"Result type: {type(result)}")
    print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
    
    # Verify we get a dict with partial results
    assert isinstance(result, dict)
    assert 'embeddings' in result
    assert 'failed_batches' in result
    assert 'successful_batches' in result
    
    # Should have some failures
    assert len(result['failed_batches']) > 0
    print(f"Failed batches: {result['failed_batches']}")
    print(f"Successful batches: {result['successful_batches']}")
    

if __name__ == "__main__":
    test_rate_limit_returns_partial_results()