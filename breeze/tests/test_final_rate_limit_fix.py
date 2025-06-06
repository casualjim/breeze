"""Final test demonstrating the rate limit fix works correctly."""

import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock

from breeze.core.embeddings import get_voyage_embeddings_with_limits


@pytest.mark.asyncio
async def test_rate_limit_fix():
    """Demonstrate that the fix allows partial results instead of complete failure."""
    
    print("\n=== Testing Rate Limit Fix ===\n")
    
    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.encode.return_value.ids = [0] * 1000  # 1000 tokens per text
    
    # Create texts that will be split into multiple batches
    # With 1000 tokens each and 120k limit, about 120 texts per batch
    texts = [f"Document {i}" for i in range(250)]  # Should create 3 batches
    
    # Mock model that simulates rate limiting
    model = MagicMock()
    call_count = 0
    
    def mock_compute(batch):
        nonlocal call_count
        call_count += 1
        print(f"API Call {call_count}: Processing {len(batch)} texts")
        
        # Simulate rate limit on batch 2 (after retries)
        if call_count in [2, 3, 4]:  # Batch 2 fails all retries
            print(f"  -> Rate limited! (attempt {call_count-1} for batch 2)")
            raise Exception("429 Too Many Requests")
        
        print(f"  -> Success!")
        return np.random.rand(len(batch), 768)
    
    model.compute_source_embeddings = mock_compute
    
    # Before fix: This would raise an exception and stop all processing
    # After fix: This returns partial results
    
    result = await get_voyage_embeddings_with_limits(
        texts,
        model,
        tokenizer,
        max_concurrent_requests=1,  # Sequential for clarity
        max_retries=3,
        retry_base_delay=0.01,
        tokens_per_minute=1000000,
        requests_per_minute=1000,
    )
    
    print(f"\n=== Results ===")
    print(f"Total texts: {len(texts)}")
    print(f"Embeddings returned: {len(result['embeddings'])}")
    print(f"Successful batches: {result['successful_batches']}")
    print(f"Failed batches: {result['failed_batches']}")
    print(f"Number of API calls made: {call_count}")
    
    # Verify the fix works
    assert isinstance(result, dict), "Should return dict with metadata"
    assert len(result['failed_batches']) > 0, "Should have some failed batches"
    assert len(result['embeddings']) > 0, "Should have some successful embeddings"
    assert len(result['embeddings']) < len(texts), "Should have partial results"
    
    print(f"\n✅ Fix verified: Got {len(result['embeddings'])} embeddings out of {len(texts)} texts")
    print(f"   Failed batches will be retried by background task")
    

if __name__ == "__main__":
    asyncio.run(test_rate_limit_fix())