"""Test to verify rate limiting fix works correctly."""

import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock

from breeze.core.embeddings import get_voyage_embeddings_with_limits


@pytest.mark.asyncio
async def test_partial_failure_handling():
    """Test that partial failures are handled correctly."""
    
    # Create mock tokenizer
    tokenizer = MagicMock()
    def encode_text(text):
        result = MagicMock()
        # Make each text have 60k tokens, so only 2 fit per batch (120k limit)
        result.ids = [0] * 60000  # Don't create huge lists
        return result
    tokenizer.encode.side_effect = encode_text
    
    # Create texts - with 60k tokens each, only 2 will fit per batch
    texts = ["text" + str(i) for i in range(10)]  # Will create 5 batches
    
    # Track API calls
    call_count = 0
    
    def mock_embeddings(batch):
        nonlocal call_count
        call_count += 1
        
        # Fail calls 2, 3, 4 to ensure it exhausts retries for at least one batch
        if call_count in [2, 3, 4, 5]:
            raise Exception("429 Rate Limit Exceeded")
        
        return np.random.rand(len(batch), 768)
    
    model = MagicMock()
    model.compute_source_embeddings = mock_embeddings
    
    # Call with low retry count to fail faster
    result = await get_voyage_embeddings_with_limits(
        texts,
        model,
        tokenizer,
        max_concurrent_requests=3,
        max_retries=1,  # Low retry count
        retry_base_delay=0.01,  # Fast retry
        tokens_per_minute=10000,
        requests_per_minute=1000,
    )
    
    # Verify we get partial results
    assert isinstance(result, dict)
    assert 'embeddings' in result
    assert 'failed_batches' in result
    assert 'successful_batches' in result
    
    # Should have some successful embeddings
    assert len(result['embeddings']) > 0
    
    print(f"Call count: {call_count}")
    print(f"Successful batches: {result['successful_batches']}")
    print(f"Failed batches: {result['failed_batches']}")
    print(f"Total embeddings returned: {len(result['embeddings'])}")
    print(f"Safe batches count: {len(result['safe_batches'])}")
    
    # Should have at least one failed batch
    assert len(result['failed_batches']) > 0
    
    # The key fix: we should get partial results, not an exception!
    assert len(result['embeddings']) < len(texts), "Should have partial results"


@pytest.mark.asyncio
async def test_all_successful():
    """Test that all successful case still works."""
    
    # Create mock tokenizer
    tokenizer = MagicMock()
    def encode_text(text):
        result = MagicMock()
        result.ids = list(range(10))
        return result
    tokenizer.encode.side_effect = encode_text
    
    # Create texts
    texts = ["text" + str(i) for i in range(10)]
    
    # Mock model that always succeeds
    model = MagicMock()
    model.compute_source_embeddings = lambda batch: np.random.rand(len(batch), 768)
    
    result = await get_voyage_embeddings_with_limits(
        texts,
        model,
        tokenizer,
        max_concurrent_requests=3,
        max_retries=3,
        retry_base_delay=0.01,
        tokens_per_minute=10000,
        requests_per_minute=1000,
    )
    
    # Should still get result dict
    assert isinstance(result, dict)
    assert len(result['embeddings']) == len(texts)
    assert len(result['failed_batches']) == 0
    assert len(result['successful_batches']) > 0


if __name__ == "__main__":
    asyncio.run(test_partial_failure_handling())
    asyncio.run(test_all_successful())