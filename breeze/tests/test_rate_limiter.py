"""Test the improved rate limiter with context managers."""

import asyncio
import time
import pytest

from breeze.core.rate_limiter import TokenBucketV2, RateLimiterV2


@pytest.mark.asyncio
async def test_token_bucket_context_manager():
    """Test token bucket with context manager."""
    # 10 tokens capacity, 5 tokens/second refill
    bucket = TokenBucketV2(capacity=10, refill_rate=5.0)
    
    # Track when operations complete
    operation_times = []
    
    async def do_operation(op_id, tokens, duration):
        """Simulate an operation that takes time."""
        start = time.time()
        try:
            async with bucket.acquire(tokens, timeout=10):
                acquired_at = time.time() - start
                print(f"Operation {op_id}: Acquired {tokens} tokens after {acquired_at:.2f}s")
                
                # Simulate work
                await asyncio.sleep(duration)
                
                completed_at = time.time() - start
                operation_times.append((op_id, acquired_at, completed_at))
                print(f"Operation {op_id}: Completed after {completed_at:.2f}s")
        except asyncio.TimeoutError:
            print(f"Operation {op_id}: Timed out")
            operation_times.append((op_id, -1, -1))
    
    # Launch 3 operations that each need 5 tokens
    # With capacity 10, only 2 can run concurrently
    tasks = [
        do_operation(1, 5, 1.0),  # 5 tokens, 1 second
        do_operation(2, 5, 1.0),  # 5 tokens, 1 second  
        do_operation(3, 5, 1.0),  # 5 tokens, 1 second - must wait
    ]
    
    await asyncio.gather(*tasks)
    
    # Check results
    print("\nOperation timings:")
    for op_id, acquired, completed in sorted(operation_times):
        print(f"  Op {op_id}: acquired at {acquired:.2f}s, completed at {completed:.2f}s")
    
    # First two should start immediately
    assert operation_times[0][1] < 0.1  # Op 1 acquired quickly
    assert operation_times[1][1] < 0.1  # Op 2 acquired quickly
    
    # Third should wait for one of the first two to complete
    third_op = [t for t in operation_times if t[0] == 3][0]
    assert third_op[1] > 0.9  # Should wait ~1 second
    
    print("✅ Token bucket context manager test passed")


@pytest.mark.asyncio
async def test_rate_limiter_realistic():
    """Test rate limiter with realistic API simulation."""
    # 120 requests/min, 3M tokens/min (Voyage AI tier 1 limits)
    limiter = RateLimiterV2(requests_per_minute=120, tokens_per_minute=3_000_000)
    
    results = []
    
    async def make_api_call(call_id, tokens):
        """Simulate an API call."""
        start = time.time()
        try:
            async with limiter.acquire(tokens, timeout=5):
                acquired_at = time.time() - start
                
                # Simulate API call taking 0.5-1 second
                await asyncio.sleep(0.5 + (call_id % 3) * 0.25)
                
                completed_at = time.time() - start
                results.append((call_id, tokens, acquired_at, completed_at, True))
                return True
        except asyncio.TimeoutError:
            failed_at = time.time() - start
            results.append((call_id, tokens, failed_at, failed_at, False))
            return False
    
    # Make 10 concurrent calls with varying token counts
    tasks = []
    for i in range(10):
        tokens = 50000 + (i * 10000)  # 50k to 140k tokens
        tasks.append(make_api_call(i, tokens))
    
    outcomes = await asyncio.gather(*tasks)
    
    # Print results
    print("\nAPI call results:")
    for call_id, tokens, acquired, completed, success in sorted(results):
        status = "✓" if success else "✗"
        print(f"  Call {call_id} ({tokens:,} tokens): "
              f"acquired at {acquired:.2f}s, completed at {completed:.2f}s {status}")
    
    # All should succeed with these limits
    assert all(outcomes)
    
    print("✅ Rate limiter realistic test passed")


@pytest.mark.asyncio 
async def test_rate_limit_burst_protection():
    """Test that rate limiter prevents bursts."""
    # Very low limits to test burst protection
    # 6 requests/min (0.1/sec), 600 tokens/min (10/sec)
    limiter = RateLimiterV2(requests_per_minute=6, tokens_per_minute=600)
    
    results = []
    
    async def burst_request(req_id):
        start = time.time()
        try:
            async with limiter.acquire(100, timeout=2):
                acquired = time.time() - start
                results.append((req_id, acquired))
                await asyncio.sleep(0.1)  # Quick operation
                return True
        except asyncio.TimeoutError:
            return False
    
    # Try to make 5 requests at once (burst)
    # With 6 requests/min, we can only handle ~1 concurrent request
    tasks = [burst_request(i) for i in range(5)]
    outcomes = await asyncio.gather(*tasks)
    
    print("\nBurst request timings:")
    for req_id, acquired in sorted(results):
        print(f"  Request {req_id}: acquired at {acquired:.2f}s")
    
    # Some should fail due to timeout
    successful = sum(outcomes)
    print(f"Successful: {successful}/5")
    
    # With such low limits and 2s timeout, expect 2-3 to succeed
    assert 1 <= successful <= 3
    
    print("✅ Burst protection test passed")


if __name__ == "__main__":
    asyncio.run(test_token_bucket_context_manager())
    asyncio.run(test_rate_limiter_realistic()) 
    asyncio.run(test_rate_limit_burst_protection())