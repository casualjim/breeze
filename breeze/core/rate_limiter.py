"""Improved token bucket rate limiter that properly handles async operations."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator
import logging

logger = logging.getLogger(__name__)


class TokenBucketV2:
    """Token bucket that properly handles in-flight requests.
    
    This implementation tracks active requests and ensures we don't
    exceed rate limits even with concurrent operations.
    """
    
    def __init__(self, capacity: int, refill_rate: float, name: str = ""):
        """Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens/active operations
            refill_rate: Number of tokens added per second
            name: Optional name for logging
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.name = name
        
        # Tokens represent available capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        
        # Track active operations
        self._active_operations = 0
        
        # Synchronization
        self._lock = asyncio.Lock()
        self._not_full = asyncio.Condition(self._lock)
    
    @asynccontextmanager
    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> AsyncIterator[None]:
        """Acquire tokens for the duration of an operation.
        
        This context manager ensures tokens are held for the entire
        duration of the operation and properly released afterward.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait for tokens
            
        Yields:
            None when tokens are acquired
            
        Raises:
            TimeoutError: If timeout expires before acquiring tokens
            ValueError: If requesting more tokens than capacity
        """
        if tokens > self.capacity:
            raise ValueError(f"Cannot acquire {tokens} tokens, bucket capacity is {self.capacity}")
        
        acquired = False
        start_time = time.monotonic()
        
        try:
            async with self._not_full:
                while True:
                    # Refill bucket
                    now = time.monotonic()
                    elapsed = now - self._last_refill
                    tokens_to_add = elapsed * self.refill_rate
                    
                    # Only add tokens up to capacity minus active operations
                    max_tokens = self.capacity - self._active_operations
                    self._tokens = min(max_tokens, self._tokens + tokens_to_add)
                    self._last_refill = now
                    
                    # Check if we can acquire
                    if self._tokens >= tokens:
                        self._tokens -= tokens
                        self._active_operations += tokens
                        acquired = True
                        break
                    
                    # Check timeout
                    if timeout is not None:
                        elapsed = time.monotonic() - start_time
                        if elapsed >= timeout:
                            raise TimeoutError(f"Failed to acquire {tokens} tokens within {timeout}s")
                    
                    # Calculate wait time
                    tokens_needed = tokens - self._tokens
                    wait_time = tokens_needed / self.refill_rate
                    
                    if timeout is not None:
                        remaining = timeout - (time.monotonic() - start_time)
                        wait_time = min(wait_time, remaining)
                    
                    # Wait for tokens or notification
                    try:
                        await asyncio.wait_for(self._not_full.wait(), timeout=wait_time)
                    except asyncio.TimeoutError:
                        # Expected - we'll loop and check again
                        pass
            
            # Tokens acquired, yield control
            yield
            
        finally:
            # Release tokens when done
            if acquired:
                async with self._not_full:
                    self._active_operations -= tokens
                    # Don't add tokens back - they were "used"
                    # Just notify waiters that capacity is available
                    self._not_full.notify_all()


class RateLimiterV2:
    """Combined rate limiter for requests and tokens using context managers."""
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        """Initialize rate limiter with both request and token limits."""
        # For requests, we track concurrent requests
        # Capacity is based on how many concurrent requests we can sustain
        # Be more conservative - assume requests take 3 seconds, limit to prevent contention
        max_concurrent_requests = max(1, min(10, requests_per_minute // 60))
        
        self.request_bucket = TokenBucketV2(
            capacity=max_concurrent_requests,
            refill_rate=requests_per_minute / 60.0,
            name="requests"
        )
        
        # For tokens, track concurrent token usage
        # Assume we can use all tokens concurrently
        self.token_bucket = TokenBucketV2(
            capacity=tokens_per_minute,
            refill_rate=tokens_per_minute / 60.0,
            name="tokens"
        )
        
        logger.info(
            f"RateLimiterV2 initialized: "
            f"max {max_concurrent_requests} concurrent requests, "
            f"{requests_per_minute} req/min, "
            f"{tokens_per_minute} tokens/min"
        )
    
    @asynccontextmanager
    async def acquire(self, tokens: int, timeout: Optional[float] = None) -> AsyncIterator[None]:
        """Acquire both a request slot and tokens for an operation.
        
        Args:
            tokens: Number of tokens needed for this request
            timeout: Maximum time to wait
            
        Yields:
            None when resources are acquired
            
        Raises:
            TimeoutError: If timeout expires
        """
        # Need to acquire both request slot and tokens
        # Use nested context managers to ensure proper cleanup
        async with self.request_bucket.acquire(1, timeout):
            async with self.token_bucket.acquire(tokens, timeout):
                yield