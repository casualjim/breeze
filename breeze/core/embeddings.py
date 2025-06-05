"""Embedding providers for Breeze, including custom Voyage AI support."""

import os
from typing import List, ClassVar
import numpy as np
import pyarrow as pa

from lancedb.embeddings import EmbeddingFunction
from lancedb.embeddings.registry import register


def sanitize_text_input(inputs) -> List[str]:
    """Sanitize the input to the embedding function."""
    if isinstance(inputs, str):
        inputs = [inputs]
    elif isinstance(inputs, list):
        # Allow plain Python lists
        pass
    elif isinstance(inputs, pa.Array):
        inputs = inputs.to_pylist()
    elif isinstance(inputs, pa.ChunkedArray):
        inputs = inputs.combine_chunks().to_pylist()
    else:
        raise ValueError(f"Input type {type(inputs)} not allowed with text model.")

    if not all(isinstance(x, str) for x in inputs):
        raise ValueError("Each input should be str.")

    return inputs


@register("voyage-code-3")
class VoyageCode3EmbeddingFunction(EmbeddingFunction):
    """
    An embedding function that uses the VoyageAI API for voyage-code-3

    This is adapted from LanceDB's built-in VoyageAI implementation to support voyage-code-3
    """

    name: str
    client: ClassVar = None

    def ndims(self):
        """Return the dimension of the embeddings."""
        if self.name == "voyage-3-lite":
            return 512
        elif self.name in ["voyage-code-2", "voyage-code-3"]:
            return 1024  # Both code models have 1024 dimensions
        elif self.name in [
            "voyage-3",
            "voyage-finance-2",
            "voyage-multilingual-2",
            "voyage-law-2",
        ]:
            return 1024
        else:
            return 1024  # Default

    def compute_query_embeddings(self, query: str, *args, **kwargs) -> List[np.ndarray]:
        """Compute the embeddings for a given user query"""
        client = VoyageCode3EmbeddingFunction._get_client()
        result = client.embed(
            texts=[query], model=self.name, input_type="query", **kwargs
        )
        return [result.embeddings[0]]

    def compute_source_embeddings(self, inputs, *args, **kwargs) -> List[np.array]:
        """Compute the embeddings for the inputs"""
        client = VoyageCode3EmbeddingFunction._get_client()
        inputs = sanitize_text_input(inputs)
        result = client.embed(
            texts=inputs, model=self.name, input_type="document", **kwargs
        )
        return result.embeddings

    @staticmethod
    def _get_client():
        if VoyageCode3EmbeddingFunction.client is None:
            try:
                import voyageai

                # Disable verbose logging
                voyageai.log = "error"
                # Also try to suppress any internal logging
                import logging

                logging.getLogger("voyageai").setLevel(logging.ERROR)
            except ImportError:
                raise ImportError("Please install voyageai: pip install voyageai")

            api_key = os.environ.get("VOYAGE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "VOYAGE_API_KEY environment variable not set. "
                    "Please set it to your Voyage AI API key."
                )
            VoyageCode3EmbeddingFunction.client = voyageai.Client(api_key)
        return VoyageCode3EmbeddingFunction.client


# Voyage token limit handling functions
async def get_voyage_embeddings_with_limits(
    texts,
    model,
    tokenizer=None,
    max_concurrent_requests=5,
    max_retries=3,
    retry_base_delay=1.0,
    tokens_per_minute=3_000_000,
    requests_per_minute=2000,
):
    """Get embeddings from Voyage AI with proper token limit handling.

    Args:
        texts: List of text strings to embed
        model: The embedding model instance
        tokenizer: HuggingFace tokenizer instance (from tokenizers library)
        max_concurrent_requests: Maximum concurrent API requests
        max_retries: Maximum number of retries for rate-limited requests
        retry_base_delay: Base delay in seconds for exponential backoff
        tokens_per_minute: Token limit per minute based on tier
        requests_per_minute: Request limit per minute based on tier

    Raises:
        Exception: If embeddings cannot be generated after all retries
    """
    import asyncio
    import time

    MAX_TOKENS_PER_BATCH = 120000  # Voyage's limit per request
    MAX_TEXTS_PER_BATCH = 128  # Voyage's limit per request
    
    # Rate limiting trackers
    request_times = []  # Track request timestamps
    token_counts = []   # Track token counts with timestamps

    # Shared rate limit state - acts as a latch
    rate_limit_lock = asyncio.Lock()
    rate_limited = False
    rate_limit_until = 0  # Timestamp when rate limit expires
    failed_batches = set()  # Track which batches failed

    def estimate_tokens(text):
        """Estimate tokens using tokenizer or character count."""
        if tokenizer:
            # For HuggingFace tokenizers
            encoded = tokenizer.encode(text)
            return len(encoded.ids)
        else:
            # Conservative estimate: ~3.5 chars per token
            return int(len(text) / 3.5)

    def create_safe_batches(texts):
        """Create batches that respect Voyage's token and count limits."""
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = estimate_tokens(text)

            # If single text exceeds limit, truncate it
            if text_tokens > MAX_TOKENS_PER_BATCH:
                max_chars = int(MAX_TOKENS_PER_BATCH * 3.5 * 0.8)
                text = text[:max_chars]
                text_tokens = estimate_tokens(text)

            # Check if adding this text would exceed limits
            if current_batch and (
                current_tokens + text_tokens
                > MAX_TOKENS_PER_BATCH * 0.8  # 80% safety margin
                or len(current_batch) >= MAX_TEXTS_PER_BATCH
            ):
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    # Create safe batches
    safe_batches = create_safe_batches(texts)

    # Log batching info
    import logging

    logger = logging.getLogger(__name__)
    if len(safe_batches) > 1:
        logger.info(
            f"Voyage AI: Processing {len(texts)} texts in {len(safe_batches)} API calls (respecting token limits)"
        )

    # Semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def check_rate_limits(batch_tokens):
        """Check if we're within rate limits."""
        async with rate_limit_lock:
            current_time = time.time()
            one_minute_ago = current_time - 60
            
            # Clean up old entries
            nonlocal request_times, token_counts
            request_times = [t for t in request_times if t > one_minute_ago]
            token_counts = [(t, c) for t, c in token_counts if t > one_minute_ago]
            
            # Check request rate
            if len(request_times) >= requests_per_minute:
                # Calculate when the oldest request will expire
                wait_time = 60 - (current_time - request_times[0])
                return False, wait_time
            
            # Check token rate
            total_tokens = sum(c for _, c in token_counts) + batch_tokens
            if total_tokens > tokens_per_minute:
                # Calculate when enough tokens will expire
                tokens_to_free = total_tokens - tokens_per_minute
                accumulated = 0
                for t, c in token_counts:
                    accumulated += c
                    if accumulated >= tokens_to_free:
                        wait_time = 60 - (current_time - t)
                        return False, wait_time
                # Fallback: wait for oldest tokens to expire
                if token_counts:
                    wait_time = 60 - (current_time - token_counts[0][0])
                    return False, wait_time
            
            return True, 0

    async def record_request(batch_tokens):
        """Record a request for rate limiting."""
        async with rate_limit_lock:
            current_time = time.time()
            request_times.append(current_time)
            token_counts.append((current_time, batch_tokens))

    async def process_batch(batch_idx, batch):
        """Process a single batch with the API."""
        nonlocal rate_limited, rate_limit_until

        import logging
        import time

        logger = logging.getLogger(__name__)
        
        # Calculate tokens in this batch
        batch_tokens = sum(estimate_tokens(text) for text in batch)

        async with semaphore:
            for attempt in range(max_retries):
                # Check tier-based rate limits
                can_proceed, wait_time = await check_rate_limits(batch_tokens)
                if not can_proceed:
                    logger.info(
                        f"Batch {batch_idx + 1}: Approaching rate limit, waiting {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                
                # Also check if we're rate limited from a 429 response
                while True:
                    wait_time = 0
                    async with rate_limit_lock:
                        current_time = time.time()
                        if rate_limited and rate_limit_until > current_time:
                            wait_time = rate_limit_until - current_time
                        else:
                            # Not rate limited or rate limit expired
                            if rate_limited and rate_limit_until <= current_time:
                                rate_limited = False
                                rate_limit_until = 0
                            break

                    # Wait outside the lock
                    if wait_time > 0:
                        logger.info(
                            f"Batch {batch_idx + 1}: Waiting {wait_time:.1f}s for rate limit to expire"
                        )
                        await asyncio.sleep(wait_time)

                try:
                    # Record the request before making the API call
                    await record_request(batch_tokens)
                    
                    # Make the API call
                    embeddings = await asyncio.to_thread(
                        model.compute_source_embeddings, batch
                    )
                    return embeddings
                except Exception as e:
                    # Check if it's a rate limit error (429)
                    error_str = str(e).lower()
                    if (
                        "429" in error_str
                        or "rate limit" in error_str
                        or "too many requests" in error_str
                    ):
                        async with rate_limit_lock:
                            if attempt < max_retries - 1:
                                # Calculate backoff delay
                                delay = retry_base_delay * (2**attempt) + (
                                    0.1 * attempt
                                )
                                rate_limited = True
                                rate_limit_until = time.time() + delay

                                logger.warning(
                                    f"Rate limited on batch {batch_idx + 1}, setting global rate limit for {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                                )

                        # Wait for the delay (outside the lock)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # For non-rate-limit errors, fail immediately
                        logger.error(f"Error in Voyage batch {batch_idx + 1}: {e}")
                        failed_batches.add(batch_idx)
                        raise

            # Exhausted all retries
            failed_batches.add(batch_idx)
            error_msg = (
                f"Failed to process batch {batch_idx + 1} after {max_retries} attempts"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

    # Process all batches concurrently
    tasks = [process_batch(idx, batch) for idx, batch in enumerate(safe_batches)]

    # Use gather with return_exceptions=True to handle failures gracefully
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for failures
    all_embeddings = []
    for idx, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.error(f"Batch {idx} failed: {result}")
            failed_batches.add(idx)
        else:
            all_embeddings.extend(result)

    # If any batches failed, raise an exception
    if failed_batches:
        failed_count = len(failed_batches)
        total_count = len(safe_batches)
        error_msg = f"Failed to generate embeddings for {failed_count}/{total_count} batches: {sorted(failed_batches)}"
        logger.error(error_msg)
        raise Exception(error_msg)

    return np.array(all_embeddings)
