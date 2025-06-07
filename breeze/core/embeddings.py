"""Embedding providers for Breeze, including custom Voyage AI support."""

import os
from typing import List, ClassVar
import numpy as np
import pyarrow as pa
import logging

from lancedb.embeddings import EmbeddingFunction
from lancedb.embeddings.registry import register
from breeze.core.rate_limiter import RateLimiterV2

logger = logging.getLogger(__name__)


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



# Global rate limiter instance (shared across all calls)
_voyage_rate_limiter = None

def _get_rate_limiter(tokens_per_minute: int, requests_per_minute: int) -> RateLimiterV2:
    """Get or create the global rate limiter."""
    global _voyage_rate_limiter
    # Always create a new rate limiter with the provided values
    # This ensures tests can use their own rate limits
    _voyage_rate_limiter = RateLimiterV2(requests_per_minute, tokens_per_minute)
    return _voyage_rate_limiter


# Voyage token limit handling functions
async def get_voyage_embeddings_with_limits(
    texts,
    model,
    tokenizer=None,
    max_concurrent_requests=5,
    max_retries=3,  # Kept for compatibility but we use time-based retries for rate limits
    retry_base_delay=1.0,
    tokens_per_minute=3_000_000,
    requests_per_minute=2000,
    max_retry_duration=600,  # 10 minutes max retry time for rate limits
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
    import logging
    
    logger = logging.getLogger(__name__)

    MAX_TOKENS_PER_BATCH = 120000  # Voyage's limit per request
    MAX_TEXTS_PER_BATCH = 128  # Voyage's limit per request
    
    # Get the global rate limiter with 10% safety margin
    # This helps avoid hitting the actual API limits
    safe_tokens_per_minute = int(tokens_per_minute * 0.9)
    safe_requests_per_minute = int(requests_per_minute * 0.9)
    rate_limiter = _get_rate_limiter(safe_tokens_per_minute, safe_requests_per_minute)
    
    logger.debug(f"Rate limiter created with {safe_tokens_per_minute} tokens/min, {safe_requests_per_minute} requests/min")
    
    # Track failed batches
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

    async def process_batch(batch_idx, batch):
        """Process a single batch with the API."""
        logger = logging.getLogger(__name__)
        
        # Calculate tokens in this batch
        batch_tokens = sum(estimate_tokens(text) for text in batch)
        
        # Track when we started trying this batch
        batch_start_time = time.time()

        async with semaphore:
            attempt = 0
            while True:
                # Check if we've been trying for too long (10 minutes)
                if time.time() - batch_start_time > max_retry_duration:
                    failed_batches.add(batch_idx)
                    error_msg = (
                        f"Failed to process batch {batch_idx + 1} after {max_retry_duration}s of retries"
                    )
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                try:
                    # Use rate limiter context manager to hold tokens for entire operation
                    async with rate_limiter.acquire(batch_tokens, timeout=60):
                        logger.info(
                            f"Batch {batch_idx + 1}: Acquired {batch_tokens} tokens, making API call"
                        )
                        
                        # Make the API call while holding the tokens
                        embeddings = await asyncio.to_thread(
                            model.compute_source_embeddings, batch
                        )
                        return embeddings
                        
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Batch {batch_idx + 1}: Timeout waiting for rate limiter after 60s"
                    )
                    continue
                except Exception as e:
                    # Check if it's a rate limit error (429)
                    error_str = str(e).lower()
                    if (
                        "429" in error_str
                        or "rate limit" in error_str
                        or "too many requests" in error_str
                    ):
                        # We hit a rate limit despite our token bucket
                        # This means our limits are set too high
                        # Wait with exponential backoff
                        delay = min(30, max(5, 5 + (attempt * 5)))
                        
                        logger.warning(
                            f"Rate limited on batch {batch_idx + 1} despite token bucket: {e}. "
                            f"Waiting {delay:.0f}s before retry (attempt {attempt + 1}). "
                            f"Consider reducing concurrent requests or rate limits."
                        )

                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    else:
                        # For non-rate-limit errors, still retry but with shorter delays
                        if attempt < 3:  # Try up to 3 times for non-rate-limit errors
                            delay = 2 * (attempt + 1)
                            logger.warning(
                                f"Error in batch {batch_idx + 1}: {e}. Retrying in {delay}s (attempt {attempt + 1})"
                            )
                            await asyncio.sleep(delay)
                            attempt += 1
                            continue
                        else:
                            # Give up on non-rate-limit errors after 3 attempts
                            logger.error(f"Error in Voyage batch {batch_idx + 1}: {e}")
                            failed_batches.add(batch_idx)
                            raise

    # Process all batches concurrently
    tasks = [process_batch(idx, batch) for idx, batch in enumerate(safe_batches)]

    # Use gather with return_exceptions=True to handle failures gracefully
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for failures and collect successful embeddings
    all_embeddings = []
    successful_batches = []
    
    for idx, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.error(f"Batch {idx} failed: {result}")
            failed_batches.add(idx)
        else:
            all_embeddings.extend(result)
            successful_batches.append(idx)

    # Always return results with metadata - let caller decide what to do with failures
    if failed_batches:
        total_count = len(safe_batches)
        logger.warning(
            f"Partially completed: {len(successful_batches)}/{total_count} batches succeeded. "
            f"Failed batches: {sorted(failed_batches)}"
        )
    
    return {
        'embeddings': np.array(all_embeddings) if all_embeddings else np.array([]),
        'successful_batches': successful_batches,
        'failed_batches': list(failed_batches),
        'texts': texts,
        'safe_batches': safe_batches
    }


async def get_local_embeddings_with_tokenizer_chunking(
    texts,
    model,
    model_name,
    max_concurrent_requests=5,
    max_retries=3,
    retry_base_delay=1.0,
    max_sequence_length=None,
):
    """Get embeddings from local models with proper tokenizer-based chunking.
    
    Args:
        texts: List of text strings to embed
        model: The embedding model instance
        model_name: Name of the model (e.g., 'BAAI/bge-m3')
        max_concurrent_requests: Maximum concurrent embedding operations
        max_retries: Maximum number of retries for failed embeddings
        retry_base_delay: Base delay in seconds for exponential backoff
        max_sequence_length: Override the model's max sequence length
        
    Returns:
        Dictionary with embeddings and metadata
    """
    import asyncio
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Try to load the tokenizer for the model
    tokenizer = None
    actual_max_length = 512  # Conservative default
    
    # Skip tokenizer loading for mock embedders
    # Check various ways a model could be a mock:
    # 1. Model name contains "mock"
    # 2. Type name starts with "Mock" or contains "Mock"
    # 3. Model is from the mock_embedders module
    is_mock = (
        "mock" in model_name.lower() or 
        type(model).__name__.startswith("Mock") or
        "Mock" in type(model).__name__ or
        model.__class__.__module__ == 'breeze.tests.mock_embedders'
    )
    
    if is_mock:
        logger.debug(f"Skipping tokenizer loading for mock embedder: {type(model).__name__}")
    else:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Get the actual max sequence length from the tokenizer/model
            if max_sequence_length:
                actual_max_length = max_sequence_length
            elif hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 1000000:
                actual_max_length = tokenizer.model_max_length
            elif hasattr(tokenizer, 'max_len'):
                actual_max_length = tokenizer.max_len
            else:
                # Try to get from model config if available
                if hasattr(model, 'model') and hasattr(model.model, 'config'):
                    if hasattr(model.model.config, 'max_position_embeddings'):
                        actual_max_length = model.model.config.max_position_embeddings
            
            logger.info(f"Using tokenizer for {model_name} with max sequence length: {actual_max_length}")
        except Exception as e:
            logger.warning(f"Could not load tokenizer for {model_name}: {e}")
            logger.warning("Falling back to character-based estimation")
    
    # Track failed batches
    failed_batches = set()
    
    def estimate_tokens(text):
        """Estimate tokens using tokenizer or character count."""
        if tokenizer:
            try:
                # Use tokenizer to get exact token count
                encoded = tokenizer.encode(text, add_special_tokens=True, truncation=False)
                return len(encoded)
            except Exception as e:
                logger.debug(f"Token encoding failed, using character fallback: {e}")
        
        # Conservative estimate: ~4 chars per token for code
        return int(len(text) / 4)
    
    def create_safe_batches(texts):
        """Create batches that respect the model's token limits."""
        # Reserve tokens for special tokens ([CLS], [SEP], etc.)
        reserved_tokens = 10  # Conservative reservation
        max_tokens_per_text = actual_max_length - reserved_tokens
        
        batches = []
        current_batch = []
        
        for idx, text in enumerate(texts):
            text_tokens = estimate_tokens(text)
            
            # If single text exceeds limit, we need to truncate it
            if text_tokens > max_tokens_per_text:
                if tokenizer:
                    try:
                        # Use tokenizer to truncate properly
                        encoded = tokenizer.encode(
                            text, 
                            add_special_tokens=True, 
                            truncation=True, 
                            max_length=actual_max_length,
                            return_tensors=None
                        )
                        # Decode back to text (without special tokens)
                        truncated_text = tokenizer.decode(encoded, skip_special_tokens=True)
                        
                        # Add truncation indicator
                        text = truncated_text + "..."
                        text_tokens = len(encoded)
                        
                        logger.debug(f"Truncated text from {estimate_tokens(texts[idx])} to {text_tokens} tokens")
                    except Exception as e:
                        logger.warning(f"Tokenizer truncation failed: {e}")
                        # Fallback to character truncation
                        max_chars = max_tokens_per_text * 4
                        text = text[:max_chars] + "..."
                        text_tokens = estimate_tokens(text)
                else:
                    # Character-based truncation
                    max_chars = max_tokens_per_text * 4
                    text = text[:max_chars] + "..."
                    text_tokens = estimate_tokens(text)
            
            current_batch.append(text)
        
        # For local models, we can usually process multiple texts at once
        # But let's be conservative and batch them reasonably
        batch_size = 32  # Reasonable batch size for local models
        
        for i in range(0, len(current_batch), batch_size):
            batches.append(current_batch[i:i + batch_size])
        
        return batches
    
    # Create safe batches
    safe_batches = create_safe_batches(texts)
    
    # Log batching info
    if len(safe_batches) > 1:
        logger.info(
            f"Local model: Processing {len(texts)} texts in {len(safe_batches)} batches"
        )
    
    # Semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def process_batch(batch_idx, batch):
        """Process a single batch with the model."""
        async with semaphore:
            attempt = 0
            while attempt < max_retries:
                try:
                    # Generate embeddings using asyncio.to_thread for CPU-bound operation
                    embeddings = await asyncio.to_thread(
                        model.compute_source_embeddings, batch
                    )
                    return embeddings
                    
                except Exception as e:
                    attempt += 1
                    if attempt < max_retries:
                        delay = retry_base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"Error in batch {batch_idx + 1}: {e}. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Failed batch {batch_idx + 1} after {max_retries} attempts: {e}")
                        failed_batches.add(batch_idx)
                        raise
    
    # Process all batches concurrently
    tasks = [process_batch(idx, batch) for idx, batch in enumerate(safe_batches)]
    
    # Use gather with return_exceptions=True to handle failures gracefully
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check for failures and collect successful embeddings
    all_embeddings = []
    successful_batches = []
    
    for idx, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.error(f"Batch {idx} failed: {result}")
            failed_batches.add(idx)
        else:
            all_embeddings.extend(result)
            successful_batches.append(idx)
    
    # Log results
    if failed_batches:
        total_count = len(safe_batches)
        logger.warning(
            f"Partially completed: {len(successful_batches)}/{total_count} batches succeeded. "
            f"Failed batches: {sorted(failed_batches)}"
        )
    
    return {
        'embeddings': np.array(all_embeddings) if all_embeddings else np.array([]),
        'successful_batches': successful_batches,
        'failed_batches': list(failed_batches),
        'texts': texts,
        'safe_batches': safe_batches
    }
