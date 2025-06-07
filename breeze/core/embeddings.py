"""Embedding providers for Breeze, including custom Voyage AI support."""

import os
from typing import List, ClassVar, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pyarrow as pa
import logging

from lancedb.embeddings import EmbeddingFunction
from lancedb.embeddings.registry import register
from transformers import AutoTokenizer
from breeze.core.rate_limiter import RateLimiterV2
from breeze.core.text_chunker import TextChunker, ChunkingConfig, FileContent, ChunkedFile

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embeddings: np.ndarray  # Array of embeddings
    successful_files: List[int]  # Indices of successfully embedded files
    failed_files: List[int]  # Indices of failed files
    chunked_files: List[ChunkedFile]  # Chunked file information
    
    
@dataclass
class VoyageEmbeddingResult(EmbeddingResult):
    """Result from Voyage embedding generation with batch information."""
    safe_batches: Optional[List[tuple]] = None  # Batches that were created
    failed_batches: Optional[List[int]] = None  # Indices of batches that failed


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


# Removed ModelAwareChunker - now using TextChunker from text_chunker.py


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


def _get_rate_limiter(
    tokens_per_minute: int, requests_per_minute: int
) -> RateLimiterV2:
    """Get or create the global rate limiter."""
    global _voyage_rate_limiter
    # Always create a new rate limiter with the provided values
    # This ensures tests can use their own rate limits
    _voyage_rate_limiter = RateLimiterV2(requests_per_minute, tokens_per_minute)
    return _voyage_rate_limiter


# Voyage token limit handling functions
async def get_voyage_embeddings_with_limits(
    file_contents: List[FileContent],
    model: EmbeddingFunction,
    tokenizer: AutoTokenizer | None = None,
    max_concurrent_requests=5,
    tokens_per_minute=3_000_000,
    requests_per_minute=2000,
    max_retry_duration=600,  # 10 minutes max retry time for rate limits
):
    """Get embeddings from Voyage AI with proper token limit handling using TextChunker.

    Args:
        file_contents: List of FileContent objects with content, path, and language
        model: The embedding model instance
        tokenizer: HuggingFace tokenizer instance (from tokenizers library)
        max_concurrent_requests: Maximum concurrent API requests
        tokens_per_minute: Token limit per minute based on tier
        requests_per_minute: Request limit per minute based on tier
        max_retry_duration: Maximum time to retry rate-limited requests

    Returns:
        Dictionary with embeddings and metadata
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

    logger.debug(
        f"Rate limiter created with {safe_tokens_per_minute} tokens/min, {safe_requests_per_minute} requests/min"
    )

    # Track failed batches
    failed_batches = set()  # Track which batches failed

    # Create chunker with 16k tokens (as per context.md)
    # This is the sweet spot for code files
    chunker = TextChunker(
        tokenizer=tokenizer, config=ChunkingConfig(max_tokens=16384, overlap_tokens=256)
    )

    # Chunk all file contents
    chunked_files = chunker.chunk_files(file_contents)

    # Log chunking info
    total_chunks = sum(len(cf.chunks) for cf in chunked_files)
    if total_chunks > len(file_contents):
        logger.info(
            f"Voyage AI: Chunked {len(file_contents)} files into {total_chunks} chunks"
        )

    # Create batches that respect Voyage's limits
    # We can fit ~7 chunks of 16k tokens in a 120k batch
    all_chunks = []
    chunk_to_file_idx = {}

    for file_idx, chunked_file in enumerate(chunked_files):
        for chunk in chunked_file.chunks:
            chunk_idx = len(all_chunks)
            all_chunks.append(chunk)
            chunk_to_file_idx[chunk_idx] = file_idx

    # Create safe batches
    safe_batches = []
    current_batch = []
    current_batch_indices = []
    current_tokens = 0

    for idx, chunk in enumerate(all_chunks):
        chunk_tokens = chunk.estimated_tokens

        # Check if adding this chunk would exceed limits
        if current_batch and (
            current_tokens + chunk_tokens
            > MAX_TOKENS_PER_BATCH * 0.8  # 80% safety margin
            or len(current_batch) >= MAX_TEXTS_PER_BATCH
        ):
            safe_batches.append((current_batch, current_batch_indices))
            current_batch = [chunk.text]
            current_batch_indices = [idx]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk.text)
            current_batch_indices.append(idx)
            current_tokens += chunk_tokens

    if current_batch:
        safe_batches.append((current_batch, current_batch_indices))

    # Log batching info
    if len(safe_batches) > 1:
        logger.info(
            f"Voyage AI: Processing {total_chunks} chunks in {len(safe_batches)} API calls"
        )

    # Semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def process_batch(batch_idx, batch_data):
        """Process a single batch with the API."""
        batch_texts, chunk_indices = batch_data
        logger = logging.getLogger(__name__)

        # Calculate tokens in this batch
        batch_tokens = sum(chunker.estimate_tokens(text) for text in batch_texts)

        # Track when we started trying this batch
        batch_start_time = time.time()

        async with semaphore:
            attempt = 0
            while True:
                # Check if we've been trying for too long (10 minutes)
                if time.time() - batch_start_time > max_retry_duration:
                    failed_batches.add(batch_idx)
                    error_msg = f"Failed to process batch {batch_idx + 1} after {max_retry_duration}s of retries"
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
                            model.compute_source_embeddings, batch_texts
                        )
                        return embeddings, chunk_indices

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
    chunk_embeddings = {}  # Maps chunk index to embedding
    successful_batches = []

    for idx, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.error(f"Batch {idx} failed: {result}")
            failed_batches.add(idx)
        else:
            embeddings, chunk_indices = result
            for i, chunk_idx in enumerate(chunk_indices):
                chunk_embeddings[chunk_idx] = embeddings[i]
            successful_batches.append(idx)

    # Combine chunk embeddings for each file
    final_embeddings = []
    successful_files = []
    failed_files = []

    for file_idx, chunked_file in enumerate(chunked_files):
        # Collect embeddings for all chunks of this file
        file_chunk_embeddings = []
        all_chunks_found = True

        # Find the global indices for this file's chunks
        global_chunk_start = sum(len(cf.chunks) for cf in chunked_files[:file_idx])

        for local_idx, chunk in enumerate(chunked_file.chunks):
            global_idx = global_chunk_start + local_idx
            if global_idx in chunk_embeddings:
                file_chunk_embeddings.append(chunk_embeddings[global_idx])
            else:
                all_chunks_found = False
                break

        if all_chunks_found and file_chunk_embeddings:
            # Combine chunk embeddings into a single embedding
            if len(file_chunk_embeddings) == 1:
                # Single chunk - use as is
                combined_embedding = file_chunk_embeddings[0]
            else:
                # Multiple chunks - combine them
                combined_embedding = chunker.combine_file_embeddings(
                    chunked_file, file_chunk_embeddings, method="weighted_average"
                )

            final_embeddings.append(combined_embedding)
            successful_files.append(file_idx)
        else:
            failed_files.append(file_idx)
            logger.warning(f"Missing embeddings for file {file_idx}")

    # Always return results with metadata - let caller decide what to do with failures
    if failed_files:
        total_count = len(file_contents)
        logger.warning(
            f"Partially completed: {len(successful_files)}/{total_count} files succeeded. "
            f"Failed file indices: {failed_files}"
        )

    return VoyageEmbeddingResult(
        embeddings=np.array(final_embeddings) if final_embeddings else np.array([]),
        successful_files=successful_files,
        failed_files=failed_files,
        chunked_files=chunked_files,
        safe_batches=safe_batches,
        failed_batches=list(failed_batches) if failed_batches else []
    )


async def get_local_embeddings_with_tokenizer_chunking(
    file_contents: List[FileContent],
    model: EmbeddingFunction,
    model_name: str,
    max_concurrent_requests: int = 5,
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    max_sequence_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Get embeddings from local models with proper tokenizer-based chunking.

    This function chunks long texts into overlapping segments, embeds each chunk,
    and combines them into a single embedding per text.

    Args:
        file_contents: List of FileContent objects with content, path, and language
        model: The LanceDB embedding model instance
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
    from breeze.core.text_chunker import create_batches_from_chunked_files

    logger = logging.getLogger(__name__)

    # Try to load the tokenizer for the model
    tokenizer = None
    actual_max_length = 8192  # Default to 8k tokens like before

    # Skip tokenizer loading for mock embedders
    # Check various ways a model could be a mock:
    # 1. Model name contains "mock"
    # 2. Type name starts with "Mock" or contains "Mock"
    # 3. Model is from the mock_embedders module
    is_mock = (
        "mock" in model_name.lower()
        or type(model).__name__.startswith("Mock")
        or "Mock" in type(model).__name__
        or model.__class__.__module__ == "breeze.tests.mock_embedders"
    )

    if is_mock:
        logger.debug(
            f"Skipping tokenizer loading for mock embedder: {type(model).__name__}"
        )
    else:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            # Get the actual max sequence length from the tokenizer/model
            if max_sequence_length:
                actual_max_length = max_sequence_length
            elif (
                hasattr(tokenizer, "model_max_length")
                and tokenizer.model_max_length < 1000000
            ):
                actual_max_length = tokenizer.model_max_length
            elif hasattr(tokenizer, "max_len"):
                actual_max_length = tokenizer.max_len
            else:
                # Try to get from model config if available
                if hasattr(model, "model") and hasattr(model.model, "config"):
                    if hasattr(model.model.config, "max_position_embeddings"):
                        actual_max_length = model.model.config.max_position_embeddings

            logger.info(
                f"Using tokenizer for {model_name} with max sequence length: {actual_max_length}"
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer for {model_name}: {e}")
            logger.warning("Falling back to character-based estimation")

    # Create chunker with the actual max length
    chunker = TextChunker(
        tokenizer=tokenizer, config=ChunkingConfig(max_tokens=actual_max_length)
    )

    # Chunk all file contents
    chunked_files = chunker.chunk_files(file_contents)

    # Log chunking info
    total_chunks = sum(len(cf.chunks) for cf in chunked_files)
    if total_chunks > len(file_contents):
        logger.info(
            f"Local model: Chunked {len(file_contents)} files into {total_chunks} chunks"
        )

    # Create batches from chunked files
    batches = create_batches_from_chunked_files(chunked_files, batch_size=32)

    # Track results
    chunk_embeddings = {}  # Maps (file_idx, chunk_index) to embedding
    failed_batches = set()

    # Semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def process_batch(batch_idx, batch):
        """Process a single batch of chunks with the model."""
        async with semaphore:
            attempt = 0
            while attempt < max_retries:
                try:
                    # Extract chunk texts
                    chunk_texts = [chunk.text for _, _, chunk in batch]

                    # Generate embeddings using asyncio.to_thread for CPU-bound operation
                    embeddings = await asyncio.to_thread(
                        model.compute_source_embeddings, chunk_texts
                    )

                    # Store results mapped by file and chunk index
                    results = []
                    for i, (file_idx, file_content, chunk) in enumerate(batch):
                        chunk_embeddings[(file_idx, chunk.chunk_index)] = embeddings[i]
                        results.append((file_idx, chunk.chunk_index, embeddings[i]))

                    return results

                except Exception as e:
                    attempt += 1
                    if attempt < max_retries:
                        delay = retry_base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"Error in batch {batch_idx + 1}: {e}. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Failed batch {batch_idx + 1} after {max_retries} attempts: {e}"
                        )
                        failed_batches.add(batch_idx)
                        raise

    # Process all batches concurrently
    tasks = [process_batch(idx, batch) for idx, batch in enumerate(batches)]

    # Use gather with return_exceptions=True to handle failures gracefully
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for failures
    successful_count = 0
    for idx, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.error(f"Batch {idx} failed: {result}")
            failed_batches.add(idx)
        else:
            successful_count += 1

    # Combine chunk embeddings for each file
    final_embeddings = []
    successful_files = []
    failed_files = []

    for file_idx, chunked_file in enumerate(chunked_files):
        # Collect embeddings for all chunks of this file
        file_chunk_embeddings = []
        all_chunks_found = True

        for chunk in chunked_file.chunks:
            key = (file_idx, chunk.chunk_index)
            if key in chunk_embeddings:
                file_chunk_embeddings.append(chunk_embeddings[key])
            else:
                all_chunks_found = False
                break

        if all_chunks_found and file_chunk_embeddings:
            # Combine chunk embeddings into a single embedding
            if len(file_chunk_embeddings) == 1:
                # Single chunk - use as is
                combined_embedding = file_chunk_embeddings[0]
            else:
                # Multiple chunks - combine them
                combined_embedding = chunker.combine_file_embeddings(
                    chunked_file, file_chunk_embeddings, method="weighted_average"
                )

            final_embeddings.append(combined_embedding)
            successful_files.append(file_idx)
        else:
            failed_files.append(file_idx)
            logger.warning(f"Missing embeddings for file {file_idx}")

    # Log results
    if failed_files:
        logger.warning(
            f"Partially completed: {len(successful_files)}/{len(file_contents)} files succeeded. "
            f"Failed file indices: {failed_files}"
        )

    return EmbeddingResult(
        embeddings=np.array(final_embeddings) if final_embeddings else np.array([]),
        successful_files=successful_files,
        failed_files=failed_files,
        chunked_files=chunked_files
    )
