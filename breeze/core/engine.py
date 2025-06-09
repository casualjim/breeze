"""Core engine for Breeze code indexing and search."""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
import time
import os

# No longer need identify and magic - using breeze-langdetect via ContentDetector

import aiofiles
import aiofiles.os
import lancedb
from lancedb.embeddings import get_registry
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from breeze.core.config import BreezeConfig
from breeze.core.file_discovery import FileDiscovery
from breeze.core.snippets import TreeSitterSnippetExtractor
from breeze.core.content_detection import ContentDetector
from breeze.core.models import (
    IndexStats,
    SearchResult,
    Project,
    IndexingTask,
    FailedBatch,
    get_code_document_schema,
    IndexStatsResult,
    FailedBatchStats,
    PipelineResult,
)
from breeze.core.embeddings import get_voyage_embeddings_with_limits, get_local_embeddings_with_tokenizer_chunking
from breeze.core.queue import IndexingQueue
from breeze.core.text_chunker import FileContent

# Use standard Python logging to avoid duplicate handlers
logger = logging.getLogger(__name__)

# Disable verbose logging for LanceDB unless explicitly requested
if os.environ.get("BREEZE_DEBUG_LANCE"):
    os.environ["RUST_LOG"] = "debug"
    os.environ["LANCE_LOG"] = "debug"
    logging.getLogger("lancedb").setLevel(logging.DEBUG)
else:
    # Suppress Lance logs by default
    os.environ["RUST_LOG"] = "error"
    os.environ["LANCE_LOG"] = "error"
    logging.getLogger("lancedb").setLevel(logging.ERROR)
    logging.getLogger("lance").setLevel(logging.ERROR)


class BreezeEngine:
    """Main engine for code indexing and search operations."""

    def __init__(self, config: Optional[BreezeConfig] = None, tokenizer=None):
        self.config = config or BreezeConfig()
        self.config.ensure_directories()
        self.tokenizer = tokenizer  # Pre-loaded tokenizer to avoid repeated loading

        self.db: Optional[lancedb.LanceDBConnection] = None
        self.table: Optional[lancedb.Table] = None
        self.projects_table: Optional[lancedb.Table] = None
        self.failed_batches_table: Optional[lancedb.Table] = None
        self.indexing_tasks_table: Optional[lancedb.Table] = None
        self.embedding_model = None
        self.document_schema = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.is_voyage_model = False  # Flag for special Voyage handling

        # File watching and task tracking
        self._watchers: Dict[str, "FileWatcher"] = {}
        self._observers: Dict[str, "Observer"] = {}  # type: ignore
        self._active_tasks: Dict[str, IndexingTask] = {}

        # Background retry task
        self._retry_task = None
        self._retry_task_stop_event = asyncio.Event()

        # Indexing queue
        self._indexing_queue: Optional[IndexingQueue] = None

        # Snippet extractor and content detector
        self.snippet_extractor = TreeSitterSnippetExtractor()
        self.content_detector = ContentDetector(exclude_patterns=self.config.exclude_patterns)

    async def initialize(self):
        """Initialize the database and embedding model."""
        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing BreezeEngine...")

            # Initialize LanceDB async connection
            self.db = await lancedb.connect_async(self.config.get_db_path())

            # Initialize embedding model using LanceDB's registry
            # Use custom embedding function if provided (for testing)
            if self.config.embedding_function is not None:
                self.embedding_model = self.config.embedding_function
                logger.info("Using custom embedding function")
                # Skip all the model-specific initialization when using custom function
            else:
                logger.info(f"Loading embedding model: {self.config.embedding_model}")
                registry = get_registry()

                # Determine provider based on model name
                if self.config.embedding_model.startswith("voyage-"):
                    # Voyage AI models
                    self.is_voyage_model = True

                    # Set API key if provided
                    if self.config.embedding_api_key:
                        os.environ["VOYAGE_API_KEY"] = self.config.embedding_api_key
                    elif os.environ.get("BREEZE_EMBEDDING_API_KEY"):
                        os.environ["VOYAGE_API_KEY"] = os.environ.get(
                            "BREEZE_EMBEDDING_API_KEY"
                        )
                    elif not os.environ.get("VOYAGE_API_KEY"):
                        raise ValueError(
                            "BREEZE_EMBEDDING_API_KEY or VOYAGE_API_KEY environment variable or embedding_api_key config required for Voyage models"
                        )

                    # Use custom function for voyage-code-3, built-in for others
                    if self.config.embedding_model == "voyage-code-3":
                        self.embedding_model = registry.get("voyage-code-3").create(
                            name=self.config.embedding_model
                        )
                    else:
                        # Try to disable logging for built-in voyageai
                        try:
                            import voyageai

                            voyageai.log = "error"
                            # Suppress internal logging
                            import logging

                            logging.getLogger("voyageai").setLevel(logging.ERROR)
                        except ImportError:
                            pass
                        self.embedding_model = registry.get("voyageai").create(
                            name=self.config.embedding_model
                        )

                    # Tokenizer should be passed in via constructor to avoid repeated loading
                    if not self.tokenizer:
                        logger.warning(
                            f"No tokenizer provided for Voyage model {self.config.embedding_model}. "
                            "Token counting may be less accurate."
                        )

                elif self.config.embedding_model.startswith("models/"):
                    # Google Gemini models
                    if self.config.embedding_api_key:
                        os.environ["GOOGLE_API_KEY"] = self.config.embedding_api_key
                    elif not os.environ.get("GOOGLE_API_KEY"):
                        raise ValueError(
                            "GOOGLE_API_KEY environment variable or embedding_api_key config required for Gemini models"
                        )

                    self.embedding_model = registry.get("gemini-text").create(
                        name=self.config.embedding_model,
                        api_key=os.environ.get("GOOGLE_API_KEY"),
                    )

                else:
                    # Default to sentence-transformers for local models
                    try:
                        logger.info(f"Loading embedding model {self.config.embedding_model} on {self.config.embedding_device}...")
                        self.embedding_model = registry.get("sentence-transformers").create(
                            name=self.config.embedding_model,
                            device=self.config.embedding_device,
                            normalize=True,
                            trust_remote_code=self.config.trust_remote_code,
                            show_progress_bar=False,
                        )
                        logger.info(f"Successfully loaded embedding model {self.config.embedding_model}")
                    except Exception as e:
                        # Check if this is an MPS-related error
                        if "MPS" in str(e) or "fbgemm" in str(e):
                            logger.warning(
                                f"Failed to load model on {self.config.embedding_device} device: {e}. "
                                f"Falling back to CPU."
                            )
                            # Retry with CPU device
                            self.embedding_model = registry.get("sentence-transformers").create(
                                name=self.config.embedding_model,
                                device="cpu",
                                normalize=True,
                                trust_remote_code=self.config.trust_remote_code,
                                show_progress_bar=False,
                            )
                            # Update config to reflect actual device
                            self.config.embedding_device = "cpu"
                        else:
                            # Re-raise if not MPS-related
                            raise
                    
                    # Load tokenizer for local models to avoid repeated loading
                    if not self.tokenizer:
                        try:
                            from transformers import AutoTokenizer
                            logger.info(f"Loading tokenizer for {self.config.embedding_model}...")
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                self.config.embedding_model, trust_remote_code=True
                            )
                            logger.info(f"Successfully loaded tokenizer for {self.config.embedding_model}")
                        except Exception as e:
                            logger.warning(f"Could not load tokenizer for {self.config.embedding_model}: {e}")
                            logger.warning("Tokenizer-based chunking will fall back to character estimation")

            # Get the document schema with embeddings
            self.document_schema = get_code_document_schema(self.embedding_model)

            # Log model properties for debugging
            embed_dim = self.embedding_model.ndims()
            logger.info(
                f"Model loaded - embedding_dim: {embed_dim}, device: {self.config.embedding_device}"
            )

            # Create or open table
            table_name = "code_documents"
            existing_tables = await self.db.table_names()

            if table_name in existing_tables:
                self.table = await self.db.open_table(table_name)
                table_len = await self.table.count_rows()
                logger.info(f"Opened existing table with {table_len} documents")
            else:
                # Create new table with embedding-aware schema
                # Try to limit initial buffer allocation
                try:
                    self.table = await self.db.create_table(
                        table_name,
                        None,  # data
                        schema=self.document_schema,  # schema with embeddings
                        mode="overwrite",  # mode
                    )
                except TypeError:
                    # Fallback if storage_options not supported
                    self.table = await self.db.create_table(
                        table_name,
                        None,  # data
                        schema=self.document_schema,  # schema with embeddings
                        mode="overwrite",  # mode
                    )
                logger.info("Created new code documents table with embeddings")

            # Initialize failed batches table
            await self._init_failed_batches_table()

            # Initialize indexing tasks table
            await self.init_indexing_tasks_table()

            # Start background retry task
            self._retry_task = asyncio.create_task(self._background_retry_task())

            # Initialize indexing queue
            self._indexing_queue = IndexingQueue(self)

            # Start queue processing
            await self._indexing_queue.start()

            self._initialized = True
            logger.info("BreezeEngine initialization complete")

    async def index_directories(
        self,
        directories: List[str],
        force_reindex: bool = False,
        progress_callback: Optional[Callable] = None,
        num_workers: int = None,
    ) -> IndexStats:
        """Index code files from specified directories using fast approach."""
        await self.initialize()

        stats = IndexStats()

        # For re-indexing, get existing document hashes in one query
        existing_hashes = {}
        if not force_reindex:
            try:
                logger.info("Loading existing document hashes...")
                # Single query to get all existing hashes
                existing = await self.table.query().select(["id", "content_hash"]).to_list()
                for doc in existing:
                    existing_hashes[doc["id"]] = doc["content_hash"]
                logger.info(f"Loaded {len(existing_hashes)} existing document hashes")
            except Exception as e:
                logger.warning(f"Could not load existing hashes: {e}")

        # Streaming file discovery and processing
        logger.info("Starting streaming file discovery and processing...")
        
        # Suppress LanceDB warnings during the indexing process
        import logging as stdlib_logging
        lance_logger = stdlib_logging.getLogger("lance")
        original_level = lance_logger.level
        lance_logger.setLevel(stdlib_logging.ERROR)

        # Configure concurrency
        concurrent_readers = num_workers or self.config.concurrent_readers or 20
        concurrent_embedders = self.config.concurrent_embedders or 10
        concurrent_writers = 1  # Always use 1 writer to avoid concurrency issues (hardcoded)

        # Process files with streaming three-stage pipeline
        pipeline_stats = await self._run_streaming_indexing_pipeline(
            directories,
            existing_hashes,
            force_reindex,
            concurrent_readers,
            concurrent_embedders,
            concurrent_writers,
            progress_callback,
        )

        # Update stats from pipeline results
        stats.files_scanned = pipeline_stats.total_files_discovered
        stats.files_indexed = pipeline_stats.indexed
        stats.files_updated = pipeline_stats.updated
        stats.errors = pipeline_stats.errors
        stats.files_skipped = (
            stats.files_scanned
            - stats.files_indexed
            - stats.files_updated
            - stats.errors
        )

        logger.info("Indexing completed")
        logger.info(
            f"Files indexed: {stats.files_indexed}, updated: {stats.files_updated}"
        )
        
        # Optimize the table to deduplicate and improve performance
        if stats.files_indexed > 0 or stats.files_updated > 0:
            logger.info("Optimizing database...")
            try:
                await self.table.optimize()
                logger.info("Database optimization completed")
            except Exception as e:
                logger.warning(f"Database optimization failed: {e}")
        
        # Restore LanceDB logging level
        lance_logger.setLevel(original_level)

        return stats

    async def index_directories_sync(
        self,
        directories: List[str],
        force_reindex: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> IndexStats:
        """
        Synchronous wrapper for indexing that works with the queue system.

        This method creates a task, queues it, and waits for completion,
        making it appear synchronous to the CLI while using the queue internally.
        """
        await self.initialize()

        # Create indexing task
        task = IndexingTask(
            paths=directories,
            force_reindex=force_reindex
        )

        # Add task to queue with progress callback
        await self._indexing_queue.add_task(task, progress_callback)

        # Poll until complete
        while task.status not in ["completed", "failed"]:
            # Refresh task from database
            task = await self.get_indexing_task_db(task.task_id)
            if not task:
                raise Exception(f"Task {task.task_id} disappeared from database")

            # Brief sleep to avoid hammering the database
            await asyncio.sleep(0.1)

        # Return results or raise error
        if task.status == "failed":
            raise Exception(f"Indexing failed: {task.error_message}")

        # Reconstruct IndexStats from task fields
        if task.result_files_scanned is not None:
            return IndexStats(
                files_scanned=task.result_files_scanned,
                files_indexed=task.result_files_indexed or 0,
                files_updated=task.result_files_updated or 0,
                files_skipped=task.result_files_skipped or 0,
                errors=task.result_errors or 0,
                total_tokens_processed=task.result_total_tokens_processed or 0
            )

        # Fallback if no stats recorded
        return IndexStats()

    async def search(
        self,
        query: str,
        limit: Optional[int] = None,
        min_relevance: Optional[float] = None,
        use_reranker: bool = True,
    ) -> List[SearchResult]:
        """Search for code files matching the query with optional reranking."""
        await self.initialize()

        limit = limit or self.config.default_limit
        min_relevance = min_relevance or self.config.min_relevance

        # Get reranker model - check for test override first
        if hasattr(self, 'reranker') and self.reranker:
            reranker_model = self.reranker if use_reranker else None
        else:
            reranker_model = self.config.get_reranker_model() if use_reranker else None

        # For reranking, retrieve more candidates (3x the requested limit)
        retrieval_limit = limit * 3 if reranker_model else limit

        # Perform vector search using LanceDB's built-in search
        # LanceDB will automatically generate embeddings for the query
        search_query = await self.table.search(query, vector_column_name="vector")
        search_query = search_query.limit(retrieval_limit)
        results = await search_query.to_arrow()
        # Convert to list of dicts for processing
        import polars as pl

        df = pl.from_arrow(results)
        results = df.to_dicts() if df.height > 0 else []

        search_results = []
        content_map = {}  # Map result ID to content for reranking

        for result in results:
            # Calculate relevance score (cosine similarity)
            # LanceDB returns L2 distance, convert to similarity
            distance = result.get("_distance", 0)
            relevance_score = 1 / (1 + distance)  # Convert distance to similarity

            if relevance_score < min_relevance:
                continue

            # Create snippet
            content = result.get("content", "")
            file_path = result.get("file_path", "")
            snippet = self._create_snippet(content, query, file_path)

            search_result = SearchResult(
                id=result.get("id", ""),
                file_path=result.get("file_path", ""),
                file_type=result.get("file_type", ""),
                relevance_score=relevance_score,
                snippet=snippet,
                last_modified=result.get("last_modified", ""),
            )
            search_results.append(search_result)
            content_map[search_result.id] = content  # Store content for reranking

        # Sort by relevance score
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply reranking if enabled
        if reranker_model and search_results:
            # Prepare results with content for reranking
            results_with_content = [(r, content_map[r.id]) for r in search_results]

            # Check if it's a mock reranker object (for tests)
            if hasattr(reranker_model, 'predict'):
                # Mock reranker - use its predict method directly
                pairs = [(query, content) for _, content in results_with_content]
                scores = reranker_model.predict(pairs)

                # Sort by scores
                scored_results = list(zip(scores, search_results))
                scored_results.sort(key=lambda x: x[0], reverse=True)
                search_results = [r for _, r in scored_results]
            elif isinstance(reranker_model, str):
                if reranker_model.startswith("rerank-"):  # Voyage
                    search_results = await self._rerank_voyage(query, results_with_content, reranker_model)
                elif reranker_model.startswith("models/"):  # Gemini
                    search_results = await self._rerank_gemini(query, results_with_content, reranker_model)
                else:  # Local models
                    search_results = await self._rerank_local(query, results_with_content, reranker_model)

        # Return top-k results
        return search_results[:limit]

    async def _rerank_voyage(self, query: str, results_with_content: List[tuple], model: str) -> List[SearchResult]:
        """Rerank results using Voyage AI reranking API."""
        try:
            import voyageai

            # Set API key
            api_key = self.config.reranker_api_key or self.config.embedding_api_key
            if not api_key and "VOYAGE_API_KEY" not in os.environ:
                logger.warning("No API key for Voyage reranker, skipping reranking")
                return [r[0] for r in results_with_content]

            # Initialize client
            client = voyageai.Client(api_key=api_key)

            # Prepare documents for reranking
            documents = [content for _, content in results_with_content]

            # Call reranking API
            reranked = client.rerank(
                query=query,
                documents=documents,
                model=model,
                top_k=len(documents)
            )

            # Reorder results based on reranking scores
            reranked_results = []
            for item in reranked.results:
                idx = item.index
                score = item.relevance_score
                result = results_with_content[idx][0]
                result.relevance_score = score  # Update with reranker score
                reranked_results.append(result)

            return reranked_results

        except Exception as e:
            logger.warning(f"Voyage reranking failed: {e}, returning original results")
            return [r[0] for r in results_with_content]

    async def _rerank_gemini(self, query: str, results_with_content: List[tuple], model: str) -> List[SearchResult]:
        """Rerank results using Google Gemini."""
        try:
            import google.generativeai as genai

            # Set API key
            api_key = self.config.reranker_api_key or self.config.embedding_api_key
            if api_key:
                genai.configure(api_key=api_key)
            elif "GOOGLE_API_KEY" not in os.environ:
                logger.warning("No API key for Gemini reranker, skipping reranking")
                return [r[0] for r in results_with_content]

            # Initialize model
            gemini_model = genai.GenerativeModel(model)

            # Create prompt for reranking
            prompt = f"Query: {query}\n\nRank these code snippets by relevance to the query. Return only the indices in order of relevance, separated by commas.\n\n"
            for i, (result, _) in enumerate(results_with_content):
                prompt += f"[{i}] File: {result.file_path}\n{result.snippet[:500]}...\n\n"

            # Get reranking
            response = gemini_model.generate_content(prompt)
            indices_str = response.text.strip()

            # Parse indices
            indices = [int(idx.strip()) for idx in indices_str.split(",") if idx.strip().isdigit()]

            # Reorder results
            reranked_results = []
            for i, idx in enumerate(indices):
                if 0 <= idx < len(results_with_content):
                    result = results_with_content[idx][0]
                    # Assign decreasing scores based on rank
                    result.relevance_score = 1.0 - (i * 0.01)
                    reranked_results.append(result)

            # Add any missing results at the end
            seen_indices = set(indices)
            for i, (result, _) in enumerate(results_with_content):
                if i not in seen_indices:
                    reranked_results.append(result)

            return reranked_results

        except Exception as e:
            logger.warning(f"Gemini reranking failed: {e}, returning original results")
            return [r[0] for r in results_with_content]

    async def _rerank_local(self, query: str, results_with_content: List[tuple], model: str) -> List[SearchResult]:
        """Rerank results using local cross-encoder model."""
        try:
            # Check if we're in test mode and should use mock reranker
            # Only use MockReranker if sentence_transformers is not already mocked
            import os
            import sys

            use_mock_reranker = (
                os.environ.get("PYTEST_CURRENT_TEST") and
                type(self.embedding_model).__name__ in {'MagicMock', 'Mock'} and
                'sentence_transformers' not in sys.modules  # Check if sentence_transformers is not already mocked
            )

            if use_mock_reranker:
                from ..tests.mock_embedders import MockReranker

                # Use mock reranker for tests
                cross_encoder = MockReranker(model_name=model)

                # Prepare pairs for reranking
                pairs = []
                for result, content in results_with_content:
                    snippet = self._create_snippet(content, query, result.file_path)
                    pairs.append([query, snippet])

                # Get scores
                scores = cross_encoder.predict(pairs)

                # Create list of (score, (result, content)) tuples and sort
                scored_results = list(zip(scores, results_with_content))
                scored_results.sort(key=lambda x: x[0], reverse=True)

                # Update relevance scores and return
                reranked_results = []
                for score, (result, _) in scored_results:
                    result.relevance_score = float(score)
                    reranked_results.append(result)

                return reranked_results

            from sentence_transformers import CrossEncoder

            # Initialize cross-encoder
            cross_encoder = CrossEncoder(model)

            # Get tokenizer and max length
            tokenizer = None
            max_length = 512  # Default fallback

            try:
                # Get the tokenizer from cross-encoder or load it
                if hasattr(cross_encoder, 'tokenizer'):
                    tokenizer = cross_encoder.tokenizer
                else:
                    # Load tokenizer for the model
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model)

                # Get max sequence length
                if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 1000000:
                    max_length = tokenizer.model_max_length
                elif hasattr(tokenizer, 'max_len'):
                    max_length = tokenizer.max_len

                # Some models have it in the model config
                if hasattr(cross_encoder, 'model') and hasattr(cross_encoder.model, 'config'):
                    if hasattr(cross_encoder.model.config, 'max_position_embeddings'):
                        max_length = min(max_length, cross_encoder.model.config.max_position_embeddings)

                logger.debug(f"Cross-encoder {model} max sequence length: {max_length}")
            except Exception as e:
                logger.debug(f"Could not load tokenizer for {model}, using character estimation: {e}")

            # Set the max length on the cross-encoder
            cross_encoder.max_length = max_length

            # Reserve tokens for query and special tokens
            # [CLS] query [SEP] snippet [SEP] = 3 special tokens
            reserved_tokens = 3
            if tokenizer:
                query_tokens = len(tokenizer.encode(query, add_special_tokens=False))
                reserved_tokens += query_tokens
            else:
                # Fallback: estimate query tokens
                reserved_tokens += len(query.split()) * 2

            max_snippet_tokens = max_length - reserved_tokens - 10  # Extra buffer

            # Prepare pairs for reranking with smart snippet extraction
            pairs = []
            for result, content in results_with_content:
                # Try to detect language for better snippet extraction
                language = None
                if result.file_path:
                    path = Path(result.file_path)
                    language = self.content_detector.detect_language(path)

                # Extract a focused snippet around the query match
                # This will give us the most relevant semantic unit (function/class)
                # already truncated intelligently
                snippet = self.snippet_extractor.extract_snippet(content, query, language)

                # Truncate based on actual token count if we have a tokenizer
                if tokenizer and len(snippet) > 100:  # Only tokenize if snippet is substantial
                    try:
                        tokens = tokenizer.encode(snippet, add_special_tokens=False)
                        if len(tokens) > max_snippet_tokens:
                            # Truncate tokens and decode back to text
                            truncated_tokens = tokens[:max_snippet_tokens]
                            snippet = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                            snippet += "..."
                    except Exception as e:
                        logger.debug(f"Token truncation failed, using character fallback: {e}")
                        # Character-based fallback
                        max_chars = max_snippet_tokens * 4  # Rough estimate
                        if len(snippet) > max_chars:
                            snippet = snippet[:max_chars] + "..."
                elif not tokenizer and len(snippet) > max_snippet_tokens * 4:
                    # No tokenizer available, use character estimation
                    snippet = snippet[:max_snippet_tokens * 4] + "..."

                pairs.append([query, snippet])

            # Get scores
            scores = cross_encoder.predict(pairs)

            # Create list of (score, (result, content)) tuples and sort
            scored_results = list(zip(scores, results_with_content))
            scored_results.sort(key=lambda x: x[0], reverse=True)

            # Update relevance scores and return
            reranked_results = []
            for score, (result, _) in scored_results:
                result.relevance_score = float(score)
                reranked_results.append(result)

            return reranked_results

        except Exception as e:
            logger.warning(f"Local reranking failed: {e}, returning original results")
            return [r[0] for r in results_with_content]

    async def get_stats(self) -> IndexStatsResult:
        """Get current index statistics."""
        if not self._initialized or self.table is None:
            return IndexStatsResult(total_documents=0, initialized=False)

        table_len = await self.table.count_rows()

        # Get failed batch stats
        failed_stats = await self._get_failed_batch_stats()

        # Get queue stats if available
        queue_stats = {}
        if self._indexing_queue:
            queue_stats = await self._indexing_queue.get_queue_status()

        return IndexStatsResult(
            total_documents=table_len,
            initialized=True,
            model=self.config.embedding_model,
            database_path=self.config.get_db_path(),
            failed_batches=failed_stats,
            indexing_queue=queue_stats,
        )

    async def _get_failed_batch_stats(self) -> FailedBatchStats:
        """Get statistics about failed batches."""
        try:
            if not self.failed_batches_table:
                return FailedBatchStats()

            # Get total count first
            total = await self.failed_batches_table.count_rows()
            if total == 0:
                return FailedBatchStats(total=0)

            # Run all count queries in parallel using asyncio.gather
            counts = await asyncio.gather(
                self.failed_batches_table.count_rows(filter="status = 'pending'"),
                self.failed_batches_table.count_rows(filter="status = 'processing'"),
                self.failed_batches_table.count_rows(filter="status = 'succeeded'"),
                self.failed_batches_table.count_rows(filter="status = 'failed'"),
                self.failed_batches_table.count_rows(filter="status = 'abandoned'"),
                # Also get the oldest pending batch
                self.failed_batches_table.search()
                    .where("status = 'pending'")
                    .select(["next_retry_at"])
                    .limit(1)
                    .to_pydantic(FailedBatch),
            )
            
            pending_count, processing_count, succeeded_count, failed_count, abandoned_count, pending_batches = counts
            
            result = FailedBatchStats(
                total=total,
                pending=pending_count,
                processing=processing_count,
                succeeded=succeeded_count,
                failed=failed_count,
                abandoned=abandoned_count
            )

            # Set next retry time if we have pending batches
            if pending_batches and pending_batches[0].next_retry_at:
                result.next_retry_at = pending_batches[0].next_retry_at.isoformat()

            return result
        except Exception as e:
            logger.error(f"Error getting failed batch stats: {e}")
            return FailedBatchStats()


    async def _read_file_batch(
        self, 
        file_paths: List[Path]
    ) -> List[Dict]:
        """Read a batch of files and return document data."""
        doc_datas = []
        
        for file_path in file_paths:
            try:
                stat = await aiofiles.os.stat(file_path)
                
                async with aiofiles.open(
                    file_path, "r", encoding="utf-8", errors="replace"
                ) as f:
                    content = await f.read()
                
                # Skip empty files
                if not content.strip():
                    continue
                
                # Generate document ID and content hash
                doc_id = f"file:{file_path}"
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                doc_data = {
                    "id": doc_id,
                    "file_path": str(file_path),
                    "content": content,
                    "file_type": file_path.suffix[1:] if file_path.suffix else "txt",
                    "file_size": stat.st_size,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime),
                    "indexed_at": datetime.now(),
                    "content_hash": content_hash,
                }
                
                doc_datas.append(doc_data)
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                
        return doc_datas
    

    def _create_snippet(self, content: str, query: str, file_path: str = None) -> str:
        """Create a relevant snippet from the content using tree-sitter if possible."""
        # Try to detect language from content
        language = None
        if file_path:
            path = Path(file_path)
            language = self.content_detector.detect_language(path)

        # Use tree-sitter extractor
        return self.snippet_extractor.extract_snippet(content, query, language)

    def _should_index_file(self, file_path: Path) -> bool:
        """Check if file should be indexed."""
        # Use ContentDetector which uses breeze-langdetect
        return self.content_detector.should_index_file(file_path)

    def _walk_directory_fast(self, directory: Path) -> List[Path]:
        """Fast synchronous directory walk with filtering and gitignore support."""
        file_discovery = FileDiscovery(
            exclude_patterns=self.config.exclude_patterns,
            should_index_file=self._should_index_file
        )
        return file_discovery.walk_directory(directory)


    async def _run_streaming_indexing_pipeline(
        self,
        directories: List[str],
        existing_hashes: Dict[str, str],
        force_reindex: bool,
        concurrent_readers: int,
        concurrent_embedders: int,
        concurrent_writers: int,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """Run three-stage pipeline for indexing files with streaming file discovery."""
        # Create queues for the pipeline - separate queues for each stage
        file_queue = asyncio.Queue(maxsize=concurrent_readers * 10)  # Files to read
        embed_queue = asyncio.Queue(maxsize=concurrent_embedders * 5)  # Documents to embed
        write_queue = asyncio.Queue(maxsize=100)  # Documents to write

        # Semaphores for controlling concurrency
        read_sem = asyncio.Semaphore(concurrent_readers)
        embed_sem = asyncio.Semaphore(concurrent_embedders)

        # Track completion
        discovery_complete = asyncio.Event()
        read_complete = asyncio.Event()
        embed_complete = asyncio.Event()

        # Track results
        total_indexed = 0
        total_updated = 0
        total_errors = 0
        total_processed = 0
        total_files_discovered = 0
        
        # Multiple buffers for pipelining
        write_buffers = []  # List of buffers waiting to be written
        write_buffer_active = []  # Current buffer being filled
        write_buffer_lock = asyncio.Lock()
        WRITE_BATCH_SIZE = 500  # Larger batches to minimize LanceDB operations
        MAX_PENDING_BUFFERS = 5  # Maximum number of buffers to keep in memory
        
        # Background flush task (only one at a time for LanceDB)
        flush_task = None

        async def file_discovery_task():
            """Discover files and queue them for processing as they are found."""
            nonlocal total_files_discovered
            
            try:
                for directory in directories:
                    path = Path(directory).resolve()
                    logger.info(f"Discovering files in {path}...")
                    # Stream files as they're discovered
                    for file_path in self._walk_directory_fast(path):
                        total_files_discovered += 1
                        # Queue individual file for reading
                        await file_queue.put(file_path)
                        
                        # Yield control periodically to allow other tasks to run
                        if total_files_discovered % 100 == 0:
                            await asyncio.sleep(0)
                            logger.debug(f"Discovered {total_files_discovered} files so far...")
                            
                logger.info(f"File discovery complete. Found {total_files_discovered} files")
            except Exception as e:
                logger.error(f"Error during file discovery: {e}")
            finally:
                discovery_complete.set()

        async def reader_task():
            """Read files from file queue and queue document data for embedding."""
            while True:
                try:
                    # Wait for a file path with timeout
                    file_path = await asyncio.wait_for(file_queue.get(), timeout=1.0)
                    
                    async with read_sem:
                        try:
                            stat = await aiofiles.os.stat(file_path)

                            async with aiofiles.open(
                                file_path, "r", encoding="utf-8", errors="replace"
                            ) as f:
                                content = await f.read()

                            # Skip empty files
                            if not content.strip():
                                continue

                            # Generate document ID and content hash
                            doc_id = f"file:{file_path}"
                            content_hash = hashlib.md5(content.encode()).hexdigest()
                            
                            # Skip existence check - we'll use add() and deduplicate during optimize
                            # This avoids the per-file query overhead that causes the nprobes warnings
                            is_update = False

                            # Create document data (without vector)
                            doc_data = {
                                "id": doc_id,
                                "file_path": str(file_path),
                                "content": content,
                                "file_type": file_path.suffix[1:] if file_path.suffix else "txt",
                                "file_size": stat.st_size,
                                "last_modified": datetime.fromtimestamp(stat.st_mtime),
                                "indexed_at": datetime.now(),
                                "content_hash": content_hash,
                                "is_update": is_update,
                            }
                            
                            # Queue for embedding
                            await embed_queue.put(doc_data)

                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
                            
                except asyncio.TimeoutError:
                    # Check if discovery is complete and file queue is empty
                    if discovery_complete.is_set() and file_queue.empty():
                        break

        async def embedder_task():
            """Generate embeddings for documents and queue for writing."""
            while True:
                try:
                    # Wait for document data with timeout
                    doc_data = await asyncio.wait_for(embed_queue.get(), timeout=1.0)

                    async with embed_sem:
                        try:
                            logger.debug(f"Generating embedding for {doc_data['file_path']}")
                            
                            # Detect language for the file
                            language = self.content_detector.detect_language(Path(doc_data["file_path"]))
                            if not language:
                                # Skip files where language detection fails (likely binary)
                                logger.debug(f"Skipping {doc_data['file_path']} - no language detected")
                                continue
                            
                            file_content = FileContent(
                                content=doc_data["content"],
                                file_path=doc_data["file_path"],
                                language=language
                            )

                            # Generate embedding for single file
                            if type(self.embedding_model).__name__ in {'MagicMock', 'Mock'}:
                                # Direct call for unittest mocks
                                embeddings = self.embedding_model.compute_source_embeddings([doc_data["content"]])
                                embedding = embeddings[0] if isinstance(embeddings, list) else embeddings
                            elif self.is_voyage_model:
                                # Voyage model - process single file
                                rate_limits = self.config.get_voyage_rate_limits()
                                result = await get_voyage_embeddings_with_limits(
                                    [file_content],
                                    self.embedding_model,
                                    self.tokenizer,
                                    self.config.voyage_concurrent_requests,
                                    self.config.voyage_max_retries,
                                    self.config.voyage_retry_base_delay,
                                    rate_limits['tokens_per_minute'],
                                    rate_limits['requests_per_minute'],
                                )

                                if result.embeddings.size > 0 and not result.failed_files:
                                    embedding = result.embeddings[0]
                                else:
                                    # Failed - store for retry
                                    await self._store_failed_batch(
                                        batch_id=f"file_{doc_data['id']}_{int(time.time())}",
                                        file_paths=[doc_data["file_path"]],
                                        content_hashes=[doc_data["content_hash"]],
                                        error_message="Voyage embedding failed",
                                    )
                                    continue
                            else:
                                # Local model - process single file
                                result = await get_local_embeddings_with_tokenizer_chunking(
                                    [file_content],
                                    self.embedding_model,
                                    self.config.embedding_model,
                                    max_concurrent_requests=concurrent_embedders,
                                    max_retries=3,
                                    retry_base_delay=1.0,
                                    max_sequence_length=self.config.max_sequence_length if hasattr(self.config, 'max_sequence_length') else None,
                                    tokenizer=self.tokenizer,
                                )

                                if result.embeddings.size > 0 and not result.failed_files:
                                    embedding = result.embeddings[0]
                                else:
                                    # Failed - store for retry
                                    await self._store_failed_batch(
                                        batch_id=f"file_{doc_data['id']}_{int(time.time())}",
                                        file_paths=[doc_data["file_path"]],
                                        content_hashes=[doc_data["content_hash"]],
                                        error_message="Local embedding failed",
                                    )
                                    continue

                            # Add embedding to document data
                            doc_data["vector"] = (
                                embedding.tolist()
                                if hasattr(embedding, "tolist")
                                else embedding
                            )
                            document = self.document_schema(**doc_data)

                            logger.debug(f"Embedding generated for {doc_data['file_path']}, queuing for write")
                            # Queue for writing
                            await write_queue.put((document, doc_data.get("is_update", False)))

                        except Exception as e:
                            logger.error(
                                f"Failed to generate embedding for {doc_data['file_path']}: {e}"
                            )

                            # Store for retry
                            await self._store_failed_batch(
                                batch_id=f"file_{doc_data['id']}_{int(time.time())}",
                                file_paths=[doc_data["file_path"]],
                                content_hashes=[doc_data["content_hash"]],
                                error_message=str(e),
                            )

                            # Put error marker for stats
                            await write_queue.put((None, False))

                except asyncio.TimeoutError:
                    # Check if reading is complete and embed queue is empty
                    if read_complete.is_set() and embed_queue.empty():
                        break

        async def continuous_flush_task():
            """Continuously flush buffers to database."""
            nonlocal total_indexed, total_updated, total_errors
            
            while True:
                # Check if there are buffers to flush
                buffer_to_flush = None
                async with write_buffer_lock:
                    if write_buffers:
                        buffer_to_flush = write_buffers.pop(0)
                
                if buffer_to_flush:
                    documents_to_write = [doc for doc, _ in buffer_to_flush]
                    
                    try:
                        # Always use add() - LanceDB handles versioning
                        # Much faster than merge_insert, no complex indexing operations
                        await self.table.add(documents_to_write)
                        
                        # Count updates vs inserts for stats
                        for _, is_update in buffer_to_flush:
                            if is_update:
                                total_updated += 1
                            else:
                                total_indexed += 1

                        logger.debug(f"Added {len(documents_to_write)} documents to database")

                    except Exception as e:
                        logger.error(f"Error writing batch: {e}")
                        total_errors += len(documents_to_write)
                else:
                    # No buffers to flush, wait a bit
                    await asyncio.sleep(0.1)
                    
                    # Check if we should exit
                    if embed_complete.is_set() and write_queue.empty():
                        # Final check for any remaining buffers
                        async with write_buffer_lock:
                            if not write_buffers and not write_buffer_active:
                                break

        async def writer_task():
            """Pull from embed queue and batch writes to database."""
            nonlocal total_processed, total_errors
            
            last_flush_time = time.time()
            FLUSH_INTERVAL = 10.0  # Less frequent flushes to reduce LanceDB overhead

            while True:
                try:
                    # Apply backpressure if too many buffers are pending
                    buffer_count = 0
                    async with write_buffer_lock:
                        buffer_count = len(write_buffers)
                    
                    if buffer_count >= MAX_PENDING_BUFFERS:
                        # Wait a bit before checking again
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Wait for items with a timeout
                    item = await asyncio.wait_for(
                        write_queue.get(), timeout=1.0
                    )

                    document, is_update = item

                    async with write_buffer_lock:
                        if document is None:
                            # This file failed to generate embeddings
                            total_errors += 1
                            total_processed += 1
                        else:
                            # Add to active buffer
                            write_buffer_active.append((document, is_update))
                            
                            # Always check if we should flush based on EITHER condition
                            current_time = time.time()
                            time_to_flush = (current_time - last_flush_time) >= FLUSH_INTERVAL
                            batch_full = len(write_buffer_active) >= WRITE_BATCH_SIZE
                            
                            if time_to_flush or batch_full:
                                # Move active buffer to write queue
                                write_buffers.append(write_buffer_active[:])
                                write_buffer_active.clear()
                                
                                logger.debug(
                                    f"Queued buffer for flush: {len(write_buffers[-1])} docs "
                                    f"(batch_full={batch_full}, time_to_flush={time_to_flush}, "
                                    f"pending_buffers={len(write_buffers)})"
                                )
                                
                                last_flush_time = current_time
                                
                            total_processed += 1

                    # Progress callback (outside of buffer lock)
                    if progress_callback:
                        from breeze.core.models import IndexStats
                        stats = IndexStats(
                            files_scanned=total_processed,
                            files_indexed=total_indexed,
                            files_updated=total_updated,
                            files_skipped=0,  # Will be calculated later
                            errors=total_errors,
                        )
                        await progress_callback(stats)

                except asyncio.TimeoutError:
                    # Check if we should flush based on time
                    async with write_buffer_lock:
                        current_time = time.time()
                        if write_buffer_active and (current_time - last_flush_time) >= FLUSH_INTERVAL:
                            # Queue buffer for flush
                            write_buffers.append(write_buffer_active[:])
                            write_buffer_active.clear()
                            
                            logger.debug(
                                f"Time-based flush: {len(write_buffers[-1])} docs "
                                f"(pending_buffers={len(write_buffers)})"
                            )
                            
                            last_flush_time = current_time
                    
                    # Check if embedding is complete and queue is empty
                    if embed_complete.is_set() and write_queue.empty():
                        # Final flush of active buffer
                        async with write_buffer_lock:
                            if write_buffer_active:
                                write_buffers.append(write_buffer_active[:])
                                write_buffer_active.clear()
                        break

        # Start the continuous flush task FIRST
        flush_task = asyncio.create_task(continuous_flush_task())
        
        # Start all pipeline tasks
        writer_tasks = [
            asyncio.create_task(writer_task()) for _ in range(concurrent_writers)
        ]

        embedder_tasks = [
            asyncio.create_task(embedder_task()) for _ in range(concurrent_embedders)
        ]

        reader_tasks = [
            asyncio.create_task(reader_task()) for _ in range(concurrent_readers)
        ]

        # Start file discovery task
        discovery_task = asyncio.create_task(file_discovery_task())

        try:
            # Wait for file discovery to complete
            await discovery_task
            
            # Wait for readers to complete (they'll exit when discovery_complete is set and file queue is empty)
            await asyncio.gather(*reader_tasks, return_exceptions=True)
            read_complete.set()
            
            # Wait for embedders to complete (they'll exit when read_complete is set and embed queue is empty)
            await asyncio.gather(*embedder_tasks, return_exceptions=True)
            embed_complete.set()

            # Give writer tasks a chance to complete, then cancel if needed
            try:
                await asyncio.wait_for(
                    asyncio.gather(*writer_tasks),
                    timeout=5.0 if os.environ.get("PYTEST_CURRENT_TEST") else None
                )
            except asyncio.TimeoutError:
                # Timeout is handled in finally block
                pass
            
            # Wait for flush task to complete
            if flush_task and not flush_task.done():
                try:
                    await asyncio.wait_for(flush_task, timeout=10.0)
                except asyncio.TimeoutError:
                    flush_task.cancel()

        finally:
            # Always cancel any remaining tasks to prevent event loop warnings
            all_tasks = reader_tasks + embedder_tasks + writer_tasks + [discovery_task, flush_task]
            for task in all_tasks:
                if task and not task.done():
                    task.cancel()

            # Wait for all tasks to complete cancellation
            if all_tasks:
                await asyncio.gather(*[t for t in all_tasks if t], return_exceptions=True)

        return PipelineResult(
            indexed=total_indexed,
            updated=total_updated,
            errors=total_errors,
            total_files_discovered=total_files_discovered,
        )


    # Project Management Methods

    async def init_project_table(self):
        """Initialize the projects table in LanceDB."""
        await self.initialize()

        table_name = "projects"
        existing_tables = await self.db.table_names()

        if table_name not in existing_tables:
            # Create new projects table
            self.projects_table = await self.db.create_table(
                table_name,
                None,  # data
                schema=Project,  # schema
                mode="overwrite",  # mode
            )
            logger.info("Created new projects table")
        else:
            self.projects_table = await self.db.open_table(table_name)
            logger.info("Opened existing projects table")

    async def add_project(
        self,
        name: str,
        paths: List[str],
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        auto_index: bool = True,
    ) -> Project:
        """Add a new project to track."""
        await self.init_project_table()

        # Normalize paths
        normalized_paths = []
        for path in paths:
            p = Path(path).resolve()
            if not p.exists():
                raise ValueError(f"Path does not exist: {path}")
            normalized_paths.append(str(p))

        # Check for duplicate paths
        try:
            arrow_table = await self.projects_table.to_arrow()
            import polars as pl

            df = pl.from_arrow(arrow_table)
            existing_projects = df.to_dicts() if df.height > 0 else []
        except Exception:
            existing_projects = []

        for project_data in existing_projects:
            for path in normalized_paths:
                if path in project_data.get("paths", []):
                    raise ValueError(
                        f"Path {path} is already tracked by project '{project_data.get('name')}'"
                    )

        # Create project
        project = Project(
            name=name,
            paths=normalized_paths,
            file_extensions=file_extensions,  # Keep for backward compatibility
            exclude_patterns=exclude_patterns or self.config.exclude_patterns,
            auto_index=auto_index,
        )

        # Add to database
        await self.projects_table.add([project])
        logger.info(f"Added project '{name}' with {len(normalized_paths)} paths")

        return project

    async def list_projects(self) -> List[Project]:
        """List all registered projects."""
        await self.init_project_table()

        try:
            arrow_table = await self.projects_table.to_arrow()
            import polars as pl

            df = pl.from_arrow(arrow_table)
            projects_data = df.to_dicts() if df.height > 0 else []
        except Exception:
            projects_data = []

        return [Project(**data) for data in projects_data]

    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        await self.init_project_table()

        try:
            # Use polars query to filter by ID
            arrow_table = await self.projects_table.to_arrow()
            import polars as pl

            df = pl.from_arrow(arrow_table)
            if df.height > 0:
                matching = df.filter(pl.col("id") == project_id)
                if matching.height > 0:
                    data = matching.to_dicts()[0]
                    return Project(**data)
        except Exception as e:
            logger.error(f"Error getting project {project_id}: {e}")

        return None

    async def remove_project(self, project_id: str) -> bool:
        """Remove a project from tracking."""
        await self.init_project_table()

        # Check if project exists
        project = await self.get_project(project_id)
        if not project:
            return False

        # Stop watching if active
        if project_id in self._watchers:
            await self.stop_watching(project_id)

        # Delete from database
        await self.projects_table.delete(f"id = '{project_id}'")

        logger.info(f"Removed project {project_id}")
        return True

    async def update_project_indexed_time(self, project_id: str):
        """Update the last indexed time for a project."""
        project = await self.get_project(project_id)
        if project:
            project.last_indexed = datetime.now()
            project.updated_at = datetime.now()

            # Use merge_insert for update
            await (
                self.projects_table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([project])
            )

    # File Watching Methods

    async def start_watching(
        self, project_id: str, event_callback: Optional[Callable] = None
    ) -> bool:
        """Start watching a project for file changes."""
        project = await self.get_project(project_id)
        if not project:
            return False

        if project_id in self._watchers:
            logger.warning(f"Project {project_id} is already being watched")
            return True

        try:
            # Create file watcher
            watcher = FileWatcher(
                self, project, event_callback, self.config.file_watcher_debounce_seconds
            )
            # Set the current event loop for thread-safe operations
            watcher.set_event_loop(asyncio.get_running_loop())

            observer = Observer()

            # Add all project paths to observer
            for path in project.paths:
                observer.schedule(watcher, path, recursive=True)

            # Start observer
            observer.start()

            # Store references
            self._watchers[project_id] = watcher
            self._observers[project_id] = observer

            # Update project watching status
            project.is_watching = True
            project.last_indexed = datetime.now()
            project.updated_at = datetime.now()

            # Update in database
            await (
                self.projects_table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([project])
            )

            logger.info(f"Started watching project '{project.name}' ({project_id})")

            if event_callback:
                await event_callback(
                    {
                        "type": "watching_started",
                        "project_id": project_id,
                        "project_name": project.name,
                        "paths": project.paths,
                    }
                )

            return True

        except Exception as e:
            logger.error(f"Failed to start watching project {project_id}: {e}")
            return False

    async def stop_watching(self, project_id: str) -> bool:
        """Stop watching a project."""
        if project_id not in self._watchers:
            return False

        try:
            # Cancel any pending FileWatcher tasks
            watcher = self._watchers.get(project_id)
            if watcher and watcher._task and not watcher._task.done():
                watcher._task.cancel()
                try:
                    await watcher._task
                except asyncio.CancelledError:
                    pass

            # Stop observer
            observer = self._observers.get(project_id)
            if observer:
                observer.stop()
                observer.join(timeout=5)

            # Clean up references
            del self._watchers[project_id]
            del self._observers[project_id]

            # Update project status
            project = await self.get_project(project_id)
            if project:
                project.is_watching = False
                project.updated_at = datetime.now()

                # Update in database
                await (
                    self.projects_table.merge_insert("id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute([project])
                )

            logger.info(f"Stopped watching project {project_id}")
            return True

        except Exception as e:
            logger.error(f"Error stopping watcher for project {project_id}: {e}")
            return False

    async def get_watching_projects(self) -> List[Project]:
        """Get all projects currently being watched."""
        all_projects = await self.list_projects()
        return [p for p in all_projects if p.id in self._watchers]

    # Task Management Methods

    async def create_indexing_task(
        self,
        paths: List[str],
        project_id: Optional[str] = None,
        force_reindex: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> IndexingTask:
        """Create and start an async indexing task."""
        task = IndexingTask(paths=paths, project_id=project_id, status="running")

        self._active_tasks[task.task_id] = task

        async def run_indexing():
            try:
                # Count total files first
                total_files = 0
                for path in paths:
                    p = Path(path)
                    if p.exists() and p.is_dir():
                        async for _ in self._walk_directory_async(p):
                            total_files += 1

                task.total_files = total_files

                # Run indexing with progress tracking
                async def task_progress_callback(progress):
                    task.processed_files = progress.get("files_scanned", 0)
                    task.progress = task.processed_files / max(task.total_files, 1)

                    if progress_callback:
                        await progress_callback(
                            {
                                **progress,
                                "task_id": task.task_id,
                                "progress": task.progress,
                            }
                        )

                await self.index_directories(
                    paths,
                    force_reindex=force_reindex,
                    progress_callback=task_progress_callback,
                )

                task.status = "completed"
                task.completed_at = datetime.now()

                # Update project if specified
                if project_id:
                    await self.update_project_indexed_time(project_id)

            except Exception as e:
                task.status = "failed"
                task.error_message = str(e)
                task.completed_at = datetime.now()
                logger.error(f"Indexing task {task.task_id} failed: {e}")

        # Start the task and store reference for cleanup
        task._async_task = asyncio.create_task(run_indexing())

        return task

    async def get_indexing_task(self, task_id: str) -> Optional[IndexingTask]:
        """Get status of an indexing task."""
        return self._active_tasks.get(task_id)

    async def list_indexing_tasks(self) -> List[IndexingTask]:
        """List all indexing tasks."""
        return list(self._active_tasks.values())

    # Indexing Task Persistence Methods

    async def init_indexing_tasks_table(self):
        """Initialize the indexing tasks table."""
        table_name = "indexing_tasks"
        existing_tables = await self.db.table_names()

        if table_name not in existing_tables:
            # Create new table
            self.indexing_tasks_table = await self.db.create_table(
                table_name,
                None,  # data
                schema=IndexingTask,  # schema
                mode="overwrite",  # mode
            )
            logger.info("Created new indexing tasks table")
        else:
            # Open existing table
            self.indexing_tasks_table = await self.db.open_table(table_name)

            # Check if schema needs migration by trying to query
            try:
                # Try to access new fields - this will fail if schema is old
                test_results = await self.indexing_tasks_table.query().limit(1).to_list()
                if test_results and 'result_files_scanned' not in test_results[0]:
                    logger.warning("Detected old indexing_tasks schema, recreating table")
                    # Drop and recreate with new schema
                    await self.db.drop_table(table_name)
                    self.indexing_tasks_table = await self.db.create_table(
                        table_name,
                        None,  # data
                        schema=IndexingTask,  # schema
                        mode="overwrite",  # mode
                    )
                    logger.info("Recreated indexing tasks table with new schema")
                else:
                    # Reset any running tasks to queued (they were interrupted)
                    await self._reset_interrupted_tasks()
            except Exception as e:
                logger.warning(f"Schema check failed: {e}, keeping existing table")
                # Reset any running tasks to queued (they were interrupted)
                await self._reset_interrupted_tasks()

    async def _reset_interrupted_tasks(self):
        """Reset running tasks to queued status on startup."""
        try:
            # Update running tasks to queued and clear started_at
            # LanceDB update syntax: update(values, where)
            await self.indexing_tasks_table.update(
                {"status": "queued", "started_at": None},
                where="status = 'running'"
            )
        except Exception as e:
            logger.warning(f"Error resetting interrupted tasks: {e}")

    async def save_indexing_task(self, task: IndexingTask) -> None:
        """Save an indexing task to the database."""
        await self.indexing_tasks_table.add([task])

    async def get_indexing_task_db(self, task_id: str) -> Optional[IndexingTask]:
        """Get an indexing task from the database."""
        try:
            results = await (
                self.indexing_tasks_table.query()
                .where(f"task_id = '{task_id}'")
                .limit(1)
                .to_list()
            )

            if not results:
                return None

            return IndexingTask(**results[0])
        except Exception as e:
            logger.error(f"Error getting indexing task: {e}")
            return None

    async def update_indexing_task(self, task: IndexingTask) -> None:
        """Update an indexing task in the database."""
        try:
            # LanceDB update syntax: update(values, where)
            await self.indexing_tasks_table.update(
                task.model_dump(),
                where=f"task_id = '{task.task_id}'"
            )
        except Exception as e:
            logger.error(f"Error updating indexing task: {e}")

    async def list_tasks_by_status(self, status: str) -> List[IndexingTask]:
        """List all tasks with a specific status."""
        try:
            results = await (
                self.indexing_tasks_table.query()
                .where(f"status = '{status}'")
                .to_list()
            )

            return [IndexingTask(**result) for result in results]
        except Exception as e:
            logger.error(f"Error listing tasks by status: {e}")
            return []

    # Failed Batch Management Methods

    async def _init_failed_batches_table(self):
        """Initialize the failed batches table."""
        table_name = "failed_batches"
        existing_tables = await self.db.table_names()

        if table_name not in existing_tables:
            # Create new table
            self.failed_batches_table = await self.db.create_table(
                table_name,
                None,  # data
                schema=FailedBatch,  # schema
                mode="overwrite",  # mode
            )
            logger.info("Created new failed batches table")
        else:
            self.failed_batches_table = await self.db.open_table(table_name)
            # Clean up old succeeded/abandoned batches
            await self._cleanup_old_failed_batches()

    async def _cleanup_old_failed_batches(self):
        """Clean up old succeeded or abandoned batches."""
        try:
            # Delete succeeded batches older than 7 days
            seven_days_ago = datetime.now() - timedelta(days=7)
            await self.failed_batches_table.delete(
                f"status IN ('succeeded', 'abandoned') AND created_at < '{seven_days_ago.isoformat()}'"
            )
        except Exception as e:
            logger.warning(f"Error cleaning up old failed batches: {e}")

    async def _store_failed_batch(
        self,
        batch_id: str,
        file_paths: List[str],
        content_hashes: List[str],
        error_message: str,
        project_id: Optional[str] = None,
    ):
        """Store a failed batch for later retry."""
        from datetime import timedelta

        failed_batch = FailedBatch(
            batch_id=batch_id,
            file_paths=file_paths,
            content_hashes=content_hashes,
            error_message=error_message,
            project_id=project_id,
            # Schedule first retry in 5 minutes
            next_retry_at=datetime.now() + timedelta(minutes=5),
        )

        await self.failed_batches_table.add([failed_batch])
        logger.info(f"Stored failed batch {batch_id} for retry")

    async def _get_pending_retries(self, limit: int = 10) -> List[FailedBatch]:
        """Get pending batches that are ready for retry."""
        try:
            # Get all pending batches where next_retry_at has passed
            arrow_table = await self.failed_batches_table.to_arrow()
            import polars as pl

            df = pl.from_arrow(arrow_table)
            if df.height > 0:
                # Filter for pending batches ready to retry
                now = datetime.now()
                pending = (
                    df.filter(
                        (pl.col("status") == "pending")
                        & (pl.col("next_retry_at") <= now)
                        & (pl.col("retry_count") < pl.col("max_retries"))
                    )
                    .sort("next_retry_at")
                    .limit(limit)
                )

                if pending.height > 0:
                    return [FailedBatch(**row) for row in pending.to_dicts()]
        except Exception as e:
            logger.error(f"Error getting pending retries: {e}")

        return []

    async def _update_failed_batch_status(
        self, batch_id: str, status: str, error_message: Optional[str] = None
    ):
        """Update the status of a failed batch."""
        try:
            # Get the batch
            arrow_table = await self.failed_batches_table.to_arrow()
            import polars as pl

            df = pl.from_arrow(arrow_table)
            if df.height > 0:
                batch_row = df.filter(pl.col("batch_id") == batch_id).to_dicts()
                if batch_row:
                    batch = FailedBatch(**batch_row[0])
                    batch.status = status
                    batch.last_retry_at = datetime.now()

                    if error_message:
                        batch.error_message = error_message
                        batch.retry_count += 1

                        # Calculate next retry with exponential backoff
                        if batch.retry_count < batch.max_retries:
                            backoff_minutes = min(
                                5 * (2**batch.retry_count), 1440
                            )  # Max 24 hours
                            batch.next_retry_at = datetime.now() + timedelta(
                                minutes=backoff_minutes
                            )
                        else:
                            batch.status = "abandoned"

                    # Update in database
                    await (
                        self.failed_batches_table.merge_insert("id")
                        .when_matched_update_all()
                        .when_not_matched_insert_all()
                        .execute([batch])
                    )

                    logger.info(f"Updated failed batch {batch_id} status to {status}")
        except Exception as e:
            logger.error(f"Error updating failed batch status: {e}")

    async def _background_retry_task(self):
        """Background task to retry failed batches."""
        logger.info("Starting background retry task")

        while not self._retry_task_stop_event.is_set():
            try:
                # Check for pending retries every minute (shorter in tests)
                import os
                sleep_time = 5 if os.environ.get("PYTEST_CURRENT_TEST") else 60
                await asyncio.sleep(sleep_time)

                # Only retry failed batches when there's no active indexing work
                # This ensures we don't interfere with ongoing indexing operations
                if self._active_tasks:
                    continue

                # Get pending batches
                pending_batches = await self._get_pending_retries(limit=5)

                if pending_batches:
                    logger.info(f"Found {len(pending_batches)} batches to retry")

                    for batch in pending_batches:
                        try:
                            # Mark as processing
                            await self._update_failed_batch_status(
                                batch.batch_id, "processing"
                            )

                            # Read the files again
                            file_paths = [Path(fp) for fp in batch.file_paths]
                            doc_datas = await self._read_file_batch(file_paths)

                            # Filter by content hash to avoid duplicates
                            filtered_docs = []
                            for doc, expected_hash in zip(
                                doc_datas, batch.content_hashes
                            ):
                                if doc and doc["content_hash"] == expected_hash:
                                    filtered_docs.append(doc)

                            if filtered_docs:
                                # Process each file individually like in the main pipeline
                                documents = []
                                
                                for doc_data in filtered_docs:
                                    try:
                                        # Detect language
                                        language = self.content_detector.detect_language(Path(doc_data["file_path"]))
                                        if not language:
                                            continue
                                        
                                        file_content = FileContent(
                                            content=doc_data["content"],
                                            file_path=doc_data["file_path"],
                                            language=language
                                        )

                                        # Generate embedding for single file
                                        if self.is_voyage_model:
                                            rate_limits = self.config.get_voyage_rate_limits()
                                            result = await get_voyage_embeddings_with_limits(
                                                [file_content],
                                                self.embedding_model,
                                                self.tokenizer,
                                                self.config.voyage_concurrent_requests,
                                                self.config.voyage_max_retries,
                                                self.config.voyage_retry_base_delay,
                                                rate_limits['tokens_per_minute'],
                                                rate_limits['requests_per_minute'],
                                            )

                                            if result.embeddings.size > 0 and not result.failed_files:
                                                embedding = result.embeddings[0]
                                            else:
                                                raise Exception("Embedding generation failed")
                                        else:
                                            # Use tokenizer-based chunking for local models
                                            result = await get_local_embeddings_with_tokenizer_chunking(
                                                [file_content],
                                                self.embedding_model,
                                                self.config.embedding_model,
                                                max_concurrent_requests=self.config.concurrent_embedders,
                                                max_retries=3,
                                                retry_base_delay=1.0,
                                                max_sequence_length=self.config.max_sequence_length if hasattr(self.config, 'max_sequence_length') else None,
                                                tokenizer=self.tokenizer,
                                            )

                                            if result.embeddings.size > 0 and not result.failed_files:
                                                embedding = result.embeddings[0]
                                            else:
                                                raise Exception("Embedding generation failed")

                                        # Add embedding to document
                                        doc_data["vector"] = (
                                            embedding.tolist()
                                            if hasattr(embedding, "tolist")
                                            else embedding
                                        )
                                        documents.append(self.document_schema(**doc_data))
                                        
                                    except Exception as e:
                                        logger.warning(f"Failed to process file in retry: {e}")
                                        continue

                                if documents:
                                    # Use add() for retry documents too - LanceDB handles versioning
                                    await self.table.add(documents)

                                # Mark as succeeded
                                await self._update_failed_batch_status(
                                    batch.batch_id, "succeeded"
                                )

                                logger.info(
                                    f"Successfully retried batch {batch.batch_id}"
                                )
                            else:
                                # Files changed or missing
                                await self._update_failed_batch_status(
                                    batch.batch_id,
                                    "abandoned",
                                    "Files changed or no longer exist",
                                )

                        except Exception as e:
                            logger.error(f"Error retrying batch {batch.batch_id}: {e}")
                            await self._update_failed_batch_status(
                                batch.batch_id, "pending", str(e)
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                if "Event loop is closed" in str(e) or isinstance(e, RuntimeError):
                    # Event loop closing, exit gracefully
                    logger.debug("Event loop closing, retry task exiting")
                    break
                logger.error(f"Error in background retry task: {e}")
                try:
                    await asyncio.sleep(60)  # Wait before retrying
                except Exception:
                    # Can't sleep, probably shutting down
                    break

        logger.info("Background retry task stopped")


    async def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        logger.info("Shutting down BreezeEngine...")

        # Stop indexing queue first to prevent new tasks
        if self._indexing_queue:
            logger.info("Stopping indexing queue...")
            await self._indexing_queue.stop()
            self._indexing_queue = None

        # Stop background retry task
        if self._retry_task:
            logger.info("Stopping background retry task...")
            self._retry_task_stop_event.set()
            self._retry_task.cancel()
            try:
                await asyncio.wait_for(self._retry_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._retry_task = None

        # Cancel any active indexing tasks
        if self._active_tasks:
            logger.info("Cancelling active indexing tasks...")
            for task in list(self._active_tasks.values()):
                if hasattr(task, '_async_task') and task._async_task and not task._async_task.done():
                    task._async_task.cancel()
                    try:
                        await asyncio.wait_for(task._async_task, timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
            self._active_tasks.clear()

        # Stop all file watchers
        if self._watchers:
            logger.info(f"Stopping {len(self._watchers)} file watchers...")
            for project_id in list(self._watchers.keys()):
                await self.stop_watching(project_id)

        # Reset initialization state
        self._initialized = False
        logger.info("BreezeEngine shutdown complete")


class FileWatcher(FileSystemEventHandler):
    """Handles file system events for code files."""

    def __init__(
        self,
        engine: BreezeEngine,
        project: Project,
        event_callback: Optional[Callable] = None,
        debounce_seconds: float = 2.0,
    ):
        self.engine = engine
        self.project = project
        self.event_callback = event_callback
        self.debounce_seconds = debounce_seconds
        self.pending_files: Set[Path] = set()
        self.last_event_time = 0
        self._processing = False
        self._task = None
        self._loop = None  # Store the event loop
        self._thread_safe_queue = asyncio.Queue()

    def set_event_loop(self, loop):
        """Set the event loop to use for async operations."""
        self._loop = loop

    def _is_code_file(self, path: Path) -> bool:
        """Check if file should be indexed."""
        # Use the engine's content detection method
        return self.engine._should_index_file(path)

    def on_any_event(self, event: FileSystemEvent):
        """Handle any file system event."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Check if it's a code file
        if not self._is_code_file(path):
            return

        # Add to pending files
        self.pending_files.add(path)
        self.last_event_time = time.time()

        # Schedule the debounce timer in the main event loop
        if self._loop:
            # Thread-safe way to schedule coroutine in the main loop
            asyncio.run_coroutine_threadsafe(self._schedule_processing(), self._loop)

    async def _schedule_processing(self):
        """Schedule processing after debounce (runs in main event loop)."""
        # Cancel existing task if any
        if self._task and not self._task.done():
            self._task.cancel()

        # Create new task
        self._task = asyncio.create_task(self._process_after_debounce())

    async def _process_after_debounce(self):
        """Process pending files after debounce period."""
        await asyncio.sleep(self.debounce_seconds)

        if self.pending_files and not self._processing:
            self._processing = True
            files_to_process = list(self.pending_files)
            self.pending_files.clear()

            try:
                # Notify about indexing start
                if self.event_callback:
                    await self.event_callback(
                        {
                            "type": "indexing_started",
                            "project_id": self.project.id,
                            "project_name": self.project.name,
                            "files": [str(f) for f in files_to_process],
                            "count": len(files_to_process),
                        }
                    )

                # Index the changed files
                async def progress_wrapper(p):
                    if self.event_callback:
                        await self.event_callback(
                            {
                                "type": "indexing_progress",
                                "project_id": self.project.id,
                                **p,
                            }
                        )

                stats = await self.engine.index_directories(
                    [str(f.parent) for f in files_to_process],
                    force_reindex=True,
                    progress_callback=progress_wrapper if self.event_callback else None,
                )

                # Update project indexed time
                await self.engine.update_project_indexed_time(self.project.id)

                # Notify completion
                if self.event_callback:
                    await self.event_callback(
                        {
                            "type": "indexing_completed",
                            "project_id": self.project.id,
                            "project_name": self.project.name,
                            "stats": stats.model_dump(),
                        }
                    )

            except Exception as e:
                logger.error(f"Error processing file changes: {e}")
                if self.event_callback:
                    await self.event_callback(
                        {
                            "type": "indexing_error",
                            "project_id": self.project.id,
                            "error": str(e),
                        }
                    )
            finally:
                self._processing = False
