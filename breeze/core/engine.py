"""Core engine for Breeze code indexing and search."""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
import time
import os

try:
    from identify import identify
except ImportError:
    raise ImportError("Please install identify: pip install identify")

try:
    import magic

    # Test if libmagic is available
    magic.Magic()
except ImportError:
    raise ImportError("Please install python-magic: pip install python-magic")
except Exception as e:
    raise RuntimeError(
        "python-magic requires libmagic to be installed.\n"
        "  macOS: brew install libmagic\n"
        "  Ubuntu/Debian: sudo apt-get install libmagic1\n"
        "  RHEL/CentOS: sudo yum install file-devel\n"
        f"Actual error: {e}"
    )

import aiofiles
import aiofiles.os
import lancedb
from lancedb.embeddings import get_registry
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from breeze.core.config import BreezeConfig
from breeze.core.models import (
    IndexStats,
    SearchResult,
    Project,
    IndexingTask,
    FailedBatch,
    get_code_document_schema,
)
from breeze.core.embeddings import get_voyage_embeddings_with_limits

# Use standard Python logging to avoid duplicate handlers
logger = logging.getLogger(__name__)

# Disable verbose logging for LanceDB unless explicitly requested
if os.environ.get("BREEZE_DEBUG_LANCE"):
    os.environ["RUST_LOG"] = "debug"
    os.environ["LANCE_LOG"] = "debug"
    logging.getLogger("lancedb").setLevel(logging.DEBUG)


class BreezeEngine:
    """Main engine for code indexing and search operations."""

    def __init__(self, config: Optional[BreezeConfig] = None):
        self.config = config or BreezeConfig()
        self.config.ensure_directories()

        self.db: Optional[lancedb.LanceDBConnection] = None
        self.table: Optional[lancedb.Table] = None
        self.projects_table: Optional[lancedb.Table] = None
        self.failed_batches_table: Optional[lancedb.Table] = None
        self.embedding_model = None
        self.document_schema = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.tokenizer = None  # For Voyage AI token counting
        self.is_voyage_model = False  # Flag for special Voyage handling

        # File watching and task tracking
        self._watchers: Dict[str, "FileWatcher"] = {}
        self._observers: Dict[str, "Observer"] = {}  # type: ignore
        self._active_tasks: Dict[str, IndexingTask] = {}

        # Background retry task
        self._retry_task = None
        self._retry_task_stop_event = asyncio.Event()

    async def initialize(self):
        """Initialize the database and embedding model."""
        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing BreezeEngine...")

            # Initialize LanceDB async connection
            self.db = await lancedb.connect_async(self.config.get_db_path())

            # Initialize embedding model using LanceDB's registry
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

                # Initialize tokenizer for Voyage
                try:
                    from tokenizers import Tokenizer

                    # Load the Voyage-specific tokenizer from HuggingFace
                    tokenizer_name = (
                        "voyageai/voyage-code-3"
                        if self.config.embedding_model == "voyage-code-3"
                        else "voyageai/voyage-code-2"
                    )
                    self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
                    logger.info(
                        f"Using HuggingFace tokenizer for accurate token counting with Voyage: {tokenizer_name}"
                    )
                except ImportError:
                    logger.warning(
                        "tokenizers not available, token limits may be exceeded"
                    )
                    logger.warning("Install with: pip install tokenizers")
                except Exception as e:
                    logger.warning(f"Failed to load Voyage tokenizer: {e}")
                    logger.warning("Token limits may be exceeded")

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
                self.embedding_model = registry.get("sentence-transformers").create(
                    name=self.config.embedding_model,
                    device=self.config.embedding_device,
                    normalize=True,
                    trust_remote_code=self.config.trust_remote_code,
                    show_progress_bar=False,
                )

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

            # Start background retry task
            self._retry_task = asyncio.create_task(self._background_retry_task())

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

        # Get existing documents for update/skip logic
        existing_docs = await self._get_existing_docs() if not force_reindex else {}

        # Phase 1: Fast directory walk
        logger.info("Phase 1: Discovering files...")
        files_to_index = []
        for directory in directories:
            path = Path(directory).resolve()
            files_to_index.extend(self._walk_directory_fast(path))
        logger.info(f"Found {len(files_to_index)} files")

        # Phase 2: Process files with three-stage pipeline
        logger.info("Phase 2: Processing files...")

        # Determine batch size
        batch_size = 100
        if self.is_voyage_model:
            batch_size = 20  # Smaller batches for Voyage AI token limits

        # Create batches
        batches = []
        for i in range(0, len(files_to_index), batch_size):
            batches.append(files_to_index[i : i + batch_size])

        # Configure concurrency
        concurrent_readers = num_workers or self.config.concurrent_readers or 20
        concurrent_embedders = self.config.concurrent_embedders or 10
        concurrent_writers = self.config.concurrent_writers or 10

        # Process batches with three-stage pipeline
        pipeline_stats = await self._run_indexing_pipeline(
            batches,
            existing_docs,
            force_reindex,
            concurrent_readers,
            concurrent_embedders,
            concurrent_writers,
            progress_callback,
        )

        # Update stats from pipeline results
        stats.files_scanned = len(files_to_index)
        stats.files_indexed = pipeline_stats["indexed"]
        stats.files_updated = pipeline_stats["updated"]
        stats.errors = pipeline_stats["errors"]
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

        return stats

    async def search(
        self,
        query: str,
        limit: Optional[int] = None,
        min_relevance: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search for code files matching the query."""
        await self.initialize()

        limit = limit or self.config.default_limit
        min_relevance = min_relevance or self.config.min_relevance

        # Perform vector search using LanceDB's built-in search
        # LanceDB will automatically generate embeddings for the query
        search_query = await self.table.search(query, vector_column_name="vector")
        search_query = search_query.limit(limit)
        results = await search_query.to_arrow()
        # Convert to list of dicts for processing
        import polars as pl

        df = pl.from_arrow(results)
        results = df.to_dicts() if df.height > 0 else []

        search_results = []
        for result in results:
            # Calculate relevance score (cosine similarity)
            # LanceDB returns L2 distance, convert to similarity
            distance = result.get("_distance", 0)
            relevance_score = 1 / (1 + distance)  # Convert distance to similarity

            if relevance_score < min_relevance:
                continue

            # Create snippet
            content = result.get("content", "")
            snippet = self._create_snippet(content, query)

            search_result = SearchResult(
                id=result.get("id", ""),
                file_path=result.get("file_path", ""),
                file_type=result.get("file_type", ""),
                relevance_score=relevance_score,
                snippet=snippet,
                last_modified=result.get("last_modified", ""),
            )
            search_results.append(search_result)

        # Sort by relevance score
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return search_results

    async def get_stats(self) -> Dict:
        """Get current index statistics."""
        if not self._initialized or self.table is None:
            return {"total_documents": 0, "initialized": False}

        table_len = await self.table.count_rows()

        # Get failed batch stats
        failed_stats = await self._get_failed_batch_stats()

        return {
            "total_documents": table_len,
            "initialized": True,
            "model": self.config.embedding_model,
            "database_path": self.config.get_db_path(),
            "failed_batches": failed_stats,
        }

    async def _get_failed_batch_stats(self) -> Dict:
        """Get statistics about failed batches."""
        try:
            if not self.failed_batches_table:
                return {}

            arrow_table = await self.failed_batches_table.to_arrow()
            import polars as pl

            df = pl.from_arrow(arrow_table)
            if df.height == 0:
                return {"total": 0}

            # Count by status
            status_counts = df.group_by("status").agg(pl.count()).to_dicts()
            stats = {row["status"]: row["count"] for row in status_counts}
            stats["total"] = df.height

            # Get oldest pending batch
            pending = df.filter(pl.col("status") == "pending")
            if pending.height > 0:
                oldest = pending.select("next_retry_at").min().item()
                if oldest:
                    stats["next_retry_at"] = oldest.isoformat()

            return stats
        except Exception as e:
            logger.error(f"Error getting failed batch stats: {e}")
            return {}

    async def _get_existing_docs(self) -> Dict[str, str]:
        """Get existing documents from the table."""
        existing_docs = {}
        try:
            # Get all data and then filter columns
            arrow_table = await self.table.to_arrow()

            # Convert Arrow table to Polars and select columns
            import polars as pl

            df = pl.from_arrow(arrow_table)

            if df.height > 0:
                # Select only the columns we need
                df_subset = df.select(["id", "content_hash"])
                for row in df_subset.iter_rows(named=True):
                    existing_docs[row["id"]] = row["content_hash"]
        except Exception as e:
            logger.warning(f"Error reading existing documents: {e}")
        return existing_docs

    def _create_snippet(self, content: str, query: str, context_lines: int = 5) -> str:
        """Create a relevant snippet from the content."""
        lines = content.split("\n")
        query_lower = query.lower()

        # Find the most relevant lines
        best_score = 0
        best_idx = 0

        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Simple scoring based on query terms
            score = sum(term in line_lower for term in query_lower.split())
            if score > best_score:
                best_score = score
                best_idx = i

        # Extract context around the best match
        start_idx = max(0, best_idx - context_lines)
        end_idx = min(len(lines), best_idx + context_lines + 1)

        snippet_lines = lines[start_idx:end_idx]
        snippet = "\n".join(snippet_lines)

        # Truncate if too long
        max_length = 1000
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."

        return snippet

    def _should_index_file(self, file_path: Path) -> bool:
        """Check if file should be indexed."""
        # Check exclude patterns first
        path_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            if pattern in path_str:
                return False

        try:
            # First try identify for known file types
            tags = identify.tags_from_path(str(file_path))

            # Skip if explicitly marked as binary
            if "binary" in tags:
                return False

            # If it has 'text' tag or any programming language tag, index it
            if "text" in tags or any(
                tag for tag in tags if tag not in {"binary", "executable"}
            ):
                return True

            # For files without clear tags, use python-magic
            try:
                mime = magic.from_file(str(file_path), mime=True)
                # Index text files, source code, config files, etc.
                if mime.startswith("text/") or mime in {
                    "application/json",
                    "application/xml",
                    "application/x-yaml",
                    "application/javascript",
                    "application/x-python-code",
                    "application/x-ruby",
                    "application/x-sh",
                    "application/x-shellscript",
                }:
                    return True
            except (OSError, IOError):
                pass

        except Exception:
            pass

        return False

    def _walk_directory_fast(self, directory: Path) -> List[Path]:
        """Fast synchronous directory walk with filtering."""
        files_to_index = []
        visited_dirs = set()

        for root, dirs, files in os.walk(directory, followlinks=False):
            root_path = Path(root)

            # Skip if we've seen this real path before
            try:
                real_path = root_path.resolve()
                if real_path in visited_dirs:
                    continue
                visited_dirs.add(real_path)
            except Exception:
                continue

            # Filter out excluded directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and not any(pattern in d for pattern in self.config.exclude_patterns)
            ]

            # Check files
            for file in files:
                if file.startswith("."):
                    continue

                file_path = root_path / file
                if self._should_index_file(file_path):
                    files_to_index.append(file_path)

        return files_to_index

    async def _run_indexing_pipeline(
        self,
        batches: List[List[Path]],
        existing_docs: Dict[str, str],
        force_reindex: bool,
        concurrent_readers: int,
        concurrent_embedders: int,
        concurrent_writers: int,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, int]:
        """Run three-stage pipeline for indexing files."""
        # Create queues for the pipeline
        embed_queue = asyncio.Queue(maxsize=concurrent_readers)
        write_queue = asyncio.Queue(maxsize=concurrent_embedders * 2)

        # Semaphores for controlling concurrency
        read_sem = asyncio.Semaphore(concurrent_readers)
        embed_sem = asyncio.Semaphore(concurrent_embedders)
        write_sem = asyncio.Semaphore(concurrent_writers)

        # Track completion
        read_complete = asyncio.Event()
        embed_complete = asyncio.Event()

        # Track results
        total_indexed = 0
        total_updated = 0
        total_errors = 0
        total_processed = 0

        async def reader_task(batch_idx: int, batch: List[Path]):
            """Read files and queue for embedding."""
            async with read_sem:
                doc_datas = await self._read_file_batch(
                    batch, existing_docs, force_reindex
                )
                await embed_queue.put((batch_idx, doc_datas))

        async def embedder_task():
            """Pull from read queue, generate embeddings, and queue for writing."""
            while True:
                try:
                    # Wait for items with a timeout
                    batch_idx, doc_datas = await asyncio.wait_for(
                        embed_queue.get(), timeout=1.0
                    )

                    async with embed_sem:
                        if doc_datas:
                            try:
                                # Extract contents for embedding
                                contents = [doc["content"] for doc in doc_datas]

                                # Generate embeddings based on model type
                                if self.is_voyage_model:
                                    # Special handling for Voyage with token limits
                                    rate_limits = self.config.get_voyage_rate_limits()
                                    result = await get_voyage_embeddings_with_limits(
                                        contents,
                                        self.embedding_model,
                                        self.tokenizer,
                                        self.config.voyage_concurrent_requests,
                                        self.config.voyage_max_retries,
                                        self.config.voyage_retry_base_delay,
                                        rate_limits['tokens_per_minute'],
                                        rate_limits['requests_per_minute'],
                                    )
                                    
                                    embeddings = result['embeddings']
                                    
                                    # Handle any failed batches
                                    if result['failed_batches']:
                                        # Find which documents failed
                                        failed_indices = []
                                        for failed_batch_idx in result['failed_batches']:
                                            batch = result['safe_batches'][failed_batch_idx]
                                            # Find original indices of texts in the failed batch
                                            start_idx = sum(len(result['safe_batches'][i]) for i in range(failed_batch_idx))
                                            for i in range(len(batch)):
                                                failed_indices.append(start_idx + i)
                                        
                                        # Separate successful and failed documents
                                        successful_doc_datas = [doc for i, doc in enumerate(doc_datas) if i not in failed_indices]
                                        failed_doc_datas = [doc for i, doc in enumerate(doc_datas) if i in failed_indices]
                                        
                                        # Store failed documents for retry
                                        if failed_doc_datas:
                                            batch_id = f"batch_{batch_idx}_voyage_retry_{int(time.time())}"
                                            file_paths = [doc["file_path"] for doc in failed_doc_datas]
                                            content_hashes = [doc["content_hash"] for doc in failed_doc_datas]
                                            
                                            await self._store_failed_batch(
                                                batch_id=batch_id,
                                                file_paths=file_paths,
                                                content_hashes=content_hashes,
                                                error_message="Rate limit exceeded - will retry",
                                            )
                                            
                                            logger.info(f"Stored {len(failed_doc_datas)} documents for later retry due to rate limits")
                                        
                                        # Continue with successful documents only
                                        doc_datas = successful_doc_datas
                                else:
                                    # Standard embedding generation
                                    embeddings = self.embedding_model.compute_source_embeddings(contents)

                                # Create document objects with embeddings
                                documents = []
                                if len(embeddings) > 0:
                                    for doc_data, embedding in zip(doc_datas, embeddings):
                                        doc_data["vector"] = (
                                            embedding.tolist()
                                            if hasattr(embedding, "tolist")
                                            else embedding
                                        )
                                        documents.append(self.document_schema(**doc_data))

                                    await write_queue.put((batch_idx, documents, doc_datas))
                                else:
                                    # No successful embeddings - all failed
                                    logger.warning(f"Batch {batch_idx}: All embeddings failed, will retry later")
                            except Exception as e:
                                logger.error(
                                    f"Failed to generate embeddings for batch {batch_idx}: {e}"
                                )

                                # Store failed batch for later retry
                                batch_id = f"batch_{batch_idx}_{int(time.time())}"
                                file_paths = [doc["file_path"] for doc in doc_datas]
                                content_hashes = [
                                    doc["content_hash"] for doc in doc_datas
                                ]

                                await self._store_failed_batch(
                                    batch_id=batch_id,
                                    file_paths=file_paths,
                                    content_hashes=content_hashes,
                                    error_message=str(e),
                                    project_id=None,  # Could be passed through context if needed
                                )

                                # Still put error marker for immediate stats tracking
                                await write_queue.put((batch_idx, None, doc_datas))
                        else:
                            await write_queue.put((batch_idx, [], []))

                except asyncio.TimeoutError:
                    # Check if reading is complete and queue is empty
                    if read_complete.is_set() and embed_queue.empty():
                        break

        async def writer_task():
            """Pull from embed queue and write to database."""
            nonlocal total_indexed, total_updated, total_errors, total_processed

            while True:
                try:
                    # Wait for items with a timeout
                    batch_idx, documents, doc_datas = await asyncio.wait_for(
                        write_queue.get(), timeout=1.0
                    )

                    async with write_sem:
                        if documents is None:
                            # This batch failed to generate embeddings
                            total_errors += len(doc_datas)
                            logger.error(
                                f"Skipping batch {batch_idx} due to embedding generation failure"
                            )
                        elif documents:
                            try:
                                await (
                                    self.table.merge_insert("id")
                                    .when_matched_update_all()
                                    .when_not_matched_insert_all()
                                    .execute(documents)
                                )

                                # Update counts
                                for doc in documents:
                                    if doc.id in existing_docs:
                                        total_updated += 1
                                    else:
                                        total_indexed += 1

                            except Exception as e:
                                logger.error(f"Error writing batch {batch_idx}: {e}")
                                total_errors += len(documents)

                        total_processed += len(doc_datas)

                        # Progress callback
                        if progress_callback:
                            await progress_callback(
                                {
                                    "files_scanned": total_processed,
                                    "files_indexed": total_indexed,
                                    "files_updated": total_updated,
                                    "files_skipped": 0,  # Will be calculated later
                                    "errors": total_errors,
                                }
                            )

                except asyncio.TimeoutError:
                    # Check if embedding is complete and queue is empty
                    if embed_complete.is_set() and write_queue.empty():
                        break

        # Start all pipeline tasks
        writer_tasks = [
            asyncio.create_task(writer_task()) for _ in range(concurrent_writers)
        ]

        embedder_tasks = [
            asyncio.create_task(embedder_task()) for _ in range(concurrent_embedders)
        ]

        reader_tasks = [reader_task(idx, batch) for idx, batch in enumerate(batches)]

        # Wait for pipeline stages to complete in order
        await asyncio.gather(*reader_tasks)
        read_complete.set()

        await asyncio.gather(*embedder_tasks)
        embed_complete.set()

        await asyncio.gather(*writer_tasks)

        return {
            "indexed": total_indexed,
            "updated": total_updated,
            "errors": total_errors,
        }

    async def _read_file_batch(
        self, file_paths: List[Path], existing_docs: Dict[str, str], force_reindex: bool
    ) -> List[Dict]:
        """Read a batch of files concurrently and return document data."""

        async def read_single_file(file_path: Path) -> Dict:
            try:
                stat = await aiofiles.os.stat(file_path)

                async with aiofiles.open(
                    file_path, "r", encoding="utf-8", errors="replace"
                ) as f:
                    content = await f.read()

                # Skip empty files
                if not content.strip():
                    return None

                # Generate document ID and content hash
                doc_id = f"file:{file_path}"
                content_hash = hashlib.md5(content.encode()).hexdigest()

                # Check if update needed
                if not force_reindex and doc_id in existing_docs:
                    if existing_docs[doc_id] == content_hash:
                        return None  # Skip unchanged files

                # Return document data (without vector)
                return {
                    "id": doc_id,
                    "file_path": str(file_path),
                    "content": content,
                    "file_type": file_path.suffix[1:] if file_path.suffix else "txt",
                    "file_size": stat.st_size,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime),
                    "indexed_at": datetime.now(),
                    "content_hash": content_hash,
                }

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return None

        # Process files concurrently
        tasks = [read_single_file(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        return [doc for doc in results if doc is not None]

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
            file_extensions=file_extensions
            or [ext.lstrip(".") for ext in self.config.code_extensions],
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
            watcher = FileWatcher(self, project, event_callback)
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
            await self.update_project_indexed_time(project_id)

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
                await self.update_project_indexed_time(project_id)

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

        # Start the task
        asyncio.create_task(run_indexing())

        return task

    async def get_indexing_task(self, task_id: str) -> Optional[IndexingTask]:
        """Get status of an indexing task."""
        return self._active_tasks.get(task_id)

    async def list_indexing_tasks(self) -> List[IndexingTask]:
        """List all indexing tasks."""
        return list(self._active_tasks.values())

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
                # Check for pending retries every minute
                await asyncio.sleep(60)

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
                            doc_datas = await self._read_file_batch(
                                file_paths,
                                {},  # existing_docs
                                True,  # force_reindex
                            )

                            # Filter by content hash to avoid duplicates
                            filtered_docs = []
                            for doc, expected_hash in zip(
                                doc_datas, batch.content_hashes
                            ):
                                if doc and doc["content_hash"] == expected_hash:
                                    filtered_docs.append(doc)

                            if filtered_docs:
                                # Try to generate embeddings
                                contents = [doc["content"] for doc in filtered_docs]

                                if self.is_voyage_model:
                                    rate_limits = self.config.get_voyage_rate_limits()
                                    result = await get_voyage_embeddings_with_limits(
                                        contents,
                                        self.embedding_model,
                                        self.tokenizer,
                                        self.config.voyage_concurrent_requests,
                                        self.config.voyage_max_retries,
                                        self.config.voyage_retry_base_delay,
                                        rate_limits['tokens_per_minute'],
                                        rate_limits['requests_per_minute'],
                                    )
                                    
                                    embeddings = result['embeddings']
                                    
                                    # If some batches failed, mark this batch for retry again
                                    if result['failed_batches']:
                                        raise Exception("Some embeddings failed - retry later")
                                else:
                                    embeddings = self.embedding_model.compute_source_embeddings(
                                        contents
                                    )

                                # Create documents and write to database
                                documents = []
                                for doc_data, embedding in zip(
                                    filtered_docs, embeddings
                                ):
                                    doc_data["vector"] = (
                                        embedding.tolist()
                                        if hasattr(embedding, "tolist")
                                        else embedding
                                    )
                                    documents.append(self.document_schema(**doc_data))

                                await (
                                    self.table.merge_insert("id")
                                    .when_matched_update_all()
                                    .when_not_matched_insert_all()
                                    .execute(documents)
                                )

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
                logger.error(f"Error in background retry task: {e}")
                await asyncio.sleep(60)  # Wait before retrying

        logger.info("Background retry task stopped")

    async def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        # Stop background retry task
        if self._retry_task:
            self._retry_task_stop_event.set()
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        # Stop all file watchers
        for project_id in list(self._watchers.keys()):
            await self.stop_watching(project_id)


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

    def _is_code_file(self, path: Path) -> bool:
        """Check if file should be indexed."""
        # Check extension
        if path.suffix.lower() not in [
            f".{ext}" for ext in self.project.file_extensions
        ]:
            return False

        # Check exclude patterns
        path_str = str(path)
        for pattern in self.project.exclude_patterns:
            if pattern in path_str:
                return False

        return True

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

        # Start or restart the debounce timer
        if self._task:
            self._task.cancel()

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
