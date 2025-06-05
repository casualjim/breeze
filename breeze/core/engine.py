"""Core engine for Breeze code indexing and search."""

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Callable, Dict, List, Optional

import lancedb
from sentence_transformers import SentenceTransformer

from breeze.core.config import BreezeConfig
from breeze.core.models import CodeDocument, IndexStats, SearchResult

logger = logging.getLogger(__name__)


class BreezeEngine:
    """Main engine for code indexing and search operations."""

    def __init__(self, config: Optional[BreezeConfig] = None):
        self.config = config or BreezeConfig()
        self.config.ensure_directories()

        self.db: Optional[lancedb.LanceDBConnection] = None
        self.table: Optional[lancedb.Table] = None
        self.model: Optional[SentenceTransformer] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._embedding_semaphore = asyncio.Semaphore(10)  # Limit concurrent embeddings

    async def initialize(self):
        """Initialize the database and embedding model."""
        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing BreezeEngine...")

            # Initialize LanceDB connection
            loop = asyncio.get_event_loop()
            self.db = await loop.run_in_executor(
                None, lancedb.connect, self.config.get_db_path()
            )

            # Initialize embedding model with trust_remote_code
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    self.config.embedding_model,
                    trust_remote_code=self.config.trust_remote_code,
                ),
            )

            # Create or open table
            table_name = "code_documents"
            existing_tables = await loop.run_in_executor(None, self.db.table_names)

            if table_name in existing_tables:
                self.table = await loop.run_in_executor(
                    None, self.db.open_table, table_name
                )
                table_len = await loop.run_in_executor(None, len, self.table)
                logger.info(f"Opened existing table with {table_len} documents")
            else:
                # Create new table with Pydantic schema
                self.table = await loop.run_in_executor(
                    None,
                    self.db.create_table,
                    table_name,
                    None,  # data
                    CodeDocument,  # schema
                    "overwrite",  # mode
                )
                logger.info("Created new code documents table")

            self._initialized = True
            logger.info("BreezeEngine initialization complete")

    async def index_directories(
        self, directories: List[str], force_reindex: bool = False, 
        progress_callback: Optional[Callable] = None, num_workers: int = 20
    ) -> IndexStats:
        """Index code files from specified directories with concurrent workers."""
        await self.initialize()

        stats = IndexStats()
        stats_lock = asyncio.Lock()
        processed_files = set()

        # Get existing documents for update/skip logic
        existing_docs = await self._get_existing_docs() if not force_reindex else {}

        # Collect all files to process
        all_files = []
        for directory in directories:
            path = Path(directory).resolve()
            if not path.exists() or not path.is_dir():
                logger.warning(f"Skipping non-existent directory: {directory}")
                stats.errors += 1
                continue

            # Collect files
            files_in_dir = 0
            async for file_path in self._walk_directory_async(path):
                if file_path not in processed_files:
                    processed_files.add(file_path)
                    all_files.append(file_path)
                    files_in_dir += 1
            logger.info(f"Found {files_in_dir} code files in {path}")

        logger.info(f"Found {len(all_files)} files to process with {num_workers} workers")
        
        import time
        start_time = time.time()

        # Create a queue for files to process
        file_queue = asyncio.Queue()
        for file_path in all_files:
            await file_queue.put(file_path)

        # Worker function
        async def worker(worker_id: int):
            while True:
                try:
                    file_path = await asyncio.wait_for(file_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    break

                try:
                    # Process the file
                    result = await self._process_file_with_result(
                        file_path, existing_docs, force_reindex
                    )
                    
                    # Update stats thread-safely
                    async with stats_lock:
                        stats.files_scanned += 1
                        if result == "indexed":
                            stats.files_indexed += 1
                        elif result == "updated":
                            stats.files_updated += 1
                        elif result == "skipped":
                            stats.files_skipped += 1
                        elif result == "error":
                            stats.errors += 1
                        
                        # Call progress callback if provided
                        if progress_callback:
                            await progress_callback({
                                "files_scanned": stats.files_scanned,
                                "files_indexed": stats.files_indexed,
                                "files_updated": stats.files_updated,
                                "files_skipped": stats.files_skipped,
                                "errors": stats.errors,
                                "current_file": str(file_path),
                                "total_files": len(all_files),
                            })
                        
                except Exception as e:
                    logger.error(f"Worker {worker_id} error processing {file_path}: {e}")
                    async with stats_lock:
                        stats.errors += 1

                file_queue.task_done()

        # Start workers
        workers = [asyncio.create_task(worker(i)) for i in range(num_workers)]
        
        # Wait for all files to be processed
        await file_queue.join()
        
        # Cancel workers
        for w in workers:
            w.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*workers, return_exceptions=True)
        
        elapsed = time.time() - start_time
        logger.info(f"Indexing completed in {elapsed:.2f} seconds ({len(all_files)/elapsed:.2f} files/sec)")

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

        # Create query embedding with proper prefix
        query_text = f"Represent this query for searching relevant code: {query}"
        query_embedding = await self._create_embedding(query_text)

        # Perform vector search
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: self.table.search(query_embedding).limit(limit).to_list()
        )

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

        loop = asyncio.get_event_loop()
        table_len = await loop.run_in_executor(None, len, self.table)

        return {
            "total_documents": table_len,
            "initialized": True,
            "model": self.config.embedding_model,
            "database_path": self.config.get_db_path(),
        }

    async def _get_existing_docs(self) -> Dict[str, str]:
        """Get existing documents from the table."""
        existing_docs = {}
        try:
            loop = asyncio.get_event_loop()
            # LanceDB's to_polars() might have issues with batch_size parameter
            # Let's use a different approach
            # Get the table as an arrow table first, then convert to polars
            arrow_table = await loop.run_in_executor(None, self.table.to_arrow)
            
            # Convert Arrow table to Polars
            import polars as pl
            df = pl.from_arrow(arrow_table)
            
            # Select only the columns we need
            df_subset = df.select(["id", "content_hash"])
            
            if df_subset.height > 0:
                for row in df_subset.iter_rows(named=True):
                    existing_docs[row["id"]] = row["content_hash"]
        except Exception as e:
            logger.warning(f"Error reading existing documents: {e}")
        return existing_docs

    async def _walk_directory_async(
        self, directory: Path
    ) -> AsyncGenerator[Path, None]:
        """Asynchronously walk directory and yield code files."""
        loop = asyncio.get_event_loop()

        # Get all files in directory tree
        def get_files():
            files = []
            for item in directory.rglob("*"):
                if not item.is_file():
                    continue

                # Skip excluded patterns
                if any(
                    pattern in str(item) for pattern in self.config.exclude_patterns
                ):
                    continue

                # Check file extension
                if item.suffix.lower() in self.config.code_extensions:
                    files.append(item)
            return files

        files = await loop.run_in_executor(None, get_files)

        # Yield files asynchronously
        for file in files:
            yield file

    async def _process_file_with_result(
        self,
        file_path: Path,
        existing_docs: Dict[str, str],
        force_reindex: bool,
    ) -> str:
        """Process a single file for indexing and return result status."""
        loop = asyncio.get_event_loop()

        try:
            # Check file size
            file_stat = await loop.run_in_executor(None, file_path.stat)
            file_size = file_stat.st_size

            if file_size > self.config.max_file_size:
                logger.debug(f"Skipping large file: {file_path} ({file_size} bytes)")
                return "skipped"

            # Read file content
            try:
                content = await loop.run_in_executor(
                    None, lambda: file_path.read_text(encoding="utf-8", errors="replace")
                )
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                return "skipped"

            # Skip empty files
            if not content.strip():
                return "skipped"

            # Generate document ID and content hash
            doc_id = f"file:{file_path}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Check if update needed
            if not force_reindex and doc_id in existing_docs:
                if existing_docs[doc_id] == content_hash:
                    return "skipped"

            # Create embedding with proper query prefix for code search
            import time
            embed_start = time.time()
            embedding = await self._create_embedding(
                f"Represent this code for searching relevant code: {content}"
            )
            embed_time = time.time() - embed_start
            if embed_time > 1.0:
                logger.debug(f"Embedding generation took {embed_time:.2f}s for {file_path.name}")

            # Prepare document
            doc = CodeDocument(
                id=doc_id,
                file_path=str(file_path),
                content=content,
                file_type=file_path.suffix[1:],  # Remove dot
                file_size=file_size,
                last_modified=datetime.fromtimestamp(file_stat.st_mtime),
                indexed_at=datetime.now(),
                content_hash=content_hash,
                vector=embedding,
            )

            # Insert or update document
            if doc_id in existing_docs:
                # Update existing document by deleting and re-adding
                await loop.run_in_executor(None, self.table.delete, f'id = "{doc_id}"')
                await loop.run_in_executor(None, self.table.add, [doc.model_dump()])
                return "updated"
            else:
                # Add new document
                await loop.run_in_executor(None, self.table.add, [doc.model_dump()])
                return "indexed"
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return "error"

    async def _create_embedding(self, text: str):
        """Create embedding for text using the model."""
        # Limit concurrent embedding generation to avoid overwhelming the model
        async with self._embedding_semaphore:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.model.encode(text, show_progress_bar=False)
            )
            return embedding

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
