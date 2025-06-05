#!/usr/bin/env python3
"""Fast concurrent indexer with parallel file processing."""

import asyncio
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from sentence_transformers import SentenceTransformer
import requests
import json
import numpy as np

try:
    from identify import identify
except ImportError:
    print("Please install identify: pip install identify")
    sys.exit(1)

try:
    import magic

    # Test if libmagic is available
    magic.Magic()
except ImportError:
    print("Please install python-magic: pip install python-magic")
    sys.exit(1)
except Exception as e:
    print("Error: python-magic requires libmagic to be installed.")
    print("  macOS: brew install libmagic")
    print("  Ubuntu/Debian: sudo apt-get install libmagic1")
    print("  RHEL/CentOS: sudo yum install file-devel")
    print(f"\nActual error: {e}")
    sys.exit(1)

import aiofiles
import aiofiles.os
import hashlib
import lancedb
from lancedb.embeddings import get_registry, EmbeddingFunction, EmbeddingFunctionRegistry
from lancedb.embeddings.base import EmbeddingFunction
from lancedb.embeddings.registry import register
from lancedb.pydantic import LanceModel, Vector
from typing import List, Union, ClassVar
import pyarrow as pa

print("Initializing fast indexer...")
# Embedding model will be initialized later based on provider
model = None
print("Fast indexer: ready to initialize embedder.")


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
    
    def compute_query_embeddings(
        self, query: str, *args, **kwargs
    ) -> List[np.ndarray]:
        """Compute the embeddings for a given user query"""
        client = VoyageCode3EmbeddingFunction._get_client()
        result = client.embed(
            texts=[query], model=self.name, input_type="query", **kwargs
        )
        return [result.embeddings[0]]
    
    def compute_source_embeddings(
        self, inputs, *args, **kwargs
    ) -> List[np.array]:
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

# Model will be initialized later with device choice
st_model = None
ollama_model = None
use_ollama = False


def get_ollama_embeddings(texts, model_name="nomic-embed-text"):
    """Get embeddings from Ollama API."""
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = []
    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": model_name,
                "prompt": text
            }
        )
        if response.status_code == 200:
            embedding = response.json()["embedding"]
            embeddings.append(embedding)
        else:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    
    return np.array(embeddings)


def get_ollama_model_info(model_name="nomic-embed-text"):
    """Get model information from Ollama."""
    response = requests.post(
        "http://localhost:11434/api/show",
        json={"model": model_name}
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Cannot get Ollama model info: {response.status_code}")


async def get_voyage_embeddings_with_limits(texts, model, tokenizer=None, max_concurrent_requests=5):
    """Get embeddings from Voyage AI with proper token limit handling."""
    import asyncio
    
    MAX_TOKENS_PER_BATCH = 120000  # Voyage's limit
    MAX_TEXTS_PER_BATCH = 128  # Voyage's limit per request
    
    def estimate_tokens(text):
        """Estimate tokens using tokenizer or character count."""
        if tokenizer:
            return len(tokenizer.encode(text))
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
            if (current_batch and 
                (current_tokens + text_tokens > MAX_TOKENS_PER_BATCH * 0.8 or  # 80% safety margin
                 len(current_batch) >= MAX_TEXTS_PER_BATCH)):
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
    print(f"Voyage: Processing {len(texts)} texts in {len(safe_batches)} API calls")
    
    # Semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def process_batch(batch_idx, batch):
        """Process a single batch with the API."""
        async with semaphore:
            try:
                # Make the API call
                embeddings = await asyncio.to_thread(
                    model.compute_source_embeddings,
                    batch
                )
                return embeddings
            except Exception as e:
                print(f"  Error in Voyage batch {batch_idx + 1}: {e}")
                # Return zero embeddings as fallback
                return [[0.0] * 1024 for _ in batch]  # Assuming 1024 dimensions
    
    # Process all batches concurrently
    tasks = [process_batch(idx, batch) for idx, batch in enumerate(safe_batches)]
    batch_results = await asyncio.gather(*tasks)
    
    # Flatten results
    all_embeddings = []
    for batch_embeddings in batch_results:
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)


def get_code_document_schema(embedding_model=None, vector_dim=None):
    """Get CodeDocument schema with embedding function configured."""

    # Get the vector dimension from the model
    if vector_dim is None and embedding_model is not None:
        vector_dim = embedding_model.ndims()
    elif vector_dim is None:
        raise ValueError("Either embedding_model or vector_dim must be provided")

    class CodeDocument(LanceModel):
        """Code document with embedding vector."""

        id: str
        file_path: str
        content: str
        file_type: str
        file_size: int
        last_modified: datetime
        indexed_at: datetime
        content_hash: str
        vector: Vector(vector_dim)

    return CodeDocument


# CodeDocument will be created after model initialization
CodeDocument = None


# Exclude patterns
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".tox",
    ".mypy_cache",
    ".coverage",
    ".hypothesis",
]


def should_index_file(file_path: Path) -> bool:
    """Check if file should be indexed - any text file not in excluded directories."""
    # Check exclude patterns first
    path_str = str(file_path)
    for pattern in EXCLUDE_PATTERNS:
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


def walk_directory_fast(directory: Path) -> list[Path]:
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
            d for d in dirs if not d.startswith(".") and d not in EXCLUDE_PATTERNS
        ]

        # Check files
        for file in files:
            if file.startswith("."):
                continue

            file_path = root_path / file
            if should_index_file(file_path):
                files_to_index.append(file_path)

    return files_to_index


async def read_file_batch(file_paths: list[Path], batch_num: int) -> list[dict]:
    """Read a batch of files concurrently."""
    doc_datas = []

    async def read_single_file(file_path: Path) -> dict:
        try:
            stat = await aiofiles.os.stat(file_path)

            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="replace"
            ) as f:
                content = await f.read()

            # Skip empty files
            if not content.strip():
                return None

            # Create document metadata (without vector)
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

            return doc_data
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    # Read files concurrently within the batch
    tasks = [read_single_file(fp) for fp in file_paths]
    results = await asyncio.gather(*tasks)

    # Filter out None results
    doc_datas = [doc for doc in results if doc is not None]

    if doc_datas:
        print(f"Batch {batch_num}: Read {len(doc_datas)} files")

    return doc_datas


async def index_directory_fast(directory: str, db_path: str = "./fast_index.db", device: str = "mps", 
                              ollama_model_name: str = None, model_name: str = "ibm-granite/granite-embedding-125m-english",
                              api_key: str = None, batch_size: int = None, 
                              concurrent_readers: int = 20, concurrent_embedders: int = 10, 
                              concurrent_writers: int = 10, voyage_concurrent_requests: int = 5):
    """Index all code files in a directory with maximum parallelism."""
    global st_model, ollama_model, use_ollama, CodeDocument, model
    
    print(f"Fast indexing {directory}...")
    start_time = time.time()
    
    # Initialize tokenizer for Voyage (if needed)
    tokenizer = None
    
    # Initialize the model based on provider
    if ollama_model_name:
        use_ollama = True
        ollama_model = ollama_model_name
        print(f"Using Ollama model: {ollama_model_name}")
        
        # Get a test embedding to determine dimension
        test_embedding = get_ollama_embeddings("test", ollama_model_name)
        vector_dim = len(test_embedding[0])
        print(f"Ollama model dimension: {vector_dim}")
        
        # Recreate schema with correct dimension
        CodeDocument = get_code_document_schema(vector_dim=vector_dim)
    elif model_name.startswith("voyage-"):
        # Use Voyage AI through LanceDB
        use_ollama = False
        print(f"Using Voyage AI model: {model_name}")
        
        if not api_key:
            api_key = os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError("VOYAGE_API_KEY environment variable or --api-key required for Voyage models")
        
        # Set API key in environment for our custom function
        os.environ["VOYAGE_API_KEY"] = api_key
        
        registry = get_registry()
        
        # Use built-in for voyage-code-2, custom for voyage-code-3
        if model_name == "voyage-code-3":
            model = registry.get("voyage-code-3").create(name=model_name)
        else:
            # Use built-in voyageai for other models
            model = registry.get("voyageai").create(name=model_name)
        
        # Recreate schema with correct dimension
        CodeDocument = get_code_document_schema(model)
        
        # For embeddings, we'll use the model directly
        st_model = model
        
        # Initialize tokenizer for Voyage
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding("cl100k_base")
            print("Using tiktoken for accurate token counting with Voyage")
        except ImportError:
            print("Warning: tiktoken not available, token limits may be exceeded")
            print("Install with: pip install tiktoken")
        
    elif model_name.startswith("models/"):
        # Use Google Gemini through LanceDB
        use_ollama = False
        print(f"Using Google Gemini model: {model_name}")
        registry = get_registry()
        
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable or --api-key required for Gemini models")
        
        model = registry.get("gemini-text").create(
            name=model_name,
            api_key=api_key
        )
        
        # Recreate schema with correct dimension
        CodeDocument = get_code_document_schema(model)
        
        # For embeddings, we'll use the model directly
        st_model = model
        
    else:
        # Use sentence-transformers
        use_ollama = False
        print(f"Loading SentenceTransformer model {model_name} on {device}...")
        
        # Use LanceDB registry for sentence-transformers
        registry = get_registry()
        model = registry.get("sentence-transformers").create(
            name=model_name,
            device=device,
            normalize=True,
            trust_remote_code=True,
            show_progress_bar=False,
        )
        
        # Also create SentenceTransformer instance for direct encoding
        st_model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        
        # Recreate schema with correct dimension
        CodeDocument = get_code_document_schema(model)
        
        print(f"Model loaded. Embedding dimension: {model.ndims()}")

    # Phase 1: Fast directory walk (synchronous for speed)
    print("Phase 1: Discovering files...")
    walk_start = time.time()
    path = Path(directory).resolve()
    files_to_index = walk_directory_fast(path)
    walk_time = time.time() - walk_start
    print(f"Found {len(files_to_index)} files in {walk_time:.2f} seconds")

    # Connect to database
    db = await lancedb.connect_async(db_path)

    # Create or open table
    table_name = "documents"
    existing_tables = await db.table_names()

    if table_name in existing_tables:
        table = await db.open_table(table_name)
        count = await table.count_rows()
        print(f"Opened existing table with {count} documents")
    else:
        # Create table - embedding function is configured in the schema
        table = await db.create_table(table_name, schema=CodeDocument)
        print("Created new table with embedding configuration")

    # Phase 2: Process files in parallel batches
    print("Phase 2: Processing files...")
    process_start = time.time()

    # Batch size - use provided or defaults
    if batch_size is None:
        batch_size = 100 if model_name == "ibm-granite/granite-embedding-125m-english" else 50
        # For Voyage, use smaller batches to avoid token limits
        if model_name.startswith("voyage-"):
            batch_size = 20

    print(
        f"Using {concurrent_readers} readers, {concurrent_embedders} embedders, "
        f"{concurrent_writers} writers for batches of ~{batch_size} files each"
    )

    # Create batches
    batches = []
    for i in range(0, len(files_to_index), batch_size):
        batches.append(files_to_index[i : i + batch_size])

    # Process batches with three-stage pipeline
    total_indexed = 0
    errors = 0
    total_read_time = 0
    total_embed_time = 0
    total_write_time = 0

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
    write_results = []

    async def reader_task(batch_idx: int, batch: list[Path]):
        """Read files and queue for embedding."""
        async with read_sem:
            read_start = time.time()
            doc_datas = await read_file_batch(batch, batch_idx)
            read_time = time.time() - read_start

            if doc_datas:
                await embed_queue.put((batch_idx, doc_datas, read_time))
            else:
                # Even empty batches need to be tracked
                await embed_queue.put((batch_idx, [], read_time))

    async def embedder_task():
        """Pull from read queue, generate embeddings, and queue for writing."""
        while True:
            try:
                # Wait for items with a timeout
                batch_idx, doc_datas, read_time = await asyncio.wait_for(
                    embed_queue.get(), timeout=1.0
                )

                async with embed_sem:
                    if doc_datas:
                        embed_start = time.time()

                        # Extract contents for embedding
                        contents = [doc["content"] for doc in doc_datas]

                        # Add timing to detect serialization
                        queue_time = time.time() - embed_start
                        print(
                            f"Batch {batch_idx}: Starting embeddings for {len(contents)} files (waited {queue_time:.2f}s in queue)..."
                        )
                        
                        # Run embedding generation
                        encode_start = time.time()
                        if use_ollama:
                            embeddings = get_ollama_embeddings(contents, ollama_model)
                        elif model_name.startswith("voyage-"):
                            # Special handling for Voyage with token limits
                            embeddings = await get_voyage_embeddings_with_limits(
                                contents, st_model, tokenizer, voyage_concurrent_requests
                            )
                        elif hasattr(st_model, 'compute_source_embeddings'):
                            # LanceDB embedding function (Gemini, etc)
                            embeddings = st_model.compute_source_embeddings(contents)
                        else:
                            # SentenceTransformer
                            embeddings = st_model.encode(
                                contents, 
                                batch_size=32,  # Process in chunks
                                show_progress_bar=False
                            )
                        encode_time = time.time() - encode_start

                        # Create CodeDocument objects with embeddings
                        documents = []
                        for doc_data, embedding in zip(doc_datas, embeddings):
                            doc_data["vector"] = embedding
                            documents.append(CodeDocument(**doc_data))

                        embed_time = time.time() - embed_start
                        print(
                            f"Batch {batch_idx}: Generated embeddings in {embed_time:.2f}s "
                            f"(encode: {encode_time:.2f}s, {len(contents)/encode_time:.1f} files/sec)"
                        )

                        await write_queue.put(
                            (batch_idx, documents, read_time, embed_time)
                        )
                    else:
                        await write_queue.put((batch_idx, [], read_time, 0))

            except asyncio.TimeoutError:
                # Check if reading is complete and queue is empty
                if read_complete.is_set() and embed_queue.empty():
                    break

    async def writer_task():
        """Pull from embed queue and write to database."""
        while True:
            try:
                # Wait for items with a timeout
                batch_idx, documents, read_time, embed_time = await asyncio.wait_for(
                    write_queue.get(), timeout=1.0
                )

                async with write_sem:
                    if documents:
                        write_start = time.time()
                        try:
                            await (
                                table.merge_insert("id")
                                .when_matched_update_all()
                                .when_not_matched_insert_all()
                                .execute(documents)
                            )
                            write_time = time.time() - write_start
                            print(
                                f"Batch {batch_idx}: Wrote {len(documents)} docs in {write_time:.2f}s "
                                f"(read: {read_time:.2f}s, embed: {embed_time:.2f}s)"
                            )
                            write_results.append(
                                (len(documents), 0, read_time, embed_time, write_time)
                            )
                        except Exception as e:
                            print(f"Error writing batch {batch_idx}: {e}")
                            write_results.append(
                                (0, len(documents), read_time, embed_time, 0)
                            )
                    else:
                        write_results.append((0, 0, read_time, embed_time, 0))

            except asyncio.TimeoutError:
                # Check if embedding is complete and queue is empty
                if embed_complete.is_set() and write_queue.empty():
                    break

    # Start all pipeline tasks in order
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

    # Aggregate results
    for indexed, errs, read_time, embed_time, write_time in write_results:
        total_indexed += indexed
        errors += errs
        total_read_time += read_time
        total_embed_time += embed_time
        total_write_time += write_time

    process_time = time.time() - process_start
    total_time = time.time() - start_time

    # Verification phase
    print("\nVerifying index...")
    verify_start = time.time()

    # Check row count
    actual_count = await table.count_rows()
    print(f"Documents in table: {actual_count}")

    # Import polars once at the beginning
    import polars as pl

    # Try to retrieve some documents to verify structure
    df = None
    try:
        # Get a few rows to check
        sample_data = await table.to_arrow()
        df = pl.from_arrow(sample_data).head(5)
        print(f"Retrieved {df.height} sample documents for verification")
    except Exception as e:
        print(f"Failed to retrieve sample documents: {e}")

    # Try vector search if we have the model
    if df is not None and df.height > 0:
        try:
            # Check if vector column exists
            if "vector" in df.columns:
                print("Vector column exists in table")
                # Try a vector search
                test_query = "function definition code"
                query_embedding = model.compute_query_embeddings(test_query)[0]

                search_query = await table.search(
                    query_embedding, vector_column_name="vector"
                )
                search_results = await search_query.limit(5).to_arrow()
                search_df = pl.from_arrow(search_results)
                print(f"Vector search returned {search_df.height} results")
            else:
                print("WARNING: No vector column found in table!")
        except Exception as e:
            print(f"Vector search test failed: {e}")

    # Check a few documents to ensure content was stored
    if df is not None and df.height > 0:
        first_doc = df.to_dicts()[0]
        print("Sample document:")
        print(f"  File: {first_doc.get('file_path', 'N/A')}")
        print(f"  Size: {first_doc.get('file_size', 0)} bytes")
        print(
            f"  Has content: {'content' in first_doc and len(first_doc['content']) > 0}"
        )
        print(
            f"  Has vector: {'vector' in first_doc and first_doc['vector'] is not None}"
        )

    verify_time = time.time() - verify_start

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"Files discovered: {len(files_to_index)}")
    print(f"Files indexed: {total_indexed}")
    print(f"Errors: {errors}")
    print(f"Batches processed: {len(batches)}")
    print("\nTiming breakdown:")
    print(f"  Discovery time: {walk_time:.2f} seconds")
    print(f"  Total read time: {total_read_time:.2f} seconds")
    print(f"  Total embed time: {total_embed_time:.2f} seconds")
    print(f"  Total write time: {total_write_time:.2f} seconds")
    print(f"  Verification time: {verify_time:.2f} seconds")
    print(f"  Total time: {total_time:.2f} seconds")
    print("\nPerformance:")
    print(f"  Overall speed: {total_indexed / total_time:.2f} files/sec")
    if total_read_time > 0:
        print(f"  Read speed: {total_indexed / total_read_time:.2f} files/sec")
    if total_embed_time > 0:
        print(f"  Embed speed: {total_indexed / total_embed_time:.2f} files/sec")
    if total_write_time > 0:
        print(f"  Write speed: {total_indexed / total_write_time:.2f} files/sec")
    print("\nConcurrency efficiency:")
    total_work_time = total_read_time + total_embed_time + total_write_time
    print(f"  Total work time: {total_work_time:.2f} seconds")
    print(f"  Pipeline efficiency: {(total_work_time / process_time * 100):.1f}%")
    print("=" * 60)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python fast_indexer.py <directory> [options]")
        print("Options:")
        print("  --device cpu|mps|cuda    Device for local models")
        print("  --model MODEL_NAME       Model name (default: ibm-granite/granite-embedding-125m-english)")
        print("  --ollama MODEL_NAME      Use Ollama model")
        print("  --api-key KEY           API key for Voyage/Gemini models")
        print("  --batch-size N          Files per batch (default: auto)")
        print("  --readers N             Concurrent file readers (default: 20)")
        print("  --embedders N           Concurrent embedders (default: 10)")
        print("  --writers N             Concurrent writers (default: 10)")
        print("  --voyage-requests N     Concurrent Voyage API requests (default: 5)")
        print("\nExamples:")
        print("  # Voyage AI with custom concurrency")
        print("  python fast_indexer.py /path --model voyage-code-3 --voyage-requests 10")
        print("  # Custom pipeline concurrency")
        print("  python fast_indexer.py /path --readers 30 --embedders 15 --writers 15")
        print("  # Google Gemini")
        print("  python fast_indexer.py /path --model models/text-embedding-004")
        print("  # Ollama")
        print("  python fast_indexer.py /path --ollama nomic-embed-text")
        sys.exit(1)

    print("Fast Indexer - Breeze MCP")

    directory = sys.argv[1]
    
    # Parse options
    device = "mps"  # Default
    ollama_model = None
    model_name = "ibm-granite/granite-embedding-125m-english"  # Default
    api_key = None
    batch_size = None
    concurrent_readers = 20
    concurrent_embedders = 10
    concurrent_writers = 10
    voyage_concurrent_requests = 5
    
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]
        elif arg == "--model" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
        elif arg == "--ollama" and i + 1 < len(sys.argv):
            ollama_model = sys.argv[i + 1]
        elif arg == "--api-key" and i + 1 < len(sys.argv):
            api_key = sys.argv[i + 1]
        elif arg == "--batch-size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
        elif arg == "--readers" and i + 1 < len(sys.argv):
            concurrent_readers = int(sys.argv[i + 1])
        elif arg == "--embedders" and i + 1 < len(sys.argv):
            concurrent_embedders = int(sys.argv[i + 1])
        elif arg == "--writers" and i + 1 < len(sys.argv):
            concurrent_writers = int(sys.argv[i + 1])
        elif arg == "--voyage-requests" and i + 1 < len(sys.argv):
            voyage_concurrent_requests = int(sys.argv[i + 1])
    
    print(f"Starting fast indexing for directory: {directory}")
    
    if ollama_model:
        print(f"Using Ollama model: {ollama_model}")
    else:
        print(f"Using model: {model_name}")
        if not model_name.startswith(("voyage-", "models/")):
            print(f"Using device: {device}")
    
    await index_directory_fast(
        directory, 
        device=device, 
        ollama_model_name=ollama_model,
        model_name=model_name,
        api_key=api_key,
        batch_size=batch_size,
        concurrent_readers=concurrent_readers,
        concurrent_embedders=concurrent_embedders,
        concurrent_writers=concurrent_writers,
        voyage_concurrent_requests=voyage_concurrent_requests
    )


if __name__ == "__main__":
    asyncio.run(main())
