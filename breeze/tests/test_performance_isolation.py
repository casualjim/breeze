"""Performance isolation tests to identify bottlenecks between embedding generation and LanceDB writes."""

import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List

import lancedb
from lancedb.pydantic import LanceModel
from pydantic import Field

from breeze.core.content_detection import ContentDetector
from breeze.core.embeddings import SentenceTransformerEmbeddings
from breeze.core.file_discovery import FileDiscovery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleCodeDocument(LanceModel):
    """Code document without embeddings - no SourceField or VectorField."""
    
    id: str = Field(description="Unique identifier for the document")
    file_path: str = Field(description="Path to the code file")
    content: str = Field(description="Full content of the code file")
    file_type: str = Field(description="File extension without dot")
    file_size: int = Field(description="Size of the file in bytes")
    last_modified: datetime = Field(description="Last modification time of the file")
    indexed_at: datetime = Field(description="Time when the file was indexed")
    content_hash: str = Field(description="MD5 hash of the file content")
    # NO vector field - this is the key difference!


def walk_directory(directory: str) -> List[Path]:
    """Walk directory and find code files using Breeze's file discovery."""
    content_detector = ContentDetector()
    file_discovery = FileDiscovery(
        exclude_patterns=[],
        should_index_file=content_detector.should_index_file
    )
    return file_discovery.walk_directory(Path(directory))


async def test_embedding_generation_only(repo_path: str = "/Users/ivan/github/kuzudb/kuzu"):
    """Test pure embedding generation performance without LanceDB."""
    
    logger.info(f"Starting embedding generation test for: {repo_path}")
    
    # Initialize embedding model directly
    embedding_model = SentenceTransformerEmbeddings(
        name="all-MiniLM-L6-v2",
        device="cpu",
        show_progress_bar=False,  # Disable progress bar for cleaner output
        normalize=True
    )
    
    # Walk directory and collect files
    logger.info("Discovering files...")
    start_discovery = time.time()
    files = walk_directory(repo_path)
    discovery_time = time.time() - start_discovery
    logger.info(f"Found {len(files)} files in {discovery_time:.2f} seconds")
    
    # Read all file contents
    logger.info("Reading file contents...")
    start_read = time.time()
    file_contents = []
    read_errors = 0
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                if content.strip():  # Skip empty files
                    file_contents.append((str(file_path), content))
        except Exception as e:
            read_errors += 1
            logger.debug(f"Error reading {file_path}: {e}")
    
    read_time = time.time() - start_read
    logger.info(f"Read {len(file_contents)} files in {read_time:.2f} seconds ({read_errors} errors)")
    
    # Measure embedding generation
    logger.info("Generating embeddings...")
    start_embed = time.time()
    
    embeddings = []
    batch_size = 32
    total_batches = (len(file_contents) + batch_size - 1) // batch_size
    
    for i in range(0, len(file_contents), batch_size):
        batch_num = i // batch_size + 1
        batch = file_contents[i:i+batch_size]
        texts = [content for _, content in batch]
        
        if batch_num % 10 == 0:
            logger.info(f"Processing batch {batch_num}/{total_batches}")
        
        # Generate embeddings
        try:
            batch_embeddings = embedding_model.compute_source_embeddings(texts)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error in batch {batch_num}: {e}")
            # Add None for failed embeddings
            embeddings.extend([None] * len(texts))
    
    embed_time = time.time() - start_embed
    successful_embeddings = sum(1 for e in embeddings if e is not None)
    
    # Print results
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION RESULTS")
    print("=" * 60)
    print(f"Repository: {repo_path}")
    print(f"Files discovered: {len(files)}")
    print(f"Files with content: {len(file_contents)}")
    print(f"Embeddings generated: {successful_embeddings}/{len(file_contents)}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"\nTiming breakdown:")
    print(f"- File discovery: {discovery_time:.2f} seconds")
    print(f"- File reading: {read_time:.2f} seconds")
    print(f"- Embedding generation: {embed_time:.2f} seconds")
    print(f"\nPerformance metrics:")
    print(f"- Files per second (discovery): {len(files) / discovery_time:.2f}")
    print(f"- Files per second (reading): {len(file_contents) / read_time:.2f}")
    print(f"- Embeddings per second: {successful_embeddings / embed_time:.2f}")
    print(f"- Average time per embedding: {embed_time / successful_embeddings * 1000:.2f} ms")
    print("=" * 60)
    
    return {
        "files_discovered": len(files),
        "files_read": len(file_contents),
        "embeddings_generated": successful_embeddings,
        "discovery_time": discovery_time,
        "read_time": read_time,
        "embed_time": embed_time,
        "total_time": discovery_time + read_time + embed_time
    }


async def test_lancedb_write_only(repo_path: str = "/Users/ivan/github/kuzudb/kuzu", db_path: str = "./test_perf_db"):
    """Test pure LanceDB write performance without embeddings."""
    
    logger.info(f"Starting LanceDB write test for: {repo_path}")
    
    # Initialize LanceDB
    logger.info(f"Connecting to LanceDB at: {db_path}")
    db = await lancedb.connect_async(db_path)
    
    # Create table with simple schema (no embeddings)
    table_name = "test_documents"
    existing_tables = await db.table_names()
    if table_name in existing_tables:
        await db.drop_table(table_name)
    
    table = await db.create_table(
        table_name,
        schema=SimpleCodeDocument,
        mode="overwrite"
    )
    logger.info("Created table with simple schema (no embeddings)")
    
    # Walk directory and collect files
    logger.info("Discovering files...")
    start_discovery = time.time()
    files = walk_directory(repo_path)
    discovery_time = time.time() - start_discovery
    logger.info(f"Found {len(files)} files in {discovery_time:.2f} seconds")
    
    # Read files and prepare documents
    logger.info("Reading files and preparing documents...")
    start_prep = time.time()
    
    all_documents = []
    read_errors = 0
    
    for file_path in files:
        try:
            stat = os.stat(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            if content.strip():  # Skip empty files
                doc = SimpleCodeDocument(
                    id=f"file:{file_path}",
                    file_path=str(file_path),
                    content=content,
                    file_type=Path(file_path).suffix[1:] if Path(file_path).suffix else "txt",
                    file_size=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime),
                    indexed_at=datetime.now(),
                    content_hash=hashlib.md5(content.encode()).hexdigest()
                )
                all_documents.append(doc)
        except Exception as e:
            read_errors += 1
            logger.debug(f"Error processing {file_path}: {e}")
    
    prep_time = time.time() - start_prep
    logger.info(f"Prepared {len(all_documents)} documents in {prep_time:.2f} seconds ({read_errors} errors)")
    
    # Write to LanceDB in batches
    logger.info("Writing to LanceDB...")
    start_write = time.time()
    
    batch_size = 100
    write_errors = 0
    total_written = 0
    total_batches = (len(all_documents) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_documents), batch_size):
        batch_num = i // batch_size + 1
        batch = all_documents[i:i+batch_size]
        
        if batch_num % 10 == 0:
            logger.info(f"Writing batch {batch_num}/{total_batches}")
        
        try:
            await table.add(batch)
            total_written += len(batch)
        except Exception as e:
            write_errors += 1
            logger.error(f"Error writing batch {batch_num}: {e}")
    
    write_time = time.time() - start_write
    
    # Verify final count
    final_count = await table.count_rows()
    
    # Print results
    print("\n" + "=" * 60)
    print("LANCEDB WRITE RESULTS")
    print("=" * 60)
    print(f"Repository: {repo_path}")
    print(f"Database: {db_path}")
    print(f"Files discovered: {len(files)}")
    print(f"Documents prepared: {len(all_documents)}")
    print(f"Documents written: {total_written}")
    print(f"Final table count: {final_count}")
    print(f"Write errors: {write_errors}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"\nTiming breakdown:")
    print(f"- File discovery: {discovery_time:.2f} seconds")
    print(f"- Document preparation: {prep_time:.2f} seconds")
    print(f"- LanceDB writes: {write_time:.2f} seconds")
    print(f"\nPerformance metrics:")
    print(f"- Files per second (discovery): {len(files) / discovery_time:.2f}")
    print(f"- Documents per second (prep): {len(all_documents) / prep_time:.2f}")
    print(f"- Writes per second: {total_written / write_time:.2f}")
    print(f"- Average time per write: {write_time / total_written * 1000:.2f} ms")
    print("=" * 60)
    
    return {
        "files_discovered": len(files),
        "documents_prepared": len(all_documents),
        "documents_written": total_written,
        "discovery_time": discovery_time,
        "prep_time": prep_time,
        "write_time": write_time,
        "total_time": discovery_time + prep_time + write_time
    }


async def test_compare_performance(repo_path: str = "/Users/ivan/github/kuzudb/kuzu"):
    """Run both tests and compare results."""
    
    print("\n" + "=" * 80)
    print("PERFORMANCE ISOLATION TEST")
    print(f"Repository: {repo_path}")
    print("=" * 80)
    
    # Test 1: Embedding generation only
    print("\n>>> TEST 1: EMBEDDING GENERATION ONLY\n")
    embed_results = await test_embedding_generation_only(repo_path)
    
    # Give some time between tests
    await asyncio.sleep(2)
    
    # Test 2: LanceDB write only
    print("\n>>> TEST 2: LANCEDB WRITE ONLY\n")
    lance_results = await test_lancedb_write_only(repo_path)
    
    # Compare results
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"Total time:")
    print(f"- Embedding generation: {embed_results['total_time']:.2f} seconds")
    print(f"- LanceDB writes: {lance_results['total_time']:.2f} seconds")
    print(f"\nBottleneck analysis:")
    
    if embed_results['total_time'] > lance_results['total_time']:
        ratio = embed_results['total_time'] / lance_results['total_time']
        print(f">>> Embedding generation is {ratio:.1f}x SLOWER than LanceDB writes")
        print(f">>> BOTTLENECK: Embedding generation")
    else:
        ratio = lance_results['total_time'] / embed_results['total_time']
        print(f">>> LanceDB writes are {ratio:.1f}x SLOWER than embedding generation")
        print(f">>> BOTTLENECK: LanceDB writes")
    
    print("\nDetailed comparison:")
    print(f"- File discovery: {embed_results['discovery_time']:.2f}s vs {lance_results['discovery_time']:.2f}s")
    print(f"- File reading: {embed_results['read_time']:.2f}s vs {lance_results['prep_time']:.2f}s")
    print(f"- Core operation: {embed_results['embed_time']:.2f}s (embed) vs {lance_results['write_time']:.2f}s (write)")
    print("=" * 80)


async def main():
    """Main entry point for the performance tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance isolation tests for Breeze")
    parser.add_argument("--repo", default="/Users/ivan/github/kuzudb/kuzu", 
                        help="Repository path to test")
    parser.add_argument("--test", choices=["embed", "lance", "both"], default="both",
                        help="Which test to run")
    parser.add_argument("--db-path", default="./test_perf_db",
                        help="Path for test LanceDB database")
    
    args = parser.parse_args()
    
    if args.test == "embed":
        await test_embedding_generation_only(args.repo)
    elif args.test == "lance":
        await test_lancedb_write_only(args.repo, args.db_path)
    else:
        await test_compare_performance(args.repo)


if __name__ == "__main__":
    asyncio.run(main())