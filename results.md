# Embedding run results for kuzu

## granite 125m english

```text
============================================================
EMBEDDINGS TEST COMPLETE
============================================================
Files discovered: 4460
Files processed: 4450
Embeddings generated: 4450
Batches processed: 45

Timing breakdown:
  Discovery time: 0.09 seconds
  Processing time: 115.22 seconds
  Total time: 115.31 seconds

Performance:
  Overall speed: 38.59 files/sec
  Processing speed: 38.62 files/sec

Memory usage:
  Final memory usage: 11811.8MB
  Max memory delta per batch: 5701.4MB
============================================================
```

## codesage small v2 (local)

```text
============================================================
EMBEDDINGS TEST COMPLETE
============================================================
Files discovered: 4460
Files processed: 4450
Embeddings generated: 4450
Batches processed: 45

Timing breakdown:
  Discovery time: 0.09 seconds
  Processing time: 420.54 seconds
  Total time: 420.63 seconds

Performance:
  Overall speed: 10.58 files/sec
  Processing speed: 10.58 files/sec

Memory usage:
  Final memory usage: 3305.5MB
  Max memory delta per batch: 1400.0MB
============================================================
```

## fast indexer

### voyage code 3

```sh
$ uv run ./fast_indexer.py . --model voyage-code-3 --readers 30 --embedders 20 --writers 20 --voyage-requests 15
Initializing fast indexer...
Fast indexer: ready to initialize embedder.
Fast Indexer - Breeze MCP
Starting fast indexing for directory: .
Using model: voyage-code-3
Fast indexing ....
Using Voyage AI model: voyage-code-3
Using tiktoken for accurate token counting with Voyage
Phase 1: Discovering files...
Found 74 files in 0.01 seconds
Opened existing table with 599 documents
Phase 2: Processing files...
Using 30 readers, 20 embedders, 20 writers for batches of ~20 files each
Batch 1: Read 20 files
Batch 1: Starting embeddings for 20 files (waited 0.00s in queue)...
Voyage: Processing 20 texts in 1 API calls
Batch 0: Read 20 files
Batch 0: Starting embeddings for 20 files (waited 0.00s in queue)...
Voyage: Processing 20 texts in 3 API calls
Batch 3: Read 14 files
Batch 2: Read 20 files
Batch 3: Starting embeddings for 14 files (waited 0.00s in queue)...
Voyage: Processing 14 texts in 1 API calls
Batch 2: Starting embeddings for 20 files (waited 0.00s in queue)...
Voyage: Processing 20 texts in 1 API calls
Batch 1: Generated embeddings in 1.82s (encode: 1.82s, 11.0 files/sec)
Batch 2: Generated embeddings in 1.72s (encode: 1.72s, 11.7 files/sec)
Batch 1: Wrote 20 docs in 0.03s (read: 0.01s, embed: 1.82s)
Batch 2: Wrote 20 docs in 0.03s (read: 0.08s, embed: 1.72s)
... elided for brevity ... 
============================================================
INDEXING COMPLETE
============================================================
Files discovered: 74
Files indexed: 74
Errors: 0
Batches processed: 4

Timing breakdown:
  Discovery time: 0.01 seconds
  Total read time: 0.18 seconds
  Total embed time: 22.98 seconds
  Total write time: 0.11 seconds
  Verification time: 0.33 seconds
  Total time: 19.09 seconds

Performance:
  Overall speed: 3.88 files/sec
  Read speed: 414.84 files/sec
  Embed speed: 3.22 files/sec
  Write speed: 699.72 files/sec

Concurrency efficiency:
  Total work time: 23.26 seconds
  Pipeline efficiency: 122.3%
============================================================
```
