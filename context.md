# Breeze Enhancement Context

This document captures the implementation context for Breeze, including recent work on chunking, rate limiting, and embeddings.

## Current State

### ✅ Resolved Issues

1. **Fixed Chunking Implementation**: Both embedders now properly chunk instead of truncating
   - `get_local_embeddings_with_tokenizer_chunking` - Uses TextChunker for semantic chunking
   - `get_voyage_embeddings_with_limits` - Uses TextChunker with 16k token chunks
   - Tests added to prevent regression

2. **Replaced Dict Returns with Dataclasses**:
   - Created `EmbeddingResult` and `VoyageEmbeddingResult` dataclasses
   - Type-safe returns instead of wishy-washy dicts
   - Better IDE support and error detection

3. **Integrated breeze-langdetect**:
   - Uses hyperpolyglot for comprehensive language detection
   - Single source of truth in `ContentDetector`
   - Supports file type categorization (TEXT, IMAGE, VIDEO, etc.)

4. **Removed ModelAwareChunker**:
   - Deleted obsolete `chunking.py` and related tests
   - Now using `TextChunker` exclusively for all chunking needs

5. **Fixed MPS Memory Issues**:
   - Batch size set to 1 for MPS devices to avoid memory allocation errors
   - Added explicit memory management for MPS devices
   - Reduced concurrent requests for MPS to prevent GPU conflicts

6. **Disabled Progress Bars**:
   - Set `TQDM_DISABLE=1` and `HF_DISABLE_PROGRESS_BAR=1` environment variables
   - Disabled sentence-transformers progress bars with `default_show_progress_bar = False`
   - Progress bars suppressed in both CLI and programmatic usage

7. **Fixed Tensor Dimension Mismatches**:
   - Added embedding shape normalization to ensure consistent 1D arrays
   - Embeddings are squeezed if they have extra dimensions
   - Proper shape validation before combining chunk embeddings

### 🚧 Future Enhancements

1. **nvim-treesitter Integration**: Use `locals.scm` files for richer semantic queries
   - Better boundary detection with scope understanding
   - Support for 100+ languages with community-maintained queries
   - More granular semantic units (imports, references, fields)

2. **Chunker Metadata**: Based on acceptance tests, chunks should include:
   - `node_type` (function, class, method, etc.)
   - `language` per chunk
   - `parent_context` for split large functions
   - `chunking_method` (semantic vs character)

## Key Decisions Made

### Chunking Strategy

- **Chunk Size**: 16k tokens (instead of 30k or 8k)
  - Sweet spot for code files
  - Most files fit in one chunk
  - Large files get 2-8 manageable chunks
  - Can batch 7 chunks per Voyage API call
  
- **Combination Method**: `weighted_average` (default)
  - Weights chunks by token count
  - Better than simple average for varying chunk sizes
  - Other options: `average`, `max_pool`, `first`

### Rate Limiting

- Voyage tiers properly configured:
  - Tier 1: 3M tokens/min, 2000 requests/min
  - Tier 2: 6M tokens/min, 4000 requests/min  
  - Tier 3: 9M tokens/min, 6000 requests/min
- 10% safety margin applied to avoid hitting limits
- `RateLimiterV2` uses token bucket algorithm with proper async handling

## Implementation Status

### ✅ Completed Tasks

1. **Python Version Update** - Changed from 3.13 to 3.12 for tree-sitter-languages compatibility
2. **Dependencies Updated** - Added gitignore_parser, tree-sitter-languages, breeze-langdetect to pyproject.toml
3. **Content Detection** - Implemented hyperpolyglot-based detection in `breeze/core/content_detection.py`
4. **Gitignore Support** - Added gitignore filtering to `_walk_directory_fast()` in engine.py
5. **Tree-sitter Snippet Extraction** - Implemented `TreeSitterSnippetExtractor` in `breeze/core/snippets.py`
6. **Tree-sitter Query Files** - Created proper query files for semantic extraction in `breeze/core/tree_sitter_queries.py`
7. **Reranker Configuration** - Added reranker fields to BreezeConfig
8. **Batch Size Configuration** - Made batch sizes configurable instead of hardcoded
9. **Reranking Pipeline** - Implemented support for Voyage, Gemini, and local rerankers in search()
10. **Pre-chunking for Local Models** - Added AutoTokenizer support for local embedding models
11. **Test Creation** - Written comprehensive tests for gitignore filtering and reranking
12. **Test Fixes** - Fixed all existing tests (config parameters, model names, API methods)
13. **Rate Limiting Test Consolidation** - Consolidated redundant rate limiting tests
14. **Fixed Local Embedder Chunking** - Now properly chunks using TextChunker instead of truncating
15. **Fixed Voyage Embedder Chunking** - Now properly chunks with TextChunker and correct rate limits
16. **Created Voyage Chunking Test** - Prevents regression of truncation bug
17. **Tree-sitter Queries Made Extensible** - Added support for Zig, shell scripts, and dynamic query loading
18. **Dynamic Language Detection** - Created `LanguageDetector` class to minimize hardcoded mappings
19. **Integrated breeze-langdetect** - Full hyperpolyglot integration with file type categorization
20. **Replaced Dict Returns with Dataclasses** - Type-safe returns for embedding functions
21. **Fixed All Test Failures** - Including queue recovery and rate limiting tests
22. **Merged Duplicate Test Files** - Consolidated _v2 test files with originals
23. **Removed ModelAwareChunker** - Deleted obsolete chunking implementation
24. **Updated Navigation Module** - Uses ContentDetector instead of LanguageDetector
25. **Fixed MPS Memory Issues** - Set batch size to 1 for MPS devices
26. **Disabled Progress Bars** - Added environment variables to suppress all progress bars
27. **Fixed Embedding Dimensions** - Normalized embeddings to consistent 1D arrays

### 📝 TODO

1. **Enhance TextChunker with metadata** - Add node_type, language, parent_context fields
2. **Implement configurable semantic granularity** - Support class-level vs function-level chunking
3. **Add incremental chunking** - Check file hash/mtime before rechunking
4. **Integrate nvim-treesitter queries** - Use locals.scm for richer semantic understanding
5. **Refactor engine.py** - Split into search, project, and task modules
6. **Add chunk merging** - Combine small adjacent chunks intelligently

## Critical Code Locations

### Embeddings (`breeze/core/embeddings.py`)

- `EmbeddingResult` & `VoyageEmbeddingResult` - Dataclass returns for type safety
- `get_voyage_embeddings_with_limits` - Voyage embeddings with proper TextChunker integration
- `get_local_embeddings_with_tokenizer_chunking` - Local embeddings using TextChunker
- `create_batches_from_chunked_files` - Utility for batch creation from chunks

### Text Chunker (`breeze/core/text_chunker.py`)

- `TextChunker` - Semantic chunking with natural boundaries
- `ChunkingConfig` - Configuration for chunk size, overlap, and behavior
- `FileContent` - Input wrapper with content, path, and language
- `ChunkedFile` - Output with source file and chunks
- `TextChunk` - Individual chunk with position and metadata

### Content Detection (`breeze/core/content_detection.py`)

- `ContentDetector` - Single source of truth for language detection
- Uses `breeze-langdetect` (hyperpolyglot) for comprehensive detection
- `BINARY_CATEGORIES` - File types to skip (IMAGE, VIDEO, AUDIO, etc.)
- Replaces the now-deleted `LanguageDetector`

### Tree-sitter Queries (`breeze/core/tree_sitter_queries.py`)

- `QueryManager` - Extensible query management system
- Supports loading custom queries from JSON files
- Built-in queries for 100+ languages including Zig and shell scripts
- Single source of truth for semantic code understanding

## Acceptance Tests (`breeze/tests/chunker_acceptance.py`)

Comprehensive test suite that defines expected chunker behavior:

### High Priority Features (from acceptance tests):
- **Semantic metadata**: chunks need `node_type`, `language`, `parent_context`
- **Empty file handling**: graceful handling of empty/whitespace-only files
- **Parser caching**: cache tree-sitter parsers for performance
- **Configurable granularity**: chunk at class vs function level

### Medium Priority Features:
- **Incremental chunking**: skip unchanged files
- **Small chunk merging**: combine tiny adjacent semantic units
- **Cross-language token estimation**: account for language differences

### Low Priority Features:
- **Advanced pooling**: structure-aware, norm-weighted, entropy-based
- **Parallel processing**: current speed is acceptable
- **Query configuration**: exclude specific semantic units

## Testing Commands

```bash
# Run all tests (recommended)
uv run pytest

# Run chunking tests
uv run pytest breeze/tests/test_local_embedder_chunking.py -v
uv run pytest breeze/tests/test_voyage_chunking.py -v
uv run pytest breeze/tests/test_text_chunker.py -v

# Run acceptance tests (many will fail - they're aspirational)
uv run pytest breeze/tests/chunker_acceptance.py -v

# Run specific test categories
uv run pytest breeze/tests/test_content_detection.py -v
uv run pytest breeze/tests/test_gitignore_filtering.py -v
uv run pytest breeze/tests/test_reranking.py -v
uv run pytest breeze/tests/test_snippets.py -v
uv run pytest breeze/tests/test_rate_limiting.py -v
```

## Environment Setup

### Required Environment Variables

```bash
# Embedding configuration
BREEZE_EMBEDDING_MODEL=voyage-code-3
BREEZE_EMBEDDING_API_KEY=your-voyage-api-key
VOYAGE_API_KEY=your-voyage-api-key  # Alternative

# Optional reranker configuration
BREEZE_RERANKER_MODEL=rerank-2  # Auto-selected if not set
BREEZE_RERANKER_API_KEY=your-key  # Falls back to embedding API key
```

## Known Issues and Solutions

1. **Model Download**: Tests use sentence-transformers/all-MiniLM-L6-v2 for speed
2. **API Keys**: Mocked in tests using sys.modules patching
3. **Async Fixtures**: Use @pytest_asyncio.fixture for async fixtures
4. **Config Parameters**: Use `data_root` not `db_path`
5. **Rate Limiting**: Partial results are now returned instead of exceptions
6. **LanceDB Concurrency**: Single writer only (`concurrent_writers: int = 1`)
7. **breeze-langdetect**: Requires reinstall if updated (not editable install)
8. **Shebang Detection**: Works correctly with breeze-langdetect
9. **Test File Creation**: Use Write tool instead of echo for proper shebangs
10. **MPS Memory**: Batch size must be 1 for MPS devices to avoid allocation errors
11. **Progress Bars**: Set TQDM_DISABLE=1 to suppress all progress bars
12. **Token Warnings**: Tokenizer warnings are expected for long files before chunking

## Architecture Notes

The chunking implementation:

- Chunks long texts into overlapping segments
- Embeds each chunk separately
- Combines chunk embeddings using weighted average
- Preserves all content instead of truncating

Rate limiting:

- Uses token bucket algorithm
- Tracks in-flight requests properly
- Applies safety margins to avoid API limits
- Returns partial results on failure

Language detection architecture:

- **Single source of truth**: All language detection flows through `ContentDetector.detect_language()`
- **Layered approach**:
  1. `breeze-langdetect` (hyperpolyglot) when available - most comprehensive
  2. `LanguageDetector` with identify + fuzzy matching - good coverage
  3. Extension-based fallbacks - last resort
- **No duplicate mappings**: Language aliases centralized in one place
- **Extensible**: Tree-sitter queries can be loaded from JSON files
- **DRY principle**: Avoided hardcoding language mappings across multiple files

## Key Improvements Summary

### Before:
- **Truncation**: Long files were cut off at 8192 tokens, losing ~80% of content
- **Dict Returns**: Untyped dictionary returns made debugging difficult
- **Hardcoded Mappings**: Language detection scattered across multiple files
- **Basic Chunking**: Character-based splitting without semantic awareness

### After:
- **Proper Chunking**: TextChunker with 16k token chunks preserves all content
- **Type Safety**: Dataclass returns with full IDE support
- **Single Source of Truth**: ContentDetector centralizes all language detection
- **Semantic Awareness**: Tree-sitter integration for natural chunk boundaries
- **Test Coverage**: Comprehensive tests prevent regression

### Next Steps:
1. Add metadata to chunks (node_type, language, parent_context)
2. Implement configurable semantic granularity
3. Integrate nvim-treesitter queries for richer understanding
4. Add incremental chunking to skip unchanged files
