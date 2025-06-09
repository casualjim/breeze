# nvim-treesitter Integration Plan

## Overview

This document outlines the plan to integrate nvim-treesitter's query files into Breeze to enhance semantic code understanding. The integration will leverage nvim-treesitter's battle-tested queries, particularly `locals.scm`, to provide richer metadata about code structure and improve chunking quality.

## Goals

1. **Enhanced Semantic Understanding**: Use nvim-treesitter's `locals.scm` queries to identify scopes, definitions, and references
2. **Improved Chunk Boundaries**: Leverage `@local.scope` captures for more natural code divisions
3. **Rich Chunk Metadata**: Include semantic information (node_type, scope, definitions) with each chunk
4. **Community-Maintained Queries**: Benefit from nvim-treesitter's extensive language support (100+ languages)
5. **Better Search Relevance**: Enable filtering and ranking by semantic type

## Background

### Current State

- Breeze uses basic tree-sitter queries focused on top-level constructs (functions, classes)
- Limited semantic understanding of code structure
- No tracking of variable scopes or symbol definitions
- Hardcoded queries for each language

### nvim-treesitter Query Structure

```text
nvim-treesitter/queries/[language]/
├── highlights.scm   # Syntax highlighting
├── locals.scm       # Scopes, definitions, references
├── folds.scm        # Code folding regions
├── indents.scm      # Indentation rules
└── injections.scm   # Language embedding
```

### locals.scm Captures

- `@local.scope` - Code blocks creating new variable scopes
- `@local.definition` - Variable/function/class definitions
- `@local.reference` - Usage of defined symbols
- Additional semantic captures for richer understanding

## Implementation Plan

### Phase 1: Query Infrastructure (Week 1)

#### 1.1 Download nvim-treesitter Queries

```python
# Create scripts/download_nvim_queries.py
- Clone nvim-treesitter repository
- Extract queries/ directory
- Filter to supported languages
- Package into breeze/queries/
```

#### 1.2 Extend QueryManager

```python
# breeze/core/tree_sitter_queries.py
class QueryManager:
    def load_scm_file(self, language: str, query_type: str) -> str:
        """Load a .scm query file for a language."""
        
    def get_locals_query(self, language: str) -> str:
        """Get locals.scm query with fallback."""
        
    def get_highlights_query(self, language: str) -> str:
        """Get highlights.scm query with fallback."""
```

#### 1.3 Query Caching System

```python
# Add to QueryManager
- Cache parsed queries in memory
- Support query composition/merging
- Handle query versioning
```

### Phase 2: Semantic Analysis (Week 1-2)

#### 2.1 Scope Analyzer

```python
# breeze/core/scope_analyzer.py
@dataclass
class ScopeInfo:
    type: str  # function, class, block, etc.
    name: Optional[str]
    start_byte: int
    end_byte: int
    parent: Optional['ScopeInfo']
    definitions: List[Definition]
    
class ScopeAnalyzer:
    def analyze_scopes(self, tree, language: str) -> List[ScopeInfo]:
        """Extract scope hierarchy using locals.scm."""
```

#### 2.2 Symbol Extractor

```python
@dataclass
class Definition:
    name: str
    type: str  # variable, function, class, etc.
    scope: ScopeInfo
    position: Tuple[int, int]

@dataclass  
class Reference:
    name: str
    definition: Optional[Definition]
    position: Tuple[int, int]
```

### Phase 3: Enhanced TextChunker (Week 2)

#### 3.1 Update Data Models

```python
# breeze/core/text_chunker.py
@dataclass
class ChunkMetadata:
    node_type: str  # function, class, method, etc.
    node_name: Optional[str]
    language: str
    parent_context: Optional[str]  # e.g., "class MyClass"
    chunking_method: str  # semantic, character
    scope_path: List[str]  # ["module", "class MyClass", "method foo"]
    definitions: List[str]  # Symbol names defined in chunk
    
@dataclass
class TextChunk:
    # existing fields...
    metadata: Optional[ChunkMetadata]
```

#### 3.2 Integrate Scope Analysis

```python
class TextChunker:
    def _chunk_semantic(self, text: str, language: str) -> List[TextChunk]:
        """Enhanced semantic chunking with scope analysis."""
        # Parse with tree-sitter
        # Extract scopes using locals.scm
        # Group by semantic boundaries
        # Add rich metadata
```

### Phase 4: Testing & Migration (Week 2-3)

#### 4.1 Test Suite

```python
# breeze/tests/test_nvim_queries.py
- Test query loading for multiple languages
- Verify scope extraction accuracy
- Test metadata population
- Performance benchmarks

# breeze/tests/test_enhanced_chunking.py  
- Test semantic boundary detection
- Verify metadata accuracy
- Test fallback behavior
```

#### 4.2 Migration Strategy

1. Add feature flag for nvim-treesitter queries
2. Run side-by-side comparison with existing queries
3. Gradual rollout by language
4. Monitor performance and accuracy

### Phase 5: Integration & Optimization (Week 3)

#### 5.1 Search Integration

- Update search to use chunk metadata
- Enable filtering by node_type
- Implement scope-aware ranking

#### 5.2 Performance Optimization

- Implement query result caching
- Optimize scope traversal
- Add incremental parsing support

## Technical Details

### Query File Structure

```scheme
; Example locals.scm for Python
(function_definition) @local.scope
(class_definition) @local.scope

(function_definition
  name: (identifier) @local.definition.function)
  
(assignment
  left: (identifier) @local.definition.var)
  
(identifier) @local.reference
```

### Integration Architecture

```
┌─────────────────────┐
│   TextChunker       │
├─────────────────────┤
│  ScopeAnalyzer      │
├─────────────────────┤
│  QueryManager       │
├─────────────────────┤
│ nvim-treesitter     │
│     queries/        │
└─────────────────────┘
```

### Backwards Compatibility

- Maintain existing CORE_LANGUAGE_QUERIES as fallback
- Support gradual migration per language
- Preserve existing API contracts
- Feature flag for enabling nvim queries

## Success Metrics

1. **Coverage**: Support for 50+ languages with nvim-treesitter queries
2. **Quality**: 90%+ accuracy in scope/definition detection
3. **Performance**: <10% overhead vs current implementation
4. **Adoption**: Positive feedback from users on improved search relevance

## Risks & Mitigations

### Risk: Query Compatibility

- **Issue**: nvim-treesitter queries may use Neovim-specific features
- **Mitigation**: Test queries thoroughly, implement compatibility layer

### Risk: Performance Impact

- **Issue**: Complex queries may slow down indexing
- **Mitigation**: Aggressive caching, query optimization, incremental parsing

### Risk: Maintenance Burden

- **Issue**: Keeping queries up-to-date with nvim-treesitter
- **Mitigation**: Automated sync scripts, version pinning

## Timeline

- **Week 1**: Query infrastructure and loading system
- **Week 1-2**: Scope analysis and symbol extraction  
- **Week 2**: Enhanced TextChunker with metadata
- **Week 2-3**: Testing and migration
- **Week 3**: Integration and optimization

## Future Enhancements

1. **Additional Query Types**: Integrate highlights.scm for better semantic understanding
2. **Incremental Indexing**: Use scope information to update only changed code
3. **Cross-File Analysis**: Track symbol definitions across files
4. **Language Server Protocol**: Expose semantic information via LSP

## References

- [nvim-treesitter repository](https://github.com/nvim-treesitter/nvim-treesitter)
- [Tree-sitter query syntax](https://tree-sitter.github.io/tree-sitter/using-parsers#query-syntax)
- [nvim-treesitter query documentation](https://github.com/nvim-treesitter/nvim-treesitter#adding-queries)
