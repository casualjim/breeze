"""Smart code snippet extraction using tree-sitter for semantic awareness."""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

from breeze.core.tree_sitter_queries import get_query_for_language

logger = logging.getLogger(__name__)


@dataclass
class SnippetConfig:
    """Configuration for snippet extraction."""
    # Maximum length for a complete snippet
    max_snippet_length: int = 1500
    # Lines to show before/after for context
    context_lines: int = 5
    # Maximum lines for a complete function/class
    max_complete_lines: int = 50
    # Lines to show at start/end for large functions
    head_lines: int = 10
    tail_lines: int = 5


class TreeSitterSnippetExtractor:
    """Extract semantically aware code snippets using tree-sitter."""
    
    def __init__(self, config: Optional[SnippetConfig] = None):
        self.config = config or SnippetConfig()
        self._parsers: Dict[str, Any] = {}
        self._initialized = False
        self._ts_languages = None
    
    def initialize(self):
        """Initialize tree-sitter languages and parsers."""
        if self._initialized:
            return
            
        try:
            import tree_sitter_language_pack
            self._ts_languages = tree_sitter_language_pack
            self._initialized = True
            logger.info("Tree-sitter language pack loaded successfully")
        except ImportError:
            logger.warning(
                "tree-sitter-language-pack not installed. "
                "Install with: pip install tree-sitter-language-pack"
            )
            self._initialized = False
    
    def _get_parser(self, language: str):
        """Get or create a parser for a language."""
        if not self._initialized:
            return None
            
        if language not in self._parsers:
            try:
                self._parsers[language] = self._ts_languages.get_parser(language)
            except Exception as e:
                logger.warning(f"Failed to get tree-sitter parser for {language}: {e}")
                self._parsers[language] = None
        
        return self._parsers.get(language)
    
    def can_parse_language(self, language: str) -> bool:
        """Check if we can parse a given language."""
        if not self._initialized:
            self.initialize()
        return self._get_parser(language) is not None
    
    def extract_snippet(
        self, 
        content: str, 
        query: str, 
        language: Optional[str] = None
    ) -> str:
        """
        Extract a semantically aware snippet from code content.
        
        Args:
            content: The full file content
            query: The search query
            language: Optional language hint (e.g., 'python', 'javascript')
            
        Returns:
            The extracted snippet
        """
        self.initialize()
        
        # Try tree-sitter based extraction if we have a language
        if language and self._initialized:
            parser = self._get_parser(language)
            if parser:
                try:
                    snippet = self._extract_semantic_snippet(
                        parser, content, query, language
                    )
                    if snippet:
                        return snippet
                except Exception as e:
                    logger.debug(f"Tree-sitter extraction failed: {e}")
        
        # Fallback to simple line-based extraction
        return self._extract_simple_snippet(content, query)
    
    def _extract_semantic_snippet(
        self, 
        parser: Any, 
        content: str, 
        query: str,
        language: str
    ) -> Optional[str]:
        """Extract snippet using tree-sitter parsing."""
        tree = parser.parse(content.encode())
        
        # Find the best matching node
        match_node = self._find_best_match_node(tree, query, content)
        if not match_node:
            return None
        
        # Get the semantic parent using queries
        semantic_parent = self._get_semantic_parent_with_query(match_node, language, tree, parser)
        
        # If query-based approach didn't work, fall back to heuristic
        if not semantic_parent:
            semantic_parent = self._get_semantic_parent(match_node, language)
        
        # Extract semantic unit content
        start_byte = semantic_parent.start_byte
        end_byte = semantic_parent.end_byte
        unit_content = content[start_byte:end_byte]
        
        # Handle size constraints
        if len(unit_content) <= self.config.max_snippet_length:
            return unit_content
        else:
            # Smart truncation for large units
            match_position = match_node.start_byte - start_byte
            return self._smart_truncate(unit_content, match_position)
    
    def _find_best_match_node(self, tree, query: str, content: str) -> Optional[Any]:
        """Find the tree node that best matches the query."""
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        best_node = None
        best_score = 0
        
        def score_node(node, text):
            """Score how well a node matches the query."""
            if not text:
                return 0
            
            text_lower = text.lower()
            score = 0
            
            # Exact match is best
            if query_lower in text_lower:
                score += 10
            
            # Count matching terms
            for term in query_terms:
                if term in text_lower:
                    score += 1
            
            # Prefer smaller, more specific nodes
            score -= len(text) / 10000
            
            return score
        
        def traverse(node):
            nonlocal best_node, best_score
            
            # Get node text
            start_byte = node.start_byte
            end_byte = node.end_byte
            node_text = content[start_byte:end_byte]
            
            # Score this node
            score = score_node(node, node_text)
            if score > best_score:
                best_score = score
                best_node = node
            
            # Traverse children
            for child in node.children:
                traverse(child)
        
        traverse(tree.root_node)
        return best_node
    
    def _get_semantic_parent_with_query(self, node: Any, language: str, tree: Any, parser: Any) -> Optional[Any]:
        """Get the semantic parent using tree-sitter queries."""
        try:
            from tree_sitter import Query
            
            # Get the language object
            lang = self._ts_languages.get_language(language)
            
            # Get the query pattern for this language
            query_pattern = get_query_for_language(language)
            
            # Create the query
            query = Query(lang, query_pattern)
            
            # Execute the query - matches returns tuples of (pattern_index, captures_dict)
            matches = query.matches(tree.root_node)
            
            # Find the smallest semantic unit containing our match node
            best_parent = None
            best_size = float('inf')
            
            for _, captures_dict in matches:
                # Iterate through all captured nodes in this match
                for _, nodes in captures_dict.items():
                    for capture_node in nodes:
                        # Check if this capture contains our match node
                        if (capture_node.start_byte <= node.start_byte and 
                            capture_node.end_byte >= node.end_byte):
                            
                            # Calculate size
                            size = capture_node.end_byte - capture_node.start_byte
                            
                            # Keep the smallest containing semantic unit
                            if size < best_size:
                                best_parent = capture_node
                                best_size = size
            
            return best_parent
            
        except Exception as e:
            logger.debug(f"Query-based semantic parent extraction failed: {e}")
            return None
    
    def _get_semantic_parent(self, node, language: str) -> Any:
        """Get the semantic parent (function/class) of a node using heuristics."""
        # Common semantic node types across languages
        semantic_types = [
            'function', 'method', 'class', 'interface', 'struct',
            'enum', 'trait', 'impl', 'module', 'namespace',
            'declaration', 'definition', 'item'
        ]
        
        current = node
        while current:
            # Check if node type contains any semantic keywords
            node_type = current.type.lower()
            if any(t in node_type for t in semantic_types):
                return current
            current = current.parent
        
        return node
    
    def _smart_truncate(self, content: str, match_position: int) -> str:
        """Intelligently truncate large code blocks."""
        lines = content.split('\n')
        
        if len(lines) <= self.config.max_complete_lines:
            return content
        
        # Calculate match line (0-based)
        pre_content = content[:match_position]
        match_line = len(pre_content.split('\n')) - 1
        
        # For very large blocks, show head + match context + tail
        result_lines = []
        
        # Head
        result_lines.extend(lines[:self.config.head_lines])
        
        # Match context (if not in head)
        if match_line > self.config.head_lines + self.config.context_lines:
            omitted_count = match_line - self.config.head_lines - self.config.context_lines
            result_lines.append(f'    ... ({omitted_count} lines omitted) ...')
            
            start = max(0, match_line - self.config.context_lines)
            end = min(len(lines), match_line + self.config.context_lines + 1)
            result_lines.extend(lines[start:end])
        else:
            # Match is within head, just show up to context after match
            end = min(len(lines), match_line + self.config.context_lines + 1)
            if end > self.config.head_lines:
                result_lines.extend(lines[self.config.head_lines:end])
        
        # Tail (if not already shown)
        last_shown_line = match_line + self.config.context_lines
        if last_shown_line < len(lines) - self.config.tail_lines - 1:
            lines_to_tail = len(lines) - last_shown_line - self.config.tail_lines - 1
            result_lines.append(f'    ... ({lines_to_tail} lines omitted) ...')
            result_lines.extend(lines[-self.config.tail_lines:])
        elif last_shown_line < len(lines) - 1:
            # Just show the remaining lines
            result_lines.extend(lines[last_shown_line + 1:])
        
        return '\n'.join(result_lines)
    
    def _extract_simple_snippet(self, content: str, query: str) -> str:
        """Fallback simple extraction method."""
        lines = content.split('\n')
        query_lower = query.lower()
        
        # Find the most relevant lines
        best_score = 0
        best_idx = 0
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Exact match gets highest score
            if query_lower in line_lower:
                score = 10
            else:
                # Simple scoring based on query terms
                score = sum(term in line_lower for term in query_lower.split())
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        # Extract context around the best match
        start_idx = max(0, best_idx - self.config.context_lines)
        end_idx = min(len(lines), best_idx + self.config.context_lines + 1)
        
        snippet_lines = lines[start_idx:end_idx]
        snippet = '\n'.join(snippet_lines)
        
        # Truncate if too long
        if len(snippet) > self.config.max_snippet_length:
            snippet = snippet[:self.config.max_snippet_length] + "..."
        
        return snippet