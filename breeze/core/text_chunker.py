"""Text chunking utilities for embedding models.

This module provides semantic code chunking using tree-sitter to split
at logical boundaries like functions and classes.
"""

import logging
from typing import List, Any, Optional, Protocol, Tuple
from dataclasses import dataclass
import numpy as np
import tree_sitter_language_pack
from tree_sitter import Query

from breeze.core.tree_sitter_queries import get_query_for_language

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    """Protocol for tokenizer interface."""
    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> Any:
        """Encode text to tokens."""
        ...
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """Decode tokens back to text."""
        ...


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    max_tokens: int  # Maximum tokens per chunk
    overlap_tokens: int = 128  # Overlap between chunks for context continuity
    reserved_tokens: int = 10  # Reserved for special tokens
    stride_ratio: float = 0.75  # How much to advance for each chunk (1 - overlap ratio)
    

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    total_chunks: int
    estimated_tokens: int


@dataclass
class FileContent:
    """Represents a file's content with its metadata."""
    content: str
    file_path: str
    language: str


@dataclass 
class ChunkedFile:
    """Represents a file that has been chunked."""
    source: FileContent
    chunks: List[TextChunk]
    
    @property
    def total_tokens(self) -> int:
        """Total estimated tokens across all chunks."""
        return sum(chunk.estimated_tokens for chunk in self.chunks)


class TextChunker:
    """Handles semantic text chunking using tree-sitter."""
    
    def __init__(self, 
                 tokenizer: Optional[Tokenizer] = None,
                 config: Optional[ChunkingConfig] = None):
        """Initialize the chunker.
        
        Args:
            tokenizer: Optional tokenizer for accurate token counting
            config: Chunking configuration
        """
        self.tokenizer = tokenizer
        self.config = config or ChunkingConfig(max_tokens=8192)
        self._parsers = {}
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if self.tokenizer:
            try:
                # For long texts, count tokens in chunks to avoid tokenizer warnings
                # about exceeding max sequence length
                chunk_size = 8000  # Safe size under most tokenizer limits
                
                if len(text) > chunk_size * 4:  # Roughly 32k chars
                    # Count tokens in chunks
                    total_tokens = 0
                    for i in range(0, len(text), chunk_size * 4):
                        chunk = text[i:i + chunk_size * 4]
                        encoded = self.tokenizer.encode(chunk, add_special_tokens=False)
                        if hasattr(encoded, 'ids'):
                            total_tokens += len(encoded.ids)
                        else:
                            total_tokens += len(encoded)
                    # Add tokens for special tokens at start/end
                    return total_tokens + 2
                else:
                    # Short text - encode directly
                    encoded = self.tokenizer.encode(text, add_special_tokens=True)
                    if hasattr(encoded, 'ids'):
                        return len(encoded.ids)
                    elif isinstance(encoded, list):
                        return len(encoded)
                    else:
                        return len(encoded)
            except Exception as e:
                logger.debug(f"Tokenizer encoding failed: {e}, using fallback")
        
        # Fallback: estimate ~4 characters per token for code
        return max(1, len(text) // 4)
    
    def chunk_file(self, file_content: FileContent) -> ChunkedFile:
        """Split a file into semantic chunks using tree-sitter.
        
        Args:
            file_content: File content with metadata
            
        Returns:
            ChunkedFile with chunks
        """
        # Trim and check for empty content
        trimmed_content = file_content.content.strip()
        if len(trimmed_content) == 0:
            return ChunkedFile(source=file_content, chunks=[])
        
        # Try semantic chunking with the file's language
        chunks = self._chunk_semantic(trimmed_content, file_content.language)
        
        # Fall back to regular chunking if semantic fails
        if not chunks:
            chunks = self._chunk_regular(trimmed_content)
        
        return ChunkedFile(source=file_content, chunks=chunks)
    
    def chunk_files(self, file_contents: List[FileContent]) -> List[ChunkedFile]:
        """Chunk multiple files.
        
        Args:
            file_contents: List of file contents with metadata
            
        Returns:
            List of chunked files
        """
        return [self.chunk_file(fc) for fc in file_contents]
    
    def _chunk_semantic(self, text: str, language: str) -> List[TextChunk]:
        """Chunk text using tree-sitter semantic boundaries."""
        parser = self._get_parser(language)
        if not parser:
            return []
        
        try:
            # Parse the code
            tree = parser.parse(text.encode())
            
            # Get semantic units (functions, classes, etc.)
            units = self._extract_semantic_units(tree, text, language)
            if not units:
                return []
            
            # Group units into chunks that fit token limit
            return self._group_units_into_chunks(units, text)
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}")
            return []
    
    def _get_parser(self, language: str):
        """Get or create parser for language."""
        if language not in self._parsers:
            try:
                self._parsers[language] = tree_sitter_language_pack.get_parser(language)
            except Exception:
                self._parsers[language] = None
        return self._parsers[language]
    
    def _extract_semantic_units(self, tree, text: str, language: str) -> List[dict]:
        """Extract functions, classes, etc. from the parse tree."""
        units = []
        
        try:
            lang = tree_sitter_language_pack.get_language(language)
            query_pattern = get_query_for_language(language)
            query = Query(lang, query_pattern)
            
            # Find all semantic units
            matches = query.matches(tree.root_node)
            
            for _, captures in matches:
                for capture_name, nodes in captures.items():
                    for node in nodes:
                        units.append({
                            'type': capture_name,
                            'start': node.start_byte,
                            'end': node.end_byte,
                            'text': text[node.start_byte:node.end_byte]
                        })
            
            # Sort by position
            units.sort(key=lambda u: u['start'])
            return units
            
        except Exception as e:
            logger.debug(f"Failed to extract semantic units: {e}")
            return []
    
    def _group_units_into_chunks(self, units: List[dict], full_text: str) -> List[TextChunk]:
        """Group semantic units into chunks respecting token limits."""
        chunks = []
        current_units = []
        current_tokens = 0
        
        for unit in units:
            unit_tokens = self.estimate_tokens(unit['text'])
            
            # If single unit is too large, split it
            if unit_tokens > self.config.max_tokens:
                # Flush current group
                if current_units:
                    chunks.append(self._units_to_chunk(current_units, full_text, len(chunks)))
                    current_units = []
                    current_tokens = 0
                
                # Split large unit
                unit_chunks = self._chunk_regular(unit['text'])
                for i, chunk in enumerate(unit_chunks):
                    # Adjust positions relative to full text
                    chunk.start_char += unit['start']
                    chunk.end_char = chunk.start_char + len(chunk.text)
                    chunk.chunk_index = len(chunks) + i
                chunks.extend(unit_chunks)
                continue
            
            # Check if adding this unit exceeds limit
            if current_units and current_tokens + unit_tokens > self.config.max_tokens:
                # Create chunk from current units
                chunks.append(self._units_to_chunk(current_units, full_text, len(chunks)))
                current_units = []
                current_tokens = 0
            
            # Add unit to current group
            current_units.append(unit)
            current_tokens += unit_tokens
        
        # Flush remaining units
        if current_units:
            chunks.append(self._units_to_chunk(current_units, full_text, len(chunks)))
        
        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _units_to_chunk(self, units: List[dict], full_text: str, chunk_index: int) -> TextChunk:
        """Convert a group of semantic units to a chunk."""
        start = min(u['start'] for u in units)
        end = max(u['end'] for u in units)
        text = full_text[start:end]
        
        return TextChunk(
            text=text,
            start_char=start,
            end_char=end,
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated
            estimated_tokens=self.estimate_tokens(text)
        )
    
    def _chunk_regular(self, text: str) -> List[TextChunk]:
        """Regular chunking when semantic chunking isn't available."""
        effective_max = self.config.max_tokens - self.config.reserved_tokens
        
        # If text fits in one chunk, return as is
        total_tokens = self.estimate_tokens(text)
        if total_tokens <= effective_max:
            return [TextChunk(
                text=text,
                start_char=0,
                end_char=len(text),
                chunk_index=0,
                total_chunks=1,
                estimated_tokens=total_tokens
            )]
        
        if self.tokenizer:
            chunks = self._chunk_with_tokenizer(text, effective_max)
        else:
            chunks = self._chunk_by_characters(text, effective_max)
            
        return chunks
    
    def _chunk_with_tokenizer(self, text: str, max_tokens: int) -> List[TextChunk]:
        """Chunk using tokenizer for precise token boundaries."""
        chunks = []
        
        # For very long texts, we need to encode in chunks to avoid tokenizer warnings
        # We'll process the text in windows and stitch the token IDs together
        try:
            window_size = 8000 * 4  # ~8000 tokens worth of characters
            
            if len(text) > window_size:
                # Encode in chunks and concatenate token IDs
                token_ids = []
                for i in range(0, len(text), window_size):
                    chunk = text[i:i + window_size]
                    encoded = self.tokenizer.encode(chunk, add_special_tokens=False)
                    if hasattr(encoded, 'ids'):
                        token_ids.extend(encoded.ids)
                    else:
                        token_ids.extend(encoded)
            else:
                # Short text - encode directly
                full_encoded = self.tokenizer.encode(text, add_special_tokens=False)
                if hasattr(full_encoded, 'ids'):
                    token_ids = full_encoded.ids
                else:
                    token_ids = full_encoded
                
            total_tokens = len(token_ids)
            
            # Calculate stride
            stride = int(max_tokens * self.config.stride_ratio)
            
            # Create chunks
            start_idx = 0
            chunk_index = 0
            
            while start_idx < total_tokens:
                # Calculate end index
                end_idx = min(start_idx + max_tokens, total_tokens)
                
                # Get chunk tokens
                chunk_tokens = token_ids[start_idx:end_idx]
                
                # Decode back to text
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                
                # Try to find the character positions in original text
                # This is approximate due to tokenization
                approx_start_char = int(start_idx / total_tokens * len(text))
                approx_end_char = int(end_idx / total_tokens * len(text))
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_char=approx_start_char,
                    end_char=approx_end_char,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will update after
                    estimated_tokens=len(chunk_tokens)
                ))
                
                # Move to next chunk with overlap
                start_idx += stride
                chunk_index += 1
                
                # Break if we've processed everything
                if end_idx >= total_tokens:
                    break
                    
        except Exception as e:
            logger.warning(f"Tokenizer chunking failed: {e}, falling back to character chunking")
            return self._chunk_by_characters(text, max_tokens)
        
        # Update total chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
            
        return chunks
    
    def _chunk_by_characters(self, text: str, max_tokens: int) -> List[TextChunk]:
        """Chunk by character count estimation."""
        chunks = []
        
        # Estimate characters per chunk (4 chars per token)
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        
        # Calculate stride in characters
        stride_chars = int(max_chars * self.config.stride_ratio)
        
        # Create chunks
        start_char = 0
        chunk_index = 0
        
        while start_char < len(text):
            # Calculate end position
            end_char = min(start_char + max_chars, len(text))
            
            # Try to break at a natural boundary (newline, space)
            if end_char < len(text):
                # Look for newline first, then space
                for sep in ['\n', ' ', '.', ',']:
                    last_sep = text.rfind(sep, start_char, end_char)
                    if last_sep > start_char + max_chars // 2:  # At least halfway through
                        end_char = last_sep + 1
                        break
            
            chunk_text = text[start_char:end_char]
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                chunk_index=chunk_index,
                total_chunks=0,  # Will update after
                estimated_tokens=self.estimate_tokens(chunk_text)
            ))
            
            # Move to next chunk
            start_char += stride_chars
            chunk_index += 1
            
            # Break if we've processed everything
            if end_char >= len(text):
                break
        
        # Update total chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
            
        return chunks
    
    def combine_chunk_embeddings(self, 
                                chunk_embeddings: List[np.ndarray], 
                                chunks: List[TextChunk],
                                method: str = "weighted_average") -> np.ndarray:
        """Combine embeddings from multiple chunks into a single embedding.
        
        Args:
            chunk_embeddings: List of embeddings for each chunk
            chunks: List of chunk metadata
            method: Combination method ('average', 'weighted_average', 'first', 'last')
            
        Returns:
            Combined embedding
        """
        if not chunk_embeddings:
            raise ValueError("No embeddings to combine")
            
        if len(chunk_embeddings) == 1:
            return chunk_embeddings[0]
            
        if method == "average":
            return np.mean(chunk_embeddings, axis=0)
            
        elif method == "weighted_average":
            # Weight by token count
            weights = np.array([chunk.estimated_tokens for chunk in chunks])
            weights = weights / weights.sum()
            
            weighted_sum = np.zeros_like(chunk_embeddings[0])
            for embedding, weight in zip(chunk_embeddings, weights):
                weighted_sum += embedding * weight
                
            return weighted_sum
            
        elif method == "first":
            return chunk_embeddings[0]
            
        elif method == "last":
            return chunk_embeddings[-1]
            
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    def combine_file_embeddings(self,
                               chunked_file: ChunkedFile,
                               chunk_embeddings: List[np.ndarray],
                               method: str = "weighted_average") -> np.ndarray:
        """Combine embeddings for all chunks of a file.
        
        Args:
            chunked_file: The chunked file with metadata
            chunk_embeddings: Embeddings for each chunk in the file
            method: Combination method
            
        Returns:
            Single embedding for the entire file
        """
        if len(chunk_embeddings) != len(chunked_file.chunks):
            raise ValueError(f"Mismatch: {len(chunk_embeddings)} embeddings for {len(chunked_file.chunks)} chunks")
        
        return self.combine_chunk_embeddings(chunk_embeddings, chunked_file.chunks, method)


def create_batches_from_chunked_files(chunked_files: List[ChunkedFile], 
                                     batch_size: int = 32) -> List[List[Tuple[int, FileContent, TextChunk]]]:
    """Create batches from chunked files, preserving file metadata with each chunk.
    
    Args:
        chunked_files: List of chunked files
        batch_size: Maximum chunks per batch
        
    Returns:
        List of batches, where each batch contains (file_index, file_content, chunk) tuples
    """
    batches = []
    current_batch = []
    
    for file_idx, chunked_file in enumerate(chunked_files):
        for chunk in chunked_file.chunks:
            current_batch.append((file_idx, chunked_file.source, chunk))
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
    
    if current_batch:
        batches.append(current_batch)
        
    return batches