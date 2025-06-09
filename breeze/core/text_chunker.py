"""Text chunking utilities for embedding models.

This module provides semantic code chunking using tree-sitter to split
at logical boundaries like functions and classes.
"""

import logging
from typing import List, Any, Optional, Protocol, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

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
    chunk_size: int  # Target chunk size in tokens
    model_max_tokens: int  # Model's maximum context length
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


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, tokenizer: Tokenizer):
        """Initialize the strategy.
        
        Args:
            tokenizer: Tokenizer for accurate token counting
        """
        self.tokenizer = tokenizer
        self._parsers = {}
    
    @abstractmethod
    def chunk(self, text: str, language: str, config: ChunkingConfig) -> List[TextChunk]:
        """Chunk the given text.
        
        Args:
            text: Text to chunk
            language: Language of the text (for semantic chunking)
            config: Chunking configuration
            
        Returns:
            List of text chunks
        """
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        try:
            # For long texts, count tokens in chunks to avoid tokenizer warnings
            chunk_size = 8000  # Safe size under most tokenizer limits
            
            if len(text) > chunk_size * 4:  # Roughly 32k chars
                total_tokens = 0
                for i in range(0, len(text), chunk_size * 4):
                    chunk = text[i:i + chunk_size * 4]
                    encoded = self.tokenizer.encode(chunk, add_special_tokens=False)
                    if hasattr(encoded, 'ids'):
                        total_tokens += len(encoded.ids)
                    else:
                        total_tokens += len(encoded)
                return total_tokens + 2
            else:
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


class SimpleChunkingStrategy(ChunkingStrategy):
    """Simple character-based chunking strategy."""
    
    def chunk(self, text: str, language: str, config: ChunkingConfig) -> List[TextChunk]:
        """Chunk text using tokenizer-based boundaries.
        
        Args:
            text: Text to chunk
            language: Language of the text (not used in this strategy)
            config: Chunking configuration
            
        Returns:
            List of text chunks
        """
        _ = language  # Not used in simple chunking
        # Ensure chunks don't exceed model's max tokens
        effective_chunk_size = min(config.chunk_size, config.model_max_tokens - config.reserved_tokens)
        
        # If text fits in one chunk, return as is
        total_tokens = self.estimate_tokens(text)
        if total_tokens <= effective_chunk_size:
            return [TextChunk(
                text=text,
                start_char=0,
                end_char=len(text),
                chunk_index=0,
                total_chunks=1,
                estimated_tokens=total_tokens
            )]
        
        return self._chunk_with_tokenizer(text, effective_chunk_size, config)
    
    def _chunk_with_tokenizer(self, text: str, max_tokens: int, config: ChunkingConfig) -> List[TextChunk]:
        """Chunk using tokenizer for precise token boundaries."""
        chunks = []
        
        try:
            # Encode the text, respecting model limits
            # We'll process in windows if needed to avoid tokenizer warnings
            window_size = min(config.model_max_tokens * 4, len(text))  # chars estimate
            
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
            stride = int(max_tokens * config.stride_ratio)
            
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
            logger.warning(f"Tokenizer chunking failed: {e}")
            raise
        
        # Update total chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
            
        return chunks


class TreeSitterChunkingStrategy(ChunkingStrategy):
    """Tree-sitter based semantic chunking strategy (placeholder for future implementation)."""
    
    def chunk(self, text: str, language: str, config: ChunkingConfig) -> List[TextChunk]:
        """Chunk text using tree-sitter semantic boundaries.
        
        Currently falls back to simple chunking. Will be replaced with proper
        tree-sitter implementation later.
        
        Args:
            text: Text to chunk
            language: Language of the text
            config: Chunking configuration
            
        Returns:
            List of text chunks
        """
        # For now, just use simple chunking
        simple_strategy = SimpleChunkingStrategy(self.tokenizer)
        return simple_strategy.chunk(text, language, config)


class TextChunker:
    """Handles text chunking using configurable strategies."""
    
    def __init__(self, 
                 strategy: ChunkingStrategy,
                 config: Optional[ChunkingConfig] = None):
        """Initialize the chunker.
        
        Args:
            strategy: Chunking strategy to use
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig(chunk_size=2048, model_max_tokens=8192)
        self.strategy = strategy
    
    def chunk_file(self, file_content: FileContent) -> ChunkedFile:
        """Split a file into chunks using the configured strategy.
        
        Args:
            file_content: File content with metadata
            
        Returns:
            ChunkedFile with chunks
        """
        # Trim and check for empty content
        trimmed_content = file_content.content.strip()
        if len(trimmed_content) == 0:
            return ChunkedFile(source=file_content, chunks=[])
        
        # Use the strategy to chunk the content
        chunks = self.strategy.chunk(trimmed_content, file_content.language, self.config)
        
        return ChunkedFile(source=file_content, chunks=chunks)
    
    def chunk_files(self, file_contents: List[FileContent]) -> List[ChunkedFile]:
        """Chunk multiple files.
        
        Args:
            file_contents: List of file contents with metadata
            
        Returns:
            List of chunked files
        """
        return [self.chunk_file(fc) for fc in file_contents]
    
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
            return np.asarray(chunk_embeddings[0])
            
        if method == "average":
            # Ensure all embeddings are numpy arrays
            embeddings_array = np.array([np.asarray(emb) for emb in chunk_embeddings])
            return np.mean(embeddings_array, axis=0)
            
        elif method == "weighted_average":
            # Weight by token count
            weights = np.array([chunk.estimated_tokens for chunk in chunks], dtype=np.float64)
            weights = weights / weights.sum()
            
            # Ensure all embeddings are numpy arrays and check dimensions
            embeddings_array = [np.asarray(emb) for emb in chunk_embeddings]
            
            # Validate all embeddings have the same shape
            first_shape = embeddings_array[0].shape
            for i, emb in enumerate(embeddings_array[1:], 1):
                if emb.shape != first_shape:
                    # Log all shapes for debugging
                    all_shapes = [e.shape for e in embeddings_array]
                    raise ValueError(
                        f"Embedding dimension mismatch: chunk 0 has shape {first_shape}, "
                        f"but chunk {i} has shape {emb.shape}. "
                        f"All shapes: {all_shapes}. All embeddings must have the same dimension."
                    )
            
            weighted_sum = np.zeros_like(embeddings_array[0])
            for embedding, weight in zip(embeddings_array, weights):
                weighted_sum += embedding * float(weight)
                
            return weighted_sum
            
        elif method == "first":
            return np.asarray(chunk_embeddings[0])
            
        elif method == "last":
            return np.asarray(chunk_embeddings[-1])
            
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