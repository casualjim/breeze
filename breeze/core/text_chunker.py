"""Text chunking utilities for embedding models.

This module provides utilities to split long texts into overlapping chunks
that fit within model token limits, ensuring no content is lost.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Protocol
from dataclasses import dataclass
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


class TextChunker:
    """Handles text chunking for embedding models."""
    
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
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if self.tokenizer:
            try:
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
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Calculate effective max tokens per chunk
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
        
        chunks = []
        
        if self.tokenizer:
            # Use tokenizer for precise chunking
            chunks = self._chunk_with_tokenizer(text, effective_max)
        else:
            # Use character-based chunking
            chunks = self._chunk_by_characters(text, effective_max)
            
        return chunks
    
    def _chunk_with_tokenizer(self, text: str, max_tokens: int) -> List[TextChunk]:
        """Chunk using tokenizer for precise token boundaries."""
        chunks = []
        
        # Encode full text
        try:
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


def create_batches_from_chunks(chunks_per_text: List[List[TextChunk]], 
                              batch_size: int = 32) -> List[List[Tuple[int, TextChunk]]]:
    """Create batches from chunks, preserving which text each chunk came from.
    
    Args:
        chunks_per_text: List of chunk lists, one per input text
        batch_size: Maximum chunks per batch
        
    Returns:
        List of batches, where each batch contains (text_index, chunk) tuples
    """
    batches = []
    current_batch = []
    
    for text_idx, chunks in enumerate(chunks_per_text):
        for chunk in chunks:
            current_batch.append((text_idx, chunk))
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
    
    if current_batch:
        batches.append(current_batch)
        
    return batches