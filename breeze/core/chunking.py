"""Enhanced text chunking with support for different embedding model tokenizers."""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkingStrategy:
    """Configuration for text chunking."""
    max_tokens: int
    overlap_ratio: float = 0.1  # 10% overlap between chunks
    chunk_by: str = "tokens"  # "tokens" or "sentences"
    combine_method: str = "weighted_average"  # How to combine chunk embeddings
    

@dataclass 
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    start_idx: int  # Start position in original text
    end_idx: int    # End position in original text
    token_count: int
    chunk_idx: int
    total_chunks: int


class ModelAwareChunker:
    """Chunker that adapts to different embedding model types and their tokenizers."""
    
    def __init__(self, model_name: str, model_type: str = None):
        """Initialize chunker for a specific model.
        
        Args:
            model_name: Name of the embedding model
            model_type: Type of model ("voyage", "sentence-transformers", "gemini")
        """
        self.model_name = model_name
        self.model_type = model_type or self._infer_model_type(model_name)
        self.tokenizer = None
        self.strategy = self._get_default_strategy()
        
        # Try to load appropriate tokenizer
        self._load_tokenizer()
        
    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from name."""
        if "voyage" in model_name.lower():
            return "voyage"
        elif model_name.startswith("models/"):
            return "gemini"
        else:
            return "sentence-transformers"
            
    def _get_default_strategy(self) -> ChunkingStrategy:
        """Get default chunking strategy for model type."""
        if self.model_type == "voyage":
            # Voyage has 120k token limit per batch, but we chunk smaller
            return ChunkingStrategy(
                max_tokens=16000,  # Conservative to leave room
                overlap_ratio=0.1,
                chunk_by="tokens",
                combine_method="weighted_average"
            )
        elif self.model_type == "gemini":
            # Gemini models - need to check their limits
            return ChunkingStrategy(
                max_tokens=8000,
                overlap_ratio=0.1, 
                chunk_by="tokens",
                combine_method="weighted_average"
            )
        else:
            # Local models - respect their actual limits
            return ChunkingStrategy(
                max_tokens=8192,  # Will be updated based on actual model
                overlap_ratio=0.1,
                chunk_by="tokens",
                combine_method="weighted_average"
            )
            
    def _load_tokenizer(self):
        """Load the appropriate tokenizer for the model."""
        try:
            if self.model_type == "voyage":
                self._load_voyage_tokenizer()
            elif self.model_type == "sentence-transformers":
                self._load_sentence_transformer_tokenizer()
            elif self.model_type == "gemini":
                self._load_gemini_tokenizer()
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {self.model_name}: {e}")
            logger.info("Will use character-based estimation")
            
    def _load_voyage_tokenizer(self):
        """Load Voyage tokenizer from HuggingFace."""
        try:
            from transformers import AutoTokenizer
            
            # Map model names to tokenizer names
            tokenizer_map = {
                "voyage-code-3": "voyageai/voyage-code-3",
                "voyage-code-2": "voyageai/voyage-code-2",
                "voyage-3": "voyageai/voyage-3",
                "voyage-3-lite": "voyageai/voyage-3-lite",
            }
            
            tokenizer_name = tokenizer_map.get(self.model_name, "voyageai/voyage-code-2")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.debug(f"Loaded Voyage tokenizer: {tokenizer_name}")
            
            # Update max tokens based on model
            if self.model_name == "voyage-code-3" and hasattr(self.tokenizer, 'model_max_length'):
                # voyage-code-3 supports 16k context
                if self.tokenizer.model_max_length and self.tokenizer.model_max_length < 1000000:
                    self.strategy.max_tokens = min(16000, self.tokenizer.model_max_length)
                    
        except ImportError:
            logger.warning("transformers package not installed for Voyage models")
            
    def _load_sentence_transformer_tokenizer(self):
        """Load tokenizer for sentence-transformer models."""
        try:
            from transformers import AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Update strategy with actual model limits
            if hasattr(self.tokenizer, 'model_max_length'):
                if self.tokenizer.model_max_length < 1000000:  # Sanity check
                    self.strategy.max_tokens = self.tokenizer.model_max_length
                    logger.debug(f"Set max tokens to {self.strategy.max_tokens} based on tokenizer")
                    
        except ImportError:
            logger.warning("transformers package not installed")
        except Exception as e:
            logger.warning(f"Could not load tokenizer for {self.model_name}: {e}")
            
    def _load_gemini_tokenizer(self):
        """Load tokenizer for Gemini models."""
        # Gemini uses a different approach - they count tokens server-side
        # For now, we'll use estimation
        logger.info("Gemini models use server-side token counting, using estimation")
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self.tokenizer:
            try:
                # Both Voyage and sentence-transformers now use AutoTokenizer
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                return len(encoded)
            except Exception as e:
                logger.debug(f"Token encoding failed: {e}, using fallback")
                
        # Fallback estimation
        if self.model_type == "voyage":
            # Voyage models: ~3.5 chars per token for code
            return max(1, int(len(text) / 3.5))
        else:
            # Other models: ~4 chars per token  
            return max(1, int(len(text) / 4))
            
    def chunk_texts(self, texts: List[str]) -> List[List[TextChunk]]:
        """Chunk multiple texts, returning chunks grouped by original text.
        
        Args:
            texts: List of texts to chunk
            
        Returns:
            List of chunk lists, one per input text
        """
        return [self.chunk_single_text(text) for text in texts]
        
    def chunk_single_text(self, text: str) -> List[TextChunk]:
        """Chunk a single text into overlapping segments.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Estimate total tokens
        total_tokens = self.estimate_tokens(text)
        
        # If it fits in one chunk, return as-is
        if total_tokens <= self.strategy.max_tokens:
            return [TextChunk(
                text=text,
                start_idx=0,
                end_idx=len(text),
                token_count=total_tokens,
                chunk_idx=0,
                total_chunks=1
            )]
            
        # Calculate overlap
        overlap_tokens = int(self.strategy.max_tokens * self.strategy.overlap_ratio)
        stride_tokens = self.strategy.max_tokens - overlap_tokens
        
        chunks = []
        
        if self.tokenizer and self.model_type in ["voyage", "sentence-transformers"]:
            # Use tokenizer for precise chunking
            chunks = self._chunk_with_tokenizer(text, stride_tokens)
        else:
            # Use character-based chunking
            chunks = self._chunk_by_characters(text, stride_tokens)
            
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
            
        return chunks
        
    def _chunk_with_tokenizer(self, text: str, stride_tokens: int) -> List[TextChunk]:
        """Chunk using tokenizer for precise boundaries."""
        chunks = []
        
        try:
            # Both Voyage and sentence-transformers now use AutoTokenizer
            # Encode without special tokens first to get clean chunks
            encoding = self.tokenizer.encode(text, add_special_tokens=False)
            
            start = 0
            chunk_idx = 0
            
            while start < len(encoding):
                # Leave room for special tokens
                chunk_size = self.strategy.max_tokens - 2  # [CLS] and [SEP] or similar
                end = min(start + chunk_size, len(encoding))
                
                # Decode chunk
                chunk_ids = encoding[start:end]
                chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                
                # Approximate character positions
                if start == 0:
                    char_start = 0
                else:
                    # Find where this chunk starts in original text
                    prefix_text = self.tokenizer.decode(encoding[:start], skip_special_tokens=True)
                    char_start = len(prefix_text)
                    
                char_end = char_start + len(chunk_text)
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_idx=char_start,
                    end_idx=char_end,
                    token_count=len(chunk_ids) + 2,  # Include special tokens
                    chunk_idx=chunk_idx,
                    total_chunks=0  # Updated later
                ))
                
                start += stride_tokens
                chunk_idx += 1
                
                # Break if we've processed everything
                if end >= len(encoding):
                    break
                    
        except Exception as e:
            logger.warning(f"Tokenizer chunking failed: {e}, falling back to character chunking")
            return self._chunk_by_characters(text, stride_tokens)
            
        return chunks
        
    def _chunk_by_characters(self, text: str, stride_tokens: int) -> List[TextChunk]:
        """Fallback character-based chunking."""
        chunks = []
        
        # Estimate characters per token
        if self.model_type == "voyage":
            chars_per_token = 3.5
        else:
            chars_per_token = 4.0
            
        max_chars = int(self.strategy.max_tokens * chars_per_token)
        stride_chars = int(stride_tokens * chars_per_token)
        
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            
            # Try to break at word boundary
            if end < len(text) and not text[end].isspace():
                # Look for last space
                space_idx = text.rfind(' ', start + max_chars // 2, end)
                if space_idx > 0:
                    end = space_idx
                    
            chunk_text = text[start:end]
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_idx=start,
                end_idx=end,
                token_count=self.estimate_tokens(chunk_text),
                chunk_idx=chunk_idx,
                total_chunks=0
            ))
            
            start += stride_chars
            chunk_idx += 1
            
        return chunks
        
    def combine_embeddings(self, 
                          embeddings: List[np.ndarray], 
                          chunks: List[TextChunk]) -> np.ndarray:
        """Combine chunk embeddings into a single embedding.
        
        Args:
            embeddings: List of embeddings for each chunk
            chunks: List of chunks with metadata
            
        Returns:
            Combined embedding vector
        """
        if not embeddings:
            raise ValueError("No embeddings to combine")
            
        if len(embeddings) == 1:
            return embeddings[0]
            
        method = self.strategy.combine_method
        
        if method == "average":
            return np.mean(embeddings, axis=0)
            
        elif method == "weighted_average":
            # Weight by token count
            weights = np.array([chunk.token_count for chunk in chunks], dtype=np.float32)
            weights = weights / weights.sum()
            
            # Weighted sum
            result = np.zeros_like(embeddings[0])
            for embedding, weight in zip(embeddings, weights):
                result += embedding * weight
            return result
            
        elif method == "max_pool":
            return np.max(embeddings, axis=0)
            
        elif method == "first":
            return embeddings[0]
            
        else:
            raise ValueError(f"Unknown combination method: {method}")
            
    def prepare_batches(self, 
                       chunks_per_text: List[List[TextChunk]], 
                       max_batch_size: int = 32) -> List[List[Tuple[int, TextChunk]]]:
        """Prepare batches of chunks for embedding.
        
        Groups chunks into batches while tracking which original text they came from.
        
        Args:
            chunks_per_text: List of chunk lists, one per original text
            max_batch_size: Maximum chunks per batch
            
        Returns:
            List of batches, each containing (text_idx, chunk) tuples
        """
        batches = []
        current_batch = []
        
        for text_idx, chunks in enumerate(chunks_per_text):
            for chunk in chunks:
                current_batch.append((text_idx, chunk))
                
                if len(current_batch) >= max_batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    
        if current_batch:
            batches.append(current_batch)
            
        return batches