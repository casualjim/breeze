"""Data models for Breeze code indexing using Pydantic v2."""

from datetime import datetime
from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field
from lancedb.pydantic import LanceModel, Vector
import uuid_utils as uuid


class RetryStatus(str, Enum):
    """Status of retry attempts."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"


class FailedBatch(LanceModel):
    """Model for tracking failed indexing batches."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    batch_id: str = Field(description="Unique identifier for this batch")
    file_paths: List[str] = Field(description="List of file paths in this batch")
    content_hashes: List[str] = Field(description="Content hashes for deduplication")
    error_message: str = Field(description="Last error message")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=5, description="Maximum retries before abandoning")
    status: str = Field(default="pending", description="Status: pending, processing, succeeded, failed, abandoned")
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    last_retry_at: Optional[datetime] = Field(default=None)
    next_retry_at: Optional[datetime] = Field(default=None, description="When to retry next")
    project_id: Optional[str] = Field(default=None, description="Associated project if any")


class CodeDocument(LanceModel):
    """Represents a code document in the index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="Unique identifier for the document")
    file_path: str = Field(description="Path to the code file")
    content: str = Field(description="Full content of the code file")
    file_type: str = Field(description="File extension without dot")
    file_size: int = Field(description="Size of the file in bytes")
    last_modified: datetime = Field(description="Last modification time of the file")
    indexed_at: datetime = Field(description="Time when the file was indexed")
    content_hash: str = Field(description="SHA256 hash of the file content")
    # Vector field will be added dynamically based on the embedding model


class SearchResult(BaseModel):
    """Represents a search result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    file_path: str
    file_type: str
    relevance_score: float
    snippet: str
    last_modified: datetime


class IndexStats(BaseModel):
    """Statistics for indexing operations."""

    files_scanned: int = 0
    files_indexed: int = 0
    files_updated: int = 0
    files_skipped: int = 0
    errors: int = 0
    total_tokens_processed: int = 0


class Project(LanceModel):
    """Represents a tracked project/repository in LanceDB."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Project name")
    paths: List[str] = Field(description="List of paths to track")
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    updated_at: datetime = Field(default_factory=lambda: datetime.now())
    last_indexed: Optional[datetime] = Field(default=None, description="Last indexing time")
    is_watching: bool = Field(default=False, description="Whether file watching is active")
    file_extensions: Optional[List[str]] = Field(default=None, description="Deprecated - content detection is now automatic")
    exclude_patterns: List[str] = Field(default_factory=list, description="Patterns to exclude")
    auto_index: bool = Field(default=True, description="Auto-index on file changes")


class IndexingTask(LanceModel):
    """Represents an indexing task with full persistence support."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    task_id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    paths: List[str] = Field(description="Directories to index")
    force_reindex: bool = Field(default=False, description="Force reindexing of all files")
    status: str = Field(default="queued", description="Status: queued, running, completed, failed")
    
    # Progress tracking
    progress: float = Field(default=0.0, description="Progress percentage (0-100)")
    total_files: int = Field(default=0, description="Total number of files to process")
    processed_files: int = Field(default=0, description="Number of files processed")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    started_at: Optional[datetime] = Field(default=None, description="When task started executing")
    completed_at: Optional[datetime] = Field(default=None, description="When task finished")
    
    # Results - proper fields for IndexStats
    result_files_scanned: Optional[int] = Field(default=None)
    result_files_indexed: Optional[int] = Field(default=None)
    result_files_updated: Optional[int] = Field(default=None)
    result_files_skipped: Optional[int] = Field(default=None)
    result_errors: Optional[int] = Field(default=None)
    result_total_tokens_processed: Optional[int] = Field(default=None)
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Queue management
    queue_position: Optional[int] = Field(default=None, description="Position in queue (0 = next)")
    attempt_count: int = Field(default=0, description="Number of execution attempts")
    
    # Optional project association
    project_id: Optional[str] = Field(default=None, description="Associated project if any")
    

def get_code_document_schema(embedding_model):
    """Get CodeDocument schema with embedding function configured.
    
    Args:
        embedding_model: The embedding model from LanceDB registry
    
    Returns:
        A CodeDocument class with vector field configured for the embedding model
    """
    class CodeDocumentWithEmbedding(LanceModel):
        """Code document with embedding vector."""
        
        model_config = ConfigDict(arbitrary_types_allowed=True)
        
        id: str = Field(description="Unique identifier for the document")
        file_path: str = Field(description="Path to the code file")
        content: str = embedding_model.SourceField()
        file_type: str = Field(description="File extension without dot")
        file_size: int = Field(description="Size of the file in bytes")
        last_modified: datetime = Field(description="Last modification time of the file")
        indexed_at: datetime = Field(description="Time when the file was indexed")
        content_hash: str = Field(description="SHA256 hash of the file content")
        vector: Vector(embedding_model.ndims()) = embedding_model.VectorField(default=None)  # type: ignore
    
    return CodeDocumentWithEmbedding
