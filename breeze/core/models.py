"""Data models for Breeze code indexing using Pydantic v2."""

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field
from lancedb.pydantic import LanceModel, Vector


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
    vector: Vector(dim=768) = Field(  # type: ignore
        description="Embedding vector"
    )  # CodeRankEmbed produces 768-dim vectors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "content": self.content,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "last_modified": self.last_modified,
            "indexed_at": self.indexed_at,
            "content_hash": self.content_hash,
            "vector": self.vector,
        }


class SearchResult(BaseModel):
    """Represents a search result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    file_path: str
    file_type: str
    relevance_score: float
    snippet: str
    last_modified: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "relevance_score": self.relevance_score,
            "snippet": self.snippet,
            "last_modified": self.last_modified.isoformat(),
        }


class IndexStats(BaseModel):
    """Statistics for indexing operations."""

    files_scanned: int = 0
    files_indexed: int = 0
    files_updated: int = 0
    files_skipped: int = 0
    errors: int = 0
    total_tokens_processed: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()
