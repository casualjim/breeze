"""Core functionality for Breeze."""
from .config import BreezeConfig
from .engine import BreezeEngine
from .models import CodeDocument, IndexStats, SearchResult

__all__ = [
    "BreezeConfig",
    "BreezeEngine", 
    "CodeDocument",
    "IndexStats",
    "SearchResult"
]