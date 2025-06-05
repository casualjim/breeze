"""Configuration for Breeze code indexing."""

import os
from dataclasses import dataclass
from typing import List, Optional
import platformdirs


@dataclass
class BreezeConfig:
    """Configuration for Breeze code indexing system."""

    # Database settings
    data_root: Optional[str] = None
    db_name: str = "code_index"

    # Model settings
    embedding_model: str = "ibm-granite/granite-embedding-125m-english"
    embedding_device: str = "cpu"  # Device for embeddings: 'cpu', 'cuda', 'mps'
    trust_remote_code: bool = True
    embedding_api_key: Optional[str] = None  # API key for cloud embedding providers

    # Concurrency settings for indexing
    concurrent_readers: int = 20
    concurrent_embedders: int = 10
    concurrent_writers: int = 10
    voyage_tier: int = 1  # Voyage AI tier (1, 2, or 3)
    voyage_concurrent_requests: int = 5  # For Voyage AI API rate limiting
    voyage_max_retries: int = 3  # Max retries for rate-limited requests
    voyage_retry_base_delay: float = 1.0  # Base delay in seconds for exponential backoff

    # Indexing settings
    code_extensions: List[str] = None
    exclude_patterns: List[str] = None

    # Search settings
    default_limit: int = 10
    min_relevance: float = 0.0

    def __post_init__(self):
        # Set platform-specific data directory if not provided
        if self.data_root is None:
            self.data_root = platformdirs.user_data_dir("breeze", "breeze-mcp")

        # Auto-detect best available device if not set
        if self.embedding_device == "cpu":
            self.embedding_device = self._detect_best_device()

        # Validate and adjust voyage tier
        if self.voyage_tier not in [1, 2, 3]:
            self.voyage_tier = 1

        # Calculate appropriate concurrent requests based on tier
        # Base tier 1: 3M tokens/min, 2000 requests/min
        # Tier 2: 2x base, Tier 3: 3x base
        base_requests_per_minute = 2000
        tier_multipliers = {1: 1, 2: 2, 3: 3}
        max_requests_per_minute = base_requests_per_minute * tier_multipliers[self.voyage_tier]
        
        # Conservative calculation: assume average processing time of 3 seconds per request
        # This gives us requests per minute / 20 as concurrent requests
        calculated_concurrent = max_requests_per_minute // 20
        
        # If voyage_concurrent_requests not explicitly set, use calculated value
        # Cap at reasonable limits: min 5, max 100
        if self.voyage_concurrent_requests == 5:  # Default value
            self.voyage_concurrent_requests = max(5, min(100, calculated_concurrent))

        if self.code_extensions is None:
            self.code_extensions = [
                ".py",
                ".js",
                ".ts",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".hpp",
                ".go",
                ".rs",
                ".jsx",
                ".tsx",
                ".php",
                ".rb",
                ".swift",
                ".kt",
                ".scala",
                ".sh",
                ".bash",
                ".zsh",
                ".fish",
                ".lua",
                ".r",
                ".R",
                ".m",
                ".mm",
                ".cs",
                ".vb",
                ".fs",
                ".clj",
                ".cljs",
                ".elm",
                ".ex",
                ".exs",
                ".erl",
                ".hrl",
                ".ml",
                ".mli",
                ".nim",
                ".cr",
                ".dart",
                ".jl",
                ".v",
                ".zig",
                ".sol",
                ".vue",
                ".svelte",
            ]

        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "__pycache__",
                ".git",
                ".svn",
                ".hg",
                ".bzr",
                "node_modules",
                "vendor",
                "venv",
                ".venv",
                "dist",
                "build",
                ".idea",
                ".vscode",
                "*.min.js",
                "*.min.css",
                "*.map",
                # Data files and directories
                "dataset",
                "datasets",
                "data",
                "*.csv",
                "*.parquet",
                "*.db",
                "*.sqlite",
                "*.sqlite3",
                "*.json",  # Often large data files
                "*.xml",  # Often large data files
                "*.log",
                "*.pkl",
                "*.pickle",
                "*.npy",
                "*.npz",
                "*.h5",
                "*.hdf5",
                "*.mat",
                "*.feather",
                "*.arrow",
                "*.msgpack",
                "*.bin",
                "*.dat",
                "*.dump",
            ]

    def get_db_path(self) -> str:
        """Get the full path to the LanceDB database."""
        return os.path.join(self.data_root, self.db_name)

    def ensure_directories(self):
        """Ensure all required directories exist."""
        os.makedirs(self.data_root, exist_ok=True)

    def get_voyage_rate_limits(self):
        """Get rate limits for the configured Voyage AI tier.
        
        Returns:
            dict: Contains 'tokens_per_minute' and 'requests_per_minute'
        """
        base_tokens = 3_000_000  # 3M tokens per minute for tier 1
        base_requests = 2000     # 2000 requests per minute for tier 1
        
        tier_multipliers = {1: 1, 2: 2, 3: 3}
        multiplier = tier_multipliers.get(self.voyage_tier, 1)
        
        return {
            'tokens_per_minute': base_tokens * multiplier,
            'requests_per_minute': base_requests * multiplier,
            'tier': self.voyage_tier,
            'tier_name': f"Tier {self.voyage_tier}",
            'concurrent_requests': self.voyage_concurrent_requests
        }

    def _detect_best_device(self) -> str:
        """Detect the best available device for embeddings."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
