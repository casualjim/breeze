"""Configuration for Breeze code indexing."""

import os
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lancedb.embeddings import EmbeddingFunction
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
    max_sequence_length: Optional[int] = None  # Override max sequence length for local models
    embedding_function: Optional["EmbeddingFunction"] = None  # Allow passing custom embedding function for testing

    # Concurrency settings for indexing
    concurrent_readers: int = 20
    concurrent_embedders: int = 10
    concurrent_writers: int = 1  # Always 1 writer to avoid concurrency issues (not configurable)
    voyage_tier: int = 1  # Voyage AI tier (1, 2, or 3)
    voyage_concurrent_requests: int = 5  # For Voyage AI API rate limiting
    voyage_max_retries: int = 3  # Max retries for rate-limited requests
    voyage_retry_base_delay: float = 1.0  # Base delay in seconds for exponential backoff

    # Indexing settings
    code_extensions: List[str] = None
    exclude_patterns: List[str] = None
    batch_size: Optional[int] = None  # Batch size for embedding generation

    # Search settings
    default_limit: int = 10
    min_relevance: float = 0.0

    # Reranker settings
    reranker_model: Optional[str] = None  # From env: BREEZE_RERANKER_MODEL
    reranker_api_key: Optional[str] = None  # From env: BREEZE_RERANKER_API_KEY

    # File watcher settings
    file_watcher_debounce_seconds: float = 2.0  # Debounce time for file change events


    def __post_init__(self):
        # Set platform-specific data directory if not provided
        if self.data_root is None:
            self.data_root = platformdirs.user_data_dir("breeze", "breeze-mcp")

        # Load reranker settings from environment if not set
        if self.reranker_model is None:
            self.reranker_model = os.environ.get("BREEZE_RERANKER_MODEL")

        if self.reranker_api_key is None:
            self.reranker_api_key = os.environ.get("BREEZE_RERANKER_API_KEY")

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

        # Very conservative calculation to avoid rate limits
        # For tier 1, be extra conservative since that's where most users are
        if self.voyage_tier == 1:
            # For tier 1, use only 10-15 concurrent requests to avoid rate limits
            calculated_concurrent = 10
        else:
            # For higher tiers, use 50% of rate limit / 20
            calculated_concurrent = int((max_requests_per_minute * 0.5) / 20)

        # If voyage_concurrent_requests not explicitly set, use calculated value
        # Cap at reasonable limits: min 5, max 30
        if self.voyage_concurrent_requests == 5:  # Default value
            self.voyage_concurrent_requests = max(5, min(30, calculated_concurrent))

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
                # MPS is available but many models have compatibility issues
                # Specifically, models using FBGemm operations fail on MPS
                # For now, default to CPU to avoid these issues
                # Users can explicitly set device='mps' if their model supports it
                import logging
                logger = logging.getLogger(__name__)
                logger.info("MPS device available but defaulting to CPU for compatibility. "
                           "Set embedding_device='mps' explicitly if your model supports it.")
                return "cpu"
        except ImportError:
            pass
        return "cpu"

    def get_reranker_model(self) -> str:
        """Get the reranker model to use based on config or embedding model."""
        if self.reranker_model:
            return self.reranker_model

        # Default reranker models based on embedding model type
        default_reranker_map = {
            'voyage-': 'rerank-2',  # Voyage AI's reranker
            'models/': 'models/gemini-2.0-flash-lite',  # Gemini reranker
            'default': 'BAAI/bge-reranker-v2-m3'  # Local reranker for everything else
        }

        # Auto-select based on embedding model
        if self.embedding_model.startswith('voyage-'):
            return default_reranker_map['voyage-']
        elif self.embedding_model.startswith('models/') or self.embedding_model.startswith('gemini-'):
            return default_reranker_map['models/']
        else:
            return default_reranker_map['default']

    def get_batch_size(self) -> int:
        """Get the appropriate batch size for embedding generation."""
        if self.batch_size is not None:
            return self.batch_size

        # Default batch sizes based on model type
        if self.embedding_model.startswith('voyage-'):
            # Smaller batches for Voyage AI due to token limits
            return 20
        elif self.embedding_model.startswith('models/'):
            # Medium batches for Gemini models
            return 50
        else:
            # Larger batches for local models
            return 100
