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
    embedding_model: str = "nomic-ai/CodeRankEmbed"
    trust_remote_code: bool = True

    # Indexing settings
    code_extensions: List[str] = None
    exclude_patterns: List[str] = None
    max_file_size: int = 1024 * 1024  # 1MB default

    # Search settings
    default_limit: int = 10
    min_relevance: float = 0.0

    def __post_init__(self):
        # Set platform-specific data directory if not provided
        if self.data_root is None:
            self.data_root = platformdirs.user_data_dir("breeze", "breeze-mcp")
        
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
            ]

    def get_db_path(self) -> str:
        """Get the full path to the LanceDB database."""
        return os.path.join(self.data_root, self.db_name)

    def ensure_directories(self):
        """Ensure all required directories exist."""
        os.makedirs(self.data_root, exist_ok=True)
