"""Content type detection for code files using identify and libmagic."""

import logging
from pathlib import Path
from typing import Optional, List, Set

try:
    from identify import identify
except ImportError:
    raise ImportError("Please install identify: pip install identify")

try:
    import magic
except ImportError:
    raise ImportError("Please install python-magic: pip install python-magic")

logger = logging.getLogger(__name__)


class ContentDetector:
    """Detects whether files contain code using identify and libmagic."""

    def __init__(self, exclude_patterns: Optional[List[str]] = None):
        """
        Initialize content detector.

        Args:
            exclude_patterns: List of patterns to exclude from indexing
        """
        self.exclude_patterns = exclude_patterns or []
        self._magic = None
        self._init_magic()

    def _init_magic(self):
        """Initialize libmagic for MIME type detection."""
        try:
            self._magic = magic.Magic(mime=True)
        except Exception as e:
            raise RuntimeError(
                "python-magic requires libmagic to be installed.\n"
                "  macOS: brew install libmagic\n"
                "  Ubuntu/Debian: sudo apt-get install libmagic1\n"
                "  RHEL/CentOS: sudo yum install file-devel\n"
                f"Actual error: {e}"
            )

    def should_index_file(self, file_path: Path) -> bool:
        """
        Check if a file should be indexed based on content detection.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file should be indexed, False otherwise
        """
        # Check exclude patterns first
        path_str = str(file_path)
        for pattern in self.exclude_patterns:
            if pattern in path_str:
                return False

        try:
            # First try identify for known file types
            tags = identify.tags_from_path(str(file_path))

            # Skip if explicitly marked as binary
            if "binary" in tags:
                return False

            # If it has 'text' tag or any programming language tag, index it
            if "text" in tags or any(
                tag for tag in tags if tag not in {"binary", "executable"}
            ):
                return True

            # For files without clear tags, use python-magic
            try:
                mime = self._magic.from_file(str(file_path))
                # Index text files, source code, config files, etc.
                if mime.startswith("text/") or mime in {
                    "application/json",
                    "application/xml",
                    "application/x-yaml",
                    "application/javascript",
                    "application/x-python-code",
                    "application/x-ruby",
                    "application/x-sh",
                    "application/x-csh",
                    "application/x-shellscript",
                    "application/x-latex",
                    "application/x-tcl",
                    "application/x-tex",
                }:
                    return True
            except (OSError, IOError):
                pass

        except Exception:
            pass

        return False

    def get_file_tags(self, file_path: Path) -> Set[str]:
        """
        Get identify tags for a file.

        Args:
            file_path: Path to the file

        Returns:
            Set of tags from identify
        """
        try:
            return identify.tags_from_path(str(file_path))
        except Exception:
            return set()
    
    def detect_language(self, file_path: Path) -> Optional[str]:
        """
        Detect the programming language of a file.

        Args:
            file_path: Path to the file

        Returns:
            Detected language name suitable for tree-sitter, or None
        """
        tags = self.get_file_tags(file_path)
        
        # Map identify tags to tree-sitter language names
        language_map = {
            'python': 'python',
            'javascript': 'javascript',
            'jsx': 'jsx',
            'typescript': 'typescript',
            'tsx': 'tsx',
            'java': 'java',
            'c++': 'cpp',
            'c': 'c',
            'c#': 'c_sharp',
            'go': 'go',
            'rust': 'rust',
            'ruby': 'ruby',
            'php': 'php',
            'swift': 'swift',
            'kotlin': 'kotlin',
            'scala': 'scala',
            'r': 'r',
            'lua': 'lua',
            'dart': 'dart',
            'bash': 'bash',
            'shell': 'bash',
            'sh': 'bash',
            'json': 'json',
            'yaml': 'yaml',
            'toml': 'toml',
            'xml': 'xml',
            'html': 'html',
            'css': 'css',
            'scss': 'scss',
            'sql': 'sql',
            'markdown': 'markdown',
            'vim': 'vim',
            'elisp': 'elisp',
            'clojure': 'clojure',
            'elixir': 'elixir',
            'haskell': 'haskell',
            'julia': 'julia',
            'ocaml': 'ocaml',
            'perl': 'perl',
            'racket': 'racket',
            'zig': 'zig',
        }
        
        # Check tags for language matches
        for tag in tags:
            if tag in language_map:
                return language_map[tag]
        
        return None
