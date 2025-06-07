"""Content type detection for code files using breeze-langdetect (hyperpolyglot)."""

import logging
from pathlib import Path
from typing import Optional, List

import breeze_langdetect

logger = logging.getLogger(__name__)

# File categories that should not be indexed (binary files)
BINARY_CATEGORIES = [
    breeze_langdetect.FileCategory.IMAGE,
    breeze_langdetect.FileCategory.VIDEO,
    breeze_langdetect.FileCategory.AUDIO,
    breeze_langdetect.FileCategory.ARCHIVE,
    breeze_langdetect.FileCategory.FONT,
    breeze_langdetect.FileCategory.APPLICATION,  # executables
]


class ContentDetector:
    """Detects whether files contain code using breeze-langdetect (hyperpolyglot)."""

    def __init__(self, exclude_patterns: Optional[List[str]] = None):
        """
        Initialize content detector.

        Args:
            exclude_patterns: List of patterns to exclude from indexing
        """
        self.exclude_patterns = exclude_patterns or []

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

        # Basic checks
        if not file_path.exists() or not file_path.is_file():
            return False

        # Skip hidden files
        if file_path.name.startswith('.'):
            return False

        # Use breeze-langdetect to detect file info
        try:
            file_info = breeze_langdetect.detect_file_info(str(file_path))
            
            # Skip binary files
            if file_info.file_type and file_info.file_type.category in BINARY_CATEGORIES:
                logger.debug(f"Skipping binary file {file_path}: {file_info.file_type.category}")
                return False
            
            # Index if we detected a programming language
            if file_info.language:
                return True
                
            # Also index plain text files (but not binary)
            if file_info.file_type and file_info.file_type.category == breeze_langdetect.FileCategory.TEXT:
                return True
                
            # Skip files we can't identify
            return False
            
        except Exception as e:
            logger.debug(f"Error detecting file info for {file_path}: {e}")
            return False

    
    def detect_language(self, file_path: Path) -> Optional[str]:
        """
        Detect the programming language of a file.

        Args:
            file_path: Path to the file

        Returns:
            Detected language name suitable for tree-sitter, or None
        """
        try:
            file_info = breeze_langdetect.detect_file_info(str(file_path))
            
            # Skip binary files
            if file_info.file_type and file_info.file_type.category in BINARY_CATEGORIES:
                return None
            
            # Return the detected language
            if file_info.language:
                # Map hyperpolyglot names to tree-sitter names
                # This is a much smaller mapping since hyperpolyglot
                # already knows about most languages
                hp_to_ts_mapping = {
                    'C++': 'cpp',
                    'C#': 'csharp',
                    'F#': 'fsharp',
                    'F*': 'fstar',
                    'Objective-C': 'objc',
                    'Objective-C++': 'objcpp',
                    'Shell': 'bash',
                    'Vim Script': 'vim',
                    'Vim script': 'vim',
                    'Emacs Lisp': 'elisp',
                    'Common Lisp': 'commonlisp',
                    'reStructuredText': 'rst',
                }
                return hp_to_ts_mapping.get(file_info.language, file_info.language.lower())
            
            # For plain text files without a specific language, return "text"
            if file_info.file_type and file_info.file_type.category == breeze_langdetect.FileCategory.TEXT:
                return "text"
                
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting language for {file_path}: {e}")
            return None
