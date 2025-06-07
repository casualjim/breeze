"""File discovery and filtering for Breeze indexing."""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from gitignore_parser import parse_gitignore

logger = logging.getLogger(__name__)


class FileDiscovery:
    """Handles file discovery with gitignore and pattern filtering."""
    
    def __init__(self, exclude_patterns: List[str], should_index_file: Callable[[Path], bool]):
        """
        Initialize file discovery.
        
        Args:
            exclude_patterns: List of patterns to exclude
            should_index_file: Callback to determine if a file should be indexed
        """
        self.exclude_patterns = exclude_patterns
        self.should_index_file = should_index_file
    
    def walk_directory(self, directory: Path) -> List[Path]:
        """
        Fast synchronous directory walk with filtering and gitignore support.
        
        Args:
            directory: Root directory to walk
            
        Returns:
            List of paths to files that should be indexed
        """
        files_to_index = []
        visited_dirs = set()
        gitignore_matchers = {}  # Cache of gitignore matchers by directory

        # Load .gitignore from the root directory and its parents
        root_matchers = self._load_gitignore_chain(directory)

        for root, dirs, files in os.walk(directory, followlinks=False):
            root_path = Path(root)

            # Skip if we've seen this real path before
            try:
                real_path = root_path.resolve()
                if real_path in visited_dirs:
                    continue
                visited_dirs.add(real_path)
            except Exception:
                continue

            # Load .gitignore for this directory if not cached
            if root_path not in gitignore_matchers:
                gitignore_matchers[root_path] = self._load_gitignore(root_path)

            # Filter directories
            dirs[:] = self._filter_directories(root_path, dirs, root_matchers, gitignore_matchers)

            # Process files
            for file in files:
                if self._should_process_file(root_path, file, root_matchers, gitignore_matchers):
                    file_path = root_path / file
                    if self.should_index_file(file_path):
                        files_to_index.append(file_path)

        return files_to_index
    
    def _load_gitignore_chain(self, path: Path) -> List[Tuple[Path, Callable]]:
        """Load all .gitignore files from path up to git root or filesystem root."""
        matchers = []
        current = path
        
        while current != current.parent:
            gitignore_path = current / ".gitignore"
            if gitignore_path.exists():
                try:
                    matcher = parse_gitignore(gitignore_path)
                    matchers.append((current, matcher))
                except Exception as e:
                    logger.warning(f"Error parsing {gitignore_path}: {e}")
            
            # Check if we've reached a git root
            if (current / ".git").exists():
                break
                
            current = current.parent
        
        return matchers
    
    def _load_gitignore(self, directory: Path) -> Optional[Callable]:
        """Load .gitignore for a specific directory."""
        gitignore_path = directory / ".gitignore"
        if gitignore_path.exists():
            try:
                return parse_gitignore(gitignore_path)
            except Exception as e:
                logger.warning(f"Error parsing {gitignore_path}: {e}")
        return None
    
    def _filter_directories(
        self, 
        root_path: Path, 
        dirs: List[str], 
        root_matchers: List[Tuple[Path, Callable]],
        gitignore_matchers: Dict[Path, Optional[Callable]]
    ) -> List[str]:
        """Filter directories based on patterns and gitignore rules."""
        filtered_dirs = []
        
        for d in dirs:
            # Skip hidden directories
            if d.startswith("."):
                continue
                
            # Skip based on config patterns
            if any(pattern in d for pattern in self.exclude_patterns):
                continue
            
            # Check gitignore rules
            dir_path = root_path / d
            if not self._is_ignored(dir_path, root_path, root_matchers, gitignore_matchers):
                filtered_dirs.append(d)
        
        return filtered_dirs
    
    def _should_process_file(
        self,
        root_path: Path,
        filename: str,
        root_matchers: List[Tuple[Path, Callable]],
        gitignore_matchers: Dict[Path, Optional[Callable]]
    ) -> bool:
        """Check if a file should be processed."""
        # Skip hidden files except .gitignore
        if filename.startswith(".") and filename != ".gitignore":
            return False
        
        file_path = root_path / filename
        
        # Check gitignore rules
        return not self._is_ignored(file_path, root_path, root_matchers, gitignore_matchers)
    
    def _is_ignored(
        self,
        path: Path,
        current_dir: Path,
        root_matchers: List[Tuple[Path, Callable]],
        gitignore_matchers: Dict[Path, Optional[Callable]]
    ) -> bool:
        """Check if a path is ignored by gitignore rules."""
        # Check against root matchers first (respects full path from project root)
        for matcher_root, matcher in root_matchers:
            if matcher(str(path)):
                return True
        
        # Check against local .gitignore
        local_matcher = gitignore_matchers.get(current_dir)
        if local_matcher and local_matcher(str(path)):
            return True
        
        return False