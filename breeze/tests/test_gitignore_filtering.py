"""Tests for gitignore filtering functionality."""

import tempfile
from pathlib import Path
import pytest

from breeze.core.file_discovery import FileDiscovery


class TestGitignoreFiltering:
    """Test gitignore pattern filtering in file discovery."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def file_discovery(self):
        """Create a FileDiscovery instance."""
        def should_index(path):
            # Simple filter for text files
            return path.suffix in {'.py', '.js', '.txt', '.md'}
        
        return FileDiscovery(
            exclude_patterns=[],
            should_index_file=should_index
        )
    
    def test_basic_gitignore_patterns(self, temp_dir, file_discovery):
        """Test basic gitignore pattern matching."""
        # Create directory structure
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        
        # Create files
        (src_dir / "main.py").write_text("print('main')")
        (src_dir / "test.py").write_text("print('test')")
        (src_dir / "temp.py").write_text("print('temp')")
        (src_dir / "build.py").write_text("print('build')")
        
        # Create .gitignore
        gitignore_content = """
# Ignore temp files
temp.py
build.py
"""
        (temp_dir / ".gitignore").write_text(gitignore_content)
        
        # Walk directory
        files = file_discovery.walk_directory(temp_dir)
        file_names = [f.name for f in files]
        
        # Check results
        assert "main.py" in file_names
        assert "test.py" in file_names
        assert "temp.py" not in file_names  # Should be ignored
        assert "build.py" not in file_names  # Should be ignored
    
    def test_wildcard_patterns(self, temp_dir, file_discovery):
        """Test wildcard patterns in gitignore."""
        # Create files
        (temp_dir / "test_file.py").write_text("test")
        (temp_dir / "test_data.txt").write_text("data")
        (temp_dir / "production.py").write_text("prod")
        (temp_dir / "README.md").write_text("readme")
        
        # Create .gitignore with wildcards
        gitignore_content = """
test_*
*.txt
"""
        (temp_dir / ".gitignore").write_text(gitignore_content)
        
        # Walk directory
        files = file_discovery.walk_directory(temp_dir)
        file_names = [f.name for f in files]
        
        # Check results
        assert "test_file.py" not in file_names  # Matches test_*
        assert "test_data.txt" not in file_names  # Matches both patterns
        assert "production.py" in file_names
        assert "README.md" in file_names
    
    def test_directory_patterns(self, temp_dir, file_discovery):
        """Test directory-specific patterns."""
        # Create nested structure
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "main.py").write_text("main")
        (temp_dir / "build").mkdir()
        (temp_dir / "build" / "output.py").write_text("output")
        (temp_dir / "dist").mkdir()
        (temp_dir / "dist" / "bundle.js").write_text("bundle")
        
        # Create .gitignore
        gitignore_content = """
build/
dist/
"""
        (temp_dir / ".gitignore").write_text(gitignore_content)
        
        # Walk directory
        files = file_discovery.walk_directory(temp_dir)
        file_paths = [str(f.relative_to(temp_dir)) for f in files]
        
        # Check results
        assert "src/main.py" in file_paths
        assert "build/output.py" not in file_paths
        assert "dist/bundle.js" not in file_paths
    
    def test_negation_patterns(self, temp_dir, file_discovery):
        """Test negation patterns in gitignore."""
        # Create structure
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir()
        (logs_dir / "debug.log").write_text("debug")
        (logs_dir / "important.log").write_text("important")
        (logs_dir / "error.log").write_text("error")
        
        # Use .txt extension for our test
        (logs_dir / "debug.txt").write_text("debug")
        (logs_dir / "important.txt").write_text("important")
        
        # Create .gitignore with negation
        gitignore_content = """
logs/*.txt
!logs/important.txt
"""
        (temp_dir / ".gitignore").write_text(gitignore_content)
        
        # Walk directory
        files = file_discovery.walk_directory(temp_dir)
        file_names = [f.name for f in files]
        
        # Check results
        assert "debug.txt" not in file_names
        assert "important.txt" in file_names  # Negation pattern
    
    def test_nested_gitignore_files(self, temp_dir, file_discovery):
        """Test nested .gitignore files in subdirectories."""
        # Create nested structure
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "main.py").write_text("main")
        (temp_dir / "src" / "temp.py").write_text("temp")
        
        sub_dir = temp_dir / "src" / "submodule"
        sub_dir.mkdir()
        (sub_dir / "module.py").write_text("module")
        (sub_dir / "local.py").write_text("local")
        
        # Root .gitignore
        (temp_dir / ".gitignore").write_text("temp.py\n")
        
        # Nested .gitignore
        (sub_dir / ".gitignore").write_text("local.py\n")
        
        # Walk directory
        files = file_discovery.walk_directory(temp_dir)
        file_names = [f.name for f in files]
        
        # Check results
        assert "main.py" in file_names
        assert "temp.py" not in file_names  # Root gitignore
        assert "module.py" in file_names
        assert "local.py" not in file_names  # Nested gitignore
    
    def test_comments_and_empty_lines(self, temp_dir, file_discovery):
        """Test that comments and empty lines are handled correctly."""
        # Create files
        (temp_dir / "keep.py").write_text("keep")
        (temp_dir / "remove.py").write_text("remove")
        
        # Create .gitignore with comments and empty lines
        gitignore_content = """
# This is a comment
remove.py

# Another comment

"""
        (temp_dir / ".gitignore").write_text(gitignore_content)
        
        # Walk directory
        files = file_discovery.walk_directory(temp_dir)
        file_names = [f.name for f in files]
        
        # Check results
        assert "keep.py" in file_names
        assert "remove.py" not in file_names
    
    def test_gitignore_with_exclude_patterns(self, temp_dir):
        """Test interaction between gitignore and exclude_patterns."""
        # Create FileDiscovery with exclude patterns
        # Note: exclude_patterns in FileDiscovery only filter directories, not files
        file_discovery = FileDiscovery(
            exclude_patterns=["node_modules", "__pycache__"],
            should_index_file=lambda p: p.suffix == '.py'
        )
        
        # Create directory structure with exclude pattern
        node_modules = temp_dir / "node_modules"
        node_modules.mkdir()
        (node_modules / "test.py").write_text("test")
        
        # Create regular files
        (temp_dir / "main.py").write_text("main")
        (temp_dir / "build.py").write_text("build")
        
        # Create .gitignore
        (temp_dir / ".gitignore").write_text("build.py\n")
        
        # Walk directory
        files = file_discovery.walk_directory(temp_dir)
        file_names = [f.name for f in files]
        
        # Check results
        assert "main.py" in file_names
        assert "test.py" not in file_names  # Excluded because in node_modules dir
        assert "build.py" not in file_names  # Excluded by gitignore
    
    def test_no_gitignore_file(self, temp_dir, file_discovery):
        """Test behavior when no .gitignore file exists."""
        # Create files without .gitignore
        (temp_dir / "main.py").write_text("main")
        (temp_dir / "test.py").write_text("test")
        (temp_dir / "temp.py").write_text("temp")
        
        # Walk directory
        files = file_discovery.walk_directory(temp_dir)
        file_names = [f.name for f in files]
        
        # All files should be included
        assert "main.py" in file_names
        assert "test.py" in file_names
        assert "temp.py" in file_names
    
    def test_glob_patterns(self, temp_dir, file_discovery):
        """Test glob patterns in gitignore."""
        # Create nested structure
        src = temp_dir / "src"
        src.mkdir()
        tests = temp_dir / "tests"
        tests.mkdir()
        
        (src / "main.py").write_text("main")
        (src / "util.py").write_text("util")
        (tests / "test_main.py").write_text("test main")
        (tests / "test_util.py").write_text("test util")
        
        # Create .gitignore with glob patterns
        gitignore_content = """
**/test_*.py
"""
        (temp_dir / ".gitignore").write_text(gitignore_content)
        
        # Walk directory
        files = file_discovery.walk_directory(temp_dir)
        file_names = [f.name for f in files]
        
        # Check results
        assert "main.py" in file_names
        assert "util.py" in file_names
        assert "test_main.py" not in file_names
        assert "test_util.py" not in file_names