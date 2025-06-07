"""Tree-sitter query definitions for semantic code understanding.

This module provides extensible tree-sitter queries for identifying
semantic units (functions, classes, structs, etc.) across programming languages.
"""

import logging
import tree_sitter_language_pack
from typing import get_args
from typing import Dict, Set, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Query patterns for different languages to find semantic units
# These queries identify functions, methods, classes, and other top-level constructs

# Core language queries that are built-in
CORE_LANGUAGE_QUERIES = {
    "python": """
        (function_definition) @function
        (class_definition) @class
        (decorated_definition) @decorated
    """,
    "javascript": """
        (function_declaration) @function
        (function) @function
        (arrow_function) @function
        (method_definition) @method
        (class_declaration) @class
        (class) @class
    """,
    "typescript": """
        (function_declaration) @function
        (function) @function
        (arrow_function) @function
        (method_definition) @method
        (method_signature) @method
        (class_declaration) @class
        (interface_declaration) @interface
        (type_alias_declaration) @type
        (enum_declaration) @enum
    """,
    "java": """
        (method_declaration) @method
        (constructor_declaration) @constructor
        (class_declaration) @class
        (interface_declaration) @interface
        (enum_declaration) @enum
    """,
    "go": """
        (function_declaration) @function
        (method_declaration) @method
        (type_declaration) @type
    """,
    "rust": """
        (function_item) @function
        (impl_item) @impl
        (struct_item) @struct
        (enum_item) @enum
        (trait_item) @trait
        (mod_item) @module
    """,
    "cpp": """
        (function_definition) @function
        (class_specifier) @class
        (struct_specifier) @struct
        (namespace_definition) @namespace
    """,
    "c": """
        (function_definition) @function
        (struct_specifier) @struct
        (enum_specifier) @enum
    """,
    "ruby": """
        (method) @method
        (singleton_method) @method
        (class) @class
        (module) @module
    """,
    "php": """
        (function_definition) @function
        (method_declaration) @method
        (class_declaration) @class
        (interface_declaration) @interface
        (trait_declaration) @trait
    """,
    "csharp": """
        (method_declaration) @method
        (constructor_declaration) @constructor
        (class_declaration) @class
        (interface_declaration) @interface
        (struct_declaration) @struct
        (enum_declaration) @enum
    """,
    "swift": """
        (function_declaration) @function
        (class_declaration) @class
        (protocol_declaration) @protocol
        (struct_declaration) @struct
        (enum_declaration) @enum
    """,
    "kotlin": """
        (function_declaration) @function
        (class_declaration) @class
        (object_declaration) @object
        (interface_declaration) @interface
    """,
    # Zig language support
    "zig": """
        (function_declaration) @function
        (test_declaration) @test
        (struct_declaration) @struct
        (enum_declaration) @enum
        (union_declaration) @union
        (error_set_declaration) @error_set
    """,
    # Shell script support (bash, sh, zsh)
    "bash": """
        (function_definition) @function
        (command
          name: (command_name) @command)
        (variable_assignment
          name: (variable_name) @variable)
        (for_statement) @loop
        (while_statement) @loop
        (if_statement) @conditional
        (case_statement) @conditional
    """,
    "shell": """
        (function_definition) @function
        (command
          name: (command_name) @command)
        (variable_assignment
          name: (variable_name) @variable)
        (for_statement) @loop
        (while_statement) @loop
        (if_statement) @conditional
        (case_statement) @conditional
    """,
    "sh": """
        (function_definition) @function
        (command
          name: (command_name) @command)
        (variable_assignment
          name: (variable_name) @variable)
        (for_statement) @loop
        (while_statement) @loop
        (if_statement) @conditional
        (case_statement) @conditional
    """,
    # Fallback for languages without specific queries
    "default": """
        (function_definition) @function
        (method_definition) @method
        (class_definition) @class
        (class_declaration) @class
    """,
}

# Additional queries loaded from external sources
EXTENDED_QUERIES: Dict[str, str] = {}


class QueryManager:
    """Manages tree-sitter queries with support for dynamic loading and extension."""

    def __init__(self):
        self._queries = CORE_LANGUAGE_QUERIES.copy()
        self._custom_queries_path: Optional[Path] = None
        self._supported_languages: Optional[Set[str]] = None

    def set_custom_queries_path(self, path: Path) -> None:
        """Set path to load custom queries from."""
        self._custom_queries_path = path
        self._load_custom_queries()

    def _load_custom_queries(self) -> None:
        """Load custom queries from JSON file if path is set."""
        if not self._custom_queries_path or not self._custom_queries_path.exists():
            return

        try:
            with open(self._custom_queries_path, "r") as f:
                custom_queries = json.load(f)

            if isinstance(custom_queries, dict):
                for lang, query in custom_queries.items():
                    if isinstance(query, str):
                        self._queries[lang] = query
                        logger.info(f"Loaded custom query for language: {lang}")

        except Exception as e:
            logger.warning(
                f"Failed to load custom queries from {self._custom_queries_path}: {e}"
            )

    def add_query(self, language: str, query: str) -> None:
        """Add or update a query for a specific language."""
        self._queries[language] = query
        logger.debug(f"Added/updated query for language: {language}")

    def get_query(self, language: str) -> str:
        """Get query for a language with fallback to default."""
        # Normalize language name
        language = language.lower()
        
        # Return specific query or default
        return self._queries.get(language, self._queries["default"])

    def get_supported_languages(self) -> Set[str]:
        """Get set of languages with custom queries."""
        if self._supported_languages is None:
            # Check which languages are actually available in tree-sitter
            try:
                # Get all supported languages from the type hint
                supported = set(get_args(tree_sitter_language_pack.SupportedLanguage))

                # Filter to only languages we have queries for
                self._supported_languages = {
                    lang
                    for lang in self._queries.keys()
                    if lang in supported or lang == "default"
                }
            except ImportError:
                # Fallback to just our query keys if tree-sitter not available
                self._supported_languages = set(self._queries.keys())

        return self._supported_languages

    def has_query(self, language: str) -> bool:
        """Check if a specific query exists for a language."""
        language = language.lower()
        return language in self._queries


# Global query manager instance
_query_manager = QueryManager()

# Backwards compatibility - expose as LANGUAGE_QUERIES
LANGUAGE_QUERIES = _query_manager._queries


def get_query_for_language(language: str) -> str:
    """Get the tree-sitter query pattern for a specific language.

    Args:
        language: The programming language name

    Returns:
        The query pattern string for that language
    """
    return _query_manager.get_query(language)


# Export additional functions for extensibility
def add_language_query(language: str, query: str) -> None:
    """Add or update a query for a specific language.

    Args:
        language: The programming language name
        query: The tree-sitter query pattern
    """
    _query_manager.add_query(language, query)


def load_custom_queries(path: Path) -> None:
    """Load custom queries from a JSON file.

    Args:
        path: Path to JSON file containing language->query mappings
    """
    _query_manager.set_custom_queries_path(path)


def get_supported_languages() -> Set[str]:
    """Get set of languages with custom queries available."""
    return _query_manager.get_supported_languages()
