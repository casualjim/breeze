"""Core navigation functionality for finding symbols, references, and implementations."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict

import tree_sitter_language_pack
from tree_sitter import Query, Node

from breeze.core.content_detection import ContentDetector
from breeze.core.tree_sitter_queries import get_query_for_language

logger = logging.getLogger(__name__)


@dataclass
class Symbol:
    """Represents a code symbol (function, class, variable, etc.)."""
    name: str
    type: str  # 'function', 'class', 'method', 'variable', etc.
    file_path: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    parent_name: Optional[str] = None  # For methods in classes
    signature: Optional[str] = None  # Full signature for functions/methods
    docstring: Optional[str] = None  # Documentation if available


@dataclass
class Reference:
    """Represents a reference to a symbol."""
    symbol_name: str
    file_path: str
    line: int
    column: int
    context: str  # Line of code containing the reference
    reference_type: str  # 'call', 'import', 'definition', 'type_hint', etc.


@dataclass
class Implementation:
    """Represents an implementation of an interface/abstract method."""
    interface_name: str
    method_name: str
    file_path: str
    line: int
    column: int
    class_name: str  # Implementing class


class CodeNavigator:
    """Provides code navigation capabilities using tree-sitter."""
    
    # Language-specific queries for finding symbol definitions
    SYMBOL_QUERIES = {
        "python": """
            (function_definition
                name: (identifier) @function.name) @function
            (class_definition
                name: (identifier) @class.name) @class
            (assignment
                left: (identifier) @variable.name) @variable
            (assignment
                left: (attribute
                    object: (identifier) @object
                    attribute: (identifier) @attribute.name)) @attribute
        """,
        "javascript": """
            (function_declaration
                name: (identifier) @function.name) @function
            (variable_declarator
                name: (identifier) @variable.name) @variable
            (class_declaration
                name: (identifier) @class.name) @class
            (method_definition
                name: (property_identifier) @method.name) @method
        """,
        "typescript": """
            (function_declaration
                name: (identifier) @function.name) @function
            (variable_declarator
                name: (identifier) @variable.name) @variable
            (class_declaration
                name: (identifier) @class.name) @class
            (method_definition
                name: (property_identifier) @method.name) @method
            (interface_declaration
                name: (type_identifier) @interface.name) @interface
        """,
        "go": """
            (function_declaration
                name: (identifier) @function.name) @function
            (method_declaration
                name: (field_identifier) @method.name
                receiver: (parameter_list
                    (parameter_declaration
                        type: (pointer_type
                            (type_identifier) @receiver.type)?
                        type: (type_identifier) @receiver.type?))) @method
            (type_declaration
                (type_spec
                    name: (type_identifier) @type.name)) @type
        """,
        "rust": """
            (function_item
                name: (identifier) @function.name) @function
            (impl_item
                type: (type_identifier) @impl.type
                body: (declaration_list
                    (function_item
                        name: (identifier) @method.name))) @impl
            (struct_item
                name: (type_identifier) @struct.name) @struct
            (trait_item
                name: (type_identifier) @trait.name) @trait
        """,
    }
    
    # Language-specific queries for finding references
    REFERENCE_QUERIES = {
        "python": """
            (call
                function: (identifier) @call.name)
            (call
                function: (attribute
                    attribute: (identifier) @method_call.name))
            (import_from_statement
                name: (dotted_name) @import.module)
            (import_from_statement
                (aliased_import
                    name: (identifier) @import.name))
        """,
        "javascript": """
            (call_expression
                function: (identifier) @call.name)
            (call_expression
                function: (member_expression
                    property: (property_identifier) @method_call.name))
            (import_specifier
                (identifier) @import.name)
        """,
        "typescript": """
            (call_expression
                function: (identifier) @call.name)
            (call_expression
                function: (member_expression
                    property: (property_identifier) @method_call.name))
            (import_specifier
                (identifier) @import.name)
            (type_identifier) @type_reference
        """,
    }
    
    def __init__(self, engine):
        """Initialize with a BreezeEngine instance."""
        self.engine = engine
        self._parsers: Dict[str, Any] = {}
        self._ts_languages = tree_sitter_language_pack
    
    def _get_parser(self, language: str):
        """Get or create a parser for a language."""
        if language not in self._parsers:
            try:
                self._parsers[language] = self._ts_languages.get_parser(language)
            except Exception as e:
                logger.warning(f"Failed to get parser for {language}: {e}")
                return None
        return self._parsers[language]
    
    async def find_symbol(self, symbol_name: str, symbol_type: Optional[str] = None) -> List[Symbol]:
        """
        Find all definitions of a symbol across the indexed codebase.
        
        Args:
            symbol_name: Name of the symbol to find
            symbol_type: Optional type filter ('function', 'class', 'method', etc.)
            
        Returns:
            List of Symbol objects representing all definitions found
        """
        # Search for files containing the symbol name
        search_results = await self.engine.search(symbol_name, limit=50)
        
        symbols = []
        for result in search_results:
            file_path = result.file_path
            
            # Get file content
            try:
                content = await self._get_file_content(file_path)
                if not content:
                    continue
                    
                # Detect language
                language = detect_language(Path(file_path))
                if not language or language not in self.SYMBOL_QUERIES:
                    continue
                
                # Parse and find symbols
                file_symbols = await self._extract_symbols_from_file(
                    content, file_path, language, symbol_name, symbol_type
                )
                symbols.extend(file_symbols)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        return symbols
    
    async def find_references(self, symbol_name: str, file_filter: Optional[str] = None) -> List[Reference]:
        """
        Find all references to a symbol.
        
        Args:
            symbol_name: Name of the symbol to find references for
            file_filter: Optional file path pattern to limit search
            
        Returns:
            List of Reference objects
        """
        # Search for files containing the symbol
        search_results = await self.engine.search(symbol_name, limit=100)
        
        references = []
        for result in search_results:
            file_path = result.file_path
            
            # Apply file filter if specified
            if file_filter and file_filter not in file_path:
                continue
            
            try:
                content = await self._get_file_content(file_path)
                if not content:
                    continue
                
                # Detect language
                language = detect_language(Path(file_path))
                if not language or language not in self.REFERENCE_QUERIES:
                    continue
                
                # Parse and find references
                file_refs = await self._extract_references_from_file(
                    content, file_path, language, symbol_name
                )
                references.extend(file_refs)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        return references
    
    async def find_implementations(self, interface_name: str, method_name: Optional[str] = None) -> List[Implementation]:
        """
        Find implementations of an interface or abstract class.
        
        Args:
            interface_name: Name of the interface/abstract class
            method_name: Optional specific method to find implementations of
            
        Returns:
            List of Implementation objects
        """
        # This is language-specific, focusing on common patterns
        implementations = []
        
        # Search for files that might contain implementations
        search_query = f"{interface_name} implements extends"
        search_results = await self.engine.search(search_query, limit=50)
        
        for result in search_results:
            file_path = result.file_path
            
            try:
                content = await self._get_file_content(file_path)
                if not content:
                    continue
                
                # Detect language
                language = detect_language(Path(file_path))
                if not language:
                    continue
                
                # Extract implementations based on language
                file_impls = await self._extract_implementations_from_file(
                    content, file_path, language, interface_name, method_name
                )
                implementations.extend(file_impls)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        return implementations
    
    async def get_symbol_hierarchy(self, file_path: str) -> Dict[str, Any]:
        """
        Get the symbol hierarchy for a file (outline view).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hierarchical structure of symbols in the file
        """
        try:
            content = await self._get_file_content(file_path)
            if not content:
                return {}
            
            language = detect_language(Path(file_path))
            if not language:
                return {}
            
            return await self._build_symbol_hierarchy(content, file_path, language)
            
        except Exception as e:
            logger.error(f"Error building symbol hierarchy for {file_path}: {e}")
            return {}
    
    async def _get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a file from the index."""
        # Query the table for the specific file
        try:
            results = await self.engine.table.query().where(f"file_path = '{file_path}'").limit(1).to_list()
            if results:
                return results[0].get('content')
        except Exception as e:
            logger.error(f"Error fetching content for {file_path}: {e}")
        return None
    
    async def _extract_symbols_from_file(
        self, 
        content: str, 
        file_path: str, 
        language: str,
        target_name: str,
        symbol_type: Optional[str] = None
    ) -> List[Symbol]:
        """Extract symbol definitions from a file."""
        parser = self._get_parser(language)
        if not parser:
            return []
        
        tree = parser.parse(content.encode())
        symbols = []
        
        # Get query for this language
        query_pattern = self.SYMBOL_QUERIES.get(language, "")
        if not query_pattern:
            return []
        
        try:
            lang_obj = self._ts_languages.get_language(language)
            query = Query(lang_obj, query_pattern)
            
            # Execute query
            matches = query.matches(tree.root_node)
            
            for pattern_idx, captures in matches:
                symbol = self._create_symbol_from_captures(
                    captures, content, file_path, target_name, symbol_type
                )
                if symbol:
                    symbols.append(symbol)
                    
        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")
        
        return symbols
    
    async def _extract_references_from_file(
        self,
        content: str,
        file_path: str,
        language: str,
        target_name: str
    ) -> List[Reference]:
        """Extract references from a file."""
        parser = self._get_parser(language)
        if not parser:
            return []
        
        tree = parser.parse(content.encode())
        references = []
        
        query_pattern = self.REFERENCE_QUERIES.get(language, "")
        if not query_pattern:
            return []
        
        try:
            lang_obj = self._ts_languages.get_language(language)
            query = Query(lang_obj, query_pattern)
            
            matches = query.matches(tree.root_node)
            
            for pattern_idx, captures in matches:
                ref = self._create_reference_from_captures(
                    captures, content, file_path, target_name
                )
                if ref:
                    references.append(ref)
                    
        except Exception as e:
            logger.error(f"Error extracting references: {e}")
        
        return references
    
    async def _extract_implementations_from_file(
        self,
        content: str,
        file_path: str,
        language: str,
        interface_name: str,
        method_name: Optional[str] = None
    ) -> List[Implementation]:
        """Extract implementations from a file."""
        implementations = []
        
        # Language-specific implementation patterns
        if language == "java":
            implementations.extend(
                await self._extract_java_implementations(
                    content, file_path, interface_name, method_name
                )
            )
        elif language == "typescript":
            implementations.extend(
                await self._extract_typescript_implementations(
                    content, file_path, interface_name, method_name
                )
            )
        elif language == "go":
            implementations.extend(
                await self._extract_go_implementations(
                    content, file_path, interface_name, method_name
                )
            )
        elif language == "rust":
            implementations.extend(
                await self._extract_rust_implementations(
                    content, file_path, interface_name, method_name
                )
            )
        
        return implementations
    
    def _create_symbol_from_captures(
        self,
        captures: Dict[str, List[Node]],
        content: str,
        file_path: str,
        target_name: str,
        symbol_type: Optional[str] = None
    ) -> Optional[Symbol]:
        """Create a Symbol object from tree-sitter captures."""
        # Extract different types of symbols
        for capture_name, nodes in captures.items():
            if not nodes:
                continue
                
            node = nodes[0]  # Take first match
            
            # Check if this is a name node
            if capture_name.endswith('.name'):
                name = content[node.start_byte:node.end_byte]
                
                # Check if name matches target
                if name != target_name:
                    continue
                
                # Determine symbol type from capture name
                sym_type = capture_name.split('.')[0]
                
                # Apply type filter if specified
                if symbol_type and sym_type != symbol_type:
                    continue
                
                # Get parent node for full context
                parent_key = sym_type
                parent_nodes = captures.get(parent_key, [])
                if not parent_nodes:
                    continue
                    
                parent_node = parent_nodes[0]
                
                # Calculate line/column positions
                start_point = parent_node.start_point
                end_point = parent_node.end_point
                
                # Extract signature for functions/methods
                signature = None
                if sym_type in ['function', 'method']:
                    signature = self._extract_function_signature(
                        parent_node, content
                    )
                
                return Symbol(
                    name=name,
                    type=sym_type,
                    file_path=file_path,
                    start_line=start_point[0] + 1,
                    start_column=start_point[1],
                    end_line=end_point[0] + 1,
                    end_column=end_point[1],
                    signature=signature
                )
        
        return None
    
    def _create_reference_from_captures(
        self,
        captures: Dict[str, List[Node]],
        content: str,
        file_path: str,
        target_name: str
    ) -> Optional[Reference]:
        """Create a Reference object from tree-sitter captures."""
        for capture_name, nodes in captures.items():
            if not nodes:
                continue
                
            node = nodes[0]
            name = content[node.start_byte:node.end_byte]
            
            # Check if name matches target
            if name != target_name:
                continue
            
            # Determine reference type from capture name
            ref_type = capture_name.split('.')[0]
            
            # Get line of code for context
            lines = content.split('\n')
            line_num = node.start_point[0]
            context = lines[line_num] if line_num < len(lines) else ""
            
            return Reference(
                symbol_name=name,
                file_path=file_path,
                line=line_num + 1,
                column=node.start_point[1],
                context=context.strip(),
                reference_type=ref_type
            )
        
        return None
    
    def _extract_function_signature(self, node: Node, content: str) -> str:
        """Extract function signature from a function node."""
        # Look for parameter list child
        for child in node.children:
            if child.type in ['parameters', 'formal_parameters', 'parameter_list']:
                # Get from function name to end of parameters
                start = node.start_byte
                end = child.end_byte
                signature = content[start:end]
                # Clean up whitespace
                return ' '.join(signature.split())
        
        # Fallback to first line
        start = node.start_byte
        end = content.find('\n', start)
        if end == -1:
            end = node.end_byte
        return content[start:end].strip()
    
    async def _build_symbol_hierarchy(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> Dict[str, Any]:
        """Build a hierarchical structure of symbols in a file."""
        parser = self._get_parser(language)
        if not parser:
            return {}
        
        tree = parser.parse(content.encode())
        
        # Build hierarchy by traversing the tree
        hierarchy = {
            "file": file_path,
            "language": language,
            "symbols": []
        }
        
        def traverse_node(node: Node, parent_context: Optional[str] = None):
            """Traverse tree and build symbol hierarchy."""
            # Check if this node represents a symbol
            symbol_info = self._node_to_symbol_info(node, content, language)
            
            if symbol_info:
                # Add to appropriate level
                if parent_context:
                    symbol_info["parent"] = parent_context
                
                # Process children
                children = []
                for child in node.children:
                    child_symbols = traverse_node(
                        child, 
                        symbol_info.get("name", parent_context)
                    )
                    if child_symbols:
                        children.extend(child_symbols)
                
                if children:
                    symbol_info["children"] = children
                
                return [symbol_info]
            else:
                # Not a symbol node, continue traversing
                all_symbols = []
                for child in node.children:
                    child_symbols = traverse_node(child, parent_context)
                    if child_symbols:
                        all_symbols.extend(child_symbols)
                return all_symbols
        
        hierarchy["symbols"] = traverse_node(tree.root_node)
        return hierarchy
    
    def _node_to_symbol_info(
        self,
        node: Node,
        content: str,
        language: str
    ) -> Optional[Dict[str, Any]]:
        """Convert a tree-sitter node to symbol information."""
        # Language-specific symbol detection
        symbol_types = {
            "python": {
                "function_definition": "function",
                "class_definition": "class",
                "decorated_definition": "function"
            },
            "javascript": {
                "function_declaration": "function",
                "class_declaration": "class",
                "method_definition": "method",
                "variable_declarator": "variable"
            },
            "typescript": {
                "function_declaration": "function",
                "class_declaration": "class",
                "interface_declaration": "interface",
                "method_definition": "method",
                "type_alias_declaration": "type"
            },
            "go": {
                "function_declaration": "function",
                "method_declaration": "method",
                "type_declaration": "type"
            },
            "rust": {
                "function_item": "function",
                "struct_item": "struct",
                "enum_item": "enum",
                "trait_item": "trait",
                "impl_item": "impl"
            }
        }
        
        lang_symbols = symbol_types.get(language, {})
        
        if node.type in lang_symbols:
            # Extract name
            name = self._extract_node_name(node, content, language)
            if not name:
                return None
            
            return {
                "name": name,
                "type": lang_symbols[node.type],
                "line": node.start_point[0] + 1,
                "column": node.start_point[1],
                "end_line": node.end_point[0] + 1,
                "end_column": node.end_point[1]
            }
        
        return None
    
    def _extract_node_name(self, node: Node, content: str, language: str) -> Optional[str]:
        """Extract the name from a symbol node."""
        # Look for identifier child nodes
        for child in node.children:
            if child.type in ['identifier', 'type_identifier', 'property_identifier', 'field_identifier']:
                return content[child.start_byte:child.end_byte]
        
        # Language-specific patterns
        if language == "python" and node.type == "function_definition":
            # Python function name is the first identifier after 'def'
            for i, child in enumerate(node.children):
                if child.type == "def" and i + 1 < len(node.children):
                    next_child = node.children[i + 1]
                    if next_child.type == "identifier":
                        return content[next_child.start_byte:next_child.end_byte]
        
        return None
    
    # Language-specific implementation extractors
    
    async def _extract_java_implementations(
        self,
        content: str,
        file_path: str,
        interface_name: str,
        method_name: Optional[str] = None
    ) -> List[Implementation]:
        """Extract Java implementations."""
        implementations = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Look for class declarations with implements/extends
            if 'class' in line and (f'implements {interface_name}' in line or 
                                   f'extends {interface_name}' in line):
                # Extract class name
                class_match = self._extract_java_class_name(line)
                if class_match:
                    if method_name:
                        # Look for specific method implementation
                        method_impls = self._find_method_in_class(
                            lines, i, class_match, method_name
                        )
                        for method_line, method_col in method_impls:
                            implementations.append(Implementation(
                                interface_name=interface_name,
                                method_name=method_name,
                                file_path=file_path,
                                line=method_line,
                                column=method_col,
                                class_name=class_match
                            ))
                    else:
                        # Just record the class implementation
                        implementations.append(Implementation(
                            interface_name=interface_name,
                            method_name="*",
                            file_path=file_path,
                            line=i + 1,
                            column=0,
                            class_name=class_match
                        ))
        
        return implementations
    
    async def _extract_typescript_implementations(
        self,
        content: str,
        file_path: str,
        interface_name: str,
        method_name: Optional[str] = None
    ) -> List[Implementation]:
        """Extract TypeScript implementations."""
        parser = self._get_parser("typescript")
        if not parser:
            return []
        
        tree = parser.parse(content.encode())
        implementations = []
        
        # TypeScript-specific query for finding implementations
        impl_query = f"""
            (class_declaration
                name: (type_identifier) @class.name
                body: (class_body) @class.body) @class
        """
        
        try:
            lang_obj = self._ts_languages.get_language("typescript")
            query = Query(lang_obj, impl_query)
            matches = query.matches(tree.root_node)
            
            for pattern_idx, captures in matches:
                class_node = captures.get("class", [None])[0]
                class_name_node = captures.get("class.name", [None])[0]
                
                if not class_node or not class_name_node:
                    continue
                
                # Check if this class implements the interface
                class_text = content[class_node.start_byte:class_node.end_byte]
                if f"implements {interface_name}" in class_text:
                    class_name = content[class_name_node.start_byte:class_name_node.end_byte]
                    
                    implementations.append(Implementation(
                        interface_name=interface_name,
                        method_name=method_name or "*",
                        file_path=file_path,
                        line=class_node.start_point[0] + 1,
                        column=class_node.start_point[1],
                        class_name=class_name
                    ))
                    
        except Exception as e:
            logger.error(f"Error extracting TypeScript implementations: {e}")
        
        return implementations
    
    async def _extract_go_implementations(
        self,
        content: str,
        file_path: str,
        interface_name: str,
        method_name: Optional[str] = None
    ) -> List[Implementation]:
        """Extract Go implementations (via receiver methods)."""
        parser = self._get_parser("go")
        if not parser:
            return []
        
        implementations = []
        # In Go, we look for methods with receivers that might implement an interface
        # This is a simplified approach - full analysis would require type checking
        
        tree = parser.parse(content.encode())
        
        # Query for method declarations
        method_query = """
            (method_declaration
                name: (field_identifier) @method.name
                receiver: (parameter_list
                    (parameter_declaration
                        type: (pointer_type
                            (type_identifier) @receiver.type)?
                        type: (type_identifier) @receiver.type?))) @method
        """
        
        try:
            lang_obj = self._ts_languages.get_language("go")
            query = Query(lang_obj, method_query)
            matches = query.matches(tree.root_node)
            
            for pattern_idx, captures in matches:
                method_name_node = captures.get("method.name", [None])[0]
                receiver_nodes = captures.get("receiver.type", [])
                
                if not method_name_node or not receiver_nodes:
                    continue
                
                method = content[method_name_node.start_byte:method_name_node.end_byte]
                receiver = content[receiver_nodes[0].start_byte:receiver_nodes[0].end_byte]
                
                # This is a heuristic - we'd need full type analysis to be sure
                implementations.append(Implementation(
                    interface_name=interface_name,
                    method_name=method,
                    file_path=file_path,
                    line=method_name_node.start_point[0] + 1,
                    column=method_name_node.start_point[1],
                    class_name=receiver
                ))
                
        except Exception as e:
            logger.error(f"Error extracting Go implementations: {e}")
        
        return implementations
    
    async def _extract_rust_implementations(
        self,
        content: str,
        file_path: str,
        interface_name: str,
        method_name: Optional[str] = None
    ) -> List[Implementation]:
        """Extract Rust trait implementations."""
        parser = self._get_parser("rust")
        if not parser:
            return []
        
        tree = parser.parse(content.encode())
        implementations = []
        
        # Rust-specific query for impl blocks
        impl_query = """
            (impl_item
                trait: (type_identifier) @trait.name
                type: (type_identifier) @type.name
                body: (declaration_list) @impl.body) @impl
        """
        
        try:
            lang_obj = self._ts_languages.get_language("rust")
            query = Query(lang_obj, impl_query)
            matches = query.matches(tree.root_node)
            
            for pattern_idx, captures in matches:
                trait_node = captures.get("trait.name", [None])[0]
                type_node = captures.get("type.name", [None])[0]
                impl_node = captures.get("impl", [None])[0]
                
                if not trait_node or not type_node or not impl_node:
                    continue
                
                trait_name = content[trait_node.start_byte:trait_node.end_byte]
                
                if trait_name == interface_name:
                    type_name = content[type_node.start_byte:type_node.end_byte]
                    
                    implementations.append(Implementation(
                        interface_name=interface_name,
                        method_name=method_name or "*",
                        file_path=file_path,
                        line=impl_node.start_point[0] + 1,
                        column=impl_node.start_point[1],
                        class_name=type_name
                    ))
                    
        except Exception as e:
            logger.error(f"Error extracting Rust implementations: {e}")
        
        return implementations
    
    def _extract_java_class_name(self, line: str) -> Optional[str]:
        """Extract class name from a Java class declaration line."""
        import re
        match = re.search(r'class\s+(\w+)', line)
        return match.group(1) if match else None
    
    def _find_method_in_class(
        self,
        lines: List[str],
        class_line: int,
        class_name: str,
        method_name: str
    ) -> List[Tuple[int, int]]:
        """Find method implementations within a class."""
        methods = []
        brace_count = 0
        in_class = False
        
        for i in range(class_line, len(lines)):
            line = lines[i]
            
            # Track braces to know when we exit the class
            if '{' in line:
                brace_count += line.count('{')
                in_class = True
            if '}' in line:
                brace_count -= line.count('}')
                if in_class and brace_count == 0:
                    break
            
            # Look for method
            if in_class and method_name in line and '(' in line:
                # Simple heuristic - could be improved with proper parsing
                col = line.find(method_name)
                methods.append((i + 1, col))
        
        return methods
