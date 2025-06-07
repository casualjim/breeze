"""Tree-sitter query definitions for semantic code understanding."""

# Query patterns for different languages to find semantic units
# These queries identify functions, methods, classes, and other top-level constructs

LANGUAGE_QUERIES = {
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
    
    # Fallback for languages without specific queries
    "default": """
        (function_definition) @function
        (method_definition) @method
        (class_definition) @class
        (class_declaration) @class
    """
}


def get_query_for_language(language: str) -> str:
    """Get the tree-sitter query pattern for a specific language.
    
    Args:
        language: The programming language name
        
    Returns:
        The query pattern string for that language
    """
    # Normalize language name
    language = language.lower()
    
    # Handle common aliases
    language_aliases = {
        "py": "python",
        "js": "javascript", 
        "ts": "typescript",
        "jsx": "javascript",
        "tsx": "typescript",
        "c++": "cpp",
        "c#": "csharp",
        "objective-c": "objc",
        "objective-c++": "objcpp",
    }
    
    language = language_aliases.get(language, language)
    
    # Return specific query or default
    return LANGUAGE_QUERIES.get(language, LANGUAGE_QUERIES["default"])