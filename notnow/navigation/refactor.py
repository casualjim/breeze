'{')
            brace_count -= lines[i].count('}')
            if brace_count < 0:
                end = i
                break
        
        return (start, end)
    
    def _find_node_at_point(self, node: Node, point: Tuple[int, int]) -> Optional[Node]:
        """Find the deepest node containing the given point."""
        if (node.start_point[0] <= point[0] <= node.end_point[0]):
            # Check children
            for child in node.children:
                result = self._find_node_at_point(child, point)
                if result:
                    return result
            return node
        return None
    
    def _find_enclosing_scope(self, node: Node) -> Optional[Node]:
        """Find the enclosing scope (function/method/block) of a node."""
        current = node
        while current:
            if current.type in [
                'function_definition', 'function_declaration',
                'method_definition', 'method_declaration',
                'block_statement', 'compound_statement',
                'class_definition', 'class_declaration'
            ]:
                return current
            current = current.parent
        return None
    
    def _find_expression_occurrences(
        self,
        lines: List[str],
        expression: str,
        start_line: int,
        end_line: int
    ) -> List[Tuple[int, int]]:
        """Find all occurrences of an expression in a scope."""
        occurrences = []
        
        for i in range(start_line - 1, min(end_line, len(lines))):
            line = lines[i]
            col = 0
            while True:
                pos = line.find(expression, col)
                if pos == -1:
                    break
                occurrences.append((i + 1, pos))
                col = pos + 1
        
        return occurrences
    
    def _find_declaration_position(
        self,
        lines: List[str],
        current_line: int,
        scope_start: int,
        language: str
    ) -> int:
        """Find the best position to insert a variable declaration."""
        # For now, simple heuristic: insert at the beginning of the scope
        # after any existing declarations
        
        insert_line = scope_start
        
        # Skip past existing variable declarations
        for i in range(scope_start - 1, min(current_line, len(lines))):
            line = lines[i].strip()
            
            # Check for variable declarations (language-specific)
            if language == "python":
                if '=' in line and not any(kw in line for kw in ['def ', 'class ', 'if ', 'for ', 'while ']):
                    insert_line = i + 2
            elif language in ["javascript", "typescript"]:
                if any(line.startswith(kw) for kw in ['const ', 'let ', 'var ']):
                    insert_line = i + 2
            else:
                # Generic
                if any(kw in line for kw in ['var ', 'let ', 'const ', ':=']):
                    insert_line = i + 2
        
        return insert_line
    
    async def _find_variable_declaration(
        self,
        content: str,
        variable_name: str,
        line_hint: Optional[int],
        language: str
    ) -> Optional[Tuple[int, str, int, int]]:
        """Find a variable declaration and its scope."""
        lines = content.split('\n')
        
        # Search for declaration pattern
        declaration_patterns = {
            "python": rf"^\s*{variable_name}\s*=\s*(.+)$",
            "javascript": rf"^\s*(const|let|var)\s+{variable_name}\s*=\s*(.+);?$",
            "typescript": rf"^\s*(const|let|var)\s+{variable_name}\s*=\s*(.+);?$",
            "go": rf"^\s*{variable_name}\s*:=\s*(.+)$",
            "rust": rf"^\s*let\s+(mut\s+)?{variable_name}\s*=\s*(.+);?$",
        }
        
        pattern = declaration_patterns.get(language, rf"^\s*\w*\s*{variable_name}\s*=\s*(.+)$")
        
        # Start searching from hint line or beginning
        start = (line_hint - 1) if line_hint else 0
        
        for i in range(start, len(lines)):
            match = re.match(pattern, lines[i])
            if match:
                # Extract value
                if language in ["javascript", "typescript"]:
                    value = match.group(2).rstrip(';').strip()
                elif language == "rust":
                    value = match.group(2).rstrip(';').strip()
                else:
                    value = match.group(1).strip()
                
                # Find scope
                scope_start, scope_end = await self._find_current_scope(
                    content, i + 1, language
                )
                
                return (i + 1, value, scope_start, scope_end)
        
        return None
    
    def _find_variable_uses(
        self,
        lines: List[str],
        variable_name: str,
        scope_start: int,
        scope_end: int,
        declaration_line: int
    ) -> List[Tuple[int, int]]:
        """Find all uses of a variable in its scope."""
        uses = []
        
        # Word boundary pattern to match whole variable names
        pattern = rf'\b{re.escape(variable_name)}\b'
        
        for i in range(scope_start - 1, min(scope_end, len(lines))):
            if i + 1 == declaration_line:
                continue  # Skip the declaration line
            
            line = lines[i]
            for match in re.finditer(pattern, line):
                uses.append((i + 1, match.start()))
        
        return uses
    
    async def apply_refactoring_plan(
        self,
        plan: RefactoringPlan,
        dry_run: bool = True
    ) -> Dict[str, str]:
        """
        Apply a refactoring plan to the codebase.
        
        Args:
            plan: The refactoring plan to apply
            dry_run: If True, return the changes without applying them
            
        Returns:
            Dictionary mapping file paths to their new content
        """
        results = {}
        
        # Group changes by file
        changes_by_file = defaultdict(list)
        for change in plan.changes:
            changes_by_file[change.file_path].append(change)
        
        for file_path, file_changes in changes_by_file.items():
            # Get current content
            content = await self._get_file_content(file_path)
            if not content:
                logger.error(f"Could not read file: {file_path}")
                continue
            
            # Sort changes by position (reverse order for safe application)
            file_changes.sort(
                key=lambda c: (c.start_line, c.start_column), 
                reverse=True
            )
            
            # Apply changes
            lines = content.split('\n')
            
            for change in file_changes:
                # Apply the change
                if change.start_line == change.end_line:
                    # Single line change
                    line = lines[change.start_line - 1]
                    new_line = (
                        line[:change.start_column] + 
                        change.new_text + 
                        line[change.end_column:]
                    )
                    lines[change.start_line - 1] = new_line
                else:
                    # Multi-line change
                    # Remove old lines
                    del lines[change.start_line - 1:change.end_line]
                    # Insert new text
                    new_lines = change.new_text.split('\n')
                    for i, new_line in enumerate(new_lines):
                        lines.insert(change.start_line - 1 + i, new_line)
            
            new_content = '\n'.join(lines)
            results[file_path] = new_content
            
            if not dry_run:
                # Actually write the changes
                try:
                    path = Path(file_path)
                    path.write_text(new_content)
                    logger.info(f"Updated file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to write file {file_path}: {e}")
        
        return results
