"""
Tool Execution Engine for SAM
Executes selected tools and integrates responses with standardized ToolResponse objects.

Sprint 5 Task 3: Tool Execution & Response Integration
"""

import logging
import subprocess
import tempfile
import json
import io
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

@dataclass
class ToolResponse:
    """Standardized response object for tool execution."""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str]
    metadata: Dict[str, Any]
    execution_time_ms: int
    timestamp: str
    provenance: Dict[str, Any]

@dataclass
class ExecutionContext:
    """Context for tool execution."""
    session_id: str
    query: str
    user_settings: Dict[str, Any]
    safety_mode: bool
    timeout_seconds: int

class ToolExecutor:
    """
    Tool execution engine with support for sequential and nested tool use.
    """
    
    def __init__(self, safety_mode: bool = True, default_timeout: int = 30):
        """
        Initialize the tool executor.
        
        Args:
            safety_mode: Enable safety restrictions for code execution
            default_timeout: Default timeout for tool execution in seconds
        """
        self.safety_mode = safety_mode
        self.default_timeout = default_timeout
        
        # Tool execution history
        self.execution_history: List[ToolResponse] = []
        
        # Safety restrictions for Python execution
        self.python_restrictions = {
            'forbidden_imports': [
                'os', 'subprocess', 'sys', 'shutil', 'glob',
                'socket', 'urllib', 'requests', 'http',
                'ftplib', 'smtplib', 'telnetlib'
            ],
            'forbidden_functions': [
                'exec', 'eval', 'compile', '__import__',
                'open', 'file', 'input', 'raw_input'
            ],
            'max_execution_time': 30,
            'max_memory_mb': 100
        }
        
        logger.info(f"Tool executor initialized (safety_mode: {safety_mode})")
    
    def execute_tool(self, tool_name: str, input_params: Dict[str, Any], 
                    context: Optional[ExecutionContext] = None) -> ToolResponse:
        """
        Execute a single tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            input_params: Parameters for tool execution
            context: Execution context
            
        Returns:
            ToolResponse object with execution results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing tool: {tool_name}")
            
            # Create default context if not provided
            if context is None:
                context = ExecutionContext(
                    session_id=f"exec_{int(start_time.timestamp())}",
                    query=input_params.get('query', ''),
                    user_settings={},
                    safety_mode=self.safety_mode,
                    timeout_seconds=self.default_timeout
                )
            
            # Route to appropriate tool executor
            if tool_name == 'python_interpreter':
                result = self._execute_python_interpreter(input_params, context)
            elif tool_name == 'table_generator':
                result = self._execute_table_generator(input_params, context)
            elif tool_name == 'multimodal_query':
                result = self._execute_multimodal_query(input_params, context)
            elif tool_name == 'web_search':
                result = self._execute_web_search(input_params, context)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Calculate execution time
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create response object
            response = ToolResponse(
                tool_name=tool_name,
                success=True,
                output=result,
                error=None,
                metadata={
                    'input_params': input_params,
                    'context_id': context.session_id,
                    'safety_mode': context.safety_mode
                },
                execution_time_ms=execution_time,
                timestamp=start_time.isoformat(),
                provenance={
                    'executor': 'SAM_ToolExecutor',
                    'version': '1.0',
                    'query': context.query
                }
            )
            
            # Add to execution history
            self.execution_history.append(response)
            
            logger.info(f"Tool {tool_name} executed successfully ({execution_time}ms)")
            return response
            
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = str(e)
            
            logger.error(f"Tool execution failed for {tool_name}: {error_msg}")
            
            # Create error response
            response = ToolResponse(
                tool_name=tool_name,
                success=False,
                output=None,
                error=error_msg,
                metadata={
                    'input_params': input_params,
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc() if logger.level <= logging.DEBUG else None
                },
                execution_time_ms=execution_time,
                timestamp=start_time.isoformat(),
                provenance={
                    'executor': 'SAM_ToolExecutor',
                    'version': '1.0',
                    'query': context.query if context else 'unknown'
                }
            )
            
            self.execution_history.append(response)
            return response
    
    def execute_tools_sequential(self, tool_configs: List[Dict[str, Any]], 
                                context: Optional[ExecutionContext] = None) -> List[ToolResponse]:
        """
        Execute multiple tools sequentially.
        
        Args:
            tool_configs: List of tool configurations
            context: Execution context
            
        Returns:
            List of ToolResponse objects
        """
        responses = []
        
        for i, tool_config in enumerate(tool_configs):
            tool_name = tool_config.get('tool_name')
            input_params = tool_config.get('input_params', {})
            
            logger.debug(f"Executing tool {i+1}/{len(tool_configs)}: {tool_name}")
            
            # Execute tool
            response = self.execute_tool(tool_name, input_params, context)
            responses.append(response)
            
            # If tool failed and it's critical, stop execution
            if not response.success and tool_config.get('critical', False):
                logger.warning(f"Critical tool {tool_name} failed, stopping sequential execution")
                break
        
        return responses
    
    def execute_tools_nested(self, primary_tool: str, primary_params: Dict[str, Any],
                           secondary_tools: List[Dict[str, Any]], 
                           context: Optional[ExecutionContext] = None) -> Dict[str, Any]:
        """
        Execute tools in nested fashion (e.g., generate → summarize).
        
        Args:
            primary_tool: Primary tool to execute first
            primary_params: Parameters for primary tool
            secondary_tools: Secondary tools to process primary output
            context: Execution context
            
        Returns:
            Dictionary with primary and secondary results
        """
        # Execute primary tool
        primary_response = self.execute_tool(primary_tool, primary_params, context)
        
        results = {
            'primary_tool': primary_tool,
            'primary_response': primary_response,
            'secondary_responses': []
        }
        
        # If primary tool succeeded, execute secondary tools
        if primary_response.success:
            for secondary_config in secondary_tools:
                # Inject primary output into secondary tool parameters
                secondary_params = secondary_config.get('input_params', {}).copy()
                secondary_params['primary_output'] = primary_response.output
                
                secondary_response = self.execute_tool(
                    secondary_config.get('tool_name'),
                    secondary_params,
                    context
                )
                
                results['secondary_responses'].append(secondary_response)
        
        return results
    
    def _execute_python_interpreter(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute Python code interpreter."""
        query = params.get('query', '')
        code = params.get('code')
        
        # If no explicit code provided, try to extract/generate from query
        if not code:
            code = self._extract_or_generate_python_code(query)
        
        if not code:
            raise ValueError("No Python code to execute")
        
        # Safety checks
        if context.safety_mode:
            self._validate_python_code_safety(code)
        
        # Execute code in isolated environment
        result = self._execute_python_code_safely(code, context.timeout_seconds)
        
        return {
            'code_executed': code,
            'output': result.get('output', ''),
            'error': result.get('error'),
            'execution_successful': result.get('success', False),
            'variables': result.get('variables', {}),
            'plots': result.get('plots', [])
        }
    
    def _execute_table_generator(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute table generator."""
        query = params.get('query', '')
        data = params.get('data')
        format_type = params.get('format', 'markdown')
        
        # Extract table requirements from query
        table_spec = self._analyze_table_requirements(query)
        
        # Generate table based on requirements
        if data:
            table_content = self._format_data_as_table(data, format_type, table_spec)
        else:
            table_content = self._generate_table_from_query(query, format_type, table_spec)
        
        return {
            'table_content': table_content,
            'format': format_type,
            'table_spec': table_spec,
            'rows': table_spec.get('estimated_rows', 0),
            'columns': table_spec.get('estimated_columns', 0)
        }
    
    def _execute_multimodal_query(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute multimodal query."""
        query = params.get('query', '')
        search_types = params.get('search_types', ['text', 'code', 'table', 'image'])
        max_results = params.get('max_results', 5)

        # CRITICAL FIX: Use memory store for search instead of separate vector manager
        try:
            from memory.memory_vectorstore import get_memory_store
            memory_store = get_memory_store()

            # Phase 3.2: Use enhanced search for multimodal queries
            try:
                if hasattr(memory_store, 'enhanced_search_memories'):
                    memories = memory_store.enhanced_search_memories(
                        query=query,
                        max_results=max_results * 2,
                        initial_candidates=max_results * 4
                    )
                    logger.info(f"Enhanced multimodal search: {len(memories)} results")
                else:
                    memories = memory_store.search_memories(query, max_results=max_results * 2)
            except Exception as e:
                logger.warning(f"Enhanced search failed in tool executor, using fallback: {e}")
                memories = memory_store.search_memories(query, max_results=max_results * 2)

            search_results = []
            for memory in memories:
                # Convert memory to search result format
                result = {
                    'chunk_id': memory.chunk_id,
                    'content': memory.content,
                    'similarity_score': getattr(memory, 'similarity_score', 0.8),  # Default high score
                    'source': memory.source,
                    'metadata': {
                        'memory_type': memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type),
                        'timestamp': memory.timestamp,
                        'importance_score': memory.importance_score,
                        'tags': memory.tags,
                        **getattr(memory, 'metadata', {})
                    }
                }
                search_results.append(result)

            # Sort by relevance and limit results
            search_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            search_results = search_results[:max_results]

            return {
                'search_results': search_results,
                'total_results': len(search_results),
                'search_types': search_types,
                'query': query,
                'memory_store_used': True,
                'total_memories_searched': len(memories)
            }
            
        except ImportError:
            logger.warning("Multimodal pipeline not available")
            return {
                'search_results': [],
                'total_results': 0,
                'error': 'Multimodal pipeline not available',
                'query': query
            }
    
    def _execute_web_search(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute web search (placeholder implementation)."""
        query = params.get('query', '')
        max_results = params.get('max_results', 5)
        safe_search = params.get('safe_search', True)
        
        # Placeholder implementation - would integrate with actual web search
        logger.warning("Web search is not implemented - returning placeholder results")
        
        return {
            'search_results': [
                {
                    'title': f'Search result for: {query}',
                    'url': 'https://example.com/placeholder',
                    'snippet': f'This is a placeholder search result for the query: {query}',
                    'relevance_score': 0.8
                }
            ],
            'total_results': 1,
            'query': query,
            'note': 'Web search not implemented - placeholder results provided'
        }
    
    def _extract_or_generate_python_code(self, query: str) -> Optional[str]:
        """Extract or generate Python code from query."""
        # Look for code blocks in query
        import re
        
        # Check for explicit code blocks
        code_block_pattern = r'```python\s*(.*?)\s*```'
        code_match = re.search(code_block_pattern, query, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Check for inline code
        inline_code_pattern = r'`([^`]+)`'
        inline_match = re.search(inline_code_pattern, query)
        if inline_match:
            code = inline_match.group(1).strip()
            if any(keyword in code for keyword in ['print', '=', '+', '-', '*', '/']):
                return code
        
        # Generate simple code for mathematical expressions
        math_pattern = r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)'
        math_match = re.search(math_pattern, query)
        if math_match:
            num1, op, num2 = math_match.groups()
            return f"result = {num1} {op} {num2}\nprint(f'Result: {{result}}')"
        
        return None
    
    def _validate_python_code_safety(self, code: str):
        """Validate Python code for safety restrictions."""
        if not self.safety_mode:
            return
        
        # Check for forbidden imports
        for forbidden in self.python_restrictions['forbidden_imports']:
            if f'import {forbidden}' in code or f'from {forbidden}' in code:
                raise ValueError(f"Forbidden import: {forbidden}")
        
        # Check for forbidden functions
        for forbidden in self.python_restrictions['forbidden_functions']:
            if forbidden in code:
                raise ValueError(f"Forbidden function: {forbidden}")
        
        # Check for file operations
        if any(pattern in code for pattern in ['open(', 'file(', 'with open']):
            raise ValueError("File operations not allowed in safety mode")
    
    def _execute_python_code_safely(self, code: str, timeout: int) -> Dict[str, Any]:
        """Execute Python code in a safe environment."""
        try:
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            # Create restricted globals
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted,
                    'reversed': reversed
                }
            }
            
            # Add math functions
            import math
            safe_globals['math'] = math
            
            # Execute code
            local_vars = {}
            exec(code, safe_globals, local_vars)
            
            # Get output
            output = captured_output.getvalue()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            return {
                'success': True,
                'output': output,
                'variables': {k: str(v) for k, v in local_vars.items() if not k.startswith('_')},
                'error': None
            }
            
        except Exception as e:
            # Restore stdout
            sys.stdout = old_stdout
            
            return {
                'success': False,
                'output': '',
                'variables': {},
                'error': str(e)
            }
    
    def _analyze_table_requirements(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine table requirements."""
        query_lower = query.lower()
        
        spec = {
            'table_type': 'comparison',
            'estimated_rows': 5,
            'estimated_columns': 3,
            'headers': [],
            'data_types': []
        }
        
        # Detect table type
        if any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs']):
            spec['table_type'] = 'comparison'
        elif any(word in query_lower for word in ['list', 'items', 'elements']):
            spec['table_type'] = 'list'
        elif any(word in query_lower for word in ['summary', 'overview', 'breakdown']):
            spec['table_type'] = 'summary'
        
        return spec
    
    def _format_data_as_table(self, data: Any, format_type: str, spec: Dict[str, Any]) -> str:
        """Format provided data as a table."""
        if format_type == 'markdown':
            if isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    # List of dictionaries
                    headers = list(data[0].keys())
                    rows = [[str(item.get(header, '')) for header in headers] for item in data]
                    
                    table_lines = []
                    table_lines.append('| ' + ' | '.join(headers) + ' |')
                    table_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
                    
                    for row in rows:
                        table_lines.append('| ' + ' | '.join(row) + ' |')
                    
                    return '\n'.join(table_lines)
                
                elif isinstance(data[0], (list, tuple)):
                    # List of lists/tuples
                    table_lines = []
                    for i, row in enumerate(data):
                        if i == 0:
                            # First row as headers
                            table_lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
                            table_lines.append('| ' + ' | '.join(['---'] * len(row)) + ' |')
                        else:
                            table_lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
                    
                    return '\n'.join(table_lines)
        
        # Fallback: simple string representation
        return str(data)
    
    def _generate_table_from_query(self, query: str, format_type: str, spec: Dict[str, Any]) -> str:
        """Generate a table based on query analysis."""
        # Placeholder implementation
        if format_type == 'markdown':
            return f"""| Item | Description | Value |
| --- | --- | --- |
| Query | {query[:30]}... | Analysis |
| Type | {spec['table_type']} | Generated |
| Format | {format_type} | Markdown |"""
        
        return f"Generated table for: {query}"
    
    def get_execution_history(self) -> List[ToolResponse]:
        """Get tool execution history."""
        return self.execution_history.copy()
    
    def clear_execution_history(self):
        """Clear tool execution history."""
        self.execution_history.clear()
        logger.info("Tool execution history cleared")

# Global tool executor instance
_tool_executor = None

def get_tool_executor(safety_mode: bool = True, default_timeout: int = 30) -> ToolExecutor:
    """Get or create a global tool executor instance."""
    global _tool_executor
    
    if _tool_executor is None:
        _tool_executor = ToolExecutor(safety_mode=safety_mode, default_timeout=default_timeout)
    
    return _tool_executor
