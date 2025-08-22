"""
Static Architecture Analyzer for SAM
====================================

Analyzes SAM's static code architecture to generate interactive blueprints
of the system's structure, modules, and interconnections.

Based on gitdiagram principles for automatic code structure analysis.

Author: SAM Development Team
Version: 1.0.0
"""

import ast
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import importlib.util


logger = logging.getLogger(__name__)


@dataclass
class ModuleNode:
    """Represents a module in the architecture."""
    name: str
    path: str
    module_type: str  # 'package', 'module', 'class', 'function'
    docstring: Optional[str]
    dependencies: List[str]
    exports: List[str]
    line_count: int
    complexity_score: float
    is_core: bool = False


@dataclass
class ArchitectureGraph:
    """Represents the complete architecture graph."""
    nodes: Dict[str, ModuleNode]
    edges: List[Tuple[str, str, str]]  # (from, to, relationship_type)
    metadata: Dict[str, Any]
    generated_at: str


class StaticArchitectureAnalyzer:
    """
    Analyzes SAM's static architecture to create interactive blueprints.
    
    Features:
    - Module dependency analysis
    - Code complexity assessment
    - Interactive graph generation
    - Documentation extraction
    """
    
    def __init__(self, sam_root_path: str):
        """
        Initialize the analyzer.
        
        Args:
            sam_root_path: Path to the SAM root directory
        """
        self.sam_root = Path(sam_root_path)
        self.logger = logging.getLogger(f"{__name__}.StaticArchitectureAnalyzer")
        self.nodes: Dict[str, ModuleNode] = {}
        self.edges: List[Tuple[str, str, str]] = []
        
        # Core SAM modules (these are the most important)
        self.core_modules = {
            'sam.core',
            'sam.memory', 
            'sam.agents',
            'sam.models',
            'sam.embedding',
            'sam.code_interpreter',
            'sam.introspection'
        }
        
    def analyze_architecture(self) -> ArchitectureGraph:
        """
        Perform complete architecture analysis.
        
        Returns:
            ArchitectureGraph: Complete architecture representation
        """
        self.logger.info("🔍 Starting static architecture analysis...")
        
        # Step 1: Discover all Python modules
        self._discover_modules()
        
        # Step 2: Analyze dependencies
        self._analyze_dependencies()
        
        # Step 3: Calculate complexity scores
        self._calculate_complexity()
        
        # Step 4: Identify core modules
        self._identify_core_modules()
        
        # Step 5: Generate graph
        graph = ArchitectureGraph(
            nodes=self.nodes,
            edges=self.edges,
            metadata={
                'total_modules': len(self.nodes),
                'total_dependencies': len(self.edges),
                'core_modules': len([n for n in self.nodes.values() if n.is_core]),
                'analysis_scope': str(self.sam_root),
                'analyzer_version': '1.0.0'
            },
            generated_at=str(datetime.now())
        )
        
        self.logger.info(f"✅ Architecture analysis complete: {len(self.nodes)} modules, {len(self.edges)} dependencies")
        return graph
    
    def _discover_modules(self):
        """Discover all Python modules in the SAM codebase."""
        self.logger.info("📂 Discovering Python modules...")
        
        for py_file in self.sam_root.rglob("*.py"):
            # Skip __pycache__ and other non-source directories
            if '__pycache__' in str(py_file) or '.git' in str(py_file):
                continue
                
            try:
                module_name = self._path_to_module_name(py_file)
                node = self._analyze_module_file(py_file, module_name)
                if node:
                    self.nodes[module_name] = node
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze {py_file}: {e}")
    
    def _path_to_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        # Get relative path from SAM root
        rel_path = file_path.relative_to(self.sam_root)
        
        # Convert to module name
        parts = list(rel_path.parts[:-1])  # Remove filename
        if rel_path.name != '__init__.py':
            parts.append(rel_path.stem)  # Add filename without .py
            
        return '.'.join(parts) if parts else 'root'
    
    def _analyze_module_file(self, file_path: Path, module_name: str) -> Optional[ModuleNode]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract information
            docstring = ast.get_docstring(tree)
            imports = self._extract_imports(tree)
            exports = self._extract_exports(tree)
            line_count = len(content.splitlines())
            
            # Determine module type
            module_type = self._determine_module_type(file_path, tree)
            
            return ModuleNode(
                name=module_name,
                path=str(file_path.relative_to(self.sam_root)),
                module_type=module_type,
                docstring=docstring,
                dependencies=imports,
                exports=exports,
                line_count=line_count,
                complexity_score=0.0,  # Will be calculated later
                is_core=False  # Will be determined later
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path}: {e}")
            return None
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Filter to only SAM-related imports
        sam_imports = [imp for imp in imports if imp.startswith('sam.')]
        return sam_imports
    
    def _extract_exports(self, tree: ast.AST) -> List[str]:
        """Extract exported names from AST."""
        exports = []
        
        # Look for __all__ definition
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and 
                len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id == '__all__'):
                
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Str):
                            exports.append(elt.s)
                        elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exports.append(elt.value)
        
        # If no __all__, extract top-level classes and functions
        if not exports:
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if not node.name.startswith('_'):  # Skip private
                        exports.append(node.name)
        
        return exports
    
    def _determine_module_type(self, file_path: Path, tree: ast.AST) -> str:
        """Determine the type of module."""
        if file_path.name == '__init__.py':
            return 'package'
        
        # Count different types of definitions
        classes = sum(1 for node in tree.body if isinstance(node, ast.ClassDef))
        functions = sum(1 for node in tree.body if isinstance(node, ast.FunctionDef))
        
        if classes > functions:
            return 'class_module'
        elif functions > 0:
            return 'function_module'
        else:
            return 'module'
    
    def _analyze_dependencies(self):
        """Analyze dependencies between modules."""
        self.logger.info("🔗 Analyzing module dependencies...")
        
        for module_name, node in self.nodes.items():
            for dependency in node.dependencies:
                if dependency in self.nodes:
                    self.edges.append((module_name, dependency, 'imports'))
    
    def _calculate_complexity(self):
        """Calculate complexity scores for modules."""
        self.logger.info("📊 Calculating complexity scores...")
        
        for node in self.nodes.values():
            # Simple complexity based on lines of code and dependencies
            complexity = (
                node.line_count * 0.1 +
                len(node.dependencies) * 2.0 +
                len(node.exports) * 1.0
            )
            node.complexity_score = min(complexity, 100.0)  # Cap at 100
    
    def _identify_core_modules(self):
        """Identify core SAM modules."""
        self.logger.info("🎯 Identifying core modules...")
        
        for module_name, node in self.nodes.items():
            # Check if module is in core modules list
            for core_pattern in self.core_modules:
                if module_name.startswith(core_pattern):
                    node.is_core = True
                    break
    
    def generate_mermaid_diagram(self, graph: ArchitectureGraph, focus_core: bool = True) -> str:
        """
        Generate Mermaid diagram representation of the architecture.
        
        Args:
            graph: Architecture graph to visualize
            focus_core: If True, only show core modules and their immediate connections
            
        Returns:
            str: Mermaid diagram syntax
        """
        self.logger.info("🎨 Generating Mermaid diagram...")
        
        # Filter nodes if focusing on core
        if focus_core:
            relevant_nodes = {name: node for name, node in graph.nodes.items() if node.is_core}
            # Add nodes that are connected to core modules
            for edge in graph.edges:
                from_node, to_node, _ = edge
                if from_node in relevant_nodes or to_node in relevant_nodes:
                    if from_node in graph.nodes:
                        relevant_nodes[from_node] = graph.nodes[from_node]
                    if to_node in graph.nodes:
                        relevant_nodes[to_node] = graph.nodes[to_node]
        else:
            relevant_nodes = graph.nodes
        
        # Generate Mermaid syntax
        mermaid = ["graph TD"]
        
        # Add nodes with styling
        for name, node in relevant_nodes.items():
            safe_name = name.replace('.', '_').replace('-', '_')
            display_name = name.split('.')[-1] if '.' in name else name
            
            # Style based on module type and importance
            if node.is_core:
                style_class = "core-module"
                shape = f"{safe_name}[{display_name}]"
            elif node.module_type == 'package':
                style_class = "package-module"
                shape = f"{safe_name}({display_name})"
            else:
                style_class = "regular-module"
                shape = f"{safe_name}[{display_name}]"
            
            mermaid.append(f"    {shape}")
        
        # Add edges
        for edge in graph.edges:
            from_node, to_node, relationship = edge
            if from_node in relevant_nodes and to_node in relevant_nodes:
                safe_from = from_node.replace('.', '_').replace('-', '_')
                safe_to = to_node.replace('.', '_').replace('-', '_')
                mermaid.append(f"    {safe_from} --> {safe_to}")
        
        # Add styling
        mermaid.extend([
            "",
            "    classDef core-module fill:#ff6b6b,stroke:#333,stroke-width:3px,color:#fff",
            "    classDef package-module fill:#4ecdc4,stroke:#333,stroke-width:2px,color:#fff", 
            "    classDef regular-module fill:#45b7d1,stroke:#333,stroke-width:1px,color:#fff"
        ])
        
        return '\n'.join(mermaid)


# Import datetime for timestamp
from datetime import datetime
