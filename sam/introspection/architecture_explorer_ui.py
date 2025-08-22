"""
Architecture Explorer UI for SAM
================================

Web-based interactive interface for exploring SAM's static architecture.
Provides real-time visualization of modules, dependencies, and code structure.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, render_template_string, jsonify, request
from dataclasses import asdict

from .static_architecture_analyzer import StaticArchitectureAnalyzer, ArchitectureGraph


logger = logging.getLogger(__name__)


class ArchitectureExplorerUI:
    """
    Web-based UI for exploring SAM's architecture.
    
    Features:
    - Interactive Mermaid.js diagrams
    - Module detail views
    - Dependency exploration
    - Code navigation links
    """
    
    def __init__(self, sam_root_path: str, port: int = 5001):
        """
        Initialize the Architecture Explorer UI.
        
        Args:
            sam_root_path: Path to SAM root directory
            port: Port to run the web server on
        """
        self.sam_root = Path(sam_root_path)
        self.port = port
        self.analyzer = StaticArchitectureAnalyzer(sam_root_path)
        self.app = Flask(__name__)
        self.architecture_graph: Optional[ArchitectureGraph] = None
        
        self.logger = logging.getLogger(f"{__name__}.ArchitectureExplorerUI")
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes for the UI."""
        
        @self.app.route('/')
        def index():
            """Main architecture explorer page."""
            return render_template_string(self._get_main_template())
        
        @self.app.route('/api/architecture')
        def get_architecture():
            """API endpoint to get architecture data."""
            if not self.architecture_graph:
                self.architecture_graph = self.analyzer.analyze_architecture()
            
            # Convert to JSON-serializable format
            return jsonify({
                'nodes': {name: asdict(node) for name, node in self.architecture_graph.nodes.items()},
                'edges': self.architecture_graph.edges,
                'metadata': self.architecture_graph.metadata
            })
        
        @self.app.route('/api/mermaid')
        def get_mermaid_diagram():
            """API endpoint to get Mermaid diagram."""
            if not self.architecture_graph:
                self.architecture_graph = self.analyzer.analyze_architecture()
            
            focus_core = request.args.get('focus_core', 'true').lower() == 'true'
            mermaid_diagram = self.analyzer.generate_mermaid_diagram(
                self.architecture_graph, 
                focus_core=focus_core
            )
            
            return jsonify({'diagram': mermaid_diagram})
        
        @self.app.route('/api/module/<path:module_name>')
        def get_module_details(module_name):
            """API endpoint to get details for a specific module."""
            if not self.architecture_graph:
                self.architecture_graph = self.analyzer.analyze_architecture()
            
            if module_name in self.architecture_graph.nodes:
                node = self.architecture_graph.nodes[module_name]
                return jsonify(asdict(node))
            else:
                return jsonify({'error': 'Module not found'}), 404
    
    def _get_main_template(self) -> str:
        """Get the main HTML template for the Architecture Explorer."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🗺️ SAM Architecture Explorer</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            margin: 0;
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .controls {
            margin-top: 15px;
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        
        .btn:hover {
            background: #2980b9;
        }
        
        .btn.active {
            background: #e74c3c;
        }
        
        .main-content {
            display: flex;
            height: calc(100vh - 140px);
        }
        
        .diagram-container {
            flex: 1;
            background: white;
            margin: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        .diagram-header {
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
            font-weight: bold;
            color: #495057;
        }
        
        .diagram-content {
            flex: 1;
            overflow: auto;
            padding: 20px;
        }
        
        .sidebar {
            width: 350px;
            background: white;
            margin: 20px 20px 20px 0;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        .sidebar-header {
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
            font-weight: bold;
            color: #495057;
        }
        
        .sidebar-content {
            flex: 1;
            overflow: auto;
            padding: 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        
        .stat-label {
            font-size: 12px;
            color: #6c757d;
            margin-top: 5px;
        }
        
        .module-details {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        
        .module-details h4 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        
        .module-details p {
            margin: 5px 0;
            font-size: 14px;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            color: #6c757d;
        }
        
        .error {
            color: #e74c3c;
            background: #fdf2f2;
            padding: 15px;
            border-radius: 8px;
            margin: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🗺️ SAM Architecture Explorer</h1>
        <div class="controls">
            <button class="btn active" id="coreViewBtn" onclick="toggleView(true)">
                🎯 Core Modules
            </button>
            <button class="btn" id="fullViewBtn" onclick="toggleView(false)">
                📊 Full Architecture
            </button>
            <button class="btn" onclick="refreshData()">
                🔄 Refresh
            </button>
        </div>
    </div>
    
    <div class="main-content">
        <div class="diagram-container">
            <div class="diagram-header">
                📈 Architecture Diagram
            </div>
            <div class="diagram-content">
                <div id="mermaid-diagram" class="loading">
                    Loading architecture diagram...
                </div>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="sidebar-header">
                📊 Architecture Statistics
            </div>
            <div class="sidebar-content">
                <div class="stats-grid" id="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number" id="total-modules">-</div>
                        <div class="stat-label">Total Modules</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="core-modules">-</div>
                        <div class="stat-label">Core Modules</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="dependencies">-</div>
                        <div class="stat-label">Dependencies</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="complexity">-</div>
                        <div class="stat-label">Avg Complexity</div>
                    </div>
                </div>
                
                <div class="module-details" id="module-details" style="display: none;">
                    <h4>Module Details</h4>
                    <div id="module-info"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFocusCore = true;
        let architectureData = null;
        
        // Initialize Mermaid
        mermaid.initialize({ 
            startOnLoad: false,
            theme: 'default',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true
            }
        });
        
        async function loadArchitectureData() {
            try {
                const response = await fetch('/api/architecture');
                architectureData = await response.json();
                updateStats();
            } catch (error) {
                console.error('Failed to load architecture data:', error);
                showError('Failed to load architecture data');
            }
        }
        
        async function loadMermaidDiagram(focusCore = true) {
            try {
                const response = await fetch(`/api/mermaid?focus_core=${focusCore}`);
                const data = await response.json();
                
                const element = document.getElementById('mermaid-diagram');
                element.innerHTML = data.diagram;
                
                await mermaid.run({
                    nodes: [element]
                });
                
            } catch (error) {
                console.error('Failed to load diagram:', error);
                showError('Failed to load diagram');
            }
        }
        
        function updateStats() {
            if (!architectureData) return;
            
            document.getElementById('total-modules').textContent = architectureData.metadata.total_modules;
            document.getElementById('core-modules').textContent = architectureData.metadata.core_modules;
            document.getElementById('dependencies').textContent = architectureData.metadata.total_dependencies;
            
            // Calculate average complexity
            const complexities = Object.values(architectureData.nodes).map(node => node.complexity_score);
            const avgComplexity = complexities.reduce((a, b) => a + b, 0) / complexities.length;
            document.getElementById('complexity').textContent = avgComplexity.toFixed(1);
        }
        
        function toggleView(focusCore) {
            currentFocusCore = focusCore;
            
            // Update button states
            document.getElementById('coreViewBtn').className = focusCore ? 'btn active' : 'btn';
            document.getElementById('fullViewBtn').className = focusCore ? 'btn' : 'btn active';
            
            // Reload diagram
            loadMermaidDiagram(focusCore);
        }
        
        function refreshData() {
            document.getElementById('mermaid-diagram').innerHTML = '<div class="loading">Refreshing...</div>';
            loadArchitectureData().then(() => loadMermaidDiagram(currentFocusCore));
        }
        
        function showError(message) {
            document.getElementById('mermaid-diagram').innerHTML = 
                `<div class="error">❌ ${message}</div>`;
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadArchitectureData().then(() => loadMermaidDiagram(true));
        });
    </script>
</body>
</html>
        '''
    
    def run(self, debug: bool = True):
        """
        Run the Architecture Explorer UI server.
        
        Args:
            debug: Whether to run in debug mode
        """
        self.logger.info(f"🚀 Starting Architecture Explorer UI on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)
    
    def generate_static_export(self, output_path: str):
        """
        Generate a static HTML export of the architecture.
        
        Args:
            output_path: Path to save the static HTML file
        """
        self.logger.info(f"📄 Generating static export to {output_path}")
        
        # Analyze architecture
        if not self.architecture_graph:
            self.architecture_graph = self.analyzer.analyze_architecture()
        
        # Generate static HTML with embedded data
        html_content = self._get_static_template()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"✅ Static export saved to {output_path}")
    
    def _get_static_template(self) -> str:
        """Generate static HTML template with embedded data."""
        # This would contain the same HTML but with embedded JSON data
        # For brevity, returning a placeholder
        return "<!-- Static HTML template would go here -->"
