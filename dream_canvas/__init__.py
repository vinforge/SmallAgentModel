#!/usr/bin/env python3
"""
SAM Dream Canvas Module
======================

Refactored Dream Canvas cognitive visualization components.

This module provides a modular architecture replacing the monolithic
dream_canvas.py with organized, maintainable components.

Components:
- dream_canvas_controller: Main Dream Canvas application orchestration
- handlers: Cognitive mapping and analysis engines
- visualization: Interactive visualization and rendering
- research: Deep research and insight generation
- utils: Data models and utilities

Author: SAM Development Team
Version: 1.0.0 - Refactored Architecture
"""

from .dream_canvas_controller import DreamCanvasController, render_dream_canvas, main
from .handlers.cognitive_mapping import (
    CognitiveMappingEngine,
    get_cognitive_mapping_engine,
    generate_cognitive_map
)
from .visualization.canvas_renderer import (
    CanvasRenderer,
    get_canvas_renderer,
    render_cognitive_map,
    render_cluster_details,
    render_map_statistics,
    render_configuration_panel
)
from .research.deep_research import (
    DeepResearchEngine,
    get_deep_research_engine,
    generate_cluster_insights,
    render_cluster_research_controls
)
from .utils.models import (
    MemoryCluster,
    CognitiveMap,
    ClusterConnection,
    ResearchInsight,
    VisualizationConfig,
    DreamCanvasState,
    VisualizationMethod,
    ClusteringMethod,
    TimeRange
)

__all__ = [
    # Main application
    'DreamCanvasController',
    'render_dream_canvas',
    'main',
    
    # Cognitive mapping
    'CognitiveMappingEngine',
    'get_cognitive_mapping_engine',
    'generate_cognitive_map',
    
    # Visualization
    'CanvasRenderer',
    'get_canvas_renderer',
    'render_cognitive_map',
    'render_cluster_details',
    'render_map_statistics',
    'render_configuration_panel',
    
    # Research
    'DeepResearchEngine',
    'get_deep_research_engine',
    'generate_cluster_insights',
    'render_cluster_research_controls',
    
    # Data models
    'MemoryCluster',
    'CognitiveMap',
    'ClusterConnection',
    'ResearchInsight',
    'VisualizationConfig',
    'DreamCanvasState',
    'VisualizationMethod',
    'ClusteringMethod',
    'TimeRange'
]

__version__ = '1.0.0'
