"""
SAM Introspection Engine
=======================

Comprehensive introspection capabilities for understanding SAM's cognitive processes.
Provides structured logging, performance monitoring, and cognitive analysis tools.

Author: SAM Development Team
Version: 1.0.0
"""

from .introspection_logger import IntrospectionLogger
from .static_architecture_analyzer import StaticArchitectureAnalyzer, ModuleNode, ArchitectureGraph
from .architecture_explorer_ui import ArchitectureExplorerUI
from .flight_recorder import (
    FlightRecorder, ReasoningStep, TraceLevel, TraceEvent, ReasoningTrace,
    CognitiveVector, get_flight_recorder, initialize_flight_recorder,
    trace_step, trace_context, TraceSession
)
from .trace_visualization_ui import TraceVisualizationUI
from .algonauts_visualization import AlgonautsVisualizer, CognitiveTrajectory, ProjectedPoint

__all__ = [
    'IntrospectionLogger',
    'StaticArchitectureAnalyzer',
    'ModuleNode',
    'ArchitectureGraph',
    'ArchitectureExplorerUI',
    'FlightRecorder',
    'ReasoningStep',
    'TraceLevel',
    'TraceEvent',
    'ReasoningTrace',
    'CognitiveVector',
    'get_flight_recorder',
    'initialize_flight_recorder',
    'trace_step',
    'trace_context',
    'TraceSession',
    'TraceVisualizationUI',
    'AlgonautsVisualizer',
    'CognitiveTrajectory',
    'ProjectedPoint'
]
