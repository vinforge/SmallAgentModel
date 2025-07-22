"""
SAM Agent Zero A* Search Planning Module

This module implements LLM-guided A* search for strategic planning in SAM.
It provides optimal action sequence planning for complex multi-step tasks.

Components:
- PlanningState: Represents a state in the search tree
- SearchNode: Wrapper for planning states with search metadata
- Frontier: Priority queue for managing search nodes
- HeuristicEstimator: LLM-based cost estimation
- ActionExpander: LLM-based action generation
- AStarPlanner: Main A* search implementation
"""

from .state import PlanningState
from .search_node import SearchNode, SearchNodeFactory
from .frontier import Frontier
from .sam_tool_registry import SAMTool, ToolCategory, SAMToolRegistry, get_sam_tool_registry
from .state_similarity import SimilarityMetrics, StateSimilarityDetector
from .sam_context_manager import (
    DocumentContext, MemoryContext, ConversationContext, SAMContextManager
)
from .heuristic_estimator import HeuristicEstimate, HeuristicEstimator
from .action_expander import ActionCandidate, ActionExpander
from .a_star_planner import PlanningResult, AStarPlanner, SAMPlannerIntegration
from .tpv_planning_controller import TPVPlanningController, PlanningProgressMetrics, PlanningControlResult
from .episodic_memory_heuristic import EpisodicMemoryHeuristic, PlanningExperience, ExperienceAdjustment
from .meta_reasoning_validator import MetaReasoningPlanValidator, PlanValidationResult, ValidationIssue, ValidationSeverity

__all__ = [
    'PlanningState',
    'SearchNode',
    'SearchNodeFactory',
    'Frontier',
    'SAMTool',
    'ToolCategory',
    'SAMToolRegistry',
    'get_sam_tool_registry',
    'SimilarityMetrics',
    'StateSimilarityDetector',
    'DocumentContext',
    'MemoryContext',
    'ConversationContext',
    'SAMContextManager',
    'HeuristicEstimate',
    'HeuristicEstimator',
    'ActionCandidate',
    'ActionExpander',
    'PlanningResult',
    'AStarPlanner',
    'SAMPlannerIntegration',
    'TPVPlanningController',
    'PlanningProgressMetrics',
    'PlanningControlResult',
    'EpisodicMemoryHeuristic',
    'PlanningExperience',
    'ExperienceAdjustment',
    'MetaReasoningPlanValidator',
    'PlanValidationResult',
    'ValidationIssue',
    'ValidationSeverity'
]

__version__ = '1.0.0'
