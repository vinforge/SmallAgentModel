"""
SAM Autonomy Module
==================

This module implements SAM's Goal & Motivation Engine, providing the foundation
for autonomous goal generation, prioritization, and execution. This is the final
component of SAM's Level 3 "Self Modification Function" that grants SAM strong
autonomy by enabling proactive task generation and pursuit.

Core Components:
- Goal: Enhanced data structure for representing autonomous goals
- GoalStack: Persistent goal management with lifecycle controls
- MotivationEngine: Autonomous goal generation from UIF analysis
- GoalSafetyValidator: Safety checks and validation for autonomous actions

Phase C Components:
- IdleTimeProcessor: Background processing during idle periods
- SystemLoadMonitor: System resource monitoring
- AutonomousExecutionEngine: Complete autonomous execution with safety
- EmergencyOverrideSystem: Emergency controls and safety circuits

Author: SAM Development Team
Version: 2.0.0 (Enhanced with safety and performance features)
"""

from .goals import Goal
from .goal_stack import GoalStack
from .motivation_engine import MotivationEngine
from .safety.goal_validator import GoalSafetyValidator

# Phase C components
from .idle_processor import IdleTimeProcessor, IdleProcessingConfig, IdleState
from .system_monitor import SystemLoadMonitor, SystemThresholds, SystemState, SystemMetrics
from .execution_engine import AutonomousExecutionEngine, ExecutionConfig, ExecutionState
from .emergency_override import EmergencyOverrideSystem, SafetyThresholds, EmergencyLevel, OverrideReason

__all__ = [
    # Core components
    'Goal',
    'GoalStack',
    'MotivationEngine',
    'GoalSafetyValidator',

    # Phase C components
    'IdleTimeProcessor',
    'IdleProcessingConfig',
    'IdleState',
    'SystemLoadMonitor',
    'SystemThresholds',
    'SystemState',
    'SystemMetrics',
    'AutonomousExecutionEngine',
    'ExecutionConfig',
    'ExecutionState',
    'EmergencyOverrideSystem',
    'SafetyThresholds',
    'EmergencyLevel',
    'OverrideReason'
]
