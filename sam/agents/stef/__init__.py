#!/usr/bin/env python3
"""
STEF (Structured Task-Execution Framework) Package
Provides structured, multi-step task execution capabilities for SAM.
"""

from .task_definitions import (
    TaskStep,
    TaskProgram, 
    ExecutionContext,
    StepStatus,
    ProgramExecutionResult
)

from .executor import STEF_Executor

from .program_registry import (
    ProgramRegistry,
    get_program_registry,
    get_program,
    find_matching_program,
    PROGRAM_REGISTRY
)

__version__ = "1.0.0"
__author__ = "SAM Development Team"

__all__ = [
    # Core classes
    'TaskStep',
    'TaskProgram', 
    'ExecutionContext',
    'StepStatus',
    'ProgramExecutionResult',
    'STEF_Executor',
    'ProgramRegistry',
    
    # Convenience functions
    'get_program_registry',
    'get_program',
    'find_matching_program',
    
    # Legacy compatibility
    'PROGRAM_REGISTRY'
]
