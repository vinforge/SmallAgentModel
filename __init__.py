"""
SAM (Small Agent Model) Package
==============================

A comprehensive AI agent framework with advanced memory, reasoning,
and introspection capabilities.

Author: SAM Development Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "SAM Development Team"

# Core modules
from . import core
from . import memory
from . import agent_zero
from . import introspection
from . import code_interpreter

__all__ = [
    'core',
    'memory', 
    'agent_zero',
    'introspection',
    'code_interpreter'
]
