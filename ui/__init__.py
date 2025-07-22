# ui/__init__.py

"""
User interface module for SAM (Small Agent Model).

This module provides chat interface capabilities and interactive memory control & visualization.
Sprint 12: Interactive Memory Control & Visualization
"""

from .chat_ui import ChatInterface, launch_chat_interface
from .memory_browser import MemoryBrowserUI
from .memory_editor import MemoryEditor
from .memory_graph import MemoryGraphVisualizer
from .memory_commands import MemoryCommandProcessor, CommandResult, get_command_processor
from .role_memory_filter import RoleBasedMemoryFilter, MemoryAccessLevel, MemoryAccessRule, RoleMemoryContext, get_role_filter
from .memory_app import main as run_memory_app

__all__ = [
    # Chat Interface
    'ChatInterface',
    'launch_chat_interface',

    # Memory Browser
    'MemoryBrowserUI',

    # Memory Editor
    'MemoryEditor',

    # Memory Graph
    'MemoryGraphVisualizer',

    # Memory Commands
    'MemoryCommandProcessor',
    'CommandResult',
    'get_command_processor',

    # Role-Based Filtering
    'RoleBasedMemoryFilter',
    'MemoryAccessLevel',
    'MemoryAccessRule',
    'RoleMemoryContext',
    'get_role_filter',

    # Main Memory App
    'run_memory_app'
]
