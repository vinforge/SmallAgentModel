#!/usr/bin/env python3
"""
SAM Memory UI Module
===================

Refactored memory interface components for the SAM Memory Control Center.

This module provides a modular architecture replacing the monolithic
memory_app.py with organized, maintainable components.

Components:
- memory_app_controller: Main memory application orchestration
- components: Memory-specific UI components (browser, editor, graph)
- handlers: Specialized handlers (system status, analytics)
- security: Memory-specific authentication and security
- utils: Memory UI utilities and helpers

Author: SAM Development Team
Version: 1.0.0 - Refactored Architecture
"""

from .memory_app_controller import MemoryAppController, main
from .components.memory_components import (
    MemoryComponentManager,
    get_memory_component_manager,
    render_memory_browser,
    render_memory_editor,
    render_memory_graph,
    render_command_interface,
    render_role_access
)
from .handlers.system_status import (
    SystemStatusHandler,
    get_system_status_handler,
    render_system_status,
    render_memory_ranking,
    render_citation_engine
)
from .security.memory_auth import (
    MemoryAuthManager,
    get_memory_auth_manager,
    check_memory_authentication,
    render_memory_authentication_required,
    render_memory_security_status,
    get_memory_security_status_indicator
)

__all__ = [
    # Main application
    'MemoryAppController',
    'main',
    
    # Memory components
    'MemoryComponentManager',
    'get_memory_component_manager',
    'render_memory_browser',
    'render_memory_editor',
    'render_memory_graph',
    'render_command_interface',
    'render_role_access',
    
    # System status and analytics
    'SystemStatusHandler',
    'get_system_status_handler',
    'render_system_status',
    'render_memory_ranking',
    'render_citation_engine',
    
    # Security and authentication
    'MemoryAuthManager',
    'get_memory_auth_manager',
    'check_memory_authentication',
    'render_memory_authentication_required',
    'render_memory_security_status',
    'get_memory_security_status_indicator'
]

__version__ = '1.0.0'
