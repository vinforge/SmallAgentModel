#!/usr/bin/env python3
"""
SAM Core Framework
==================

Core framework components that standardize patterns across all SAM UI modules.
This provides the foundation for consistent, maintainable, and scalable SAM applications.

This module provides:
- Base controller classes with standard patterns
- Unified utility functions
- Component interface standards
- Common error handling and logging
- Standardized UI patterns

Author: SAM Development Team
Version: 1.0.0 - Standardization Framework
"""

from .base_controller import BaseController, ComponentManager, get_component_manager
from .unified_utils import (
    # Formatting utilities
    format_file_size,
    format_duration,
    format_timestamp,
    format_number,

    # Validation utilities
    validate_json,
    validate_url,
    validate_email,
    sanitize_filename,

    # UI helper functions
    render_status_badge,
    render_metric_card,
    render_progress_bar,
    render_info_box,
    create_download_link,
    create_tabs,
    render_collapsible_section,
    render_key_value_pairs,

    # Data utilities
    safe_get_nested,
    truncate_text,
    extract_result_content,
    generate_unique_id,

    # Health check utilities
    health_check,
    render_health_status,

    # Theme and styling utilities
    get_theme_colors,
    format_code_block,

    # Session utilities
    get_session_info,
    clear_session_prefix,

    # Error handling utilities
    handle_error,
    safe_execute
)
from .performance import (
    # Caching
    cached,
    get_cache,
    render_cache_dashboard,

    # Lazy Loading
    lazy_load,
    render_lazy_loading_dashboard,

    # Memory Management
    track_resource,
    render_memory_dashboard,

    # Performance Monitoring
    performance_timer,
    render_performance_dashboard,

    # Comprehensive Performance
    optimize_sam_application,
    render_comprehensive_performance_dashboard
)
from .component_interface import (
    BaseComponent,
    RenderableComponent,
    DataComponent,
    InteractiveComponent,
    ComponentRegistry,
    get_component_registry,
    register_component,
    get_component,
    render_component_group,
    render_component_status_dashboard,
    component_error_handler,
    cached_component_method
)

__all__ = [
    # Base controller framework
    'BaseController',
    'ComponentManager',
    'get_component_manager',

    # Formatting utilities
    'format_file_size',
    'format_duration',
    'format_timestamp',
    'format_number',

    # Validation utilities
    'validate_json',
    'validate_url',
    'validate_email',
    'sanitize_filename',

    # UI helper functions
    'render_status_badge',
    'render_metric_card',
    'render_progress_bar',
    'render_info_box',
    'create_download_link',
    'create_tabs',
    'render_collapsible_section',
    'render_key_value_pairs',

    # Data utilities
    'safe_get_nested',
    'truncate_text',
    'extract_result_content',
    'generate_unique_id',

    # Health check utilities
    'health_check',
    'render_health_status',

    # Theme and styling utilities
    'get_theme_colors',
    'format_code_block',

    # Session utilities
    'get_session_info',
    'clear_session_prefix',

    # Error handling utilities
    'handle_error',
    'safe_execute',

    # Component interface
    'BaseComponent',
    'RenderableComponent',
    'DataComponent',
    'InteractiveComponent',
    'ComponentRegistry',
    'get_component_registry',
    'register_component',
    'get_component',
    'render_component_group',
    'render_component_status_dashboard',
    'component_error_handler',
    'cached_component_method',

    # Performance optimization
    'cached',
    'get_cache',
    'render_cache_dashboard',
    'lazy_load',
    'render_lazy_loading_dashboard',
    'track_resource',
    'render_memory_dashboard',
    'performance_timer',
    'render_performance_dashboard',
    'optimize_sam_application',
    'render_comprehensive_performance_dashboard'
]

__version__ = '1.0.0'

# Framework metadata
FRAMEWORK_INFO = {
    'name': 'SAM Core Framework',
    'version': __version__,
    'description': 'Standardized framework for SAM UI applications',
    'components': [
        'BaseController - Standard application controller pattern',
        'UnifiedUtils - Common utility functions',
        'ComponentInterface - Standard component patterns',
        'ErrorHandling - Unified error handling',
        'UIPatterns - Consistent UI patterns'
    ],
    'benefits': [
        'Eliminates code duplication',
        'Ensures consistent patterns',
        'Improves maintainability',
        'Standardizes error handling',
        'Accelerates development'
    ]
}


def get_framework_info() -> dict:
    """Get information about the SAM Core Framework."""
    return FRAMEWORK_INFO


def print_framework_info():
    """Print framework information to console."""
    info = get_framework_info()
    
    print(f"\n{info['name']} v{info['version']}")
    print("=" * 50)
    print(f"Description: {info['description']}")
    
    print("\nComponents:")
    for component in info['components']:
        print(f"  • {component}")
    
    print("\nBenefits:")
    for benefit in info['benefits']:
        print(f"  ✓ {benefit}")
    
    print("\n" + "=" * 50)


# Auto-import commonly used functions for convenience
from .unified_utils import format_file_size, format_duration, handle_error
from .component_interface import BaseComponent, register_component
