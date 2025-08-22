#!/usr/bin/env python3
"""
SAM UI Utilities
================

Common utilities and helper functions for the SAM UI.

Author: SAM Development Team
Version: 1.0.0
"""

from .helpers import (
    extract_result_content,
    format_file_size,
    format_duration,
    format_timestamp,
    sanitize_filename,
    validate_json,
    safe_get_nested,
    render_metric_card,
    render_status_badge,
    create_download_link,
    render_progress_bar,
    render_collapsible_section,
    validate_url,
    validate_email,
    truncate_text,
    render_info_box,
    get_theme_colors,
    render_loading_spinner,
    create_tabs,
    render_sidebar_section,
    format_code_block,
    render_key_value_pairs,
    health_check,
    render_health_status
)

__all__ = [
    'extract_result_content',
    'format_file_size',
    'format_duration',
    'format_timestamp',
    'sanitize_filename',
    'validate_json',
    'safe_get_nested',
    'render_metric_card',
    'render_status_badge',
    'create_download_link',
    'render_progress_bar',
    'render_collapsible_section',
    'validate_url',
    'validate_email',
    'truncate_text',
    'render_info_box',
    'get_theme_colors',
    'render_loading_spinner',
    'create_tabs',
    'render_sidebar_section',
    'format_code_block',
    'render_key_value_pairs',
    'health_check',
    'render_health_status'
]
