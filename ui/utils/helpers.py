#!/usr/bin/env python3
"""
UI Helper Utilities for SAM
===========================

Common utility functions and helpers for the SAM UI
extracted from the monolithic secure_streamlit_app.py.

This module provides:
- Common UI utilities
- Data formatting functions
- Validation helpers
- Configuration utilities

Author: SAM Development Team
Version: 1.0.0 - Refactored from secure_streamlit_app.py
"""

import streamlit as st
import logging
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_result_content(result: Any) -> Tuple[str, Dict[str, Any]]:
    """Extract content and metadata from various result types."""
    try:
        if hasattr(result, 'content'):
            content = str(result.content)
            metadata = getattr(result, 'metadata', {})
        elif hasattr(result, 'text'):
            content = str(result.text)
            metadata = getattr(result, 'metadata', {})
        elif isinstance(result, dict):
            content = result.get('content', result.get('text', str(result)))
            metadata = result.get('metadata', {})
        elif isinstance(result, str):
            content = result
            metadata = {}
        else:
            content = str(result)
            metadata = {}
        
        return content, metadata
        
    except Exception as e:
        logger.error(f"Error extracting result content: {e}")
        return str(result), {}


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_timestamp(timestamp: Union[str, datetime, float]) -> str:
    """Format timestamp in human-readable format."""
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, datetime):
            dt = timestamp
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        else:
            return str(timestamp)
        
        return dt.strftime("%Y-%m-%d %H:%M:%S")
        
    except Exception as e:
        logger.error(f"Error formatting timestamp: {e}")
        return str(timestamp)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        max_name_length = 255 - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized


def validate_json(json_string: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """Validate JSON string and return parsed data."""
    try:
        data = json.loads(json_string)
        return True, data, None
    except json.JSONDecodeError as e:
        return False, None, str(e)


def safe_get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Safely get nested dictionary value."""
    try:
        for key in keys:
            data = data[key]
        return data
    except (KeyError, TypeError):
        return default


def render_metric_card(title: str, value: str, delta: Optional[str] = None, 
                      help_text: Optional[str] = None):
    """Render a metric card with consistent styling."""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text
    )


def render_status_badge(status: str, message: str = "") -> str:
    """Render a status badge with appropriate styling."""
    status_configs = {
        'success': {'emoji': '✅', 'color': 'green'},
        'error': {'emoji': '❌', 'color': 'red'},
        'warning': {'emoji': '⚠️', 'color': 'orange'},
        'info': {'emoji': 'ℹ️', 'color': 'blue'},
        'processing': {'emoji': '⏳', 'color': 'yellow'},
        'pending': {'emoji': '⏸️', 'color': 'gray'}
    }
    
    config = status_configs.get(status.lower(), {'emoji': '❓', 'color': 'gray'})
    
    if message:
        return f"{config['emoji']} {message}"
    else:
        return f"{config['emoji']} {status.title()}"


def create_download_link(data: Union[str, bytes], filename: str, 
                        mime_type: str = "text/plain") -> None:
    """Create a download button for data."""
    st.download_button(
        label=f"📥 Download {filename}",
        data=data,
        file_name=filename,
        mime=mime_type
    )


def render_progress_bar(current: int, total: int, label: str = "Progress") -> None:
    """Render a progress bar with percentage."""
    if total > 0:
        progress = current / total
        st.progress(progress, text=f"{label}: {current}/{total} ({progress*100:.1f}%)")
    else:
        st.progress(0, text=f"{label}: 0/0 (0%)")


def render_collapsible_section(title: str, content: str, expanded: bool = False) -> None:
    """Render a collapsible section with content."""
    with st.expander(title, expanded=expanded):
        st.markdown(content)


def validate_url(url: str) -> bool:
    """Validate URL format."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None


def validate_email(email: str) -> bool:
    """Validate email format."""
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return email_pattern.match(email) is not None


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def render_info_box(message: str, box_type: str = "info") -> None:
    """Render an information box with appropriate styling."""
    box_functions = {
        'info': st.info,
        'success': st.success,
        'warning': st.warning,
        'error': st.error
    }
    
    box_function = box_functions.get(box_type, st.info)
    box_function(message)


def get_theme_colors() -> Dict[str, str]:
    """Get theme colors for consistent styling."""
    return {
        'primary': '#FF6B6B',
        'secondary': '#4ECDC4',
        'success': '#45B7D1',
        'warning': '#FFA07A',
        'error': '#FF6B6B',
        'info': '#96CEB4',
        'light': '#FFEAA7',
        'dark': '#2D3436'
    }


def render_loading_spinner(message: str = "Loading...") -> None:
    """Render a loading spinner with message."""
    with st.spinner(message):
        time.sleep(0.1)  # Brief pause for visual effect


def create_tabs(tab_names: List[str]) -> List:
    """Create tabs with consistent styling."""
    return st.tabs([f"📋 {name}" for name in tab_names])


def render_sidebar_section(title: str, content_func, expanded: bool = True) -> None:
    """Render a sidebar section with consistent styling."""
    with st.sidebar:
        if expanded:
            st.markdown(f"### {title}")
            content_func()
        else:
            with st.expander(title, expanded=False):
                content_func()


def format_code_block(code: str, language: str = "python") -> str:
    """Format code block with syntax highlighting."""
    return f"```{language}\n{code}\n```"


def render_key_value_pairs(data: Dict[str, Any], title: str = "Details") -> None:
    """Render key-value pairs in a formatted table."""
    st.markdown(f"**{title}:**")
    
    for key, value in data.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{key}:**")
        with col2:
            st.write(str(value))


def health_check() -> Dict[str, Any]:
    """Perform basic health check of the application."""
    try:
        # Check session state
        session_valid = 'session_id' in st.session_state
        
        # Check file system access
        temp_file = Path("temp_health_check.txt")
        try:
            temp_file.write_text("test")
            temp_file.unlink()
            filesystem_ok = True
        except Exception:
            filesystem_ok = False
        
        # Check current time
        current_time = datetime.now()
        
        return {
            'status': 'healthy' if session_valid and filesystem_ok else 'degraded',
            'session_valid': session_valid,
            'filesystem_access': filesystem_ok,
            'current_time': current_time.isoformat(),
            'uptime': time.time() - st.session_state.get('app_start_time', time.time())
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'current_time': datetime.now().isoformat()
        }


def render_health_status():
    """Render application health status."""
    health = health_check()
    
    if health['status'] == 'healthy':
        st.success("✅ Application is healthy")
    elif health['status'] == 'degraded':
        st.warning("⚠️ Application is running with degraded performance")
    else:
        st.error("❌ Application health check failed")
    
    with st.expander("Health Details", expanded=False):
        render_key_value_pairs(health, "Health Check Results")
