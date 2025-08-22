#!/usr/bin/env python3
"""
SAM UI Module
=============

Refactored UI components for the SAM (Small Agent Model) application.

This module provides a modular architecture replacing the monolithic
secure_streamlit_app.py with organized, maintainable components.

Components:
- app_controller: Main application orchestration
- components: Reusable UI components (chat, forms, etc.)
- handlers: Specialized handlers (document, auth, etc.)
- security: Authentication and session management
- utils: Common utilities and helpers

Author: SAM Development Team
Version: 1.0.0 - Refactored Architecture
"""

from .app_controller import SAMAppController, main
from .components.chat_interface import ChatInterface, get_chat_interface, render_chat_interface
from .handlers.document_handler import (
    render_document_upload_section,
    render_uploaded_documents_list,
    generate_enhanced_summary_prompt,
    generate_enhanced_questions_prompt,
    generate_enhanced_analysis_prompt
)
from .security.session_manager import (
    SessionManager,
    get_session_manager,
    initialize_secure_session,
    check_authentication
)
from .utils.helpers import (
    extract_result_content,
    format_file_size,
    format_duration,
    format_timestamp,
    health_check,
    render_health_status
)

__all__ = [
    # Main application
    'SAMAppController',
    'main',
    
    # Chat interface
    'ChatInterface',
    'get_chat_interface',
    'render_chat_interface',
    
    # Document handling
    'render_document_upload_section',
    'render_uploaded_documents_list',
    'generate_enhanced_summary_prompt',
    'generate_enhanced_questions_prompt',
    'generate_enhanced_analysis_prompt',
    
    # Security and sessions
    'SessionManager',
    'get_session_manager',
    'initialize_secure_session',
    'check_authentication',
    
    # Utilities
    'extract_result_content',
    'format_file_size',
    'format_duration',
    'format_timestamp',
    'health_check',
    'render_health_status'
]

__version__ = '1.0.0'
