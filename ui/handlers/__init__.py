#!/usr/bin/env python3
"""
SAM UI Handlers
===============

Specialized handlers for different UI functionality.

Author: SAM Development Team
Version: 1.0.0
"""

from .document_handler import (
    render_document_upload_section,
    render_uploaded_documents_list,
    generate_enhanced_summary_prompt,
    generate_enhanced_questions_prompt,
    generate_enhanced_analysis_prompt,
    validate_uploaded_file,
    process_uploaded_document
)

__all__ = [
    'render_document_upload_section',
    'render_uploaded_documents_list',
    'generate_enhanced_summary_prompt',
    'generate_enhanced_questions_prompt', 
    'generate_enhanced_analysis_prompt',
    'validate_uploaded_file',
    'process_uploaded_document'
]
