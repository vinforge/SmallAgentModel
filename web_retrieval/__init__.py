"""
Web Retrieval Module for SAM - Phase 7.1: Agent-Zero Foundation

This module provides secure, isolated web content retrieval capabilities
using agent-zero technology with process isolation and manual airlock controls.

Components:
- WebFetcher: Main interface for web content retrieval
- ProcessManager: Handles isolated subprocess execution
- DataContracts: Structured data validation and schemas
- Exceptions: Custom exception handling

Security Features:
- Process isolation prevents browser crashes from affecting SAM
- Manual airlock system for controlled content ingestion
- Quarantine directory for untrusted web content
- Structured error handling and logging

Usage:
    from web_retrieval import WebFetcher
    
    fetcher = WebFetcher()
    result = fetcher.fetch_url_content("https://example.com")
    
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Content: {result.content[:100]}...")
"""

from .web_fetcher import WebFetcher, WebFetchResult
from .data_contracts import WebContentData, ValidationError
from .exceptions import (
    WebRetrievalError,
    ProcessIsolationError,
    ContentExtractionError,
    TimeoutError
)

__version__ = "1.0.0"
__author__ = "SAM Development Team"

__all__ = [
    'WebFetcher',
    'WebFetchResult', 
    'WebContentData',
    'ValidationError',
    'WebRetrievalError',
    'ProcessIsolationError',
    'ContentExtractionError',
    'TimeoutError'
]
