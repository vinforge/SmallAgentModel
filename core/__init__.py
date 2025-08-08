# core/__init__.py

"""
Core module for SAM (Small Agent Model).

This module provides core functionality for query routing and semantic processing.
"""

from .query_router import SemanticQueryRouter, QueryResult

__all__ = [
    'SemanticQueryRouter',
    'QueryResult'
]
