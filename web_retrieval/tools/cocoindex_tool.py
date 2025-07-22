#!/usr/bin/env python3
"""
CocoIndex Tool for Phase 8.5: Intelligent Web Retrieval with cocoindex

This module provides the CocoIndexTool class that acts as a clean interface
to the cocoindex library for intelligent web content retrieval and indexing.

Features:
- Asynchronous intelligent search using cocoindex
- Configurable search parameters (num_pages, search_provider)
- Robust error handling with graceful fallbacks
- Clean integration with SAM's existing vetting pipeline
- Support for both Community and Pro editions

Author: SAM Development Team
Version: 1.0.0
"""

import asyncio
import logging
import subprocess
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class CocoIndexError(Exception):
    """Base exception for CocoIndex-related errors"""
    pass

class CocoIndexTool:
    """
    Tool for performing intelligent web searches using cocoindex.
    
    CocoIndex provides a sophisticated search and content extraction pipeline
    that combines search APIs with intelligent scraping and local indexing.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 search_provider: str = "serper",
                 num_pages: int = 5,
                 timeout: int = 60):
        """
        Initialize the CocoIndex tool.
        
        Args:
            api_key: API key for the search provider (Serper, etc.)
            search_provider: Search provider to use ("serper", "duckduckgo")
            num_pages: Number of pages to fetch and index
            timeout: Timeout for search operations in seconds
        """
        self.api_key = api_key
        self.search_provider = search_provider
        self.num_pages = num_pages
        self.timeout = timeout
        self.cocoindex_available = False
        
        # Check if cocoindex is available
        self._check_cocoindex_availability()
        
        logger.info(f"CocoIndexTool initialized - Available: {self.cocoindex_available}")
    
    def _check_cocoindex_availability(self) -> None:
        """Check if cocoindex is available and install if needed."""
        try:
            import cocoindex
            self.cocoindex_available = True
            logger.info("cocoindex library is available")
        except ImportError:
            logger.warning("cocoindex not available - attempting installation...")
            try:
                # Try to install cocoindex
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 'cocoindex'
                ])
                import cocoindex
                self.cocoindex_available = True
                logger.info("cocoindex successfully installed and imported")
            except Exception as e:
                logger.error(f"Failed to install cocoindex: {e}")
                self.cocoindex_available = False
    
    async def intelligent_search(self, query: str) -> Dict[str, Any]:
        """
        Perform intelligent web search using cocoindex.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            logger.info(f"Starting intelligent search with cocoindex: '{query}'")
            
            # Check if cocoindex is available
            if not self.cocoindex_available:
                return self._fallback_error_response(
                    "cocoindex not available - please install or configure manually"
                )
            
            # Check API key for paid providers - auto-switch to DuckDuckGo if no key
            if self.search_provider == "serper" and not self.api_key:
                logger.warning("No Serper API key provided - switching to DuckDuckGo for free search")
                self.search_provider = "duckduckgo"
            
            # Run cocoindex search in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._execute_cocoindex_search, 
                query
            )
            
            return result
            
        except Exception as e:
            logger.error(f"CocoIndex search failed: {e}")
            return self._fallback_error_response(str(e))
    
    def _execute_cocoindex_search(self, query: str) -> Dict[str, Any]:
        """Execute the actual cocoindex search (runs in thread executor)."""
        try:
            import cocoindex
            
            # Configure cocoindex parameters
            config = {
                'search_provider': self.search_provider,
                'num_pages': self.num_pages,
                'timeout': self.timeout
            }
            
            # Add API key if available
            if self.api_key:
                config['api_key'] = self.api_key
            
            logger.info(f"Executing cocoindex search with config: {config}")
            
            # TEMPORARY FIX: CocoIndex API has changed, disable for now
            # TODO: Implement proper cocoindex flow-based search
            logger.warning("CocoIndex search temporarily disabled due to API compatibility issues")
            raise Exception("CocoIndex tool temporarily disabled - API interface changed")
            
            # Process and format results
            if search_result and hasattr(search_result, 'chunks'):
                chunks = search_result.chunks
                
                # Format chunks for SAM's vetting pipeline
                formatted_chunks = []
                for i, chunk in enumerate(chunks):
                    formatted_chunk = {
                        'content': chunk.text if hasattr(chunk, 'text') else str(chunk),
                        'source_url': chunk.url if hasattr(chunk, 'url') else '',
                        'title': chunk.title if hasattr(chunk, 'title') else f'Chunk {i+1}',
                        'relevance_score': chunk.score if hasattr(chunk, 'score') else 0.0,
                        'chunk_index': i,
                        'extraction_method': 'cocoindex',
                        'timestamp': datetime.now().isoformat()
                    }
                    formatted_chunks.append(formatted_chunk)
                
                return {
                    'success': True,
                    'tool_used': 'cocoindex_tool',
                    'query': query,
                    'chunks': formatted_chunks,
                    'total_chunks': len(formatted_chunks),
                    'search_provider': self.search_provider,
                    'num_pages_fetched': self.num_pages,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'cocoindex_version': getattr(cocoindex, '__version__', 'unknown'),
                        'processing_time_seconds': getattr(search_result, 'processing_time', 0),
                        'sources_processed': getattr(search_result, 'sources_count', 0)
                    }
                }
            else:
                return self._fallback_error_response(
                    "cocoindex returned no results or invalid format"
                )
                
        except ImportError:
            return self._fallback_error_response(
                "cocoindex library not available"
            )
        except Exception as e:
            logger.error(f"cocoindex execution failed: {e}")
            return self._fallback_error_response(f"cocoindex error: {str(e)}")
    
    def _fallback_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate a standardized error response."""
        return {
            'success': False,
            'tool_used': 'cocoindex_tool',
            'error': error_message,
            'chunks': [],
            'total_chunks': 0,
            'timestamp': datetime.now().isoformat(),
            'fallback_available': True,
            'fallback_suggestion': 'Consider using legacy search tools or configuring API keys'
        }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about the CocoIndex tool."""
        return {
            'name': 'CocoIndexTool',
            'description': 'Intelligent web search and content extraction using cocoindex',
            'version': '1.0.0',
            'available': self.cocoindex_available,
            'search_provider': self.search_provider,
            'num_pages': self.num_pages,
            'api_key_configured': bool(self.api_key),
            'capabilities': [
                'Intelligent web search',
                'Automatic content extraction',
                'Local indexing and ranking',
                'Multi-source aggregation',
                'Relevance scoring'
            ],
            'requirements': [
                'cocoindex library',
                'API key for paid providers (optional)',
                'Internet connection'
            ]
        }
    
    def update_config(self, 
                     api_key: Optional[str] = None,
                     search_provider: Optional[str] = None,
                     num_pages: Optional[int] = None) -> None:
        """Update tool configuration."""
        if api_key is not None:
            self.api_key = api_key
        if search_provider is not None:
            self.search_provider = search_provider
        if num_pages is not None:
            self.num_pages = num_pages
            
        logger.info(f"CocoIndexTool configuration updated: provider={self.search_provider}, pages={self.num_pages}")
