#!/usr/bin/env python3
"""
Firecrawl Tool for SAM's Intelligent Web Retrieval System

This module provides the FirecrawlTool class that integrates Firecrawl's advanced
web crawling and extraction capabilities with SAM's existing vetting pipeline.

Features:
- Advanced web crawling with anti-bot mechanisms
- Interactive content extraction with actions
- Batch processing for multiple URLs
- Full integration with SAM's content vetting system
- Graceful fallback to existing tools

Author: SAM Development Team
Version: 1.0.0
"""

import asyncio
import logging
import subprocess
import sys
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)


class FirecrawlTool:
    """
    Firecrawl integration tool for advanced web crawling and extraction.
    
    This tool provides SAM with advanced web crawling capabilities while
    maintaining full integration with SAM's content vetting pipeline.
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize the Firecrawl tool.
        
        Args:
            api_key: Firecrawl API key (optional for self-hosted)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self.firecrawl_available = False
        self.firecrawl_app = None
        
        # Initialize Firecrawl
        self._initialize_firecrawl()
        
        logger.info(f"FirecrawlTool initialized - Available: {self.firecrawl_available}")
    
    def _initialize_firecrawl(self):
        """Initialize Firecrawl with graceful fallback."""
        try:
            # Try to import and initialize Firecrawl
            import firecrawl
            from firecrawl import FirecrawlApp
            
            if self.api_key:
                self.firecrawl_app = FirecrawlApp(api_key=self.api_key)
                logger.info("Firecrawl initialized with API key")
            else:
                # Try to initialize without API key (for self-hosted)
                self.firecrawl_app = FirecrawlApp()
                logger.info("Firecrawl initialized without API key (self-hosted mode)")
            
            self.firecrawl_available = True
            
        except ImportError:
            logger.warning("Firecrawl not installed. Install with: pip install firecrawl-py")
            self.firecrawl_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize Firecrawl: {e}")
            self.firecrawl_available = False
    
    async def intelligent_crawl(self, url: str, max_pages: int = 10, 
                               include_subdomains: bool = False) -> Dict[str, Any]:
        """
        Crawl an entire website with Firecrawl's advanced capabilities.
        
        Args:
            url: Starting URL to crawl
            max_pages: Maximum number of pages to crawl
            include_subdomains: Whether to include subdomains
            
        Returns:
            Dictionary containing crawl results formatted for SAM's vetting pipeline
        """
        try:
            logger.info(f"Starting intelligent crawl with Firecrawl: '{url}' (max_pages: {max_pages})")
            
            if not self.firecrawl_available:
                return self._fallback_error_response(
                    "Firecrawl not available - please install firecrawl-py or configure API key"
                )
            
            # Run Firecrawl crawl in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._execute_firecrawl_crawl, 
                url, max_pages, include_subdomains
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Firecrawl crawl failed: {e}")
            return self._fallback_error_response(str(e))
    
    def _execute_firecrawl_crawl(self, url: str, max_pages: int, 
                                include_subdomains: bool) -> Dict[str, Any]:
        """Execute the actual Firecrawl crawl (runs in thread executor)."""
        try:
            # Configure crawl parameters
            crawl_params = {
                'limit': max_pages,
                'scrapeOptions': {
                    'formats': ['markdown', 'html'],
                    'includeTags': ['title', 'meta', 'h1', 'h2', 'h3', 'p', 'article'],
                    'excludeTags': ['script', 'style', 'nav', 'footer', 'aside']
                }
            }
            
            if not include_subdomains:
                crawl_params['allowedDomains'] = [self._extract_domain(url)]
            
            logger.info(f"Executing Firecrawl crawl with params: {crawl_params}")
            
            # Execute crawl
            crawl_result = self.firecrawl_app.crawl_url(url, **crawl_params)
            
            if crawl_result and crawl_result.get('success'):
                # Format results for SAM's vetting pipeline
                formatted_result = self._format_crawl_results(crawl_result, url)
                
                logger.info(f"Firecrawl crawl completed successfully: {formatted_result['total_chunks']} chunks extracted")
                return formatted_result
            else:
                error_msg = crawl_result.get('error', 'Unknown crawl error') if crawl_result else 'No result returned'
                return self._fallback_error_response(f"Crawl failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Firecrawl crawl execution failed: {e}")
            return self._fallback_error_response(str(e))
    
    async def extract_with_actions(self, url: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract content after performing interactive actions.
        
        Args:
            url: URL to extract from
            actions: List of actions to perform (click, scroll, input, etc.)
            
        Returns:
            Dictionary containing extraction results formatted for SAM's vetting pipeline
        """
        try:
            logger.info(f"Starting interactive extraction with Firecrawl: '{url}' with {len(actions)} actions")
            
            if not self.firecrawl_available:
                return self._fallback_error_response(
                    "Firecrawl not available - please install firecrawl-py or configure API key"
                )
            
            # Run Firecrawl scrape with actions in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._execute_firecrawl_actions, 
                url, actions
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Firecrawl interactive extraction failed: {e}")
            return self._fallback_error_response(str(e))
    
    def _execute_firecrawl_actions(self, url: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Firecrawl scrape with actions (runs in thread executor)."""
        try:
            # Configure scrape parameters with actions
            scrape_params = {
                'formats': ['markdown', 'html'],
                'actions': actions,
                'waitFor': 2000  # Wait 2 seconds after actions
            }
            
            logger.info(f"Executing Firecrawl scrape with actions: {scrape_params}")
            
            # Execute scrape with actions
            scrape_result = self.firecrawl_app.scrape_url(url, **scrape_params)
            
            if scrape_result and scrape_result.get('success'):
                # Format results for SAM's vetting pipeline
                formatted_result = self._format_scrape_results(scrape_result, url, 'interactive_extraction')
                
                logger.info(f"Firecrawl interactive extraction completed successfully")
                return formatted_result
            else:
                error_msg = scrape_result.get('error', 'Unknown scrape error') if scrape_result else 'No result returned'
                return self._fallback_error_response(f"Interactive extraction failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Firecrawl actions execution failed: {e}")
            return self._fallback_error_response(str(e))
    
    async def batch_scrape(self, urls: List[str], max_concurrent: int = 5) -> Dict[str, Any]:
        """
        Batch scrape multiple URLs simultaneously.
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary containing batch results formatted for SAM's vetting pipeline
        """
        try:
            logger.info(f"Starting batch scrape with Firecrawl: {len(urls)} URLs")
            
            if not self.firecrawl_available:
                return self._fallback_error_response(
                    "Firecrawl not available - please install firecrawl-py or configure API key"
                )
            
            # Run batch scrape in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._execute_firecrawl_batch, 
                urls, max_concurrent
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Firecrawl batch scrape failed: {e}")
            return self._fallback_error_response(str(e))
    
    def _execute_firecrawl_batch(self, urls: List[str], max_concurrent: int) -> Dict[str, Any]:
        """Execute Firecrawl batch scrape (runs in thread executor)."""
        try:
            # Configure batch scrape parameters
            batch_params = {
                'urls': urls,
                'formats': ['markdown', 'html']
            }
            
            logger.info(f"Executing Firecrawl batch scrape: {len(urls)} URLs")
            
            # Execute batch scrape
            batch_result = self.firecrawl_app.batch_scrape(**batch_params)
            
            if batch_result and batch_result.get('success'):
                # Format results for SAM's vetting pipeline
                formatted_result = self._format_batch_results(batch_result, urls)
                
                logger.info(f"Firecrawl batch scrape completed: {formatted_result['total_chunks']} chunks extracted")
                return formatted_result
            else:
                error_msg = batch_result.get('error', 'Unknown batch error') if batch_result else 'No result returned'
                return self._fallback_error_response(f"Batch scrape failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Firecrawl batch execution failed: {e}")
            return self._fallback_error_response(str(e))

    def _format_crawl_results(self, crawl_result: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Format Firecrawl crawl results for SAM's vetting pipeline."""
        try:
            chunks = []
            data = crawl_result.get('data', [])

            for i, page in enumerate(data):
                chunk = {
                    'content': page.get('markdown', ''),
                    'title': page.get('metadata', {}).get('title', f'Page {i+1}'),
                    'url': page.get('metadata', {}).get('sourceURL', url),
                    'source': self._extract_domain(page.get('metadata', {}).get('sourceURL', url)),
                    'timestamp': datetime.now().isoformat(),
                    'extraction_method': 'firecrawl_crawl',
                    'metadata': {
                        'html': page.get('html', ''),
                        'language': page.get('metadata', {}).get('language', 'unknown'),
                        'description': page.get('metadata', {}).get('description', ''),
                        'status_code': page.get('metadata', {}).get('statusCode', 200),
                        'firecrawl_metadata': page.get('metadata', {})
                    }
                }
                chunks.append(chunk)

            return {
                'success': True,
                'tool_used': 'firecrawl_tool',
                'extraction_method': 'crawl',
                'chunks': chunks,
                'total_chunks': len(chunks),
                'metadata': {
                    'crawl_url': url,
                    'pages_crawled': len(data),
                    'timestamp': datetime.now().isoformat(),
                    'firecrawl_version': 'firecrawl-py'
                }
            }

        except Exception as e:
            logger.error(f"Failed to format crawl results: {e}")
            return self._fallback_error_response(f"Result formatting failed: {e}")

    def _format_scrape_results(self, scrape_result: Dict[str, Any], url: str, method: str) -> Dict[str, Any]:
        """Format Firecrawl scrape results for SAM's vetting pipeline."""
        try:
            data = scrape_result.get('data', {})

            chunk = {
                'content': data.get('markdown', ''),
                'title': data.get('metadata', {}).get('title', 'Extracted Content'),
                'url': url,
                'source': self._extract_domain(url),
                'timestamp': datetime.now().isoformat(),
                'extraction_method': f'firecrawl_{method}',
                'metadata': {
                    'html': data.get('html', ''),
                    'language': data.get('metadata', {}).get('language', 'unknown'),
                    'description': data.get('metadata', {}).get('description', ''),
                    'status_code': data.get('metadata', {}).get('statusCode', 200),
                    'firecrawl_metadata': data.get('metadata', {})
                }
            }

            return {
                'success': True,
                'tool_used': 'firecrawl_tool',
                'extraction_method': method,
                'chunks': [chunk],
                'total_chunks': 1,
                'metadata': {
                    'scrape_url': url,
                    'timestamp': datetime.now().isoformat(),
                    'firecrawl_version': 'firecrawl-py'
                }
            }

        except Exception as e:
            logger.error(f"Failed to format scrape results: {e}")
            return self._fallback_error_response(f"Result formatting failed: {e}")

    def _format_batch_results(self, batch_result: Dict[str, Any], urls: List[str]) -> Dict[str, Any]:
        """Format Firecrawl batch results for SAM's vetting pipeline."""
        try:
            chunks = []
            data = batch_result.get('data', [])

            for i, page in enumerate(data):
                chunk = {
                    'content': page.get('markdown', ''),
                    'title': page.get('metadata', {}).get('title', f'Batch Item {i+1}'),
                    'url': page.get('metadata', {}).get('sourceURL', urls[i] if i < len(urls) else 'unknown'),
                    'source': self._extract_domain(page.get('metadata', {}).get('sourceURL', urls[i] if i < len(urls) else 'unknown')),
                    'timestamp': datetime.now().isoformat(),
                    'extraction_method': 'firecrawl_batch',
                    'metadata': {
                        'html': page.get('html', ''),
                        'language': page.get('metadata', {}).get('language', 'unknown'),
                        'description': page.get('metadata', {}).get('description', ''),
                        'status_code': page.get('metadata', {}).get('statusCode', 200),
                        'firecrawl_metadata': page.get('metadata', {}),
                        'batch_index': i
                    }
                }
                chunks.append(chunk)

            return {
                'success': True,
                'tool_used': 'firecrawl_tool',
                'extraction_method': 'batch',
                'chunks': chunks,
                'total_chunks': len(chunks),
                'metadata': {
                    'batch_urls': urls,
                    'urls_processed': len(data),
                    'timestamp': datetime.now().isoformat(),
                    'firecrawl_version': 'firecrawl-py'
                }
            }

        except Exception as e:
            logger.error(f"Failed to format batch results: {e}")
            return self._fallback_error_response(f"Result formatting failed: {e}")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc or parsed.path.split('/')[0]
        except:
            return url

    def _fallback_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate standardized error response for fallback scenarios."""
        return {
            'success': False,
            'tool_used': 'firecrawl_tool',
            'error': error_message,
            'chunks': [],
            'total_chunks': 0,
            'metadata': {
                'error_timestamp': datetime.now().isoformat(),
                'fallback_reason': error_message
            }
        }

    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about the Firecrawl tool."""
        return {
            'name': 'FirecrawlTool',
            'description': 'Advanced web crawling and extraction using Firecrawl',
            'available': self.firecrawl_available,
            'capabilities': [
                'Full website crawling',
                'Interactive content extraction',
                'Batch URL processing',
                'Anti-bot mechanisms',
                'JavaScript rendering',
                'Media file extraction'
            ],
            'vetting_integration': True,
            'requires_api_key': self.api_key is not None,
            'version': '1.0.0'
        }
