#!/usr/bin/env python3
"""
URL Content Extractor - Uses Playwright + Trafilatura for smart content extraction
"""

import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
import sys
import tempfile
import json

logger = logging.getLogger(__name__)

class URLContentExtractor:
    """Tool for extracting clean content from individual URLs."""
    
    def __init__(self):
        self.timeout = 30
        self.user_agent = 'SAM-ContentExtractor/1.0 (+https://sam-ai.com)'
        self.max_content_length = 50000
        
    def extract_content(self, url: str) -> Dict[str, Any]:
        """Extract clean content from a URL."""
        try:
            logger.info(f"Extracting content from URL: {url}")
            
            # Try Trafilatura first (faster, no browser needed)
            trafilatura_result = self._extract_with_trafilatura(url)
            if trafilatura_result['success'] and trafilatura_result.get('content'):
                logger.info("Successfully extracted content with Trafilatura")
                return trafilatura_result
            
            # Fallback to requests + basic parsing
            logger.info("Trafilatura failed, trying requests fallback")
            return self._extract_with_requests(url)
            
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'content': '',
                'metadata': {}
            }
    
    def _extract_with_trafilatura(self, url: str) -> Dict[str, Any]:
        """Extract content using Trafilatura library."""
        try:
            # Try to import trafilatura
            try:
                import trafilatura
            except ImportError:
                logger.warning("Trafilatura not available, installing...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trafilatura'])
                import trafilatura
            
            # Download and extract content
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return {'success': False, 'error': 'Failed to download content'}
            
            # Extract main content
            content = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if not content:
                return {'success': False, 'error': 'No content extracted'}
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            
            # Limit content length
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "... [Content truncated]"
            
            return {
                'success': True,
                'content': content,
                'metadata': {
                    'title': metadata.title if metadata else '',
                    'author': metadata.author if metadata else '',
                    'date': metadata.date if metadata else '',
                    'description': metadata.description if metadata else '',
                    'sitename': metadata.sitename if metadata else '',
                    'url': url,
                    'extraction_method': 'trafilatura',
                    'content_length': len(content),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Trafilatura extraction failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_with_requests(self, url: str) -> Dict[str, Any]:
        """Fallback extraction using requests + BeautifulSoup."""
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Try to import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                logger.warning("BeautifulSoup not available, installing...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'beautifulsoup4'])
                from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ''
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Extract main content
            content_parts = []
            
            # Try to find main content areas
            main_selectors = [
                'main', 'article', '.content', '.post-content', 
                '.entry-content', '.article-content', '#content'
            ]
            
            main_content = None
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                # Extract text from main content area
                for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    text = p.get_text().strip()
                    if text and len(text) > 20:  # Filter out very short paragraphs
                        content_parts.append(text)
            else:
                # Fallback: extract all paragraphs
                for p in soup.find_all('p'):
                    text = p.get_text().strip()
                    if text and len(text) > 20:
                        content_parts.append(text)
            
            content = '\n\n'.join(content_parts)
            
            # Limit content length
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "... [Content truncated]"
            
            # Extract metadata
            description = ''
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                description = desc_tag.get('content', '')
            
            return {
                'success': True,
                'content': content,
                'metadata': {
                    'title': title,
                    'description': description,
                    'url': url,
                    'extraction_method': 'requests_beautifulsoup',
                    'content_length': len(content),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Requests extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'content': '',
                'metadata': {}
            }
    
    def extract_multiple_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Extract content from multiple URLs."""
        try:
            results = []
            successful_extractions = 0
            
            for url in urls:
                try:
                    result = self.extract_content(url)
                    results.append(result)
                    if result['success']:
                        successful_extractions += 1
                except Exception as e:
                    logger.error(f"Failed to extract from {url}: {e}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'url': url,
                        'content': '',
                        'metadata': {}
                    })
            
            return {
                'success': successful_extractions > 0,
                'results': results,
                'total_urls': len(urls),
                'successful_extractions': successful_extractions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Multiple URL extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about this tool."""
        return {
            'name': 'URLContentExtractor',
            'description': 'Extracts clean content from individual URLs using advanced parsing',
            'capabilities': [
                'Smart content extraction',
                'HTML cleaning',
                'Metadata extraction',
                'Multiple extraction methods',
                'Content length limiting'
            ],
            'best_for': [
                'Article content extraction',
                'Blog post reading',
                'Documentation parsing',
                'Clean text extraction'
            ]
        }
