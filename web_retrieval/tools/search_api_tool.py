#!/usr/bin/env python3
"""
Search API Tool - Uses search engines to find relevant URLs and snippets
"""

import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SearchAPITool:
    """Tool for performing web searches using search APIs."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
        self.timeout = 10
        
    def search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Perform web search and return URLs with snippets."""
        try:
            logger.info(f"Performing search API query: '{query}'")
            
            if self.api_key:
                # Use Serper API if available
                return self._search_with_serper(query, num_results)
            else:
                # Fallback to DuckDuckGo search
                return self._search_with_duckduckgo(query, num_results)
                
        except Exception as e:
            logger.error(f"Search API failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def _search_with_serper(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using Serper API."""
        try:
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'q': query,
                'num': num_results,
                'gl': 'us',
                'hl': 'en'
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Process organic results
                for item in data.get('organic', []):
                    result = {
                        'title': item.get('title', ''),
                        'url': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'serper_search',
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                
                return {
                    'success': True,
                    'results': results,
                    'total_results': len(results),
                    'query': query,
                    'source': 'serper'
                }
            else:
                logger.error(f"Serper API error: {response.status_code}")
                return self._search_with_duckduckgo(query, num_results)
                
        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return self._search_with_duckduckgo(query, num_results)
    
    def _search_with_duckduckgo(self, query: str, num_results: int) -> Dict[str, Any]:
        """Fallback search using DuckDuckGo."""
        try:
            # Simple DuckDuckGo instant answer API
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(
                'https://api.duckduckgo.com/',
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Process related topics
                for topic in data.get('RelatedTopics', [])[:num_results]:
                    if isinstance(topic, dict) and 'FirstURL' in topic:
                        result = {
                            'title': topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', ''),
                            'url': topic.get('FirstURL', ''),
                            'snippet': topic.get('Text', ''),
                            'source': 'duckduckgo_search',
                            'timestamp': datetime.now().isoformat()
                        }
                        results.append(result)
                
                # If no related topics, create a basic search result
                if not results and data.get('AbstractURL'):
                    result = {
                        'title': data.get('Heading', query),
                        'url': data.get('AbstractURL', ''),
                        'snippet': data.get('Abstract', ''),
                        'source': 'duckduckgo_search',
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                
                return {
                    'success': True,
                    'results': results,
                    'total_results': len(results),
                    'query': query,
                    'source': 'duckduckgo'
                }
            else:
                # Handle different status codes more gracefully
                if response.status_code == 202:
                    logger.warning(f"DuckDuckGo API returned 202 (Accepted) - request is being processed")
                    # For 202, try to parse any partial results
                    try:
                        data = response.json()
                        if data and (data.get('RelatedTopics') or data.get('AbstractURL')):
                            # Process partial results
                            results = []
                            for topic in data.get('RelatedTopics', []):
                                if isinstance(topic, dict) and topic.get('FirstURL'):
                                    result = {
                                        'title': topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', ''),
                                        'url': topic.get('FirstURL', ''),
                                        'snippet': topic.get('Text', ''),
                                        'source': 'duckduckgo_search_partial',
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    results.append(result)

                            if results:
                                logger.info(f"DuckDuckGo returned partial results despite 202 status")
                                return {
                                    'success': True,
                                    'results': results,
                                    'total_results': len(results),
                                    'query': query,
                                    'source': 'duckduckgo_partial'
                                }
                    except:
                        pass

                logger.error(f"DuckDuckGo API error: {response.status_code}")
                return {
                    'success': False,
                    'error': f'DuckDuckGo API returned {response.status_code}',
                    'results': []
                }
                
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about this tool."""
        return {
            'name': 'SearchAPITool',
            'description': 'Performs web searches using search APIs to find relevant URLs and snippets',
            'capabilities': [
                'General web search',
                'URL discovery',
                'Content snippets',
                'Multiple search engines'
            ],
            'best_for': [
                'General knowledge queries',
                'Finding specific websites',
                'Research topics',
                'URL discovery'
            ]
        }
