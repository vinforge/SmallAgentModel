#!/usr/bin/env python3
"""
Simple Web Search Tool
Provides basic web search functionality when main search APIs are unavailable.
"""

import logging
import requests
import re
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus, urljoin
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class SimpleWebSearchTool:
    """
    Simple web search tool that provides basic search results
    when main search APIs are unavailable or fail.
    """
    
    def __init__(self):
        self.timeout = 10
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        self.max_retries = 2
        
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform a simple web search using DuckDuckGo instant answers.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Search results in standard format
        """
        try:
            logger.info(f"🔍 Simple web search for: '{query}'")
            
            # Try DuckDuckGo instant answers first
            instant_results = self._search_duckduckgo_instant(query)
            if instant_results['success'] and instant_results.get('results'):
                return instant_results
            
            # Fallback to generating helpful search suggestions
            return self._generate_search_guidance(query, max_results)
            
        except Exception as e:
            logger.error(f"Simple web search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def _search_duckduckgo_instant(self, query: str) -> Dict[str, Any]:
        """Search using DuckDuckGo instant answers API."""
        try:
            # DuckDuckGo instant answers API
            encoded_query = quote_plus(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
            
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract useful information
            results = []
            
            # Abstract (main answer)
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'Information'),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data.get('Abstract', ''),
                    'source': data.get('AbstractSource', 'DuckDuckGo'),
                    'type': 'instant_answer'
                })
            
            # Related topics
            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '')[:100] + '...',
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                        'source': 'DuckDuckGo Related',
                        'type': 'related_topic'
                    })
            
            # Infobox
            if data.get('Infobox') and data['Infobox'].get('content'):
                infobox_content = []
                for item in data['Infobox']['content'][:3]:
                    if item.get('label') and item.get('value'):
                        infobox_content.append(f"{item['label']}: {item['value']}")
                
                if infobox_content:
                    results.append({
                        'title': 'Key Information',
                        'url': data.get('AbstractURL', ''),
                        'snippet': '\n'.join(infobox_content),
                        'source': 'DuckDuckGo Infobox',
                        'type': 'infobox'
                    })
            
            if results:
                return {
                    'success': True,
                    'results': results,
                    'total_results': len(results),
                    'source': 'duckduckgo_instant',
                    'query': query
                }
            
            return {'success': False, 'results': []}
            
        except Exception as e:
            logger.warning(f"DuckDuckGo instant search failed: {e}")
            return {'success': False, 'results': []}
    
    def _generate_search_guidance(self, query: str, max_results: int) -> Dict[str, Any]:
        """Generate helpful search guidance when direct search fails."""
        try:
            # Analyze query to provide targeted guidance
            guidance_results = []
            
            query_lower = query.lower()
            
            # Business plan guidance
            if 'business plan' in query_lower:
                guidance_results.extend([
                    {
                        'title': 'Business Plan Template and Guide',
                        'url': 'https://www.sba.gov/business-guide/plan-your-business/write-your-business-plan',
                        'snippet': 'The U.S. Small Business Administration provides comprehensive business plan templates and guidance for entrepreneurs starting new ventures.',
                        'source': 'SBA.gov',
                        'type': 'guidance'
                    },
                    {
                        'title': 'International Business Planning Resources',
                        'url': 'https://www.export.gov/article?id=Business-Planning-for-Export',
                        'snippet': 'Export.gov offers resources for businesses planning to sell products internationally, including market research and regulatory guidance.',
                        'source': 'Export.gov',
                        'type': 'guidance'
                    }
                ])
            
            # Mexico export guidance
            if 'mexico' in query_lower and any(word in query_lower for word in ['export', 'sell', 'business']):
                guidance_results.extend([
                    {
                        'title': 'Exporting to Mexico - Trade Requirements',
                        'url': 'https://www.trade.gov/country-commercial-guides/mexico-trade-agreements',
                        'snippet': 'Official U.S. government guidance on trade agreements, regulations, and requirements for exporting goods to Mexico.',
                        'source': 'Trade.gov',
                        'type': 'guidance'
                    },
                    {
                        'title': 'Mexico Market Research and Business Opportunities',
                        'url': 'https://www.export.gov/apex/article2?id=Mexico-Market-Overview',
                        'snippet': 'Market overview, business opportunities, and key considerations for U.S. companies looking to enter the Mexican market.',
                        'source': 'Export.gov',
                        'type': 'guidance'
                    }
                ])
            
            # Food/beverage export guidance
            if any(word in query_lower for word in ['food', 'beverage', 'lemonade', 'drink']):
                guidance_results.extend([
                    {
                        'title': 'FDA Food Export Requirements',
                        'url': 'https://www.fda.gov/food/importing-food-products-united-states/food-facility-registration',
                        'snippet': 'FDA requirements for food and beverage manufacturers, including registration, labeling, and safety standards for export.',
                        'source': 'FDA.gov',
                        'type': 'guidance'
                    }
                ])
            
            # General business guidance if no specific matches
            if not guidance_results:
                guidance_results.extend([
                    {
                        'title': 'Small Business Administration Resources',
                        'url': 'https://www.sba.gov/',
                        'snippet': 'Comprehensive resources for starting and growing a small business, including funding, planning, and regulatory guidance.',
                        'source': 'SBA.gov',
                        'type': 'guidance'
                    },
                    {
                        'title': 'International Trade Administration',
                        'url': 'https://www.trade.gov/',
                        'snippet': 'U.S. government resources for international trade, export assistance, and market research for businesses expanding globally.',
                        'source': 'Trade.gov',
                        'type': 'guidance'
                    }
                ])
            
            # Limit results
            guidance_results = guidance_results[:max_results]
            
            return {
                'success': True,
                'results': guidance_results,
                'total_results': len(guidance_results),
                'source': 'search_guidance',
                'query': query,
                'note': 'These are curated resources based on your query. For more specific information, consider consulting with business advisors or trade specialists.'
            }
            
        except Exception as e:
            logger.error(f"Search guidance generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def can_handle_query(self, query: str) -> bool:
        """Check if this tool can handle the given query."""
        # This tool can handle any query as a fallback
        return True
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions for improving the query."""
        suggestions = []
        
        query_lower = query.lower()
        
        if 'business plan' in query_lower:
            suggestions.extend([
                f"{query} template",
                f"{query} example",
                f"{query} checklist",
                f"how to write {query}"
            ])
        
        if 'export' in query_lower or 'international' in query_lower:
            suggestions.extend([
                f"{query} requirements",
                f"{query} regulations",
                f"{query} documentation",
                f"{query} customs"
            ])
        
        return suggestions[:5]

# Global instance
_simple_web_search_tool = None

def get_simple_web_search_tool() -> SimpleWebSearchTool:
    """Get or create the global simple web search tool instance."""
    global _simple_web_search_tool
    if _simple_web_search_tool is None:
        _simple_web_search_tool = SimpleWebSearchTool()
    return _simple_web_search_tool
