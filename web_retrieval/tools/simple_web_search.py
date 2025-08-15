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
        Perform a simple web search using DuckDuckGo instant answers and web search.

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
                logger.info(f"✅ DuckDuckGo instant answers found {len(instant_results['results'])} results")
                return instant_results

            # Try DuckDuckGo web search as primary fallback
            web_search_results = self._search_duckduckgo_web(query, max_results)
            if web_search_results['success'] and web_search_results.get('results'):
                logger.info(f"✅ DuckDuckGo web search found {len(web_search_results['results'])} results")
                return web_search_results

            # Try alternative search method as secondary fallback
            alt_search_results = self._search_alternative_method(query, max_results)
            if alt_search_results['success'] and alt_search_results.get('results'):
                logger.info(f"✅ Alternative search found {len(alt_search_results['results'])} results")
                return alt_search_results

            # Only use guidance as last resort and mark it clearly
            logger.warning(f"⚠️ No web search results found for '{query}', providing guidance resources")
            guidance_results = self._generate_search_guidance(query, max_results)
            guidance_results['is_guidance'] = True  # Mark as guidance, not actual search results
            guidance_results['note'] = 'No current web results found. Showing curated guidance resources.'
            return guidance_results

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

    def _search_duckduckgo_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform actual web search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            Search results with actual web sources
        """
        try:
            import requests
            from urllib.parse import quote_plus
            import re
            from bs4 import BeautifulSoup

            # DuckDuckGo web search URL
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            results = []

            # Parse search results
            result_divs = soup.find_all('div', class_='result')

            for result_div in result_divs[:max_results]:
                try:
                    # Extract title and URL
                    title_link = result_div.find('a', class_='result__a')
                    if not title_link:
                        continue

                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')

                    # Extract snippet
                    snippet_div = result_div.find('div', class_='result__snippet')
                    snippet = snippet_div.get_text(strip=True) if snippet_div else ''

                    # Clean up URL (DuckDuckGo sometimes wraps URLs)
                    if url.startswith('/l/?uddg='):
                        # Extract actual URL from DuckDuckGo redirect
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                        if 'uddg' in parsed:
                            url = urllib.parse.unquote(parsed['uddg'][0])

                    if title and url:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'source': self._extract_domain_from_url(url),
                            'type': 'web_search'
                        })

                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue

            if results:
                return {
                    'success': True,
                    'results': results,
                    'total_results': len(results),
                    'source': 'duckduckgo_web_search',
                    'query': query
                }
            else:
                return {'success': False, 'results': []}

        except Exception as e:
            logger.warning(f"DuckDuckGo web search failed: {e}")
            return {'success': False, 'results': []}

    def _search_alternative_method(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Alternative search method using a different approach.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            Search results from alternative method
        """
        try:
            import requests
            from urllib.parse import quote_plus
            import json

            # Try using a different search approach
            # This could be expanded to use other search APIs or methods

            # For now, try a simple search using a different user agent and approach
            search_url = f"https://duckduckgo.com/lite/?q={quote_plus(query)}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse the lite version which is simpler
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []

            # Find result links in the lite version
            result_links = soup.find_all('a', href=True)

            for link in result_links[:max_results * 2]:  # Get more to filter
                href = link.get('href', '')
                text = link.get_text(strip=True)

                # Filter for actual result links (not navigation)
                if (href.startswith('http') and
                    not href.startswith('https://duckduckgo.com') and
                    len(text) > 10 and
                    not any(skip in href.lower() for skip in ['javascript:', 'mailto:', '#'])):

                    results.append({
                        'title': text,
                        'url': href,
                        'snippet': f"Search result for: {query}",
                        'source': self._extract_domain_from_url(href),
                        'type': 'web_search'
                    })

                    if len(results) >= max_results:
                        break

            if results:
                return {
                    'success': True,
                    'results': results,
                    'total_results': len(results),
                    'source': 'alternative_web_search',
                    'query': query
                }
            else:
                return {'success': False, 'results': []}

        except Exception as e:
            logger.warning(f"Alternative search method failed: {e}")
            return {'success': False, 'results': []}

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain name from URL for source attribution."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            # Return domain if valid, otherwise return original URL
            return domain if domain else url
        except Exception:
            return url
    
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
