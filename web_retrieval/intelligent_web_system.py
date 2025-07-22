#!/usr/bin/env python3
"""
Intelligent Web Retrieval System - Main orchestrator for SAM's web intelligence
Coordinates the router and tools for optimal web content retrieval
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import tools
from .tools.cocoindex_tool import CocoIndexTool
from .tools.search_api_tool import SearchAPITool
from .tools.news_api_tool import NewsAPITool
from .tools.rss_reader_tool import RSSReaderTool
from .tools.url_content_extractor import URLContentExtractor
from .tools.firecrawl_tool import FirecrawlTool
from .tools.simple_web_search import get_simple_web_search_tool
from .query_router import QueryRouter

logger = logging.getLogger(__name__)

class IntelligentWebSystem:
    """Main orchestrator for intelligent web content retrieval."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the intelligent web system."""
        self.api_keys = api_keys or {}
        self.config = config or {}

        # Initialize tools
        self.cocoindex_tool = CocoIndexTool(
            api_key=self.api_keys.get('serper'),
            search_provider=self.config.get('cocoindex_search_provider', 'serper'),
            num_pages=self.config.get('cocoindex_num_pages', 5)
        )
        self.search_tool = SearchAPITool(api_key=self.api_keys.get('serper'))
        self.news_tool = NewsAPITool(api_key=self.api_keys.get('newsapi'))
        self.rss_tool = RSSReaderTool()
        self.url_extractor = URLContentExtractor()
        self.firecrawl_tool = FirecrawlTool(
            api_key=self.api_keys.get('firecrawl'),
            timeout=self.config.get('firecrawl_timeout', 30)
        )

        # Initialize router
        self.router = QueryRouter()

        # Tool mapping
        self.tools = {
            'cocoindex_tool': self.cocoindex_tool,
            'search_api_tool': self.search_tool,
            'news_api_tool': self.news_tool,
            'rss_reader_tool': self.rss_tool,
            'url_content_extractor': self.url_extractor,
            'firecrawl_tool': self.firecrawl_tool
        }

        logger.info("Intelligent Web System initialized with all tools including CocoIndex")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using the intelligent routing system."""
        try:
            logger.info(f"Processing query with intelligent web system: '{query}'")
            
            # Step 1: Route the query
            routing_decision = self.router.route_query(query)
            
            # Step 2: Execute primary tool
            primary_result = self._execute_tool(
                routing_decision['primary_tool'], 
                query, 
                routing_decision.get('parameters', {})
            )
            
            # Step 3: Check if primary tool succeeded
            if primary_result['success'] and self._has_sufficient_content(primary_result):
                logger.info(f"Primary tool {routing_decision['primary_tool']} succeeded")
                return self._format_final_result(primary_result, routing_decision, query)
            
            # Step 4: Execute fallback chain
            logger.info(f"Primary tool failed or insufficient content, trying fallbacks")
            for fallback_tool in routing_decision['fallback_chain']:
                try:
                    fallback_result = self._execute_tool(fallback_tool, query, {})

                    if fallback_result['success'] and self._has_sufficient_content(fallback_result):
                        logger.info(f"Fallback tool {fallback_tool} succeeded")
                        return self._format_final_result(fallback_result, routing_decision, query, fallback_tool)

                except Exception as e:
                    logger.warning(f"Fallback tool {fallback_tool} failed: {e}")
                    continue

            # Step 5: Final fallback - Simple Web Search
            logger.info("All configured tools failed, trying simple web search as final fallback")
            try:
                simple_search_tool = get_simple_web_search_tool()
                simple_result = simple_search_tool.search(query, max_results=5)

                if simple_result['success'] and simple_result.get('results'):
                    logger.info("Simple web search succeeded as final fallback")
                    return self._format_simple_search_result(simple_result, routing_decision, query)

            except Exception as e:
                logger.warning(f"Simple web search fallback failed: {e}")

            # Step 6: All tools failed
            return {
                'success': False,
                'error': 'All tools in the fallback chain failed, including simple web search',
                'query': query,
                'routing_decision': routing_decision,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Intelligent web system processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_tool(self, tool_name: str, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with the given query and parameters."""
        try:
            tool = self.tools.get(tool_name)
            if not tool:
                return {'success': False, 'error': f'Tool {tool_name} not found'}
            
            logger.info(f"Executing tool: {tool_name}")
            
            # Execute based on tool type
            if tool_name == 'cocoindex_tool':
                return self._execute_cocoindex_tool(tool, query, parameters)
            elif tool_name == 'search_api_tool':
                return self._execute_search_tool(tool, query, parameters)
            elif tool_name == 'news_api_tool':
                return self._execute_news_tool(tool, query, parameters)
            elif tool_name == 'rss_reader_tool':
                return self._execute_rss_tool(tool, query, parameters)
            elif tool_name == 'url_content_extractor':
                return self._execute_url_tool(tool, query, parameters)
            elif tool_name == 'firecrawl_tool':
                return self._execute_firecrawl_tool(tool, query, parameters)
            else:
                return {'success': False, 'error': f'Unknown tool: {tool_name}'}
                
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_search_tool(self, tool: SearchAPITool, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search API tool with enhanced error handling."""
        num_results = parameters.get('num_results', 10)

        try:
            result = tool.search(query, num_results)

            # If search succeeded, extract content from top URLs
            if result['success'] and result.get('results'):
                top_urls = [item['url'] for item in result['results'][:3]]
                content_results = self.url_extractor.extract_multiple_urls(top_urls)

                return {
                    'success': True,
                    'tool_used': 'search_api_tool',
                    'search_results': result['results'],
                    'extracted_content': content_results.get('results', []),
                    'total_sources': len(top_urls),
                    'timestamp': datetime.now().isoformat()
                }

            # Search API failed or returned no results
            logger.warning(f"Search API failed or returned no results for query: {query}")
            return {
                'success': False,
                'error': 'Search API failed or returned no results',
                'tool_used': 'search_api_tool',
                'fallback_recommended': True,
                'query': query
            }

        except Exception as e:
            logger.error(f"Search API tool execution failed: {e}")
            return {
                'success': False,
                'error': f'Search API execution error: {str(e)}',
                'tool_used': 'search_api_tool',
                'fallback_recommended': True,
                'query': query
            }

    def _execute_cocoindex_tool(self, tool: CocoIndexTool, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CocoIndex tool for intelligent web search."""
        try:
            import asyncio

            # Run the async search
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(tool.intelligent_search(query))
            finally:
                loop.close()

            if result['success']:
                return {
                    'success': True,
                    'tool_used': 'cocoindex_tool',
                    'chunks': result['chunks'],
                    'total_chunks': result['total_chunks'],
                    'search_provider': result.get('search_provider'),
                    'metadata': result.get('metadata', {}),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # CocoIndex failed, return the error for potential fallback
                return result

        except Exception as e:
            logger.error(f"CocoIndex tool execution failed: {e}")
            return {
                'success': False,
                'error': f'CocoIndex execution error: {str(e)}',
                'tool_used': 'cocoindex_tool',
                'fallback_recommended': True
            }

    def _execute_news_tool(self, tool: NewsAPITool, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute news API tool."""
        num_articles = parameters.get('num_articles', 20)
        result = tool.get_news(query, num_articles)

        if result['success']:
            # Convert articles to chunks format for consistency
            chunks = []
            for article in result['articles']:
                chunk = {
                    'content': f"{article.get('title', '')}\n\n{article.get('description', '')}",
                    'title': article.get('title', ''),
                    'source_url': article.get('url', ''),
                    'source_name': article.get('source', {}).get('name', '') if isinstance(article.get('source'), dict) else str(article.get('source', '')),
                    'timestamp': article.get('timestamp', ''),
                    'published_at': article.get('published_at', ''),
                    'tool_source': 'news_api_tool',
                    'content_type': 'news_article'
                }
                chunks.append(chunk)

            return {
                'success': True,
                'tool_used': 'news_api_tool',
                'articles': result['articles'],  # Keep original articles
                'chunks': chunks,  # Add chunks format
                'total_articles': len(result['articles']),
                'total_chunks': len(chunks),
                'source': result.get('source', 'news_api'),
                'timestamp': datetime.now().isoformat()
            }

        return result
    
    def _execute_rss_tool(self, tool: RSSReaderTool, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RSS reader tool."""
        # Get relevant RSS feeds based on query and parameters
        rss_feeds = self._get_rss_feeds_for_query(query, parameters)

        result = tool.read_multiple_feeds(rss_feeds, max_items_per_feed=10)

        if result['success']:
            # Convert articles to chunks format for consistency
            chunks = []
            for article in result['articles']:
                chunk = {
                    'content': f"{article.get('title', '')}\n\n{article.get('description', '')}",
                    'title': article.get('title', ''),
                    'source_url': article.get('link', ''),
                    'source_name': article.get('source', ''),
                    'timestamp': article.get('timestamp', ''),
                    'pub_date': article.get('pub_date', ''),
                    'tool_source': 'rss_reader_tool',
                    'content_type': 'news_article'
                }
                chunks.append(chunk)

            return {
                'success': True,
                'tool_used': 'rss_reader_tool',
                'articles': result['articles'],  # Keep original articles
                'chunks': chunks,  # Add chunks format
                'total_articles': len(result['articles']),
                'total_chunks': len(chunks),
                'successful_feeds': result['successful_feeds'],
                'timestamp': datetime.now().isoformat()
            }

        return result
    
    def _execute_url_tool(self, tool: URLContentExtractor, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute URL content extractor tool."""
        url = parameters.get('url')
        if not url:
            return {'success': False, 'error': 'No URL provided for extraction'}
        
        result = tool.extract_content(url)
        
        if result['success']:
            return {
                'success': True,
                'tool_used': 'url_content_extractor',
                'content': result['content'],
                'metadata': result['metadata'],
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
        
        return result

    def _execute_firecrawl_tool(self, tool: FirecrawlTool, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Firecrawl tool for advanced web crawling and extraction."""
        try:
            import asyncio

            # Determine operation mode based on parameters
            operation_mode = parameters.get('operation_mode', 'crawl')

            # Run the appropriate Firecrawl operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if operation_mode == 'crawl':
                    # Full website crawling
                    url = parameters.get('url', query)
                    max_pages = parameters.get('max_pages', 10)
                    include_subdomains = parameters.get('include_subdomains', False)
                    result = loop.run_until_complete(
                        tool.intelligent_crawl(url, max_pages, include_subdomains)
                    )
                elif operation_mode == 'interactive':
                    # Interactive extraction with actions
                    url = parameters.get('url', query)
                    actions = parameters.get('actions', [])
                    result = loop.run_until_complete(
                        tool.extract_with_actions(url, actions)
                    )
                elif operation_mode == 'batch':
                    # Batch processing multiple URLs
                    urls = parameters.get('urls', [query])
                    max_concurrent = parameters.get('max_concurrent', 5)
                    result = loop.run_until_complete(
                        tool.batch_scrape(urls, max_concurrent)
                    )
                else:
                    # Default to single URL scraping
                    url = parameters.get('url', query)
                    result = loop.run_until_complete(
                        tool.intelligent_crawl(url, 1, False)  # Single page crawl
                    )
            finally:
                loop.close()

            if result['success']:
                # Format result for SAM's vetting pipeline
                formatted_result = {
                    'success': True,
                    'tool_used': 'firecrawl_tool',
                    'extraction_method': result.get('extraction_method', operation_mode),
                    'chunks': result['chunks'],
                    'total_chunks': result['total_chunks'],
                    'metadata': result.get('metadata', {}),
                    'timestamp': datetime.now().isoformat(),
                    'vetting_ready': True  # Mark as ready for SAM's vetting pipeline
                }

                logger.info(f"Firecrawl tool completed successfully: {formatted_result['total_chunks']} chunks extracted")
                return formatted_result
            else:
                # Firecrawl failed, return the error for potential fallback
                logger.warning(f"Firecrawl tool failed: {result.get('error', 'Unknown error')}")
                return result

        except Exception as e:
            logger.error(f"Firecrawl tool execution failed: {e}")
            return {
                'success': False,
                'error': f'Firecrawl execution error: {str(e)}',
                'tool_used': 'firecrawl_tool',
                'fallback_recommended': True
            }

    def _get_rss_feeds_for_query(self, query: str, parameters: Dict[str, Any]) -> List[str]:
        """Get relevant RSS feeds based on query and parameters."""
        query_lower = query.lower()
        topic_category = parameters.get('topic_category')
        specific_source = parameters.get('specific_source')
        
        # Source-specific feeds
        if specific_source == 'cnn':
            return ["http://rss.cnn.com/rss/cnn_latest.rss"]
        elif specific_source == 'bbc':
            return ["https://feeds.bbci.co.uk/news/rss.xml"]
        elif specific_source == 'nytimes':
            return ["https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"]
        
        # Topic-specific feeds
        if topic_category == 'politics':
            return [
                "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
                "https://feeds.bbci.co.uk/news/politics/rss.xml",
                "http://rss.cnn.com/rss/cnn_latest.rss"
            ]
        elif topic_category == 'technology':
            return [
                "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
                "https://feeds.bbci.co.uk/news/technology/rss.xml",
                "http://rss.cnn.com/rss/cnn_latest.rss"
            ]
        elif topic_category == 'business':
            return [
                "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
                "https://feeds.bbci.co.uk/news/business/rss.xml",
                "http://rss.cnn.com/rss/cnn_latest.rss"
            ]
        elif topic_category == 'health':
            return [
                "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
                "https://feeds.bbci.co.uk/news/health/rss.xml",
                "http://rss.cnn.com/rss/cnn_latest.rss"
            ]
        
        # Default general news feeds
        return [
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
            "http://rss.cnn.com/rss/cnn_latest.rss"
        ]
    
    def _has_sufficient_content(self, result: Dict[str, Any]) -> bool:
        """Check if the result has sufficient content."""
        if not result.get('success'):
            return False

        # Check for cocoindex chunks (Phase 8.5)
        chunks = result.get('chunks', [])
        if chunks and len(chunks) > 0:
            # Check if chunks have meaningful content
            total_content_length = sum(len(chunk.get('content', '')) for chunk in chunks)
            return total_content_length > 200

        # Check for articles
        articles = result.get('articles', [])
        if articles and len(articles) > 0:
            return True

        # Check for search results
        search_results = result.get('search_results', [])
        if search_results and len(search_results) > 0:
            return True

        # Check for extracted content
        content = result.get('content', '')
        if content and len(content.strip()) > 100:
            return True

        return False
    
    def _format_final_result(self, result: Dict[str, Any], routing_decision: Dict[str, Any],
                           query: str, actual_tool: Optional[str] = None) -> Dict[str, Any]:
        """Format the final result for return."""
        return {
            'success': True,
            'query': query,
            'tool_used': actual_tool or routing_decision['primary_tool'],
            'routing_decision': routing_decision,
            'data': result,
            'timestamp': datetime.now().isoformat()
        }

    def _format_simple_search_result(self, result: Dict[str, Any], routing_decision: Dict[str, Any],
                                   query: str) -> Dict[str, Any]:
        """Format simple web search results."""
        return {
            'success': True,
            'tool_used': 'simple_web_search',
            'data': result,
            'query': query,
            'routing_decision': routing_decision,
            'timestamp': datetime.now().isoformat(),
            'note': 'Results from simple web search fallback'
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the intelligent web system."""
        return {
            'name': 'IntelligentWebSystem',
            'description': 'Advanced web retrieval system with intelligent routing',
            'available_tools': list(self.tools.keys()),
            'router_info': self.router.get_router_info(),
            'capabilities': [
                'Intelligent query routing',
                'Multi-tool fallback chains',
                'Content quality assessment',
                'CocoIndex intelligent search (Phase 8.5)',
                'Firecrawl advanced web crawling (NEW)',
                'Interactive content extraction (NEW)',
                'Batch URL processing (NEW)',
                'Anti-bot mechanisms (NEW)',
                'API integration',
                'RSS feed processing',
                'URL content extraction'
            ]
        }


# Global instance
_intelligent_web_system = None


def get_intelligent_web_system(api_keys: Optional[Dict[str, str]] = None,
                              config: Optional[Dict[str, Any]] = None) -> IntelligentWebSystem:
    """
    Get or create global intelligent web system instance.

    Args:
        api_keys: Optional API keys dictionary
        config: Optional configuration dictionary

    Returns:
        IntelligentWebSystem instance
    """
    global _intelligent_web_system

    if _intelligent_web_system is None:
        # Try to load API keys from config if not provided
        if api_keys is None:
            try:
                from config.api_keys import get_api_keys
                api_keys = get_api_keys()
            except ImportError:
                api_keys = {}

        _intelligent_web_system = IntelligentWebSystem(api_keys, config)

    return _intelligent_web_system
