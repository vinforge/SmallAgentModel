#!/usr/bin/env python3
"""
Intelligent Query Router - The brain of SAM's web retrieval system
Routes queries to the most appropriate tool with fallback chains
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class QueryRouter:
    """Intelligent router that selects the best tool for each query."""
    
    def __init__(self):
        # News keywords - more specific to avoid false positives
        self.news_keywords = [
            'breaking news', 'latest news', 'news update', 'headlines', 'happening now',
            'just announced', 'breaking', 'urgent news', 'news report', 'press release'
        ]

        # Time-sensitive news keywords (require news context)
        self.time_news_keywords = [
            'latest', 'current', 'recent', 'today', 'update', 'report', 'announced'
        ]

        self.search_keywords = [
            'what is', 'how to', 'explain', 'define', 'features', 'benefits',
            'comparison', 'vs', 'versus', 'difference', 'guide', 'tutorial',
            'average', 'typical', 'standard', 'specifications', 'specs',
            # Enhanced search triggers (preserving 100% of functionality)
            'search up', 'search for', 'search about', 'look up', 'look for',
            'find out', 'find information', 'information about', 'details about',
            'tell me about', 'learn about', 'research', 'investigate',
            'discover', 'explore', 'understand', 'clarify', 'describe'
        ]

        self.topic_categories = {
            'politics': ['politics', 'political', 'election', 'government', 'congress', 'senate', 'president'],
            'technology': [
                'technology', 'tech', 'ai', 'artificial intelligence', 'software', 'programming',
                'computer', 'monitor', 'lcd', 'led', 'oled', 'display', 'screen', 'refresh rate',
                'hz', 'hertz', 'fps', 'resolution', 'pixel', 'graphics', 'gpu', 'cpu', 'hardware',
                'specifications', 'specs', 'performance', 'benchmark', 'gaming', 'processor'
            ],
            'business': ['business', 'economy', 'finance', 'market', 'stock', 'economic', 'financial'],
            'health': ['health', 'medical', 'medicine', 'covid', 'pandemic', 'disease', 'healthcare'],
            'sports': ['sports', 'football', 'basketball', 'baseball', 'soccer', 'olympics', 'game'],
            'science': ['science', 'research', 'study', 'discovery', 'scientific', 'climate']
        }
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route a query to the most appropriate tool with fallback chain."""
        try:
            logger.info(f"Routing query: '{query}'")
            
            # Analyze the query
            analysis = self._analyze_query(query)
            
            # Determine primary tool and fallback chain
            routing_decision = self._make_routing_decision(query, analysis)
            
            logger.info(f"Routing decision: Primary={routing_decision['primary_tool']}, "
                       f"Fallbacks={routing_decision['fallback_chain']}")
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Query routing failed: {e}")
            return self._get_default_routing()
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics."""
        query_lower = query.lower()
        
        analysis = {
            'has_url': self._contains_url(query),
            'is_news_query': self._is_news_query(query_lower),
            'is_search_query': self._is_search_query(query_lower),
            'topic_category': self._detect_topic_category(query_lower),
            'urgency_level': self._assess_urgency(query_lower),
            'query_length': len(query.split()),
            'contains_specific_source': self._detect_specific_source(query_lower),
            # NEW: Firecrawl-specific analysis
            'requires_site_crawling': self._requires_site_crawling(query_lower),
            'requires_interaction': self._requires_interaction(query_lower),
            'multiple_urls': self._has_multiple_urls(query),
            'extracted_urls': self._extract_all_urls(query),
            'suggested_actions': self._suggest_actions(query_lower)
        }
        
        return analysis
    
    def _contains_url(self, query: str) -> bool:
        """Check if query contains a URL."""
        url_pattern = r'https?://[^\s]+'
        return bool(re.search(url_pattern, query))
    
    def _is_news_query(self, query_lower: str) -> bool:
        """Determine if this is a news-related query with improved context awareness."""
        # Check for explicit news keywords first
        explicit_news_score = sum(1 for keyword in self.news_keywords if keyword in query_lower)
        if explicit_news_score >= 1:
            return True

        # Check for time-sensitive keywords but only if they appear in news context
        time_keywords_found = [keyword for keyword in self.time_news_keywords if keyword in query_lower]
        if time_keywords_found:
            # Look for news context indicators
            news_context_indicators = [
                'news', 'article', 'story', 'headline', 'journalist', 'reporter',
                'press', 'media', 'publication', 'newspaper', 'magazine'
            ]

            # Also check for topic categories that suggest technical/factual queries (not news)
            technical_indicators = [
                'specification', 'specs', 'feature', 'performance', 'benchmark',
                'average', 'typical', 'standard', 'rate', 'frequency', 'measurement'
            ]

            # If technical indicators are present, it's likely not a news query
            if any(indicator in query_lower for indicator in technical_indicators):
                return False

            # If news context indicators are present, it's likely a news query
            if any(indicator in query_lower for indicator in news_context_indicators):
                return True

        return False
    
    def _is_search_query(self, query_lower: str) -> bool:
        """Determine if this is a general search query."""
        search_score = sum(1 for keyword in self.search_keywords if keyword in query_lower)
        return search_score >= 1
    
    def _detect_topic_category(self, query_lower: str) -> Optional[str]:
        """Detect the topic category of the query."""
        for category, keywords in self.topic_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return None
    
    def _assess_urgency(self, query_lower: str) -> str:
        """Assess the urgency level of the query."""
        urgent_keywords = ['breaking', 'urgent', 'emergency', 'now', 'immediately']
        high_keywords = ['latest', 'current', 'today', 'recent']
        
        if any(keyword in query_lower for keyword in urgent_keywords):
            return 'urgent'
        elif any(keyword in query_lower for keyword in high_keywords):
            return 'high'
        else:
            return 'normal'
    
    def _detect_specific_source(self, query_lower: str) -> Optional[str]:
        """Detect if query mentions a specific news source."""
        sources = {
            'cnn': 'cnn',
            'bbc': 'bbc',
            'nytimes': 'nytimes',
            'new york times': 'nytimes',
            'reuters': 'reuters',
            'associated press': 'ap',
            'ap news': 'ap'
        }
        
        for source_name, source_key in sources.items():
            if source_name in query_lower:
                return source_key
        return None
    
    def _make_routing_decision(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make the routing decision based on analysis with Firecrawl integration."""

        # Rule 1: Website crawling request -> Firecrawl Tool
        if analysis.get('requires_site_crawling', False):
            return {
                'primary_tool': 'firecrawl_tool',
                'fallback_chain': ['cocoindex_tool', 'search_api_tool'],
                'reasoning': 'Full website crawling required - using Firecrawl',
                'confidence': 0.90,
                'parameters': {
                    'operation_mode': 'crawl',
                    'url': self._extract_url_from_query(query) or query,
                    'max_pages': 20,
                    'include_subdomains': False
                }
            }

        # Rule 2: Interactive content needed -> Firecrawl Tool with actions
        if analysis.get('requires_interaction', False):
            return {
                'primary_tool': 'firecrawl_tool',
                'fallback_chain': ['url_content_extractor'],
                'reasoning': 'Interactive content extraction required - using Firecrawl',
                'confidence': 0.85,
                'parameters': {
                    'operation_mode': 'interactive',
                    'url': self._extract_url_from_query(query) or query,
                    'actions': analysis.get('suggested_actions', [])
                }
            }

        # Rule 3: Multiple URLs -> Firecrawl Batch Tool
        if analysis.get('multiple_urls', False):
            return {
                'primary_tool': 'firecrawl_tool',
                'fallback_chain': ['cocoindex_tool'],
                'reasoning': 'Batch processing multiple URLs - using Firecrawl',
                'confidence': 0.80,
                'parameters': {
                    'operation_mode': 'batch',
                    'urls': analysis.get('extracted_urls', []),
                    'max_concurrent': 5
                }
            }

        # Rule 4: URL in query -> Enhanced routing (Firecrawl vs URL Extractor)
        if analysis['has_url']:
            url = self._extract_url_from_query(query)
            # Use Firecrawl for complex sites, URL extractor for simple ones
            if self._is_complex_site(url):
                return {
                    'primary_tool': 'firecrawl_tool',
                    'fallback_chain': ['url_content_extractor', 'search_api_tool'],
                    'reasoning': 'Complex site detected - using Firecrawl for better extraction',
                    'confidence': 0.88,
                    'parameters': {
                        'operation_mode': 'crawl',
                        'url': url,
                        'max_pages': 5
                    }
                }
            else:
                return {
                    'primary_tool': 'url_content_extractor',
                    'fallback_chain': ['firecrawl_tool', 'search_api_tool'],
                    'reasoning': 'Simple URL - using standard content extraction',
                    'confidence': 0.95,
                    'parameters': {
                        'url': url
                    }
                }

        # Rule 2: Topic-specific query -> Search API Tool (PRIORITIZED)
        # CocoIndex temporarily disabled, using Search API as primary
        if analysis['topic_category']:
            # Check if this is a news-related query even within topic categories
            is_news_query = self._is_news_related_query(query)

            if is_news_query or analysis['topic_category'] in ['politics', 'sports', 'entertainment']:
                # News-related topics can use RSS feeds
                fallback_chain = ['rss_reader_tool', 'news_api_tool', 'url_content_extractor']
            else:
                # Business/informational topics should avoid RSS feeds
                fallback_chain = ['url_content_extractor', 'cocoindex_tool']

            return {
                'primary_tool': 'search_api_tool',
                'fallback_chain': fallback_chain,
                'reasoning': f'Topic-specific query: {analysis["topic_category"]} - {"news-focused" if is_news_query else "informational"} routing',
                'confidence': 0.85,
                'parameters': {
                    'topic_category': analysis['topic_category'],
                    'query_type': 'topic_specific',
                    'is_news_query': is_news_query
                }
            }

        # Rule 3: General search query -> Search API Tool with better fallback
        if analysis['is_search_query'] or analysis['query_length'] > 8:
            # Check if this is a business/informational query vs news query
            is_news_query = self._is_news_related_query(query)

            if is_news_query:
                # News-related queries can use RSS as fallback
                fallback_chain = ['rss_reader_tool', 'url_content_extractor']
            else:
                # Business/informational queries should not use RSS feeds
                fallback_chain = ['url_content_extractor', 'cocoindex_tool']

            return {
                'primary_tool': 'search_api_tool',
                'fallback_chain': fallback_chain,
                'reasoning': f'General search query - {"news-related" if is_news_query else "informational"} (CocoIndex temporarily disabled)',
                'confidence': 0.85,
                'parameters': {
                    'query_type': 'general_search',
                    'is_news_query': is_news_query
                }
            }

        # Rule 4: News query -> News API Tool (now lower priority)
        if analysis['is_news_query']:
            fallback_chain = ['rss_reader_tool', 'search_api_tool']

            return {
                'primary_tool': 'news_api_tool',
                'fallback_chain': fallback_chain,
                'reasoning': 'News-related query detected',
                'confidence': 0.80,
                'parameters': {
                    'urgency': analysis['urgency_level'],
                    'specific_source': analysis['contains_specific_source']
                }
            }

        # Rule 5: Specific source mentioned -> RSS Reader Tool
        if analysis['contains_specific_source']:
            return {
                'primary_tool': 'rss_reader_tool',
                'fallback_chain': ['news_api_tool', 'search_api_tool'],
                'reasoning': f'Specific source mentioned: {analysis["contains_specific_source"]}',
                'confidence': 0.75,
                'parameters': {
                    'source': analysis['contains_specific_source']
                }
            }

        # Default: CocoIndex Tool for intelligent search
        return self._get_default_routing()
    
    def _extract_url_from_query(self, query: str) -> str:
        """Extract URL from query."""
        url_pattern = r'https?://[^\s]+'
        match = re.search(url_pattern, query)
        return match.group(0) if match else ''

    def _is_news_related_query(self, query: str) -> bool:
        """Check if a query is news-related and should use RSS feeds."""
        query_lower = query.lower()

        # News-specific keywords
        news_keywords = [
            'news', 'breaking', 'latest', 'current events', 'headlines',
            'today', 'yesterday', 'this week', 'recent', 'update',
            'politics', 'election', 'government', 'policy',
            'celebrity', 'entertainment', 'sports', 'weather'
        ]

        # Business/informational keywords that should NOT use RSS
        business_keywords = [
            'business plan', 'how to', 'guide', 'tutorial', 'steps',
            'process', 'strategy', 'analysis', 'research', 'study',
            'export', 'import', 'selling', 'marketing', 'startup',
            'investment', 'finance', 'legal requirements', 'regulations'
        ]

        # Check for business/informational keywords first (higher priority)
        if any(keyword in query_lower for keyword in business_keywords):
            return False

        # Check for news keywords
        if any(keyword in query_lower for keyword in news_keywords):
            return True

        # Default: not news-related for general queries
        return False

    def _get_default_routing(self) -> Dict[str, Any]:
        """Get default routing when no specific rules match."""
        return {
            'primary_tool': 'search_api_tool',
            'fallback_chain': ['url_content_extractor', 'cocoindex_tool'],
            'reasoning': 'Default routing - search API with content extraction fallback (avoiding RSS for non-news queries)',
            'confidence': 0.70,
            'parameters': {
                'query_type': 'default'
            }
        }
    
    def get_tool_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """Get ranked tool recommendations for a query."""
        analysis = self._analyze_query(query)
        routing = self.route_query(query)
        
        recommendations = []
        
        # Primary tool
        recommendations.append({
            'tool': routing['primary_tool'],
            'confidence': routing['confidence'],
            'reasoning': routing['reasoning'],
            'rank': 1
        })
        
        # Fallback tools
        for i, tool in enumerate(routing['fallback_chain'], 2):
            recommendations.append({
                'tool': tool,
                'confidence': max(0.3, routing['confidence'] - (i * 0.15)),
                'reasoning': f'Fallback option {i-1}',
                'rank': i
            })
        
        return recommendations
    
    def explain_routing_decision(self, query: str) -> str:
        """Provide human-readable explanation of routing decision."""
        analysis = self._analyze_query(query)
        routing = self.route_query(query)
        
        explanation_parts = [
            f"Query Analysis for: '{query}'",
            f"",
            f"Primary Tool: {routing['primary_tool']}",
            f"Confidence: {routing['confidence']:.2f}",
            f"Reasoning: {routing['reasoning']}",
            f"",
            f"Fallback Chain: {' → '.join(routing['fallback_chain'])}",
            f"",
            f"Query Characteristics:",
            f"  - Contains URL: {analysis['has_url']}",
            f"  - News Query: {analysis['is_news_query']}",
            f"  - Search Query: {analysis['is_search_query']}",
            f"  - Topic Category: {analysis['topic_category'] or 'None'}",
            f"  - Urgency Level: {analysis['urgency_level']}",
            f"  - Specific Source: {analysis['contains_specific_source'] or 'None'}"
        ]
        
        return "\n".join(explanation_parts)

    def get_router_info(self) -> Dict[str, Any]:
        """Get information about the router."""
        return {
            'name': 'QueryRouter',
            'description': 'Intelligent router that selects the best tool for each query',
            'available_tools': [
                'firecrawl_tool',
                'cocoindex_tool',
                'search_api_tool',
                'news_api_tool',
                'rss_reader_tool',
                'url_content_extractor'
            ],
            'routing_rules': [
                'Website crawling → Firecrawl Tool (NEW)',
                'Interactive content → Firecrawl Tool (NEW)',
                'Multiple URLs → Firecrawl Batch Tool (NEW)',
                'Complex sites → Firecrawl Tool (NEW)',
                'Simple URLs → URL Content Extractor',
                'Topic-specific query → CocoIndex Tool',
                'General search query → CocoIndex Tool',
                'News keywords → News API Tool',
                'Specific source → RSS Reader Tool',
                'Default → CocoIndex Tool (intelligent search)'
            ]
        }

    # NEW: Firecrawl-specific analysis methods
    def _requires_site_crawling(self, query_lower: str) -> bool:
        """Detect if query requires full website crawling."""
        crawling_keywords = [
            'crawl', 'entire site', 'whole website', 'all pages', 'site map',
            'complete website', 'full site', 'entire domain', 'all content',
            'website analysis', 'site analysis', 'comprehensive analysis'
        ]
        return any(keyword in query_lower for keyword in crawling_keywords)

    def _requires_interaction(self, query_lower: str) -> bool:
        """Detect if query requires interactive content extraction."""
        interaction_keywords = [
            'login', 'sign in', 'form', 'submit', 'click', 'button',
            'dynamic content', 'javascript', 'interactive', 'ajax',
            'behind login', 'requires authentication', 'member content'
        ]
        return any(keyword in query_lower for keyword in interaction_keywords)

    def _has_multiple_urls(self, query: str) -> bool:
        """Check if query contains multiple URLs."""
        urls = self._extract_all_urls(query)
        return len(urls) > 1

    def _extract_all_urls(self, query: str) -> List[str]:
        """Extract all URLs from query."""
        url_pattern = r'https?://[^\s]+'
        return re.findall(url_pattern, query)

    def _suggest_actions(self, query_lower: str) -> List[Dict[str, Any]]:
        """Suggest Firecrawl actions based on query content."""
        actions = []

        if 'login' in query_lower or 'sign in' in query_lower:
            actions.append({'type': 'wait', 'milliseconds': 2000})
            actions.append({'type': 'click', 'selector': 'input[type="email"], input[name="username"]'})

        if 'search' in query_lower:
            actions.append({'type': 'wait', 'milliseconds': 1000})
            actions.append({'type': 'click', 'selector': 'input[type="search"], input[name="q"]'})

        if 'form' in query_lower or 'submit' in query_lower:
            actions.append({'type': 'wait', 'milliseconds': 2000})

        return actions

    def _is_complex_site(self, url: str) -> bool:
        """Determine if a site is complex and would benefit from Firecrawl."""
        if not url:
            return False

        # Sites known to be complex or have anti-bot measures
        complex_domains = [
            'linkedin.com', 'facebook.com', 'twitter.com', 'instagram.com',
            'amazon.com', 'ebay.com', 'airbnb.com', 'booking.com',
            'netflix.com', 'spotify.com', 'youtube.com', 'tiktok.com',
            'reddit.com', 'quora.com', 'medium.com', 'substack.com'
        ]

        # Check if URL contains any complex domain
        url_lower = url.lower()
        return any(domain in url_lower for domain in complex_domains)
