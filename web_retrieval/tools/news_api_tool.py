#!/usr/bin/env python3
"""
News API Tool - Uses NewsAPI.org and other news sources for current news
"""

import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class NewsAPITool:
    """Tool for fetching current news using news APIs."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.newsapi_url = "https://newsapi.org/v2/everything"
        self.timeout = 15
        
    def get_news(self, query: str, num_articles: int = 20) -> Dict[str, Any]:
        """Get current news articles for a query."""
        try:
            logger.info(f"Fetching news for query: '{query}'")
            
            if self.api_key:
                # Use NewsAPI if available
                return self._fetch_with_newsapi(query, num_articles)
            else:
                # Fallback to RSS-based news fetching
                return self._fetch_with_rss_fallback(query, num_articles)
                
        except Exception as e:
            logger.error(f"News API failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'articles': []
            }
    
    def _fetch_with_newsapi(self, query: str, num_articles: int) -> Dict[str, Any]:
        """Fetch news using NewsAPI.org."""
        try:
            # Calculate date range (last 7 days)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)
            
            params = {
                'q': query,
                'apiKey': self.api_key,
                'sortBy': 'publishedAt',
                'pageSize': min(num_articles, 100),
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'language': 'en'
            }
            
            response = requests.get(
                self.newsapi_url,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for article in data.get('articles', []):
                    processed_article = {
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'published_at': article.get('publishedAt', ''),
                        'author': article.get('author', ''),
                        'url_to_image': article.get('urlToImage', ''),
                        'content_preview': article.get('content', ''),
                        'tool_source': 'newsapi',
                        'timestamp': datetime.now().isoformat()
                    }
                    articles.append(processed_article)
                
                return {
                    'success': True,
                    'articles': articles,
                    'total_results': data.get('totalResults', len(articles)),
                    'query': query,
                    'source': 'newsapi'
                }
            else:
                logger.error(f"NewsAPI error: {response.status_code}")
                return self._fetch_with_rss_fallback(query, num_articles)
                
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return self._fetch_with_rss_fallback(query, num_articles)
    
    def _fetch_with_rss_fallback(self, query: str, num_articles: int) -> Dict[str, Any]:
        """Fallback to RSS-based news fetching."""
        try:
            # Use the existing RSS system as fallback
            rss_feeds = self._get_relevant_rss_feeds(query)
            articles = []
            
            for feed_url in rss_feeds:
                try:
                    feed_articles = self._fetch_rss_feed(feed_url, query)
                    articles.extend(feed_articles)
                except Exception as e:
                    logger.warning(f"RSS feed {feed_url} failed: {e}")
                    continue
            
            # Limit to requested number
            articles = articles[:num_articles]
            
            return {
                'success': True,
                'articles': articles,
                'total_results': len(articles),
                'query': query,
                'source': 'rss_fallback'
            }
            
        except Exception as e:
            logger.error(f"RSS fallback failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'articles': []
            }
    
    def _get_relevant_rss_feeds(self, query: str) -> List[str]:
        """Get relevant RSS feeds based on query."""
        query_lower = query.lower()
        
        # Topic-specific RSS feeds
        if any(word in query_lower for word in ['politics', 'political', 'election']):
            return [
                "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
                "https://feeds.bbci.co.uk/news/politics/rss.xml",
                "http://rss.cnn.com/rss/cnn_latest.rss"
            ]
        elif any(word in query_lower for word in ['technology', 'tech', 'ai']):
            return [
                "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
                "https://feeds.bbci.co.uk/news/technology/rss.xml",
                "http://rss.cnn.com/rss/cnn_latest.rss"
            ]
        elif any(word in query_lower for word in ['business', 'economy', 'finance']):
            return [
                "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
                "https://feeds.bbci.co.uk/news/business/rss.xml",
                "http://rss.cnn.com/rss/cnn_latest.rss"
            ]
        elif any(word in query_lower for word in ['health', 'medical', 'medicine']):
            return [
                "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
                "https://feeds.bbci.co.uk/news/health/rss.xml",
                "http://rss.cnn.com/rss/cnn_latest.rss"
            ]
        else:
            # General news feeds
            return [
                "https://feeds.bbci.co.uk/news/rss.xml",
                "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
                "http://rss.cnn.com/rss/cnn_latest.rss"
            ]
    
    def _fetch_rss_feed(self, feed_url: str, query: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed."""
        try:
            import xml.etree.ElementTree as ET
            import re
            
            response = requests.get(feed_url, timeout=self.timeout)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            items = root.findall('.//item')
            
            articles = []
            for item in items[:10]:  # Limit per feed
                title_elem = item.find('title')
                desc_elem = item.find('description')
                link_elem = item.find('link')
                date_elem = item.find('pubDate')
                
                if title_elem is not None and title_elem.text:
                    title = title_elem.text.strip()
                    description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ''
                    
                    # Clean HTML from description
                    description = re.sub(r'<[^>]+>', '', description)
                    
                    article = {
                        'title': title,
                        'description': description,
                        'url': link_elem.text.strip() if link_elem is not None and link_elem.text else '',
                        'source': feed_url,
                        'published_at': date_elem.text.strip() if date_elem is not None and date_elem.text else '',
                        'tool_source': 'rss_fallback',
                        'timestamp': datetime.now().isoformat()
                    }
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"RSS feed parsing failed for {feed_url}: {e}")
            return []
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about this tool."""
        return {
            'name': 'NewsAPITool',
            'description': 'Fetches current news articles using news APIs and RSS feeds',
            'capabilities': [
                'Current news retrieval',
                'Topic-specific news',
                'Multiple news sources',
                'Recent articles (last 7 days)'
            ],
            'best_for': [
                'Latest news queries',
                'Breaking news',
                'Current events',
                'Topic-specific news'
            ]
        }
