#!/usr/bin/env python3
"""
RSS Reader Tool - Specialized tool for reading and parsing RSS feeds
"""

import logging
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from datetime import datetime
import html
import re

logger = logging.getLogger(__name__)

class RSSReaderTool:
    """Tool for reading and parsing RSS feeds."""
    
    def __init__(self):
        self.timeout = 20
        self.user_agent = 'SAM-RSS-Reader/1.0 (+https://sam-ai.com)'
        
    def read_feed(self, feed_url: str, max_items: int = 15) -> Dict[str, Any]:
        """Read and parse an RSS feed."""
        try:
            logger.info(f"Reading RSS feed: {feed_url}")
            
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/rss+xml, application/xml, text/xml, application/atom+xml'
            }
            
            response = requests.get(feed_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse RSS content
            articles = self._parse_rss_content(response.content, feed_url, max_items)
            
            return {
                'success': True,
                'articles': articles,
                'feed_url': feed_url,
                'total_items': len(articles),
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error reading RSS feed {feed_url}: {e}")
            return {
                'success': False,
                'error': f'HTTP error: {e}',
                'articles': []
            }
        except ET.ParseError as e:
            logger.error(f"XML parsing error for RSS feed {feed_url}: {e}")
            return {
                'success': False,
                'error': f'XML parsing error: {e}',
                'articles': []
            }
        except Exception as e:
            logger.error(f"RSS feed reading failed for {feed_url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'articles': []
            }
    
    def _parse_rss_content(self, content: bytes, feed_url: str, max_items: int) -> List[Dict[str, Any]]:
        """Parse RSS XML content and extract articles."""
        try:
            root = ET.fromstring(content)
            
            # Find items (RSS 2.0, Atom, etc.)
            items = (root.findall('.//item') or 
                    root.findall('.//{http://www.w3.org/2005/Atom}entry') or
                    root.findall('.//entry'))
            
            logger.info(f"Found {len(items)} items in RSS feed")
            
            articles = []
            
            for i, item in enumerate(items[:max_items]):
                try:
                    article = self._extract_article_data(item, feed_url)
                    if article and article.get('title'):
                        articles.append(article)
                        
                        # Debug logging for first few items
                        if i < 3:
                            logger.info(f"✅ Extracted article: {article['title'][:50]}...")
                            
                except Exception as e:
                    logger.warning(f"Error extracting article {i}: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(articles)} articles from RSS feed")
            return articles
            
        except Exception as e:
            logger.error(f"RSS content parsing failed: {e}")
            return []
    
    def _extract_article_data(self, item: ET.Element, feed_url: str) -> Optional[Dict[str, Any]]:
        """Extract data from a single RSS item."""
        try:
            # Extract title
            title = self._get_element_text(item, ['title'])
            if not title or len(title.strip()) < 3:
                return None
            
            # Extract description/summary
            description = self._get_element_text(item, [
                'description', 'summary', 
                '{http://www.w3.org/2005/Atom}summary',
                '{http://www.w3.org/2005/Atom}content'
            ])
            
            # Clean HTML from description
            if description:
                description = re.sub(r'<[^>]+>', '', description)
                description = html.unescape(description).strip()
                if len(description) > 300:
                    description = description[:300] + "..."
            
            # Extract link
            link = self._get_element_text(item, ['link'])
            if not link:
                # Try Atom-style link
                link_elem = item.find('.//{http://www.w3.org/2005/Atom}link')
                if link_elem is not None:
                    link = link_elem.get('href', '')
            
            # Extract publication date
            pub_date = self._get_element_text(item, [
                'pubDate', 'published', 
                '{http://www.w3.org/2005/Atom}published',
                'dc:date'
            ])
            
            # Extract category
            category = self._get_element_text(item, ['category'])
            
            # Extract author
            author = self._get_element_text(item, [
                'author', 'dc:creator',
                '{http://www.w3.org/2005/Atom}author'
            ])
            
            return {
                'title': title.strip(),
                'description': description or '',
                'link': link.strip() if link else '',
                'pub_date': pub_date.strip() if pub_date else '',
                'category': category.strip() if category else '',
                'author': author.strip() if author else '',
                'source': feed_url,
                'tool_source': 'rss_reader',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Article data extraction failed: {e}")
            return None
    
    def _get_element_text(self, item: ET.Element, tag_names: List[str]) -> Optional[str]:
        """Get text from the first matching element."""
        for tag_name in tag_names:
            elem = item.find(tag_name)
            if elem is not None and elem.text:
                return html.unescape(elem.text).strip()
        return None
    
    def read_multiple_feeds(self, feed_urls: List[str], max_items_per_feed: int = 10) -> Dict[str, Any]:
        """Read multiple RSS feeds and combine results."""
        try:
            all_articles = []
            successful_feeds = []
            failed_feeds = []
            
            for feed_url in feed_urls:
                try:
                    result = self.read_feed(feed_url, max_items_per_feed)
                    if result['success']:
                        all_articles.extend(result['articles'])
                        successful_feeds.append(feed_url)
                    else:
                        failed_feeds.append({'url': feed_url, 'error': result.get('error', 'Unknown error')})
                        
                except Exception as e:
                    failed_feeds.append({'url': feed_url, 'error': str(e)})
                    logger.error(f"Failed to read feed {feed_url}: {e}")
            
            # Sort articles by publication date (newest first)
            all_articles.sort(key=lambda x: x.get('pub_date', ''), reverse=True)
            
            return {
                'success': len(all_articles) > 0,
                'articles': all_articles,
                'total_articles': len(all_articles),
                'successful_feeds': successful_feeds,
                'failed_feeds': failed_feeds,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Multiple feeds reading failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'articles': []
            }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about this tool."""
        return {
            'name': 'RSSReaderTool',
            'description': 'Specialized tool for reading and parsing RSS feeds',
            'capabilities': [
                'RSS 2.0 parsing',
                'Atom feed parsing',
                'Multiple feed support',
                'Content extraction',
                'HTML cleaning'
            ],
            'best_for': [
                'News feeds',
                'Blog updates',
                'Content syndication',
                'Regular content monitoring'
            ]
        }
