#!/usr/bin/env python3
"""
SAM Scrapy Manager - Intelligent web content extraction using Scrapy
"""

import logging
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
import sys
import os

logger = logging.getLogger(__name__)

class ScrapyManager:
    """Manages Scrapy spiders for intelligent web content extraction."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure Scrapy settings
        self.scrapy_settings = {
            'USER_AGENT': 'SAM-SecureBot/1.0 (+https://sam-ai.com)',
            'ROBOTSTXT_OBEY': True,
            'CONCURRENT_REQUESTS': 2,
            'DOWNLOAD_DELAY': 1,
            'RANDOMIZE_DOWNLOAD_DELAY': True,
            'AUTOTHROTTLE_ENABLED': True,
            'AUTOTHROTTLE_START_DELAY': 1,
            'AUTOTHROTTLE_MAX_DELAY': 10,
            'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
            'FEEDS': {},
            'LOG_LEVEL': 'WARNING'
        }
        
    def scrape_news_content(self, query: str, urls: List[str]) -> Dict[str, Any]:
        """Scrape news content from multiple URLs based on query."""
        try:
            logger.info(f"Starting Scrapy news extraction for query: '{query}' from {len(urls)} URLs")
            
            # Create temporary output file
            output_file = self.output_dir / f"news_scrape_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Configure feed output
            self.scrapy_settings['FEEDS'] = {
                str(output_file): {
                    'format': 'json',
                    'overwrite': True
                }
            }
            
            # Determine appropriate spider based on URLs
            spider_name = self._select_spider(urls)
            
            # Run Scrapy spider
            result = self._run_spider(spider_name, urls, query, output_file)
            
            if result['success']:
                # Load and process scraped data
                scraped_data = self._load_scraped_data(output_file)
                processed_data = self._process_scraped_data(scraped_data, query)
                
                # Cleanup
                if output_file.exists():
                    output_file.unlink()
                
                return {
                    'success': True,
                    'data': processed_data,
                    'source_count': len(processed_data.get('articles', [])),
                    'query': query
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Scrapy extraction failed'),
                    'data': None
                }
                
        except Exception as e:
            logger.error(f"Scrapy news extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
    
    def _select_spider(self, urls: List[str]) -> str:
        """Select appropriate spider based on URLs."""
        # Analyze URLs to determine best spider
        domains = [self._extract_domain(url) for url in urls]
        
        if any('cnn.com' in domain for domain in domains):
            return 'cnn_spider'
        elif any('nytimes.com' in domain for domain in domains):
            return 'nytimes_spider'
        elif any('bbc.co.uk' in domain or 'bbc.com' in domain for domain in domains):
            return 'bbc_spider'
        else:
            return 'general_news_spider'
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except:
            return url.lower()
    
    def _run_spider(self, spider_name: str, urls: List[str], query: str, output_file: Path) -> Dict[str, Any]:
        """Run Scrapy spider with specified parameters."""
        try:
            # Create spider command
            spider_script = self._create_spider_script(spider_name, urls, query, output_file)
            
            # Execute spider
            result = subprocess.run([
                sys.executable, '-c', spider_script
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return {'success': True}
            else:
                logger.error(f"Spider execution failed: {result.stderr}")
                return {'success': False, 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Spider execution timed out'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_spider_script(self, spider_name: str, urls: List[str], query: str, output_file: Path) -> str:
        """Create Python script to run the spider."""
        urls_str = json.dumps(urls)
        output_str = str(output_file)
        
        script = f'''
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import json
import re
from datetime import datetime

class NewsSpider(scrapy.Spider):
    name = "{spider_name}"
    start_urls = {urls_str}
    custom_settings = {{
        'USER_AGENT': 'SAM-SecureBot/1.0 (+https://sam-ai.com)',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 2,
        'DOWNLOAD_DELAY': 1,
        'FEEDS': {{
            "{output_str}": {{"format": "json", "overwrite": True}}
        }},
        'LOG_LEVEL': 'WARNING'
    }}
    
    def parse(self, response):
        # Extract articles based on site structure
        articles = self.extract_articles(response)
        
        for article in articles:
            if self.is_relevant_article(article, "{query}"):
                yield article
    
    def extract_articles(self, response):
        articles = []
        
        # CNN-specific selectors
        if 'cnn.com' in response.url:
            for article in response.css('article, .card, .cd__content'):
                title = article.css('h1::text, h2::text, h3::text, .headline::text, .cd__headline::text').get()
                summary = article.css('p::text, .summary::text, .cd__description::text').get()
                link = article.css('a::attr(href)').get()
                
                if title and len(title.strip()) > 10:
                    articles.append({{
                        'title': title.strip(),
                        'summary': summary.strip() if summary else '',
                        'link': response.urljoin(link) if link else response.url,
                        'source': 'CNN',
                        'scraped_at': datetime.now().isoformat()
                    }})
        
        # BBC-specific selectors
        elif 'bbc.co.uk' in response.url or 'bbc.com' in response.url:
            for article in response.css('article, .gs-c-promo, .media'):
                title = article.css('h1::text, h2::text, h3::text, .gs-c-promo-heading__title::text').get()
                summary = article.css('p::text, .gs-c-promo-summary::text').get()
                link = article.css('a::attr(href)').get()
                
                if title and len(title.strip()) > 10:
                    articles.append({{
                        'title': title.strip(),
                        'summary': summary.strip() if summary else '',
                        'link': response.urljoin(link) if link else response.url,
                        'source': 'BBC',
                        'scraped_at': datetime.now().isoformat()
                    }})
        
        # NYTimes-specific selectors
        elif 'nytimes.com' in response.url:
            for article in response.css('article, .story, .css-1l4spti'):
                title = article.css('h1::text, h2::text, h3::text, .css-1kv6qi0::text').get()
                summary = article.css('p::text, .summary::text, .css-1echdzn::text').get()
                link = article.css('a::attr(href)').get()
                
                if title and len(title.strip()) > 10:
                    articles.append({{
                        'title': title.strip(),
                        'summary': summary.strip() if summary else '',
                        'link': response.urljoin(link) if link else response.url,
                        'source': 'New York Times',
                        'scraped_at': datetime.now().isoformat()
                    }})
        
        # General news selectors
        else:
            for article in response.css('article, .article, .news-item, .story'):
                title = article.css('h1::text, h2::text, h3::text, .title::text, .headline::text').get()
                summary = article.css('p::text, .summary::text, .description::text').get()
                link = article.css('a::attr(href)').get()
                
                if title and len(title.strip()) > 10:
                    articles.append({{
                        'title': title.strip(),
                        'summary': summary.strip() if summary else '',
                        'link': response.urljoin(link) if link else response.url,
                        'source': response.url,
                        'scraped_at': datetime.now().isoformat()
                    }})
        
        return articles[:10]  # Limit to top 10 articles
    
    def is_relevant_article(self, article, query):
        # Simple relevance check
        query_words = query.lower().split()
        title_lower = article.get('title', '').lower()
        summary_lower = article.get('summary', '').lower()
        
        # Check if any query words appear in title or summary
        for word in query_words:
            if len(word) > 3 and (word in title_lower or word in summary_lower):
                return True
        
        return True  # Include all articles for now

# Run the spider
process = CrawlerProcess()
process.crawl(NewsSpider)
process.start()
'''
        return script
    
    def _load_scraped_data(self, output_file: Path) -> List[Dict[str, Any]]:
        """Load scraped data from output file."""
        try:
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else [data]
            return []
        except Exception as e:
            logger.error(f"Failed to load scraped data: {e}")
            return []
    
    def _process_scraped_data(self, scraped_data: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Process and format scraped data."""
        try:
            articles = []
            sources = set()
            
            for item in scraped_data:
                if isinstance(item, dict) and item.get('title'):
                    articles.append({
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'link': item.get('link', ''),
                        'source': item.get('source', 'Unknown'),
                        'scraped_at': item.get('scraped_at', datetime.now().isoformat())
                    })
                    sources.add(item.get('source', 'Unknown'))
            
            return {
                'query': query,
                'articles': articles,
                'source_count': len(sources),
                'article_count': len(articles),
                'sources': list(sources),
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process scraped data: {e}")
            return {
                'query': query,
                'articles': [],
                'source_count': 0,
                'article_count': 0,
                'sources': [],
                'scraped_at': datetime.now().isoformat()
            }
