#!/usr/bin/env python3
"""
Content Formatter for Scrapy Results
Converts scraped news data into readable format for SAM
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ContentFormatter:
    """Formats scraped content for SAM's consumption."""
    
    def __init__(self):
        self.max_articles = 15
        self.max_summary_length = 300
    
    def format_news_content(self, scraped_data: Dict[str, Any]) -> str:
        """Format scraped news data into readable content."""
        try:
            if not scraped_data.get('articles'):
                return "No news articles were successfully extracted from the web sources."
            
            articles = scraped_data['articles'][:self.max_articles]
            sources = scraped_data.get('sources', [])
            
            # Create formatted content
            content_parts = []
            
            # Header
            content_parts.append(f"**Latest News Results for: {scraped_data.get('query', 'News Search')}**")
            content_parts.append(f"*Found {len(articles)} articles from {len(sources)} sources*")
            content_parts.append("")
            
            # Articles
            for i, article in enumerate(articles, 1):
                article_content = self._format_article(article, i)
                if article_content:
                    content_parts.append(article_content)
                    content_parts.append("---")
            
            # Remove last separator
            if content_parts and content_parts[-1] == "---":
                content_parts.pop()
            
            # Footer with sources
            content_parts.append("")
            content_parts.append("**Sources:**")
            for source in sources:
                content_parts.append(f"• {source}")
            
            content_parts.append("")
            content_parts.append(f"*Content scraped at: {scraped_data.get('scraped_at', datetime.now().isoformat())}*")
            
            return "\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Content formatting failed: {e}")
            return f"Error formatting scraped content: {e}"
    
    def _format_article(self, article: Dict[str, Any], index: int) -> str:
        """Format a single article."""
        try:
            parts = []
            
            # Title
            title = article.get('title', '').strip()
            if title:
                parts.append(f"**{index}. {title}**")
            
            # Summary
            summary = article.get('summary', '').strip()
            if summary:
                # Truncate if too long
                if len(summary) > self.max_summary_length:
                    summary = summary[:self.max_summary_length] + "..."
                parts.append(summary)
            
            # Source and link
            source = article.get('source', 'Unknown')
            link = article.get('link', '')
            
            if link and link != 'Unknown':
                parts.append(f"*Source: {source}* | [Read more]({link})")
            else:
                parts.append(f"*Source: {source}*")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"Article formatting failed: {e}")
            return ""
    
    def create_summary_for_ai(self, scraped_data: Dict[str, Any]) -> str:
        """Create a concise summary for AI processing."""
        try:
            if not scraped_data.get('articles'):
                return "No news content available for processing."
            
            articles = scraped_data['articles'][:10]  # Limit for AI processing
            
            summary_parts = []
            summary_parts.append(f"News Summary for: {scraped_data.get('query', 'News Search')}")
            summary_parts.append(f"Articles found: {len(articles)}")
            summary_parts.append("")
            
            for i, article in enumerate(articles, 1):
                title = article.get('title', '').strip()
                summary = article.get('summary', '').strip()
                source = article.get('source', 'Unknown')
                
                if title:
                    article_summary = f"{i}. {title}"
                    if summary:
                        # Limit summary for AI processing
                        short_summary = summary[:200] + "..." if len(summary) > 200 else summary
                        article_summary += f" - {short_summary}"
                    article_summary += f" (Source: {source})"
                    summary_parts.append(article_summary)
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"AI summary creation failed: {e}")
            return f"Error creating AI summary: {e}"
    
    def extract_key_topics(self, scraped_data: Dict[str, Any]) -> List[str]:
        """Extract key topics from scraped articles."""
        try:
            topics = set()
            articles = scraped_data.get('articles', [])
            
            for article in articles:
                title = article.get('title', '').lower()
                summary = article.get('summary', '').lower()
                
                # Simple keyword extraction
                text = f"{title} {summary}"
                words = text.split()
                
                # Extract potential topics (words longer than 4 characters)
                for word in words:
                    clean_word = ''.join(c for c in word if c.isalpha())
                    if len(clean_word) > 4:
                        topics.add(clean_word.title())
            
            # Return top topics
            return sorted(list(topics))[:10]
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return []
    
    def get_content_stats(self, scraped_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about scraped content."""
        try:
            articles = scraped_data.get('articles', [])
            
            stats = {
                'total_articles': len(articles),
                'sources_count': len(scraped_data.get('sources', [])),
                'articles_with_summary': len([a for a in articles if a.get('summary')]),
                'articles_with_links': len([a for a in articles if a.get('link')]),
                'average_title_length': 0,
                'average_summary_length': 0
            }
            
            if articles:
                title_lengths = [len(a.get('title', '')) for a in articles]
                summary_lengths = [len(a.get('summary', '')) for a in articles if a.get('summary')]
                
                stats['average_title_length'] = sum(title_lengths) / len(title_lengths)
                if summary_lengths:
                    stats['average_summary_length'] = sum(summary_lengths) / len(summary_lengths)
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats calculation failed: {e}")
            return {
                'total_articles': 0,
                'sources_count': 0,
                'articles_with_summary': 0,
                'articles_with_links': 0,
                'average_title_length': 0,
                'average_summary_length': 0
            }
