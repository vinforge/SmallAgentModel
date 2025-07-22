"""
NewsApiTool - Current News and Recent Events Retrieval
======================================================

Provides current news and recent events retrieval using NewsAPI and RSS feeds.
Optimized for breaking news, current events, and recent developments.

Author: SAM Development Team
Version: 1.0.0
"""

import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from ..uif import SAM_UIF
from .base import BaseSkillModule, SkillExecutionError
from ..security import get_security_manager, SecurityPolicy

# Import the existing NewsAPITool from web_retrieval
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from web_retrieval.tools.news_api_tool import NewsAPITool as WebNewsAPITool
    NEWS_API_AVAILABLE = True
except ImportError:
    NEWS_API_AVAILABLE = False

logger = logging.getLogger(__name__)


class NewsApiTool(BaseSkillModule):
    """
    Current news and recent events retrieval tool.
    
    Features:
    - Real-time news from 80,000+ sources via NewsAPI
    - RSS feed fallback for reliability
    - Optimized for breaking news and current events
    - Time-sensitive content with recency scoring
    - Integration with SAM's vetting pipeline
    """
    
    skill_name = "NewsApiTool"
    skill_version = "1.0.0"
    skill_description = "Retrieves current news and recent events from news APIs and RSS feeds"
    skill_category = "tools"
    
    # Dependency declarations
    required_inputs = []  # Can extract from input_query if news_query not provided
    optional_inputs = ["news_query", "time_range", "num_articles", "news_category"]
    output_keys = ["news_articles", "news_confidence", "news_source", "article_count"]
    
    # Skill characteristics
    requires_external_access = True
    requires_vetting = True  # News content needs vetting for bias and accuracy
    can_run_parallel = True
    estimated_execution_time = 4.0
    max_execution_time = 20.0
    
    def __init__(self, newsapi_key: Optional[str] = None):
        super().__init__()
        self.newsapi_key = newsapi_key
        self._security_manager = get_security_manager()
        self._setup_security_policy()
        
        # Initialize the underlying NewsAPI tool
        if NEWS_API_AVAILABLE:
            self._news_tool = WebNewsAPITool(api_key=newsapi_key)
        else:
            self._news_tool = None
            self.logger.warning("NewsAPI tool not available, will use fallback methods")
    
    def _setup_security_policy(self) -> None:
        """Set up security policy for news operations."""
        policy = SecurityPolicy(
            allow_network_access=True,
            allow_file_system_access=False,
            max_execution_time=20.0,
            sandbox_enabled=True,
            allowed_commands=[],
            blocked_commands=["rm", "del", "format", "sudo", "ssh"]
        )
        
        self._security_manager.register_tool_policy(self.skill_name, policy)
    
    def execute(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Execute news retrieval with security controls and content vetting.
        
        Args:
            uif: Universal Interface Format with news request
            
        Returns:
            Updated UIF with news results (marked for vetting)
        """
        try:
            # Extract news query
            news_query = self._extract_news_query(uif)
            if not news_query:
                raise SkillExecutionError("No news query found")
            
            self.logger.info(f"Performing news search: {news_query}")
            
            # Get query parameters
            time_range = uif.intermediate_data.get("time_range", "7d")
            num_articles = uif.intermediate_data.get("num_articles", 20)
            news_category = uif.intermediate_data.get("news_category", "general")
            
            # Perform secure news search
            news_results = self._perform_secure_news_search(
                news_query, time_range, num_articles, news_category
            )
            
            # Mark content for vetting (news can contain bias)
            uif.requires_vetting = True
            
            # Store results in UIF (will be vetted before use)
            uif.intermediate_data["news_articles"] = news_results["articles"]
            uif.intermediate_data["news_confidence"] = news_results["confidence"]
            uif.intermediate_data["news_source"] = news_results["source"]
            uif.intermediate_data["article_count"] = len(news_results["articles"])
            
            # Add to security context for vetting
            uif.security_context["external_content"] = {
                "source": "news_search",
                "query": news_query,
                "articles_count": len(news_results["articles"]),
                "requires_vetting": True,
                "content_type": "news"
            }
            
            # Set skill outputs
            uif.set_skill_output(self.skill_name, {
                "query": news_query,
                "articles_found": len(news_results["articles"]),
                "confidence": news_results["confidence"],
                "source": news_results["source"],
                "vetting_required": True
            })
            
            self.logger.info(f"News search completed: {len(news_results['articles'])} articles found")
            
            return uif
            
        except Exception as e:
            self.logger.exception("Error during news search")
            raise SkillExecutionError(f"News search failed: {str(e)}")
    
    def _extract_news_query(self, uif: SAM_UIF) -> Optional[str]:
        """
        Extract news query from UIF or user query.
        
        Returns:
            News query or None if not found
        """
        # Check intermediate data first
        if "news_query" in uif.intermediate_data:
            return uif.intermediate_data["news_query"]
        
        # Try to extract from user query
        query = uif.input_query
        
        # Look for news-related patterns
        news_patterns = [
            r'(?:latest\s+)?news\s+(?:about\s+)?(.+)',
            r'(?:recent\s+)?(?:breaking\s+)?news\s+(?:on\s+)?(.+)',
            r'what\'?s\s+happening\s+(?:with\s+)?(.+)',
            r'current\s+events\s+(?:about\s+)?(.+)',
            r'recent\s+developments\s+(?:in\s+)?(.+)',
            r'updates\s+(?:on\s+)?(.+)',
            r'headlines\s+(?:about\s+)?(.+)',
        ]
        
        query_lower = query.lower()
        for pattern in news_patterns:
            match = re.search(pattern, query_lower)
            if match:
                news_query = match.group(1).strip()
                # Clean up the query
                news_query = self._clean_news_query(news_query)
                return news_query
        
        # If no pattern matches but query contains news keywords, use entire query
        if self._is_news_query(query):
            return self._clean_news_query(query)
        
        return None
    
    def _clean_news_query(self, query: str) -> str:
        """
        Clean and normalize news query.
        
        Returns:
            Cleaned news query
        """
        # Remove common question words and punctuation
        words_to_remove = ['?', '.', '!', 'please', 'can you', 'could you', 'tell me about']
        
        cleaned = query
        for word in words_to_remove:
            cleaned = cleaned.replace(word, ' ')
        
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def _is_news_query(self, query: str) -> bool:
        """
        Check if query appears to be a news request.
        
        Returns:
            True if query appears to be news-related
        """
        news_indicators = [
            'news', 'breaking', 'latest', 'recent', 'current events',
            'headlines', 'updates', 'developments', 'happening',
            'today', 'yesterday', 'this week', 'recently'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in news_indicators)
    
    def _perform_secure_news_search(self, query: str, time_range: str, 
                                  num_articles: int, news_category: str) -> Dict[str, Any]:
        """
        Perform news search with security controls.
        
        Returns:
            Dictionary with news results
        """
        def search_func():
            return self._execute_news_search(query, time_range, num_articles, news_category)
        
        # Execute with security manager
        security_result = self._security_manager.execute_tool_safely(
            self.skill_name,
            search_func
        )
        
        if not security_result.success:
            if security_result.rate_limited:
                raise SkillExecutionError("News search rate limit exceeded")
            elif security_result.security_violations:
                raise SkillExecutionError(f"Security violations: {security_result.security_violations}")
            else:
                raise SkillExecutionError(f"News search failed: {security_result.error_message}")
        
        return security_result.output
    
    def _execute_news_search(self, query: str, time_range: str, 
                           num_articles: int, news_category: str) -> Dict[str, Any]:
        """
        Execute news search using NewsAPI or fallback methods.
        
        Returns:
            Dictionary with news articles and metadata
        """
        try:
            if self._news_tool:
                # Use the existing NewsAPI tool
                result = self._news_tool.get_news(query, num_articles)
                
                if result['success']:
                    # Process articles for SAM's format
                    processed_articles = self._process_news_articles(result['articles'], query)
                    
                    return {
                        "articles": processed_articles,
                        "confidence": self._calculate_news_confidence(result, query),
                        "source": result.get('source', 'newsapi'),
                        "query": query,
                        "total_found": result.get('total_results', len(processed_articles))
                    }
                else:
                    # NewsAPI failed, use fallback
                    return self._fallback_news_search(query, num_articles)
            else:
                # No NewsAPI available, use fallback
                return self._fallback_news_search(query, num_articles)
                
        except Exception as e:
            self.logger.error(f"News search execution failed: {e}")
            return {
                "articles": [],
                "confidence": 0.0,
                "source": "error",
                "query": query,
                "error": str(e)
            }
    
    def _process_news_articles(self, articles: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Process news articles for SAM's format.
        
        Returns:
            List of processed articles
        """
        processed = []
        
        for article in articles:
            # Calculate recency score
            recency_score = self._calculate_recency_score(article.get('published_at', ''))
            
            # Calculate relevance score
            relevance_score = self._calculate_article_relevance(article, query)
            
            processed_article = {
                "title": article.get('title', ''),
                "description": article.get('description', ''),
                "content": article.get('content_preview', ''),
                "url": article.get('url', ''),
                "source": article.get('source', 'Unknown'),
                "published_at": article.get('published_at', ''),
                "author": article.get('author', ''),
                "image_url": article.get('url_to_image', ''),
                "recency_score": recency_score,
                "relevance_score": relevance_score,
                "combined_score": (recency_score + relevance_score) / 2,
                "tool_source": "news_api_tool",
                "timestamp": datetime.now().isoformat()
            }
            
            processed.append(processed_article)
        
        # Sort by combined score (recency + relevance)
        processed.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return processed
    
    def _calculate_recency_score(self, published_at: str) -> float:
        """
        Calculate recency score for an article.
        
        Returns:
            Recency score between 0.0 and 1.0
        """
        if not published_at:
            return 0.5  # Default score for unknown dates
        
        try:
            # Parse the published date
            pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            now = datetime.now(pub_date.tzinfo)
            
            # Calculate hours since publication
            hours_ago = (now - pub_date).total_seconds() / 3600
            
            # Score decreases with age (1.0 for very recent, 0.0 for very old)
            if hours_ago <= 1:
                return 1.0
            elif hours_ago <= 24:
                return 0.8
            elif hours_ago <= 168:  # 1 week
                return 0.6
            elif hours_ago <= 720:  # 1 month
                return 0.4
            else:
                return 0.2
                
        except Exception:
            return 0.5  # Default score for parsing errors
    
    def _calculate_article_relevance(self, article: Dict[str, Any], query: str) -> float:
        """
        Calculate relevance score for an article.
        
        Returns:
            Relevance score between 0.0 and 1.0
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = f"{title} {description}"
        
        query_words = query.lower().split()
        
        # Count matching words
        matches = sum(1 for word in query_words if word in content)
        base_score = matches / len(query_words) if query_words else 0.0
        
        # Boost for title matches
        title_matches = sum(1 for word in query_words if word in title)
        title_boost = (title_matches / len(query_words)) * 0.3 if query_words else 0.0
        
        return min(1.0, base_score + title_boost)
    
    def _calculate_news_confidence(self, result: Dict[str, Any], query: str) -> float:
        """
        Calculate overall confidence in news results.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        articles = result.get('articles', [])
        
        if not articles:
            return 0.0
        
        # Base confidence from number of articles
        base_confidence = min(0.8, 0.3 + (len(articles) * 0.05))
        
        # Boost for successful API response
        if result.get('success', False):
            base_confidence += 0.1
        
        # Boost for recent articles
        recent_articles = sum(1 for article in articles 
                            if self._calculate_recency_score(article.get('published_at', '')) > 0.6)
        recency_boost = (recent_articles / len(articles)) * 0.1
        
        return min(0.95, base_confidence + recency_boost)
    
    def _fallback_news_search(self, query: str, num_articles: int) -> Dict[str, Any]:
        """
        Fallback news search when NewsAPI is not available.
        
        Returns:
            Dictionary with fallback news results
        """
        # This would implement RSS-based news search or other fallback methods
        # For now, return a placeholder
        fallback_articles = [{
            "title": f"News about {query} (Fallback)",
            "description": f"Fallback news search results for {query}. NewsAPI not available.",
            "content": f"This is a fallback result for news about {query}.",
            "url": "https://example.com/news",
            "source": "Fallback",
            "published_at": datetime.now().isoformat(),
            "recency_score": 0.5,
            "relevance_score": 0.5,
            "combined_score": 0.5,
            "tool_source": "fallback"
        }]
        
        return {
            "articles": fallback_articles,
            "confidence": 0.3,  # Lower confidence for fallback
            "source": "fallback",
            "query": query,
            "total_found": len(fallback_articles)
        }
    
    def can_handle_query(self, query: str) -> bool:
        """
        Check if this tool can handle the given query.
        
        Args:
            query: User query to check
            
        Returns:
            True if query appears to be a news request
        """
        return self._is_news_query(query)
