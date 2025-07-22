"""
Web Search & External Knowledge Access for SAM
Real-time web search with security controls and result summarization.

Sprint 9 Task 3: Web Search & External Knowledge Access
"""

import logging
import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
import urllib.parse

logger = logging.getLogger(__name__)

class SearchEngine(Enum):
    """Supported search engines."""
    DUCKDUCKGO = "duckduckgo"
    BING = "bing"
    GOOGLE = "google"

class SearchResultType(Enum):
    """Types of web search results."""
    WEB_PAGE = "web_page"
    NEWS = "news"
    IMAGE = "image"
    VIDEO = "video"

@dataclass
class WebSearchResult:
    """A single web search result."""
    result_id: str
    title: str
    url: str
    snippet: str
    result_type: SearchResultType
    confidence_score: float
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class SearchQuery:
    """A web search query with metadata."""
    query_id: str
    query_text: str
    search_engine: SearchEngine
    user_id: str
    session_id: str
    timestamp: str
    results_count: int
    processing_time_ms: int
    metadata: Dict[str, Any]

@dataclass
class SearchSummary:
    """Summary of web search results."""
    summary_id: str
    query_id: str
    query_text: str
    summary_text: str
    key_insights: List[str]
    source_urls: List[str]
    confidence_score: float
    created_at: str
    metadata: Dict[str, Any]

class WebSearchEngine:
    """
    Web search engine with security controls and result summarization.
    """
    
    def __init__(self, search_logs_file: str = "web_search_logs.json",
                 enable_web_access: bool = False):
        """
        Initialize the web search engine.
        
        Args:
            search_logs_file: Path to search logs storage file
            enable_web_access: Whether web access is enabled
        """
        self.search_logs_file = Path(search_logs_file)
        self.enable_web_access = enable_web_access
        
        # Storage
        self.search_queries: Dict[str, SearchQuery] = {}
        self.search_summaries: Dict[str, SearchSummary] = {}
        
        # Configuration
        self.config = {
            'max_results': 10,
            'timeout_seconds': 30,
            'user_agent': 'SAM-Assistant/1.0',
            'rate_limit_delay': 1.0,  # seconds between requests
            'allowed_domains': [],  # Empty means all domains allowed
            'blocked_domains': ['malware.com', 'spam.com'],  # Example blocked domains
            'log_all_queries': True,
            'enable_summarization': True
        }
        
        # Load existing logs
        self._load_search_logs()
        
        logger.info(f"Web search engine initialized (web access: {enable_web_access})")
    
    def search(self, query: str, user_id: str, session_id: str,
              search_engine: SearchEngine = SearchEngine.DUCKDUCKGO,
              max_results: int = 5) -> Tuple[List[WebSearchResult], str]:
        """
        Perform a web search.
        
        Args:
            query: Search query
            user_id: User performing the search
            session_id: Session ID
            search_engine: Search engine to use
            max_results: Maximum number of results
            
        Returns:
            Tuple of (search results, query ID)
        """
        try:
            if not self.enable_web_access:
                logger.warning("Web access is disabled")
                return [], ""
            
            query_id = f"query_{uuid.uuid4().hex[:12]}"
            start_time = datetime.now()
            
            # Validate and sanitize query
            sanitized_query = self._sanitize_query(query)
            if not sanitized_query:
                raise ValueError("Invalid or empty query")
            
            logger.info(f"Performing web search: {sanitized_query} (engine: {search_engine.value})")
            
            # Perform search based on engine
            if search_engine == SearchEngine.DUCKDUCKGO:
                results = self._search_duckduckgo(sanitized_query, max_results)
            elif search_engine == SearchEngine.BING:
                results = self._search_bing(sanitized_query, max_results)
            elif search_engine == SearchEngine.GOOGLE:
                results = self._search_google(sanitized_query, max_results)
            else:
                raise ValueError(f"Unsupported search engine: {search_engine}")
            
            # Filter results by domain restrictions
            filtered_results = self._filter_results(results)
            
            # Calculate processing time
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create search query record
            search_query = SearchQuery(
                query_id=query_id,
                query_text=sanitized_query,
                search_engine=search_engine,
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                results_count=len(filtered_results),
                processing_time_ms=processing_time_ms,
                metadata={'original_query': query}
            )
            
            self.search_queries[query_id] = search_query
            
            # Log the search
            if self.config['log_all_queries']:
                self._log_search_query(search_query, filtered_results)
            
            logger.info(f"Web search completed: {len(filtered_results)} results in {processing_time_ms}ms")
            return filtered_results, query_id
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return [], ""
    
    def summarize_results(self, query_id: str, results: List[WebSearchResult]) -> Optional[SearchSummary]:
        """
        Summarize web search results.
        
        Args:
            query_id: Query ID
            results: Search results to summarize
            
        Returns:
            Search summary or None if failed
        """
        try:
            if not self.config['enable_summarization'] or not results:
                return None
            
            query = self.search_queries.get(query_id)
            if not query:
                logger.error(f"Query not found: {query_id}")
                return None
            
            summary_id = f"summary_{uuid.uuid4().hex[:12]}"
            
            # Extract key information from results
            all_snippets = [result.snippet for result in results if result.snippet]
            source_urls = [result.url for result in results]
            
            # Generate summary (simplified implementation)
            summary_text = self._generate_summary(query.query_text, all_snippets)
            key_insights = self._extract_key_insights(all_snippets)
            
            # Calculate confidence based on result quality
            confidence_score = self._calculate_summary_confidence(results)
            
            # Create search summary
            search_summary = SearchSummary(
                summary_id=summary_id,
                query_id=query_id,
                query_text=query.query_text,
                summary_text=summary_text,
                key_insights=key_insights,
                source_urls=source_urls,
                confidence_score=confidence_score,
                created_at=datetime.now().isoformat(),
                metadata={'results_count': len(results)}
            )
            
            self.search_summaries[summary_id] = search_summary
            
            logger.info(f"Generated search summary: {summary_id}")
            return search_summary
            
        except Exception as e:
            logger.error(f"Error summarizing search results: {e}")
            return None
    
    def get_search_history(self, user_id: Optional[str] = None,
                          days_back: int = 7) -> List[SearchQuery]:
        """
        Get search history for a user.
        
        Args:
            user_id: Optional user ID filter
            days_back: Number of days to look back
            
        Returns:
            List of search queries
        """
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Filter queries
            filtered_queries = []
            for query in self.search_queries.values():
                query_date = datetime.fromisoformat(query.timestamp)
                
                if query_date >= cutoff_date:
                    if user_id is None or query.user_id == user_id:
                        filtered_queries.append(query)
            
            # Sort by timestamp (newest first)
            filtered_queries.sort(key=lambda q: q.timestamp, reverse=True)
            
            return filtered_queries
            
        except Exception as e:
            logger.error(f"Error getting search history: {e}")
            return []
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize search query for security."""
        try:
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[<>"\']', '', query.strip())
            
            # Limit length
            if len(sanitized) > 200:
                sanitized = sanitized[:200]
            
            # Remove excessive whitespace
            sanitized = re.sub(r'\s+', ' ', sanitized)
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing query: {e}")
            return ""
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Search using DuckDuckGo (simulated implementation)."""
        try:
            # This would integrate with actual DuckDuckGo API or web scraping
            # For now, return simulated results
            
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            # Simulate search results
            results = []
            for i in range(min(max_results, 5)):
                result = WebSearchResult(
                    result_id=f"ddg_{uuid.uuid4().hex[:8]}",
                    title=f"Search Result {i+1} for '{query}'",
                    url=f"https://example{i+1}.com/search-result",
                    snippet=f"This is a simulated search result snippet for '{query}'. It contains relevant information about the search topic.",
                    result_type=SearchResultType.WEB_PAGE,
                    confidence_score=0.8 - (i * 0.1),
                    timestamp=datetime.now().isoformat(),
                    metadata={'search_engine': 'duckduckgo', 'rank': i+1}
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {e}")
            return []
    
    def _search_bing(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Search using Bing API (simulated implementation)."""
        try:
            # This would integrate with actual Bing Search API
            # For now, return simulated results
            
            logger.info(f"Searching Bing for: {query}")
            
            # Simulate search results
            results = []
            for i in range(min(max_results, 5)):
                result = WebSearchResult(
                    result_id=f"bing_{uuid.uuid4().hex[:8]}",
                    title=f"Bing Result {i+1}: {query}",
                    url=f"https://bing-result{i+1}.com/page",
                    snippet=f"Bing search result snippet for '{query}'. This provides comprehensive information about the search topic with high relevance.",
                    result_type=SearchResultType.WEB_PAGE,
                    confidence_score=0.85 - (i * 0.1),
                    timestamp=datetime.now().isoformat(),
                    metadata={'search_engine': 'bing', 'rank': i+1}
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Bing: {e}")
            return []
    
    def _search_google(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Search using Google Custom Search API (simulated implementation)."""
        try:
            # This would integrate with actual Google Custom Search API
            # For now, return simulated results
            
            logger.info(f"Searching Google for: {query}")
            
            # Simulate search results
            results = []
            for i in range(min(max_results, 5)):
                result = WebSearchResult(
                    result_id=f"google_{uuid.uuid4().hex[:8]}",
                    title=f"Google Result {i+1} - {query}",
                    url=f"https://google-result{i+1}.com/content",
                    snippet=f"Google search result for '{query}'. This snippet contains detailed and relevant information about the search query topic.",
                    result_type=SearchResultType.WEB_PAGE,
                    confidence_score=0.9 - (i * 0.1),
                    timestamp=datetime.now().isoformat(),
                    metadata={'search_engine': 'google', 'rank': i+1}
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Google: {e}")
            return []
    
    def _filter_results(self, results: List[WebSearchResult]) -> List[WebSearchResult]:
        """Filter search results based on domain restrictions."""
        try:
            filtered_results = []
            
            for result in results:
                url_domain = self._extract_domain(result.url)
                
                # Check blocked domains
                if url_domain in self.config['blocked_domains']:
                    logger.debug(f"Blocked result from domain: {url_domain}")
                    continue
                
                # Check allowed domains (if specified)
                if self.config['allowed_domains'] and url_domain not in self.config['allowed_domains']:
                    logger.debug(f"Domain not in allowed list: {url_domain}")
                    continue
                
                filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error filtering results: {e}")
            return results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc.lower()
        except:
            return ""
    
    def _generate_summary(self, query: str, snippets: List[str]) -> str:
        """Generate summary from search result snippets."""
        try:
            if not snippets:
                return f"No detailed information found for '{query}'."
            
            # Simple summary generation (would use more sophisticated NLP in practice)
            combined_text = ' '.join(snippets)
            
            # Extract key sentences
            sentences = re.split(r'[.!?]+', combined_text)
            relevant_sentences = []
            
            query_words = set(query.lower().split())
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Minimum sentence length
                    sentence_words = set(sentence.lower().split())
                    # Check if sentence contains query words
                    if query_words.intersection(sentence_words):
                        relevant_sentences.append(sentence)
            
            # Take top 3 most relevant sentences
            summary_sentences = relevant_sentences[:3]
            
            if summary_sentences:
                summary = '. '.join(summary_sentences) + '.'
            else:
                summary = f"Search results for '{query}' provide various information on the topic."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Summary generation failed for query: {query}"
    
    def _extract_key_insights(self, snippets: List[str]) -> List[str]:
        """Extract key insights from search result snippets."""
        try:
            insights = []
            
            # Simple insight extraction (would use more sophisticated analysis in practice)
            combined_text = ' '.join(snippets).lower()
            
            # Look for common insight patterns
            insight_patterns = [
                r'according to ([^,]+)',
                r'research shows ([^.]+)',
                r'studies indicate ([^.]+)',
                r'experts believe ([^.]+)',
                r'data suggests ([^.]+)'
            ]
            
            for pattern in insight_patterns:
                matches = re.findall(pattern, combined_text)
                for match in matches:
                    if len(match.strip()) > 10:
                        insights.append(match.strip().capitalize())
            
            # Limit to top 5 insights
            return insights[:5]
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return []
    
    def _calculate_summary_confidence(self, results: List[WebSearchResult]) -> float:
        """Calculate confidence score for summary."""
        try:
            if not results:
                return 0.0
            
            # Average confidence of individual results
            avg_confidence = sum(result.confidence_score for result in results) / len(results)
            
            # Bonus for multiple sources
            source_bonus = min(0.2, len(results) * 0.05)
            
            # Penalty for very few results
            if len(results) < 3:
                avg_confidence *= 0.8
            
            return min(1.0, avg_confidence + source_bonus)
            
        except Exception as e:
            logger.error(f"Error calculating summary confidence: {e}")
            return 0.5
    
    def _log_search_query(self, query: SearchQuery, results: List[WebSearchResult]):
        """Log search query and results."""
        try:
            log_entry = {
                'query_id': query.query_id,
                'query_text': query.query_text,
                'search_engine': query.search_engine.value,
                'user_id': query.user_id,
                'session_id': query.session_id,
                'timestamp': query.timestamp,
                'results_count': len(results),
                'processing_time_ms': query.processing_time_ms,
                'result_urls': [result.url for result in results],
                'metadata': query.metadata
            }
            
            # Append to log file
            with open(self.search_logs_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            logger.debug(f"Logged search query: {query.query_id}")
            
        except Exception as e:
            logger.error(f"Error logging search query: {e}")
    
    def _load_search_logs(self):
        """Load recent search logs."""
        try:
            if self.search_logs_file.exists():
                # Load recent queries (last 1000 lines)
                with open(self.search_logs_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Process recent lines
                for line in lines[-1000:]:
                    try:
                        log_entry = json.loads(line.strip())
                        
                        search_query = SearchQuery(
                            query_id=log_entry['query_id'],
                            query_text=log_entry['query_text'],
                            search_engine=SearchEngine(log_entry['search_engine']),
                            user_id=log_entry['user_id'],
                            session_id=log_entry['session_id'],
                            timestamp=log_entry['timestamp'],
                            results_count=log_entry['results_count'],
                            processing_time_ms=log_entry['processing_time_ms'],
                            metadata=log_entry.get('metadata', {})
                        )
                        
                        self.search_queries[search_query.query_id] = search_query
                        
                    except Exception as e:
                        logger.warning(f"Error parsing log entry: {e}")
                
                logger.info(f"Loaded {len(self.search_queries)} search queries from logs")
            
        except Exception as e:
            logger.error(f"Error loading search logs: {e}")

# Global web search engine instance
_web_search_engine = None

def get_web_search_engine(search_logs_file: str = "web_search_logs.json",
                         enable_web_access: bool = False) -> WebSearchEngine:
    """Get or create a global web search engine instance."""
    global _web_search_engine
    
    if _web_search_engine is None:
        _web_search_engine = WebSearchEngine(
            search_logs_file=search_logs_file,
            enable_web_access=enable_web_access
        )
    
    return _web_search_engine
