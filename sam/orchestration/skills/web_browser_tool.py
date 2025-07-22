"""
AgentZeroWebBrowserTool - Web Browsing and Search Skill
=======================================================

Provides secure web browsing and search capabilities with content vetting.
Integrates with SAM's security framework for safe external content access.
"""

import re
import time
import logging
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from ..uif import SAM_UIF
from .base import BaseSkillModule, SkillExecutionError
from ..security import get_security_manager, SecurityPolicy

logger = logging.getLogger(__name__)


class AgentZeroWebBrowserTool(BaseSkillModule):
    """
    Secure web browsing tool for information retrieval.
    
    Features:
    - Web page content extraction
    - Search query processing
    - Content sanitization and vetting
    - Rate limiting and security controls
    - Integration with SAM's vetting pipeline
    """
    
    skill_name = "AgentZeroWebBrowserTool"
    skill_version = "1.0.0"
    skill_description = "Performs secure web browsing and content retrieval"
    skill_category = "tools"
    
    # Dependency declarations
    required_inputs = ["web_search_query"]
    optional_inputs = ["search_context", "max_results", "content_filter"]
    output_keys = ["web_search_results", "extracted_content", "search_confidence"]
    
    # Skill characteristics
    requires_external_access = True
    requires_vetting = True  # Web content needs vetting
    can_run_parallel = True
    estimated_execution_time = 5.0
    max_execution_time = 30.0
    
    def __init__(self):
        super().__init__()
        self._security_manager = get_security_manager()
        self._setup_security_policy()
        self._session = requests.Session()
        self._setup_session()
    
    def _setup_security_policy(self) -> None:
        """Set up security policy for web browsing operations."""
        policy = SecurityPolicy(
            allow_network_access=True,
            allow_file_system_access=False,
            max_execution_time=30.0,
            sandbox_enabled=True,
            allowed_commands=["curl", "wget"],
            blocked_commands=["rm", "del", "format", "sudo", "ssh"]
        )
        
        self._security_manager.register_tool_policy(self.skill_name, policy)
    
    def _setup_session(self) -> None:
        """Set up HTTP session with security headers."""
        self._session.headers.update({
            'User-Agent': 'SAM-Agent/1.0 (Secure AI Assistant)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Set timeouts
        self._session.timeout = 10
    
    def execute(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Execute web browsing with security controls and content vetting.

        Args:
            uif: Universal Interface Format with search request

        Returns:
            Updated UIF with search results (marked for vetting)
        """
        # Initialize tracing
        trace_id = uif.intermediate_data.get('trace_id')
        start_time = time.time()

        if trace_id:
            self._log_trace_event(
                trace_id=trace_id,
                event_type="start",
                severity="info",
                message="Starting web search and content extraction",
                payload={
                    "tool": self.skill_name,
                    "input_query": uif.input_query,
                    "requires_external_access": self.requires_external_access,
                    "requires_vetting": self.requires_vetting
                }
            )

        try:
            # Extract search query
            search_query = self._extract_search_query(uif)
            if not search_query:
                if trace_id:
                    self._log_trace_event(
                        trace_id=trace_id,
                        event_type="error",
                        severity="error",
                        message="No web search query found",
                        payload={"input_query": uif.input_query}
                    )
                raise SkillExecutionError("No web search query found")

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="data_in",
                    severity="info",
                    message=f"Extracted web search query: {search_query}",
                    payload={
                        "search_query": search_query,
                        "query_length": len(search_query),
                        "is_question": any(q in search_query.lower() for q in ['what', 'how', 'why', 'when', 'where', 'who'])
                    }
                )

            self.logger.info(f"Performing web search: {search_query}")

            # Get search parameters
            search_context = uif.intermediate_data.get("search_context", {})
            max_results = uif.intermediate_data.get("max_results", 5)
            content_filter = uif.intermediate_data.get("content_filter", "safe")

            # Perform secure web search
            search_start_time = time.time()
            search_results = self._perform_secure_search(
                search_query, search_context, max_results, content_filter
            )
            search_duration = (time.time() - search_start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="tool_call",
                    severity="info",
                    message=f"Web search completed: {len(search_results['results'])} results found",
                    duration_ms=search_duration,
                    payload={
                        "search_query": search_query,
                        "results_count": len(search_results["results"]),
                        "content_count": len(search_results["content"]),
                        "confidence": search_results["confidence"],
                        "search_duration_ms": search_duration,
                        "max_results_requested": max_results
                    }
                )

            # Mark content for vetting
            uif.requires_vetting = True

            # Store results in UIF (will be vetted before use)
            uif.intermediate_data["web_search_results"] = search_results["results"]
            uif.intermediate_data["extracted_content"] = search_results["content"]
            uif.intermediate_data["search_confidence"] = search_results["confidence"]

            # Add to security context for vetting
            uif.security_context["external_content"] = {
                "source": "web_search",
                "query": search_query,
                "results_count": len(search_results["results"]),
                "requires_vetting": True
            }

            # Set skill outputs
            uif.set_skill_output(self.skill_name, {
                "search_query": search_query,
                "results_count": len(search_results["results"]),
                "confidence": search_results["confidence"],
                "vetting_required": True
            })

            total_duration = (time.time() - start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="data_out",
                    severity="info",
                    message=f"Web browser tool execution completed - content marked for vetting",
                    duration_ms=total_duration,
                    payload={
                        "results_count": len(search_results["results"]),
                        "content_extracted": len(search_results["content"]),
                        "confidence": search_results["confidence"],
                        "requires_vetting": True,
                        "execution_time_ms": total_duration,
                        "search_time_ms": search_duration,
                        "overhead_ms": total_duration - search_duration
                    }
                )

            self.logger.info(f"Web search completed: {len(search_results['results'])} results found")

            return uif

        except Exception as e:
            total_duration = (time.time() - start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="error",
                    severity="error",
                    message=f"Web browser tool execution failed: {str(e)}",
                    duration_ms=total_duration,
                    payload={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "execution_time_ms": total_duration
                    }
                )

            self.logger.exception("Error during web search")
            raise SkillExecutionError(f"Web search failed: {str(e)}")
    
    def _extract_search_query(self, uif: SAM_UIF) -> Optional[str]:
        """
        Extract search query from UIF or user query.
        
        Returns:
            Search query or None if not found
        """
        # Check intermediate data first
        if "web_search_query" in uif.intermediate_data:
            return uif.intermediate_data["web_search_query"]
        
        # Try to extract from user query
        query = uif.input_query
        
        # Look for search-related patterns (enhanced for better detection)
        search_patterns = [
            r'search\s+(?:for\s+|up\s+|about\s+)?(.+)',
            r'find\s+(?:information\s+(?:about\s+)?)?(.+)',
            r'look\s+up\s+(.+)',
            r'browse\s+(?:for\s+)?(.+)',
            r'what\s+(?:is|are)\s+(.+)',
            r'tell\s+me\s+about\s+(.+)',
            # Enhanced patterns to catch queries like "search up, information about, why do clouds form?"
            r'search\s+up,?\s*(?:information\s+about,?\s*)?(.+)',
            r'information\s+about,?\s*(.+)',
            r'learn\s+about\s+(.+)',
            r'research\s+(.+)',
            r'investigate\s+(.+)',
            r'discover\s+(.+)',
            r'explore\s+(.+)',
            r'understand\s+(.+)',
            r'explain\s+(.+)',
            r'clarify\s+(.+)',
            r'describe\s+(.+)',
            r'details\s+about\s+(.+)',
        ]
        
        query_lower = query.lower()
        for pattern in search_patterns:
            match = re.search(pattern, query_lower)
            if match:
                search_query = match.group(1).strip()
                # Clean up the query
                search_query = self._clean_search_query(search_query)
                return search_query
        
        # If no pattern matches, use the entire query if it seems like a search
        if self._is_search_query(query):
            return self._clean_search_query(query)
        
        return None
    
    def _clean_search_query(self, query: str) -> str:
        """
        Clean and normalize search query.
        
        Returns:
            Cleaned search query
        """
        # Remove common question words and punctuation
        words_to_remove = ['?', '.', '!', 'please', 'can you', 'could you']
        
        cleaned = query
        for word in words_to_remove:
            cleaned = cleaned.replace(word, ' ')
        
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def _is_search_query(self, query: str) -> bool:
        """
        Check if query appears to be a search request.
        
        Returns:
            True if query appears to be a search
        """
        search_indicators = [
            'search', 'find', 'look up', 'browse', 'what is', 'what are',
            'tell me about', 'information about', 'details about',
            # Enhanced search detection (preserving 100% of functionality)
            'search up', 'search for', 'search about', 'look for',
            'find out', 'find information', 'learn about', 'research',
            'investigate', 'discover', 'explore', 'understand', 'clarify'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in search_indicators)
    
    def _perform_secure_search(self, query: str, context: Dict[str, Any], 
                             max_results: int, content_filter: str) -> Dict[str, Any]:
        """
        Perform web search with security controls.
        
        Returns:
            Dictionary with search results
        """
        def search_func():
            return self._execute_web_search(query, context, max_results, content_filter)
        
        # Execute with security manager
        security_result = self._security_manager.execute_tool_safely(
            self.skill_name,
            search_func
        )
        
        if not security_result.success:
            if security_result.rate_limited:
                raise SkillExecutionError("Web search rate limit exceeded")
            elif security_result.security_violations:
                raise SkillExecutionError(f"Security violations: {security_result.security_violations}")
            else:
                raise SkillExecutionError(f"Web search failed: {security_result.error_message}")
        
        return security_result.output
    
    def _execute_web_search(self, query: str, context: Dict[str, Any], 
                          max_results: int, content_filter: str) -> Dict[str, Any]:
        """
        Execute web search and extract content.
        
        Returns:
            Dictionary with search results and extracted content
        """
        results = []
        extracted_content = []
        confidence = 0.7
        
        try:
            # For demonstration, we'll use a simple search approach
            # In production, this would integrate with search APIs like Google, Bing, etc.
            
            # Simulate search results (in production, use actual search API)
            search_results = self._simulate_search_results(query, max_results)
            
            for result in search_results:
                try:
                    # Extract content from each URL
                    content = self._extract_page_content(result["url"])
                    
                    if content:
                        results.append({
                            "title": result["title"],
                            "url": result["url"],
                            "snippet": result["snippet"],
                            "content_length": len(content),
                            "extracted_at": "2024-01-01T00:00:00Z"  # Would be actual timestamp
                        })
                        
                        extracted_content.append({
                            "url": result["url"],
                            "title": result["title"],
                            "content": content[:2000],  # Limit content length
                            "full_content_available": len(content) > 2000
                        })
                
                except Exception as e:
                    self.logger.warning(f"Failed to extract content from {result['url']}: {e}")
                    continue
            
            # Calculate confidence based on results quality
            if results:
                confidence = min(0.9, 0.5 + (len(results) * 0.1))
            
            return {
                "results": results,
                "content": extracted_content,
                "confidence": confidence,
                "query": query,
                "total_found": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Web search execution failed: {e}")
            return {
                "results": [],
                "content": [],
                "confidence": 0.0,
                "query": query,
                "error": str(e)
            }
    
    def _simulate_search_results(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """
        Simulate search results for demonstration.
        In production, this would call actual search APIs.
        
        Returns:
            List of simulated search results
        """
        # This is a placeholder - in production, integrate with:
        # - Google Custom Search API
        # - Bing Search API
        # - DuckDuckGo API
        # - Other search providers
        
        simulated_results = [
            {
                "title": f"Information about {query} - Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                "snippet": f"Comprehensive information about {query} from Wikipedia..."
            },
            {
                "title": f"{query} - Official Website",
                "url": f"https://example.com/{query.replace(' ', '-')}",
                "snippet": f"Official information and resources about {query}..."
            },
            {
                "title": f"Latest news about {query}",
                "url": f"https://news.example.com/{query.replace(' ', '-')}",
                "snippet": f"Recent news and updates related to {query}..."
            }
        ]
        
        return simulated_results[:max_results]
    
    def _extract_page_content(self, url: str) -> Optional[str]:
        """
        Extract text content from a web page.
        
        Returns:
            Extracted text content or None if extraction fails
        """
        try:
            # Validate URL
            if not self._is_safe_url(url):
                self.logger.warning(f"Unsafe URL blocked: {url}")
                return None
            
            # Fetch page content
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in str(line).split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Failed to extract content from {url}: {e}")
            return None
    
    def _is_safe_url(self, url: str) -> bool:
        """
        Check if URL is safe to access.
        
        Returns:
            True if URL is considered safe
        """
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Check for blocked domains
            blocked_domains = [
                'localhost', '127.0.0.1', '0.0.0.0',
                '192.168.', '10.', '172.16.', '172.17.', '172.18.',
                '172.19.', '172.20.', '172.21.', '172.22.', '172.23.',
                '172.24.', '172.25.', '172.26.', '172.27.', '172.28.',
                '172.29.', '172.30.', '172.31.'
            ]
            
            hostname = parsed.hostname or ''
            for blocked in blocked_domains:
                if hostname.startswith(blocked):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def can_handle_query(self, query: str) -> bool:
        """
        Check if this tool can handle the given query.
        
        Args:
            query: User query to check
            
        Returns:
            True if query appears to be a web search request
        """
        search_keywords = [
            'search', 'find', 'look up', 'browse', 'google', 'web',
            'internet', 'online', 'website', 'url', 'link'
        ]
        
        query_lower = query.lower()
        
        # Check for search keywords
        if any(keyword in query_lower for keyword in search_keywords):
            return True
        
        # Check for question patterns that might need web search
        question_patterns = [
            'what is', 'what are', 'who is', 'who are', 'where is', 'where are',
            'when is', 'when are', 'how is', 'how are', 'why is', 'why are'
        ]
        
        if any(pattern in query_lower for pattern in question_patterns):
            return True
        
        return False

    def _log_trace_event(self, trace_id: str, event_type: str, severity: str,
                        message: str, duration_ms: Optional[float] = None,
                        payload: Optional[Dict[str, Any]] = None) -> None:
        """Log a trace event for the web browser tool."""
        try:
            from sam.cognition.trace_logger import log_event
            log_event(
                trace_id=trace_id,
                source_module=self.skill_name,
                event_type=event_type,
                severity=severity,
                message=message,
                duration_ms=duration_ms,
                payload=payload or {},
                metadata={
                    "tool_version": self.skill_version,
                    "tool_category": self.skill_category,
                    "requires_external_access": self.requires_external_access,
                    "requires_vetting": self.requires_vetting
                }
            )
        except ImportError:
            # Tracing not available, continue without logging
            pass
        except Exception as e:
            # Don't let tracing errors break the tool
            self.logger.debug(f"Trace logging failed: {e}")
