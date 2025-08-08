"""
FinancialDataTool - Specialized Financial and Factual Data Retrieval
===================================================================

Provides specialized financial data retrieval using Serper API for direct,
factual lookups. Optimized for specific data points like stock prices,
market capitalization, and financial metrics.

Author: SAM Development Team
Version: 1.0.0
"""

import re
import time
import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..uif import SAM_UIF
from .base import BaseSkillModule, SkillExecutionError
from ..security import get_security_manager, SecurityPolicy

logger = logging.getLogger(__name__)


class FinancialDataTool(BaseSkillModule):
    """
    Specialized tool for financial and factual data retrieval.
    
    Features:
    - Direct financial data lookup using Serper API
    - Optimized for specific data points (market cap, stock prices, etc.)
    - Structured data extraction from search results
    - High confidence scoring for factual data
    - Integration with SAM's security framework
    """
    
    skill_name = "FinancialDataTool"
    skill_version = "1.0.0"
    skill_description = "Retrieves specific financial data and factual information using optimized search"
    skill_category = "tools"
    
    # Dependency declarations
    required_inputs = []  # Can extract from input_query if financial_query not provided
    optional_inputs = ["financial_query", "data_type", "company_symbol", "specific_metric"]
    output_keys = ["financial_data", "data_confidence", "data_source", "extracted_value"]
    
    # Skill characteristics
    requires_external_access = True
    requires_vetting = False  # Financial data is typically factual and doesn't need content vetting
    can_run_parallel = True
    estimated_execution_time = 3.0
    max_execution_time = 15.0
    
    def __init__(self, serper_api_key: Optional[str] = None):
        super().__init__()
        self.serper_api_key = serper_api_key
        self._security_manager = get_security_manager()
        self._setup_security_policy()
        self._session = requests.Session()
        self._setup_session()
    
    def _setup_security_policy(self) -> None:
        """Set up security policy for financial data operations."""
        policy = SecurityPolicy(
            allow_network_access=True,
            allow_file_system_access=False,
            max_execution_time=15.0,
            sandbox_enabled=True,
            allowed_commands=[],
            blocked_commands=["rm", "del", "format", "sudo", "ssh"]
        )
        
        self._security_manager.register_tool_policy(self.skill_name, policy)
    
    def _setup_session(self) -> None:
        """Set up HTTP session with security headers."""
        self._session.headers.update({
            'User-Agent': 'SAM-FinancialDataTool/1.0 (Financial Data Assistant)',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        })
        
        # Set timeouts
        self._session.timeout = 10
    
    def execute(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Execute financial data retrieval with security controls.

        Args:
            uif: Universal Interface Format with financial data request

        Returns:
            Updated UIF with financial data results
        """
        # Initialize tracing
        trace_id = uif.intermediate_data.get('trace_id')
        start_time = time.time()

        if trace_id:
            self._log_trace_event(
                trace_id=trace_id,
                event_type="start",
                severity="info",
                message="Starting financial data retrieval",
                payload={
                    "tool": self.skill_name,
                    "input_query": uif.input_query,
                    "has_serper_api": bool(self.serper_api_key),
                    "requires_external_access": self.requires_external_access
                }
            )

        try:
            # Extract financial query
            financial_query = self._extract_financial_query(uif)
            if not financial_query:
                if trace_id:
                    self._log_trace_event(
                        trace_id=trace_id,
                        event_type="error",
                        severity="error",
                        message="No financial query found",
                        payload={"input_query": uif.input_query}
                    )
                raise SkillExecutionError("No financial query found")

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="data_in",
                    severity="info",
                    message=f"Extracted financial query: {financial_query}",
                    payload={
                        "financial_query": financial_query,
                        "query_length": len(financial_query),
                        "is_stock_query": any(term in financial_query.lower() for term in ['stock', 'share', 'price']),
                        "is_market_cap_query": 'market cap' in financial_query.lower()
                    }
                )

            self.logger.info(f"Performing financial data lookup: {financial_query}")

            # Get query parameters
            data_type = uif.intermediate_data.get("data_type", "general")
            company_symbol = uif.intermediate_data.get("company_symbol", "")
            specific_metric = uif.intermediate_data.get("specific_metric", "")

            # Perform secure financial data search
            search_start_time = time.time()
            financial_results = self._perform_secure_financial_search(
                financial_query, data_type, company_symbol, specific_metric
            )
            search_duration = (time.time() - search_start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="tool_call",
                    severity="info",
                    message=f"Financial data search completed: {len(financial_results['data'])} results found",
                    duration_ms=search_duration,
                    payload={
                        "financial_query": financial_query,
                        "results_count": len(financial_results["data"]),
                        "confidence": financial_results["confidence"],
                        "data_source": financial_results["source"],
                        "extracted_value": financial_results["extracted_value"],
                        "search_duration_ms": search_duration
                    }
                )

            # Store results in UIF
            uif.intermediate_data["financial_data"] = financial_results["data"]
            uif.intermediate_data["data_confidence"] = financial_results["confidence"]
            uif.intermediate_data["data_source"] = financial_results["source"]
            uif.intermediate_data["extracted_value"] = financial_results["extracted_value"]

            # Set skill outputs
            uif.set_skill_output(self.skill_name, {
                "query": financial_query,
                "data_found": len(financial_results["data"]) > 0,
                "confidence": financial_results["confidence"],
                "extracted_value": financial_results["extracted_value"],
                "source": financial_results["source"]
            })

            total_duration = (time.time() - start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="data_out",
                    severity="info",
                    message=f"Financial data tool execution completed successfully",
                    duration_ms=total_duration,
                    payload={
                        "data_found": len(financial_results["data"]) > 0,
                        "confidence": financial_results["confidence"],
                        "extracted_value": financial_results["extracted_value"],
                        "execution_time_ms": total_duration,
                        "search_time_ms": search_duration,
                        "overhead_ms": total_duration - search_duration
                    }
                )

            self.logger.info(f"Financial data lookup completed: confidence={financial_results['confidence']:.2f}")

            return uif

        except Exception as e:
            total_duration = (time.time() - start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="error",
                    severity="error",
                    message=f"Financial data tool execution failed: {str(e)}",
                    duration_ms=total_duration,
                    payload={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "execution_time_ms": total_duration
                    }
                )

            self.logger.exception("Error during financial data lookup")
            raise SkillExecutionError(f"Financial data lookup failed: {str(e)}")
    
    def _extract_financial_query(self, uif: SAM_UIF) -> Optional[str]:
        """
        Extract financial query from UIF or user query.
        
        Returns:
            Financial query or None if not found
        """
        # Check intermediate data first
        if "financial_query" in uif.intermediate_data:
            return uif.intermediate_data["financial_query"]
        
        # Try to extract from user query
        query = uif.input_query
        
        # Look for financial data patterns
        financial_patterns = [
            r'market\s+cap(?:italization)?\s+(?:of\s+)?(.+)',
            r'stock\s+price\s+(?:of\s+)?(.+)',
            r'share\s+price\s+(?:of\s+)?(.+)',
            r'(?:current\s+)?value\s+(?:of\s+)?(.+)',
            r'(?:what\s+is\s+)?(.+)\s+(?:market\s+cap|stock\s+price|share\s+price)',
            r'(?:what\s+is\s+)?(.+)\s+worth',
            r'revenue\s+(?:of\s+)?(.+)',
            r'earnings\s+(?:of\s+)?(.+)',
            r'financial\s+data\s+(?:for\s+)?(.+)',
        ]
        
        query_lower = query.lower()
        for pattern in financial_patterns:
            match = re.search(pattern, query_lower)
            if match:
                financial_query = match.group(1).strip()
                # Clean up the query
                financial_query = self._clean_financial_query(financial_query)
                return financial_query
        
        # If no pattern matches but query contains financial keywords, use entire query
        if self._is_financial_query(query):
            return self._clean_financial_query(query)
        
        return None
    
    def _clean_financial_query(self, query: str) -> str:
        """
        Clean and normalize financial query.

        Returns:
            Cleaned financial query
        """
        # Remove common question words and punctuation
        words_to_remove = ['?', '.', '!', 'please', 'can you', 'could you', 'what is', 'what are']

        cleaned = query
        for word in words_to_remove:
            cleaned = cleaned.replace(word, ' ')

        # Remove calculation parts from the query (everything after "then" or "and then")
        calculation_separators = [' then ', ' and then ', '. then', '. and then']
        for separator in calculation_separators:
            if separator in cleaned.lower():
                cleaned = cleaned.lower().split(separator)[0]
                break

        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())

        return cleaned.strip()
    
    def _is_financial_query(self, query: str) -> bool:
        """
        Check if query appears to be a financial data request.
        
        Returns:
            True if query appears to be financial
        """
        financial_indicators = [
            'market cap', 'market capitalization', 'stock price', 'share price',
            'financial data', 'revenue', 'earnings', 'valuation', 'worth',
            'nasdaq', 'nyse', 'ticker', 'symbol', 'dividend', 'pe ratio',
            'price to earnings', 'market value', 'enterprise value',
            'cost', 'price', 'value', 'trading', 'current price', 'today',
            'stock', 'shares', 'equity', 'investment', 'finance', 'financial'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in financial_indicators)
    
    def _perform_secure_financial_search(self, query: str, data_type: str, 
                                       company_symbol: str, specific_metric: str) -> Dict[str, Any]:
        """
        Perform financial data search with security controls.
        
        Returns:
            Dictionary with financial data results
        """
        def search_func():
            return self._execute_financial_search(query, data_type, company_symbol, specific_metric)
        
        # Execute with security manager
        security_result = self._security_manager.execute_tool_safely(
            self.skill_name,
            search_func
        )
        
        if not security_result.success:
            if security_result.rate_limited:
                raise SkillExecutionError("Financial data search rate limit exceeded")
            elif security_result.security_violations:
                raise SkillExecutionError(f"Security violations: {security_result.security_violations}")
            else:
                raise SkillExecutionError(f"Financial data search failed: {security_result.error_message}")
        
        return security_result.output
    
    def _execute_financial_search(self, query: str, data_type: str, 
                                company_symbol: str, specific_metric: str) -> Dict[str, Any]:
        """
        Execute financial data search using Serper API.
        
        Returns:
            Dictionary with financial data and extracted values
        """
        try:
            if self.serper_api_key:
                # Use Serper API for high-quality financial data
                return self._search_with_serper(query, data_type, company_symbol, specific_metric)
            else:
                # Fallback to simulated financial data
                return self._simulate_financial_data(query, data_type, company_symbol, specific_metric)
                
        except Exception as e:
            self.logger.error(f"Financial search execution failed: {e}")
            return {
                "data": [],
                "confidence": 0.0,
                "source": "error",
                "extracted_value": None,
                "error": str(e)
            }
    
    def _search_with_serper(self, query: str, data_type: str, 
                          company_symbol: str, specific_metric: str) -> Dict[str, Any]:
        """
        Search for financial data using Serper API.
        
        Returns:
            Dictionary with search results and extracted financial data
        """
        try:
            # Optimize query for financial data
            optimized_query = self._optimize_financial_query(query, data_type, company_symbol, specific_metric)
            
            # Serper API request
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": optimized_query,
                "num": 10,
                "gl": "us",
                "hl": "en"
            }
            
            response = self._session.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            search_data = response.json()
            
            # Extract financial data from search results
            financial_data = self._extract_financial_data_from_results(search_data, query, specific_metric)
            
            return financial_data
            
        except Exception as e:
            self.logger.error(f"Serper API search failed: {e}")
            # Fallback to simulation
            return self._simulate_financial_data(query, data_type, company_symbol, specific_metric)
    
    def _optimize_financial_query(self, query: str, data_type: str, 
                                company_symbol: str, specific_metric: str) -> str:
        """
        Optimize query for better financial data retrieval.
        
        Returns:
            Optimized search query
        """
        # Add specific financial terms to improve search accuracy
        if "market cap" in query.lower() or "market capitalization" in query.lower():
            return f"{query} market capitalization current value"
        elif "stock price" in query.lower() or "share price" in query.lower():
            return f"{query} current stock price today"
        elif "revenue" in query.lower():
            return f"{query} annual revenue financial data"
        elif "earnings" in query.lower():
            return f"{query} quarterly earnings financial results"
        else:
            return f"{query} financial data current"
    
    def _extract_financial_data_from_results(self, search_data: Dict[str, Any], 
                                           original_query: str, specific_metric: str) -> Dict[str, Any]:
        """
        Extract specific financial data from Serper search results.
        
        Returns:
            Dictionary with extracted financial data
        """
        extracted_data = []
        extracted_value = None
        confidence = 0.0
        
        try:
            # Check organic results for financial data
            organic_results = search_data.get("organic", [])
            
            for result in organic_results:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                
                # Extract numerical values from title and snippet
                extracted_numbers = self._extract_numbers_from_text(f"{title} {snippet}")
                
                if extracted_numbers:
                    extracted_data.append({
                        "title": title,
                        "snippet": snippet,
                        "url": link,
                        "extracted_numbers": extracted_numbers,
                        "relevance_score": self._calculate_relevance_score(title, snippet, original_query)
                    })
            
            # Check knowledge graph for direct answers
            knowledge_graph = search_data.get("knowledgeGraph", {})
            if knowledge_graph:
                kg_data = self._extract_from_knowledge_graph(knowledge_graph, original_query)
                if kg_data:
                    extracted_data.append(kg_data)
            
            # Check answer box for direct financial data
            answer_box = search_data.get("answerBox", {})
            if answer_box:
                answer_data = self._extract_from_answer_box(answer_box, original_query)
                if answer_data:
                    extracted_data.append(answer_data)
                    extracted_value = answer_data.get("primary_value")
            
            # Calculate overall confidence
            if extracted_data:
                confidence = min(0.95, 0.6 + (len(extracted_data) * 0.1))
                
                # Find the most likely value if not already found
                if not extracted_value and extracted_data:
                    extracted_value = self._find_most_likely_value(extracted_data, original_query)
            
            return {
                "data": extracted_data,
                "confidence": confidence,
                "source": "serper_api",
                "extracted_value": extracted_value,
                "query": original_query,
                "total_results": len(extracted_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting financial data: {e}")
            return {
                "data": [],
                "confidence": 0.0,
                "source": "error",
                "extracted_value": None,
                "error": str(e)
            }
    
    def _extract_numbers_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract numerical values from text that might be financial data.
        
        Returns:
            List of extracted numbers with context
        """
        numbers = []
        
        # Patterns for financial numbers
        patterns = [
            # Market cap patterns: $2.5 trillion, $500 billion, etc.
            (r'\$(\d+(?:\.\d+)?)\s*(trillion|billion|million|thousand)', 'currency'),
            # Stock price patterns: $150.25, $50.00, etc.
            (r'\$(\d+(?:\.\d+)?)', 'currency'),
            # Percentage patterns: 15.5%, 3.2%, etc.
            (r'(\d+(?:\.\d+)?)\s*%', 'percentage'),
            # Large numbers with commas: 1,500,000,000
            (r'(\d{1,3}(?:,\d{3})+)', 'number'),
            # Decimal numbers: 2.5, 150.25, etc.
            (r'(\d+\.\d+)', 'decimal'),
            # Whole numbers: 1500, 2500, etc.
            (r'\b(\d+)\b', 'integer')
        ]
        
        for pattern, number_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                context = text[max(0, match.start()-20):match.end()+20]
                
                numbers.append({
                    "value": value,
                    "type": number_type,
                    "context": context.strip(),
                    "full_match": match.group(0)
                })
        
        return numbers
    
    def _calculate_relevance_score(self, title: str, snippet: str, query: str) -> float:
        """
        Calculate relevance score for a search result.
        
        Returns:
            Relevance score between 0.0 and 1.0
        """
        text = f"{title} {snippet}".lower()
        query_words = query.lower().split()
        
        # Count matching words
        matches = sum(1 for word in query_words if word in text)
        base_score = matches / len(query_words) if query_words else 0.0
        
        # Boost for financial keywords
        financial_keywords = ['market cap', 'stock price', 'revenue', 'earnings', 'billion', 'trillion']
        financial_boost = sum(0.1 for keyword in financial_keywords if keyword in text)
        
        return min(1.0, base_score + financial_boost)
    
    def _extract_from_knowledge_graph(self, kg_data: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """Extract financial data from knowledge graph."""
        # Implementation would extract structured data from Google's knowledge graph
        # This is a placeholder for the actual implementation
        return None
    
    def _extract_from_answer_box(self, answer_data: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """Extract financial data from answer box."""
        # Implementation would extract direct answers from Google's answer box
        # This is a placeholder for the actual implementation
        return None
    
    def _find_most_likely_value(self, extracted_data: List[Dict[str, Any]], query: str) -> Optional[str]:
        """Find the most likely financial value from extracted data."""
        # Implementation would analyze extracted data to find the most relevant value
        # This is a placeholder for the actual implementation
        if extracted_data and extracted_data[0].get("extracted_numbers"):
            return extracted_data[0]["extracted_numbers"][0].get("value")
        return None
    
    def _simulate_financial_data(self, query: str, data_type: str,
                               company_symbol: str, specific_metric: str) -> Dict[str, Any]:
        """
        Simulate financial data when Serper API is not available.

        Returns:
            Dictionary with simulated financial data
        """
        # Provide realistic simulated data based on the query
        query_lower = query.lower()

        if 'nvidia' in query_lower:
            if 'market cap' in query_lower or 'market capitalization' in query_lower:
                simulated_data = [{
                    "title": "NVIDIA Corporation (NVDA) Market Capitalization",
                    "snippet": "NVIDIA Corporation has a current market capitalization of approximately $2.5 trillion as of recent trading.",
                    "url": "https://finance.yahoo.com/quote/NVDA",
                    "extracted_numbers": [{"value": "2.5", "type": "currency", "context": "market capitalization $2.5 trillion", "full_match": "$2.5 trillion"}],
                    "relevance_score": 0.95,
                    "primary_value": "$2.5 trillion"
                }]
                extracted_value = "$2.5 trillion"
            else:
                simulated_data = [{
                    "title": f"NVIDIA Financial Information - {query}",
                    "snippet": f"NVIDIA Corporation financial data related to: {query}",
                    "url": "https://finance.yahoo.com/quote/NVDA",
                    "extracted_numbers": [{"value": "150.25", "type": "currency", "context": "simulated financial data"}],
                    "relevance_score": 0.8
                }]
                extracted_value = "$150.25"
        else:
            # Generic simulation for other companies
            simulated_data = [{
                "title": f"Financial data for {query}",
                "snippet": f"Financial information about {query}. Market data and metrics.",
                "url": "https://example.com/financial-data",
                "extracted_numbers": [{"value": "100.50", "type": "currency", "context": "simulated data"}],
                "relevance_score": 0.7
            }]
            extracted_value = "$100.50"
        
        return {
            "data": simulated_data,
            "confidence": 0.5,  # Lower confidence for simulated data
            "source": "simulation",
            "extracted_value": extracted_value,
            "query": query,
            "total_results": len(simulated_data)
        }
    
    def can_handle_query(self, query: str) -> bool:
        """
        Check if this tool can handle the given query.
        
        Args:
            query: User query to check
            
        Returns:
            True if query appears to be a financial data request
        """
        return self._is_financial_query(query)

    def _log_trace_event(self, trace_id: str, event_type: str, severity: str,
                        message: str, duration_ms: Optional[float] = None,
                        payload: Optional[Dict[str, Any]] = None) -> None:
        """Log a trace event for the financial data tool."""
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
                    "has_serper_api": bool(self.serper_api_key)
                }
            )
        except ImportError:
            # Tracing not available, continue without logging
            pass
        except Exception as e:
            # Don't let tracing errors break the tool
            self.logger.debug(f"Trace logging failed: {e}")
