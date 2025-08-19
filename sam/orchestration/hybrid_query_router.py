#!/usr/bin/env python3
"""
Hybrid Query Router
===================

Implements the 4-stage hybrid routing system that combines SSRL self-search
with SAM's existing fast-path and external search capabilities.

Stage 1: Fast-Path Triage (deterministic routing)
Stage 2: Self-Search Attempt (SSRL reasoning)
Stage 3: Confidence-Based Escalation (decision making)
Stage 4: Document-First & Web Search (existing logic)

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingStage(Enum):
    """Stages in the hybrid routing process."""
    FAST_PATH = "FAST_PATH"
    SELF_SEARCH = "SELF_SEARCH"
    CONFIDENCE_ESCALATION = "CONFIDENCE_ESCALATION"
    EXTERNAL_SEARCH = "EXTERNAL_SEARCH"


class RoutingDecision(Enum):
    """Possible routing decisions."""
    CALCULATOR = "CALCULATOR"
    CODE_INTERPRETER = "CODE_INTERPRETER"
    TABLE_ANALYSIS = "TABLE_ANALYSIS"
    SELF_SEARCH_SUCCESS = "SELF_SEARCH_SUCCESS"
    DOCUMENT_SEARCH = "DOCUMENT_SEARCH"
    WEB_SEARCH = "WEB_SEARCH"
    ESCALATION_REQUIRED = "ESCALATION_REQUIRED"


@dataclass
class RoutingResult:
    """Result from the hybrid routing process."""
    decision: RoutingDecision
    stage_reached: RoutingStage
    content: str
    confidence_score: float
    execution_time: float
    reasoning_steps: List[str]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class HybridQueryRouter:
    """
    Hybrid Query Router implementing the 4-stage decision flow.
    
    This router combines the speed of deterministic routing with the
    intelligence of SSRL self-search and the reliability of existing
    external search systems.
    """
    
    def __init__(self, 
                 self_search_confidence_threshold: float = 0.8,
                 enable_self_search: bool = True,
                 fast_path_enabled: bool = True):
        """
        Initialize the HybridQueryRouter.
        
        Args:
            self_search_confidence_threshold: Minimum confidence to trust self-search
            enable_self_search: Whether to use SSRL self-search stage
            fast_path_enabled: Whether to use fast-path triage
        """
        self.self_search_confidence_threshold = self_search_confidence_threshold
        self.enable_self_search = enable_self_search
        self.fast_path_enabled = fast_path_enabled
        
        # Performance tracking
        self.routing_stats = {
            'total_queries': 0,
            'fast_path_hits': 0,
            'self_search_successes': 0,
            'external_escalations': 0,
            'avg_response_time': 0.0
        }
        
        logger.info(f"HybridQueryRouter initialized with confidence_threshold={self_search_confidence_threshold}")
    
    def route_query(self, query: str, context: Dict[str, Any] = None) -> RoutingResult:
        """
        Route a query through the 4-stage hybrid system.
        
        Args:
            query: The user's question or request
            context: Additional context including conversation history, uploaded files, etc.
            
        Returns:
            RoutingResult with the final decision and content
        """
        start_time = time.time()
        self.routing_stats['total_queries'] += 1
        
        try:
            # Stage 1: Fast-Path Triage
            if self.fast_path_enabled:
                fast_path_result = self._stage1_fast_path_triage(query, context)
                if fast_path_result:
                    self.routing_stats['fast_path_hits'] += 1
                    execution_time = time.time() - start_time
                    self._update_avg_response_time(execution_time)
                    return fast_path_result
            
            # Stage 2: Self-Search Attempt
            if self.enable_self_search:
                self_search_result = self._stage2_self_search_attempt(query, context)
                if self_search_result:
                    # Stage 3: Confidence-Based Escalation
                    if self_search_result.confidence_score >= self.self_search_confidence_threshold:
                        self.routing_stats['self_search_successes'] += 1
                        execution_time = time.time() - start_time
                        self._update_avg_response_time(execution_time)
                        return self_search_result
            
            # Stage 4: Document-First & Web Search (External)
            external_result = self._stage4_external_search(query, context)
            self.routing_stats['external_escalations'] += 1
            execution_time = time.time() - start_time
            self._update_avg_response_time(execution_time)
            return external_result
            
        except Exception as e:
            logger.error(f"HybridQueryRouter failed: {e}")
            execution_time = time.time() - start_time
            return RoutingResult(
                decision=RoutingDecision.ESCALATION_REQUIRED,
                stage_reached=RoutingStage.FAST_PATH,
                content="",
                confidence_score=0.0,
                execution_time=execution_time,
                reasoning_steps=[f"Router error: {e}"],
                metadata={'error': str(e)},
                success=False,
                error=str(e)
            )
    
    def _stage1_fast_path_triage(self, query: str, context: Dict[str, Any]) -> Optional[RoutingResult]:
        """
        Stage 1: Fast-Path Triage for obvious, deterministic cases.
        
        Args:
            query: User query
            context: Query context
            
        Returns:
            RoutingResult if fast-path routing is possible, None otherwise
        """
        try:
            # Check for pure math expressions
            if self._is_pure_calculation(query):
                return RoutingResult(
                    decision=RoutingDecision.CALCULATOR,
                    stage_reached=RoutingStage.FAST_PATH,
                    content=f"ROUTE_TO_CALCULATOR: {query}",
                    confidence_score=1.0,
                    execution_time=0.0,
                    reasoning_steps=["Fast-path: Pure calculation detected"],
                    metadata={'tool': 'calculator', 'query': query},
                    success=True
                )
            
            # Check for CSV analysis
            if self._has_csv_context(query, context):
                return RoutingResult(
                    decision=RoutingDecision.CODE_INTERPRETER,
                    stage_reached=RoutingStage.FAST_PATH,
                    content=f"ROUTE_TO_CSV_ANALYSIS: {query}",
                    confidence_score=1.0,
                    execution_time=0.0,
                    reasoning_steps=["Fast-path: CSV analysis detected"],
                    metadata={'tool': 'code_interpreter', 'query': query},
                    success=True
                )
            
            # Check for table analysis
            if self._is_table_analysis_query(query):
                return RoutingResult(
                    decision=RoutingDecision.TABLE_ANALYSIS,
                    stage_reached=RoutingStage.FAST_PATH,
                    content=f"ROUTE_TO_TABLE_ANALYSIS: {query}",
                    confidence_score=1.0,
                    execution_time=0.0,
                    reasoning_steps=["Fast-path: Table analysis detected"],
                    metadata={'tool': 'table_analysis', 'query': query},
                    success=True
                )
            
            # No fast-path match
            return None
            
        except Exception as e:
            logger.error(f"Fast-path triage failed: {e}")
            return None
    
    def _stage2_self_search_attempt(self, query: str, context: Dict[str, Any]) -> Optional[RoutingResult]:
        """
        Stage 2: Self-Search Attempt using SSRL reasoning.
        
        Args:
            query: User query
            context: Query context
            
        Returns:
            RoutingResult with self-search outcome, None if failed
        """
        try:
            from sam.orchestration.skills.self_search_tool import get_self_search_tool
            
            # Get the self-search tool
            self_search_tool = get_self_search_tool()
            
            # Perform self-search
            ssrl_result = self_search_tool.execute(query, context)
            
            if ssrl_result.success:
                return RoutingResult(
                    decision=RoutingDecision.SELF_SEARCH_SUCCESS,
                    stage_reached=RoutingStage.SELF_SEARCH,
                    content=ssrl_result.content,
                    confidence_score=ssrl_result.confidence_score,
                    execution_time=ssrl_result.execution_time,
                    reasoning_steps=ssrl_result.reasoning_steps,
                    metadata={
                        'ssrl_confidence_level': ssrl_result.confidence_level.value,
                        'search_iterations': ssrl_result.search_iterations,
                        'tool': 'self_search'
                    },
                    success=True
                )
            else:
                logger.warning(f"Self-search failed: {ssrl_result.error}")
                return None
                
        except Exception as e:
            logger.error(f"Self-search attempt failed: {e}")
            return None
    
    def _stage4_external_search(self, query: str, context: Dict[str, Any]) -> RoutingResult:
        """
        Stage 4: Document-First & Web Search using existing SAM logic.
        
        Args:
            query: User query
            context: Query context
            
        Returns:
            RoutingResult with external search routing decision
        """
        try:
            # This is a placeholder that will integrate with existing SAM routing logic
            # In the actual implementation, this would call the existing routing functions
            
            # For now, return a placeholder that indicates external routing is needed
            return RoutingResult(
                decision=RoutingDecision.ESCALATION_REQUIRED,
                stage_reached=RoutingStage.EXTERNAL_SEARCH,
                content=f"ROUTE_TO_EXTERNAL_SEARCH: {query}",
                confidence_score=0.5,
                execution_time=0.0,
                reasoning_steps=["External search routing required"],
                metadata={
                    'tool': 'external_search',
                    'query': query,
                    'requires_integration': True
                },
                success=True
            )
            
        except (ImportError, AttributeError, ValueError) as e:
            logger.error(f"External search routing failed: {e}")
            return RoutingResult(
                decision=RoutingDecision.ESCALATION_REQUIRED,
                stage_reached=RoutingStage.EXTERNAL_SEARCH,
                content="",
                confidence_score=0.0,
                execution_time=0.0,
                reasoning_steps=[f"External search error: {e}"],
                metadata={'error': str(e)},
                success=False,
                error=str(e)
            )

    def _is_pure_calculation(self, query: str) -> bool:
        """
        Check if query is a pure mathematical calculation.

        Args:
            query: User query

        Returns:
            True if query is a pure calculation
        """
        import re

        # Remove whitespace and convert to lowercase
        clean_query = query.strip().lower()

        # Patterns that indicate pure calculations
        calc_patterns = [
            r'^[\d\s\+\-\*\/\(\)\.\^%]+$',  # Basic math expression
            r'^\d+[\s]*[\+\-\*\/][\s]*\d+',  # Simple arithmetic
            r'^calculate[\s]+[\d\s\+\-\*\/\(\)\.\^%]+$',  # "calculate X"
            r'^what[\s]+is[\s]+[\d\s\+\-\*\/\(\)\.\^%]+[\?]*$',  # "what is X"
        ]

        for pattern in calc_patterns:
            if re.match(pattern, clean_query):
                return True

        # Check for math keywords with simple expressions
        math_keywords = ['calculate', 'compute', 'solve', 'what is']
        has_math_keyword = any(keyword in clean_query for keyword in math_keywords)

        if has_math_keyword:
            # Check if it contains mostly numbers and operators
            math_chars = re.findall(r'[\d\+\-\*\/\(\)\.\^%]', clean_query)
            total_chars = len(re.findall(r'[a-zA-Z\d\+\-\*\/\(\)\.\^%]', clean_query))

            if total_chars > 0 and len(math_chars) / total_chars > 0.6:
                return True

        return False

    def _has_csv_context(self, query: str, context: Dict[str, Any]) -> bool:
        """
        Check if query has CSV context (uploaded CSV files).

        Args:
            query: User query
            context: Query context

        Returns:
            True if CSV context is available
        """
        if not context:
            return False

        # Check for uploaded CSV files in context
        if context.get('uploaded_csv_files'):
            return True

        # Check session state for CSV files (if available)
        try:
            import streamlit as st
            if (hasattr(st, 'session_state') and
                hasattr(st.session_state, 'uploaded_csv_files') and
                st.session_state.uploaded_csv_files):
                return True
        except (ImportError, AttributeError):
            pass

        # Check for CSV-related keywords in query
        csv_keywords = [
            'csv', 'spreadsheet', 'data', 'dataset', 'table',
            'average', 'mean', 'sum', 'count', 'calculate',
            'salary', 'employee', 'column', 'row'
        ]

        query_lower = query.lower()
        csv_keyword_count = sum(1 for keyword in csv_keywords if keyword in query_lower)

        # If multiple CSV keywords, likely a data analysis query
        return csv_keyword_count >= 2

    def _is_table_analysis_query(self, query: str) -> bool:
        """
        Check if query is requesting table analysis.

        Args:
            query: User query

        Returns:
            True if query is for table analysis
        """
        table_keywords = [
            'table', 'chart', 'graph', 'visualization', 'plot',
            'create table', 'make table', 'show table', 'generate table',
            'tabulate', 'format as table'
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in table_keywords)

    def _update_avg_response_time(self, execution_time: float):
        """Update the average response time statistic."""
        total_queries = self.routing_stats['total_queries']
        current_avg = self.routing_stats['avg_response_time']

        # Calculate new average
        new_avg = ((current_avg * (total_queries - 1)) + execution_time) / total_queries
        self.routing_stats['avg_response_time'] = new_avg

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get current routing statistics."""
        stats = self.routing_stats.copy()

        # Calculate percentages
        total = stats['total_queries']
        if total > 0:
            stats['fast_path_percentage'] = (stats['fast_path_hits'] / total) * 100
            stats['self_search_percentage'] = (stats['self_search_successes'] / total) * 100
            stats['external_percentage'] = (stats['external_escalations'] / total) * 100
        else:
            stats['fast_path_percentage'] = 0
            stats['self_search_percentage'] = 0
            stats['external_percentage'] = 0

        return stats

    def reset_stats(self):
        """Reset routing statistics."""
        self.routing_stats = {
            'total_queries': 0,
            'fast_path_hits': 0,
            'self_search_successes': 0,
            'external_escalations': 0,
            'avg_response_time': 0.0
        }

    def configure(self, **kwargs):
        """
        Update router configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        if 'self_search_confidence_threshold' in kwargs:
            self.self_search_confidence_threshold = kwargs['self_search_confidence_threshold']
            logger.info(f"Updated confidence threshold to {self.self_search_confidence_threshold}")

        if 'enable_self_search' in kwargs:
            self.enable_self_search = kwargs['enable_self_search']
            logger.info(f"Self-search enabled: {self.enable_self_search}")

        if 'fast_path_enabled' in kwargs:
            self.fast_path_enabled = kwargs['fast_path_enabled']
            logger.info(f"Fast-path enabled: {self.fast_path_enabled}")


# Global instance for easy access
_hybrid_router = None

def get_hybrid_query_router() -> HybridQueryRouter:
    """Get the global HybridQueryRouter instance."""
    global _hybrid_router
    if _hybrid_router is None:
        _hybrid_router = HybridQueryRouter()
    return _hybrid_router


def route_query_hybrid(query: str, context: Dict[str, Any] = None) -> RoutingResult:
    """
    Convenience function to route a query through the hybrid system.

    Args:
        query: User query
        context: Query context

    Returns:
        RoutingResult with routing decision and content
    """
    router = get_hybrid_query_router()
    return router.route_query(query, context)
