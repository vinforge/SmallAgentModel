#!/usr/bin/env python3
"""
SSRL Integration Layer
======================

Integration layer that connects the HybridQueryRouter with SAM's existing
query processing systems. This module provides a seamless interface for
incorporating SSRL capabilities into the current SAM architecture.

Author: SAM Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class SSRLIntegration:
    """
    Integration layer for SSRL functionality with existing SAM systems.
    
    This class provides methods to integrate the HybridQueryRouter with
    SAM's current query processing pipeline while maintaining backward
    compatibility.
    """
    
    def __init__(self, enable_ssrl: bool = True):
        """
        Initialize SSRL integration.
        
        Args:
            enable_ssrl: Whether SSRL functionality is enabled
        """
        self.enable_ssrl = enable_ssrl
        self.integration_stats = {
            'total_queries': 0,
            'ssrl_handled': 0,
            'fallback_to_existing': 0,
            'errors': 0
        }
        
        logger.info(f"SSRL Integration initialized (enabled: {enable_ssrl})")
    
    def process_query_with_ssrl(self, 
                               query: str, 
                               context: Dict[str, Any] = None,
                               force_existing_system: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Process a query using the SSRL hybrid system with fallback to existing logic.
        
        Args:
            query: User query
            context: Query context including conversation history, files, etc.
            force_existing_system: If True, bypass SSRL and use existing system
            
        Returns:
            Tuple of (response_content, metadata)
        """
        self.integration_stats['total_queries'] += 1
        
        try:
            # Check if SSRL should be used
            if not self.enable_ssrl or force_existing_system:
                return self._fallback_to_existing_system(query, context)
            
            # Use the hybrid router
            from sam.orchestration.hybrid_query_router import get_hybrid_query_router
            
            router = get_hybrid_query_router()
            routing_result = router.route_query(query, context)
            
            if routing_result.success:
                # Handle the routing decision
                response_content, metadata = self._handle_routing_decision(
                    routing_result, query, context
                )
                
                self.integration_stats['ssrl_handled'] += 1
                return response_content, metadata
            else:
                # SSRL failed, fallback to existing system
                logger.warning(f"SSRL routing failed: {routing_result.error}")
                return self._fallback_to_existing_system(query, context)
                
        except Exception as e:
            logger.error(f"SSRL integration error: {e}")
            self.integration_stats['errors'] += 1
            return self._fallback_to_existing_system(query, context)
    
    def _handle_routing_decision(self, 
                               routing_result, 
                               query: str, 
                               context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Handle the routing decision from the HybridQueryRouter.
        
        Args:
            routing_result: Result from HybridQueryRouter
            query: Original query
            context: Query context
            
        Returns:
            Tuple of (response_content, metadata)
        """
        from sam.orchestration.hybrid_query_router import RoutingDecision
        
        decision = routing_result.decision
        metadata = {
            'routing_stage': routing_result.stage_reached.value,
            'confidence_score': routing_result.confidence_score,
            'execution_time': routing_result.execution_time,
            'reasoning_steps': routing_result.reasoning_steps,
            'ssrl_metadata': routing_result.metadata
        }
        
        if decision == RoutingDecision.CALCULATOR:
            # Route to calculator tool
            response = self._route_to_calculator(query, context)
            metadata['tool_used'] = 'calculator'
            return response, metadata
            
        elif decision == RoutingDecision.CODE_INTERPRETER:
            # Route to CSV/code analysis
            response = self._route_to_code_interpreter(query, context)
            metadata['tool_used'] = 'code_interpreter'
            return response, metadata
            
        elif decision == RoutingDecision.TABLE_ANALYSIS:
            # Route to table analysis
            response = self._route_to_table_analysis(query, context)
            metadata['tool_used'] = 'table_analysis'
            return response, metadata
            
        elif decision == RoutingDecision.SELF_SEARCH_SUCCESS:
            # Use SSRL result directly
            metadata['tool_used'] = 'ssrl_self_search'
            return routing_result.content, metadata
            
        elif decision == RoutingDecision.ESCALATION_REQUIRED:
            # Escalate to existing SAM systems
            response, existing_metadata = self._fallback_to_existing_system(query, context)
            metadata.update(existing_metadata)
            metadata['tool_used'] = 'existing_system_fallback'
            return response, metadata
            
        else:
            # Unknown decision, fallback
            logger.warning(f"Unknown routing decision: {decision}")
            response, existing_metadata = self._fallback_to_existing_system(query, context)
            metadata.update(existing_metadata)
            metadata['tool_used'] = 'unknown_fallback'
            return response, metadata
    
    def _route_to_calculator(self, query: str, context: Dict[str, Any]) -> str:
        """
        Route query to calculator tool.
        
        Args:
            query: Mathematical query
            context: Query context
            
        Returns:
            Calculator result
        """
        try:
            # Import and use existing calculator functionality
            # This is a placeholder - integrate with actual calculator tool
            
            # For now, return a placeholder response
            return f"CALCULATOR_RESULT: {query} (Integration with calculator tool needed)"
            
        except Exception as e:
            logger.error(f"Calculator routing failed: {e}")
            return f"Calculator error: {e}"
    
    def _route_to_code_interpreter(self, query: str, context: Dict[str, Any]) -> str:
        """
        Route query to code interpreter for CSV analysis.
        
        Args:
            query: Data analysis query
            context: Query context
            
        Returns:
            Code interpreter result
        """
        try:
            # Import and use existing CSV analysis functionality
            # This integrates with the existing generate_tool_enhanced_response
            
            # For now, return a placeholder that indicates integration is needed
            return f"CSV_ANALYSIS_RESULT: {query} (Integration with existing CSV analysis needed)"
            
        except Exception as e:
            logger.error(f"Code interpreter routing failed: {e}")
            return f"Code interpreter error: {e}"
    
    def _route_to_table_analysis(self, query: str, context: Dict[str, Any]) -> str:
        """
        Route query to table analysis tool.
        
        Args:
            query: Table analysis query
            context: Query context
            
        Returns:
            Table analysis result
        """
        try:
            # Import and use existing table analysis functionality
            # This is a placeholder - integrate with actual table analysis tool
            
            return f"TABLE_ANALYSIS_RESULT: {query} (Integration with table analysis tool needed)"
            
        except Exception as e:
            logger.error(f"Table analysis routing failed: {e}")
            return f"Table analysis error: {e}"
    
    def _fallback_to_existing_system(self, 
                                   query: str, 
                                   context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Fallback to existing SAM query processing system.
        
        Args:
            query: User query
            context: Query context
            
        Returns:
            Tuple of (response_content, metadata)
        """
        self.integration_stats['fallback_to_existing'] += 1
        
        try:
            # This is where we would integrate with the existing SAM query processing
            # For now, return a placeholder that indicates where integration is needed
            
            response = f"EXISTING_SYSTEM_RESULT: {query} (Integration with existing SAM system needed)"
            metadata = {
                'tool_used': 'existing_system',
                'integration_status': 'placeholder'
            }
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Existing system fallback failed: {e}")
            return f"System error: {e}", {'error': str(e)}
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = self.integration_stats.copy()
        
        total = stats['total_queries']
        if total > 0:
            stats['ssrl_percentage'] = (stats['ssrl_handled'] / total) * 100
            stats['fallback_percentage'] = (stats['fallback_to_existing'] / total) * 100
            stats['error_percentage'] = (stats['errors'] / total) * 100
        else:
            stats['ssrl_percentage'] = 0
            stats['fallback_percentage'] = 0
            stats['error_percentage'] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset integration statistics."""
        self.integration_stats = {
            'total_queries': 0,
            'ssrl_handled': 0,
            'fallback_to_existing': 0,
            'errors': 0
        }
    
    def enable(self):
        """Enable SSRL integration."""
        self.enable_ssrl = True
        logger.info("SSRL integration enabled")
    
    def disable(self):
        """Disable SSRL integration."""
        self.enable_ssrl = False
        logger.info("SSRL integration disabled")


# Global instance for easy access
_ssrl_integration = None

def get_ssrl_integration() -> SSRLIntegration:
    """Get the global SSRLIntegration instance."""
    global _ssrl_integration
    if _ssrl_integration is None:
        _ssrl_integration = SSRLIntegration()
    return _ssrl_integration


def process_query_with_ssrl_integration(query: str, 
                                      context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to process a query with SSRL integration.
    
    Args:
        query: User query
        context: Query context
        
    Returns:
        Tuple of (response_content, metadata)
    """
    integration = get_ssrl_integration()
    return integration.process_query_with_ssrl(query, context)
