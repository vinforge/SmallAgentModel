#!/usr/bin/env python3
"""
SLP Fallback Service
Clean, modular SLP fallback generator using extracted services.
"""

import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class SLPFallbackService:
    """
    Clean SLP fallback service that uses extracted services for
    context building and response generation.
    """
    
    def __init__(self):
        self.context_builder = None
        self.response_generator = None
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize required services."""
        try:
            from services.context_builder_service import get_context_builder_service
            from services.response_generator_service import get_response_generator_service
            
            self.context_builder = get_context_builder_service()
            self.response_generator = get_response_generator_service()
            
            logger.info("✅ SLP Fallback Service initialized with extracted services")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize some services: {e}")
    
    def create_fallback_generator(self, memoir_context: Dict[str, Any]) -> Callable:
        """
        Create a clean fallback generator function.
        
        Args:
            memoir_context: MEMOIR context data
            
        Returns:
            Fallback generator function
        """
        def clean_fallback_generator(query: str, context: Dict[str, Any]) -> str:
            """
            Clean fallback generator using extracted services.
            
            Args:
                query: User query
                context: Context data including conversation history, etc.
                
            Returns:
                Generated response
            """
            try:
                logger.info(f"🔄 SLP Fallback: Processing query '{query[:50]}...'")
                
                # Step 1: Build comprehensive context
                context_data = self._prepare_context_data(query, context, memoir_context)
                context_parts = self._build_context(query, context_data)
                
                # Step 2: Generate response
                full_prompt = "\n".join(context_parts)
                response = self._generate_response(full_prompt)
                
                logger.info(f"✅ SLP Fallback: Generated response ({len(response)} chars)")
                return response
                
            except Exception as e:
                logger.error(f"❌ SLP Fallback generator error: {e}")
                return self._create_error_response(query, e)
        
        return clean_fallback_generator
    
    def _prepare_context_data(self, query: str, context: Dict[str, Any], memoir_context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context data for the context builder."""
        context_data = {
            # Core context
            'conversation_history': context.get('conversation_history', ''),
            'reduce_conversation_weight': context.get('reduce_conversation_weight', False),
            'document_query_detected': context.get('document_query_detected', False),
            'sources': context.get('sources', []),
            
            # MEMOIR context
            'memoir_context': memoir_context,
            
            # Search preferences
            'use_smart_search_router': context.get('use_smart_search_router', False),
        }
        
        # Log context preparation
        doc_query = context_data['document_query_detected']
        conv_history_len = len(context_data['conversation_history'])
        reduce_weight = context_data['reduce_conversation_weight']
        
        logger.info(f"🏗️ Context prepared: doc_query={doc_query}, conv_len={conv_history_len}, reduce_weight={reduce_weight}")
        
        return context_data
    
    def _build_context(self, query: str, context_data: Dict[str, Any]) -> list:
        """Build context using the context builder service."""
        if self.context_builder:
            try:
                return self.context_builder.build_context(query, context_data)
            except Exception as e:
                logger.warning(f"⚠️ Context builder failed, using fallback: {e}")
                return self._build_fallback_context(query, context_data)
        else:
            logger.warning("⚠️ Context builder not available, using fallback")
            return self._build_fallback_context(query, context_data)
    
    def _build_fallback_context(self, query: str, context_data: Dict[str, Any]) -> list:
        """Fallback context building if service is unavailable."""
        context_parts = []
        
        # Add conversation history if appropriate
        if self._should_include_conversation_history(context_data):
            conversation_history = context_data['conversation_history']
            context_parts.extend([
                "--- RECENT CONVERSATION HISTORY (Most recent first) ---",
                conversation_history,
                "--- END OF CONVERSATION HISTORY ---\n"
            ])
        
        # Add query
        context_parts.append(f"Question: {query}")
        
        # Add basic instruction
        if context_data.get('document_query_detected', False):
            context_parts.append("\nPlease provide a comprehensive response based on any available document content.")
        else:
            context_parts.append("\nPlease provide a comprehensive, helpful response.")
        
        return context_parts
    
    def _should_include_conversation_history(self, context_data: Dict[str, Any]) -> bool:
        """Determine if conversation history should be included."""
        conversation_history = context_data.get('conversation_history', '')
        reduce_conversation_weight = context_data.get('reduce_conversation_weight', False)
        
        return (
            conversation_history and 
            conversation_history != "No recent conversation history." and 
            not reduce_conversation_weight
        )
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using the response generator service."""
        if self.response_generator:
            try:
                return self.response_generator.generate_response(prompt)
            except Exception as e:
                logger.warning(f"⚠️ Response generator failed, using fallback: {e}")
                return self._create_simple_response(prompt)
        else:
            logger.warning("⚠️ Response generator not available, using fallback")
            return self._create_simple_response(prompt)
    
    def _create_simple_response(self, prompt: str) -> str:
        """Create a simple response when services are unavailable."""
        # Extract query from prompt
        lines = prompt.split('\n')
        query_line = None
        
        for line in lines:
            if line.startswith('Question:'):
                query_line = line.replace('Question:', '').strip()
                break
        
        if query_line:
            return f"I understand you're asking about: {query_line}"
        else:
            return "I understand your question and I'm processing it."
    
    def _create_error_response(self, query: str, error: Exception) -> str:
        """Create an error response."""
        if "timeout" in str(error).lower():
            return f"I apologize for the delay processing your question about '{query}'. Please try again in a moment."
        else:
            return f"I understand you're asking about: {query}"
    
    def test_services(self) -> Dict[str, bool]:
        """Test if all required services are available."""
        results = {
            'context_builder': self.context_builder is not None,
            'response_generator': self.response_generator is not None,
        }
        
        if self.response_generator:
            results['ollama_connection'] = self.response_generator.test_connection()
        
        return results

# Global instance for easy access
_slp_fallback_service = None

def get_slp_fallback_service() -> SLPFallbackService:
    """Get or create the global SLP fallback service instance."""
    global _slp_fallback_service
    if _slp_fallback_service is None:
        _slp_fallback_service = SLPFallbackService()
    return _slp_fallback_service

def create_clean_fallback_generator(memoir_context: Dict[str, Any]) -> Callable:
    """
    Convenience function to create a clean fallback generator.
    
    Args:
        memoir_context: MEMOIR context data
        
    Returns:
        Clean fallback generator function
    """
    service = get_slp_fallback_service()
    return service.create_fallback_generator(memoir_context)
