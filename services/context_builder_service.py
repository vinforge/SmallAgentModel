#!/usr/bin/env python3
"""
Context Builder Service
Builds comprehensive context for response generation by combining multiple sources.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ContextSection:
    """Represents a section of context with metadata."""
    title: str
    content: str
    source_type: str
    priority: int = 0
    max_length: Optional[int] = None

class ContextBuilderService:
    """
    Service for building comprehensive context from multiple sources
    including conversation history, documents, MEMOIR knowledge, and other sources.
    """
    
    def __init__(self):
        self.max_total_length = 4000  # Maximum total context length
        self.section_priorities = {
            'document_content': 10,
            'user_corrections': 9,
            'memoir_knowledge': 8,
            'conversation_history': 7,
            'general_context': 6,
            'web_content': 5
        }
    
    def build_context(self, query: str, context_data: Dict[str, Any]) -> List[str]:
        """
        Build comprehensive context from multiple sources.
        
        Args:
            query: The user query
            context_data: Dictionary containing various context sources
            
        Returns:
            List of context strings ready for prompt construction
        """
        logger.info(f"🏗️ Building context for query: '{query[:50]}...'")
        
        sections = []
        
        # Add conversation history section
        if self._should_include_conversation_history(context_data):
            conv_section = self._build_conversation_section(context_data)
            if conv_section:
                sections.append(conv_section)
        
        # Add document content section (highest priority for document queries)
        if context_data.get('document_query_detected', False):
            doc_section = self._build_document_section(query, context_data)
            if doc_section:
                sections.append(doc_section)
        
        # Add MEMOIR knowledge section
        memoir_section = self._build_memoir_section(context_data)
        if memoir_section:
            sections.append(memoir_section)
        
        # Add general context section
        general_section = self._build_general_context_section(context_data)
        if general_section:
            sections.append(general_section)
        
        # Sort sections by priority and build final context
        sections.sort(key=lambda x: x.priority, reverse=True)
        
        # Build final context with length management
        final_context = self._assemble_final_context(sections, query, context_data)
        
        logger.info(f"🏗️ Built context with {len(sections)} sections, {len(final_context)} final parts")
        return final_context
    
    def _should_include_conversation_history(self, context_data: Dict[str, Any]) -> bool:
        """Determine if conversation history should be included."""
        conversation_history = context_data.get('conversation_history', '')
        reduce_conversation_weight = context_data.get('reduce_conversation_weight', False)
        
        return (
            conversation_history and 
            conversation_history != "No recent conversation history." and 
            not reduce_conversation_weight
        )
    
    def _build_conversation_section(self, context_data: Dict[str, Any]) -> Optional[ContextSection]:
        """Build conversation history section."""
        conversation_history = context_data.get('conversation_history', '')
        
        if not conversation_history:
            return None
        
        return ContextSection(
            title="RECENT CONVERSATION HISTORY (Most recent first)",
            content=conversation_history,
            source_type='conversation_history',
            priority=self.section_priorities['conversation_history'],
            max_length=1500
        )
    
    def _build_document_section(self, query: str, context_data: Dict[str, Any]) -> Optional[ContextSection]:
        """Build document content section using search."""
        try:
            # Use smart search router if available
            use_smart_router = context_data.get('use_smart_search_router', False)
            
            if use_smart_router:
                from services.search_router import smart_search
                document_results = smart_search(query, max_results=5)
            else:
                # Fallback to legacy search
                from secure_streamlit_app import search_unified_memory
                document_results = search_unified_memory(query, max_results=5)
            
            logger.info(f"📄 Document search returned {len(document_results)} results")
            
            if not document_results:
                return None
            
            # Build document content using standardized result processing
            content_parts = []
            try:
                # Use the unified result processor
                from services.result_processor_service import get_result_processor_service
                processor = get_result_processor_service()

                standardized_results = processor.process_results(document_results)

                for i, std_result in enumerate(standardized_results[:3], 1):
                    content_parts.append(f"{i}. From {std_result.source}:")
                    content_parts.append(f"   {std_result.content[:800]}")  # Limit per result

                    # Log if we found target content
                    if any(term in std_result.content.lower() for term in ['chroma', 'sam story', 'ethan hayes']):
                        logger.info(f"📄 ✅ Found target document content in result {i}")

            except Exception as e:
                logger.warning(f"📄 ResultProcessor failed, using fallback: {e}")
                # Fallback to legacy processing
                for i, result in enumerate(document_results[:3], 1):
                    try:
                        content, source = self._extract_result_content(result)

                        if content and source:
                            content_parts.append(f"{i}. From {source}:")
                            content_parts.append(f"   {content[:800]}")

                            if any(term in content.lower() for term in ['chroma', 'sam story', 'ethan hayes']):
                                logger.info(f"📄 ✅ Found target document content in result {i}")

                    except Exception as e:
                        logger.warning(f"📄 Error processing document result {i}: {e}")
                        continue
            
            if not content_parts:
                return None
            
            return ContextSection(
                title="UPLOADED DOCUMENT CONTENT",
                content="\n".join(content_parts),
                source_type='document_content',
                priority=self.section_priorities['document_content'],
                max_length=2400  # Higher limit for documents
            )
            
        except Exception as e:
            logger.error(f"📄 Document section building failed: {e}")
            return None
    
    def _build_memoir_section(self, context_data: Dict[str, Any]) -> Optional[ContextSection]:
        """Build MEMOIR knowledge section."""
        memoir_context = context_data.get('memoir_context', {})
        relevant_edits = memoir_context.get('relevant_edits', [])
        
        if not relevant_edits:
            return None
        
        content_parts = ["Learned knowledge from previous corrections:"]
        
        for i, edit in enumerate(relevant_edits[:2]):
            if edit.get('edit_type') == 'correction':
                content_parts.append(f"• Previous correction: {edit.get('correction', '')}")
            elif edit.get('edit_type') == 'memory_correction':
                content = edit.get('content', '')[:200]
                content_parts.append(f"• Learned: {content}")
        
        return ContextSection(
            title="MEMOIR KNOWLEDGE",
            content="\n".join(content_parts),
            source_type='memoir_knowledge',
            priority=self.section_priorities['memoir_knowledge'],
            max_length=600
        )
    
    def _build_general_context_section(self, context_data: Dict[str, Any]) -> Optional[ContextSection]:
        """Build general context section."""
        sources = context_data.get('sources', [])
        
        if not sources:
            return None
        
        content_parts = ["Relevant context:"]
        
        for i, source in enumerate(sources[:3]):
            content = source.get('content', '')[:500]
            content_parts.append(f"{i+1}. {content}")
        
        return ContextSection(
            title="GENERAL CONTEXT",
            content="\n".join(content_parts),
            source_type='general_context',
            priority=self.section_priorities['general_context'],
            max_length=800
        )
    
    def _extract_result_content(self, result: Any) -> tuple:
        """Extract content and source from various result types."""
        try:
            # Debug logging
            logger.debug(f"Extracting content from result type: {type(result)}")

            # Handle RankedMemoryResult (Phase 3) - has content directly
            if hasattr(result, 'content') and hasattr(result, 'metadata'):
                source = result.metadata.get('source_path', result.metadata.get('source_name', 'Unknown'))
                return result.content, source

            # Handle MemorySearchResult with chunk attribute
            elif hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                return result.chunk.content, result.chunk.source

            # Handle MemorySearchResult with memory_chunk attribute
            elif hasattr(result, 'memory_chunk'):
                return result.memory_chunk.content, result.memory_chunk.source

            # Handle direct content objects
            elif hasattr(result, 'content'):
                return result.content, getattr(result, 'source', 'Unknown')

            else:
                logger.warning(f"Unknown result structure: {type(result)}")
                logger.warning(f"Available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                return None, None

        except Exception as e:
            logger.error(f"Error extracting result content from {type(result)}: {e}")
            logger.error(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            return None, None
    
    def _assemble_final_context(self, sections: List[ContextSection], query: str, context_data: Dict[str, Any]) -> List[str]:
        """Assemble final context with length management."""
        context_parts = []
        total_length = 0
        
        # Add query
        context_parts.append(f"Question: {query}")
        total_length += len(query) + 20
        
        # Add sections in priority order
        for section in sections:
            section_content = section.content
            
            # Apply section-specific length limits
            if section.max_length and len(section_content) > section.max_length:
                section_content = section_content[:section.max_length] + "..."
            
            # Check total length limit
            section_length = len(section.title) + len(section_content) + 20
            if total_length + section_length > self.max_total_length:
                logger.info(f"🏗️ Reached total length limit, truncating remaining sections")
                break
            
            # Add section
            context_parts.append(f"\n--- {section.title} ---")
            context_parts.append(section_content)
            context_parts.append(f"--- END OF {section.title} ---\n")
            
            total_length += section_length
        
        # Add appropriate instruction based on query type
        if context_data.get('document_query_detected', False):
            context_parts.append("\nPlease provide a comprehensive response based on the uploaded document content above. Use specific details, quotes, and information from the document to answer the question thoroughly.")
        else:
            context_parts.append("\nPlease provide a comprehensive, helpful response using the learned knowledge.")
        
        return context_parts

# Global instance for easy access
_context_builder_service = None

def get_context_builder_service() -> ContextBuilderService:
    """Get or create the global context builder service instance."""
    global _context_builder_service
    if _context_builder_service is None:
        _context_builder_service = ContextBuilderService()
    return _context_builder_service
