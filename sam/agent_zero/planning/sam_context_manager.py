"""
SAM Context Manager Module

Manages SAM-specific context for planning states, including document analysis results,
memory consolidation, conversation history, and other SAM capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentContext:
    """Context information about documents available to SAM."""
    
    documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Dictionary of document filename -> document metadata"""
    
    analysis_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Cached analysis results for documents"""
    
    processing_status: Dict[str, str] = field(default_factory=dict)
    """Processing status for each document (uploaded, analyzed, etc.)"""
    
    def add_document(self, filename: str, metadata: Dict[str, Any]):
        """Add a document to the context."""
        self.documents[filename] = metadata
        self.processing_status[filename] = "uploaded"
        logger.debug(f"Added document to context: {filename}")
    
    def set_analysis_result(self, filename: str, analysis_type: str, result: Any):
        """Store analysis result for a document."""
        if filename not in self.analysis_results:
            self.analysis_results[filename] = {}
        
        self.analysis_results[filename][analysis_type] = result
        logger.debug(f"Stored {analysis_type} analysis for {filename}")
    
    def get_analysis_result(self, filename: str, analysis_type: str) -> Optional[Any]:
        """Get cached analysis result for a document."""
        return self.analysis_results.get(filename, {}).get(analysis_type)
    
    def has_document(self, filename: str) -> bool:
        """Check if a document is available."""
        return filename in self.documents
    
    def get_document_list(self) -> List[str]:
        """Get list of available document filenames."""
        return list(self.documents.keys())
    
    def get_processed_documents(self) -> List[str]:
        """Get list of documents that have been processed."""
        return [
            filename for filename, status in self.processing_status.items()
            if status in ["analyzed", "processed"]
        ]


@dataclass
class MemoryContext:
    """Context information about SAM's memory system."""
    
    relevant_memories: List[Dict[str, Any]] = field(default_factory=list)
    """List of memories relevant to current task"""
    
    memory_queries: List[str] = field(default_factory=list)
    """History of memory queries performed"""
    
    consolidation_results: List[Dict[str, Any]] = field(default_factory=list)
    """Results of knowledge consolidation operations"""
    
    memory_types: Set[str] = field(default_factory=set)
    """Types of memories available (document, episodic, etc.)"""
    
    def add_relevant_memory(self, memory: Dict[str, Any]):
        """Add a relevant memory to the context."""
        self.relevant_memories.append(memory)
        if 'memory_type' in memory:
            self.memory_types.add(memory['memory_type'])
        logger.debug(f"Added relevant memory: {memory.get('id', 'unknown')}")
    
    def add_memory_query(self, query: str):
        """Record a memory query."""
        self.memory_queries.append(query)
        logger.debug(f"Recorded memory query: {query[:50]}...")
    
    def add_consolidation_result(self, result: Dict[str, Any]):
        """Add knowledge consolidation result."""
        self.consolidation_results.append(result)
        logger.debug(f"Added consolidation result: {result.get('id', 'unknown')}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory context."""
        return {
            'relevant_memories_count': len(self.relevant_memories),
            'memory_queries_count': len(self.memory_queries),
            'consolidation_results_count': len(self.consolidation_results),
            'memory_types': list(self.memory_types)
        }


@dataclass
class ConversationContext:
    """Context information about the current conversation."""
    
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    """Recent conversation messages"""
    
    user_intent: Optional[str] = None
    """Inferred user intent from conversation"""
    
    conversation_topics: List[str] = field(default_factory=list)
    """Topics discussed in conversation"""
    
    clarification_needed: List[str] = field(default_factory=list)
    """Areas where clarification is needed"""
    
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    """Session-level metadata"""
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.conversation_history.append(message)
        
        # Keep only recent messages (last 20)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        logger.debug(f"Added {role} message to conversation context")
    
    def set_user_intent(self, intent: str):
        """Set the inferred user intent."""
        self.user_intent = intent
        logger.debug(f"Set user intent: {intent}")
    
    def add_topic(self, topic: str):
        """Add a conversation topic."""
        if topic not in self.conversation_topics:
            self.conversation_topics.append(topic)
            logger.debug(f"Added conversation topic: {topic}")
    
    def add_clarification_need(self, clarification: str):
        """Add an area needing clarification."""
        if clarification not in self.clarification_needed:
            self.clarification_needed.append(clarification)
            logger.debug(f"Added clarification need: {clarification}")
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation messages."""
        return self.conversation_history[-count:] if self.conversation_history else []


class SAMContextManager:
    """
    Manages all SAM-specific context for planning operations.
    
    This class provides a unified interface for accessing and managing
    document context, memory context, conversation context, and other
    SAM-specific information needed for planning.
    """
    
    def __init__(self):
        """Initialize the context manager."""
        self.document_context = DocumentContext()
        self.memory_context = MemoryContext()
        self.conversation_context = ConversationContext()
        self._context_cache: Dict[str, Any] = {}
        self._last_updated = datetime.now()
    
    def update_from_session_state(self, session_state: Dict[str, Any]):
        """
        Update context from SAM's session state.
        
        Args:
            session_state: Current session state from SAM
        """
        # Update document context
        if 'uploaded_documents' in session_state:
            for filename, metadata in session_state['uploaded_documents'].items():
                self.document_context.add_document(filename, metadata)
        
        # Update conversation context
        if 'chat_history' in session_state:
            for message in session_state['chat_history']:
                if isinstance(message, dict) and 'role' in message and 'content' in message:
                    self.conversation_context.add_message(message['role'], message['content'])
        
        # Update memory context from session
        if 'relevant_memories' in session_state:
            for memory in session_state['relevant_memories']:
                self.memory_context.add_relevant_memory(memory)
        
        self._last_updated = datetime.now()
        logger.debug("Updated SAM context from session state")
    
    def get_planning_context(self) -> Dict[str, Any]:
        """
        Get context dictionary suitable for planning operations.
        
        Returns:
            Dictionary with all context information
        """
        return {
            'documents': self.document_context.documents if self.document_context.documents else None,
            'memory': self.memory_context.relevant_memories if self.memory_context.relevant_memories else None,
            'conversation': {
                'history': self.conversation_context.conversation_history,
                'intent': self.conversation_context.user_intent,
                'topics': self.conversation_context.conversation_topics
            } if self.conversation_context.conversation_history else None,
            'last_updated': self._last_updated.isoformat()
        }
    
    def get_context_for_tool(self, tool_name: str) -> Dict[str, Any]:
        """
        Get context specific to a particular tool.
        
        Args:
            tool_name: Name of the tool needing context
            
        Returns:
            Context dictionary tailored for the tool
        """
        context = {}
        
        # Document analysis tools need document context
        if 'document' in tool_name.lower():
            context['documents'] = self.document_context.documents
            context['analysis_results'] = self.document_context.analysis_results
        
        # Memory tools need memory context
        if 'memory' in tool_name.lower():
            context['memories'] = self.memory_context.relevant_memories
            context['memory_types'] = list(self.memory_context.memory_types)
        
        # Conversation tools need conversation context
        if 'conversation' in tool_name.lower() or 'clarify' in tool_name.lower():
            context['conversation'] = self.conversation_context.conversation_history
            context['user_intent'] = self.conversation_context.user_intent
            context['topics'] = self.conversation_context.conversation_topics
        
        return context
    
    def cache_computation_result(self, key: str, result: Any):
        """Cache a computation result for reuse."""
        self._context_cache[key] = {
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        logger.debug(f"Cached computation result: {key}")
    
    def get_cached_result(self, key: str, max_age_minutes: int = 30) -> Optional[Any]:
        """
        Get a cached computation result if it's still valid.
        
        Args:
            key: Cache key
            max_age_minutes: Maximum age of cached result in minutes
            
        Returns:
            Cached result or None if not found/expired
        """
        if key not in self._context_cache:
            return None
        
        cached_item = self._context_cache[key]
        cached_time = datetime.fromisoformat(cached_item['timestamp'])
        age_minutes = (datetime.now() - cached_time).total_seconds() / 60
        
        if age_minutes <= max_age_minutes:
            logger.debug(f"Retrieved cached result: {key}")
            return cached_item['result']
        else:
            # Remove expired cache entry
            del self._context_cache[key]
            logger.debug(f"Expired cached result removed: {key}")
            return None
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of all context information."""
        return {
            'documents': {
                'count': len(self.document_context.documents),
                'processed': len(self.document_context.get_processed_documents()),
                'filenames': self.document_context.get_document_list()
            },
            'memory': self.memory_context.get_memory_summary(),
            'conversation': {
                'message_count': len(self.conversation_context.conversation_history),
                'topics': self.conversation_context.conversation_topics,
                'user_intent': self.conversation_context.user_intent,
                'clarifications_needed': len(self.conversation_context.clarification_needed)
            },
            'cache': {
                'cached_items': len(self._context_cache),
                'last_updated': self._last_updated.isoformat()
            }
        }
    
    def clear_context(self):
        """Clear all context information."""
        self.document_context = DocumentContext()
        self.memory_context = MemoryContext()
        self.conversation_context = ConversationContext()
        self._context_cache.clear()
        self._last_updated = datetime.now()
        logger.info("Cleared all SAM context")
    
    def export_context(self) -> Dict[str, Any]:
        """Export context for serialization/storage."""
        return {
            'document_context': {
                'documents': self.document_context.documents,
                'analysis_results': self.document_context.analysis_results,
                'processing_status': self.document_context.processing_status
            },
            'memory_context': {
                'relevant_memories': self.memory_context.relevant_memories,
                'memory_queries': self.memory_context.memory_queries,
                'consolidation_results': self.memory_context.consolidation_results,
                'memory_types': list(self.memory_context.memory_types)
            },
            'conversation_context': {
                'conversation_history': self.conversation_context.conversation_history,
                'user_intent': self.conversation_context.user_intent,
                'conversation_topics': self.conversation_context.conversation_topics,
                'clarification_needed': self.conversation_context.clarification_needed,
                'session_metadata': self.conversation_context.session_metadata
            },
            'last_updated': self._last_updated.isoformat()
        }
    
    def import_context(self, context_data: Dict[str, Any]):
        """Import context from serialized data."""
        if 'document_context' in context_data:
            doc_data = context_data['document_context']
            self.document_context.documents = doc_data.get('documents', {})
            self.document_context.analysis_results = doc_data.get('analysis_results', {})
            self.document_context.processing_status = doc_data.get('processing_status', {})
        
        if 'memory_context' in context_data:
            mem_data = context_data['memory_context']
            self.memory_context.relevant_memories = mem_data.get('relevant_memories', [])
            self.memory_context.memory_queries = mem_data.get('memory_queries', [])
            self.memory_context.consolidation_results = mem_data.get('consolidation_results', [])
            self.memory_context.memory_types = set(mem_data.get('memory_types', []))
        
        if 'conversation_context' in context_data:
            conv_data = context_data['conversation_context']
            self.conversation_context.conversation_history = conv_data.get('conversation_history', [])
            self.conversation_context.user_intent = conv_data.get('user_intent')
            self.conversation_context.conversation_topics = conv_data.get('conversation_topics', [])
            self.conversation_context.clarification_needed = conv_data.get('clarification_needed', [])
            self.conversation_context.session_metadata = conv_data.get('session_metadata', {})
        
        self._last_updated = datetime.now()
        logger.info("Imported SAM context from data")
