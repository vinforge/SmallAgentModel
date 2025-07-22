"""
Memory-Driven Reasoning Engine for SAM
Automatic recall and context injection for enhanced reasoning with long-term memory.

Sprint 11 Task 4: Memory-Driven Reasoning
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from .memory_vectorstore import MemoryVectorStore, MemoryType, MemorySearchResult, get_memory_store

logger = logging.getLogger(__name__)

@dataclass
class MemoryContext:
    """Memory context injected into reasoning."""
    query: str
    recalled_memories: List[MemorySearchResult]
    context_summary: str
    relevance_score: float
    memory_count: int
    oldest_memory_date: Optional[str]
    newest_memory_date: Optional[str]

@dataclass
class ReasoningSession:
    """A reasoning session with memory integration."""
    session_id: str
    user_id: str
    query: str
    memory_context: MemoryContext
    reasoning_steps: List[Dict[str, Any]]
    final_response: str
    confidence_score: float
    memory_updated: bool
    created_at: str
    completed_at: Optional[str]

class MemoryDrivenReasoningEngine:
    """
    Reasoning engine that automatically recalls and integrates long-term memory.
    """
    
    def __init__(self, memory_store: MemoryVectorStore = None):
        """
        Initialize the memory-driven reasoning engine.
        
        Args:
            memory_store: Memory vector store instance
        """
        self.memory_store = memory_store or get_memory_store()
        
        # Configuration
        self.config = {
            'max_recalled_memories': 5,
            'memory_relevance_threshold': 0.6,
            'auto_memory_update': True,
            'memory_importance_threshold': 0.7,
            'context_window_size': 2000,
            'enable_memory_injection': True
        }
        
        # Session tracking
        self.active_sessions: Dict[str, ReasoningSession] = {}
        
        logger.info("Memory-driven reasoning engine initialized")
    
    def reason_with_memory(self, query: str, user_id: str, session_id: str = None,
                          context: Dict[str, Any] = None) -> ReasoningSession:
        """
        Perform reasoning with automatic memory recall and injection.
        
        Args:
            query: User query
            user_id: User ID
            session_id: Optional session ID
            context: Additional context
            
        Returns:
            ReasoningSession with memory-enhanced results
        """
        try:
            if not session_id:
                import uuid
                session_id = f"session_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Starting memory-driven reasoning: {query[:50]}...")
            
            # Step 1: Recall relevant memories
            memory_context = self._recall_relevant_memories(query, user_id)
            
            # Step 2: Perform reasoning with memory context
            reasoning_steps = self._perform_memory_enhanced_reasoning(query, memory_context, context)
            
            # Step 3: Generate final response
            final_response = self._generate_memory_aware_response(query, memory_context, reasoning_steps)
            
            # Step 4: Calculate confidence
            confidence_score = self._calculate_reasoning_confidence(memory_context, reasoning_steps)
            
            # Step 5: Update memory if significant
            memory_updated = False
            if self.config['auto_memory_update']:
                memory_updated = self._update_memory_from_reasoning(query, final_response, confidence_score, user_id)
            
            # Create reasoning session
            session = ReasoningSession(
                session_id=session_id,
                user_id=user_id,
                query=query,
                memory_context=memory_context,
                reasoning_steps=reasoning_steps,
                final_response=final_response,
                confidence_score=confidence_score,
                memory_updated=memory_updated,
                created_at=datetime.now().isoformat(),
                completed_at=datetime.now().isoformat()
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Memory-driven reasoning completed: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error in memory-driven reasoning: {e}")
            raise
    
    def add_important_memory(self, content: str, memory_type: MemoryType, source: str,
                           user_id: str, tags: List[str] = None, 
                           importance_score: float = 0.8) -> str:
        """
        Add an important memory that should be remembered.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            source: Source of the memory
            user_id: User ID
            tags: Optional tags
            importance_score: Importance score
            
        Returns:
            Memory chunk ID
        """
        try:
            # Add user-specific tag
            tags = tags or []
            if f"user:{user_id}" not in tags:
                tags.append(f"user:{user_id}")
            
            # Add importance tag
            if importance_score >= 0.8:
                tags.append("high_importance")
            elif importance_score >= 0.6:
                tags.append("medium_importance")
            
            chunk_id = self.memory_store.add_memory(
                content=content,
                memory_type=memory_type,
                source=source,
                tags=tags,
                importance_score=importance_score,
                metadata={'user_id': user_id, 'marked_important': True}
            )
            
            logger.info(f"Added important memory: {chunk_id}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error adding important memory: {e}")
            raise
    
    def forget_memory(self, chunk_id: str, user_id: str) -> bool:
        """
        Remove a memory from the store.
        
        Args:
            chunk_id: Memory chunk ID
            user_id: User ID (for authorization)
            
        Returns:
            True if successful
        """
        try:
            # Get memory to check ownership
            memory = self.memory_store.get_memory(chunk_id)
            if not memory:
                logger.error(f"Memory not found: {chunk_id}")
                return False
            
            # Check if user has permission to delete
            if memory.metadata.get('user_id') != user_id:
                logger.error(f"User {user_id} not authorized to delete memory {chunk_id}")
                return False
            
            # Delete memory
            success = self.memory_store.delete_memory(chunk_id)
            
            if success:
                logger.info(f"Forgot memory: {chunk_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error forgetting memory: {e}")
            return False
    
    def get_memory_summary(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get a summary of memories in the store.
        
        Args:
            user_id: Optional user ID filter
            
        Returns:
            Memory summary
        """
        try:
            # Get overall stats
            stats = self.memory_store.get_memory_stats()
            
            # Filter by user if specified
            user_memories = []
            if user_id:
                for chunk in self.memory_store.memory_chunks.values():
                    if chunk.metadata.get('user_id') == user_id:
                        user_memories.append(chunk)
            
            summary = {
                'total_memories': stats['total_memories'],
                'user_memories': len(user_memories) if user_id else None,
                'memory_types': stats['memory_types'],
                'storage_size_mb': stats['total_size_mb'],
                'oldest_memory': stats['oldest_memory'],
                'newest_memory': stats['newest_memory'],
                'store_type': stats['store_type']
            }
            
            if user_id and user_memories:
                # User-specific stats
                user_types = {}
                for memory in user_memories:
                    mem_type = memory.memory_type.value
                    user_types[mem_type] = user_types.get(mem_type, 0) + 1
                
                summary['user_memory_types'] = user_types
                summary['user_oldest_memory'] = min(user_memories, key=lambda m: m.timestamp).timestamp
                summary['user_newest_memory'] = max(user_memories, key=lambda m: m.timestamp).timestamp
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {'error': str(e)}
    
    def search_memories(self, query: str, user_id: str = None, 
                       memory_types: List[MemoryType] = None,
                       max_results: int = 10) -> List[MemorySearchResult]:
        """
        Search memories with optional user filtering.
        
        Args:
            query: Search query
            user_id: Optional user ID filter
            memory_types: Optional memory type filter
            max_results: Maximum results
            
        Returns:
            List of memory search results
        """
        try:
            # Search all memories
            results = self.memory_store.search_memories(
                query=query,
                max_results=max_results * 2,  # Get more to filter
                memory_types=memory_types
            )
            
            # Filter by user if specified
            if user_id:
                filtered_results = []
                for result in results:
                    if result.chunk.metadata.get('user_id') == user_id:
                        filtered_results.append(result)
                        if len(filtered_results) >= max_results:
                            break
                results = filtered_results
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def _recall_relevant_memories(self, query: str, user_id: str) -> MemoryContext:
        """Recall memories relevant to the query."""
        try:
            # Phase 3.2: Use enhanced search for memory reasoning
            try:
                if hasattr(self.memory_store, 'enhanced_search_memories'):
                    # Enhanced search with hybrid ranking for reasoning
                    memory_results = self.memory_store.enhanced_search_memories(
                        query=query,
                        max_results=self.config['max_recalled_memories'],
                        initial_candidates=self.config['max_recalled_memories'] * 2
                    )
                    logger.info(f"Enhanced memory reasoning search: {len(memory_results)} results")
                else:
                    # Fallback to regular search
                    memory_results = self.search_memories(
                        query=query,
                        user_id=user_id,
                        max_results=self.config['max_recalled_memories']
                    )
            except Exception as e:
                logger.warning(f"Enhanced search failed in memory reasoning, using fallback: {e}")
                memory_results = self.search_memories(
                    query=query,
                    user_id=user_id,
                    max_results=self.config['max_recalled_memories']
                )
            
            # Filter by relevance threshold
            relevant_memories = [
                result for result in memory_results
                if result.similarity_score >= self.config['memory_relevance_threshold']
            ]
            
            # Create context summary
            context_summary = self._create_memory_context_summary(relevant_memories)
            
            # Calculate overall relevance
            relevance_score = 0.0
            if relevant_memories:
                relevance_score = sum(r.similarity_score for r in relevant_memories) / len(relevant_memories)
            
            # Get date range
            oldest_date = None
            newest_date = None
            if relevant_memories:
                dates = [r.chunk.timestamp for r in relevant_memories]
                oldest_date = min(dates)
                newest_date = max(dates)
            
            memory_context = MemoryContext(
                query=query,
                recalled_memories=relevant_memories,
                context_summary=context_summary,
                relevance_score=relevance_score,
                memory_count=len(relevant_memories),
                oldest_memory_date=oldest_date,
                newest_memory_date=newest_date
            )
            
            logger.info(f"Recalled {len(relevant_memories)} relevant memories")
            return memory_context
            
        except Exception as e:
            logger.error(f"Error recalling memories: {e}")
            return MemoryContext(
                query=query,
                recalled_memories=[],
                context_summary="No relevant memories found",
                relevance_score=0.0,
                memory_count=0,
                oldest_memory_date=None,
                newest_memory_date=None
            )
    
    def _create_memory_context_summary(self, memory_results: List[MemorySearchResult]) -> str:
        """Create a summary of recalled memories for context injection."""
        try:
            if not memory_results:
                return "No relevant memories found."
            
            summary_parts = [
                f"Relevant memories ({len(memory_results)} found):",
                ""
            ]
            
            for i, result in enumerate(memory_results, 1):
                memory = result.chunk
                
                # Format memory entry
                memory_entry = [
                    f"{i}. [{memory.memory_type.value.title()}] {memory.source}",
                    f"   Content: {memory.content[:150]}{'...' if len(memory.content) > 150 else ''}",
                    f"   Date: {memory.timestamp[:10]}",
                    f"   Relevance: {result.similarity_score:.1%}",
                    f"   Tags: {', '.join(memory.tags[:3])}{'...' if len(memory.tags) > 3 else ''}",
                    ""
                ]
                
                summary_parts.extend(memory_entry)
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating memory context summary: {e}")
            return "Error creating memory summary."
    
    def _perform_memory_enhanced_reasoning(self, query: str, memory_context: MemoryContext,
                                         context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform reasoning enhanced with memory context."""
        try:
            reasoning_steps = []
            
            # Step 1: Memory integration
            if memory_context.memory_count > 0:
                reasoning_steps.append({
                    'step': 'memory_integration',
                    'description': f'Integrated {memory_context.memory_count} relevant memories',
                    'details': {
                        'memory_count': memory_context.memory_count,
                        'relevance_score': memory_context.relevance_score,
                        'date_range': f"{memory_context.oldest_memory_date} to {memory_context.newest_memory_date}"
                    }
                })
            
            # Step 2: Context analysis
            reasoning_steps.append({
                'step': 'context_analysis',
                'description': 'Analyzed query in context of available memories',
                'details': {
                    'query_length': len(query),
                    'memory_enhanced': memory_context.memory_count > 0,
                    'additional_context': context is not None
                }
            })
            
            # Step 3: Memory-informed reasoning
            if memory_context.memory_count > 0:
                reasoning_steps.append({
                    'step': 'memory_informed_reasoning',
                    'description': 'Applied insights from recalled memories to reasoning process',
                    'details': {
                        'memory_types': list(set(r.chunk.memory_type.value for r in memory_context.recalled_memories)),
                        'key_insights': self._extract_key_insights(memory_context.recalled_memories)
                    }
                })
            
            return reasoning_steps
            
        except Exception as e:
            logger.error(f"Error in memory-enhanced reasoning: {e}")
            return []
    
    def _generate_memory_aware_response(self, query: str, memory_context: MemoryContext,
                                      reasoning_steps: List[Dict[str, Any]]) -> str:
        """Generate response that incorporates memory context."""
        try:
            response_parts = []
            
            # Main response based on query
            if memory_context.memory_count > 0:
                response_parts.append(f"Based on your query and {memory_context.memory_count} relevant memories:")
                response_parts.append("")
                
                # Incorporate memory insights
                for result in memory_context.recalled_memories[:3]:  # Top 3 memories
                    memory = result.chunk
                    response_parts.append(f"• From {memory.source} ({memory.timestamp[:10]}): {memory.content[:100]}...")
                
                response_parts.append("")
                response_parts.append("Considering this historical context, here's my analysis:")
            else:
                response_parts.append("Based on your query:")
            
            response_parts.append("")
            response_parts.append(f"Query: {query}")
            response_parts.append("")
            response_parts.append("Analysis: This appears to be a request that would benefit from comprehensive reasoning.")
            
            if memory_context.memory_count > 0:
                response_parts.append(f"The recalled memories provide relevant context with {memory_context.relevance_score:.1%} average relevance.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating memory-aware response: {e}")
            return f"Error generating response for query: {query}"
    
    def _calculate_reasoning_confidence(self, memory_context: MemoryContext,
                                      reasoning_steps: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for reasoning."""
        try:
            base_confidence = 0.6
            
            # Boost confidence with relevant memories
            memory_boost = min(0.3, memory_context.relevance_score * 0.3)
            
            # Boost confidence with reasoning steps
            reasoning_boost = min(0.1, len(reasoning_steps) * 0.03)
            
            total_confidence = base_confidence + memory_boost + reasoning_boost
            
            return min(1.0, total_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating reasoning confidence: {e}")
            return 0.5
    
    def _update_memory_from_reasoning(self, query: str, response: str, confidence: float,
                                    user_id: str) -> bool:
        """Update memory with reasoning results if significant."""
        try:
            # Only store high-confidence reasoning
            if confidence < self.config['memory_importance_threshold']:
                return False
            
            # Create memory content
            memory_content = f"Query: {query}\n\nResponse: {response}"
            
            # Add to memory
            chunk_id = self.memory_store.add_memory(
                content=memory_content,
                memory_type=MemoryType.REASONING,
                source="sam_reasoning_session",
                tags=[f"user:{user_id}", "reasoning", "auto_generated"],
                importance_score=confidence,
                metadata={
                    'user_id': user_id,
                    'confidence': confidence,
                    'auto_generated': True
                }
            )
            
            logger.info(f"Updated memory with reasoning: {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory from reasoning: {e}")
            return False
    
    def _extract_key_insights(self, memory_results: List[MemorySearchResult]) -> List[str]:
        """Extract key insights from recalled memories."""
        try:
            insights = []
            
            for result in memory_results:
                memory = result.chunk
                
                # Extract key phrases (simplified)
                content_words = memory.content.lower().split()
                if len(content_words) > 10:
                    # Take first sentence or first 20 words
                    first_sentence = memory.content.split('.')[0]
                    if len(first_sentence) < 100:
                        insights.append(first_sentence.strip())
                    else:
                        insights.append(' '.join(content_words[:20]) + '...')
            
            return insights[:5]  # Top 5 insights
            
        except Exception as e:
            logger.error(f"Error extracting key insights: {e}")
            return []

# Global memory-driven reasoning engine instance
_memory_reasoning_engine = None

def get_memory_reasoning_engine(memory_store: MemoryVectorStore = None) -> MemoryDrivenReasoningEngine:
    """Get or create a global memory-driven reasoning engine instance."""
    global _memory_reasoning_engine
    
    if _memory_reasoning_engine is None:
        _memory_reasoning_engine = MemoryDrivenReasoningEngine(memory_store=memory_store)
    
    return _memory_reasoning_engine
