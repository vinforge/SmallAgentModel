#!/usr/bin/env python3
"""
SAM Persona Memory Retrieval System - Task 30 Phase 2
=====================================================

Implements persona memory retrieval for Post Persona Alignment (PPA).
Retrieves user preferences, conversation summaries, and learned facts
from long-term memory to enable consistent persona across sessions.

Part of Task 30: Advanced Conversational Coherence Engine
Author: SAM Development Team
Version: 1.0.0
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class PersonaMemory:
    """Represents a piece of persona-related memory."""
    memory_id: str
    content: str
    memory_type: str  # 'preference', 'fact', 'conversation_summary', 'correction'
    user_id: str
    confidence: float
    last_accessed: str
    importance_score: float
    metadata: Dict[str, Any]

class PersonaMemoryRetriever:
    """
    Retrieves persona-related memories for response refinement.
    
    Features:
    - User preference retrieval
    - Conversation summary extraction
    - Learned fact identification
    - Importance-based ranking
    - Temporal relevance scoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the persona memory retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.PersonaMemoryRetriever")
        
        # Default configuration
        self.config = {
            'max_persona_memories': 5,
            'preference_weight': 1.0,
            'fact_weight': 0.8,
            'summary_weight': 0.6,
            'recency_weight': 0.3,
            'similarity_threshold': 0.4,
            'temporal_decay_days': 30
        }
        
        if config:
            self.config.update(config)
        
        self.logger.info("PersonaMemoryRetriever initialized")
    
    def retrieve_persona_memories(self, query_text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve relevant persona memories for the given query.
        
        Args:
            query_text: The draft response or query text to analyze
            user_id: User identifier for personalized retrieval
            
        Returns:
            Dictionary containing retrieved persona memories
        """
        try:
            persona_context = {
                'preferences': [],
                'learned_facts': [],
                'conversation_summaries': [],
                'corrections': [],
                'total_memories': 0,
                'retrieval_confidence': 0.0
            }
            
            # Get memory stores
            memory_stores = self._get_memory_stores()
            if not memory_stores:
                self.logger.warning("No memory stores available for persona retrieval")
                return persona_context
            
            # Extract key concepts from query for targeted retrieval
            key_concepts = self._extract_key_concepts(query_text)
            
            # Retrieve from each memory store
            all_memories = []
            for store_name, store in memory_stores.items():
                try:
                    memories = self._retrieve_from_store(store, key_concepts, user_id)
                    all_memories.extend(memories)
                    self.logger.debug(f"Retrieved {len(memories)} memories from {store_name}")
                except Exception as e:
                    self.logger.warning(f"Error retrieving from {store_name}: {e}")
            
            # Classify and rank memories
            classified_memories = self._classify_memories(all_memories, query_text)
            
            # Select top memories by category
            persona_context = self._select_top_memories(classified_memories)
            
            # Calculate overall confidence
            persona_context['retrieval_confidence'] = self._calculate_confidence(persona_context)
            
            self.logger.info(f"Retrieved {persona_context['total_memories']} persona memories for user {user_id}")
            return persona_context
            
        except Exception as e:
            self.logger.error(f"Error retrieving persona memories: {e}")
            return {
                'preferences': [],
                'learned_facts': [],
                'conversation_summaries': [],
                'corrections': [],
                'total_memories': 0,
                'retrieval_confidence': 0.0,
                'error': str(e)
            }
    
    def _get_memory_stores(self) -> Dict[str, Any]:
        """Get available memory stores for retrieval."""
        stores = {}
        
        try:
            # Try to get secure memory store
            import streamlit as st
            if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                stores['secure'] = st.session_state.secure_memory_store
        except:
            pass
        
        try:
            # Try to get regular memory store
            from memory.memory_vectorstore import get_memory_store
            stores['regular'] = get_memory_store()
        except:
            pass
        
        try:
            # Try to get user-taught knowledge from session state
            import streamlit as st
            if hasattr(st.session_state, 'user_taught_knowledge') and st.session_state.user_taught_knowledge:
                stores['session'] = st.session_state.user_taught_knowledge
        except:
            pass
        
        return stores
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for targeted retrieval."""
        # Simple keyword extraction - could be enhanced with NLP
        text_lower = text.lower()
        
        # Common persona-related keywords
        persona_keywords = [
            'prefer', 'like', 'dislike', 'usually', 'always', 'never',
            'style', 'approach', 'method', 'way', 'format', 'language',
            'detailed', 'brief', 'technical', 'simple', 'formal', 'casual'
        ]
        
        # Extract words that might indicate preferences or facts
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Filter for relevant concepts
        concepts = []
        for word in words:
            if len(word) > 3 and (word in persona_keywords or word.endswith('ing') or word.endswith('ed')):
                concepts.append(word)
        
        # Add domain-specific terms
        domain_terms = ['python', 'programming', 'ai', 'machine learning', 'data', 'analysis']
        for term in domain_terms:
            if term in text_lower:
                concepts.append(term)
        
        return list(set(concepts))[:10]  # Limit to top 10 concepts
    
    def _retrieve_from_store(self, store, key_concepts: List[str], user_id: Optional[str]) -> List[PersonaMemory]:
        """Retrieve memories from a specific store."""
        memories = []
        
        try:
            if hasattr(store, 'search_memories'):
                # Vector store search
                for concept in key_concepts[:3]:  # Limit searches
                    try:
                        results = store.search_memories(concept, max_results=5)
                        for result in results:
                            memory = self._convert_to_persona_memory(result, 'vector_search')
                            if memory and (not user_id or self._is_user_relevant(memory, user_id)):
                                memories.append(memory)
                    except Exception as e:
                        self.logger.debug(f"Search failed for concept '{concept}': {e}")
            
            elif isinstance(store, list):
                # Session state knowledge list
                for item in store:
                    memory = self._convert_to_persona_memory(item, 'session_knowledge')
                    if memory:
                        memories.append(memory)
            
        except Exception as e:
            self.logger.warning(f"Error retrieving from store: {e}")
        
        return memories
    
    def _convert_to_persona_memory(self, result, source_type: str) -> Optional[PersonaMemory]:
        """Convert search result to PersonaMemory object."""
        try:
            if hasattr(result, 'content'):
                # Vector store result
                content = result.content
                memory_id = getattr(result, 'chunk_id', f"{source_type}_{hash(content)}")
                metadata = getattr(result, 'metadata', {})
            elif isinstance(result, dict):
                # Dictionary result (session knowledge)
                content = result.get('content', '')
                memory_id = result.get('id', f"{source_type}_{hash(content)}")
                metadata = result
            else:
                return None
            
            # Determine memory type based on content
            memory_type = self._classify_memory_type(content)
            
            return PersonaMemory(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                user_id=metadata.get('user_id', 'unknown'),
                confidence=0.7,  # Default confidence
                last_accessed=datetime.now().isoformat(),
                importance_score=metadata.get('importance', 0.5),
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.debug(f"Error converting result to persona memory: {e}")
            return None
    
    def _classify_memory_type(self, content: str) -> str:
        """Classify memory type based on content."""
        content_lower = content.lower()
        
        # Preference indicators
        if any(word in content_lower for word in ['prefer', 'like', 'usually', 'always', 'style']):
            return 'preference'
        
        # Correction indicators
        if any(word in content_lower for word in ['actually', 'correction', 'wrong', 'should be']):
            return 'correction'
        
        # Fact indicators (user-taught information)
        if any(word in content_lower for word in ['secret', 'capital', 'new information', 'remember']):
            return 'fact'
        
        # Conversation summary indicators
        if any(word in content_lower for word in ['discussed', 'talked about', 'conversation', 'session']):
            return 'conversation_summary'
        
        return 'fact'  # Default to fact
    
    def _is_user_relevant(self, memory: PersonaMemory, user_id: str) -> bool:
        """Check if memory is relevant to the specific user."""
        # Check if memory is user-specific
        if memory.user_id == user_id:
            return True
        
        # Check if memory contains user-specific information
        if f"user:{user_id}" in memory.metadata.get('tags', []):
            return True
        
        # For now, include general memories too
        return True
    
    def _classify_memories(self, memories: List[PersonaMemory], query_text: str) -> Dict[str, List[PersonaMemory]]:
        """Classify memories by type and calculate relevance scores."""
        classified = {
            'preferences': [],
            'learned_facts': [],
            'conversation_summaries': [],
            'corrections': []
        }
        
        for memory in memories:
            # Calculate relevance score
            relevance_score = self._calculate_relevance(memory, query_text)
            memory.confidence = relevance_score
            
            # Classify by type
            if memory.memory_type == 'preference':
                classified['preferences'].append(memory)
            elif memory.memory_type == 'fact':
                classified['learned_facts'].append(memory)
            elif memory.memory_type == 'conversation_summary':
                classified['conversation_summaries'].append(memory)
            elif memory.memory_type == 'correction':
                classified['corrections'].append(memory)
        
        # Sort each category by confidence
        for category in classified:
            classified[category].sort(key=lambda m: m.confidence, reverse=True)
        
        return classified
    
    def _calculate_relevance(self, memory: PersonaMemory, query_text: str) -> float:
        """Calculate relevance score for a memory."""
        score = 0.0
        
        # Base importance score
        score += memory.importance_score * 0.4
        
        # Content similarity (simple keyword matching)
        memory_words = set(memory.content.lower().split())
        query_words = set(query_text.lower().split())
        common_words = memory_words.intersection(query_words)
        if memory_words:
            similarity = len(common_words) / len(memory_words)
            score += similarity * 0.4
        
        # Temporal relevance (more recent = higher score)
        try:
            last_accessed = datetime.fromisoformat(memory.last_accessed.replace('Z', '+00:00'))
            days_ago = (datetime.now() - last_accessed).days
            temporal_score = max(0, 1 - (days_ago / self.config['temporal_decay_days']))
            score += temporal_score * self.config['recency_weight']
        except:
            pass
        
        # Type-specific weighting
        type_weights = {
            'preference': self.config['preference_weight'],
            'fact': self.config['fact_weight'],
            'conversation_summary': self.config['summary_weight'],
            'correction': 1.0  # Corrections are always important
        }
        score *= type_weights.get(memory.memory_type, 0.5)
        
        return min(1.0, score)
    
    def _select_top_memories(self, classified_memories: Dict[str, List[PersonaMemory]]) -> Dict[str, Any]:
        """Select top memories from each category."""
        max_per_category = max(1, self.config['max_persona_memories'] // 4)
        
        result = {
            'preferences': classified_memories['preferences'][:max_per_category],
            'learned_facts': classified_memories['learned_facts'][:max_per_category],
            'conversation_summaries': classified_memories['conversation_summaries'][:max_per_category],
            'corrections': classified_memories['corrections'][:max_per_category],
            'total_memories': 0
        }
        
        # Count total memories
        for category in ['preferences', 'learned_facts', 'conversation_summaries', 'corrections']:
            result['total_memories'] += len(result[category])
        
        return result
    
    def _calculate_confidence(self, persona_context: Dict[str, Any]) -> float:
        """Calculate overall confidence in persona retrieval."""
        if persona_context['total_memories'] == 0:
            return 0.0
        
        total_confidence = 0.0
        total_memories = 0
        
        for category in ['preferences', 'learned_facts', 'conversation_summaries', 'corrections']:
            for memory in persona_context[category]:
                total_confidence += memory.confidence
                total_memories += 1
        
        return total_confidence / total_memories if total_memories > 0 else 0.0

# Global persona memory retriever instance
_persona_retriever: Optional[PersonaMemoryRetriever] = None

def get_persona_memory_retriever(config: Optional[Dict[str, Any]] = None) -> PersonaMemoryRetriever:
    """Get the global persona memory retriever instance."""
    global _persona_retriever
    
    if _persona_retriever is None:
        _persona_retriever = PersonaMemoryRetriever(config)
    return _persona_retriever
