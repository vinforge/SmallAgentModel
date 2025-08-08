"""
MemoryTool - Explicit Memory Interface for Agent Zero
====================================================

This tool provides Agent Zero with a formal, auditable interface for interacting
with SAM's internal memory stores. It makes the agent's reasoning more transparent
and intelligent by providing explicit memory operations.

The MemoryTool serves as a unified interface to SAM's various memory stores,
intelligently routing requests to the appropriate existing backend without
having its own database.

Core Functions:
- search_conversations(): Search through archived conversation content
- search_knowledge_base(): Perform RAG search on long-term knowledge base  
- add_to_knowledge_base(): Add verified information to knowledge base

Part of SAM's mem0-inspired Memory Augmentation (Task 33, Phase 1)
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from ..uif import SAM_UIF, UIFStatus
from .base import BaseSkillModule, SkillExecutionError

logger = logging.getLogger(__name__)

class MemoryTool(BaseSkillModule):
    """
    Explicit memory interface tool for Agent Zero.
    
    Provides a unified, auditable interface to SAM's memory systems:
    - Conversation search via Conversational Intelligence Engine
    - Knowledge base search via vector store RAG
    - Knowledge base updates for closing research loops
    
    This tool makes Agent Zero's memory interactions explicit and transparent,
    improving reasoning quality and enabling better decision-making.
    """
    
    skill_name = "MemoryTool"
    skill_version = "1.0.0"
    skill_description = "Explicit memory interface for searching conversations and knowledge base"
    skill_category = "memory"
    
    # Dependency declarations
    required_inputs = ["memory_operation", "query"]
    optional_inputs = ["search_context", "max_results", "thread_ids", "content_metadata"]
    output_keys = ["memory_results", "search_confidence", "operation_status"]
    
    def __init__(self):
        """Initialize the MemoryTool with connections to SAM's memory systems."""
        super().__init__()
        
        # Memory system connections (initialized lazily)
        self._memory_store = None
        self._integrated_memory = None
        self._conversation_engine = None
        
        # Configuration
        self.config = {
            'max_conversation_results': 20,
            'max_knowledge_results': 15,
            'conversation_search_context': 3,  # surrounding messages
            'knowledge_search_threshold': 0.7,
            'enable_cross_memory_search': True
        }
        
        self.logger.info("MemoryTool initialized")
    
    def _initialize_memory_systems(self) -> None:
        """Initialize connections to SAM's memory systems."""
        try:
            # Import and initialize memory store
            from memory.memory_vectorstore import get_memory_store
            from memory.integrated_memory import get_integrated_memory
            
            self._memory_store = get_memory_store()
            self._integrated_memory = get_integrated_memory()
            
            # Import conversation intelligence engine
            from sam.conversation.contextual_relevance import ContextualRelevanceEngine
            self._conversation_engine = ContextualRelevanceEngine()
            
            self.logger.info("Memory systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory systems: {e}")
            raise SkillExecutionError(f"Memory system initialization failed: {e}")
    
    async def execute(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Execute memory operations based on the requested operation type.
        
        Supported operations:
        - search_conversations: Search archived conversation content
        - search_knowledge_base: Search long-term knowledge base
        - add_to_knowledge_base: Add new information to knowledge base
        - cross_memory_search: Search both conversations and knowledge base
        """
        try:
            # Initialize memory systems if needed
            if self._memory_store is None:
                self._initialize_memory_systems()
            
            # Extract operation parameters
            memory_operation = uif.get_data("memory_operation")
            query = uif.get_data("query")
            
            if not memory_operation or not query:
                raise SkillExecutionError("memory_operation and query are required")
            
            self.logger.info(f"Executing memory operation: {memory_operation} with query: {query[:100]}...")
            
            # Route to appropriate memory operation
            if memory_operation == "search_conversations":
                result = await self._search_conversations(uif)
            elif memory_operation == "search_knowledge_base":
                result = await self._search_knowledge_base(uif)
            elif memory_operation == "add_to_knowledge_base":
                result = await self._add_to_knowledge_base(uif)
            elif memory_operation == "cross_memory_search":
                result = await self._cross_memory_search(uif)
            else:
                raise SkillExecutionError(f"Unknown memory operation: {memory_operation}")
            
            # Update UIF with results
            uif.add_data("memory_results", result.get("results", []))
            uif.add_data("search_confidence", result.get("confidence", 0.0))
            uif.add_data("operation_status", result.get("status", "completed"))
            uif.add_data("operation_metadata", result.get("metadata", {}))
            
            uif.status = UIFStatus.COMPLETED
            self.logger.info(f"Memory operation {memory_operation} completed successfully")
            
            return uif
            
        except Exception as e:
            self.logger.error(f"MemoryTool execution failed: {e}")
            uif.add_data("error", str(e))
            uif.add_data("operation_status", "failed")
            uif.status = UIFStatus.FAILED
            return uif
    
    async def _search_conversations(self, uif: SAM_UIF) -> Dict[str, Any]:
        """Search through archived conversation content using Task 31 Advanced Search."""
        try:
            query = uif.get_data("query")
            max_results = uif.get_data("max_results", self.config['max_conversation_results'])
            thread_ids = uif.get_data("thread_ids")  # Optional specific threads
            
            self.logger.info(f"Searching conversations for: {query}")
            
            # Use the Advanced Search functionality from Task 31
            search_results = self._conversation_engine.search_within_threads(
                query=query,
                thread_ids=thread_ids,
                limit=max_results
            )
            
            # Calculate overall confidence based on relevance scores
            if search_results:
                avg_relevance = sum(r.get('relevance_score', 0) for r in search_results) / len(search_results)
                confidence = min(avg_relevance, 1.0)
            else:
                confidence = 0.0
            
            # Format results for Agent Zero
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    'type': 'conversation',
                    'thread_id': result.get('thread_id'),
                    'thread_title': result.get('thread_title'),
                    'content': result.get('message_content'),
                    'relevance_score': result.get('relevance_score', 0),
                    'context_messages': result.get('context_messages', []),
                    'source': 'archived_conversations'
                })
            
            return {
                'results': formatted_results,
                'confidence': confidence,
                'status': 'completed',
                'metadata': {
                    'search_type': 'conversations',
                    'total_results': len(formatted_results),
                    'query': query,
                    'searched_threads': len(thread_ids) if thread_ids else 'all',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Conversation search failed: {e}")
            return {
                'results': [],
                'confidence': 0.0,
                'status': 'failed',
                'metadata': {'error': str(e)}
            }
    
    async def _search_knowledge_base(self, uif: SAM_UIF) -> Dict[str, Any]:
        """Perform RAG search on the long-term knowledge base."""
        try:
            query = uif.get_data("query")
            max_results = uif.get_data("max_results", self.config['max_knowledge_results'])
            search_threshold = self.config['knowledge_search_threshold']
            
            self.logger.info(f"Searching knowledge base for: {query}")
            
            # Perform vector search on memory store
            search_results = self._memory_store.search_memories(
                query=query,
                limit=max_results,
                similarity_threshold=search_threshold
            )
            
            # Calculate confidence based on similarity scores
            if search_results:
                avg_similarity = sum(r.similarity_score for r in search_results) / len(search_results)
                confidence = avg_similarity
            else:
                confidence = 0.0
            
            # Format results for Agent Zero
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    'type': 'knowledge',
                    'chunk_id': result.chunk_id,
                    'content': result.content,
                    'similarity_score': result.similarity_score,
                    'memory_type': result.memory_type.value if hasattr(result, 'memory_type') else 'unknown',
                    'source': result.source if hasattr(result, 'source') else 'knowledge_base',
                    'tags': result.tags if hasattr(result, 'tags') else [],
                    'importance_score': result.importance_score if hasattr(result, 'importance_score') else 0.5
                })
            
            return {
                'results': formatted_results,
                'confidence': confidence,
                'status': 'completed',
                'metadata': {
                    'search_type': 'knowledge_base',
                    'total_results': len(formatted_results),
                    'query': query,
                    'similarity_threshold': search_threshold,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Knowledge base search failed: {e}")
            return {
                'results': [],
                'confidence': 0.0,
                'status': 'failed',
                'metadata': {'error': str(e)}
            }

    async def _add_to_knowledge_base(self, uif: SAM_UIF) -> Dict[str, Any]:
        """Add verified information to the long-term knowledge base."""
        try:
            query = uif.get_data("query")  # Content to add
            content_metadata = uif.get_data("content_metadata", {})

            self.logger.info(f"Adding content to knowledge base: {query[:100]}...")

            # Extract metadata for the memory chunk
            source = content_metadata.get('source', 'agent_zero_research')
            tags = content_metadata.get('tags', ['agent_verified'])
            importance_score = content_metadata.get('importance_score', 0.8)
            memory_type_str = content_metadata.get('memory_type', 'RESEARCH')

            # Import memory types
            from memory.memory_vectorstore import MemoryType

            # Convert string to MemoryType enum
            try:
                memory_type = MemoryType[memory_type_str.upper()]
            except KeyError:
                memory_type = MemoryType.RESEARCH

            # Add to memory store
            chunk_id = self._memory_store.add_memory(
                content=query,
                memory_type=memory_type,
                source=source,
                tags=tags,
                importance_score=importance_score,
                metadata=content_metadata
            )

            return {
                'results': [{
                    'type': 'knowledge_addition',
                    'chunk_id': chunk_id,
                    'content_preview': query[:200] + "..." if len(query) > 200 else query,
                    'memory_type': memory_type.value,
                    'source': source,
                    'tags': tags,
                    'importance_score': importance_score
                }],
                'confidence': 1.0,  # High confidence for successful addition
                'status': 'completed',
                'metadata': {
                    'operation_type': 'knowledge_addition',
                    'chunk_id': chunk_id,
                    'content_length': len(query),
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Knowledge base addition failed: {e}")
            return {
                'results': [],
                'confidence': 0.0,
                'status': 'failed',
                'metadata': {'error': str(e)}
            }

    async def _cross_memory_search(self, uif: SAM_UIF) -> Dict[str, Any]:
        """Search both conversations and knowledge base, combining results."""
        try:
            query = uif.get_data("query")

            self.logger.info(f"Performing cross-memory search for: {query}")

            # Search both memory systems
            conversation_result = await self._search_conversations(uif)
            knowledge_result = await self._search_knowledge_base(uif)

            # Combine results
            all_results = []
            all_results.extend(conversation_result.get('results', []))
            all_results.extend(knowledge_result.get('results', []))

            # Sort by relevance/similarity scores
            all_results.sort(key=lambda x: x.get('relevance_score', x.get('similarity_score', 0)), reverse=True)

            # Calculate combined confidence
            conv_confidence = conversation_result.get('confidence', 0)
            kb_confidence = knowledge_result.get('confidence', 0)

            # Weighted average (favor knowledge base slightly for factual queries)
            combined_confidence = (conv_confidence * 0.4 + kb_confidence * 0.6)

            return {
                'results': all_results,
                'confidence': combined_confidence,
                'status': 'completed',
                'metadata': {
                    'search_type': 'cross_memory',
                    'conversation_results': len(conversation_result.get('results', [])),
                    'knowledge_results': len(knowledge_result.get('results', [])),
                    'total_results': len(all_results),
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Cross-memory search failed: {e}")
            return {
                'results': [],
                'confidence': 0.0,
                'status': 'failed',
                'metadata': {'error': str(e)}
            }

    def get_capability_description(self) -> Dict[str, Any]:
        """Return detailed capability description for Agent Zero's tool selection."""
        return {
            'tool_name': self.skill_name,
            'description': self.skill_description,
            'operations': {
                'search_conversations': {
                    'description': 'Search through archived conversation history',
                    'use_cases': ['Find past discussions', 'Recall previous answers', 'Check conversation context'],
                    'inputs': ['query', 'optional: thread_ids, max_results'],
                    'outputs': ['conversation_results', 'relevance_scores', 'context_messages']
                },
                'search_knowledge_base': {
                    'description': 'Search the long-term knowledge base using RAG',
                    'use_cases': ['Find factual information', 'Retrieve research data', 'Access stored knowledge'],
                    'inputs': ['query', 'optional: max_results'],
                    'outputs': ['knowledge_results', 'similarity_scores', 'source_information']
                },
                'add_to_knowledge_base': {
                    'description': 'Add verified information to the knowledge base',
                    'use_cases': ['Store research findings', 'Save important facts', 'Close research loops'],
                    'inputs': ['content', 'optional: metadata, tags, importance_score'],
                    'outputs': ['chunk_id', 'storage_confirmation']
                },
                'cross_memory_search': {
                    'description': 'Search both conversations and knowledge base simultaneously',
                    'use_cases': ['Comprehensive information retrieval', 'Find all related information'],
                    'inputs': ['query', 'optional: max_results'],
                    'outputs': ['combined_results', 'ranked_by_relevance']
                }
            },
            'decision_guidance': {
                'when_to_use': [
                    'Before searching external web sources',
                    'When user asks about past conversations',
                    'When needing to verify stored information',
                    'When adding new research findings'
                ],
                'priority': 'high',
                'efficiency_benefit': 'Reduces external API calls and improves response accuracy'
            }
        }
