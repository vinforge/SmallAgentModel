#!/usr/bin/env python3
"""
Enhanced Context Router
Intelligent routing and assembly of context for SAM responses.
Enhanced with Sprint 15 features: Memory Ranking and Citation Engine.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ContextAssembly:
    """Assembled context with ranking and citation information."""
    context_text: str
    ranked_memories: List[Any]
    citations: List[Any]
    transparency_score: float
    context_quality_score: float
    routing_explanation: str

class EnhancedContextRouter:
    """
    Enhanced context router that uses memory ranking and citation engine
    to provide high-quality, transparent context for SAM responses.
    """
    
    def __init__(self):
        """Initialize the enhanced context router."""
        self.ranking_framework = None
        self.citation_engine = None
        self.query_router = None
        self.memory_store = None
        
        self._initialize_components()
        
        logger.info("Enhanced context router initialized")
    
    def _initialize_components(self):
        """Initialize all required components."""
        try:
            # Initialize memory ranking framework
            from memory.memory_ranking import get_memory_ranking_framework
            self.ranking_framework = get_memory_ranking_framework()
            
            # Initialize citation engine
            from memory.citation_engine import get_citation_engine
            self.citation_engine = get_citation_engine()
            
            # Initialize query router
            from utils.query_router import get_query_router
            self.query_router = get_query_router()
            
            # Initialize memory store
            from memory.memory_vectorstore import get_memory_store
            self.memory_store = get_memory_store()
            
            logger.info("All context router components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing context router components: {e}")
    
    def assemble_context(self, query: str, max_context_length: int = 2000) -> ContextAssembly:
        """
        Assemble high-quality context for a query using ranking and citations.
        
        Args:
            query: User query
            max_context_length: Maximum length of context text
            
        Returns:
            ContextAssembly with ranked context and citations
        """
        try:
            logger.info(f"Assembling context for query: {query[:50]}...")
            
            # Step 1: Analyze query using query router
            available_documents = self._get_available_documents()
            analysis = self.query_router.analyze_query(query, available_documents)
            
            # Step 2: Search for relevant memories
            relevant_memories = self._search_relevant_memories(query, analysis)
            
            if not relevant_memories:
                return self._create_empty_context(query, "No relevant memories found")
            
            # Step 3: Rank memories using ranking framework
            ranked_scores = self.ranking_framework.rank_memories(
                relevant_memories, 
                query=query,
                context={'query_analysis': analysis}
            )
            
            # Step 4: Select top-ranked memories for context
            selected_memories = self._select_context_memories(ranked_scores, max_context_length)
            
            # Step 5: Generate citations using citation engine
            cited_response = self.citation_engine.inject_citations(
                "",  # Empty response text for now
                [score.memory_id for score in selected_memories],
                query=query
            )
            
            # Step 6: Assemble final context
            context_text = self._build_context_text(selected_memories, cited_response)
            
            # Step 7: Calculate quality scores
            context_quality_score = self._calculate_context_quality(selected_memories)
            
            # Step 8: Generate routing explanation
            routing_explanation = self._generate_routing_explanation(analysis, selected_memories)
            
            return ContextAssembly(
                context_text=context_text,
                ranked_memories=selected_memories,
                citations=cited_response.citations,
                transparency_score=cited_response.transparency_score,
                context_quality_score=context_quality_score,
                routing_explanation=routing_explanation
            )
            
        except Exception as e:
            logger.error(f"Error assembling context: {e}")
            return self._create_empty_context(query, f"Error: {e}")
    
    def _get_available_documents(self) -> List[str]:
        """Get list of available documents for query analysis."""
        try:
            memory_stats = self.memory_store.get_memory_stats()
            if 'sources' in memory_stats:
                return [src for src in memory_stats['sources'] if src.startswith('document:')]
            return []
        except Exception as e:
            logger.debug(f"Error getting available documents: {e}")
            return []
    
    def _search_relevant_memories(self, query: str, analysis: Any) -> List[Any]:
        """Search for relevant memories based on query analysis."""
        try:
            # Get search filters from query router
            search_filters = self.query_router.get_search_filters(analysis)
            
            # Determine number of results based on query type
            max_results = 8 if analysis.query_type.value == "document_specific" else 6
            
            # Phase 3.2: Use enhanced search with hybrid ranking
            try:
                if hasattr(self.memory_store, 'enhanced_search_memories'):
                    # Enhanced search with ranking
                    memory_results = self.memory_store.enhanced_search_memories(
                        query=query,
                        max_results=max_results,
                        memory_types=search_filters.get('memory_types'),
                        tags=search_filters.get('tags'),
                        initial_candidates=max_results * 3
                    )
                    logger.info(f"Enhanced search returned {len(memory_results)} ranked results")
                else:
                    # Fallback to regular search
                    memory_results = self.memory_store.search_memories(
                        query,
                        max_results=max_results,
                        memory_types=search_filters.get('memory_types'),
                        tags=search_filters.get('tags')
                    )
            except Exception as e:
                logger.warning(f"Enhanced search failed in context router, using fallback: {e}")
                memory_results = self.memory_store.search_memories(
                    query,
                    max_results=max_results,
                    memory_types=search_filters.get('memory_types'),
                    tags=search_filters.get('tags')
                )
            
            logger.info(f"Found {len(memory_results)} relevant memories")
            return memory_results
            
        except Exception as e:
            logger.error(f"Error searching relevant memories: {e}")
            return []
    
    def _select_context_memories(self, ranked_scores: List[Any], 
                                max_length: int) -> List[Any]:
        """Select the best memories for context based on ranking and length constraints."""
        try:
            selected_memories = []
            current_length = 0
            
            # Prioritize pinned and high-priority memories
            priority_memories = [score for score in ranked_scores if score.is_priority or score.is_pinned]
            regular_memories = [score for score in ranked_scores if not (score.is_priority or score.is_pinned)]
            
            # Process priority memories first
            for score in priority_memories:
                memory = self._get_memory_from_score(score)
                if memory:
                    content_length = len(self._get_memory_content(memory))
                    if current_length + content_length <= max_length:
                        selected_memories.append(score)
                        current_length += content_length
                    else:
                        break
            
            # Add regular memories if space allows
            for score in regular_memories:
                memory = self._get_memory_from_score(score)
                if memory:
                    content_length = len(self._get_memory_content(memory))
                    if current_length + content_length <= max_length:
                        selected_memories.append(score)
                        current_length += content_length
                    else:
                        break
            
            logger.info(f"Selected {len(selected_memories)} memories for context ({current_length} chars)")
            return selected_memories
            
        except Exception as e:
            logger.error(f"Error selecting context memories: {e}")
            return ranked_scores[:3]  # Fallback to top 3
    
    def _get_memory_from_score(self, score: Any) -> Any:
        """Get memory object from ranking score."""
        try:
            # CRITICAL FIX: Get the actual memory from the memory store using the memory_id
            if hasattr(score, 'memory_id'):
                # Get memory from the memory store
                all_memories = self.memory_store.get_all_memories()
                for memory in all_memories:
                    if memory.chunk_id == score.memory_id:
                        return memory

            # Fallback: if score object has memory data directly
            if hasattr(score, 'content') or (hasattr(score, 'chunk') and hasattr(score.chunk, 'content')):
                return score

            return None
        except Exception as e:
            logger.debug(f"Error getting memory from score: {e}")
            return None
    
    def _get_memory_content(self, memory: Any) -> str:
        """Get content from memory object."""
        try:
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'content'):
                return memory.chunk.content
            elif hasattr(memory, 'content'):
                return memory.content
            return ""
        except Exception:
            return ""
    
    def _build_context_text(self, selected_memories: List[Any], 
                           cited_response: Any) -> str:
        """Build the final context text with citations."""
        try:
            context_parts = []
            
            # Add header based on memory types
            if any(self._is_document_memory(mem) for mem in selected_memories):
                context_parts.append("📚 **From uploaded documents and memory:**")
            else:
                context_parts.append("🧠 **Relevant context:**")
            
            # Add each memory with ranking information
            for i, memory_score in enumerate(selected_memories):
                memory = self._get_memory_from_score(memory_score)
                if not memory:
                    continue
                
                content = self._get_memory_content(memory)
                source = self._get_memory_source(memory)
                
                # Format content - use full content for document memories, preview for others
                if self._is_document_memory(memory):
                    # For document memories, use more content (up to 1500 chars) to provide better context
                    content_preview = content[:1500] if len(content) > 1500 else content
                    if len(content) > 1500:
                        content_preview += "... [content continues]"
                else:
                    # For conversation memories, use shorter preview
                    content_preview = content[:300] if len(content) > 300 else content
                    if len(content) > 300:
                        content_preview += "..."
                
                # Add priority indicator
                priority_icon = "⭐" if memory_score.is_priority else "🔸" if memory_score.is_pinned else "•"
                
                # Add ranking score for transparency
                score_info = f"(Score: {memory_score.overall_score:.2f})"
                
                # Format source nicely
                source_display = self._format_source_display(source)
                
                context_parts.append(f"{priority_icon} {source_display} {score_info}: {content_preview}...")
            
            # Add citations if available
            if cited_response.citations:
                context_parts.append("\n**Sources:**")
                for citation in cited_response.citations:
                    context_parts.append(f"- {citation.citation_label}: \"{citation.quote_text}\"")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building context text: {e}")
            return "Error building context"
    
    def _is_document_memory(self, memory: Any) -> bool:
        """Check if memory is from a document."""
        try:
            source = self._get_memory_source(memory)
            return source.startswith("document:")
        except Exception:
            return False
    
    def _get_memory_source(self, memory: Any) -> str:
        """Get source from memory object."""
        try:
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'source'):
                return memory.chunk.source
            elif hasattr(memory, 'source'):
                return memory.source
            return "unknown"
        except Exception:
            return "unknown"
    
    def _format_source_display(self, source: str) -> str:
        """Format source for display."""
        try:
            if source.startswith("document:"):
                # Extract filename from path
                file_path = source.replace("document:", "")
                if ":" in file_path:
                    file_name = file_path.split(":")[-1].replace("block_", "Block ")
                    file_base = file_path.split(":")[0].split("/")[-1]
                    return f"📄 {file_base} ({file_name})"
                else:
                    return f"📄 {file_path.split('/')[-1]}"
            else:
                return f"💭 {source}"
        except Exception:
            return f"📄 {source}"
    
    def _calculate_context_quality(self, selected_memories: List[Any]) -> float:
        """Calculate overall context quality score."""
        try:
            if not selected_memories:
                return 0.0
            
            # Average ranking scores
            avg_ranking_score = sum(mem.overall_score for mem in selected_memories) / len(selected_memories)
            
            # Diversity score (different sources)
            sources = set(self._get_memory_source(self._get_memory_from_score(mem)) for mem in selected_memories)
            diversity_score = min(1.0, len(sources) / 3.0)  # Normalize to 3 sources
            
            # Priority coverage (how many priority memories included)
            priority_count = sum(1 for mem in selected_memories if mem.is_priority)
            priority_score = min(1.0, priority_count / 2.0)  # Normalize to 2 priority memories
            
            # Combined quality score
            quality_score = (avg_ranking_score * 0.5 + diversity_score * 0.3 + priority_score * 0.2)
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.debug(f"Error calculating context quality: {e}")
            return 0.5
    
    def _generate_routing_explanation(self, analysis: Any, selected_memories: List[Any]) -> str:
        """Generate explanation of routing and ranking decisions."""
        try:
            explanation_parts = []
            
            # Query analysis
            explanation_parts.append(f"Query Type: {analysis.query_type.value}")
            explanation_parts.append(f"Confidence: {analysis.confidence:.2f}")
            
            # Memory selection
            explanation_parts.append(f"Selected {len(selected_memories)} memories")
            
            # Priority breakdown
            priority_count = sum(1 for mem in selected_memories if mem.is_priority)
            pinned_count = sum(1 for mem in selected_memories if mem.is_pinned)
            
            if priority_count > 0:
                explanation_parts.append(f"Priority memories: {priority_count}")
            if pinned_count > 0:
                explanation_parts.append(f"Pinned memories: {pinned_count}")
            
            return " | ".join(explanation_parts)
            
        except Exception as e:
            logger.debug(f"Error generating routing explanation: {e}")
            return "Routing explanation unavailable"
    
    def _create_empty_context(self, query: str, reason: str) -> ContextAssembly:
        """Create empty context assembly for error cases."""
        return ContextAssembly(
            context_text=f"No relevant context found: {reason}",
            ranked_memories=[],
            citations=[],
            transparency_score=0.0,
            context_quality_score=0.0,
            routing_explanation=reason
        )

# Global context router instance
_context_router = None

def get_enhanced_context_router() -> EnhancedContextRouter:
    """Get or create a global enhanced context router instance."""
    global _context_router
    
    if _context_router is None:
        _context_router = EnhancedContextRouter()
    
    return _context_router
