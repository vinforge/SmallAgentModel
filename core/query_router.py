"""
Semantic Query Router for SAM
Routes queries through semantic retrieval and formats context for the model.

Sprint 2 Task 3: Semantic Query Router
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of query processing with context and metadata."""
    formatted_prompt: str
    context_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    semantic_mode: bool
    query_stats: Dict[str, Any]

class SemanticQueryRouter:
    """
    Routes user queries through semantic retrieval and formats context for the model.
    Provides intelligent context injection based on vector similarity.
    """
    
    def __init__(self, vector_manager, embedding_manager, config: Dict[str, Any] = None):
        """
        Initialize the semantic query router.
        
        Args:
            vector_manager: VectorManager instance for semantic search
            embedding_manager: EmbeddingManager instance for query embedding
            config: Configuration dictionary
        """
        self.vector_manager = vector_manager
        self.embedding_manager = embedding_manager
        
        # Configuration
        self.config = config or {}
        self.default_top_k = self.config.get('default_top_k', 5)
        self.score_threshold = self.config.get('score_threshold', 0.1)
        self.max_context_length = self.config.get('max_context_length', 4000)
        self.context_template = self.config.get('context_template', self._default_context_template())
        
        logger.info("Semantic query router initialized")
    
    def _default_context_template(self) -> str:
        """Default template for formatting context."""
        return """## Relevant Context

Based on your query, here are the most relevant pieces of information from the knowledge base:

{context_blocks}

## Query
{user_query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, please indicate what additional information might be needed."""
    
    def route_query(self, user_query: str, semantic_mode: bool = True, 
                   top_k: Optional[int] = None, tag_filters: Optional[List[str]] = None) -> QueryResult:
        """
        Route a user query through semantic retrieval or direct processing.
        
        Args:
            user_query: The user's query
            semantic_mode: Whether to use semantic retrieval
            top_k: Number of top chunks to retrieve
            tag_filters: Optional tag filters for retrieval
            
        Returns:
            QueryResult with formatted prompt and metadata
        """
        try:
            start_time = datetime.now()
            
            if not semantic_mode:
                # Direct mode - no semantic retrieval
                return self._direct_query(user_query, start_time)
            
            # Semantic mode - retrieve relevant context
            return self._semantic_query(user_query, top_k, tag_filters, start_time)
            
        except Exception as e:
            logger.error(f"Error routing query: {e}")
            return self._error_result(user_query, str(e))
    
    def _direct_query(self, user_query: str, start_time: datetime) -> QueryResult:
        """Process query without semantic retrieval."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResult(
            formatted_prompt=user_query,
            context_chunks=[],
            metadata={
                'mode': 'direct',
                'processing_time_seconds': processing_time
            },
            semantic_mode=False,
            query_stats={
                'chunks_retrieved': 0,
                'processing_time': processing_time
            }
        )
    
    def _semantic_query(self, user_query: str, top_k: Optional[int], 
                       tag_filters: Optional[List[str]], start_time: datetime) -> QueryResult:
        """Process query with semantic retrieval."""
        try:
            # Use default top_k if not specified
            if top_k is None:
                top_k = self.default_top_k
            
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_query(user_query)
            
            # Perform semantic search
            search_results = self.vector_manager.search(
                query_vector=query_embedding,
                top_k=top_k,
                score_threshold=self.score_threshold
            )
            
            # Apply tag filtering if specified
            if tag_filters:
                search_results = self._filter_by_tags(search_results, tag_filters)
            
            # Format context and create prompt
            context_blocks = self._format_context_blocks(search_results)
            formatted_prompt = self._create_formatted_prompt(user_query, context_blocks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                formatted_prompt=formatted_prompt,
                context_chunks=search_results,
                metadata={
                    'mode': 'semantic',
                    'chunks_used': len(search_results),
                    'top_k_requested': top_k,
                    'score_threshold': self.score_threshold,
                    'tag_filters': tag_filters,
                    'processing_time_seconds': processing_time
                },
                semantic_mode=True,
                query_stats={
                    'chunks_retrieved': len(search_results),
                    'avg_similarity_score': sum(r['similarity_score'] for r in search_results) / len(search_results) if search_results else 0,
                    'processing_time': processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error in semantic query processing: {e}")
            return self._error_result(user_query, str(e))
    
    def _filter_by_tags(self, results: List[Dict[str, Any]], tag_filters: List[str]) -> List[Dict[str, Any]]:
        """Filter results based on tag preferences."""
        try:
            # Determine if filters are required or preferred
            required_tags = [tag for tag in tag_filters if tag.startswith('require:')]
            preferred_tags = [tag for tag in tag_filters if not tag.startswith('require:')]
            
            # Clean up required tags (remove 'require:' prefix)
            required_tags = [tag.replace('require:', '') for tag in required_tags]
            
            # Apply filtering and ranking
            filtered_results = self.vector_manager.filter_by_tags(
                results=results,
                required_tags=required_tags if required_tags else None,
                preferred_tags=preferred_tags if preferred_tags else None
            )
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error filtering by tags: {e}")
            return results
    
    def _format_context_blocks(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context blocks."""
        if not search_results:
            return "No relevant context found in the knowledge base."
        
        context_blocks = []
        total_length = 0
        
        for i, result in enumerate(search_results, 1):
            # Extract metadata
            metadata = result.get('metadata', {})
            source_file = metadata.get('source_file', 'Unknown')
            tags = metadata.get('tags', [])
            enrichment_score = metadata.get('enrichment_score', 0)
            similarity_score = result.get('similarity_score', 0)
            
            # Format context block
            block = f"""### Context {i} (Similarity: {similarity_score:.3f})
**Source:** {source_file}
**Tags:** {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}
**Enrichment Score:** {enrichment_score:.1f}

{result['text']}

---"""
            
            # Check if adding this block would exceed max context length
            if total_length + len(block) > self.max_context_length:
                logger.debug(f"Context length limit reached, using {i-1} chunks")
                break
            
            context_blocks.append(block)
            total_length += len(block)
        
        return '\n\n'.join(context_blocks)
    
    def _create_formatted_prompt(self, user_query: str, context_blocks: str) -> str:
        """Create the final formatted prompt with context."""
        try:
            formatted_prompt = self.context_template.format(
                context_blocks=context_blocks,
                user_query=user_query
            )
            
            return formatted_prompt
            
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return f"Context:\n{context_blocks}\n\nQuery: {user_query}"
    
    def _error_result(self, user_query: str, error_message: str) -> QueryResult:
        """Create an error result."""
        return QueryResult(
            formatted_prompt=user_query,
            context_chunks=[],
            metadata={
                'mode': 'error',
                'error': error_message
            },
            semantic_mode=False,
            query_stats={
                'chunks_retrieved': 0,
                'error': error_message
            }
        )
    
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze a query to suggest optimal retrieval parameters.
        
        Args:
            user_query: The user's query
            
        Returns:
            Analysis results with suggestions
        """
        try:
            analysis = {
                'query_length': len(user_query),
                'word_count': len(user_query.split()),
                'suggested_tags': [],
                'suggested_top_k': self.default_top_k,
                'query_type': 'general'
            }
            
            # Analyze query content for tag suggestions
            query_lower = user_query.lower()
            
            # ML/AI related queries
            if any(term in query_lower for term in ['machine learning', 'ml', 'ai', 'neural', 'model', 'training']):
                analysis['suggested_tags'].extend(['ml:', 'concept:machine-learning'])
                analysis['query_type'] = 'ml_related'
            
            # GPU/Performance related
            if any(term in query_lower for term in ['gpu', 'cuda', 'performance', 'optimization']):
                analysis['suggested_tags'].extend(['gpu', 'performance'])
                analysis['query_type'] = 'performance_related'
            
            # Code/Programming related
            if any(term in query_lower for term in ['code', 'function', 'class', 'python', 'programming']):
                analysis['suggested_tags'].extend(['python', 'functions:', 'classes:'])
                analysis['query_type'] = 'code_related'
            
            # Error/Debugging related
            if any(term in query_lower for term in ['error', 'bug', 'debug', 'exception', 'fix']):
                analysis['suggested_tags'].extend(['error-handling', 'debugging'])
                analysis['query_type'] = 'debugging_related'
            
            # Adjust top_k based on query type
            if analysis['query_type'] in ['ml_related', 'code_related']:
                analysis['suggested_top_k'] = 7  # More context for complex topics
            elif analysis['query_type'] == 'debugging_related':
                analysis['suggested_top_k'] = 3  # Focused context for debugging
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        vector_stats = self.vector_manager.get_stats() if self.vector_manager else {}
        embedding_info = self.embedding_manager.get_model_info() if self.embedding_manager else {}
        
        return {
            'vector_store': vector_stats,
            'embedding_model': embedding_info,
            'config': {
                'default_top_k': self.default_top_k,
                'score_threshold': self.score_threshold,
                'max_context_length': self.max_context_length
            }
        }
