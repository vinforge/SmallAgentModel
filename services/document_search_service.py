#!/usr/bin/env python3
"""
Document Search Service
Unified service for searching and retrieving document content across memory stores.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Standardized search result structure."""
    content: str
    source: str
    similarity_score: float
    source_type: str
    chunk_id: Optional[str] = None

class DocumentSearchService:
    """
    Unified document search service that handles multiple memory stores
    and provides consistent result processing with document reference resolution.
    """

    def __init__(self):
        self.secure_memory_store = None
        self.regular_memory_store = None
        self.document_resolver = None
        self._initialize_stores()
        self._initialize_resolver()
    
    def _initialize_stores(self):
        """Initialize memory stores with proper error handling."""
        try:
            # Initialize secure memory store
            import streamlit as st
            if hasattr(st.session_state, 'secure_memory_store') and st.session_state.secure_memory_store:
                self.secure_memory_store = st.session_state.secure_memory_store
            else:
                self._initialize_secure_store()
        except Exception as e:
            logger.warning(f"Secure memory store initialization failed: {e}")
        
        try:
            # Initialize regular memory store
            from memory.memory_vectorstore import get_memory_store
            self.regular_memory_store = get_memory_store()
        except Exception as e:
            logger.error(f"Regular memory store initialization failed: {e}")

    def _initialize_resolver(self):
        """Initialize document reference resolver."""
        try:
            from .document_reference_resolver import get_document_reference_resolver
            # Use the primary memory store (secure if available, otherwise regular)
            primary_store = self.secure_memory_store or self.regular_memory_store
            self.document_resolver = get_document_reference_resolver(primary_store)
            logger.info("Document reference resolver initialized")
        except Exception as e:
            logger.error(f"Document resolver initialization failed: {e}")
            self.document_resolver = None
    
    def _initialize_secure_store(self):
        """Initialize secure memory store as fallback."""
        try:
            import streamlit as st
            from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType
            
            st.session_state.secure_memory_store = get_secure_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384,
                security_manager=None
            )
            self.secure_memory_store = st.session_state.secure_memory_store
            logger.info("✅ Secure memory store initialized as fallback")
        except Exception as e:
            logger.warning(f"Secure memory store fallback initialization failed: {e}")
    
    def search_documents(self, query: str, max_results: int = 5, user_id: str = None) -> List[SearchResult]:
        """
        Search for documents across all available memory stores with intelligent caching.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            user_id: Optional user ID for personalized caching

        Returns:
            List of standardized SearchResult objects
        """
        import time

        # PHASE 3: Intelligent caching integration
        start_time = time.time()
        cache_key = self._generate_cache_key(query, max_results)

        try:
            # Try intelligent cache first
            from services.intelligent_cache_service import get_intelligent_cache_service
            cache_service = get_intelligent_cache_service()

            cached_results = cache_service.get(cache_key, user_id)
            if cached_results:
                response_time_ms = (time.time() - start_time) * 1000
                logger.info(f"🎯 Cache hit for query: '{query}' ({response_time_ms:.1f}ms)")

                # Record performance analytics
                self._record_performance(query, response_time_ms, len(cached_results),
                                       cache_hit=True, user_id=user_id)
                return cached_results

        except Exception as e:
            logger.warning(f"⚠️ Cache lookup failed: {e}")

        # PHASE 1: Check for specific document references
        logger.info(f"🔍 Enhanced document search: '{query}' (max_results: {max_results})")

        if self.document_resolver:
            resolved_doc = self.document_resolver.resolve_document_reference(query)
            if resolved_doc and resolved_doc.confidence > 0.7:
                logger.info(f"📄 Document reference resolved: {resolved_doc.filename}")
                return self._get_document_specific_results(resolved_doc, query, max_results)

        # Cache miss - perform actual search
        all_results = []

        # Priority 1: Search secure memory store
        secure_results = self._search_secure_store(query, max_results * 2)

        # Check if secure results contain relevant document content
        has_relevant_content = self._has_relevant_document_content(secure_results)

        if has_relevant_content:
            all_results.extend(secure_results)
            logger.info(f"✅ Found relevant content in secure store: {len(secure_results)} results")
        else:
            # Priority 2: Search regular memory store for document content
            logger.info("🔄 No relevant content in secure store, searching regular memory store...")
            regular_results = self._search_regular_store(query, max_results * 2)

            if regular_results:
                all_results.extend(regular_results)
                logger.info(f"✅ Found relevant content in regular store: {len(regular_results)} results")
            else:
                # Fallback: Include secure results even if not highly relevant
                all_results.extend(secure_results)
                logger.warning("⚠️ No highly relevant content found, using available results")

        # Sort by similarity score and source type priority
        all_results.sort(key=lambda x: (
            x.source_type == 'uploaded_documents',  # Prioritize uploaded documents
            x.similarity_score
        ), reverse=True)

        final_results = all_results[:max_results]

        # PHASE 3: Cache the results with intelligent caching
        try:
            cache_service.put(cache_key, final_results, query, user_id)
            logger.debug(f"📦 Cached results for query: '{query}'")
        except Exception as e:
            logger.warning(f"⚠️ Cache storage failed: {e}")

        # Record performance analytics
        response_time_ms = (time.time() - start_time) * 1000
        self._record_performance(query, response_time_ms, len(final_results),
                               cache_hit=False, user_id=user_id)

        logger.info(f"📊 Returning {len(final_results)} final results ({response_time_ms:.1f}ms)")
        return final_results
    
    def _search_secure_store(self, query: str, max_results: int) -> List[SearchResult]:
        """Search secure memory store and return standardized results."""
        if not self.secure_memory_store:
            return []
        
        try:
            # Try enhanced search first
            if hasattr(self.secure_memory_store, 'enhanced_search_memories'):
                raw_results = self.secure_memory_store.enhanced_search_memories(
                    query=f"{query} uploaded document whitepaper pdf",
                    max_results=max_results
                )
            else:
                raw_results = self.secure_memory_store.search_memories(
                    f"{query} uploaded document",
                    max_results
                )
            
            return self._standardize_results(raw_results, 'uploaded_documents')
            
        except Exception as e:
            logger.error(f"Secure store search failed: {e}")
            return []
    
    def _search_regular_store(self, query: str, max_results: int) -> List[SearchResult]:
        """Search regular memory store for document content."""
        if not self.regular_memory_store:
            return []
        
        try:
            # Search with document-specific terms
            search_query = f"{query} SAM story Chroma Ethan Hayes neural network"
            raw_results = self.regular_memory_store.search_memories(search_query, max_results)
            
            # Filter for relevant document content
            relevant_results = []
            for result in raw_results:
                if self._is_relevant_document_content(result):
                    relevant_results.append(result)
            
            return self._standardize_results(relevant_results, 'uploaded_documents')
            
        except Exception as e:
            logger.error(f"Regular store search failed: {e}")
            return []
    
    def _standardize_results(self, raw_results: List[Any], source_type: str) -> List[SearchResult]:
        """Convert raw search results to standardized SearchResult objects using ResultProcessor."""
        try:
            # Use the unified result processor service
            from services.result_processor_service import get_result_processor_service
            processor = get_result_processor_service()

            standardized_results = processor.process_results(raw_results)

            # Convert to our SearchResult format and apply source_type
            search_results = []
            for std_result in standardized_results:
                search_results.append(SearchResult(
                    content=std_result.content,
                    source=std_result.source,
                    similarity_score=std_result.similarity_score,
                    source_type=source_type,  # Override with our source_type
                    chunk_id=std_result.chunk_id
                ))

            logger.info(f"📊 Standardized {len(search_results)} results using ResultProcessor")
            return search_results

        except Exception as e:
            logger.warning(f"⚠️ ResultProcessor failed, using fallback: {e}")
            return self._standardize_results_fallback(raw_results, source_type)

    def _standardize_results_fallback(self, raw_results: List[Any], source_type: str) -> List[SearchResult]:
        """Fallback standardization method if ResultProcessor fails."""
        standardized = []

        for result in raw_results:
            try:
                # Extract content and source based on result structure
                content, source, similarity, chunk_id = self._extract_result_data(result)

                if content and source:
                    standardized.append(SearchResult(
                        content=content,
                        source=source,
                        similarity_score=similarity,
                        source_type=source_type,
                        chunk_id=chunk_id
                    ))

            except Exception as e:
                logger.warning(f"Error standardizing result: {e}")
                continue

        return standardized

    def _extract_result_data(self, result: Any) -> tuple:
        """Extract content, source, similarity, and chunk_id from various result types."""
        try:
            # Debug: Log the result type and attributes
            logger.debug(f"Processing result type: {type(result)}")
            logger.debug(f"Result attributes: {dir(result)}")

            # Handle RankedMemoryResult (secure store) - check multiple possible structures
            if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                return (
                    result.chunk.content,
                    result.chunk.source,
                    getattr(result, 'similarity_score', 0.0),
                    getattr(result.chunk, 'chunk_id', None)
                )

            # Handle RankedMemoryResult with direct memory_chunk access
            elif hasattr(result, 'memory_chunk') and hasattr(result.memory_chunk, 'content'):
                return (
                    result.memory_chunk.content,
                    getattr(result.memory_chunk, 'source', 'Unknown'),
                    getattr(result, 'similarity_score', getattr(result, 'score', 0.0)),
                    getattr(result.memory_chunk, 'chunk_id', None)
                )

            # Handle RankedMemoryResult with direct content access
            elif hasattr(result, 'content'):
                return (
                    result.content,
                    getattr(result, 'source', 'Unknown'),
                    getattr(result, 'similarity_score', getattr(result, 'score', 0.0)),
                    getattr(result, 'chunk_id', None)
                )

            # Handle objects with 'memory' attribute
            elif hasattr(result, 'memory') and hasattr(result.memory, 'content'):
                return (
                    result.memory.content,
                    getattr(result.memory, 'source', 'Unknown'),
                    getattr(result, 'similarity_score', getattr(result, 'score', 0.0)),
                    getattr(result.memory, 'chunk_id', None)
                )

            # Handle MemorySearchResult (regular store)
            elif hasattr(result, 'memory_chunk'):
                return (
                    result.memory_chunk.content,
                    result.memory_chunk.source,
                    getattr(result, 'score', 0.0),
                    getattr(result.memory_chunk, 'chunk_id', None)
                )

            else:
                logger.warning(f"Unknown result structure: {type(result)}")
                logger.warning(f"Available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                return None, None, 0.0, None

        except Exception as e:
            logger.error(f"Error extracting result data from {type(result)}: {e}")
            logger.error(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            return None, None, 0.0, None
    
    def _has_relevant_document_content(self, results: List[SearchResult]) -> bool:
        """Check if results contain actual document content (not just metadata)."""
        for result in results[:3]:  # Check top 3 results
            content_lower = result.content.lower()
            if any(term in content_lower for term in [
                'chroma', 'ethan hayes', 'neural network', 'university lab', 'project chroma'
            ]):
                return True
        return False
    
    def _is_relevant_document_content(self, result: Any) -> bool:
        """Check if a single result contains relevant document content."""
        try:
            # Extract content based on result structure - handle multiple types
            content = None

            if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                content = result.chunk.content.lower()
            elif hasattr(result, 'memory_chunk') and hasattr(result.memory_chunk, 'content'):
                content = result.memory_chunk.content.lower()
            elif hasattr(result, 'memory') and hasattr(result.memory, 'content'):
                content = result.memory.content.lower()
            elif hasattr(result, 'content'):
                content = result.content.lower()
            else:
                logger.debug(f"No content found in result type: {type(result)}")
                return False

            if not content:
                return False
            
            return any(term in content for term in [
                'chroma', 'ethan hayes', 'neural network', 'university lab', 'project chroma'
            ])
            
        except Exception:
            return False

    def _generate_cache_key(self, query: str, max_results: int) -> str:
        """Generate cache key for query and parameters."""
        import hashlib

        # Create a normalized cache key
        normalized_query = query.lower().strip()
        cache_data = f"{normalized_query}|{max_results}"

        # Generate hash for consistent key length
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _record_performance(self, query: str, response_time_ms: float, result_count: int,
                           cache_hit: bool = False, user_id: str = None):
        """Record performance metrics for analytics."""
        try:
            from services.performance_analytics_service import get_performance_analytics_service
            analytics = get_performance_analytics_service()

            analytics.record_search_performance(
                query=query,
                response_time_ms=response_time_ms,
                result_count=result_count,
                cache_hit=cache_hit,
                service_name="document_search_service"
            )

        except Exception as e:
            logger.warning(f"⚠️ Performance recording failed: {e}")

    def _record_error(self, error_type: str, error_message: str):
        """Record error for analytics."""
        try:
            from services.performance_analytics_service import get_performance_analytics_service
            analytics = get_performance_analytics_service()

            analytics.record_service_error(
                service_name="document_search_service",
                error_type=error_type,
                error_message=error_message
            )

        except Exception as e:
            logger.warning(f"⚠️ Error recording failed: {e}")

    def _get_document_specific_results(self, resolved_doc, query: str, max_results: int) -> List[SearchResult]:
        """Get results for a specific resolved document."""
        try:
            # Get all chunks for the resolved document
            document_chunks = self.document_resolver.get_document_chunks(resolved_doc)

            if not document_chunks:
                logger.warning(f"No chunks found for resolved document: {resolved_doc.filename}")
                return []

            # Create search results from document chunks
            results = []
            for i, chunk in enumerate(document_chunks[:max_results]):
                result = SearchResult(
                    content=chunk.content,
                    source=f"Document: {resolved_doc.filename}",
                    similarity_score=0.95,  # High score for exact document matches
                    source_type='resolved_document',
                    chunk_id=chunk.chunk_id
                )
                results.append(result)

            logger.info(f"📄 Retrieved {len(results)} chunks from document: {resolved_doc.filename}")

            # If we have a title, add it as context
            if resolved_doc.title and resolved_doc.title.upper() in query.upper():
                # Add title information as first result
                title_result = SearchResult(
                    content=f"Document Title: {resolved_doc.title}\n\nThis document contains {resolved_doc.total_chunks} sections. Here are the most relevant parts:",
                    source=f"Document: {resolved_doc.filename} (Title)",
                    similarity_score=1.0,
                    source_type='document_metadata',
                    chunk_id=f"title_{resolved_doc.document_id}"
                )
                results.insert(0, title_result)

            return results

        except Exception as e:
            logger.error(f"Error getting document-specific results: {e}")
            return []

# Global instance for easy access
_document_search_service = None

def get_document_search_service() -> DocumentSearchService:
    """Get or create the global document search service instance."""
    global _document_search_service
    if _document_search_service is None:
        _document_search_service = DocumentSearchService()
    return _document_search_service
