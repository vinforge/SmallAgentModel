#!/usr/bin/env python3
"""
Result Processor Service
Unified service for processing search results from various sources with consistent interfaces.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ResultType(Enum):
    """Types of search result objects."""
    RANKED_MEMORY_RESULT = "ranked_memory_result"
    MEMORY_SEARCH_RESULT = "memory_search_result"
    SECURE_MEMORY_RESULT = "secure_memory_result"
    DIRECT_CONTENT = "direct_content"
    UNKNOWN = "unknown"

@dataclass
class StandardizedResult:
    """Standardized result object with consistent interface."""
    content: str
    source: str
    similarity_score: float
    result_type: ResultType
    chunk_id: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    source_type: Optional[str] = None
    
    # Backward compatibility properties
    @property
    def chunk(self):
        """Backward compatibility: provide chunk-like access."""
        return self

class ResultProcessorService:
    """
    Unified service for processing search results from various sources.
    Handles all known result types and provides consistent interfaces.
    """
    
    def __init__(self):
        self.processors = {
            ResultType.RANKED_MEMORY_RESULT: self._process_ranked_memory_result,
            ResultType.MEMORY_SEARCH_RESULT: self._process_memory_search_result,
            ResultType.SECURE_MEMORY_RESULT: self._process_secure_memory_result,
            ResultType.DIRECT_CONTENT: self._process_direct_content,
        }
    
    def process_results(self, results: List[Any]) -> List[StandardizedResult]:
        """
        Process a list of search results into standardized format.
        
        Args:
            results: List of search result objects of various types
            
        Returns:
            List of StandardizedResult objects
        """
        if not results:
            return []
        
        logger.info(f"🔄 Processing {len(results)} search results")
        
        standardized_results = []
        
        for i, result in enumerate(results):
            try:
                result_type = self._identify_result_type(result)
                processor = self.processors.get(result_type, self._process_unknown_result)
                
                standardized = processor(result)
                if standardized:
                    standardized_results.append(standardized)
                    logger.debug(f"   {i+1}. {result_type.value}: {standardized.source}")
                else:
                    logger.warning(f"   {i+1}. Failed to process {result_type.value}")
                    
            except Exception as e:
                logger.warning(f"   {i+1}. Error processing result: {e}")
                # Try fallback processing
                fallback_result = self._process_fallback(result)
                if fallback_result:
                    standardized_results.append(fallback_result)
        
        logger.info(f"✅ Processed {len(standardized_results)}/{len(results)} results successfully")
        return standardized_results
    
    def _identify_result_type(self, result: Any) -> ResultType:
        """Identify the type of search result object."""
        # RankedMemoryResult (Phase 3 enhanced results)
        if hasattr(result, 'content') and hasattr(result, 'final_score') and hasattr(result, 'metadata'):
            return ResultType.RANKED_MEMORY_RESULT
        
        # MemorySearchResult (legacy results with chunk)
        elif hasattr(result, 'chunk') and hasattr(result, 'similarity_score'):
            return ResultType.MEMORY_SEARCH_RESULT
        
        # MemorySearchResult (alternative structure with memory_chunk)
        elif hasattr(result, 'memory_chunk'):
            return ResultType.MEMORY_SEARCH_RESULT
        
        # Secure memory store results
        elif hasattr(result, 'chunk') and hasattr(result.chunk, 'content') and hasattr(result.chunk, 'source'):
            return ResultType.SECURE_MEMORY_RESULT
        
        # Direct content objects
        elif hasattr(result, 'content') and hasattr(result, 'source'):
            return ResultType.DIRECT_CONTENT
        
        else:
            return ResultType.UNKNOWN
    
    def _process_ranked_memory_result(self, result: Any) -> Optional[StandardizedResult]:
        """Process RankedMemoryResult (Phase 3 enhanced)."""
        try:
            return StandardizedResult(
                content=result.content,
                source=result.metadata.get('source_path', result.metadata.get('source_name', 'Unknown')),
                similarity_score=getattr(result, 'final_score', getattr(result, 'confidence_score', 0.0)),
                result_type=ResultType.RANKED_MEMORY_RESULT,
                chunk_id=getattr(result, 'chunk_id', None),
                tags=getattr(result, 'tags', []),
                metadata=result.metadata,
                timestamp=result.metadata.get('created_at'),
                source_type='ranked_memory'
            )
        except Exception as e:
            logger.warning(f"Error processing RankedMemoryResult: {e}")
            return None
    
    def _process_memory_search_result(self, result: Any) -> Optional[StandardizedResult]:
        """Process MemorySearchResult (legacy with chunk or memory_chunk)."""
        try:
            # Handle different MemorySearchResult structures
            if hasattr(result, 'chunk'):
                chunk = result.chunk
                content = getattr(chunk, 'content', str(chunk))
                source = getattr(chunk, 'source', 'Unknown')
                chunk_id = getattr(chunk, 'chunk_id', None)
                tags = getattr(chunk, 'tags', [])
                timestamp = getattr(chunk, 'timestamp', None)
                metadata = getattr(chunk, 'metadata', {})
            elif hasattr(result, 'memory_chunk'):
                chunk = result.memory_chunk
                content = chunk.content
                source = chunk.source
                chunk_id = getattr(chunk, 'chunk_id', None)
                tags = getattr(chunk, 'tags', [])
                timestamp = getattr(chunk, 'timestamp', None)
                metadata = getattr(chunk, 'metadata', {})
            else:
                return None
            
            return StandardizedResult(
                content=content,
                source=source,
                similarity_score=getattr(result, 'similarity_score', getattr(result, 'score', 0.0)),
                result_type=ResultType.MEMORY_SEARCH_RESULT,
                chunk_id=chunk_id,
                tags=tags,
                metadata=metadata,
                timestamp=timestamp,
                source_type='memory_search'
            )
        except Exception as e:
            logger.warning(f"Error processing MemorySearchResult: {e}")
            return None
    
    def _process_secure_memory_result(self, result: Any) -> Optional[StandardizedResult]:
        """Process secure memory store results."""
        try:
            chunk = result.chunk
            return StandardizedResult(
                content=chunk.content,
                source=chunk.source,
                similarity_score=getattr(result, 'similarity_score', 0.0),
                result_type=ResultType.SECURE_MEMORY_RESULT,
                chunk_id=getattr(chunk, 'chunk_id', None),
                tags=getattr(chunk, 'tags', []),
                metadata=getattr(chunk, 'metadata', {}),
                timestamp=getattr(chunk, 'timestamp', None),
                source_type='secure_memory'
            )
        except Exception as e:
            logger.warning(f"Error processing secure memory result: {e}")
            return None
    
    def _process_direct_content(self, result: Any) -> Optional[StandardizedResult]:
        """Process direct content objects."""
        try:
            return StandardizedResult(
                content=result.content,
                source=getattr(result, 'source', 'Unknown'),
                similarity_score=getattr(result, 'similarity_score', getattr(result, 'score', 0.0)),
                result_type=ResultType.DIRECT_CONTENT,
                chunk_id=getattr(result, 'chunk_id', getattr(result, 'id', None)),
                tags=getattr(result, 'tags', []),
                metadata=getattr(result, 'metadata', {}),
                timestamp=getattr(result, 'timestamp', None),
                source_type=getattr(result, 'source_type', 'direct_content')
            )
        except Exception as e:
            logger.warning(f"Error processing direct content: {e}")
            return None
    
    def _process_unknown_result(self, result: Any) -> Optional[StandardizedResult]:
        """Process unknown result types with best-effort extraction."""
        logger.warning(f"Processing unknown result type: {type(result)}")
        return self._process_fallback(result)
    
    def _process_fallback(self, result: Any) -> Optional[StandardizedResult]:
        """Fallback processing for any result type."""
        try:
            # Try to extract basic information
            content = None
            source = "Unknown"
            similarity_score = 0.0
            
            # Try various content extraction methods
            if hasattr(result, 'content'):
                content = str(result.content)
            elif hasattr(result, 'text'):
                content = str(result.text)
            elif hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                content = str(result.chunk.content)
            else:
                content = str(result)[:500]  # Fallback to string representation
            
            # Try various source extraction methods
            if hasattr(result, 'source'):
                source = str(result.source)
            elif hasattr(result, 'chunk') and hasattr(result.chunk, 'source'):
                source = str(result.chunk.source)
            elif hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                source = result.metadata.get('source', result.metadata.get('source_path', 'Unknown'))
            
            # Try various similarity score extraction methods
            for attr in ['similarity_score', 'score', 'final_score', 'confidence_score']:
                if hasattr(result, attr):
                    try:
                        similarity_score = float(getattr(result, attr))
                        break
                    except (ValueError, TypeError):
                        continue
            
            if content:
                return StandardizedResult(
                    content=content,
                    source=source,
                    similarity_score=similarity_score,
                    result_type=ResultType.UNKNOWN,
                    chunk_id=getattr(result, 'chunk_id', getattr(result, 'id', None)),
                    tags=getattr(result, 'tags', []),
                    metadata=getattr(result, 'metadata', {}),
                    timestamp=getattr(result, 'timestamp', None),
                    source_type='fallback'
                )
            else:
                return None
                
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            return None
    
    def create_backward_compatible_results(self, standardized_results: List[StandardizedResult]) -> List[Any]:
        """
        Create backward-compatible result objects for existing code.
        
        Args:
            standardized_results: List of StandardizedResult objects
            
        Returns:
            List of backward-compatible result objects
        """
        compatible_results = []
        
        for result in standardized_results:
            # Create a backward-compatible object
            class CompatibleResult:
                def __init__(self, std_result: StandardizedResult):
                    self.content = std_result.content
                    self.source = std_result.source
                    self.similarity_score = std_result.similarity_score
                    self.source_type = std_result.source_type
                    self.chunk_id = std_result.chunk_id
                    self.tags = std_result.tags or []
                    self.metadata = std_result.metadata or {}
                    self.timestamp = std_result.timestamp
                    
                    # For backward compatibility with chunk access
                    self.chunk = self
                    
                    # For memory_chunk access pattern
                    self.memory_chunk = self
            
            compatible_results.append(CompatibleResult(result))
        
        return compatible_results
    
    def get_processing_stats(self, results: List[Any]) -> Dict[str, int]:
        """Get statistics about result types processed."""
        stats = {result_type.value: 0 for result_type in ResultType}
        
        for result in results:
            result_type = self._identify_result_type(result)
            stats[result_type.value] += 1
        
        return stats

# Global instance for easy access
_result_processor_service = None

def get_result_processor_service() -> ResultProcessorService:
    """Get or create the global result processor service instance."""
    global _result_processor_service
    if _result_processor_service is None:
        _result_processor_service = ResultProcessorService()
    return _result_processor_service

def process_search_results(results: List[Any]) -> List[StandardizedResult]:
    """
    Convenience function for processing search results.
    
    Args:
        results: List of search result objects of various types
        
    Returns:
        List of StandardizedResult objects
    """
    processor = get_result_processor_service()
    return processor.process_results(results)
