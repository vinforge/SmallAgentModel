"""
Memory Result Adapter for Phase 3.2 Integration
Provides compatibility between RankedMemoryResult and MemorySearchResult objects.
"""

import logging
from typing import List, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

def extract_content_from_memory_result(memory_result: Any) -> str:
    """
    Extract content from either RankedMemoryResult or MemorySearchResult.

    Args:
        memory_result: Memory result object (any type)

    Returns:
        Content string
    """
    try:
        # Debug logging
        logger.debug(f"Extracting content from memory result type: {type(memory_result)}")

        # Phase 3: RankedMemoryResult (has content and metadata directly)
        if hasattr(memory_result, 'content') and hasattr(memory_result, 'metadata'):
            return memory_result.content

        # Legacy: MemorySearchResult with chunk
        elif hasattr(memory_result, 'chunk') and hasattr(memory_result.chunk, 'content'):
            return memory_result.chunk.content

        # Legacy: MemorySearchResult with memory_chunk
        elif hasattr(memory_result, 'memory_chunk') and hasattr(memory_result.memory_chunk, 'content'):
            return memory_result.memory_chunk.content

        # Fallback: try direct content attribute
        elif hasattr(memory_result, 'content'):
            return memory_result.content

        # Last resort: string representation
        else:
            logger.warning(f"Unknown memory result structure: {type(memory_result)}")
            logger.warning(f"Available attributes: {[attr for attr in dir(memory_result) if not attr.startswith('_')]}")
            return str(memory_result)

    except Exception as e:
        logger.error(f"Error extracting content from memory result {type(memory_result)}: {e}")
        return str(memory_result)

def extract_metadata_from_memory_result(memory_result: Any) -> Dict[str, Any]:
    """
    Extract metadata from either RankedMemoryResult or MemorySearchResult.

    Args:
        memory_result: Memory result object (any type)

    Returns:
        Metadata dictionary
    """
    try:
        # Phase 3: RankedMemoryResult (has metadata directly)
        if hasattr(memory_result, 'metadata') and hasattr(memory_result, 'chunk_id'):
            return memory_result.metadata

        # Legacy: MemorySearchResult with chunk
        elif hasattr(memory_result, 'chunk') and hasattr(memory_result.chunk, 'metadata'):
            return memory_result.chunk.metadata

        # Legacy: MemorySearchResult with memory_chunk
        elif hasattr(memory_result, 'memory_chunk') and hasattr(memory_result.memory_chunk, 'metadata'):
            return memory_result.memory_chunk.metadata

        # Fallback: empty metadata
        else:
            logger.debug(f"No metadata found in memory result type: {type(memory_result)}")
            return {}

    except Exception as e:
        logger.warning(f"Error extracting metadata from memory result: {e}")
        return {}

def extract_source_info_from_memory_result(memory_result: Any) -> Dict[str, Any]:
    """
    Extract source information from memory result.
    
    Args:
        memory_result: Memory result object (any type)
        
    Returns:
        Source information dictionary
    """
    try:
        metadata = extract_metadata_from_memory_result(memory_result)
        
        # Phase 3: Enhanced metadata with rich source info
        if 'source_name' in metadata:
            return {
                'name': metadata.get('source_name', 'Unknown'),
                'page_number': metadata.get('page_number', 1),
                'chunk_index': metadata.get('chunk_index', 0),
                'section_title': metadata.get('section_title', ''),
                'confidence_indicator': metadata.get('confidence_indicator', '?'),
                'full_path': metadata.get('source_path', ''),
                'document_position': metadata.get('document_position', 0.0)
            }
        
        # Legacy: Extract from source field
        else:
            source = ""
            if hasattr(memory_result, 'chunk') and hasattr(memory_result.chunk, 'source'):
                source = memory_result.chunk.source
            elif hasattr(memory_result, 'source'):
                source = memory_result.source
            
            # Parse legacy source format
            return parse_legacy_source(source)
            
    except Exception as e:
        logger.warning(f"Error extracting source info from memory result: {e}")
        return {
            'name': 'Unknown',
            'page_number': 1,
            'chunk_index': 0,
            'section_title': '',
            'confidence_indicator': '?',
            'full_path': '',
            'document_position': 0.0
        }

def parse_legacy_source(source: str) -> Dict[str, Any]:
    """Parse legacy source string format."""
    try:
        import re
        from pathlib import Path
        
        # Default values
        result = {
            'name': 'Unknown',
            'page_number': 1,
            'chunk_index': 0,
            'section_title': '',
            'confidence_indicator': '?',
            'full_path': source,
            'document_position': 0.0
        }
        
        if not source:
            return result
        
        # Extract document name from path
        if ':' in source:
            parts = source.split(':')
            if len(parts) > 1:
                path_part = parts[1]
                if path_part.startswith('uploads/'):
                    result['name'] = Path(path_part).name
                else:
                    result['name'] = path_part
        else:
            result['name'] = Path(source).name if source else 'Unknown'
        
        # Extract page number
        page_match = re.search(r'page[_\s]*(\d+)', source, re.IGNORECASE)
        if page_match:
            result['page_number'] = int(page_match.group(1))
        
        # Extract chunk/block index
        chunk_match = re.search(r'(?:chunk|block)[_\s]*(\d+)', source, re.IGNORECASE)
        if chunk_match:
            result['chunk_index'] = int(chunk_match.group(1))
        
        return result
        
    except Exception as e:
        logger.warning(f"Error parsing legacy source: {e}")
        return {
            'name': 'Unknown',
            'page_number': 1,
            'chunk_index': 0,
            'section_title': '',
            'confidence_indicator': '?',
            'full_path': source,
            'document_position': 0.0
        }

def extract_confidence_score_from_memory_result(memory_result: Any) -> float:
    """
    Extract confidence score from memory result.
    
    Args:
        memory_result: Memory result object (any type)
        
    Returns:
        Confidence score (0.0-1.0)
    """
    try:
        # Phase 3: RankedMemoryResult with confidence_score
        if hasattr(memory_result, 'confidence_score'):
            return float(memory_result.confidence_score)
        
        # Phase 3: RankedMemoryResult with final_score
        elif hasattr(memory_result, 'final_score'):
            return float(memory_result.final_score)
        
        # Legacy: MemorySearchResult with similarity_score
        elif hasattr(memory_result, 'similarity_score'):
            return float(memory_result.similarity_score)
        
        # Legacy: MemoryChunk with importance_score
        elif hasattr(memory_result, 'chunk') and hasattr(memory_result.chunk, 'importance_score'):
            return float(memory_result.chunk.importance_score)
        
        # Fallback
        else:
            return 0.5
            
    except Exception as e:
        logger.warning(f"Error extracting confidence score from memory result: {e}")
        return 0.5

def format_enhanced_citation(memory_result: Any, index: int = 1) -> str:
    """
    Format enhanced citation from memory result.
    
    Args:
        memory_result: Memory result object (any type)
        index: Citation index number
        
    Returns:
        Formatted citation string
    """
    try:
        source_info = extract_source_info_from_memory_result(memory_result)
        confidence = extract_confidence_score_from_memory_result(memory_result)
        
        # Build citation components
        citation_parts = []
        
        # Source name
        citation_parts.append(f"📚 **{source_info['name']}**")
        
        # Confidence indicator
        confidence_dots = "●" * min(5, max(1, int(confidence * 5)))
        confidence_empty = "○" * (5 - len(confidence_dots))
        citation_parts.append(f"{confidence_dots}{confidence_empty} ({confidence:.1%})")
        
        # Page/section info if available
        if source_info['page_number'] > 1:
            citation_parts.append(f"p.{source_info['page_number']}")
        
        if source_info['section_title']:
            citation_parts.append(f"§{source_info['section_title']}")
        
        return f"{index}. {' '.join(citation_parts)}"
        
    except Exception as e:
        logger.warning(f"Error formatting enhanced citation: {e}")
        return f"{index}. 📚 **Unknown Source** ○○○○○ (0.0%)"

def convert_to_legacy_format(ranked_results: List[Any]) -> List[Any]:
    """
    Convert RankedMemoryResult objects to legacy MemorySearchResult format for compatibility.
    
    Args:
        ranked_results: List of RankedMemoryResult objects
        
    Returns:
        List of objects compatible with legacy code
    """
    try:
        from memory.memory_vectorstore import MemorySearchResult, MemoryChunk, MemoryType
        
        legacy_results = []
        
        for i, result in enumerate(ranked_results):
            try:
                # Check if this is already a legacy result
                if hasattr(result, 'chunk') and hasattr(result, 'similarity_score'):
                    legacy_results.append(result)
                    continue
                
                # Convert RankedMemoryResult to MemorySearchResult
                if hasattr(result, 'content') and hasattr(result, 'final_score'):
                    # Create a MemoryChunk from RankedMemoryResult
                    chunk = MemoryChunk(
                        chunk_id=result.chunk_id,
                        content=result.content,
                        content_hash="",  # Not available in RankedMemoryResult
                        embedding=None,   # Not needed for display
                        memory_type=MemoryType.DOCUMENT,  # Default
                        source=result.metadata.get('source_path', ''),
                        timestamp=result.metadata.get('created_at', ''),
                        tags=[],  # Not available
                        importance_score=result.confidence_score,
                        access_count=0,  # Not available
                        last_accessed='',  # Not available
                        metadata=result.metadata
                    )
                    
                    # Create MemorySearchResult
                    legacy_result = MemorySearchResult(
                        chunk=chunk,
                        similarity_score=result.semantic_score,
                        rank=i + 1
                    )
                    
                    legacy_results.append(legacy_result)
                else:
                    # Unknown format, pass through
                    legacy_results.append(result)
                    
            except Exception as e:
                logger.warning(f"Error converting result {i}: {e}")
                legacy_results.append(result)
        
        return legacy_results
        
    except Exception as e:
        logger.error(f"Error converting to legacy format: {e}")
        return ranked_results  # Return original on error
