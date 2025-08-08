#!/usr/bin/env python3
"""
Document Context Assembler
Formats document chunks into structured context for LLM consumption with proper citations.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .semantic_document_search import DocumentChunk, DocumentSearchResult

logger = logging.getLogger(__name__)

@dataclass
class AssembledContext:
    """Assembled document context ready for LLM consumption."""
    formatted_context: str
    source_count: int
    document_count: int
    total_word_count: int
    confidence_level: str
    context_metadata: Dict[str, Any]
    citation_map: Dict[str, str]  # Maps source references to full citations

class DocumentContextAssembler:
    """
    Takes raw document chunks and formats them into clean, structured context
    for the LLM with proper citation metadata.
    """
    
    def __init__(self, max_context_length: int = 4000):
        """
        Initialize the context assembler.
        
        Args:
            max_context_length: Maximum character length for assembled context
        """
        self.max_context_length = max_context_length
        self.logger = logging.getLogger(__name__)
    
    def assemble_context(self, search_result: DocumentSearchResult) -> AssembledContext:
        """
        Assemble document chunks into formatted context for LLM.
        
        Args:
            search_result: Result from semantic document search
            
        Returns:
            AssembledContext with formatted text and metadata
        """
        try:
            if not search_result.chunks:
                return self._create_empty_context(search_result.query)
            
            self.logger.info(f"📝 Assembling context from {len(search_result.chunks)} chunks")
            
            # Sort chunks by relevance and priority
            sorted_chunks = self._sort_chunks_for_context(search_result.chunks)
            
            # Build context sections
            context_sections = []
            citation_map = {}
            total_word_count = 0
            current_length = 0
            
            # Add header
            header = "--- Begin Context from Uploaded Documents ---\n"
            context_sections.append(header)
            current_length += len(header)
            
            # Process each chunk
            for i, chunk in enumerate(sorted_chunks):
                # Create citation reference
                citation_ref = f"Doc{i+1}"
                citation_map[citation_ref] = self._create_full_citation(chunk)
                
                # Format chunk section
                chunk_section = self._format_chunk_section(chunk, citation_ref)
                
                # Check if adding this chunk would exceed length limit
                if current_length + len(chunk_section) > self.max_context_length:
                    self.logger.info(f"⚠️ Context length limit reached, truncating at {i} chunks")
                    break
                
                context_sections.append(chunk_section)
                current_length += len(chunk_section)
                total_word_count += chunk.word_count
            
            # Add footer
            footer = "--- End Context from Uploaded Documents ---\n"
            context_sections.append(footer)
            
            # Combine all sections
            formatted_context = "\n".join(context_sections)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(search_result)
            
            # Count unique documents
            unique_documents = len(set(chunk.document_name for chunk in sorted_chunks))
            
            assembled_context = AssembledContext(
                formatted_context=formatted_context,
                source_count=len(sorted_chunks),
                document_count=unique_documents,
                total_word_count=total_word_count,
                confidence_level=confidence_level,
                context_metadata={
                    'assembly_timestamp': datetime.now().isoformat(),
                    'original_query': search_result.query,
                    'highest_relevance': search_result.highest_relevance_score,
                    'average_relevance': search_result.average_relevance_score,
                    'context_length': len(formatted_context),
                    'chunks_included': len(sorted_chunks),
                    'chunks_available': len(search_result.chunks),
                    'truncated': len(sorted_chunks) < len(search_result.chunks)
                },
                citation_map=citation_map
            )
            
            self.logger.info(f"✅ Context assembled: {len(sorted_chunks)} chunks, {unique_documents} documents, {confidence_level} confidence")
            return assembled_context
            
        except Exception as e:
            self.logger.error(f"❌ Context assembly failed: {e}")
            return self._create_empty_context(search_result.query)
    
    def _sort_chunks_for_context(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Sort chunks for optimal context presentation."""
        # Sort by: 1) Similarity score (desc), 2) Priority score (desc), 3) Document name
        return sorted(chunks, key=lambda x: (
            -x.similarity_score,
            -x.priority_score,
            x.document_name,
            x.block_number or 0
        ))
    
    def _format_chunk_section(self, chunk: DocumentChunk, citation_ref: str) -> str:
        """Format a single chunk into a context section."""
        # FIXED: Get document name from metadata if chunk.document_name is empty/unknown
        document_name = chunk.document_name

        # If document name is empty, unknown, or generic, try to get from metadata
        if not document_name or document_name.lower() in ['unknown', '', 'unknown document']:
            if hasattr(chunk, 'metadata') and chunk.metadata:
                # Try various metadata fields for the filename
                document_name = (
                    chunk.metadata.get('extra_filename') or
                    chunk.metadata.get('filename') or
                    chunk.metadata.get('source_type') or
                    chunk.metadata.get('file_name') or
                    'Unknown Document'
                )

        # Create source line with metadata
        source_line = f"[Source: {citation_ref} - '{document_name}'"

        if chunk.block_number is not None:
            source_line += f", Block: {chunk.block_number}"

        source_line += f", Relevance: {chunk.similarity_score:.2f}"

        if chunk.content_type:
            source_line += f", Type: {chunk.content_type}"

        source_line += "]\n"
        
        # Clean and format content
        cleaned_content = self._clean_content(chunk.content)
        
        # Create complete section
        section = f"{source_line}{cleaned_content}\n"
        
        return section
    
    def _clean_content(self, content: str) -> str:
        """Clean and format content for LLM consumption."""
        # Remove excessive whitespace
        cleaned = " ".join(content.split())
        
        # Ensure reasonable length
        if len(cleaned) > 500:
            cleaned = cleaned[:497] + "..."
        
        return cleaned
    
    def _create_full_citation(self, chunk: DocumentChunk) -> str:
        """Create a full citation string for the chunk."""
        citation_parts = [f"'{chunk.document_name}'"]
        
        if chunk.block_number is not None:
            citation_parts.append(f"Block {chunk.block_number}")
        
        if chunk.content_type and chunk.content_type != "unknown":
            citation_parts.append(f"({chunk.content_type})")
        
        citation_parts.append(f"Relevance: {chunk.similarity_score:.2f}")
        
        return ", ".join(citation_parts)
    
    def _determine_confidence_level(self, search_result: DocumentSearchResult) -> str:
        """Determine overall confidence level for the assembled context."""
        if search_result.highest_relevance_score >= 0.85:
            return "HIGH"
        elif search_result.highest_relevance_score >= 0.65:
            return "MEDIUM"
        elif search_result.highest_relevance_score >= 0.45:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _create_empty_context(self, query: str) -> AssembledContext:
        """Create an empty context when no documents are found."""
        return AssembledContext(
            formatted_context="--- No relevant documents found ---\n",
            source_count=0,
            document_count=0,
            total_word_count=0,
            confidence_level="NONE",
            context_metadata={
                'assembly_timestamp': datetime.now().isoformat(),
                'original_query': query,
                'highest_relevance': 0.0,
                'average_relevance': 0.0,
                'context_length': 0,
                'chunks_included': 0,
                'chunks_available': 0,
                'truncated': False
            },
            citation_map={}
        )
    
    def create_citation_summary(self, assembled_context: AssembledContext) -> str:
        """Create a summary of citations for display to user."""
        if not assembled_context.citation_map:
            return "No sources found in uploaded documents."
        
        summary_parts = [
            f"Information retrieved from {assembled_context.source_count} sources across {assembled_context.document_count} documents:"
        ]
        
        for ref, citation in assembled_context.citation_map.items():
            summary_parts.append(f"{ref}: {citation}")
        
        return "\n".join(summary_parts)
    
    def get_context_quality_metrics(self, assembled_context: AssembledContext) -> Dict[str, Any]:
        """Get quality metrics for the assembled context."""
        return {
            'confidence_level': assembled_context.confidence_level,
            'source_diversity': assembled_context.document_count,
            'content_richness': assembled_context.total_word_count,
            'relevance_score': assembled_context.context_metadata.get('highest_relevance', 0.0),
            'completeness': not assembled_context.context_metadata.get('truncated', False),
            'context_length': assembled_context.context_metadata.get('context_length', 0)
        }
