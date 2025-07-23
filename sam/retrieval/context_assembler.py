#!/usr/bin/env python3
"""
V2 Context Assembler for SAM MUVERA Pipeline
Assembles retrieved documents into structured context for LLM consumption.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global context assembler instance
_v2_context_assembler = None

@dataclass
class ContextConfig:
    """Configuration for context assembly."""
    max_context_length: int = 4000    # Maximum context length in characters
    max_documents: int = 10           # Maximum documents to include
    include_metadata: bool = True     # Include document metadata
    include_scores: bool = True       # Include similarity scores
    context_format: str = "structured"  # Context format ('structured', 'simple')
    enable_summarization: bool = False   # Enable content summarization
    chunk_overlap_handling: str = "merge"  # How to handle overlapping chunks

@dataclass
class V2AssembledContext:
    """Assembled context from v2 retrieval."""
    formatted_context: str           # Formatted context for LLM
    source_documents: List[str]      # Source document IDs
    document_count: int              # Number of documents included
    total_length: int                # Total context length
    similarity_scores: Dict[str, float]  # Document similarity scores
    assembly_time: float             # Time taken for assembly
    truncated: bool                  # Whether context was truncated
    metadata: Dict[str, Any]         # Additional metadata

class V2ContextAssembler:
    """
    Context assembler for v2 retrieval results.
    
    Takes retrieved documents and assembles them into structured context
    suitable for LLM consumption with proper citations and metadata.
    """
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize the v2 context assembler.
        
        Args:
            config: Context assembly configuration
        """
        self.config = config or ContextConfig()
        
        # Components (loaded lazily)
        self.storage_manager = None
        self.is_initialized = False
        
        logger.info(f"📝 V2ContextAssembler initialized")
        logger.info(f"📏 Max length: {self.config.max_context_length}, max docs: {self.config.max_documents}")
    
    def _initialize_components(self) -> bool:
        """Initialize context assembly components."""
        if self.is_initialized:
            return True
        
        try:
            logger.info("🔄 Initializing v2 context assembly components...")
            
            # Initialize storage manager
            from sam.storage import get_v2_storage_manager
            self.storage_manager = get_v2_storage_manager()
            
            self.is_initialized = True
            logger.info("✅ V2 context assembly components initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize v2 context assembly components: {e}")
            return False
    
    def assemble_context(self, 
                        query: str,
                        document_ids: List[str],
                        similarity_scores: Optional[Dict[str, float]] = None) -> Optional[V2AssembledContext]:
        """
        Assemble context from retrieved documents.
        
        Args:
            query: Original user query
            document_ids: List of document IDs to include
            similarity_scores: Optional similarity scores for documents
            
        Returns:
            V2AssembledContext with assembled context
        """
        try:
            start_time = time.time()
            
            logger.debug(f"📝 Assembling context from {len(document_ids)} documents")
            
            # Initialize components
            if not self._initialize_components():
                logger.error("❌ Failed to initialize context assembly components")
                return None
            
            # Limit number of documents
            if len(document_ids) > self.config.max_documents:
                logger.warning(f"⚠️  Limiting context to {self.config.max_documents} documents")
                document_ids = document_ids[:self.config.max_documents]
            
            # Retrieve document records
            document_records = []
            for doc_id in document_ids:
                record = self.storage_manager.retrieve_document(doc_id)
                if record:
                    document_records.append(record)
                else:
                    logger.warning(f"⚠️  Failed to retrieve document: {doc_id}")
            
            if not document_records:
                logger.error("❌ No valid documents found for context assembly")
                return None
            
            # Assemble context based on format
            if self.config.context_format == "structured":
                formatted_context = self._assemble_structured_context(
                    query, document_records, similarity_scores
                )
            else:
                formatted_context = self._assemble_simple_context(
                    query, document_records, similarity_scores
                )
            
            # Check if truncation is needed
            truncated = False
            if len(formatted_context) > self.config.max_context_length:
                formatted_context = self._truncate_context(formatted_context)
                truncated = True
                logger.debug(f"📏 Context truncated to {len(formatted_context)} characters")
            
            assembly_time = time.time() - start_time
            
            # Create result
            result = V2AssembledContext(
                formatted_context=formatted_context,
                source_documents=[record.document_id for record in document_records],
                document_count=len(document_records),
                total_length=len(formatted_context),
                similarity_scores=similarity_scores or {},
                assembly_time=assembly_time,
                truncated=truncated,
                metadata={
                    'query': query,
                    'config': self.config.__dict__,
                    'assembly_timestamp': time.time(),
                    'original_document_count': len(document_ids),
                    'retrieved_document_count': len(document_records)
                }
            )
            
            logger.debug(f"✅ Context assembled: {len(document_records)} docs, {len(formatted_context)} chars, {assembly_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Context assembly failed: {e}")
            return None
    
    def _assemble_structured_context(self, 
                                   query: str,
                                   document_records: List[Any],
                                   similarity_scores: Optional[Dict[str, float]]) -> str:
        """Assemble structured context with proper formatting."""
        context_parts = []
        
        # Add header
        context_parts.append("=== RETRIEVED DOCUMENTS ===")
        context_parts.append(f"Query: {query}")
        context_parts.append(f"Documents found: {len(document_records)}")
        context_parts.append("")
        
        # Add each document
        for i, record in enumerate(document_records, 1):
            doc_id = record.document_id
            similarity_score = similarity_scores.get(doc_id, 0.0) if similarity_scores else 0.0
            
            # Document header
            context_parts.append(f"--- Document {i}: {record.filename} ---")
            
            if self.config.include_scores and similarity_score > 0:
                context_parts.append(f"Relevance Score: {similarity_score:.4f}")
            
            if self.config.include_metadata:
                context_parts.append(f"Document ID: {doc_id}")
                context_parts.append(f"Tokens: {record.num_tokens}")
                context_parts.append(f"Processing Date: {record.processing_timestamp}")
            
            context_parts.append("")
            
            # Document content
            content = record.text_content
            if len(content) > 2000:  # Limit per document
                content = content[:2000] + "... [content truncated]"
            
            context_parts.append(content)
            context_parts.append("")
        
        # Add footer
        context_parts.append("=== END RETRIEVED DOCUMENTS ===")
        
        return "\n".join(context_parts)
    
    def _assemble_simple_context(self, 
                                query: str,
                                document_records: List[Any],
                                similarity_scores: Optional[Dict[str, float]]) -> str:
        """Assemble simple context without extensive formatting."""
        context_parts = []
        
        # Add query context
        context_parts.append(f"Based on the query '{query}', here are the relevant documents:")
        context_parts.append("")
        
        # Add each document simply
        for i, record in enumerate(document_records, 1):
            doc_id = record.document_id
            similarity_score = similarity_scores.get(doc_id, 0.0) if similarity_scores else 0.0
            
            # Simple document header
            header = f"[Document {i}: {record.filename}]"
            if self.config.include_scores and similarity_score > 0:
                header += f" (Relevance: {similarity_score:.3f})"
            
            context_parts.append(header)
            
            # Document content
            content = record.text_content
            if len(content) > 1500:  # Limit per document for simple format
                content = content[:1500] + "..."
            
            context_parts.append(content)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _truncate_context(self, context: str) -> str:
        """Truncate context to fit within length limits."""
        if len(context) <= self.config.max_context_length:
            return context
        
        # Try to truncate at sentence boundaries
        truncated = context[:self.config.max_context_length]
        
        # Find last sentence ending
        last_sentence = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence > self.config.max_context_length * 0.8:  # If we can keep 80% with sentence boundary
            truncated = truncated[:last_sentence + 1]
        
        # Add truncation indicator
        truncated += "\n\n[Context truncated due to length limits]"
        
        return truncated
    
    def create_citation_map(self, document_records: List[Any]) -> Dict[str, str]:
        """Create a citation map for document references."""
        citation_map = {}
        
        for i, record in enumerate(document_records, 1):
            citation_key = f"doc_{i}"
            citation_value = f"{record.filename} (ID: {record.document_id})"
            citation_map[citation_key] = citation_value
        
        return citation_map
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context assembler statistics."""
        try:
            return {
                'config': self.config.__dict__,
                'is_initialized': self.is_initialized,
                'components': {
                    'storage_manager': self.storage_manager is not None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get context stats: {e}")
            return {'error': str(e)}
    
    def preview_context(self, 
                       document_ids: List[str],
                       max_preview_length: int = 500) -> str:
        """
        Generate a preview of what the context would look like.
        
        Args:
            document_ids: List of document IDs
            max_preview_length: Maximum preview length
            
        Returns:
            Preview string
        """
        try:
            if not self._initialize_components():
                return "Error: Failed to initialize components"
            
            preview_parts = []
            preview_parts.append(f"Context Preview ({len(document_ids)} documents):")
            preview_parts.append("")
            
            for i, doc_id in enumerate(document_ids[:3], 1):  # Preview first 3 docs
                record = self.storage_manager.retrieve_document(doc_id)
                if record:
                    preview_parts.append(f"{i}. {record.filename} ({record.num_tokens} tokens)")
                    content_preview = record.text_content[:100] + "..." if len(record.text_content) > 100 else record.text_content
                    preview_parts.append(f"   {content_preview}")
                else:
                    preview_parts.append(f"{i}. [Document not found: {doc_id}]")
            
            if len(document_ids) > 3:
                preview_parts.append(f"... and {len(document_ids) - 3} more documents")
            
            preview = "\n".join(preview_parts)
            
            if len(preview) > max_preview_length:
                preview = preview[:max_preview_length] + "..."
            
            return preview
            
        except Exception as e:
            logger.error(f"❌ Context preview failed: {e}")
            return f"Error generating preview: {str(e)}"

def get_v2_context_assembler(config: Optional[ContextConfig] = None) -> V2ContextAssembler:
    """
    Get or create a v2 context assembler instance.
    
    Args:
        config: Context assembly configuration
        
    Returns:
        V2ContextAssembler instance
    """
    global _v2_context_assembler
    
    if _v2_context_assembler is None:
        _v2_context_assembler = V2ContextAssembler(config)
    
    return _v2_context_assembler

def assemble_context_v2(query: str,
                       document_ids: List[str],
                       similarity_scores: Optional[Dict[str, float]] = None) -> Optional[V2AssembledContext]:
    """
    Convenience function to assemble context using v2 pipeline.
    
    Args:
        query: Original user query
        document_ids: List of document IDs to include
        similarity_scores: Optional similarity scores for documents
        
    Returns:
        V2AssembledContext with assembled context
    """
    assembler = get_v2_context_assembler()
    return assembler.assemble_context(query, document_ids, similarity_scores)
