#!/usr/bin/env python3
"""
V2 Query Handler for SAM
Handles document queries using v1/v2 RAG pipeline routing with chat integration.
"""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def query_documents_v2(query: str, 
                      session_id: str = None, 
                      force_pipeline: str = None,
                      max_results: int = 10) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Query documents using v2 RAG pipeline with automatic routing.
    
    Args:
        query: User question
        session_id: Session identifier
        force_pipeline: Force specific pipeline ('v1_chunking' or 'v2_muvera')
        max_results: Maximum number of results
        
    Returns:
        Tuple of (success, response_text, metadata)
    """
    try:
        logger.info(f"🔍 V2 document query: '{query[:50]}...'")
        
        # Route query through RAG pipeline router
        from sam.document_rag.rag_pipeline_router import route_rag_query
        
        routing_result = route_rag_query(query, force_pipeline)
        
        if not routing_result:
            return False, "Failed to route query to RAG pipeline", {}
        
        # Extract RAG result
        rag_result = routing_result.rag_result
        pipeline_used = routing_result.selected_pipeline.value
        
        logger.info(f"📊 Pipeline used: {pipeline_used}, fallback: {routing_result.fallback_used}")
        
        # Process result based on pipeline type
        if pipeline_used == "v2_muvera":
            return _process_v2_result(rag_result, routing_result, session_id)
        else:
            return _process_v1_result(rag_result, routing_result, session_id)
            
    except Exception as e:
        logger.error(f"❌ V2 document query failed: {e}")
        return False, f"Query processing failed: {str(e)}", {'error': str(e)}

def _process_v2_result(rag_result, routing_result, session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Process v2 RAG pipeline result."""
    try:
        if not rag_result.success:
            error_msg = rag_result.error_message or "Unknown v2 pipeline error"
            return False, f"v2 pipeline failed: {error_msg}", {
                'pipeline': 'v2_muvera',
                'error': error_msg,
                'session_id': session_id
            }
        
        # Check if documents were found
        if rag_result.document_count == 0:
            return True, "No relevant documents found in your uploaded files.", {
                'pipeline': 'v2_muvera',
                'document_count': 0,
                'context_available': False,
                'session_id': session_id,
                'processing_time': rag_result.total_time,
                'fallback_used': routing_result.fallback_used
            }
        
        # Format response with context
        context = rag_result.formatted_context
        if not context:
            return False, "Failed to generate document context", {
                'pipeline': 'v2_muvera',
                'error': 'No context generated',
                'session_id': session_id
            }
        
        # Create response message
        response_parts = []
        response_parts.append("Based on your uploaded documents:")
        response_parts.append("")
        response_parts.append(context)
        response_parts.append("")
        
        # Add source attribution
        citations = _get_v2_citations(rag_result)
        if citations:
            response_parts.append("Sources:")
            for citation in citations:
                response_parts.append(f"• {citation}")
        
        response_text = "\n".join(response_parts)
        
        # Create metadata
        metadata = {
            'pipeline': 'v2_muvera',
            'success': True,
            'document_count': rag_result.document_count,
            'context_length': rag_result.context_length,
            'context_available': True,
            'source_documents': rag_result.source_documents,
            'similarity_scores': rag_result.similarity_scores,
            'processing_time': rag_result.total_time,
            'retrieval_time': rag_result.retrieval_time,
            'context_assembly_time': rag_result.context_assembly_time,
            'session_id': session_id,
            'fallback_used': routing_result.fallback_used,
            'routing_reason': routing_result.routing_reason,
            'citations': citations
        }
        
        logger.info(f"✅ v2 query successful: {rag_result.document_count} docs, {rag_result.total_time:.3f}s")
        
        return True, response_text, metadata
        
    except Exception as e:
        logger.error(f"❌ Failed to process v2 result: {e}")
        return False, f"Failed to process v2 result: {str(e)}", {
            'pipeline': 'v2_muvera',
            'error': str(e),
            'session_id': session_id
        }

def _process_v1_result(rag_result, routing_result, session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Process v1 RAG pipeline result."""
    try:
        # v1 result is a dictionary
        if not rag_result.get('success', True):
            error_msg = rag_result.get('error', 'Unknown v1 pipeline error')
            return False, f"v1 pipeline failed: {error_msg}", {
                'pipeline': 'v1_chunking',
                'error': error_msg,
                'session_id': session_id
            }
        
        # Check if document context is available
        use_document_context = rag_result.get('use_document_context', False)
        document_context = rag_result.get('document_context')
        
        if not use_document_context or not document_context:
            return True, "No relevant documents found in your uploaded files.", {
                'pipeline': 'v1_chunking',
                'document_count': 0,
                'context_available': False,
                'session_id': session_id,
                'fallback_used': routing_result.fallback_used
            }
        
        # Format v1 response
        formatted_context = document_context.get('formatted_context', '')
        if not formatted_context:
            return False, "Failed to generate document context", {
                'pipeline': 'v1_chunking',
                'error': 'No context generated',
                'session_id': session_id
            }
        
        # Create response message
        response_parts = []
        response_parts.append("Based on your uploaded documents:")
        response_parts.append("")
        response_parts.append(formatted_context)
        response_parts.append("")
        
        # Add v1 source attribution
        source_count = document_context.get('source_count', 0)
        document_count = document_context.get('document_count', 0)
        
        if source_count > 0:
            response_parts.append(f"Sources: {source_count} sources from {document_count} documents")
        
        response_text = "\n".join(response_parts)
        
        # Create metadata
        metadata = {
            'pipeline': 'v1_chunking',
            'success': True,
            'document_count': document_count,
            'source_count': source_count,
            'context_available': True,
            'context_length': len(formatted_context),
            'session_id': session_id,
            'fallback_used': routing_result.fallback_used,
            'routing_reason': routing_result.routing_reason,
            'routing_decision': rag_result.get('routing_decision', {}),
            'document_context': document_context
        }
        
        logger.info(f"✅ v1 query successful: {document_count} docs, {source_count} sources")
        
        return True, response_text, metadata
        
    except Exception as e:
        logger.error(f"❌ Failed to process v1 result: {e}")
        return False, f"Failed to process v1 result: {str(e)}", {
            'pipeline': 'v1_chunking',
            'error': str(e),
            'session_id': session_id
        }

def _get_v2_citations(rag_result) -> list:
    """Get formatted citations from v2 RAG result."""
    try:
        from sam.storage import get_v2_storage_manager
        
        citations = []
        storage_manager = get_v2_storage_manager()
        
        for i, doc_id in enumerate(rag_result.source_documents, 1):
            try:
                record = storage_manager.retrieve_document(doc_id)
                if record:
                    score = rag_result.similarity_scores.get(doc_id, 0.0)
                    citation = f"{record.filename}"
                    if score > 0:
                        citation += f" (relevance: {score:.3f})"
                    citations.append(citation)
                else:
                    citations.append(f"Document {doc_id}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to get citation for {doc_id}: {e}")
                citations.append(f"Document {doc_id}")
        
        return citations
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to generate citations: {e}")
        return []

def get_query_handler_status() -> Dict[str, Any]:
    """Get status of the v2 query handler."""
    try:
        from sam.document_rag.rag_pipeline_router import get_rag_pipeline_router
        
        router = get_rag_pipeline_router()
        router_status = router.get_router_status()
        
        return {
            'handler_available': True,
            'router_status': router_status,
            'supported_pipelines': ['v1_chunking', 'v2_muvera'],
            'default_pipeline': router_status.get('default_pipeline', 'unknown'),
            'fallback_enabled': router_status.get('enable_fallback', False)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get query handler status: {e}")
        return {
            'handler_available': False,
            'error': str(e)
        }

def switch_pipeline(pipeline: str) -> Tuple[bool, str]:
    """
    Switch the default RAG pipeline.
    
    Args:
        pipeline: Pipeline to switch to ('v1_chunking' or 'v2_muvera')
        
    Returns:
        Tuple of (success, message)
    """
    try:
        from sam.document_rag.rag_pipeline_router import get_rag_pipeline_router, PipelineSelection
        
        if pipeline == "v1_chunking":
            selection = PipelineSelection.V1_CHUNKING
        elif pipeline == "v2_muvera":
            selection = PipelineSelection.V2_MUVERA
        else:
            return False, f"Unknown pipeline: {pipeline}"
        
        router = get_rag_pipeline_router()
        router.set_default_pipeline(selection)
        
        # Also update configuration file
        from sam.document_processing.v2_upload_handler import set_pipeline_version
        set_pipeline_version(pipeline)
        
        logger.info(f"🔧 Switched to pipeline: {pipeline}")
        return True, f"Successfully switched to {pipeline} pipeline"
        
    except Exception as e:
        logger.error(f"❌ Failed to switch pipeline: {e}")
        return False, f"Failed to switch pipeline: {str(e)}"

# Backward compatibility function
def query_pdf_for_sam_v2(query: str, session_id: str = None, pdf_name: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Backward compatible PDF query function using v2 routing.
    
    Args:
        query: Question to ask
        session_id: Session identifier
        pdf_name: PDF name (for compatibility, not used in v2)
        
    Returns:
        Tuple of (success, response, metadata)
    """
    return query_documents_v2(query, session_id)

# Main query function as specified in task52.md
def query_v2_documents(query: str, session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Main v2 document query function as specified in task52.md.
    
    Args:
        query: User question
        session_id: Session identifier
        
    Returns:
        Tuple of (success, response_text, metadata)
    """
    try:
        logger.info(f"🔍 Querying v2 documents: '{query[:50]}...'")
        
        # Force v2 pipeline
        success, response, metadata = query_documents_v2(
            query=query,
            session_id=session_id,
            force_pipeline="v2_muvera"
        )
        
        if success:
            logger.info(f"✅ v2 document query successful")
        else:
            logger.error(f"❌ v2 document query failed: {response}")
        
        return success, response, metadata
        
    except Exception as e:
        logger.error(f"❌ v2 document query failed: {e}")
        return False, f"v2 query failed: {str(e)}", {'error': str(e)}
