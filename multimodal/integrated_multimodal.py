"""
Integrated Multimodal System for SAM
Combines all Sprint 9 multimodal capabilities into a unified system.

Sprint 9: Multimodal Input, Web Search, and Visual Reasoning Integration
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import base64

from .ingestion_engine import MultimodalIngestionEngine, get_ingestion_engine
from .local_search import LocalFileSearchEngine, get_local_search_engine
from .web_search import WebSearchEngine, SearchEngine, get_web_search_engine
from .reasoning_engine import MultimodalReasoningEngine, get_reasoning_engine

logger = logging.getLogger(__name__)

@dataclass
class MultimodalRequest:
    """Request for multimodal processing."""
    request_id: str
    user_id: str
    session_id: str
    query: str
    text_inputs: List[str]
    image_data: List[Dict[str, Any]]  # {'data': bytes/base64, 'filename': str}
    audio_data: List[Dict[str, Any]]  # {'data': bytes, 'filename': str}
    document_data: List[Dict[str, Any]]  # {'data': bytes, 'filename': str}
    enable_web_search: bool
    enable_local_search: bool
    context: Dict[str, Any]

@dataclass
class MultimodalResponse:
    """Response from multimodal processing."""
    request_id: str
    answer: str
    confidence_level: str
    overall_confidence: float
    source_attributions: List[str]
    reasoning_trace: str
    media_processed: Dict[str, int]
    processing_time_ms: int
    created_at: str

class IntegratedMultimodalSystem:
    """
    Unified multimodal system that integrates all Sprint 9 capabilities.
    """
    
    def __init__(self, enable_web_access: bool = False,
                 knowledge_directory: str = "knowledge",
                 storage_directory: str = "multimodal_storage"):
        """
        Initialize the integrated multimodal system.
        
        Args:
            enable_web_access: Whether web search is enabled
            knowledge_directory: Directory for local knowledge files
            storage_directory: Directory for multimodal storage
        """
        # Initialize all multimodal components
        self.ingestion_engine = get_ingestion_engine(storage_directory=storage_directory)
        self.local_search_engine = get_local_search_engine(knowledge_directory=knowledge_directory)
        self.web_search_engine = get_web_search_engine(enable_web_access=enable_web_access)
        self.reasoning_engine = get_reasoning_engine(
            ingestion_engine=self.ingestion_engine,
            local_search_engine=self.local_search_engine,
            web_search_engine=self.web_search_engine
        )
        
        # Configuration
        self.config = {
            'auto_index_knowledge': True,
            'max_concurrent_processing': 5,
            'enable_cross_modal_reasoning': True,
            'default_confidence_threshold': 0.3
        }
        
        # Auto-index knowledge directory if enabled
        if self.config['auto_index_knowledge']:
            self._auto_index_knowledge()
        
        logger.info("Integrated multimodal system initialized")
    
    def process_multimodal_request(self, request: MultimodalRequest) -> MultimodalResponse:
        """
        Process a complete multimodal request.
        
        Args:
            request: MultimodalRequest to process
            
        Returns:
            MultimodalResponse with complete results
        """
        try:
            start_time = datetime.now()
            
            logger.info(f"Processing multimodal request: {request.query[:50]}...")
            
            # Step 1: Ingest multimodal content
            media_ids = self._ingest_multimodal_content(request)
            
            # Step 2: Perform multimodal reasoning
            reasoning_response = self.reasoning_engine.reason_multimodal(
                query=request.query,
                user_id=request.user_id,
                session_id=request.session_id,
                text_inputs=request.text_inputs,
                image_ids=media_ids.get('images', []),
                audio_ids=media_ids.get('audio', []),
                document_ids=media_ids.get('documents', []),
                enable_web_search=request.enable_web_search,
                enable_local_search=request.enable_local_search
            )
            
            # Step 3: Format response
            formatted_answer = self.reasoning_engine.format_response(
                reasoning_response,
                include_reasoning_trace=True,
                include_source_details=True
            )
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Create integrated response
            response = MultimodalResponse(
                request_id=request.request_id,
                answer=formatted_answer,
                confidence_level=reasoning_response.confidence_level.value,
                overall_confidence=reasoning_response.overall_confidence,
                source_attributions=reasoning_response.source_attributions,
                reasoning_trace=self._extract_reasoning_trace(reasoning_response),
                media_processed={
                    'images': len(media_ids.get('images', [])),
                    'audio': len(media_ids.get('audio', [])),
                    'documents': len(media_ids.get('documents', []))
                },
                processing_time_ms=processing_time_ms,
                created_at=datetime.now().isoformat()
            )
            
            logger.info(f"Multimodal request completed: {processing_time_ms}ms, "
                       f"confidence: {reasoning_response.overall_confidence:.1%}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing multimodal request: {e}")
            return self._create_error_response(request, str(e))
    
    def ingest_image(self, image_data: Union[bytes, str], filename: str) -> str:
        """
        Ingest an image for processing.
        
        Args:
            image_data: Image data (bytes or base64 string)
            filename: Original filename
            
        Returns:
            Media ID
        """
        try:
            return self.ingestion_engine.ingest_image(
                image_data=image_data,
                filename=filename,
                perform_ocr=True,
                generate_caption=True
            )
            
        except Exception as e:
            logger.error(f"Error ingesting image {filename}: {e}")
            raise
    
    def ingest_audio(self, audio_data: bytes, filename: str) -> str:
        """
        Ingest an audio file for processing.
        
        Args:
            audio_data: Audio data in bytes
            filename: Original filename
            
        Returns:
            Media ID
        """
        try:
            return self.ingestion_engine.ingest_audio(
                audio_data=audio_data,
                filename=filename,
                perform_transcription=True
            )
            
        except Exception as e:
            logger.error(f"Error ingesting audio {filename}: {e}")
            raise
    
    def ingest_document(self, document_data: bytes, filename: str) -> str:
        """
        Ingest a document for processing.
        
        Args:
            document_data: Document data in bytes
            filename: Original filename
            
        Returns:
            Media ID
        """
        try:
            return self.ingestion_engine.ingest_document(
                document_data=document_data,
                filename=filename
            )
            
        except Exception as e:
            logger.error(f"Error ingesting document {filename}: {e}")
            raise
    
    def search_local_knowledge(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search through local knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            results = self.local_search_engine.search(query, max_results=max_results)
            
            # Convert to dict format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'title': result.title,
                    'filename': result.filename,
                    'content_preview': result.content_preview,
                    'confidence_score': result.confidence_score,
                    'attribution': self.local_search_engine.create_citation(result),
                    'metadata': result.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching local knowledge: {e}")
            return []
    
    def search_web(self, query: str, user_id: str, session_id: str,
                  max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            user_id: User performing the search
            session_id: Session ID
            max_results: Maximum number of results
            
        Returns:
            List of web search results
        """
        try:
            if not self.web_search_engine.enable_web_access:
                logger.warning("Web access is disabled")
                return []
            
            results, query_id = self.web_search_engine.search(
                query=query,
                user_id=user_id,
                session_id=session_id,
                search_engine=SearchEngine.DUCKDUCKGO,
                max_results=max_results
            )
            
            # Convert to dict format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'confidence_score': result.confidence_score,
                    'result_type': result.result_type.value,
                    'metadata': result.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            return []
    
    def get_media_info(self, media_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about processed media.
        
        Args:
            media_id: Media ID
            
        Returns:
            Media information or None if not found
        """
        try:
            metadata = self.ingestion_engine.get_media_metadata(media_id)
            result = self.ingestion_engine.get_processing_result(media_id)
            
            if metadata and result:
                return {
                    'media_id': media_id,
                    'filename': metadata.original_filename,
                    'media_type': metadata.media_type.value,
                    'file_size': metadata.file_size,
                    'processing_status': metadata.processing_status.value,
                    'extracted_text': result.extracted_text,
                    'caption': result.caption,
                    'transcription': result.transcription,
                    'confidence_scores': result.confidence_scores,
                    'processing_time_ms': result.processing_time_ms
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting media info: {e}")
            return None
    
    def index_knowledge_directory(self, directory_path: Optional[str] = None) -> int:
        """
        Index files in the knowledge directory.
        
        Args:
            directory_path: Optional specific directory to index
            
        Returns:
            Number of files indexed
        """
        try:
            return self.local_search_engine.index_directory(directory_path)
            
        except Exception as e:
            logger.error(f"Error indexing knowledge directory: {e}")
            return 0
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                'ingestion_engine': {
                    'media_count': len(self.ingestion_engine.media_registry),
                    'processing_results': len(self.ingestion_engine.processing_results),
                    'available': True
                },
                'local_search_engine': {
                    'indexed_items': len(self.local_search_engine.search_index),
                    'available': True
                },
                'web_search_engine': {
                    'web_access_enabled': self.web_search_engine.enable_web_access,
                    'search_queries': len(self.web_search_engine.search_queries),
                    'available': True
                },
                'reasoning_engine': {
                    'reasoning_sessions': len(self.reasoning_engine.reasoning_sessions),
                    'cross_modal_fusion_enabled': self.reasoning_engine.config['enable_cross_modal_fusion'],
                    'available': True
                },
                'system_ready': True
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e), 'system_ready': False}
    
    def _ingest_multimodal_content(self, request: MultimodalRequest) -> Dict[str, List[str]]:
        """Ingest all multimodal content from request."""
        media_ids = {
            'images': [],
            'audio': [],
            'documents': []
        }
        
        try:
            # Process images
            for image_item in request.image_data:
                try:
                    media_id = self.ingestion_engine.ingest_image(
                        image_data=image_item['data'],
                        filename=image_item['filename'],
                        perform_ocr=True,
                        generate_caption=True
                    )
                    media_ids['images'].append(media_id)
                except Exception as e:
                    logger.error(f"Error processing image {image_item['filename']}: {e}")
            
            # Process audio
            for audio_item in request.audio_data:
                try:
                    media_id = self.ingestion_engine.ingest_audio(
                        audio_data=audio_item['data'],
                        filename=audio_item['filename'],
                        perform_transcription=True
                    )
                    media_ids['audio'].append(media_id)
                except Exception as e:
                    logger.error(f"Error processing audio {audio_item['filename']}: {e}")
            
            # Process documents
            for doc_item in request.document_data:
                try:
                    media_id = self.ingestion_engine.ingest_document(
                        document_data=doc_item['data'],
                        filename=doc_item['filename']
                    )
                    media_ids['documents'].append(media_id)
                except Exception as e:
                    logger.error(f"Error processing document {doc_item['filename']}: {e}")
            
            return media_ids
            
        except Exception as e:
            logger.error(f"Error ingesting multimodal content: {e}")
            return media_ids
    
    def _extract_reasoning_trace(self, reasoning_response) -> str:
        """Extract reasoning trace from response."""
        try:
            trace_parts = []
            
            for step in reasoning_response.reasoning_steps:
                trace_parts.append(f"{step.step_type}: {step.reasoning_trace}")
            
            return " → ".join(trace_parts)
            
        except Exception as e:
            logger.error(f"Error extracting reasoning trace: {e}")
            return "Reasoning trace unavailable"
    
    def _auto_index_knowledge(self):
        """Automatically index the knowledge directory."""
        try:
            indexed_count = self.local_search_engine.index_directory()
            logger.info(f"Auto-indexed {indexed_count} knowledge files")
            
        except Exception as e:
            logger.error(f"Error auto-indexing knowledge: {e}")
    
    def _create_error_response(self, request: MultimodalRequest, error_message: str) -> MultimodalResponse:
        """Create an error response."""
        return MultimodalResponse(
            request_id=request.request_id,
            answer=f"Error processing multimodal request: {error_message}",
            confidence_level="very_low",
            overall_confidence=0.0,
            source_attributions=[],
            reasoning_trace="Error occurred during processing",
            media_processed={'images': 0, 'audio': 0, 'documents': 0},
            processing_time_ms=0,
            created_at=datetime.now().isoformat()
        )

# Global integrated multimodal system instance
_integrated_multimodal_system = None

def get_integrated_multimodal_system(enable_web_access: bool = False,
                                   knowledge_directory: str = "knowledge",
                                   storage_directory: str = "multimodal_storage") -> IntegratedMultimodalSystem:
    """Get or create a global integrated multimodal system instance."""
    global _integrated_multimodal_system
    
    if _integrated_multimodal_system is None:
        _integrated_multimodal_system = IntegratedMultimodalSystem(
            enable_web_access=enable_web_access,
            knowledge_directory=knowledge_directory,
            storage_directory=storage_directory
        )
    
    return _integrated_multimodal_system
