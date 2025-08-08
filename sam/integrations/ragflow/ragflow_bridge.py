"""
RAGFlow Bridge
High-level interface between SAM and RAGFlow.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio
from dataclasses import dataclass
from datetime import datetime

from .ragflow_client import RAGFlowClient
from .document_processor import RAGFlowDocumentProcessor
from .hybrid_retrieval import HybridRetrievalEngine
from .knowledge_sync import KnowledgeBaseSynchronizer

logger = logging.getLogger(__name__)

@dataclass
class DocumentProcessingResult:
    """Result of document processing through RAGFlow."""
    success: bool
    document_id: Optional[str] = None
    chunks_count: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class QueryResult:
    """Result of query processing through RAGFlow."""
    success: bool
    chunks: List[Dict] = None
    citations: List[str] = None
    confidence_score: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None

class RAGFlowBridge:
    """
    High-level bridge between SAM and RAGFlow.
    
    Provides simplified interface for:
    - Document upload and processing
    - Intelligent query routing
    - Hybrid retrieval operations
    - Knowledge base synchronization
    """
    
    def __init__(self, 
                 ragflow_url: str = "http://localhost:9380/api/v1",
                 api_key: Optional[str] = None,
                 knowledge_base_name: str = "sam_documents",
                 enable_sync: bool = True):
        """
        Initialize RAGFlow bridge.
        
        Args:
            ragflow_url: RAGFlow API URL
            api_key: API key for authentication
            knowledge_base_name: Default knowledge base name
            enable_sync: Enable bidirectional sync with SAM
        """
        self.client = RAGFlowClient(ragflow_url, api_key)
        self.knowledge_base_name = knowledge_base_name
        self.knowledge_base_id = None
        self.enable_sync = enable_sync
        
        # Initialize components
        self.document_processor = RAGFlowDocumentProcessor(self.client)
        self.hybrid_retrieval = HybridRetrievalEngine(self.client)
        
        if enable_sync:
            self.synchronizer = KnowledgeBaseSynchronizer(self.client)
        
        # Initialize knowledge base
        self._initialize_knowledge_base()
        
        logger.info(f"RAGFlow bridge initialized for knowledge base: {knowledge_base_name}")
    
    def _initialize_knowledge_base(self):
        """Initialize or get existing knowledge base."""
        try:
            # Try to create knowledge base (will fail if exists)
            response = self.client.create_knowledge_base(
                name=self.knowledge_base_name,
                description="SAM Document Store with RAGFlow integration",
                embedding_model="BAAI/bge-large-en-v1.5",
                chunk_template="intelligent",
                language="English"
            )
            self.knowledge_base_id = response.get('id')
            logger.info(f"Created new knowledge base: {self.knowledge_base_id}")
            
        except Exception as e:
            # Knowledge base might already exist
            logger.info(f"Knowledge base might already exist: {e}")
            # In a real implementation, we'd query for existing knowledge bases
            # For now, we'll use a default ID
            self.knowledge_base_id = "sam_documents"
    
    def process_document(self, 
                        file_path: Union[str, Path],
                        chunk_template: Optional[str] = None,
                        metadata: Optional[Dict] = None,
                        sync_to_sam: bool = True) -> DocumentProcessingResult:
        """
        Process document through RAGFlow with enhanced understanding.
        
        Args:
            file_path: Path to document file
            chunk_template: Override chunking template
            metadata: Additional metadata
            sync_to_sam: Whether to sync results back to SAM
            
        Returns:
            Document processing result
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing document through RAGFlow: {file_path}")
            
            # Upload and process document
            upload_result = self.document_processor.process_document(
                knowledge_base_id=self.knowledge_base_id,
                file_path=file_path,
                chunk_template=chunk_template,
                metadata=metadata
            )
            
            if not upload_result.get('success'):
                return DocumentProcessingResult(
                    success=False,
                    error_message=upload_result.get('error', 'Upload failed')
                )
            
            document_id = upload_result.get('document_id')
            
            # Wait for processing to complete
            processing_result = self.document_processor.wait_for_processing(document_id)
            
            if not processing_result.get('success'):
                return DocumentProcessingResult(
                    success=False,
                    document_id=document_id,
                    error_message=processing_result.get('error', 'Processing failed')
                )
            
            # Get processing statistics
            chunks_count = processing_result.get('chunks_count', 0)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Sync to SAM if enabled
            if sync_to_sam and self.enable_sync:
                try:
                    self.synchronizer.sync_document_to_sam(document_id)
                    logger.info(f"Document synced to SAM: {document_id}")
                except Exception as e:
                    logger.warning(f"Failed to sync document to SAM: {e}")
            
            return DocumentProcessingResult(
                success=True,
                document_id=document_id,
                chunks_count=chunks_count,
                processing_time=processing_time,
                metadata=processing_result.get('metadata')
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return DocumentProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def query_documents(self, 
                       query: str,
                       max_results: int = 5,
                       similarity_threshold: float = 0.3,
                       use_hybrid_retrieval: bool = True,
                       include_sam_results: bool = True) -> QueryResult:
        """
        Query documents using RAGFlow's advanced retrieval.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            use_hybrid_retrieval: Use hybrid retrieval strategy
            include_sam_results: Include results from SAM memory store
            
        Returns:
            Query result with chunks and citations
        """
        start_time = datetime.now()
        
        try:
            logger.debug(f"Querying RAGFlow: {query}")
            
            if use_hybrid_retrieval:
                # Use hybrid retrieval engine
                result = self.hybrid_retrieval.query(
                    query=query,
                    knowledge_base_id=self.knowledge_base_id,
                    max_results=max_results,
                    similarity_threshold=similarity_threshold,
                    include_sam_results=include_sam_results
                )
            else:
                # Use direct RAGFlow query
                result = self.client.query_knowledge_base(
                    knowledge_base_id=self.knowledge_base_id,
                    query=query,
                    max_results=max_results,
                    similarity_threshold=similarity_threshold
                )
            
            if not result.get('success'):
                return QueryResult(
                    success=False,
                    error_message=result.get('error', 'Query failed')
                )
            
            # Extract results
            chunks = result.get('chunks', [])
            citations = result.get('citations', [])
            confidence_score = result.get('confidence_score', 0.0)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                success=True,
                chunks=chunks,
                citations=citations,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return QueryResult(
                success=False,
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def get_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """
        Get document chunks for visual intervention.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document chunks with metadata
        """
        try:
            return self.client.get_document_chunks(document_id)
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_chunk(self, 
                    chunk_id: str,
                    content: Optional[str] = None,
                    keywords: Optional[List[str]] = None,
                    sync_to_sam: bool = True) -> Dict[str, Any]:
        """
        Update document chunk with manual intervention.
        
        Args:
            chunk_id: Chunk ID to update
            content: New chunk content
            keywords: Keywords to add
            sync_to_sam: Whether to sync changes to SAM
            
        Returns:
            Update result
        """
        try:
            result = self.client.update_chunk(
                chunk_id=chunk_id,
                content=content,
                keywords=keywords
            )
            
            # Sync changes to SAM if enabled
            if sync_to_sam and self.enable_sync and result.get('success'):
                try:
                    self.synchronizer.sync_chunk_to_sam(chunk_id)
                    logger.info(f"Chunk changes synced to SAM: {chunk_id}")
                except Exception as e:
                    logger.warning(f"Failed to sync chunk changes to SAM: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update chunk: {e}")
            return {'success': False, 'error': str(e)}
    
    def sync_with_sam(self, direction: str = "bidirectional") -> Dict[str, Any]:
        """
        Manually trigger synchronization with SAM.
        
        Args:
            direction: Sync direction ("to_sam", "from_sam", "bidirectional")
            
        Returns:
            Synchronization result
        """
        if not self.enable_sync:
            return {'success': False, 'error': 'Synchronization disabled'}
        
        try:
            return self.synchronizer.sync(direction=direction)
        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of RAGFlow integration.
        
        Returns:
            Health status information
        """
        try:
            ragflow_healthy = self.client.health_check()
            
            return {
                'success': True,
                'ragflow_healthy': ragflow_healthy,
                'knowledge_base_id': self.knowledge_base_id,
                'sync_enabled': self.enable_sync,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
