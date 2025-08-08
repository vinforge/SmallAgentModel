"""
RAGFlow Document Processor
Enhanced document processing using RAGFlow's DeepDoc capabilities.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .ragflow_client import RAGFlowClient

logger = logging.getLogger(__name__)

class ChunkTemplate(Enum):
    """Available chunking templates in RAGFlow."""
    INTELLIGENT = "intelligent"
    MANUAL = "manual"
    NAIVE = "naive"
    PAPER = "paper"
    BOOK = "book"
    LAWS = "laws"
    PRESENTATION = "presentation"
    PICTURE = "picture"
    ONE = "one"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    TABLE = "table"
    RESUME = "resume"
    EMAIL = "email"
    QA = "qa"

class ProcessingStatus(Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class DocumentMetadata:
    """Enhanced document metadata."""
    filename: str
    file_size: int
    file_type: str
    upload_time: str
    chunk_template: str
    language: str
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    has_images: bool = False
    has_tables: bool = False
    processing_options: Optional[Dict] = None

class RAGFlowDocumentProcessor:
    """
    Enhanced document processor using RAGFlow's DeepDoc capabilities.
    
    Features:
    - Deep document understanding with layout analysis
    - Multi-modal processing (text, images, tables)
    - Template-based intelligent chunking
    - OCR and document parsing for complex formats
    - Visual chunking intervention support
    """
    
    def __init__(self, client: RAGFlowClient):
        """
        Initialize document processor.
        
        Args:
            client: RAGFlow client instance
        """
        self.client = client
        self.supported_formats = {
            '.pdf', '.doc', '.docx', '.txt', '.md', '.mdx',
            '.csv', '.xlsx', '.xls', '.jpeg', '.jpg', '.png', 
            '.tif', '.gif', '.ppt', '.pptx'
        }
        
        logger.info("RAGFlow document processor initialized")
    
    def get_optimal_chunk_template(self, file_path: Union[str, Path]) -> ChunkTemplate:
        """
        Determine optimal chunking template based on file type and content.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Recommended chunking template
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        # Template mapping based on file type
        template_mapping = {
            '.pdf': ChunkTemplate.INTELLIGENT,
            '.doc': ChunkTemplate.INTELLIGENT,
            '.docx': ChunkTemplate.INTELLIGENT,
            '.txt': ChunkTemplate.NAIVE,
            '.md': ChunkTemplate.BOOK,
            '.mdx': ChunkTemplate.BOOK,
            '.csv': ChunkTemplate.TABLE,
            '.xlsx': ChunkTemplate.TABLE,
            '.xls': ChunkTemplate.TABLE,
            '.ppt': ChunkTemplate.PRESENTATION,
            '.pptx': ChunkTemplate.PRESENTATION,
            '.jpg': ChunkTemplate.PICTURE,
            '.jpeg': ChunkTemplate.PICTURE,
            '.png': ChunkTemplate.PICTURE,
            '.gif': ChunkTemplate.PICTURE,
            '.tif': ChunkTemplate.PICTURE
        }
        
        # Check for specific document types based on filename
        filename_lower = file_path.name.lower()
        
        if any(keyword in filename_lower for keyword in ['resume', 'cv']):
            return ChunkTemplate.RESUME
        elif any(keyword in filename_lower for keyword in ['law', 'legal', 'regulation']):
            return ChunkTemplate.LAWS
        elif any(keyword in filename_lower for keyword in ['paper', 'research', 'journal']):
            return ChunkTemplate.PAPER
        elif any(keyword in filename_lower for keyword in ['email', 'mail']):
            return ChunkTemplate.EMAIL
        elif any(keyword in filename_lower for keyword in ['qa', 'faq', 'question']):
            return ChunkTemplate.QA
        
        return template_mapping.get(file_ext, ChunkTemplate.INTELLIGENT)
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate file for processing.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Validation result with file metadata
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            return {
                'valid': False,
                'error': f'File not found: {file_path}'
            }
        
        # Check file extension
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            return {
                'valid': False,
                'error': f'Unsupported file format: {file_ext}',
                'supported_formats': list(self.supported_formats)
            }
        
        # Get file metadata
        file_stats = file_path.stat()
        file_size = file_stats.st_size
        
        # Check file size (100MB limit)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            return {
                'valid': False,
                'error': f'File too large: {file_size / (1024*1024):.1f}MB (max: 100MB)'
            }
        
        return {
            'valid': True,
            'filename': file_path.name,
            'file_size': file_size,
            'file_type': file_ext,
            'recommended_template': self.get_optimal_chunk_template(file_path).value
        }
    
    def process_document(self, 
                        knowledge_base_id: str,
                        file_path: Union[str, Path],
                        chunk_template: Optional[str] = None,
                        metadata: Optional[Dict] = None,
                        processing_options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process document through RAGFlow with enhanced understanding.
        
        Args:
            knowledge_base_id: Target knowledge base ID
            file_path: Path to document file
            chunk_template: Override chunking template
            metadata: Additional metadata
            processing_options: Processing configuration options
            
        Returns:
            Processing result with document ID
        """
        # Validate file
        validation = self.validate_file(file_path)
        if not validation['valid']:
            return {
                'success': False,
                'error': validation['error']
            }
        
        # Determine chunk template
        if not chunk_template:
            chunk_template = validation['recommended_template']
        
        # Prepare enhanced metadata
        enhanced_metadata = {
            'filename': validation['filename'],
            'file_size': validation['file_size'],
            'file_type': validation['file_type'],
            'chunk_template': chunk_template,
            'upload_source': 'sam_ragflow_integration',
            'processing_options': processing_options or {}
        }
        
        if metadata:
            enhanced_metadata.update(metadata)
        
        try:
            logger.info(f"Processing document with template '{chunk_template}': {file_path}")
            
            # Upload document to RAGFlow
            upload_result = self.client.upload_document(
                knowledge_base_id=knowledge_base_id,
                file_path=file_path,
                chunk_template=chunk_template,
                metadata=enhanced_metadata
            )
            
            if not upload_result.get('success'):
                return {
                    'success': False,
                    'error': upload_result.get('error', 'Upload failed')
                }
            
            document_id = upload_result.get('document_id')
            
            return {
                'success': True,
                'document_id': document_id,
                'chunk_template': chunk_template,
                'metadata': enhanced_metadata,
                'message': f'Document uploaded successfully with template: {chunk_template}'
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def wait_for_processing(self, 
                          document_id: str,
                          timeout: int = 300,
                          poll_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for document processing to complete.
        
        Args:
            document_id: Document ID to monitor
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds
            
        Returns:
            Processing completion result
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                status_result = self.client.get_document_status(document_id)
                
                if not status_result.get('success'):
                    return {
                        'success': False,
                        'error': 'Failed to get document status'
                    }
                
                status = status_result.get('status', '').lower()
                progress = status_result.get('progress', 0)
                
                logger.debug(f"Document {document_id} status: {status} ({progress}%)")
                
                if status == ProcessingStatus.SUCCESS.value:
                    return {
                        'success': True,
                        'status': status,
                        'progress': progress,
                        'chunks_count': status_result.get('chunks_count', 0),
                        'processing_time': status_result.get('processing_time', 0),
                        'metadata': status_result.get('metadata', {})
                    }
                elif status == ProcessingStatus.FAILED.value:
                    return {
                        'success': False,
                        'status': status,
                        'error': status_result.get('error', 'Processing failed')
                    }
                elif status == ProcessingStatus.CANCELLED.value:
                    return {
                        'success': False,
                        'status': status,
                        'error': 'Processing was cancelled'
                    }
                
                # Still processing, wait and retry
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error checking document status: {e}")
                time.sleep(poll_interval)
        
        # Timeout reached
        return {
            'success': False,
            'error': f'Processing timeout after {timeout} seconds'
        }
    
    def get_processing_statistics(self, document_id: str) -> Dict[str, Any]:
        """
        Get detailed processing statistics for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Processing statistics and metadata
        """
        try:
            # Get document status
            status_result = self.client.get_document_status(document_id)
            
            if not status_result.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get document status'
                }
            
            # Get document chunks
            chunks_result = self.client.get_document_chunks(document_id)
            
            if not chunks_result.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get document chunks'
                }
            
            chunks = chunks_result.get('chunks', [])
            
            # Calculate statistics
            total_chunks = len(chunks)
            total_characters = sum(len(chunk.get('content', '')) for chunk in chunks)
            avg_chunk_size = total_characters / total_chunks if total_chunks > 0 else 0
            
            # Analyze chunk types
            chunk_types = {}
            for chunk in chunks:
                chunk_type = chunk.get('type', 'text')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            return {
                'success': True,
                'document_id': document_id,
                'status': status_result.get('status'),
                'total_chunks': total_chunks,
                'total_characters': total_characters,
                'average_chunk_size': avg_chunk_size,
                'chunk_types': chunk_types,
                'processing_time': status_result.get('processing_time', 0),
                'metadata': status_result.get('metadata', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {
                'success': False,
                'error': str(e)
            }
