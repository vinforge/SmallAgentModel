#!/usr/bin/env python3
"""
V2 Document Processor for SAM MUVERA Retrieval Pipeline
Processes documents using multi-vector embeddings and FDE transformation.
"""

import os
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Global processor instance
_v2_document_processor = None

@dataclass
class V2ProcessingResult:
    """Result from v2 document processing."""
    document_id: str              # Unique document identifier
    filename: str                 # Original filename
    file_path: str               # Path to original file
    text_content: str            # Extracted text content
    token_embeddings: Any        # Token embeddings from ColBERT
    fde_vector: Any              # Fixed dimensional encoding
    chunks: List[str]            # Text chunks for processing
    processing_time: float       # Total processing time
    success: bool                # Whether processing succeeded
    error_message: Optional[str] # Error message if failed
    metadata: Dict[str, Any]     # Additional metadata

class V2DocumentProcessor:
    """
    V2 Document processor using multi-vector embeddings and FDE transformation.
    
    Integrates:
    - Text extraction (PDF, DOCX, etc.)
    - Multi-vector embeddings (ColBERTv2)
    - FDE transformation (MUVERA)
    - Storage in v2 schema
    """
    
    def __init__(self,
                 embedder_model: str = "colbert-ir/colbertv2.0",
                 fde_dim: int = 768,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        Initialize the v2 document processor.
        
        Args:
            embedder_model: ColBERT model to use
            fde_dim: FDE output dimension
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedder_model = embedder_model
        self.fde_dim = fde_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Components (loaded lazily)
        self.embedder = None
        self.fde_transformer = None
        self.storage_manager = None
        self.is_initialized = False
        
        logger.info(f"🔄 V2DocumentProcessor initialized")
        logger.info(f"📊 Model: {embedder_model}, FDE dim: {fde_dim}")
    
    def _initialize_components(self) -> bool:
        """Initialize all processing components."""
        if self.is_initialized:
            return True
        
        try:
            logger.info("🔄 Initializing v2 processing components...")
            
            # Initialize multi-vector embedder
            from sam.embedding import get_multivector_embedder
            self.embedder = get_multivector_embedder(
                model_name=self.embedder_model,
                max_length=self.chunk_size
            )
            
            # Initialize FDE transformer
            from sam.cognition import get_muvera_fde
            self.fde_transformer = get_muvera_fde(fde_dim=self.fde_dim)
            
            # Initialize storage manager
            from sam.storage import get_v2_storage_manager
            self.storage_manager = get_v2_storage_manager()
            
            self.is_initialized = True
            logger.info("✅ V2 processing components initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize v2 components: {e}")
            return False
    
    def _extract_text(self, file_path: str) -> Tuple[bool, str, str]:
        """
        Extract text from various file formats.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (success, text_content, error_message)
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False, "", f"File not found: {file_path}"
            
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_ext in ['.txt', '.md']:
                return self._extract_text_file(file_path)
            elif file_ext == '.docx':
                return self._extract_docx_text(file_path)
            else:
                # Try as text file
                return self._extract_text_file(file_path)
                
        except Exception as e:
            return False, "", f"Text extraction failed: {str(e)}"
    
    def _extract_pdf_text(self, file_path: Path) -> Tuple[bool, str, str]:
        """Extract text from PDF file."""
        try:
            # Use the existing PDF processor
            from sam.document_processing.proven_pdf_processor import ProvenPDFProcessor
            
            pdf_processor = ProvenPDFProcessor()
            
            # Extract text using existing processor
            success, message = pdf_processor.process_pdf(str(file_path), file_path.stem)
            
            if success:
                # Get the extracted text (this is a simplified approach)
                # In practice, we'd need to modify the PDF processor to return text
                with open(file_path, 'rb') as f:
                    # Use PyPDF2 directly for text extraction
                    try:
                        import PyPDF2
                        if hasattr(PyPDF2, 'PdfReader'):
                            reader = PyPDF2.PdfReader(f)
                            text = ""
                            for page in reader.pages:
                                text += page.extract_text() + "\n"
                        else:
                            reader = PyPDF2.PdfFileReader(f)
                            text = ""
                            for i in range(reader.numPages):
                                page = reader.getPage(i)
                                text += page.extractText() + "\n"
                        
                        return True, text, ""
                    except Exception as e:
                        return False, "", f"PyPDF2 extraction failed: {e}"
            else:
                return False, "", message
                
        except Exception as e:
            return False, "", f"PDF extraction failed: {str(e)}"
    
    def _extract_text_file(self, file_path: Path) -> Tuple[bool, str, str]:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return True, text, ""
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                return True, text, ""
            except Exception as e:
                return False, "", f"Text file reading failed: {str(e)}"
        except Exception as e:
            return False, "", f"Text file reading failed: {str(e)}"
    
    def _extract_docx_text(self, file_path: Path) -> Tuple[bool, str, str]:
        """Extract text from DOCX file."""
        try:
            import docx
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return True, text, ""
        except ImportError:
            return False, "", "python-docx not available. Install with: pip install python-docx"
        except Exception as e:
            return False, "", f"DOCX extraction failed: {str(e)}"
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create text chunks for processing."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start, end - 100)
                sentence_end = -1
                
                for i in range(end, search_start, -1):
                    if text[i] in '.!?':
                        sentence_end = i + 1
                        break
                
                if sentence_end > 0:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def process_document(self, 
                        file_path: str,
                        document_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> V2ProcessingResult:
        """
        Process a document using the v2 pipeline.
        
        Args:
            file_path: Path to the document file
            document_id: Optional document ID (auto-generated if None)
            metadata: Optional metadata
            
        Returns:
            V2ProcessingResult with processing results
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Processing document v2: {file_path}")
            
            # Initialize components
            if not self._initialize_components():
                return V2ProcessingResult(
                    document_id="",
                    filename="",
                    file_path=file_path,
                    text_content="",
                    token_embeddings=None,
                    fde_vector=None,
                    chunks=[],
                    processing_time=0.0,
                    success=False,
                    error_message="Failed to initialize v2 components",
                    metadata={}
                )
            
            # Generate document ID if not provided
            if not document_id:
                file_name = Path(file_path).name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                content_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
                document_id = f"v2_{timestamp}_{content_hash}"
            
            # Extract text
            success, text_content, error_msg = self._extract_text(file_path)
            if not success:
                return V2ProcessingResult(
                    document_id=document_id,
                    filename=Path(file_path).name,
                    file_path=file_path,
                    text_content="",
                    token_embeddings=None,
                    fde_vector=None,
                    chunks=[],
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=error_msg,
                    metadata=metadata or {}
                )
            
            logger.info(f"📄 Extracted {len(text_content)} characters")
            
            # Create chunks
            chunks = self._create_chunks(text_content)
            logger.info(f"🧩 Created {len(chunks)} chunks")
            
            # Process the full document with multi-vector embeddings
            embedding_result = self.embedder.embed_document(text_content, doc_id=document_id)
            if not embedding_result:
                return V2ProcessingResult(
                    document_id=document_id,
                    filename=Path(file_path).name,
                    file_path=file_path,
                    text_content=text_content,
                    token_embeddings=None,
                    fde_vector=None,
                    chunks=chunks,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Failed to generate multi-vector embeddings",
                    metadata=metadata or {}
                )
            
            logger.info(f"🧠 Generated embeddings: {embedding_result.num_tokens} tokens")
            
            # Generate FDE vector
            fde_result = self.fde_transformer.generate_fde(
                embedding_result.token_embeddings, 
                doc_id=document_id
            )
            if not fde_result:
                return V2ProcessingResult(
                    document_id=document_id,
                    filename=Path(file_path).name,
                    file_path=file_path,
                    text_content=text_content,
                    token_embeddings=embedding_result,
                    fde_vector=None,
                    chunks=chunks,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Failed to generate FDE vector",
                    metadata=metadata or {}
                )
            
            logger.info(f"🔄 Generated FDE: {fde_result.fde_dim}D vector")
            
            # Store in v2 storage
            storage_success = self.storage_manager.store_document(
                document_id=document_id,
                filename=Path(file_path).name,
                file_path=file_path,
                text_content=text_content,
                token_embeddings=embedding_result.token_embeddings,
                fde_vector=fde_result.fde_vector,
                metadata={
                    **(metadata or {}),
                    'processing_version': 'v2',
                    'embedding_model': self.embedder_model,
                    'fde_dim': self.fde_dim,
                    'num_chunks': len(chunks),
                    'embedding_result': {
                        'num_tokens': embedding_result.num_tokens,
                        'embedding_dim': embedding_result.embedding_dim,
                        'processing_time': embedding_result.processing_time
                    },
                    'fde_result': {
                        'fde_dim': fde_result.fde_dim,
                        'compression_ratio': fde_result.compression_ratio,
                        'processing_time': fde_result.processing_time
                    }
                }
            )
            
            if not storage_success:
                logger.warning("⚠️  Storage failed, but processing succeeded")
            
            processing_time = time.time() - start_time
            
            result = V2ProcessingResult(
                document_id=document_id,
                filename=Path(file_path).name,
                file_path=file_path,
                text_content=text_content,
                token_embeddings=embedding_result,
                fde_vector=fde_result,
                chunks=chunks,
                processing_time=processing_time,
                success=True,
                error_message=None,
                metadata=metadata or {}
            )
            
            logger.info(f"✅ Document processed successfully: {document_id}")
            logger.info(f"⏱️  Total processing time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return V2ProcessingResult(
                document_id=document_id or "",
                filename=Path(file_path).name if file_path else "",
                file_path=file_path,
                text_content="",
                token_embeddings=None,
                fde_vector=None,
                chunks=[],
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                metadata=metadata or {}
            )

def get_v2_document_processor(embedder_model: str = "colbert-ir/colbertv2.0",
                             fde_dim: int = 768,
                             chunk_size: int = 512,
                             chunk_overlap: int = 50) -> V2DocumentProcessor:
    """
    Get or create a v2 document processor instance.
    
    Args:
        embedder_model: ColBERT model to use
        fde_dim: FDE output dimension
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        V2DocumentProcessor instance
    """
    global _v2_document_processor
    
    if _v2_document_processor is None:
        _v2_document_processor = V2DocumentProcessor(
            embedder_model=embedder_model,
            fde_dim=fde_dim,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    return _v2_document_processor

def process_document_v2(file_path: str,
                       document_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> V2ProcessingResult:
    """
    Convenience function to process a document using v2 pipeline.
    
    Args:
        file_path: Path to the document file
        document_id: Optional document ID
        metadata: Optional metadata
        
    Returns:
        V2ProcessingResult with processing results
    """
    processor = get_v2_document_processor()
    return processor.process_document(file_path, document_id, metadata)
