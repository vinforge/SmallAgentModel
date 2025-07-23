#!/usr/bin/env python3
"""
V2 Ingestion Pipeline for SAM MUVERA Retrieval System
Orchestrates the complete v2 document ingestion process.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Global pipeline instance
_v2_ingestion_pipeline = None

@dataclass
class V2IngestionConfig:
    """Configuration for v2 ingestion pipeline."""
    embedder_model: str = "colbert-ir/colbertv2.0"
    fde_dim: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_workers: int = 4
    batch_size: int = 10
    storage_root: str = "uploads"
    chroma_db_path: str = "chroma_db_v2"
    enable_deduplication: bool = True
    enable_progress_tracking: bool = True
    fallback_to_v1: bool = True

@dataclass
class V2IngestionResult:
    """Result from v2 ingestion pipeline."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    processing_time: float
    document_results: List[Any]  # List of V2ProcessingResult
    errors: List[str]
    metadata: Dict[str, Any]

class V2IngestionPipeline:
    """
    V2 Ingestion pipeline for SAM's MUVERA retrieval system.
    
    Orchestrates:
    - Batch document processing
    - Multi-threaded ingestion
    - Progress tracking
    - Error handling and fallbacks
    - Deduplication
    """
    
    def __init__(self, config: Optional[V2IngestionConfig] = None):
        """
        Initialize the v2 ingestion pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or V2IngestionConfig()
        
        # Components
        self.document_processor = None
        self.storage_manager = None
        self.is_initialized = False
        
        # Progress tracking
        self.processed_count = 0
        self.total_count = 0
        self.current_batch = 0
        
        logger.info(f"🔄 V2IngestionPipeline initialized")
        logger.info(f"📊 Config: {self.config.max_workers} workers, batch size {self.config.batch_size}")
    
    def _initialize_components(self) -> bool:
        """Initialize pipeline components."""
        if self.is_initialized:
            return True
        
        try:
            logger.info("🔄 Initializing v2 ingestion components...")
            
            # Initialize document processor
            from sam.ingestion.v2_document_processor import get_v2_document_processor
            self.document_processor = get_v2_document_processor(
                embedder_model=self.config.embedder_model,
                fde_dim=self.config.fde_dim,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            # Initialize storage manager
            from sam.storage import get_v2_storage_manager
            self.storage_manager = get_v2_storage_manager(
                storage_root=self.config.storage_root,
                chroma_db_path=self.config.chroma_db_path
            )
            
            self.is_initialized = True
            logger.info("✅ V2 ingestion components initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize v2 ingestion components: {e}")
            return False
    
    def _check_duplicate(self, file_path: str) -> Optional[str]:
        """Check if document is already processed."""
        if not self.config.enable_deduplication:
            return None
        
        try:
            # Simple file-based deduplication
            file_name = Path(file_path).name
            file_size = os.path.getsize(file_path)
            
            # Check existing documents
            existing_docs = self.storage_manager.list_documents()
            
            for doc_id in existing_docs:
                record = self.storage_manager.retrieve_document(doc_id)
                if record and record.filename == file_name and record.file_size == file_size:
                    logger.info(f"📋 Duplicate detected: {file_name} (existing: {doc_id})")
                    return doc_id
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️  Duplicate check failed: {e}")
            return None
    
    def _process_single_document(self, 
                                file_path: str,
                                document_id: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Process a single document."""
        try:
            # Check for duplicates
            if self.config.enable_deduplication:
                existing_id = self._check_duplicate(file_path)
                if existing_id:
                    logger.info(f"⏭️  Skipping duplicate: {file_path}")
                    # Return a mock successful result for duplicate
                    from sam.ingestion.v2_document_processor import V2ProcessingResult
                    return V2ProcessingResult(
                        document_id=existing_id,
                        filename=Path(file_path).name,
                        file_path=file_path,
                        text_content="",
                        token_embeddings=None,
                        fde_vector=None,
                        chunks=[],
                        processing_time=0.0,
                        success=True,
                        error_message=None,
                        metadata={'duplicate': True, 'original_id': existing_id}
                    )
            
            # Process document
            result = self.document_processor.process_document(
                file_path=file_path,
                document_id=document_id,
                metadata=metadata
            )
            
            # Update progress
            self.processed_count += 1
            
            if self.config.enable_progress_tracking:
                progress = (self.processed_count / self.total_count) * 100
                logger.info(f"📊 Progress: {self.processed_count}/{self.total_count} ({progress:.1f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process document {file_path}: {e}")
            # Return failed result
            from sam.ingestion.v2_document_processor import V2ProcessingResult
            return V2ProcessingResult(
                document_id=document_id or "",
                filename=Path(file_path).name,
                file_path=file_path,
                text_content="",
                token_embeddings=None,
                fde_vector=None,
                chunks=[],
                processing_time=0.0,
                success=False,
                error_message=str(e),
                metadata=metadata or {}
            )
    
    def ingest_documents(self, 
                        file_paths: List[str],
                        document_ids: Optional[List[str]] = None,
                        metadata_list: Optional[List[Dict[str, Any]]] = None) -> V2IngestionResult:
        """
        Ingest multiple documents using the v2 pipeline.
        
        Args:
            file_paths: List of file paths to process
            document_ids: Optional list of document IDs
            metadata_list: Optional list of metadata dicts
            
        Returns:
            V2IngestionResult with ingestion results
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting v2 ingestion: {len(file_paths)} documents")
            
            # Initialize components
            if not self._initialize_components():
                return V2IngestionResult(
                    total_documents=len(file_paths),
                    successful_documents=0,
                    failed_documents=len(file_paths),
                    processing_time=0.0,
                    document_results=[],
                    errors=["Failed to initialize v2 ingestion components"],
                    metadata={}
                )
            
            # Setup progress tracking
            self.total_count = len(file_paths)
            self.processed_count = 0
            self.current_batch = 0
            
            # Prepare document data
            document_data = []
            for i, file_path in enumerate(file_paths):
                doc_id = document_ids[i] if document_ids and i < len(document_ids) else None
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
                document_data.append((file_path, doc_id, metadata))
            
            # Process documents in batches with threading
            all_results = []
            errors = []
            
            # Split into batches
            batches = [document_data[i:i + self.config.batch_size] 
                      for i in range(0, len(document_data), self.config.batch_size)]
            
            logger.info(f"📦 Processing {len(batches)} batches with {self.config.max_workers} workers")
            
            for batch_idx, batch in enumerate(batches):
                self.current_batch = batch_idx + 1
                logger.info(f"🔄 Processing batch {self.current_batch}/{len(batches)}")
                
                # Process batch with thread pool
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    # Submit all documents in batch
                    future_to_doc = {
                        executor.submit(self._process_single_document, file_path, doc_id, metadata): 
                        (file_path, doc_id, metadata)
                        for file_path, doc_id, metadata in batch
                    }
                    
                    # Collect results
                    for future in as_completed(future_to_doc):
                        file_path, doc_id, metadata = future_to_doc[future]
                        try:
                            result = future.result()
                            all_results.append(result)
                            
                            if not result.success:
                                errors.append(f"{file_path}: {result.error_message}")
                                
                        except Exception as e:
                            error_msg = f"Exception processing {file_path}: {str(e)}"
                            errors.append(error_msg)
                            logger.error(f"❌ {error_msg}")
            
            # Calculate statistics
            successful_docs = sum(1 for result in all_results if result.success)
            failed_docs = len(all_results) - successful_docs
            processing_time = time.time() - start_time
            
            # Create result
            ingestion_result = V2IngestionResult(
                total_documents=len(file_paths),
                successful_documents=successful_docs,
                failed_documents=failed_docs,
                processing_time=processing_time,
                document_results=all_results,
                errors=errors,
                metadata={
                    'config': self.config.__dict__,
                    'batches_processed': len(batches),
                    'avg_processing_time': processing_time / len(file_paths) if file_paths else 0,
                    'ingestion_timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ V2 ingestion completed: {successful_docs}/{len(file_paths)} successful")
            logger.info(f"⏱️  Total time: {processing_time:.2f}s, avg: {processing_time/len(file_paths):.2f}s/doc")
            
            if errors:
                logger.warning(f"⚠️  {len(errors)} errors occurred during ingestion")
            
            return ingestion_result
            
        except Exception as e:
            logger.error(f"❌ V2 ingestion failed: {e}")
            return V2IngestionResult(
                total_documents=len(file_paths),
                successful_documents=0,
                failed_documents=len(file_paths),
                processing_time=time.time() - start_time,
                document_results=[],
                errors=[str(e)],
                metadata={}
            )
    
    def ingest_directory(self, 
                        directory_path: str,
                        file_patterns: List[str] = None,
                        recursive: bool = True) -> V2IngestionResult:
        """
        Ingest all documents from a directory.
        
        Args:
            directory_path: Path to directory
            file_patterns: File patterns to match (e.g., ['*.pdf', '*.txt'])
            recursive: Whether to search recursively
            
        Returns:
            V2IngestionResult with ingestion results
        """
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists():
                logger.error(f"❌ Directory not found: {directory_path}")
                return V2IngestionResult(
                    total_documents=0,
                    successful_documents=0,
                    failed_documents=0,
                    processing_time=0.0,
                    document_results=[],
                    errors=[f"Directory not found: {directory_path}"],
                    metadata={}
                )
            
            # Find files
            file_paths = []
            patterns = file_patterns or ['*.pdf', '*.txt', '*.md', '*.docx']
            
            for pattern in patterns:
                if recursive:
                    files = directory_path.rglob(pattern)
                else:
                    files = directory_path.glob(pattern)
                
                file_paths.extend([str(f) for f in files if f.is_file()])
            
            logger.info(f"📁 Found {len(file_paths)} files in {directory_path}")
            
            if not file_paths:
                logger.warning(f"⚠️  No files found matching patterns: {patterns}")
                return V2IngestionResult(
                    total_documents=0,
                    successful_documents=0,
                    failed_documents=0,
                    processing_time=0.0,
                    document_results=[],
                    errors=[],
                    metadata={'directory': str(directory_path), 'patterns': patterns}
                )
            
            # Ingest files
            return self.ingest_documents(file_paths)
            
        except Exception as e:
            logger.error(f"❌ Directory ingestion failed: {e}")
            return V2IngestionResult(
                total_documents=0,
                successful_documents=0,
                failed_documents=0,
                processing_time=0.0,
                document_results=[],
                errors=[str(e)],
                metadata={}
            )

def get_v2_ingestion_pipeline(config: Optional[V2IngestionConfig] = None) -> V2IngestionPipeline:
    """
    Get or create a v2 ingestion pipeline instance.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        V2IngestionPipeline instance
    """
    global _v2_ingestion_pipeline
    
    if _v2_ingestion_pipeline is None:
        _v2_ingestion_pipeline = V2IngestionPipeline(config)
    
    return _v2_ingestion_pipeline

def ingest_document_v2(file_path: str,
                      document_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> V2IngestionResult:
    """
    Convenience function to ingest a single document using v2 pipeline.
    
    Args:
        file_path: Path to the document file
        document_id: Optional document ID
        metadata: Optional metadata
        
    Returns:
        V2IngestionResult with ingestion results
    """
    pipeline = get_v2_ingestion_pipeline()
    return pipeline.ingest_documents(
        file_paths=[file_path],
        document_ids=[document_id] if document_id else None,
        metadata_list=[metadata] if metadata else None
    )
