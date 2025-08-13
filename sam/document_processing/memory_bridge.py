#!/usr/bin/env python3
"""
Document Memory Bridge
=====================

Bridges the gap between proven PDF processor and SAM memory stores.
"""

import logging
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentMemoryBridge:
    """Bridge between proven PDF processor and SAM memory stores."""
    
    def __init__(self):
        """Initialize the bridge."""
        self.memory_store = None
        self._initialize_memory_store()
    
    def _initialize_memory_store(self):
        """Initialize the SAM memory store."""
        try:
            # Try to get secure memory store first
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'secure_memory_store'):
                    self.memory_store = st.session_state.secure_memory_store
                    logger.info("✅ Using existing secure memory store")
                    return
            except:
                pass
            
            # Fallback to creating new memory store
            try:
                from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType
                self.memory_store = get_secure_memory_store(
                    store_type=VectorStoreType.CHROMA,
                    storage_directory="memory_store",
                    embedding_dimension=384,
                    enable_encryption=False  # Disable encryption for compatibility
                )
                logger.info("✅ Created new secure memory store")
            except Exception as e:
                logger.warning(f"Secure memory store failed: {e}")
                
                # Final fallback to regular memory store
                from memory.memory_vectorstore import get_memory_store, VectorStoreType
                self.memory_store = get_memory_store(
                    store_type=VectorStoreType.CHROMA,
                    storage_directory="memory_store",
                    embedding_dimension=384
                )
                logger.info("✅ Using regular memory store as fallback")
                
        except Exception as e:
            logger.error(f"Failed to initialize memory store: {e}")
            self.memory_store = None
    
    def store_document_in_memory(self, pdf_name: str, chunks: List[str], metadata: Dict[str, Any] = None) -> bool:
        """Store document chunks in SAM memory store."""
        if not self.memory_store:
            logger.error("Memory store not available")
            return False
        
        try:
            stored_count = 0
            base_metadata = metadata or {}
            
            for i, chunk_content in enumerate(chunks):
                if not chunk_content.strip():
                    continue
                
                # Ensure a document_id is present for downstream citations
                derived_doc_id = base_metadata.get('document_id') or Path(pdf_name).stem

                chunk_metadata = {
                    **base_metadata,
                    'document_name': pdf_name,
                    'document_id': derived_doc_id,
                    'chunk_index': i,
                    'content_type': 'pdf_document',
                    'source_type': 'uploaded_document',
                    'upload_method': 'proven_pdf_processor_bridge'
                }
                
                # Store in memory with proper format
                if hasattr(self.memory_store, 'add_memory'):
                    # Use add_memory for SecureMemoryVectorStore or MemoryVectorStore
                    from memory.memory_vectorstore import MemoryType
                    self.memory_store.add_memory(
                        content=chunk_content,
                        memory_type=MemoryType.DOCUMENT,
                        source=f"Document: {pdf_name} (Block {i+1})",
                        tags=["uploaded_document", "pdf"],
                        importance_score=0.8,
                        metadata=chunk_metadata
                    )
                else:
                    # Fallback for other memory store types
                    logger.warning("Memory store doesn't have add_memory method")
                    continue
                
                stored_count += 1
            
            logger.info(f"✅ Stored {stored_count} chunks for {pdf_name} in SAM memory")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document in memory: {e}")
            return False
    
    def enhanced_pdf_upload_handler(self, pdf_path: str, filename: str = None, session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Enhanced PDF upload handler that stores in both systems."""
        try:
            # First, use the proven PDF processor
            from sam.document_processing.proven_pdf_integration import handle_pdf_upload_for_sam
            
            success, message, metadata = handle_pdf_upload_for_sam(pdf_path, filename, session_id)
            
            if success:
                # Also store in SAM memory system
                pdf_name = metadata.get('pdf_name', Path(filename or pdf_path).stem)
                
                # Get chunks from proven processor
                from sam.document_processing.proven_pdf_integration import get_sam_pdf_integration
                integration = get_sam_pdf_integration()
                processor = integration.processor
                
                if pdf_name in processor.vector_stores:
                    vector_store = processor.vector_stores[pdf_name]
                    chunks = []
                    
                    # Extract chunks
                    if isinstance(vector_store, dict) and 'chunks' in vector_store:
                        chunks = vector_store['chunks']
                    else:
                        # Try to extract from FAISS
                        try:
                            if hasattr(vector_store, 'docstore') and hasattr(vector_store.docstore, '_dict'):
                                for doc_id, doc in vector_store.docstore._dict.items():
                                    if hasattr(doc, 'page_content'):
                                        chunks.append(doc.page_content)
                                    elif hasattr(doc, 'content'):
                                        chunks.append(doc.content)
                        except:
                            pass
                    
                    if chunks:
                        # Store in SAM memory
                        bridge_success = self.store_document_in_memory(pdf_name, chunks, {
                            'original_filename': filename,
                            'session_id': session_id,
                            'processing_method': 'proven_pdf_with_bridge',
                            'document_id': Path(filename or pdf_path).stem
                        })
                        
                        if bridge_success:
                            metadata['stored_in_sam_memory'] = True
                            metadata['sam_memory_chunks'] = len(chunks)
                            message += f" and stored {len(chunks)} chunks in SAM memory"
                        else:
                            metadata['stored_in_sam_memory'] = False
                            logger.warning("Failed to store in SAM memory, but proven processor succeeded")
                    else:
                        logger.warning("No chunks found to store in SAM memory")
                        metadata['stored_in_sam_memory'] = False
                else:
                    logger.warning("PDF not found in proven processor vector stores")
                    metadata['stored_in_sam_memory'] = False
            
            return success, message, metadata
            
        except Exception as e:
            logger.error(f"Enhanced PDF upload failed: {e}")
            return False, f"Upload failed: {str(e)}", {}

# Global bridge instance
_document_bridge = None

def get_document_bridge() -> DocumentMemoryBridge:
    """Get the global document bridge instance."""
    global _document_bridge
    if _document_bridge is None:
        _document_bridge = DocumentMemoryBridge()
    return _document_bridge

def enhanced_handle_pdf_upload_for_sam(pdf_path: str, filename: str = None, session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Enhanced PDF upload handler that uses the bridge."""
    bridge = get_document_bridge()
    return bridge.enhanced_pdf_upload_handler(pdf_path, filename, session_id)
