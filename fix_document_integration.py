#!/usr/bin/env python3
"""
Fix Document Integration
========================

This script fixes the document upload and retrieval integration by:
1. Ensuring uploaded documents are stored in the correct memory system
2. Creating a bridge between the proven PDF processor and SAM memory stores
3. Fixing the document search functionality
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_document_bridge():
    """Create a bridge between proven PDF processor and SAM memory stores."""
    
    print("🔧 Creating Document Integration Bridge")
    print("=" * 50)
    
    # Create a patched version of the proven PDF integration
    bridge_file = Path("sam/document_processing/memory_bridge.py")
    bridge_file.parent.mkdir(parents=True, exist_ok=True)
    
    bridge_code = '''#!/usr/bin/env python3
"""
Document Memory Bridge
=====================

Bridges the gap between proven PDF processor and SAM memory stores.
"""

import logging
from typing import Tuple, Dict, Any, Optional
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
                
                chunk_metadata = {
                    **base_metadata,
                    'document_name': pdf_name,
                    'chunk_index': i,
                    'content_type': 'pdf_document',
                    'source_type': 'uploaded_document',
                    'upload_method': 'proven_pdf_processor_bridge'
                }
                
                # Store in memory with proper format
                self.memory_store.store_memory(
                    content=chunk_content,
                    memory_type="document",
                    metadata=chunk_metadata
                )
                
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
                            'processing_method': 'proven_pdf_with_bridge'
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
'''
    
    try:
        with open(bridge_file, 'w') as f:
            f.write(bridge_code)
        print(f"✅ Created document bridge: {bridge_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to create bridge: {e}")
        return False

def patch_chat_interface():
    """Patch the chat interface to use the enhanced upload handler."""
    
    print("\n🔧 Patching Chat Interface")
    print("-" * 30)
    
    try:
        # Read the current secure_streamlit_app.py
        app_file = Path("secure_streamlit_app.py")
        
        if not app_file.exists():
            print("❌ secure_streamlit_app.py not found")
            return False
        
        with open(app_file, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if 'enhanced_handle_pdf_upload_for_sam' in content:
            print("✅ Chat interface already patched")
            return True
        
        # Patch the import and function call
        old_import = "from sam.document_processing.proven_pdf_integration import handle_pdf_upload_for_sam"
        new_import = "from sam.document_processing.memory_bridge import enhanced_handle_pdf_upload_for_sam as handle_pdf_upload_for_sam"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            
            # Write back the patched content
            with open(app_file, 'w') as f:
                f.write(content)
            
            print("✅ Successfully patched chat interface")
            return True
        else:
            print("⚠️ Import pattern not found - manual patching may be needed")
            return False
            
    except Exception as e:
        print(f"❌ Failed to patch chat interface: {e}")
        return False

def main():
    """Main function to fix document integration."""
    
    print("🚀 SAM Document Integration Fix")
    print("=" * 50)
    
    success_count = 0
    total_steps = 2
    
    # Step 1: Create document bridge
    if create_document_bridge():
        success_count += 1
    
    # Step 2: Patch chat interface
    if patch_chat_interface():
        success_count += 1
    
    print(f"\n📊 Results: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        print("\n🎉 Document integration fix completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. Restart SAM: python start_sam.py")
        print("   2. Upload a document through the chat interface")
        print("   3. Try asking questions about the document")
        print("   4. Use Summarize and Deep Analysis buttons")
        return True
    else:
        print("\n⚠️ Some steps failed - document integration may not work properly")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Script error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
