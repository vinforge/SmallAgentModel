#!/usr/bin/env python3
"""
Fix Document Storage Integration
===============================

This script fixes the disconnect between document upload and document search
by ensuring uploaded documents are properly stored in the SAM memory system.
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

def fix_document_storage():
    """Fix the document storage integration issue."""
    
    print("🔧 SAM Document Storage Fix")
    print("=" * 50)
    
    try:
        # Check if we have uploaded documents in the proven PDF processor
        from sam.document_processing.proven_pdf_integration import get_sam_pdf_integration
        
        integration = get_sam_pdf_integration()
        status = integration.get_integration_status()
        
        print("📋 Current Document Status:")
        print(f"   • Processor available: {'✅' if status['processor_available'] else '❌'}")
        print(f"   • Total processed PDFs: {status['total_processed_pdfs']}")
        print(f"   • Current session: {status['current_session']}")
        print(f"   • Session PDFs: {status['session_pdfs']}")
        
        if status['total_processed_pdfs'] == 0:
            print("\n❌ No documents found in proven PDF processor!")
            print("📋 To fix this:")
            print("   1. Upload a document through the SAM chat interface")
            print("   2. Run this fix script again")
            return False
        
        # Get the secure memory store
        print("\n🔍 Checking SAM memory stores...")
        
        try:
            import streamlit as st
            # Initialize session state if not available
            if not hasattr(st, 'session_state'):
                st.session_state = type('SessionState', (), {})()
            
            from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType
            
            # Get or create secure memory store
            if not hasattr(st.session_state, 'secure_memory_store'):
                st.session_state.secure_memory_store = get_secure_memory_store(
                    store_type=VectorStoreType.CHROMA,
                    storage_directory="memory_store",
                    embedding_dimension=384,
                    enable_encryption=True
                )
            
            memory_store = st.session_state.secure_memory_store
            print("   • Secure memory store: ✅ Available")
            
        except Exception as e:
            print(f"   • Secure memory store: ❌ Error: {e}")
            
            # Fallback to regular memory store
            try:
                from memory.memory_vectorstore import get_memory_store, VectorStoreType
                memory_store = get_memory_store(
                    store_type=VectorStoreType.CHROMA,
                    storage_directory="memory_store",
                    embedding_dimension=384
                )
                print("   • Regular memory store: ✅ Available (fallback)")
            except Exception as e2:
                print(f"   • Regular memory store: ❌ Error: {e2}")
                return False
        
        # Migrate documents from proven PDF processor to SAM memory store
        print("\n🔄 Migrating documents to SAM memory store...")
        
        processor = integration.processor
        migrated_count = 0
        
        for pdf_name in processor.list_processed_pdfs():
            try:
                print(f"   📄 Migrating: {pdf_name}")
                
                # Get PDF info and vector store
                pdf_info = processor.get_pdf_info(pdf_name)
                vector_store = processor.vector_stores.get(pdf_name)
                
                if not vector_store:
                    print(f"      ❌ No vector store found for {pdf_name}")
                    continue
                
                # Extract chunks from the vector store
                chunks = []
                if isinstance(vector_store, dict) and 'chunks' in vector_store:
                    # Fallback format (no FAISS)
                    chunks = vector_store['chunks']
                else:
                    # FAISS format - need to extract texts
                    try:
                        if hasattr(vector_store, 'docstore') and hasattr(vector_store.docstore, '_dict'):
                            # Extract texts from FAISS docstore
                            for doc_id, doc in vector_store.docstore._dict.items():
                                if hasattr(doc, 'page_content'):
                                    chunks.append(doc.page_content)
                                elif hasattr(doc, 'content'):
                                    chunks.append(doc.content)
                    except Exception as extract_error:
                        print(f"      ⚠️ Could not extract chunks from FAISS: {extract_error}")
                        continue
                
                if not chunks:
                    print(f"      ❌ No chunks found for {pdf_name}")
                    continue
                
                print(f"      📊 Found {len(chunks)} chunks")
                
                # Store chunks in SAM memory store
                stored_count = 0
                for i, chunk_content in enumerate(chunks):
                    try:
                        # Create memory entry
                        memory_entry = {
                            'content': chunk_content,
                            'source': f"Document: {pdf_name} (Block {i+1})",
                            'metadata': {
                                'document_name': pdf_name,
                                'chunk_index': i,
                                'content_type': 'pdf_document',
                                'upload_method': 'proven_pdf_processor',
                                'migrated_from': 'proven_pdf_processor'
                            }
                        }
                        
                        # Store in memory
                        memory_store.store_memory(
                            content=chunk_content,
                            memory_type="document",
                            metadata=memory_entry['metadata']
                        )
                        
                        stored_count += 1
                        
                    except Exception as store_error:
                        print(f"      ⚠️ Failed to store chunk {i}: {store_error}")
                
                print(f"      ✅ Stored {stored_count}/{len(chunks)} chunks")
                migrated_count += 1
                
            except Exception as migrate_error:
                print(f"      ❌ Failed to migrate {pdf_name}: {migrate_error}")
        
        if migrated_count > 0:
            print(f"\n🎉 Successfully migrated {migrated_count} documents to SAM memory store!")
            print("\n📋 Next Steps:")
            print("   1. Restart SAM: python start_sam.py")
            print("   2. Try asking questions about your documents")
            print("   3. Use the Summarize and Deep Analysis buttons")
            
            return True
        else:
            print("\n❌ No documents were migrated!")
            print("📋 This could be due to:")
            print("   1. Documents are in an unsupported format")
            print("   2. Vector stores are corrupted")
            print("   3. Memory store is not accessible")
            
            return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = fix_document_storage()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Script error: {e}")
        exit(1)
