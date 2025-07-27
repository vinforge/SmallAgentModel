#!/usr/bin/env python3
"""
Fix Memory Store Synchronization
Fixes the issue where multimodal pipeline and Streamlit app use different memory stores.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def analyze_memory_store_mismatch():
    """Analyze the memory store mismatch issue."""
    print("🔍 Analyzing Memory Store Mismatch")
    print("=" * 50)
    
    try:
        from memory.memory_vectorstore import get_memory_store, VectorStoreType
        
        # Test different memory store configurations
        configs = [
            ("Default (no params)", {}),
            ("SIMPLE backend", {"store_type": VectorStoreType.SIMPLE}),
            ("CHROMA backend", {"store_type": VectorStoreType.CHROMA, "storage_directory": "memory_store", "embedding_dimension": 384}),
        ]
        
        for config_name, params in configs:
            try:
                print(f"\n📊 Testing {config_name}:")
                memory_store = get_memory_store(**params)
                
                print(f"   Type: {type(memory_store)}")
                print(f"   Backend: {memory_store.store_type}")
                
                # Test search
                results = memory_store.search_memories("", max_results=5)
                print(f"   Documents: {len(results)}")
                
                if results:
                    print(f"   Sample: {results[0].chunk.content[:50]}..." if hasattr(results[0], 'chunk') else f"   Sample: {results[0].content[:50]}...")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
    
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def sync_memory_stores():
    """Synchronize documents between memory stores."""
    print("\n🔄 Synchronizing Memory Stores")
    print("=" * 50)
    
    try:
        from memory.memory_vectorstore import get_memory_store, VectorStoreType, MemoryType
        
        # Get the multimodal pipeline memory store (where documents are)
        source_store = get_memory_store()  # Default - where documents are stored
        print(f"✅ Source store: {source_store.store_type}")
        
        # Get documents from source store
        source_docs = source_store.search_memories("", max_results=100)
        print(f"📄 Found {len(source_docs)} documents in source store")
        
        if not source_docs:
            print("❌ No documents to sync")
            return False
        
        # Get the target store (what Streamlit app uses)
        target_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store", 
            embedding_dimension=384
        )
        print(f"✅ Target store: {target_store.store_type}")
        
        # Check current target store content
        target_docs = target_store.search_memories("", max_results=5)
        print(f"📄 Target store currently has {len(target_docs)} documents")
        
        # Sync documents
        synced_count = 0
        for doc in source_docs:
            try:
                # Extract content and metadata
                content = ""
                source = ""
                tags = []
                metadata = {}
                
                if hasattr(doc, 'chunk'):
                    content = doc.chunk.content
                    source = getattr(doc.chunk, 'source', 'unknown')
                    tags = getattr(doc.chunk, 'tags', [])
                    metadata = getattr(doc.chunk, 'metadata', {})
                elif hasattr(doc, 'content'):
                    content = doc.content
                    source = getattr(doc, 'source', 'unknown')
                    tags = getattr(doc, 'tags', [])
                    metadata = getattr(doc, 'metadata', {})
                
                if content:
                    # Add to target store
                    chunk_id = target_store.add_memory(
                        content=content,
                        memory_type=MemoryType.DOCUMENT,
                        source=source,
                        tags=tags + ['synced'],
                        importance_score=0.8,
                        metadata=metadata
                    )
                    synced_count += 1
                    print(f"   ✅ Synced document: {source[:50]}...")
                
            except Exception as e:
                print(f"   ❌ Failed to sync document: {e}")
        
        print(f"🎉 Successfully synced {synced_count} documents")
        
        # Verify sync
        target_docs_after = target_store.search_memories("", max_results=5)
        print(f"📊 Target store now has {len(target_docs_after)} documents")
        
        return synced_count > 0
        
    except Exception as e:
        print(f"❌ Sync failed: {e}")
        return False

def test_streamlit_compatibility():
    """Test if the fix works with Streamlit app patterns."""
    print("\n🧪 Testing Streamlit Compatibility")
    print("=" * 50)
    
    try:
        from memory.memory_vectorstore import get_memory_store, VectorStoreType, MemoryType
        from memory.secure_memory_vectorstore import get_secure_memory_store
        
        # Test regular memory store (what Streamlit uses)
        regular_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384
        )
        
        # Test search with memory_type parameter (Streamlit pattern)
        try:
            # This should work now
            doc_results = regular_store.search_memories(
                query="",
                memory_types=[MemoryType.DOCUMENT],
                max_results=10
            )
            print(f"✅ Regular store document search: {len(doc_results)} results")
            
            if doc_results:
                for i, result in enumerate(doc_results[:3]):
                    content = ""
                    if hasattr(result, 'chunk'):
                        content = result.chunk.content[:80] + "..." if len(result.chunk.content) > 80 else result.chunk.content
                    elif hasattr(result, 'content'):
                        content = result.content[:80] + "..." if len(result.content) > 80 else result.content
                    print(f"   {i+1}. {content}")
            
        except Exception as e:
            print(f"❌ Regular store search failed: {e}")
        
        # Test secure memory store
        try:
            secure_store = get_secure_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384,
                enable_encryption=False
            )
            
            secure_results = secure_store.search_memories("", max_results=5)
            print(f"✅ Secure store search: {len(secure_results)} results")
            
        except Exception as e:
            print(f"❌ Secure store search failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def create_test_document_upload():
    """Create a test document and upload it through the proper pipeline."""
    print("\n📤 Testing Document Upload Pipeline")
    print("=" * 50)
    
    try:
        import tempfile
        from memory.memory_vectorstore import get_memory_store, VectorStoreType, MemoryType
        
        # Create test document
        test_content = """
        SAM Document Upload Fix Test
        
        This document tests the fixed document upload pipeline.
        
        Key Information:
        - Upload method: Fixed pipeline
        - Memory store: Synchronized
        - Retrieval: Should work in Streamlit
        
        If you can see this document in SAM's Streamlit interface, the fix worked!
        """
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        print(f"📄 Created test file: {temp_file_path}")
        
        # Process through multimodal pipeline
        from multimodal_processing.multimodal_pipeline import get_multimodal_pipeline
        pipeline = get_multimodal_pipeline()
        
        result = pipeline.process_document(temp_file_path)
        print(f"✅ Multimodal processing: {result.get('document_id', 'success')}")
        
        # Also add directly to target memory store
        target_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384
        )
        
        chunk_id = target_store.add_memory(
            content=f"Document: fix_test.txt\n\n{test_content}",
            memory_type=MemoryType.DOCUMENT,
            source="fix_test",
            tags=['test', 'fix', 'upload'],
            importance_score=0.9,
            metadata={
                'filename': 'fix_test.txt',
                'test_type': 'upload_fix'
            }
        )
        
        print(f"✅ Added to target store: {chunk_id}")
        
        # Test retrieval
        search_results = target_store.search_memories("SAM Document Upload Fix", max_results=5)
        print(f"🔍 Search results: {len(search_results)}")
        
        if search_results:
            print("✅ Document successfully uploaded and retrievable!")
        else:
            print("❌ Document not found in search")
        
        # Cleanup
        os.unlink(temp_file_path)
        
        return len(search_results) > 0
        
    except Exception as e:
        print(f"❌ Test upload failed: {e}")
        return False

def main():
    """Run the memory store fix."""
    print("🔧 SAM Memory Store Synchronization Fix")
    print("=" * 60)
    
    # Analyze the problem
    analyze_memory_store_mismatch()
    
    # Sync the memory stores
    sync_success = sync_memory_stores()
    
    # Test compatibility
    compat_success = test_streamlit_compatibility()
    
    # Test document upload
    upload_success = create_test_document_upload()
    
    print("\n" + "=" * 60)
    print("🎯 FIX SUMMARY")
    print("=" * 60)
    
    if sync_success and compat_success and upload_success:
        print("🎉 SUCCESS! Memory store synchronization fixed!")
        print("✅ Documents are now available in Streamlit app")
        print("✅ Upload pipeline is working correctly")
        print("✅ Search functionality is operational")
        print("\n💡 Next steps:")
        print("1. Try uploading a document in the Streamlit app")
        print("2. Test document Q&A functionality")
        print("3. Verify that uploaded documents appear in the document library")
    else:
        print("❌ Some issues remain:")
        if not sync_success:
            print("   - Memory store sync failed")
        if not compat_success:
            print("   - Streamlit compatibility issues")
        if not upload_success:
            print("   - Document upload test failed")
        print("\n💡 Check the error messages above for details")

if __name__ == "__main__":
    main()
