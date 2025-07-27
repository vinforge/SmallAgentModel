#!/usr/bin/env python3
"""
SAM Document Processing Diagnostic Script
Identifies why SAM cannot read uploaded documents.
"""

import os
import sys
import logging
from pathlib import Path
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_file_system():
    """Check file system and directory structure."""
    print("🔍 STEP 1: File System Check")
    print("=" * 50)
    
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check for key directories
    directories_to_check = [
        "memory_store",
        "chroma_db",
        "data/uploads", 
        "multimodal_processing",
        "memory",
        "logs"
    ]
    
    for directory in directories_to_check:
        exists = os.path.exists(directory)
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"📂 {directory}: {status}")
        
        if exists and os.path.isdir(directory):
            try:
                files = os.listdir(directory)
                file_count = len(files)
                print(f"   📄 Contains {file_count} files/folders")
                if file_count > 0 and file_count <= 5:
                    print(f"   📋 Contents: {files}")
            except PermissionError:
                print("   ⚠️  Permission denied")
    
    # Check for ChromaDB files specifically
    print("\n🔍 ChromaDB Files Search:")
    for root, dirs, files in os.walk("."):
        for file in files:
            if "chroma" in file.lower() or file.endswith(".sqlite3"):
                print(f"   📄 {os.path.join(root, file)}")
    
    print()

def check_multimodal_pipeline():
    """Check if multimodal pipeline can be imported and initialized."""
    print("🔍 STEP 2: Multimodal Pipeline Check")
    print("=" * 50)
    
    try:
        from multimodal_processing.multimodal_pipeline import get_multimodal_pipeline
        print("✅ Multimodal pipeline import successful")
        
        try:
            pipeline = get_multimodal_pipeline()
            print("✅ Multimodal pipeline initialization successful")
            print(f"   📊 Pipeline type: {type(pipeline)}")
            
            # Check if pipeline has required methods
            required_methods = ['process_document', 'process_documents_batch']
            for method in required_methods:
                if hasattr(pipeline, method):
                    print(f"   ✅ Method '{method}' available")
                else:
                    print(f"   ❌ Method '{method}' MISSING")
            
            return pipeline
            
        except Exception as e:
            print(f"❌ Multimodal pipeline initialization failed: {e}")
            return None
            
    except ImportError as e:
        print(f"❌ Multimodal pipeline import failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def check_memory_stores():
    """Check memory store initialization and status."""
    print("🔍 STEP 3: Memory Store Check")
    print("=" * 50)
    
    # Check regular memory store
    try:
        from memory.memory_vectorstore import get_memory_store, VectorStoreType
        print("✅ Regular memory store import successful")
        
        try:
            memory_store = get_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384
            )
            print("✅ Regular memory store initialization successful")
            
            # Check if store has documents
            try:
                # Try to search for any documents
                results = memory_store.search_memories(
                    query="",
                    memory_type="document",
                    max_results=10
                )
                print(f"   📄 Found {len(results)} documents in regular store")
                
                if results:
                    for i, result in enumerate(results[:3]):
                        filename = "Unknown"
                        if hasattr(result, 'metadata') and result.metadata:
                            filename = result.metadata.get('filename', 'Unknown')
                        elif hasattr(result, 'chunk') and hasattr(result.chunk, 'source'):
                            filename = result.chunk.source
                        print(f"   📋 Document {i+1}: {filename}")
                        
            except Exception as e:
                print(f"   ⚠️  Could not search documents: {e}")
                
        except Exception as e:
            print(f"❌ Regular memory store initialization failed: {e}")
            
    except ImportError as e:
        print(f"❌ Regular memory store import failed: {e}")
    
    print()
    
    # Check secure memory store
    try:
        from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType
        print("✅ Secure memory store import successful")
        
        try:
            secure_store = get_secure_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384,
                enable_encryption=False  # Disable encryption for testing
            )
            print("✅ Secure memory store initialization successful")
            
            # Check security status
            try:
                status = secure_store.get_security_status()
                print(f"   🔐 Security status: {status}")
            except Exception as e:
                print(f"   ⚠️  Could not get security status: {e}")
                
        except Exception as e:
            print(f"❌ Secure memory store initialization failed: {e}")
            
    except ImportError as e:
        print(f"❌ Secure memory store import failed: {e}")

def test_document_processing(pipeline):
    """Test document processing with a sample file."""
    print("🔍 STEP 4: Document Processing Test")
    print("=" * 50)
    
    if not pipeline:
        print("❌ Cannot test - multimodal pipeline not available")
        return
    
    # Create a test document
    test_content = """
    Test Document for SAM Diagnostic
    
    This is a test document to verify that SAM's document processing pipeline is working correctly.
    
    Key Information:
    - Document type: Test file
    - Purpose: Diagnostic testing
    - Content: Sample text for processing
    
    If you can read this in SAM's responses, the document processing is working!
    """
    
    try:
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        print(f"📄 Created test file: {temp_file_path}")
        
        # Test multimodal pipeline processing
        try:
            print("🔄 Testing multimodal pipeline processing...")
            result = pipeline.process_document(temp_file_path)
            
            if result:
                print("✅ Multimodal pipeline processing successful!")
                print(f"   📊 Result type: {type(result)}")
                print(f"   📋 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (str, int, float, bool)):
                            print(f"   📝 {key}: {value}")
                        else:
                            print(f"   📝 {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
            else:
                print("❌ Multimodal pipeline returned None/empty result")
                
        except Exception as e:
            print(f"❌ Multimodal pipeline processing failed: {e}")
            import traceback
            print(f"   🔍 Traceback: {traceback.format_exc()}")
        
        # Test memory store addition
        try:
            print("\n🔄 Testing memory store addition...")
            from memory.memory_vectorstore import get_memory_store, VectorStoreType
            
            memory_store = get_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384
            )
            
            # Add test document to memory store
            chunk_id = memory_store.add_memory(
                content=f"Test Document: diagnostic_test.txt\n\n{test_content}",
                source="diagnostic_test",
                tags=['test', 'diagnostic'],
                importance_score=0.8,
                metadata={
                    'filename': 'diagnostic_test.txt',
                    'test_document': True,
                    'diagnostic_timestamp': str(os.path.getmtime(temp_file_path))
                }
            )
            
            print(f"✅ Successfully added to memory store with ID: {chunk_id}")
            
            # Test retrieval
            search_results = memory_store.search_memories(
                query="diagnostic test document",
                max_results=5
            )
            
            print(f"   🔍 Search found {len(search_results)} results")
            for i, result in enumerate(search_results):
                if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                    content_preview = result.chunk.content[:100] + "..." if len(result.chunk.content) > 100 else result.chunk.content
                    print(f"   📄 Result {i+1}: {content_preview}")
                    
        except Exception as e:
            print(f"❌ Memory store test failed: {e}")
            import traceback
            print(f"   🔍 Traceback: {traceback.format_exc()}")
        
        # Cleanup
        try:
            os.unlink(temp_file_path)
            print(f"🧹 Cleaned up test file: {temp_file_path}")
        except Exception as e:
            print(f"⚠️  Could not clean up test file: {e}")
            
    except Exception as e:
        print(f"❌ Test document creation failed: {e}")

def check_dependencies():
    """Check required dependencies."""
    print("🔍 STEP 5: Dependencies Check")
    print("=" * 50)
    
    required_packages = [
        'chromadb',
        'sentence_transformers', 
        'PyPDF2',
        'streamlit'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Available")
        except ImportError:
            print(f"❌ {package}: MISSING")
    
    # Check specific components
    components_to_check = [
        ('memory.memory_vectorstore', 'MemoryVectorStore'),
        ('multimodal_processing.multimodal_pipeline', 'MultimodalProcessingPipeline'),
        ('multimodal_processing.document_parser', 'DocumentParser'),
        ('multimodal_processing.knowledge_consolidator', 'KnowledgeConsolidator')
    ]
    
    print("\n🔍 SAM Components Check:")
    for module_name, component_name in components_to_check:
        try:
            module = __import__(module_name, fromlist=[component_name])
            if hasattr(module, component_name):
                print(f"✅ {module_name}.{component_name}: Available")
            else:
                print(f"⚠️  {module_name}: Module exists but {component_name} not found")
        except ImportError as e:
            print(f"❌ {module_name}: Import failed - {e}")

def main():
    """Run complete diagnostic."""
    print("🔧 SAM Document Processing Diagnostic")
    print("=" * 60)
    print("This script will identify why SAM cannot read uploaded documents.\n")
    
    # Run all diagnostic steps
    check_file_system()
    pipeline = check_multimodal_pipeline()
    check_memory_stores()
    test_document_processing(pipeline)
    check_dependencies()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Review the output above for any ❌ FAILED items")
    print("2. If multimodal pipeline failed, check multimodal_processing/ directory")
    print("3. If memory stores failed, check ChromaDB installation")
    print("4. If test processing failed, check the specific error messages")
    print("5. Run this script again after fixing any issues")
    print("\n💡 If you see ✅ SUCCESS for all steps but still can't read documents,")
    print("   the issue might be in the Streamlit app initialization or security.")

if __name__ == "__main__":
    main()
