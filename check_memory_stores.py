#!/usr/bin/env python3
"""
Quick Memory Store Checker
Shows what documents are currently stored in SAM's memory.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_regular_memory_store():
    """Check the regular memory store."""
    print("🔍 Regular Memory Store")
    print("=" * 30)
    
    try:
        from memory.memory_vectorstore import get_memory_store, VectorStoreType
        
        memory_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384
        )
        
        print("✅ Memory store connected")
        
        # Search for all documents
        all_docs = memory_store.search_memories(
            query="",
            memory_type="document", 
            max_results=100
        )
        
        print(f"📄 Total documents: {len(all_docs)}")
        
        if all_docs:
            print("\n📋 Documents found:")
            filenames = set()
            for i, doc in enumerate(all_docs):
                filename = "Unknown"
                source = "Unknown"
                
                # Extract filename and source
                if hasattr(doc, 'metadata') and doc.metadata:
                    filename = doc.metadata.get('filename', 'Unknown')
                    source = doc.metadata.get('source', 'Unknown')
                elif hasattr(doc, 'chunk'):
                    if hasattr(doc.chunk, 'metadata') and doc.chunk.metadata:
                        filename = doc.chunk.metadata.get('filename', 'Unknown')
                    source = getattr(doc.chunk, 'source', 'Unknown')
                
                filenames.add(filename)
                
                # Show content preview
                content = ""
                if hasattr(doc, 'content'):
                    content = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                elif hasattr(doc, 'chunk') and hasattr(doc.chunk, 'content'):
                    content = doc.chunk.content[:100] + "..." if len(doc.chunk.content) > 100 else doc.chunk.content
                
                print(f"   {i+1}. {filename} (Source: {source})")
                print(f"      Content: {content}")
                print()
            
            print(f"📊 Unique filenames: {list(filenames)}")
        else:
            print("❌ No documents found in regular memory store")
            
    except Exception as e:
        print(f"❌ Error accessing regular memory store: {e}")

def check_secure_memory_store():
    """Check the secure memory store."""
    print("\n🔐 Secure Memory Store")
    print("=" * 30)
    
    try:
        from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType
        
        secure_store = get_secure_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384,
            enable_encryption=False  # Disable encryption for checking
        )
        
        print("✅ Secure memory store connected")
        
        # Get security status
        try:
            status = secure_store.get_security_status()
            print(f"🔐 Security status: {status}")
        except:
            print("⚠️  Could not get security status")
        
        # Search for all documents
        all_docs = secure_store.search_memories(
            query="",
            memory_type="document",
            max_results=100
        )
        
        print(f"📄 Total documents: {len(all_docs)}")
        
        if all_docs:
            print("\n📋 Documents found:")
            for i, doc in enumerate(all_docs):
                filename = "Unknown"
                
                # Extract filename
                if hasattr(doc, 'metadata') and doc.metadata:
                    filename = doc.metadata.get('filename', 'Unknown')
                
                # Show content preview
                content = ""
                if hasattr(doc, 'content'):
                    content = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                
                print(f"   {i+1}. {filename}")
                print(f"      Content: {content}")
                print()
        else:
            print("❌ No documents found in secure memory store")
            
    except Exception as e:
        print(f"❌ Error accessing secure memory store: {e}")

def check_chromadb_files():
    """Check for ChromaDB files on disk."""
    print("\n💾 ChromaDB Files on Disk")
    print("=" * 30)
    
    # Check current directory and subdirectories
    chromadb_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if "chroma" in file.lower() or file.endswith(".sqlite3"):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                chromadb_files.append((file_path, file_size))
    
    if chromadb_files:
        print("📄 ChromaDB files found:")
        for file_path, file_size in chromadb_files:
            print(f"   {file_path} ({file_size} bytes)")
    else:
        print("❌ No ChromaDB files found")
    
    # Check specific directories
    directories_to_check = ["memory_store", "chroma_db", "data"]
    for directory in directories_to_check:
        if os.path.exists(directory):
            print(f"\n📂 Contents of {directory}:")
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path)
                        print(f"   📄 {item} ({size} bytes)")
                    else:
                        print(f"   📁 {item}/")
            except PermissionError:
                print(f"   ⚠️  Permission denied")

def test_simple_query():
    """Test a simple query to see if documents are retrievable."""
    print("\n🔍 Simple Query Test")
    print("=" * 30)
    
    try:
        from memory.memory_vectorstore import get_memory_store, VectorStoreType
        
        memory_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384
        )
        
        # Test queries
        test_queries = [
            "document",
            "test",
            "pdf",
            "upload",
            ""  # Empty query to get all
        ]
        
        for query in test_queries:
            results = memory_store.search_memories(
                query=query,
                max_results=5
            )
            print(f"Query '{query}': {len(results)} results")
            
            if results and query:  # Show details for non-empty queries
                for i, result in enumerate(results[:2]):  # Show first 2
                    content = ""
                    if hasattr(result, 'content'):
                        content = result.content[:50] + "..." if len(result.content) > 50 else result.content
                    elif hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                        content = result.chunk.content[:50] + "..." if len(result.chunk.content) > 50 else result.chunk.content
                    print(f"   {i+1}. {content}")
        
    except Exception as e:
        print(f"❌ Query test failed: {e}")

def main():
    """Run all checks."""
    print("🔍 SAM Memory Store Checker")
    print("=" * 50)
    
    check_regular_memory_store()
    check_secure_memory_store()
    check_chromadb_files()
    test_simple_query()
    
    print("\n" + "=" * 50)
    print("🎯 MEMORY CHECK COMPLETE")
    print("=" * 50)
    print("\n💡 What this tells us:")
    print("- If both stores show 0 documents, uploads aren't working")
    print("- If ChromaDB files exist but are small, they might be empty")
    print("- If query test returns 0 results, documents aren't indexed properly")
    print("- Check the diagnostic script for more detailed analysis")

if __name__ == "__main__":
    main()
