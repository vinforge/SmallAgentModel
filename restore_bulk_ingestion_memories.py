#!/usr/bin/env python3
"""
Script to restore bulk ingestion memories to the memory store.
"""

import sys
import sqlite3
from pathlib import Path
sys.path.append('.')
sys.path.append('scripts')

def get_processed_files():
    """Get list of successfully processed files from ingestion database."""
    print("🔍 Checking bulk ingestion database...")
    
    db_path = Path('data/ingestion_state.db')
    if not db_path.exists():
        print("❌ No ingestion state database found")
        return []
    
    processed_files = []
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute('''
                SELECT filepath, chunks_created, processed_at 
                FROM processed_files 
                WHERE status = "success" AND chunks_created > 0
                ORDER BY processed_at DESC
            ''')
            
            for filepath, chunks_created, processed_at in cursor.fetchall():
                file_path = Path(filepath)
                if file_path.exists():
                    processed_files.append({
                        'path': file_path,
                        'chunks_created': chunks_created,
                        'processed_at': processed_at
                    })
                else:
                    print(f"⚠️  File not found: {filepath}")
            
        print(f"📊 Found {len(processed_files)} successfully processed files")
        return processed_files
        
    except Exception as e:
        print(f"❌ Error reading ingestion database: {e}")
        return []

def restore_memories_from_files(processed_files, max_files=None):
    """Restore memories by re-processing files through multimodal pipeline."""
    print(f"\n🔄 Restoring memories from {len(processed_files)} files...")
    
    if max_files:
        processed_files = processed_files[:max_files]
        print(f"📊 Limited to first {max_files} files for testing")
    
    try:
        from memory.memory_vectorstore import get_memory_store
        from multimodal_processing.multimodal_pipeline import MultimodalProcessingPipeline
        
        # Get initial memory count
        memory_store = get_memory_store()
        initial_stats = memory_store.get_memory_stats()
        initial_count = initial_stats.get('total_memories', 0)
        print(f"📊 Initial memory count: {initial_count}")
        
        # Initialize pipeline
        pipeline = MultimodalProcessingPipeline()
        
        # Process files
        restored_count = 0
        failed_count = 0
        
        for i, file_info in enumerate(processed_files, 1):
            file_path = file_info['path']
            expected_chunks = file_info['chunks_created']
            
            print(f"📈 Progress: {i}/{len(processed_files)} - {file_path.name}")
            print(f"   Expected chunks: {expected_chunks}")
            
            try:
                # Process through pipeline
                result = pipeline.process_document(file_path)
                
                if result:
                    memory_storage = result.get('memory_storage', {})
                    actual_chunks = memory_storage.get('total_chunks_stored', 0)
                    
                    print(f"   ✅ Processed: {actual_chunks} chunks stored")
                    restored_count += 1
                    
                    if actual_chunks != expected_chunks:
                        print(f"   ⚠️  Chunk count mismatch: expected {expected_chunks}, got {actual_chunks}")
                else:
                    print(f"   ❌ Processing failed")
                    failed_count += 1
                    
            except Exception as e:
                print(f"   ❌ Error processing {file_path.name}: {e}")
                failed_count += 1
        
        # Check final memory count
        final_stats = memory_store.get_memory_stats()
        final_count = final_stats.get('total_memories', 0)
        memories_added = final_count - initial_count
        
        print(f"\n📊 Restoration Summary:")
        print(f"   Files processed: {restored_count}")
        print(f"   Files failed: {failed_count}")
        print(f"   Initial memories: {initial_count}")
        print(f"   Final memories: {final_count}")
        print(f"   Memories added: {memories_added}")
        
        return memories_added > 0
        
    except Exception as e:
        print(f"❌ Error during restoration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main restoration process."""
    print("🚀 Bulk Ingestion Memory Restoration")
    print("=" * 60)
    
    # Get processed files
    processed_files = get_processed_files()
    
    if not processed_files:
        print("❌ No processed files found to restore")
        return
    
    # Show summary
    total_expected_chunks = sum(f['chunks_created'] for f in processed_files)
    print(f"📊 Restoration Plan:")
    print(f"   Files to process: {len(processed_files)}")
    print(f"   Expected total chunks: {total_expected_chunks}")
    
    # Ask for confirmation
    print(f"\n⚠️  This will re-process all {len(processed_files)} files.")
    print(f"   This may take 10-30 minutes depending on file sizes.")
    
    # For testing, limit to first 5 files
    test_mode = True
    if test_mode:
        print(f"\n🧪 TEST MODE: Processing first 5 files only")
        max_files = 5
    else:
        response = input(f"\nProceed with full restoration? (y/N): ")
        if response.lower() != 'y':
            print("❌ Restoration cancelled")
            return
        max_files = None
    
    # Restore memories
    success = restore_memories_from_files(processed_files, max_files)
    
    if success:
        print(f"\n✅ Memory restoration completed!")
        print(f"\n🎨 Next steps:")
        print(f"   1. Go to Dream Canvas")
        print(f"   2. Click 'Generate Dream Canvas'")
        print(f"   3. Should now process many more memories")
        print(f"   4. Should take longer and show meaningful clusters")
    else:
        print(f"\n❌ Memory restoration failed")
        print(f"\n💡 Troubleshooting:")
        print(f"   - Check if files still exist at original locations")
        print(f"   - Verify multimodal pipeline is working")
        print(f"   - Check memory store configuration")

if __name__ == "__main__":
    main()
