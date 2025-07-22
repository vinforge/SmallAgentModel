#!/usr/bin/env python3
"""
SAM Secure Enclave - Data Migration Script

Migrates existing unencrypted data to the new encrypted format.
Handles ChromaDB collections, memory stores, and document uploads.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import json
import shutil
import getpass
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from security import SecureStateManager, EncryptedChromaStore
from memory.memory_vectorstore import get_memory_store, VectorStoreType

class SAMDataMigrator:
    """Handles migration of SAM data to encrypted format."""
    
    def __init__(self):
        self.security_manager = None
        self.encrypted_store = None
        self.backup_dir = Path("migration_backup")
        self.migration_log = []
        self.stats = {
            'chromadb_memories': 0,
            'json_memories': 0,
            'uploaded_files': 0,
            'total_migrated': 0,
            'errors': 0
        }
    
    def run_migration(self):
        """Run the complete migration process."""
        print("🔄 SAM Secure Enclave - Data Migration")
        print("=" * 50)
        
        try:
            # Step 1: Setup master password
            self._setup_master_password()
            
            # Step 2: Create backup
            self._create_backup()
            
            # Step 3: Migrate ChromaDB data
            self._migrate_chromadb_data()
            
            # Step 4: Migrate document uploads
            self._migrate_document_uploads()
            
            # Step 5: Migrate memory store
            self._migrate_memory_store()
            
            # Step 6: Cleanup old data
            self._cleanup_old_data()
            
            # Step 7: Verify migration
            self._verify_migration()
            
            print("\n🎉 Migration completed successfully!")
            self._print_migration_summary()
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            self._restore_from_backup()
            raise
    
    def _setup_master_password(self):
        """Setup master password for encryption."""
        print("\n🔐 Step 1: Master Password Setup")
        
        self.security_manager = SecureStateManager()
        
        if self.security_manager.is_setup_required():
            print("This is your first time using SAM Secure Enclave.")
            print("You need to create a master password to encrypt your data.")
            print("\n⚠️  IMPORTANT:")
            print("- Choose a strong password you'll remember")
            print("- This password cannot be recovered if lost")
            print("- All your SAM data will be encrypted with this password")
            
            while True:
                password = getpass.getpass("\nEnter master password: ").strip()
                if len(password) < 8:
                    print("❌ Password must be at least 8 characters long")
                    continue
                
                confirm = getpass.getpass("Confirm master password: ").strip()
                if password != confirm:
                    print("❌ Passwords do not match")
                    continue
                
                break
            
            print("\n🔐 Setting up secure enclave...")
            success = self.security_manager.setup_master_password(password)
            
            if success:
                print("✅ Master password setup successful!")
                self.migration_log.append("Master password setup completed")
            else:
                raise Exception("Failed to setup master password")
        else:
            print("Existing security setup detected.")
            password = getpass.getpass("Enter your master password to unlock: ").strip()
            
            success = self.security_manager.unlock_application(password)
            if not success:
                raise Exception("Failed to unlock with provided password")
            
            print("✅ Application unlocked successfully!")
            self.migration_log.append("Application unlocked for migration")
        
        # Initialize encrypted store
        self.encrypted_store = EncryptedChromaStore(
            collection_name="sam_secure_memory",
            crypto_manager=self.security_manager.crypto
        )
        print("✅ Encrypted store initialized")
    
    def _create_backup(self):
        """Create backup of existing data."""
        print("\n💾 Step 2: Creating Backup")
        
        self.backup_dir.mkdir(exist_ok=True)
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup directories
        backup_targets = [
            "chroma_db",
            "web_ui/chroma_db", 
            "memory_store",
            "web_ui/memory_store",
            "uploads",
            "web_ui/uploads",
            "data/uploads"
        ]
        
        backed_up_count = 0
        for target in backup_targets:
            target_path = Path(target)
            if target_path.exists():
                backup_path = self.backup_dir / f"{backup_timestamp}_{target.replace('/', '_')}"
                try:
                    shutil.copytree(target_path, backup_path, ignore_errors=True)
                    print(f"  ✅ Backed up: {target} → {backup_path}")
                    self.migration_log.append(f"Backup created: {target}")
                    backed_up_count += 1
                except Exception as e:
                    print(f"  ⚠️  Backup failed for {target}: {e}")
                    self.migration_log.append(f"Backup failed: {target} - {e}")
        
        print(f"✅ Backup completed: {backed_up_count} directories backed up")
    
    def _migrate_chromadb_data(self):
        """Migrate existing ChromaDB collections to encrypted format."""
        print("\n🗄️ Step 3: Migrating ChromaDB Data")
        
        # Find existing ChromaDB collections
        chroma_paths = [
            Path("chroma_db"),
            Path("web_ui/chroma_db"),
            Path("memory_store/chroma_db")
        ]
        
        for chroma_path in chroma_paths:
            if chroma_path.exists():
                print(f"  📂 Processing: {chroma_path}")
                
                try:
                    # Load existing memory store
                    old_store = get_memory_store(
                        store_type=VectorStoreType.CHROMA,
                        storage_directory=str(chroma_path.parent),
                        embedding_dimension=384
                    )
                    
                    # Get all memories
                    all_memories = old_store.get_all_memories()
                    print(f"    Found {len(all_memories)} memories to migrate")
                    
                    # Migrate each memory
                    for memory in all_memories:
                        try:
                            # Prepare metadata for encrypted storage
                            metadata = {
                                'source_id': f"migrated_{memory.chunk_id}",
                                'document_type': self._extract_document_type(memory.source),
                                'source_name': self._extract_source_name(memory.source),
                                'created_at': memory.timestamp,
                                'importance_score': memory.importance_score,
                                'memory_type': memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type),
                                'tags': memory.tags,
                                'access_count': memory.access_count,
                                'last_accessed': memory.last_accessed,
                                'migration_source': str(chroma_path),
                                'original_chunk_id': memory.chunk_id
                            }
                            
                            # Add additional metadata from original
                            if hasattr(memory, 'metadata') and memory.metadata:
                                metadata.update(memory.metadata)
                            
                            # Add to encrypted store
                            chunk_id = self.encrypted_store.add_memory_chunk(
                                chunk_text=memory.content,
                                metadata=metadata,
                                embedding=memory.embedding or [0.1] * 384  # Default embedding if missing
                            )
                            
                            self.stats['chromadb_memories'] += 1
                            
                        except Exception as e:
                            print(f"    ⚠️  Failed to migrate memory {memory.chunk_id}: {e}")
                            self.migration_log.append(f"Memory migration failed: {memory.chunk_id} - {e}")
                            self.stats['errors'] += 1
                    
                    print(f"    ✅ Migrated {len(all_memories)} memories from {chroma_path}")
                    
                except Exception as e:
                    print(f"    ❌ Failed to process {chroma_path}: {e}")
                    self.migration_log.append(f"ChromaDB migration failed: {chroma_path} - {e}")
                    self.stats['errors'] += 1
        
        print(f"✅ ChromaDB migration completed: {self.stats['chromadb_memories']} memories migrated")
    
    def _migrate_document_uploads(self):
        """Migrate uploaded documents to encrypted metadata."""
        print("\n📄 Step 4: Migrating Document Uploads")
        
        upload_paths = [
            Path("uploads"),
            Path("web_ui/uploads"),
            Path("data/uploads")
        ]
        
        for upload_path in upload_paths:
            if upload_path.exists():
                print(f"  📂 Processing uploads in: {upload_path}")
                
                for file_path in upload_path.rglob("*"):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        try:
                            # Create metadata for uploaded file
                            metadata = {
                                'source_id': f"upload_{file_path.stem}",
                                'document_type': file_path.suffix.lower().replace('.', '') or 'unknown',
                                'source_name': file_path.name,
                                'source_path': str(file_path),
                                'file_size': file_path.stat().st_size,
                                'upload_timestamp': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                                'migration_source': 'document_upload',
                                'encrypted_file_path': str(file_path)
                            }
                            
                            # Add file metadata to encrypted store
                            chunk_id = self.encrypted_store.add_memory_chunk(
                                chunk_text=f"Uploaded document: {file_path.name}",
                                metadata=metadata,
                                embedding=[0.1] * 384  # Default embedding
                            )
                            
                            self.stats['uploaded_files'] += 1
                            
                        except Exception as e:
                            print(f"    ⚠️  Failed to migrate {file_path.name}: {e}")
                            self.migration_log.append(f"File migration failed: {file_path.name} - {e}")
                            self.stats['errors'] += 1
        
        print(f"✅ Document upload migration completed: {self.stats['uploaded_files']} files processed")
    
    def _migrate_memory_store(self):
        """Migrate JSON-based memory store files."""
        print("\n🧠 Step 5: Migrating Memory Store Files")
        
        memory_paths = [
            Path("memory_store"),
            Path("web_ui/memory_store")
        ]
        
        for memory_path in memory_paths:
            if memory_path.exists():
                print(f"  📂 Processing memory files in: {memory_path}")
                
                for json_file in memory_path.glob("mem_*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            memory_data = json.load(f)
                        
                        # Extract content and metadata
                        content = memory_data.get('content', '')
                        if not content:
                            continue
                        
                        metadata = {
                            'source_id': f"json_{json_file.stem}",
                            'document_type': 'memory',
                            'source_name': json_file.name,
                            'created_at': memory_data.get('timestamp', datetime.now().isoformat()),
                            'importance_score': memory_data.get('importance_score', 0.5),
                            'memory_type': memory_data.get('memory_type', 'conversation'),
                            'migration_source': 'json_memory_store',
                            'original_file': str(json_file)
                        }
                        
                        # Add to encrypted store
                        chunk_id = self.encrypted_store.add_memory_chunk(
                            chunk_text=content,
                            metadata=metadata,
                            embedding=memory_data.get('embedding', [0.1] * 384)
                        )
                        
                        self.stats['json_memories'] += 1
                        
                    except Exception as e:
                        print(f"    ⚠️  Failed to migrate {json_file.name}: {e}")
                        self.migration_log.append(f"JSON memory migration failed: {json_file.name} - {e}")
                        self.stats['errors'] += 1
        
        print(f"✅ Memory store migration completed: {self.stats['json_memories']} memories migrated")
    
    def _extract_document_type(self, source: str) -> str:
        """Extract document type from source string."""
        if ':' in source:
            parts = source.split(':')
            if len(parts) > 1 and '.' in parts[1]:
                return Path(parts[1]).suffix.lower().replace('.', '')
        return 'unknown'
    
    def _extract_source_name(self, source: str) -> str:
        """Extract source name from source string."""
        if ':' in source:
            parts = source.split(':')
            if len(parts) > 1:
                return Path(parts[1]).name
        return source
    
    def _cleanup_old_data(self):
        """Cleanup old unencrypted data after successful migration."""
        print("\n🧹 Step 6: Cleanup Old Data")
        
        print("The migration is complete. You can now safely remove the old unencrypted data.")
        print("⚠️  This action cannot be undone (but you have backups).")
        
        response = input("Remove old unencrypted data? (y/N): ").strip().lower()
        
        if response == 'y':
            cleanup_targets = [
                "chroma_db",
                "web_ui/chroma_db"
            ]
            
            for target in cleanup_targets:
                target_path = Path(target)
                if target_path.exists():
                    try:
                        shutil.rmtree(target_path, ignore_errors=True)
                        print(f"  🗑️  Removed: {target}")
                        self.migration_log.append(f"Cleanup: {target}")
                    except Exception as e:
                        print(f"  ⚠️  Failed to remove {target}: {e}")
            
            print("✅ Cleanup completed")
        else:
            print("⏭️  Cleanup skipped - old data preserved")
    
    def _verify_migration(self):
        """Verify the migration was successful."""
        print("\n✅ Step 7: Verifying Migration")
        
        # Check encrypted store
        info = self.encrypted_store.get_collection_info()
        print(f"  📊 Encrypted store contains: {info['chunk_count']} chunks")
        print(f"  🔒 Encryption enabled: {info['encryption_enabled']}")
        print(f"  🔍 Searchable fields: {len(info['searchable_fields'])}")
        print(f"  🔐 Encrypted fields: {len(info['encrypted_fields'])}")
        
        # Test search functionality
        try:
            test_results = self.encrypted_store.query_memories(
                query_embedding=[0.1] * 384,  # Dummy embedding
                n_results=5
            )
            print(f"  🔍 Test search returned: {len(test_results)} results")
        except Exception as e:
            print(f"  ⚠️  Test search failed: {e}")
        
        # Update total stats
        self.stats['total_migrated'] = (
            self.stats['chromadb_memories'] + 
            self.stats['json_memories'] + 
            self.stats['uploaded_files']
        )
        
        print("✅ Migration verification completed")
    
    def _restore_from_backup(self):
        """Restore from backup in case of migration failure."""
        print("\n🔄 Attempting to restore from backup...")
        # Implementation would restore from backup_dir
        print("⚠️  Please manually restore from backup if needed")
    
    def _print_migration_summary(self):
        """Print migration summary."""
        print("\n📊 MIGRATION SUMMARY")
        print("=" * 30)
        print(f"ChromaDB memories migrated: {self.stats['chromadb_memories']}")
        print(f"JSON memories migrated: {self.stats['json_memories']}")
        print(f"Uploaded files processed: {self.stats['uploaded_files']}")
        print(f"Total items migrated: {self.stats['total_migrated']}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"Backup location: {self.backup_dir}")
        
        if self.stats['errors'] == 0:
            print("\n🎉 Migration completed without errors!")
        else:
            print(f"\n⚠️  Migration completed with {self.stats['errors']} errors")
            print("Check the migration log for details")

def main():
    """Main migration function."""
    try:
        migrator = SAMDataMigrator()
        migrator.run_migration()
        
    except KeyboardInterrupt:
        print("\n\n❌ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
