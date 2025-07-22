"""
Memory Backend Migration Tool (Task 33, Phase 3)
===============================================

Tool for migrating data between different memory backends.
Enables safe testing and transition between backend implementations.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from .base_backend import BaseMemoryBackend, BackendType
from .backend_factory import create_memory_backend, MemoryBackendConfig

logger = logging.getLogger(__name__)

class MemoryMigrationTool:
    """Tool for migrating memories between different backends."""
    
    def __init__(self):
        """Initialize the migration tool."""
        self.logger = logging.getLogger(f"{__name__}.MemoryMigrationTool")
    
    def migrate_backend_to_backend(self,
                                  source_backend: BaseMemoryBackend,
                                  target_backend: BaseMemoryBackend,
                                  batch_size: int = 100,
                                  dry_run: bool = False) -> Dict[str, Any]:
        """
        Migrate memories from one backend to another.
        
        Args:
            source_backend: Source memory backend
            target_backend: Target memory backend
            batch_size: Number of memories to process at once
            dry_run: If True, only simulate the migration
            
        Returns:
            Migration results and statistics
        """
        try:
            self.logger.info(f"Starting migration from {source_backend.__class__.__name__} to {target_backend.__class__.__name__}")
            
            # Get source statistics
            source_stats = source_backend.get_memory_stats()
            self.logger.info(f"Source backend stats: {source_stats}")
            
            # Export from source
            export_path = f"migration_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            if not dry_run:
                export_success = source_backend.export_memories(export_path)
                if not export_success:
                    raise RuntimeError("Failed to export from source backend")
            
            # Import to target
            imported_count = 0
            if not dry_run:
                imported_count = target_backend.import_memories(export_path)
                
                # Clean up export file
                try:
                    Path(export_path).unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to clean up export file: {e}")
            
            # Get target statistics
            target_stats = target_backend.get_memory_stats()
            
            migration_result = {
                'status': 'completed' if not dry_run else 'dry_run',
                'source_backend': source_backend.__class__.__name__,
                'target_backend': target_backend.__class__.__name__,
                'source_stats': source_stats,
                'target_stats': target_stats,
                'imported_count': imported_count,
                'dry_run': dry_run,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Migration completed: {migration_result}")
            return migration_result
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def migrate_native_to_mem0(self, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate from SAM Native to Mem0 backend."""
        try:
            # Create backends
            native_config = MemoryBackendConfig(backend_type=BackendType.SAM_NATIVE)
            native_backend = create_memory_backend(BackendType.SAM_NATIVE, native_config)
            
            mem0_config = MemoryBackendConfig(backend_type=BackendType.MEM0)
            mem0_backend = create_memory_backend(BackendType.MEM0, mem0_config)
            
            return self.migrate_backend_to_backend(native_backend, mem0_backend, dry_run=dry_run)
            
        except Exception as e:
            self.logger.error(f"Native to Mem0 migration failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def migrate_mem0_to_native(self, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate from Mem0 to SAM Native backend."""
        try:
            # Create backends
            mem0_config = MemoryBackendConfig(backend_type=BackendType.MEM0)
            mem0_backend = create_memory_backend(BackendType.MEM0, mem0_config)
            
            native_config = MemoryBackendConfig(backend_type=BackendType.SAM_NATIVE)
            native_backend = create_memory_backend(BackendType.SAM_NATIVE, native_config)
            
            return self.migrate_backend_to_backend(mem0_backend, native_backend, dry_run=dry_run)
            
        except Exception as e:
            self.logger.error(f"Mem0 to Native migration failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def validate_migration(self,
                          source_backend: BaseMemoryBackend,
                          target_backend: BaseMemoryBackend,
                          sample_size: int = 10) -> Dict[str, Any]:
        """
        Validate that migration was successful by comparing samples.
        
        Args:
            source_backend: Source backend
            target_backend: Target backend
            sample_size: Number of memories to sample for validation
            
        Returns:
            Validation results
        """
        try:
            self.logger.info("Starting migration validation")
            
            # Get sample memories from source
            sample_query = "test validation sample"
            source_results = source_backend.search_memories(sample_query, limit=sample_size)
            
            validation_results = {
                'total_sampled': len(source_results),
                'successful_matches': 0,
                'failed_matches': 0,
                'content_mismatches': 0,
                'missing_memories': 0,
                'validation_details': []
            }
            
            for source_result in source_results:
                # Search for equivalent in target
                target_results = target_backend.search_memories(
                    source_result.content[:100],  # Use content snippet as query
                    limit=1
                )
                
                if not target_results:
                    validation_results['missing_memories'] += 1
                    validation_results['validation_details'].append({
                        'chunk_id': source_result.chunk_id,
                        'status': 'missing',
                        'content_preview': source_result.content[:50]
                    })
                    continue
                
                target_result = target_results[0]
                
                # Compare content
                if source_result.content.strip() == target_result.content.strip():
                    validation_results['successful_matches'] += 1
                    validation_results['validation_details'].append({
                        'chunk_id': source_result.chunk_id,
                        'status': 'match',
                        'similarity_score': target_result.similarity_score
                    })
                else:
                    validation_results['content_mismatches'] += 1
                    validation_results['validation_details'].append({
                        'chunk_id': source_result.chunk_id,
                        'status': 'content_mismatch',
                        'source_preview': source_result.content[:50],
                        'target_preview': target_result.content[:50]
                    })
            
            # Calculate success rate
            total_checks = validation_results['total_sampled']
            successful = validation_results['successful_matches']
            validation_results['success_rate'] = successful / total_checks if total_checks > 0 else 0
            
            self.logger.info(f"Validation completed: {successful}/{total_checks} successful matches")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Migration validation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def create_backup(self, backend: BaseMemoryBackend, backup_path: str) -> bool:
        """
        Create a backup of a memory backend.
        
        Args:
            backend: Backend to backup
            backup_path: Path for backup file
            
        Returns:
            True if backup successful
        """
        try:
            self.logger.info(f"Creating backup to {backup_path}")
            
            success = backend.export_memories(backup_path)
            
            if success:
                self.logger.info("Backup created successfully")
            else:
                self.logger.error("Backup creation failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return False
    
    def restore_backup(self, backend: BaseMemoryBackend, backup_path: str) -> int:
        """
        Restore a memory backend from backup.
        
        Args:
            backend: Backend to restore to
            backup_path: Path to backup file
            
        Returns:
            Number of memories restored
        """
        try:
            self.logger.info(f"Restoring backup from {backup_path}")
            
            restored_count = backend.import_memories(backup_path)
            
            self.logger.info(f"Restored {restored_count} memories from backup")
            return restored_count
            
        except Exception as e:
            self.logger.error(f"Backup restoration failed: {e}")
            return 0

# Convenience functions
def migrate_to_mem0(dry_run: bool = False) -> Dict[str, Any]:
    """Migrate current SAM memories to Mem0 backend."""
    tool = MemoryMigrationTool()
    return tool.migrate_native_to_mem0(dry_run=dry_run)

def migrate_to_native(dry_run: bool = False) -> Dict[str, Any]:
    """Migrate current Mem0 memories to SAM Native backend."""
    tool = MemoryMigrationTool()
    return tool.migrate_mem0_to_native(dry_run=dry_run)

def validate_backends() -> Dict[str, Any]:
    """Validate that both backends are working correctly."""
    try:
        from .backend_factory import compare_backends
        return compare_backends()
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def create_memory_backup(backend_type: str = "sam_native", backup_path: str = None) -> bool:
    """Create a backup of the specified backend."""
    try:
        if backup_path is None:
            backup_path = f"sam_memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create backend
        if backend_type.lower() == "mem0":
            config = MemoryBackendConfig(backend_type=BackendType.MEM0)
            backend = create_memory_backend(BackendType.MEM0, config)
        else:
            config = MemoryBackendConfig(backend_type=BackendType.SAM_NATIVE)
            backend = create_memory_backend(BackendType.SAM_NATIVE, config)
        
        tool = MemoryMigrationTool()
        return tool.create_backup(backend, backup_path)
        
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        return False
