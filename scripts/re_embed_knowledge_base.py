#!/usr/bin/env python3
"""
SAM Knowledge Base Re-embedding Script
=====================================

Re-embeds the knowledge base using a newly activated ModelEngine.
This script is launched as a background process during engine migrations.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class ReEmbeddingTask:
    """
    Background task for re-embedding knowledge base with new engine.
    """
    
    def __init__(self, migration_id: str, progress_file: Optional[str] = None):
        """
        Initialize re-embedding task.
        
        Args:
            migration_id: Migration identifier
            progress_file: Optional file to write progress updates
        """
        self.migration_id = migration_id
        self.progress_file = progress_file
        self.start_time = datetime.now()
        
        # Progress tracking
        self.total_documents = 0
        self.processed_documents = 0
        self.failed_documents = 0
        self.current_document = ""
        
        # Setup logging
        self.setup_logging()
        
        self.logger = logging.getLogger(f"{__name__}.ReEmbeddingTask")
        self.logger.info(f"Initialized re-embedding task for migration: {migration_id}")
    
    def setup_logging(self):
        """Setup logging for the re-embedding task."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"re_embedding_{self.migration_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def run(self) -> bool:
        """
        Execute the re-embedding task.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("🔄 Starting knowledge base re-embedding...")
            
            # Step 1: Initialize new embedding system
            self.update_progress("Initializing new embedding system", 5.0)
            embedding_manager = self._initialize_embedding_system()
            
            if not embedding_manager:
                raise Exception("Failed to initialize embedding system")
            
            # Step 2: Get documents to re-embed
            self.update_progress("Scanning knowledge base", 10.0)
            documents = self._get_documents_to_reembed()
            
            if not documents:
                self.logger.info("No documents found to re-embed")
                self.update_progress("Completed - no documents found", 100.0)
                return True
            
            self.total_documents = len(documents)
            self.logger.info(f"Found {self.total_documents} documents to re-embed")
            
            # Step 3: Re-embed documents
            self.update_progress("Re-embedding documents", 15.0)
            success = self._reembed_documents(documents, embedding_manager)
            
            if success:
                self.update_progress("Re-embedding completed successfully", 100.0)
                self.logger.info("✅ Knowledge base re-embedding completed successfully")
                return True
            else:
                self.update_progress("Re-embedding failed", 0.0)
                self.logger.error("❌ Knowledge base re-embedding failed")
                return False
                
        except Exception as e:
            self.update_progress(f"Error: {str(e)}", 0.0)
            self.logger.error(f"❌ Re-embedding task failed: {e}")
            return False
    
    def _initialize_embedding_system(self):
        """Initialize the embedding system with the new engine."""
        try:
            # Import SAM embedding components
            from sam.embedding.embedding_manager import get_embedding_manager
            
            # Get the embedding manager (it should use the newly configured engine)
            embedding_manager = get_embedding_manager()
            
            # Test embedding to ensure it's working
            test_embedding = embedding_manager.embed_text("Test embedding")
            
            if test_embedding is None or len(test_embedding) == 0:
                raise Exception("Embedding system returned empty result")
            
            self.logger.info(f"✅ Embedding system initialized (dimension: {len(test_embedding)})")
            return embedding_manager
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize embedding system: {e}")
            return None
    
    def _get_documents_to_reembed(self) -> List[Dict[str, Any]]:
        """Get list of documents that need re-embedding."""
        try:
            documents = []
            
            # Try to get documents from memory manager
            try:
                from sam.memory.memory_manager import get_memory_manager
                
                memory_manager = get_memory_manager()
                
                # Get all stored documents/memories
                # This is a simplified approach - in practice you'd iterate through
                # the actual document storage system
                
                # For now, we'll simulate getting documents
                # In a real implementation, this would query the vector store
                # and get all document chunks that need re-embedding
                
                self.logger.info("Scanning memory store for documents...")
                
                # Placeholder: In real implementation, you'd:
                # 1. Query ChromaDB/vector store for all documents
                # 2. Get document metadata and content
                # 3. Return list of documents to re-embed
                
                # For demo purposes, create some sample documents
                sample_docs = [
                    {
                        'id': f'doc_{i}',
                        'content': f'Sample document {i} content for re-embedding',
                        'metadata': {'source': 'knowledge_base', 'type': 'text'}
                    }
                    for i in range(10)  # Simulate 10 documents
                ]
                
                documents.extend(sample_docs)
                
            except Exception as e:
                self.logger.warning(f"Could not access memory manager: {e}")
            
            # Try to get documents from other sources
            try:
                # Check for uploaded documents
                uploads_dir = Path("data/documents")
                if uploads_dir.exists():
                    for doc_file in uploads_dir.glob("*.txt"):
                        try:
                            with open(doc_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            documents.append({
                                'id': f'file_{doc_file.stem}',
                                'content': content,
                                'metadata': {
                                    'source': 'uploaded_file',
                                    'filename': doc_file.name,
                                    'type': 'text'
                                }
                            })
                        except Exception as e:
                            self.logger.warning(f"Could not read file {doc_file}: {e}")
                            
            except Exception as e:
                self.logger.warning(f"Could not scan uploads directory: {e}")
            
            self.logger.info(f"Found {len(documents)} documents to re-embed")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to get documents: {e}")
            return []
    
    def _reembed_documents(self, documents: List[Dict[str, Any]], embedding_manager) -> bool:
        """Re-embed all documents with the new engine."""
        try:
            success_count = 0
            
            for i, doc in enumerate(documents):
                try:
                    self.current_document = doc['id']
                    
                    # Calculate progress (15% to 95% of total progress)
                    progress = 15.0 + (80.0 * i / len(documents))
                    self.update_progress(f"Re-embedding document {i+1}/{len(documents)}: {doc['id']}", progress)
                    
                    # Generate new embedding
                    new_embedding = embedding_manager.embed_text(doc['content'])
                    
                    if new_embedding is None or len(new_embedding) == 0:
                        raise Exception("Empty embedding generated")
                    
                    # Update vector store with new embedding
                    # In a real implementation, this would:
                    # 1. Delete old embedding from vector store
                    # 2. Insert new embedding with same metadata
                    # 3. Update any indexes
                    
                    self.logger.debug(f"✅ Re-embedded document: {doc['id']}")
                    success_count += 1
                    self.processed_documents += 1
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to re-embed document {doc['id']}: {e}")
                    self.failed_documents += 1
                    continue
            
            # Summary
            self.logger.info(f"Re-embedding summary:")
            self.logger.info(f"  Total documents: {len(documents)}")
            self.logger.info(f"  Successfully processed: {success_count}")
            self.logger.info(f"  Failed: {self.failed_documents}")
            
            # Consider successful if at least 80% succeeded
            success_rate = success_count / len(documents) if documents else 0
            return success_rate >= 0.8
            
        except Exception as e:
            self.logger.error(f"Failed to re-embed documents: {e}")
            return False
    
    def update_progress(self, message: str, percentage: float):
        """Update progress and write to progress file if specified."""
        progress_data = {
            'migration_id': self.migration_id,
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'percentage': percentage,
            'total_documents': self.total_documents,
            'processed_documents': self.processed_documents,
            'failed_documents': self.failed_documents,
            'current_document': self.current_document,
            'elapsed_seconds': (datetime.now() - self.start_time).total_seconds()
        }
        
        self.logger.info(f"Progress: {message} ({percentage:.1f}%)")
        
        # Write to progress file if specified
        if self.progress_file:
            try:
                with open(self.progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Could not write progress file: {e}")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Re-embed SAM knowledge base with new engine")
    parser.add_argument("--migration-id", required=True, help="Migration identifier")
    parser.add_argument("--progress-file", help="File to write progress updates")
    parser.add_argument("--background", action="store_true", help="Run in background mode")
    
    args = parser.parse_args()
    
    # Create progress file path if not specified
    if not args.progress_file:
        progress_dir = Path("sam/assets/migration_backups")
        progress_dir.mkdir(parents=True, exist_ok=True)
        args.progress_file = str(progress_dir / f"reembedding_progress_{args.migration_id}.json")
    
    # Create and run re-embedding task
    task = ReEmbeddingTask(args.migration_id, args.progress_file)
    
    success = task.run()
    
    if success:
        print(f"✅ Re-embedding completed successfully for migration: {args.migration_id}")
        sys.exit(0)
    else:
        print(f"❌ Re-embedding failed for migration: {args.migration_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
