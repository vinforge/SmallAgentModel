#!/usr/bin/env python3
"""
SAM Bulk Document Ingestion Tool - Phase 1
Command-line tool for bulk importing documents into SAM's knowledge base.

Usage:
    python scripts/bulk_ingest.py --source /path/to/documents
    python scripts/bulk_ingest.py --source /path/to/documents --dry-run
    python scripts/bulk_ingest.py --source /path/to/documents --file-types pdf,txt,md
"""

import os
import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
import sqlite3

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/bulk_ingest.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class BulkIngestionState:
    """Manages state tracking for bulk ingestion to avoid reprocessing files."""
    
    def __init__(self, state_file: str = "data/ingestion_state.db"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for state tracking."""
        with sqlite3.connect(self.state_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    filepath TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    last_modified REAL NOT NULL,
                    processed_at TEXT NOT NULL,
                    chunks_created INTEGER DEFAULT 0,
                    enrichment_score REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'success'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON processed_files(file_hash)
            """)
            conn.commit()
    
    def get_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file content."""
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return ""
    
    def is_file_processed(self, filepath: Path) -> bool:
        """Check if file has been processed and hasn't changed."""
        try:
            current_hash = self.get_file_hash(filepath)
            current_mtime = filepath.stat().st_mtime
            
            with sqlite3.connect(self.state_file) as conn:
                cursor = conn.execute(
                    "SELECT file_hash, last_modified FROM processed_files WHERE filepath = ?",
                    (str(filepath),)
                )
                result = cursor.fetchone()
                
                if result:
                    stored_hash, stored_mtime = result
                    return stored_hash == current_hash and abs(stored_mtime - current_mtime) < 1.0
                
                return False
        except Exception as e:
            logger.error(f"Error checking file state for {filepath}: {e}")
            return False
    
    def mark_file_processed(self, filepath: Path, chunks_created: int = 0, 
                          enrichment_score: float = 0.0, status: str = 'success'):
        """Mark file as processed in the state database."""
        try:
            file_hash = self.get_file_hash(filepath)
            file_size = filepath.stat().st_size
            last_modified = filepath.stat().st_mtime
            processed_at = datetime.now().isoformat()
            
            with sqlite3.connect(self.state_file) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO processed_files 
                    (filepath, file_hash, file_size, last_modified, processed_at, 
                     chunks_created, enrichment_score, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (str(filepath), file_hash, file_size, last_modified, 
                      processed_at, chunks_created, enrichment_score, status))
                conn.commit()
        except Exception as e:
            logger.error(f"Error marking file as processed {filepath}: {e}")
    
    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        try:
            with sqlite3.connect(self.state_file) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_files,
                        SUM(chunks_created) as total_chunks,
                        AVG(enrichment_score) as avg_enrichment,
                        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                    FROM processed_files
                """)
                result = cursor.fetchone()
                
                return {
                    'total_files': result[0] or 0,
                    'total_chunks': result[1] or 0,
                    'avg_enrichment': result[2] or 0.0,
                    'successful': result[3] or 0,
                    'failed': result[4] or 0
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

class BulkDocumentIngestor:
    """Core bulk document ingestion engine."""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.txt', '.md', '.docx', '.doc', '.rtf',
        '.py', '.js', '.html', '.htm', '.css', '.json',
        '.xml', '.yaml', '.yml', '.csv', '.tsv'
    }
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.state = BulkIngestionState()
        self.processed_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        
        # Initialize SAM components
        self._init_sam_components()
    
    def _init_sam_components(self):
        """Initialize SAM's document processing components."""
        try:
            from multimodal_processing.multimodal_pipeline import MultimodalProcessingPipeline
            from memory.memory_vectorstore import get_memory_store

            self.pipeline = MultimodalProcessingPipeline()
            self.memory_store = get_memory_store()

            logger.info("✅ SAM components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SAM components: {e}")
            if not self.dry_run:
                raise
    
    def scan_folder(self, source_path: Path, file_types: Optional[Set[str]] = None) -> List[Path]:
        """Scan folder for supported document files."""
        if file_types:
            extensions = {f'.{ext.lower().lstrip(".")}' for ext in file_types}
        else:
            extensions = self.SUPPORTED_EXTENSIONS
        
        files = []
        try:
            for file_path in source_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    files.append(file_path)
            
            logger.info(f"📁 Found {len(files)} supported files in {source_path}")
            return sorted(files)
        
        except Exception as e:
            logger.error(f"❌ Error scanning folder {source_path}: {e}")
            return []
    
    def process_file(self, filepath: Path) -> Tuple[bool, Dict]:
        """Process a single file through SAM's ingestion pipeline."""
        try:
            logger.info(f"📄 Processing: {filepath.name}")
            
            if self.dry_run:
                # Simulate processing for dry run
                return True, {
                    'chunks_created': 5,
                    'enrichment_score': 0.75,
                    'file_size': filepath.stat().st_size
                }
            
            # Process through SAM's pipeline
            result = self.pipeline.process_document(str(filepath))

            if result:
                # Extract information from the processing result
                chunks_created = result.get('content_blocks', 0)
                enrichment_score = result.get('enrichment_score', 0.0)
                memory_storage = result.get('memory_storage', {})
                total_chunks_stored = memory_storage.get('total_chunks_stored', 0)

                logger.info(f"✅ Successfully processed {filepath.name}: "
                          f"{chunks_created} content blocks, {total_chunks_stored} memory chunks, "
                          f"score: {enrichment_score:.2f}")

                return True, {
                    'chunks_created': total_chunks_stored,
                    'enrichment_score': enrichment_score,
                    'file_size': filepath.stat().st_size,
                    'content_blocks': chunks_created
                }
            else:
                logger.error(f"❌ Failed to process {filepath.name}")
                return False, {}
        
        except Exception as e:
            logger.error(f"❌ Error processing {filepath}: {e}")
            return False, {}
    
    def ingest_folder(self, source_path: Path, file_types: Optional[Set[str]] = None) -> Dict:
        """Ingest all supported files from a folder."""
        logger.info(f"🚀 Starting bulk ingestion from: {source_path}")
        
        if self.dry_run:
            logger.info("🔍 DRY RUN MODE - No files will be actually processed")
        
        # Scan for files
        files = self.scan_folder(source_path, file_types)
        
        if not files:
            logger.warning("⚠️ No supported files found")
            return self._get_summary()
        
        # Filter out already processed files
        new_files = []
        for filepath in files:
            if self.state.is_file_processed(filepath):
                logger.info(f"⏭️ Skipping already processed: {filepath.name}")
                self.skipped_count += 1
            else:
                new_files.append(filepath)
        
        logger.info(f"📊 Found {len(new_files)} new files to process "
                   f"({self.skipped_count} already processed)")
        
        # Process new files
        for i, filepath in enumerate(new_files, 1):
            logger.info(f"📈 Progress: {i}/{len(new_files)} - {filepath.name}")
            
            success, result = self.process_file(filepath)
            
            if success:
                self.processed_count += 1
                if not self.dry_run:
                    self.state.mark_file_processed(
                        filepath,
                        chunks_created=result.get('chunks_created', 0),
                        enrichment_score=result.get('enrichment_score', 0.0),
                        status='success'
                    )
            else:
                self.failed_count += 1
                if not self.dry_run:
                    self.state.mark_file_processed(filepath, status='failed')
        
        summary = self._get_summary()
        logger.info(f"🎉 Bulk ingestion complete: {summary}")
        
        return summary
    
    def _get_summary(self) -> Dict:
        """Get ingestion summary."""
        return {
            'processed': self.processed_count,
            'skipped': self.skipped_count,
            'failed': self.failed_count,
            'total_found': self.processed_count + self.skipped_count + self.failed_count
        }

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SAM Bulk Document Ingestion Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/bulk_ingest.py --source /path/to/documents
  python scripts/bulk_ingest.py --source /path/to/documents --dry-run
  python scripts/bulk_ingest.py --source /path/to/documents --file-types pdf,txt,md
  python scripts/bulk_ingest.py --stats
        """
    )
    
    parser.add_argument('--source', type=str, help='Source folder path to ingest')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Simulate ingestion without actually processing files')
    parser.add_argument('--file-types', type=str,
                       help='Comma-separated list of file extensions (e.g., pdf,txt,md)')
    parser.add_argument('--stats', action='store_true',
                       help='Show ingestion statistics and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Show stats and exit
    if args.stats:
        state = BulkIngestionState()
        stats = state.get_stats()
        print("\n📊 Bulk Ingestion Statistics:")
        print(f"   Total files processed: {stats.get('total_files', 0)}")
        print(f"   Total chunks created: {stats.get('total_chunks', 0)}")
        print(f"   Average enrichment score: {stats.get('avg_enrichment', 0.0):.2f}")
        print(f"   Successful: {stats.get('successful', 0)}")
        print(f"   Failed: {stats.get('failed', 0)}")
        return
    
    # Validate source path
    if not args.source:
        parser.error("--source is required (or use --stats to view statistics)")
    
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"❌ Source path does not exist: {source_path}")
        sys.exit(1)
    
    if not source_path.is_dir():
        logger.error(f"❌ Source path is not a directory: {source_path}")
        sys.exit(1)
    
    # Parse file types
    file_types = None
    if args.file_types:
        file_types = {ext.strip() for ext in args.file_types.split(',')}
        logger.info(f"🎯 Filtering for file types: {file_types}")
    
    # Run bulk ingestion
    try:
        ingestor = BulkDocumentIngestor(dry_run=args.dry_run)
        summary = ingestor.ingest_folder(source_path, file_types)
        
        print(f"\n🎉 Bulk Ingestion Summary:")
        print(f"   📄 Processed: {summary['processed']}")
        print(f"   ⏭️ Skipped: {summary['skipped']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   📊 Total found: {summary['total_found']}")
        
        if args.dry_run:
            print("\n🔍 This was a dry run - no files were actually processed")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Bulk ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Bulk ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
