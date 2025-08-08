"""
Vetting Pipeline for Phase 7.2: Automated Vetting Engine

This module provides the complete automated vetting pipeline that processes
quarantined web content through comprehensive security and quality analysis.

Features:
- Batch processing of quarantined files
- Comprehensive vetting workflow
- Result enrichment and storage
- Archive management
- Performance monitoring
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
import time

from .content_evaluator import ContentEvaluator
from .exceptions import WebRetrievalError, ValidationError


class VettingPipeline:
    """
    Complete automated vetting pipeline for quarantined web content.
    
    This class manages the entire workflow from quarantined files through
    analysis to vetted results, including file management and archiving.
    """
    
    def __init__(self, 
                 content_evaluator: ContentEvaluator,
                 quarantine_dir: str = "quarantine",
                 vetted_dir: str = "vetted",
                 archive_dir: str = "archive"):
        """
        Initialize the vetting pipeline.
        
        Args:
            content_evaluator: ContentEvaluator instance for analysis
            quarantine_dir: Directory containing quarantined files
            vetted_dir: Directory for vetted results
            archive_dir: Directory for archived processed files
        """
        self.evaluator = content_evaluator
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.quarantine_dir = Path(quarantine_dir)
        self.vetted_dir = Path(vetted_dir)
        self.archive_dir = Path(archive_dir)
        
        # Create directories if they don't exist
        self._setup_directories()
        
        # Processing statistics
        self.stats = {
            'files_processed': 0,
            'files_passed': 0,
            'files_failed': 0,
            'files_review': 0,
            'total_processing_time': 0.0,
            'errors': 0
        }
        
        self.logger.info(f"VettingPipeline initialized with evaluator: {content_evaluator.__class__.__name__}")
    
    def process_quarantined_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Process a single quarantined file through the complete vetting pipeline.
        
        Args:
            filepath: Path to quarantined JSON file
            
        Returns:
            Processing result with status and file locations
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing quarantined file: {filepath.name}")
            
            # Step 1: Validate file
            if not self._validate_quarantined_file(filepath):
                raise ValidationError(f"Invalid quarantined file: {filepath}")
            
            # Step 2: Run content evaluation
            enriched_data = self.evaluator.evaluate_quarantined_file(filepath)
            
            # Step 3: Save vetted result
            vetted_path = self._save_vetted_result(enriched_data, filepath.name)
            
            # Step 4: Archive original file
            archive_path = self._archive_original_file(filepath)
            
            # Step 5: Update statistics
            processing_time = time.time() - start_time
            self._update_stats(enriched_data, processing_time)
            
            # Step 6: Generate result summary
            result = {
                'status': 'success',
                'original_file': str(filepath),
                'vetted_file': str(vetted_path),
                'archived_file': str(archive_path),
                'processing_time': processing_time,
                'recommendation': enriched_data['vetting_results']['recommendation'],
                'overall_score': enriched_data['vetting_results']['overall_score'],
                'confidence': enriched_data['vetting_results']['confidence']
            }
            
            self.logger.info(f"Successfully processed {filepath.name}: "
                           f"{result['recommendation']} "
                           f"(score: {result['overall_score']:.3f})")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['errors'] += 1
            
            self.logger.error(f"Error processing {filepath.name}: {e}")
            
            return {
                'status': 'error',
                'original_file': str(filepath),
                'error': str(e),
                'processing_time': processing_time
            }
    
    def process_all_quarantined_files(self, 
                                    file_pattern: str = "*.json",
                                    exclude_metadata: bool = True) -> Dict[str, Any]:
        """
        Process all quarantined files in batch mode.
        
        Args:
            file_pattern: File pattern to match (default: *.json)
            exclude_metadata: Whether to exclude metadata files
            
        Returns:
            Batch processing results and statistics
        """
        start_time = time.time()
        
        # Find all matching files
        all_files = list(self.quarantine_dir.glob(file_pattern))
        
        # Filter out metadata files if requested
        if exclude_metadata:
            files_to_process = [
                f for f in all_files 
                if f.name not in ['metadata.json', 'README.md']
            ]
        else:
            files_to_process = all_files
        
        if not files_to_process:
            self.logger.info("No quarantined files found to process")
            return {
                'status': 'no_files',
                'files_found': 0,
                'files_processed': 0,
                'results': []
            }
        
        self.logger.info(f"Starting batch processing of {len(files_to_process)} files")
        
        # Process each file
        results = []
        for filepath in files_to_process:
            result = self.process_quarantined_file(filepath)
            results.append(result)
        
        # Calculate batch statistics
        total_time = time.time() - start_time
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        # Categorize successful results by recommendation
        passed = [r for r in successful if r.get('recommendation') == 'PASS']
        review = [r for r in successful if r.get('recommendation') == 'REVIEW']
        rejected = [r for r in successful if r.get('recommendation') == 'FAIL']
        
        batch_result = {
            'status': 'completed',
            'total_files': len(files_to_process),
            'successful': len(successful),
            'failed': len(failed),
            'recommendations': {
                'PASS': len(passed),
                'REVIEW': len(review),
                'FAIL': len(rejected)
            },
            'total_processing_time': total_time,
            'average_processing_time': total_time / len(files_to_process) if files_to_process else 0,
            'results': results,
            'statistics': self.get_pipeline_stats()
        }
        
        self.logger.info(f"Batch processing complete: {len(successful)} successful, "
                        f"{len(failed)} failed, {total_time:.2f}s total")
        
        return batch_result
    
    def _setup_directories(self):
        """Create required directories if they don't exist."""
        for directory in [self.quarantine_dir, self.vetted_dir, self.archive_dir]:
            directory.mkdir(exist_ok=True)
            self.logger.debug(f"Directory ready: {directory}")
    
    def _validate_quarantined_file(self, filepath: Path) -> bool:
        """Validate that a quarantined file is ready for processing."""
        if not filepath.exists():
            return False
        
        if not filepath.suffix == '.json':
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check for required fields
            required_fields = ['url', 'content', 'timestamp']
            return all(field in data for field in required_fields)
            
        except (json.JSONDecodeError, Exception):
            return False
    
    def _save_vetted_result(self, enriched_data: Dict[str, Any], original_filename: str) -> Path:
        """Save vetted result to vetted directory."""
        
        # Generate vetted filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = original_filename.replace('.json', '')
        vetted_filename = f"vetted_{timestamp}_{base_name}.json"
        
        vetted_path = self.vetted_dir / vetted_filename
        
        # Save enriched data
        with open(vetted_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Vetted result saved: {vetted_path}")
        return vetted_path
    
    def _archive_original_file(self, filepath: Path) -> Path:
        """Move original file to archive directory."""
        
        # Generate archive filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = filepath.stem
        archive_filename = f"archived_{timestamp}_{base_name}.json"
        
        archive_path = self.archive_dir / archive_filename
        
        # Move file to archive
        shutil.move(str(filepath), str(archive_path))
        
        self.logger.debug(f"Original file archived: {archive_path}")
        return archive_path
    
    def _update_stats(self, enriched_data: Dict[str, Any], processing_time: float):
        """Update processing statistics."""
        self.stats['files_processed'] += 1
        self.stats['total_processing_time'] += processing_time
        
        recommendation = enriched_data['vetting_results']['recommendation']
        if recommendation == 'PASS':
            self.stats['files_passed'] += 1
        elif recommendation == 'REVIEW':
            self.stats['files_review'] += 1
        elif recommendation == 'FAIL':
            self.stats['files_failed'] += 1
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        total_files = self.stats['files_processed']
        
        return {
            'files_processed': total_files,
            'files_passed': self.stats['files_passed'],
            'files_review': self.stats['files_review'],
            'files_failed': self.stats['files_failed'],
            'errors': self.stats['errors'],
            'total_processing_time': self.stats['total_processing_time'],
            'average_processing_time': (
                self.stats['total_processing_time'] / total_files 
                if total_files > 0 else 0.0
            ),
            'pass_rate': (
                self.stats['files_passed'] / total_files 
                if total_files > 0 else 0.0
            ),
            'directories': {
                'quarantine': str(self.quarantine_dir),
                'vetted': str(self.vetted_dir),
                'archive': str(self.archive_dir)
            },
            'evaluator_config': self.evaluator.get_evaluator_stats()
        }
    
    def cleanup_old_archives(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up old archived files."""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        archived_files = list(self.archive_dir.glob("*.json"))
        old_files = []
        
        for filepath in archived_files:
            if filepath.stat().st_mtime < cutoff_time:
                old_files.append(filepath)
        
        # Remove old files
        removed_count = 0
        for filepath in old_files:
            try:
                filepath.unlink()
                removed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to remove old archive {filepath}: {e}")
        
        self.logger.info(f"Cleaned up {removed_count} old archive files (older than {days_old} days)")
        
        return {
            'files_removed': removed_count,
            'files_checked': len(archived_files),
            'cutoff_days': days_old
        }
    
    def get_quarantine_summary(self) -> Dict[str, Any]:
        """Get summary of current quarantine directory contents."""
        json_files = list(self.quarantine_dir.glob("*.json"))
        
        # Exclude metadata files
        content_files = [f for f in json_files if f.name not in ['metadata.json']]
        
        return {
            'total_files': len(json_files),
            'content_files': len(content_files),
            'metadata_files': len(json_files) - len(content_files),
            'ready_for_processing': len(content_files),
            'quarantine_directory': str(self.quarantine_dir)
        }
