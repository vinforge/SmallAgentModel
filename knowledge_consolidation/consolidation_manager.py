#!/usr/bin/env python3
"""
Consolidation Manager - Main orchestrator for knowledge consolidation
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .content_processor import ContentProcessor
from .knowledge_integrator import KnowledgeIntegrator

logger = logging.getLogger(__name__)

class ConsolidationManager:
    """Main manager for the knowledge consolidation process."""
    
    def __init__(self):
        self.content_processor = ContentProcessor()
        self.knowledge_integrator = KnowledgeIntegrator()
        self.consolidation_log = []
        
    def consolidate_approved_content(self, approved_dir: str = "approved") -> Dict[str, Any]:
        """Main consolidation process for approved content."""
        try:
            logger.info("Starting knowledge consolidation process")
            
            # Step 1: Process approved content
            processing_result = self.content_processor.batch_process_approved_content(approved_dir)
            
            if not processing_result['success']:
                return {
                    'success': False,
                    'error': f"Content processing failed: {processing_result.get('error')}",
                    'stage': 'processing'
                }
            
            if processing_result['total_items'] == 0:
                return {
                    'success': True,
                    'message': 'No approved content found to consolidate',
                    'total_items': 0,
                    'stage': 'processing'
                }
            
            # Step 2: Filter and validate knowledge items
            knowledge_items = self.content_processor.filter_knowledge_items(
                processing_result['knowledge_items']
            )
            
            if not knowledge_items:
                return {
                    'success': True,
                    'message': 'No valid knowledge items after filtering',
                    'total_items': 0,
                    'stage': 'filtering'
                }
            
            # Step 3: Integrate into knowledge base
            integration_result = self.knowledge_integrator.integrate_knowledge_items(knowledge_items)
            
            if not integration_result['success']:
                return {
                    'success': False,
                    'error': f"Knowledge integration failed: {integration_result.get('error')}",
                    'stage': 'integration'
                }
            
            # Step 4: Verify integration
            sample_queries = self._generate_sample_queries(knowledge_items)
            verification_result = self.knowledge_integrator.verify_integration(sample_queries)
            
            # Step 5: Log consolidation
            consolidation_record = self._create_consolidation_record(
                processing_result, integration_result, verification_result
            )
            self.consolidation_log.append(consolidation_record)
            
            # Step 6: Move processed files to archive
            archive_result = self._archive_processed_files(approved_dir)
            
            return {
                'success': True,
                'processing_result': processing_result,
                'integration_result': integration_result,
                'verification_result': verification_result,
                'archive_result': archive_result,
                'consolidation_record': consolidation_record,
                'total_items_processed': processing_result['total_items'],
                'total_items_integrated': integration_result['integrated_count'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Knowledge consolidation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': 'consolidation_manager'
            }
    
    def _generate_sample_queries(self, knowledge_items: List[Dict[str, Any]]) -> List[str]:
        """Generate sample queries for verification based on integrated content."""
        queries = []
        
        # Extract key terms from titles and content
        for item in knowledge_items[:5]:  # Sample from first 5 items
            title = item.get('title', '')
            content = item.get('content', '')
            
            # Extract key words from title
            if title:
                title_words = [word.strip('.,!?') for word in title.split() if len(word) > 4]
                if title_words:
                    queries.append(title_words[0])
            
            # Extract key phrases from content
            if content:
                content_words = [word.strip('.,!?') for word in content.split() if len(word) > 5]
                if content_words:
                    queries.append(content_words[0])
        
        # Add some generic queries
        queries.extend(['health news', 'latest news', 'current events'])
        
        return list(set(queries))[:10]  # Return unique queries, max 10
    
    def _create_consolidation_record(self, processing_result: Dict[str, Any], 
                                   integration_result: Dict[str, Any],
                                   verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a record of the consolidation process."""
        return {
            'consolidation_id': f"consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'processing': {
                'files_processed': processing_result.get('processed_files', 0),
                'files_failed': processing_result.get('failed_files', 0),
                'total_items': processing_result.get('total_items', 0)
            },
            'integration': {
                'items_integrated': integration_result.get('integrated_count', 0),
                'items_failed': integration_result.get('failed_count', 0)
            },
            'verification': {
                'queries_tested': verification_result.get('total_queries', 0),
                'queries_successful': verification_result.get('successful_queries', 0),
                'success_rate': verification_result.get('success_rate', 0)
            },
            'status': 'completed' if integration_result.get('integrated_count', 0) > 0 else 'no_content'
        }
    
    def _archive_processed_files(self, approved_dir: str) -> Dict[str, Any]:
        """Move processed files to archive directory."""
        try:
            approved_path = Path(approved_dir)
            archive_path = approved_path.parent / "archive" / datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if not approved_path.exists():
                return {'success': True, 'message': 'No approved directory to archive'}
            
            approved_files = list(approved_path.glob('*.json'))
            if not approved_files:
                return {'success': True, 'message': 'No files to archive'}
            
            # Create archive directory
            archive_path.mkdir(parents=True, exist_ok=True)
            
            # Move files
            moved_count = 0
            for file_path in approved_files:
                try:
                    new_path = archive_path / file_path.name
                    file_path.rename(new_path)
                    moved_count += 1
                except Exception as e:
                    logger.error(f"Failed to move {file_path}: {e}")
            
            return {
                'success': True,
                'moved_count': moved_count,
                'archive_path': str(archive_path)
            }
            
        except Exception as e:
            logger.error(f"Archive process failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_consolidation_history(self) -> List[Dict[str, Any]]:
        """Get history of consolidation processes."""
        return self.consolidation_log.copy()
    
    def get_consolidation_statistics(self) -> Dict[str, Any]:
        """Get statistics about consolidation processes."""
        if not self.consolidation_log:
            return {
                'total_consolidations': 0,
                'total_items_processed': 0,
                'total_items_integrated': 0,
                'average_success_rate': 0
            }
        
        total_consolidations = len(self.consolidation_log)
        total_items_processed = sum(record['processing']['total_items'] for record in self.consolidation_log)
        total_items_integrated = sum(record['integration']['items_integrated'] for record in self.consolidation_log)
        
        success_rates = [record['verification']['success_rate'] for record in self.consolidation_log 
                        if record['verification']['success_rate'] is not None]
        average_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        return {
            'total_consolidations': total_consolidations,
            'total_items_processed': total_items_processed,
            'total_items_integrated': total_items_integrated,
            'average_success_rate': average_success_rate,
            'last_consolidation': self.consolidation_log[-1]['timestamp'] if self.consolidation_log else None
        }
    
    def manual_consolidation_trigger(self) -> Dict[str, Any]:
        """Manually trigger consolidation process."""
        logger.info("Manual consolidation triggered")
        return self.consolidate_approved_content()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status for consolidation."""
        try:
            integration_status = self.knowledge_integrator.get_integration_status()
            
            return {
                'consolidation_manager': 'active',
                'content_processor': 'active',
                'knowledge_integrator': integration_status,
                'consolidation_history_count': len(self.consolidation_log),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
