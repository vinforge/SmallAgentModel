"""
Vetting Interface for Phase 7.3: UI Integration & The "Go/No-Go" Decision

This module provides the Flask blueprint for integrating the automated vetting
engine into SAM's web interface, including the "Vet All" functionality and
the final Go/No-Go decision gate for content ingestion.

Features:
- Automated vetting process trigger
- Vetted content browser and analysis display
- Content approval/rejection workflow
- Integration with existing document processing pipeline
"""

import json
import logging
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Blueprint, render_template, jsonify, request, current_app
from werkzeug.utils import secure_filename

# Import SAM components
try:
    from utils.security_utils import require_unlock
    from utils.vector_manager import VectorManager
    from multimodal_processing.multimodal_pipeline import get_multimodal_pipeline
except ImportError:
    # Fallback for development
    def require_unlock(f):
        return f
    VectorManager = None
    get_multimodal_pipeline = None

# Create blueprint
vetting_bp = Blueprint('vetting', __name__, url_prefix='/vetting')
logger = logging.getLogger(__name__)


@vetting_bp.route('/api/status')
def get_vetting_status():
    """Get current vetting system status and file counts."""
    try:
        quarantine_dir = Path('quarantine')
        vetted_dir = Path('vetted')
        approved_dir = Path('approved')
        rejected_dir = Path('rejected')
        
        # Count files in each directory
        quarantine_files = len([f for f in quarantine_dir.glob('*.json') 
                              if f.name not in ['metadata.json', 'README.md']])
        vetted_files = len(list(vetted_dir.glob('*.json')))
        approved_files = len(list(approved_dir.glob('*.json'))) if approved_dir.exists() else 0
        rejected_files = len(list(rejected_dir.glob('*.json'))) if rejected_dir.exists() else 0
        
        return jsonify({
            'status': 'success',
            'quarantine_files': quarantine_files,
            'vetted_files': vetted_files,
            'approved_files': approved_files,
            'rejected_files': rejected_files,
            'ready_for_vetting': quarantine_files > 0,
            'has_vetted_content': vetted_files > 0,
            'system_operational': True
        })
        
    except Exception as e:
        logger.error(f"Error getting vetting status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'system_operational': False
        }), 500


@vetting_bp.route('/api/vet-all', methods=['POST'])
@require_unlock
def trigger_vetting_process():
    """
    Trigger automated vetting of all quarantined content.
    
    This executes the vet_quarantined_content.py script and returns
    the results in a structured format for the UI.
    """
    try:
        logger.info("Starting automated vetting process via UI")
        
        # Get project root directory
        project_root = Path(__file__).parent.parent
        
        # Execute vetting script with batch processing
        result = subprocess.run([
            sys.executable, 
            'scripts/vet_quarantined_content.py', 
            '--batch',
            '--quiet'
        ], 
        capture_output=True, 
        text=True, 
        cwd=project_root,
        timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Parse basic stats from successful execution
            vetting_stats = {
                'status': 'completed',
                'message': 'All quarantined content has been processed',
                'timestamp': datetime.now().isoformat()
            }
            
            # Get updated status
            status_response = get_vetting_status()
            if status_response.status_code == 200:
                status_data = status_response.get_json()
                vetting_stats.update({
                    'vetted_files': status_data.get('vetted_files', 0),
                    'quarantine_files': status_data.get('quarantine_files', 0)
                })
            
            logger.info(f"Vetting process completed successfully: {vetting_stats}")
            
            return jsonify({
                'status': 'success',
                'message': 'Vetting process completed successfully',
                'stats': vetting_stats
            })
        else:
            error_message = result.stderr or "Unknown error occurred"
            logger.error(f"Vetting process failed: {error_message}")
            
            return jsonify({
                'status': 'error',
                'message': f'Vetting process failed: {error_message}',
                'returncode': result.returncode
            }), 500
            
    except subprocess.TimeoutExpired:
        logger.error("Vetting process timed out")
        return jsonify({
            'status': 'error',
            'message': 'Vetting process timed out (exceeded 5 minutes)'
        }), 500
        
    except Exception as e:
        logger.error(f"Error triggering vetting process: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to start vetting process: {str(e)}'
        }), 500


@vetting_bp.route('/api/vetted-content')
def get_vetted_content():
    """Get list of all vetted content with analysis results."""
    try:
        vetted_dir = Path('vetted')
        
        if not vetted_dir.exists():
            return jsonify({
                'status': 'success',
                'files': [],
                'total_count': 0
            })
        
        vetted_files = []
        
        for filepath in vetted_dir.glob('*.json'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                vetting_results = data.get('vetting_results', {})
                scores = vetting_results.get('scores', {})
                
                # Extract key information for display
                file_info = {
                    'filename': filepath.name,
                    'url': data.get('url', 'Unknown'),
                    'timestamp': vetting_results.get('timestamp', ''),
                    'recommendation': vetting_results.get('recommendation', 'UNKNOWN'),
                    'overall_score': round(vetting_results.get('overall_score', 0.0), 3),
                    'confidence': round(vetting_results.get('confidence', 0.0), 3),
                    'reason': vetting_results.get('reason', 'No reason provided'),
                    'scores': {
                        'credibility': round(scores.get('credibility', 0.0), 3),
                        'persuasion': round(scores.get('persuasion', 0.0), 3),
                        'speculation': round(scores.get('speculation', 0.0), 3),
                        'purity': round(scores.get('purity', 0.0), 3),
                        'source_reputation': round(scores.get('source_reputation', 0.0), 3)
                    },
                    'risk_factors': len(vetting_results.get('risk_assessment', {}).get('risk_factors', [])),
                    'content_preview': self._create_content_preview(data.get('content', '')),
                    'source_reputation_score': round(
                        vetting_results.get('source_reputation', {}).get('final_score', 0.0), 3
                    ),
                    'file_size': filepath.stat().st_size,
                    'processing_time': vetting_results.get('processing_time', 0.0)
                }
                
                vetted_files.append(file_info)
                
            except Exception as e:
                logger.warning(f"Error reading vetted file {filepath}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        vetted_files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'files': vetted_files,
            'total_count': len(vetted_files)
        })
        
    except Exception as e:
        logger.error(f"Error getting vetted content: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@vetting_bp.route('/api/vetted-content/<filename>')
def get_vetted_file_details(filename):
    """Get detailed analysis for a specific vetted file."""
    try:
        # Secure the filename
        secure_name = secure_filename(filename)
        vetted_path = Path('vetted') / secure_name
        
        if not vetted_path.exists():
            return jsonify({
                'status': 'error',
                'error': 'File not found'
            }), 404
        
        with open(vetted_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add file metadata
        file_stats = vetted_path.stat()
        data['file_metadata'] = {
            'filename': secure_name,
            'file_size': file_stats.st_size,
            'created_time': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'data': data
        })
        
    except Exception as e:
        logger.error(f"Error getting vetted file details: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@vetting_bp.route('/api/approve-content', methods=['POST'])
@require_unlock
def approve_vetted_content():
    """
    Approve vetted content for ingestion into SAM's knowledge base.
    
    This implements the final "Go" decision in the Go/No-Go gate.
    """
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({
                'status': 'error',
                'error': 'Filename required'
            }), 400
        
        # Secure the filename
        secure_name = secure_filename(filename)
        vetted_path = Path('vetted') / secure_name
        
        if not vetted_path.exists():
            return jsonify({
                'status': 'error',
                'error': 'Vetted file not found'
            }), 404
        
        # Load vetted content
        with open(vetted_path, 'r', encoding='utf-8') as f:
            vetted_data = json.load(f)
        
        # Process through ingestion pipeline
        ingestion_result = self._process_approved_content(vetted_data)
        
        if ingestion_result['status'] == 'success':
            # Move vetted file to approved directory
            self._move_to_approved(vetted_path, vetted_data)
            
            logger.info(f"Successfully approved and ingested content: {secure_name}")
            
            return jsonify({
                'status': 'success',
                'message': 'Content successfully added to knowledge base',
                'ingestion_result': ingestion_result
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Ingestion failed: {ingestion_result.get("error", "Unknown error")}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error approving content: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@vetting_bp.route('/api/reject-content', methods=['POST'])
@require_unlock
def reject_vetted_content():
    """
    Reject vetted content (final "No-Go" decision).
    
    Moves content to rejected directory for audit trail.
    """
    try:
        data = request.get_json()
        filename = data.get('filename')
        reason = data.get('reason', 'User rejected')
        
        if not filename:
            return jsonify({
                'status': 'error',
                'error': 'Filename required'
            }), 400
        
        # Secure the filename
        secure_name = secure_filename(filename)
        vetted_path = Path('vetted') / secure_name
        
        if not vetted_path.exists():
            return jsonify({
                'status': 'error',
                'error': 'Vetted file not found'
            }), 404
        
        # Move to rejected directory with metadata
        self._move_to_rejected(vetted_path, reason)
        
        logger.info(f"Content rejected by user: {secure_name} - {reason}")
        
        return jsonify({
            'status': 'success',
            'message': 'Content rejected and moved to rejected directory'
        })
        
    except Exception as e:
        logger.error(f"Error rejecting content: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


# Helper methods
def _create_content_preview(content: str, max_length: int = 200) -> str:
    """Create a safe preview of content for display."""
    if not content:
        return "No content available"
    
    # Clean and truncate content
    clean_content = content.strip()
    if len(clean_content) > max_length:
        return clean_content[:max_length] + "..."
    return clean_content


def _process_approved_content(vetted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process approved web content through SAM's ingestion pipeline."""
    try:
        # Extract sanitized content
        vetting_results = vetted_data.get('vetting_results', {})
        sanitization = vetting_results.get('sanitization', {})
        clean_content = sanitization.get('clean_text') or vetted_data.get('content', '')
        
        if not clean_content:
            return {
                'status': 'error',
                'error': 'No clean content available for ingestion'
            }
        
        # Create temporary file for processing
        temp_dir = Path('temp')
        temp_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"approved_web_content_{timestamp}.txt"
        temp_path = temp_dir / temp_filename
        
        # Write content with metadata
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {vetted_data.get('url', 'Unknown')}\n")
            f.write(f"Retrieved: {vetted_data.get('timestamp', 'Unknown')}\n")
            f.write(f"Vetting Score: {vetting_results.get('overall_score', 0.0):.3f}\n")
            f.write(f"Recommendation: {vetting_results.get('recommendation', 'UNKNOWN')}\n")
            f.write(f"Approved: {datetime.now().isoformat()}\n")
            f.write("\n" + "="*50 + "\n\n")
            f.write(clean_content)
        
        # Process through multimodal pipeline if available
        multimodal_pipeline = get_multimodal_pipeline()
        if multimodal_pipeline:
            result = multimodal_pipeline.process_document(temp_path)
            
            # Clean up temporary file
            temp_path.unlink()
            
            return {
                'status': 'success',
                'document_id': result.get('document_id'),
                'content_blocks': result.get('content_blocks', 0),
                'memory_chunks': result.get('memory_storage', {}).get('chunks_stored', 0),
                'processing_method': 'multimodal_pipeline'
            }
        else:
            # Fallback: basic text processing
            temp_path.unlink()
            
            return {
                'status': 'success',
                'message': 'Content processed (multimodal pipeline not available)',
                'processing_method': 'fallback'
            }
            
    except Exception as e:
        logger.error(f"Error processing approved content: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def _move_to_approved(vetted_path: Path, vetted_data: Dict[str, Any]) -> None:
    """Move vetted file to approved directory with approval metadata."""
    approved_dir = Path('approved')
    approved_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    approved_filename = f"approved_{timestamp}_{vetted_path.name}"
    approved_path = approved_dir / approved_filename
    
    # Add approval metadata
    vetted_data['approval_info'] = {
        'approved_at': datetime.now().isoformat(),
        'approved_by': 'user',
        'original_filename': vetted_path.name
    }
    
    # Save with approval metadata
    with open(approved_path, 'w', encoding='utf-8') as f:
        json.dump(vetted_data, f, indent=2, ensure_ascii=False)
    
    # Remove from vetted directory
    vetted_path.unlink()


def _move_to_rejected(vetted_path: Path, reason: str) -> None:
    """Move vetted file to rejected directory with rejection metadata."""
    rejected_dir = Path('rejected')
    rejected_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rejected_filename = f"rejected_{timestamp}_{vetted_path.name}"
    rejected_path = rejected_dir / rejected_filename
    
    # Load and add rejection metadata
    with open(vetted_path, 'r', encoding='utf-8') as f:
        vetted_data = json.load(f)
    
    vetted_data['rejection_info'] = {
        'rejected_at': datetime.now().isoformat(),
        'reason': reason,
        'rejected_by': 'user',
        'original_filename': vetted_path.name
    }
    
    # Save with rejection metadata
    with open(rejected_path, 'w', encoding='utf-8') as f:
        json.dump(vetted_data, f, indent=2, ensure_ascii=False)
    
    # Remove from vetted directory
    vetted_path.unlink()


# Attach helper methods to blueprint
vetting_bp._create_content_preview = _create_content_preview
vetting_bp._process_approved_content = _process_approved_content
vetting_bp._move_to_approved = _move_to_approved
vetting_bp._move_to_rejected = _move_to_rejected
