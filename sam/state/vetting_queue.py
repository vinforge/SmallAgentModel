"""
Vetting Queue Manager - Research Paper Vetting System
====================================================

Manages the quarantine and vetting process for downloaded research papers
with automated analysis, scoring, and approval workflows.

Features:
- Quarantine file management
- Automated security and relevance scoring
- Manual review queue
- Auto-approval based on thresholds
- Audit trail and logging

Part of SAM's Task 27: Automated "Dream & Discover" Engine
Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class VettingStatus(Enum):
    """Status of files in the vetting queue."""
    PENDING_ANALYSIS = "pending_analysis"
    PENDING_APPROVAL = "pending_approval"
    AUTO_APPROVED = "auto_approved"
    MANUALLY_APPROVED = "manually_approved"
    REJECTED = "rejected"
    REQUIRES_MANUAL_REVIEW = "requires_manual_review"

@dataclass
class VettingScores:
    """Scoring results for vetting analysis."""
    security_risk_score: float  # 0.0 (safe) to 1.0 (high risk)
    relevance_score: float      # 0.0 (irrelevant) to 1.0 (highly relevant)
    credibility_score: float    # 0.0 (low credibility) to 1.0 (high credibility)
    overall_score: float        # Composite score
    
@dataclass
class VettingEntry:
    """Entry in the vetting queue."""
    file_id: str
    original_filename: str
    quarantine_path: str
    paper_metadata: Dict[str, Any]
    original_insight_text: str
    status: VettingStatus
    scores: Optional[VettingScores]
    created_at: str
    analyzed_at: Optional[str]
    approved_at: Optional[str]
    approved_by: Optional[str]  # 'auto' or user identifier
    rejection_reason: Optional[str]
    notes: List[str]

class VettingQueueManager:
    """
    Manages the vetting queue for downloaded research papers.
    
    Handles:
    - File quarantine management
    - Automated analysis and scoring
    - Manual review workflows
    - Auto-approval based on thresholds
    - Audit trail and reporting
    """
    
    def __init__(self, queue_file: Optional[str] = None, quarantine_dir: Optional[str] = None):
        """Initialize the vetting queue manager."""
        self.logger = logging.getLogger(__name__)
        
        # Queue file configuration
        if queue_file:
            self.queue_file = Path(queue_file)
        else:
            # Use SAM's standard state directory
            state_dir = Path.home() / ".sam"
            state_dir.mkdir(exist_ok=True)
            self.queue_file = state_dir / "vetting_queue.json"
        
        # Quarantine directory configuration
        if quarantine_dir:
            self.quarantine_dir = Path(quarantine_dir)
        else:
            self.quarantine_dir = Path("memory/quarantine")
        
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory queue
        self._queue: Dict[str, VettingEntry] = {}
        
        # Auto-approval thresholds (can be configured)
        self.auto_approval_thresholds = {
            'security_risk_max': 0.3,      # Max security risk for auto-approval
            'relevance_min': 0.7,          # Min relevance for auto-approval
            'credibility_min': 0.6,        # Min credibility for auto-approval
            'overall_min': 0.7             # Min overall score for auto-approval
        }
        
        # Initialize queue file if it doesn't exist
        self._initialize_queue_file()
        
        # Load existing queue
        self._load_queue()
        
        self.logger.info(f"VettingQueueManager initialized with {len(self._queue)} entries")
    
    def add_file_to_queue(self, quarantine_path: str, paper_metadata: Dict[str, Any], 
                         original_insight_text: str) -> str:
        """Add a file to the vetting queue."""
        with self._lock:
            file_id = str(uuid.uuid4())
            
            entry = VettingEntry(
                file_id=file_id,
                original_filename=paper_metadata.get('title', 'Unknown'),
                quarantine_path=quarantine_path,
                paper_metadata=paper_metadata,
                original_insight_text=original_insight_text,
                status=VettingStatus.PENDING_ANALYSIS,
                scores=None,
                created_at=datetime.now().isoformat(),
                analyzed_at=None,
                approved_at=None,
                approved_by=None,
                rejection_reason=None,
                notes=[]
            )
            
            self._queue[file_id] = entry
            self._save_queue()
            
            self.logger.info(f"Added file to vetting queue: {file_id} - {entry.original_filename}")
            return file_id
    
    def update_analysis_results(self, file_id: str, scores: VettingScores) -> bool:
        """Update a file with analysis results."""
        with self._lock:
            if file_id not in self._queue:
                self.logger.error(f"File not found in queue: {file_id}")
                return False
            
            entry = self._queue[file_id]
            entry.scores = scores
            entry.analyzed_at = datetime.now().isoformat()
            entry.status = VettingStatus.PENDING_APPROVAL
            
            # Check for auto-approval
            if self._check_auto_approval(scores):
                entry.status = VettingStatus.AUTO_APPROVED
                entry.approved_at = datetime.now().isoformat()
                entry.approved_by = "auto"
                self.logger.info(f"File auto-approved: {file_id}")
            else:
                entry.status = VettingStatus.REQUIRES_MANUAL_REVIEW
                self.logger.info(f"File requires manual review: {file_id}")
            
            self._save_queue()
            return True
    
    def approve_file(self, file_id: str, approved_by: str, notes: Optional[str] = None) -> bool:
        """Manually approve a file."""
        with self._lock:
            if file_id not in self._queue:
                self.logger.error(f"File not found in queue: {file_id}")
                return False
            
            entry = self._queue[file_id]
            entry.status = VettingStatus.MANUALLY_APPROVED
            entry.approved_at = datetime.now().isoformat()
            entry.approved_by = approved_by
            
            if notes:
                entry.notes.append(f"Approval: {notes}")
            
            self._save_queue()
            self.logger.info(f"File manually approved: {file_id} by {approved_by}")
            return True
    
    def reject_file(self, file_id: str, reason: str, rejected_by: str) -> bool:
        """Reject a file."""
        with self._lock:
            if file_id not in self._queue:
                self.logger.error(f"File not found in queue: {file_id}")
                return False
            
            entry = self._queue[file_id]
            entry.status = VettingStatus.REJECTED
            entry.rejection_reason = reason
            entry.notes.append(f"Rejected by {rejected_by}: {reason}")
            
            # Move file to rejected folder
            self._move_to_rejected(entry)
            
            self._save_queue()
            self.logger.info(f"File rejected: {file_id} - {reason}")
            return True
    
    def get_pending_review_files(self) -> List[VettingEntry]:
        """Get files that require manual review."""
        with self._lock:
            return [
                entry for entry in self._queue.values()
                if entry.status == VettingStatus.REQUIRES_MANUAL_REVIEW
            ]
    
    def get_approved_files(self) -> List[VettingEntry]:
        """Get approved files ready for ingestion."""
        with self._lock:
            return [
                entry for entry in self._queue.values()
                if entry.status in [VettingStatus.AUTO_APPROVED, VettingStatus.MANUALLY_APPROVED]
            ]
    
    def get_file_entry(self, file_id: str) -> Optional[VettingEntry]:
        """Get a specific file entry."""
        with self._lock:
            return self._queue.get(file_id)
    
    def get_all_entries(self) -> List[VettingEntry]:
        """Get all vetting queue entries."""
        with self._lock:
            return list(self._queue.values())
    
    def get_queue_summary(self) -> Dict[str, int]:
        """Get summary statistics of the vetting queue."""
        with self._lock:
            summary = {}
            for status in VettingStatus:
                summary[status.value] = sum(
                    1 for entry in self._queue.values()
                    if entry.status == status
                )
            summary['total'] = len(self._queue)
            return summary
    
    def update_auto_approval_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update auto-approval thresholds."""
        with self._lock:
            self.auto_approval_thresholds.update(thresholds)
            self.logger.info(f"Updated auto-approval thresholds: {thresholds}")
    
    def get_auto_approval_thresholds(self) -> Dict[str, float]:
        """Get current auto-approval thresholds."""
        with self._lock:
            return self.auto_approval_thresholds.copy()
    
    def cleanup_old_entries(self, days_old: int = 30) -> int:
        """Clean up old entries from the queue."""
        with self._lock:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            entries_to_remove = []
            for file_id, entry in self._queue.items():
                entry_date = datetime.fromisoformat(entry.created_at).timestamp()
                if entry_date < cutoff_date and entry.status in [
                    VettingStatus.REJECTED, 
                    VettingStatus.AUTO_APPROVED, 
                    VettingStatus.MANUALLY_APPROVED
                ]:
                    entries_to_remove.append(file_id)
            
            for file_id in entries_to_remove:
                del self._queue[file_id]
            
            if entries_to_remove:
                self._save_queue()
                self.logger.info(f"Cleaned up {len(entries_to_remove)} old queue entries")
            
            return len(entries_to_remove)
    
    def _check_auto_approval(self, scores: VettingScores) -> bool:
        """Check if scores meet auto-approval criteria."""
        thresholds = self.auto_approval_thresholds
        
        return (
            scores.security_risk_score <= thresholds['security_risk_max'] and
            scores.relevance_score >= thresholds['relevance_min'] and
            scores.credibility_score >= thresholds['credibility_min'] and
            scores.overall_score >= thresholds['overall_min']
        )
    
    def _move_to_rejected(self, entry: VettingEntry) -> None:
        """Move a file to the rejected folder."""
        try:
            rejected_dir = self.quarantine_dir / "rejected"
            rejected_dir.mkdir(exist_ok=True)
            
            source_path = Path(entry.quarantine_path)
            if source_path.exists():
                dest_path = rejected_dir / source_path.name
                source_path.rename(dest_path)
                entry.quarantine_path = str(dest_path)
                self.logger.debug(f"Moved rejected file: {dest_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to move rejected file: {e}")
    
    def _initialize_queue_file(self) -> None:
        """Initialize queue file if it doesn't exist."""
        if not self.queue_file.exists():
            try:
                with open(self.queue_file, 'w') as f:
                    json.dump({}, f, indent=2)
                self.logger.info(f"Initialized new vetting queue file: {self.queue_file}")
            except Exception as e:
                self.logger.error(f"Failed to initialize queue file: {e}")
                raise
    
    def _load_queue(self) -> None:
        """Load vetting queue from disk."""
        try:
            if not self.queue_file.exists():
                self._queue = {}
                return
            
            with open(self.queue_file, 'r') as f:
                data = json.load(f)
            
            # Convert dict data back to VettingEntry objects
            self._queue = {}
            for file_id, entry_data in data.items():
                # Convert scores if present
                scores = None
                if entry_data.get('scores'):
                    scores = VettingScores(**entry_data['scores'])
                
                entry = VettingEntry(
                    file_id=entry_data['file_id'],
                    original_filename=entry_data['original_filename'],
                    quarantine_path=entry_data['quarantine_path'],
                    paper_metadata=entry_data['paper_metadata'],
                    original_insight_text=entry_data['original_insight_text'],
                    status=VettingStatus(entry_data['status']),
                    scores=scores,
                    created_at=entry_data['created_at'],
                    analyzed_at=entry_data.get('analyzed_at'),
                    approved_at=entry_data.get('approved_at'),
                    approved_by=entry_data.get('approved_by'),
                    rejection_reason=entry_data.get('rejection_reason'),
                    notes=entry_data.get('notes', [])
                )
                
                self._queue[file_id] = entry
            
            self.logger.debug(f"Loaded vetting queue: {len(self._queue)} entries")
            
        except Exception as e:
            self.logger.error(f"Failed to load vetting queue: {e}")
            self._queue = {}
    
    def _save_queue(self) -> None:
        """Save vetting queue to disk."""
        try:
            # Convert VettingEntry objects to dict for JSON serialization
            data = {}
            for file_id, entry in self._queue.items():
                entry_dict = asdict(entry)
                # Convert enum to string
                entry_dict['status'] = entry.status.value
                data[file_id] = entry_dict
            
            with open(self.queue_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Saved vetting queue: {len(self._queue)} entries")
            
        except Exception as e:
            self.logger.error(f"Failed to save vetting queue: {e}")
            raise

# Global instance for easy access
_vetting_queue_manager = None

def get_vetting_queue_manager() -> VettingQueueManager:
    """Get the global vetting queue manager instance."""
    global _vetting_queue_manager
    if _vetting_queue_manager is None:
        _vetting_queue_manager = VettingQueueManager()
    return _vetting_queue_manager
