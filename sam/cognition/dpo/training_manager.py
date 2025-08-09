"""
DPO Training Manager

High-level orchestration and management for DPO training processes.
Provides progress monitoring, job management, and integration with the UI.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import json
import logging
import subprocess
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Represents a DPO training job."""
    job_id: str
    user_id: str
    status: TrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Configuration
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Progress tracking
    current_step: int = 0
    total_steps: int = 0
    current_loss: float = 0.0
    best_loss: float = float('inf')
    
    # Results
    output_dir: Optional[str] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    
    # Process management
    process_id: Optional[int] = None
    log_file: Optional[str] = None


class DPOTrainingManager:
    """
    High-level manager for DPO training jobs with progress monitoring.
    """
    
    def __init__(self, base_output_dir: str = "./models/personalized"):
        """
        Initialize the DPO training manager.
        
        Args:
            base_output_dir: Base directory for training outputs
        """
        self.logger = logging.getLogger(f"{__name__}.DPOTrainingManager")
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Job management
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.job_history: List[TrainingJob] = []
        self.max_concurrent_jobs = 1  # Limit concurrent training jobs
        
        # Progress monitoring
        self.progress_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # Job persistence
        self.jobs_file = self.base_output_dir / "training_jobs.json"
        self.load_job_history()
        
        self.logger.info("DPO Training Manager initialized")
    
    def create_training_job(self, user_id: str, config_overrides: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new DPO training job.
        
        Args:
            user_id: User identifier
            config_overrides: Configuration overrides
            
        Returns:
            Job ID
        """
        # Generate unique job ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"dpo_{user_id}_{timestamp}"
        
        # Create job
        job = TrainingJob(
            job_id=job_id,
            user_id=user_id,
            status=TrainingStatus.PENDING,
            created_at=datetime.now(),
            config_overrides=config_overrides or {}
        )
        
        # Set output directory
        job.output_dir = str(self.base_output_dir / user_id / job_id)
        
        # Store job
        self.active_jobs[job_id] = job
        self.job_history.append(job)
        self.save_job_history()
        
        self.logger.info(f"Created training job: {job_id} for user: {user_id}")
        return job_id
    
    def start_training_job(self, job_id: str) -> bool:
        """
        Start a training job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if started successfully
        """
        if job_id not in self.active_jobs:
            self.logger.error(f"Job not found: {job_id}")
            return False
        
        job = self.active_jobs[job_id]
        
        # Check if already running
        if job.status == TrainingStatus.RUNNING:
            self.logger.warning(f"Job already running: {job_id}")
            return False
        
        # Check concurrent job limit
        running_jobs = [j for j in self.active_jobs.values() if j.status == TrainingStatus.RUNNING]
        if len(running_jobs) >= self.max_concurrent_jobs:
            self.logger.warning(f"Maximum concurrent jobs reached: {self.max_concurrent_jobs}")
            return False
        
        try:
            # Prepare command
            script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "run_dpo_tuning.py"
            
            cmd = [
                "python", str(script_path),
                "--user_id", job.user_id
            ]
            
            # Add configuration overrides
            if job.config_overrides:
                if 'output' in job.config_overrides and 'output_dir' in job.config_overrides['output']:
                    cmd.extend(["--output_dir", job.config_overrides['output']['output_dir']])
                
                if 'training' in job.config_overrides:
                    training_config = job.config_overrides['training']
                    if 'learning_rate' in training_config:
                        cmd.extend(["--learning_rate", str(training_config['learning_rate'])])
                    if 'num_train_epochs' in training_config:
                        cmd.extend(["--num_epochs", str(training_config['num_train_epochs'])])
                    if 'per_device_train_batch_size' in training_config:
                        cmd.extend(["--batch_size", str(training_config['per_device_train_batch_size'])])
                    if 'beta' in training_config:
                        cmd.extend(["--beta", str(training_config['beta'])])
                
                if 'lora' in job.config_overrides and 'r' in job.config_overrides['lora']:
                    cmd.extend(["--lora_rank", str(job.config_overrides['lora']['r'])])
            
            # Set up logging
            log_dir = Path(job.output_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            job.log_file = str(log_dir / "training.log")
            
            # Start process
            with open(job.log_file, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path(__file__).parent.parent.parent.parent)
                )
            
            # Update job status
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            job.process_id = process.pid
            
            # Start monitoring if not already active
            if not self.monitoring_active:
                self.start_monitoring()
            
            self.save_job_history()
            self.logger.info(f"Started training job: {job_id} (PID: {process.pid})")
            
            # Notify callbacks
            self._notify_progress(job_id, {
                'status': 'started',
                'message': 'Training job started',
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            self.save_job_history()
            self.logger.error(f"Failed to start training job {job_id}: {e}")
            return False
    
    def cancel_training_job(self, job_id: str) -> bool:
        """
        Cancel a running training job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancelled successfully
        """
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        
        if job.status != TrainingStatus.RUNNING:
            return False
        
        try:
            # Terminate process
            if job.process_id:
                import psutil
                try:
                    process = psutil.Process(job.process_id)
                    process.terminate()
                    process.wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass  # Process already terminated
                except psutil.TimeoutExpired:
                    process.kill()  # Force kill if terminate doesn't work
            
            # Update job status
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            
            self.save_job_history()
            self.logger.info(f"Cancelled training job: {job_id}")
            
            # Notify callbacks
            self._notify_progress(job_id, {
                'status': 'cancelled',
                'message': 'Training job cancelled',
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling job {job_id}: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a training job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary
        """
        if job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[job_id]
        
        # Calculate duration
        duration = None
        if job.started_at:
            end_time = job.completed_at or datetime.now()
            duration = (end_time - job.started_at).total_seconds()
        
        # Calculate progress
        progress = 0.0
        if job.total_steps > 0:
            progress = job.current_step / job.total_steps
        
        return {
            'job_id': job.job_id,
            'user_id': job.user_id,
            'status': job.status.value,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'duration_seconds': duration,
            'progress': progress,
            'current_step': job.current_step,
            'total_steps': job.total_steps,
            'current_loss': job.current_loss,
            'best_loss': job.best_loss if job.best_loss != float('inf') else None,
            'output_dir': job.output_dir,
            'model_path': job.model_path,
            'error_message': job.error_message,
            'log_file': job.log_file
        }
    
    def get_user_jobs(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all jobs for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of job status dictionaries
        """
        user_jobs = [job for job in self.job_history if job.user_id == user_id]
        return [self.get_job_status(job.job_id) for job in user_jobs if job.job_id in self.active_jobs]
    
    def start_monitoring(self):
        """Start the progress monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started training progress monitoring")
    
    def stop_monitoring(self):
        """Stop the progress monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Stopped training progress monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check running jobs
                running_jobs = [job for job in self.active_jobs.values() if job.status == TrainingStatus.RUNNING]
                
                for job in running_jobs:
                    self._check_job_progress(job)
                
                # Stop monitoring if no running jobs
                if not running_jobs:
                    self.monitoring_active = False
                    break
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _check_job_progress(self, job: TrainingJob):
        """Check progress of a specific job."""
        try:
            # Check if process is still running
            if job.process_id:
                import psutil
                try:
                    process = psutil.Process(job.process_id)
                    if not process.is_running():
                        # Process finished, check results
                        self._finalize_job(job)
                        return
                except psutil.NoSuchProcess:
                    # Process not found, check results
                    self._finalize_job(job)
                    return
            
            # Parse log file for progress
            if job.log_file and Path(job.log_file).exists():
                self._parse_training_log(job)
            
        except Exception as e:
            self.logger.error(f"Error checking job progress {job.job_id}: {e}")
    
    def _parse_training_log(self, job: TrainingJob):
        """Parse training log for progress information."""
        try:
            with open(job.log_file, 'r') as f:
                lines = f.readlines()
            
            # Look for progress indicators in recent lines
            for line in lines[-50:]:  # Check last 50 lines
                # Look for step information
                if "step" in line.lower() and "/" in line:
                    # Try to extract step information
                    # This is a simplified parser - real implementation would be more robust
                    pass
                
                # Look for loss information
                if "loss" in line.lower():
                    # Try to extract loss information
                    pass
            
        except Exception as e:
            self.logger.debug(f"Error parsing log for job {job.job_id}: {e}")
    
    def _finalize_job(self, job: TrainingJob):
        """Finalize a completed job."""
        try:
            # Check for results file
            results_file = Path(job.output_dir) / "training_results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                if results.get('success'):
                    job.status = TrainingStatus.COMPLETED
                    job.model_path = results.get('output_dir')
                else:
                    job.status = TrainingStatus.FAILED
                    job.error_message = results.get('error', 'Unknown error')
            else:
                job.status = TrainingStatus.FAILED
                job.error_message = "No results file found"
            
            job.completed_at = datetime.now()
            self.save_job_history()
            
            # Notify callbacks
            self._notify_progress(job.job_id, {
                'status': job.status.value,
                'message': f'Training job {job.status.value}',
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Finalized job {job.job_id}: {job.status.value}")
            
        except Exception as e:
            self.logger.error(f"Error finalizing job {job.job_id}: {e}")
            job.status = TrainingStatus.FAILED
            job.error_message = f"Finalization error: {e}"
            job.completed_at = datetime.now()
            self.save_job_history()
    
    def add_progress_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, job_id: str, progress_data: Dict[str, Any]):
        """Notify all progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(job_id, progress_data)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
    
    def save_job_history(self):
        """Save job history to disk."""
        try:
            job_data = []
            for job in self.job_history:
                job_dict = {
                    'job_id': job.job_id,
                    'user_id': job.user_id,
                    'status': job.status.value,
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'config_overrides': job.config_overrides,
                    'output_dir': job.output_dir,
                    'model_path': job.model_path,
                    'error_message': job.error_message
                }
                job_data.append(job_dict)
            
            with open(self.jobs_file, 'w') as f:
                json.dump(job_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving job history: {e}")
    
    def load_job_history(self):
        """Load job history from disk."""
        try:
            if not self.jobs_file.exists():
                return
            
            with open(self.jobs_file, 'r') as f:
                job_data = json.load(f)
            
            for job_dict in job_data:
                job = TrainingJob(
                    job_id=job_dict['job_id'],
                    user_id=job_dict['user_id'],
                    status=TrainingStatus(job_dict['status']),
                    created_at=datetime.fromisoformat(job_dict['created_at']),
                    started_at=datetime.fromisoformat(job_dict['started_at']) if job_dict['started_at'] else None,
                    completed_at=datetime.fromisoformat(job_dict['completed_at']) if job_dict['completed_at'] else None,
                    config_overrides=job_dict.get('config_overrides', {}),
                    output_dir=job_dict.get('output_dir'),
                    model_path=job_dict.get('model_path'),
                    error_message=job_dict.get('error_message')
                )
                
                self.job_history.append(job)
                
                # Add to active jobs if not completed
                if job.status in [TrainingStatus.PENDING, TrainingStatus.RUNNING]:
                    self.active_jobs[job.job_id] = job
            
            self.logger.info(f"Loaded {len(self.job_history)} jobs from history")
            
        except Exception as e:
            self.logger.error(f"Error loading job history: {e}")


# Global training manager instance
_training_manager = None

def get_dpo_training_manager() -> DPOTrainingManager:
    """Get or create a global DPO training manager instance."""
    global _training_manager
    
    if _training_manager is None:
        _training_manager = DPOTrainingManager()
    
    return _training_manager
