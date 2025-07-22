"""
Discovery Cycle Orchestrator - Automated "Dream & Discover" Engine
================================================================

The DiscoveryCycleOrchestrator implements SAM's automated knowledge discovery pipeline,
orchestrating the complete cycle from bulk ingestion through insight synthesis to
research initiation.

This module provides:
- Automated bulk document ingestion
- Dream Canvas clustering and synthesis
- State management for discovery notifications
- Error handling and recovery mechanisms
- Progress tracking and logging

Part of SAM's Task 27: Automated "Dream & Discover" Engine
Author: SAM Development Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class DiscoveryStage(Enum):
    """Stages of the discovery cycle."""
    IDLE = "idle"
    BULK_INGESTION = "bulk_ingestion"
    CLUSTERING = "clustering"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    FAILED = "failed"

@dataclass
class DiscoveryProgress:
    """Progress tracking for discovery cycle."""
    stage: DiscoveryStage
    started_at: str
    current_step: str
    steps_completed: List[str]
    errors: List[str]
    progress_percentage: float
    estimated_completion: Optional[str] = None

@dataclass
class DiscoveryResult:
    """Result of a complete discovery cycle."""
    cycle_id: str
    started_at: str
    completed_at: str
    status: str
    stages_completed: List[str]
    errors: List[str]
    bulk_ingestion_summary: Optional[Dict[str, Any]] = None
    clustering_summary: Optional[Dict[str, Any]] = None
    synthesis_summary: Optional[Dict[str, Any]] = None
    insights_generated: int = 0
    new_insights_available: bool = False

class DiscoveryCycleOrchestrator:
    """
    Main orchestrator for SAM's automated discovery cycle.
    
    Coordinates the complete "Dream & Discover" workflow:
    1. Bulk document ingestion from configured sources
    2. Dream Canvas clustering of memory vectors
    3. Insight synthesis from concept clusters
    4. State management for user notifications
    """
    
    def __init__(self, state_file: Optional[str] = None):
        """Initialize the discovery cycle orchestrator."""
        self.logger = logging.getLogger(__name__)
        
        # State file for persistent storage
        if state_file:
            self.state_file = Path(state_file)
        else:
            # Use SAM's standard state directory
            state_dir = Path.home() / ".sam"
            state_dir.mkdir(exist_ok=True)
            self.state_file = state_dir / "sam_state.json"
        
        # Current cycle tracking
        self.current_cycle_id: Optional[str] = None
        self.current_progress: Optional[DiscoveryProgress] = None
        
        # Configuration
        self.max_retry_attempts = 3
        self.retry_delay = 5  # seconds
        
        self.logger.info("DiscoveryCycleOrchestrator initialized")
    
    async def run_full_cycle(self, sources: Optional[List[str]] = None) -> DiscoveryResult:
        """
        Execute the complete discovery cycle with comprehensive error handling.
        
        Args:
            sources: Optional list of source paths for bulk ingestion
            
        Returns:
            DiscoveryResult with complete cycle information
        """
        cycle_id = f"discovery_{int(time.time())}"
        started_at = datetime.now().isoformat()
        
        self.current_cycle_id = cycle_id
        self.current_progress = DiscoveryProgress(
            stage=DiscoveryStage.BULK_INGESTION,
            started_at=started_at,
            current_step="Initializing discovery cycle",
            steps_completed=[],
            errors=[],
            progress_percentage=0.0
        )
        
        result = DiscoveryResult(
            cycle_id=cycle_id,
            started_at=started_at,
            completed_at="",
            status="running",
            stages_completed=[],
            errors=[]
        )
        
        try:
            self.logger.info(f"🚀 Starting discovery cycle: {cycle_id}")
            
            # Step 1: Bulk Ingestion (25% of progress)
            self._update_progress(DiscoveryStage.BULK_INGESTION, "Executing bulk ingestion", 10.0)
            bulk_result = await self._execute_with_retry(
                "bulk_ingestion", 
                self._execute_bulk_ingestion,
                sources
            )
            result.bulk_ingestion_summary = bulk_result
            result.stages_completed.append("bulk_ingestion")
            self._update_progress(DiscoveryStage.BULK_INGESTION, "Bulk ingestion completed", 25.0)
            
            # Step 2: Dream Canvas Clustering (50% of progress)
            self._update_progress(DiscoveryStage.CLUSTERING, "Executing Dream Canvas clustering", 35.0)
            clustering_result = await self._execute_with_retry(
                "clustering",
                self._execute_dream_canvas_clustering
            )
            result.clustering_summary = clustering_result
            result.stages_completed.append("clustering")
            self._update_progress(DiscoveryStage.CLUSTERING, "Clustering completed", 50.0)
            
            # Step 3: Insight Synthesis (75% of progress)
            self._update_progress(DiscoveryStage.SYNTHESIS, "Executing insight synthesis", 60.0)
            synthesis_result = await self._execute_with_retry(
                "synthesis",
                self._execute_dream_canvas_synthesis
            )
            result.synthesis_summary = synthesis_result
            result.insights_generated = synthesis_result.get('insights_generated', 0)
            result.stages_completed.append("synthesis")
            self._update_progress(DiscoveryStage.SYNTHESIS, "Synthesis completed", 75.0)
            
            # Step 4: State Update (100% of progress)
            self._update_progress(DiscoveryStage.COMPLETE, "Updating state flags", 90.0)
            if result.insights_generated > 0:
                self._set_new_insights_flag()
                result.new_insights_available = True
                self.logger.info(f"✨ New insights flag set - {result.insights_generated} insights generated")
            
            # Complete the cycle
            result.status = "completed"
            result.completed_at = datetime.now().isoformat()
            self._update_progress(DiscoveryStage.COMPLETE, "Discovery cycle completed", 100.0)
            
            self.logger.info(f"🎉 Discovery cycle completed successfully: {cycle_id}")
            self.logger.info(f"   📄 Stages: {', '.join(result.stages_completed)}")
            self.logger.info(f"   💡 Insights: {result.insights_generated}")
            
        except Exception as e:
            result.status = "failed"
            result.completed_at = datetime.now().isoformat()
            result.errors.append(str(e))
            self._update_progress(DiscoveryStage.FAILED, f"Cycle failed: {str(e)}", 0.0)
            self.logger.error(f"❌ Discovery cycle failed: {e}")
        
        finally:
            # Save final result
            self._save_cycle_result(result)
            self.current_cycle_id = None
            self.current_progress = None
        
        return result
    
    async def _execute_with_retry(self, step_name: str, func, *args) -> Dict[str, Any]:
        """Execute a function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retry_attempts):
            try:
                self.logger.info(f"🔄 Executing {step_name} (attempt {attempt + 1}/{self.max_retry_attempts})")
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args)
                else:
                    result = func(*args)
                
                self.logger.info(f"✅ {step_name} completed successfully")
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"⚠️ {step_name} attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retry_attempts - 1:
                    self.logger.info(f"🔄 Retrying {step_name} in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
        
        # All attempts failed
        error_msg = f"{step_name} failed after {self.max_retry_attempts} attempts: {last_exception}"
        self.logger.error(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    def _execute_bulk_ingestion(self, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute bulk ingestion process."""
        try:
            from scripts.bulk_ingest import BulkDocumentIngestor
            
            # Initialize bulk ingestor
            ingestor = BulkDocumentIngestor(dry_run=False)
            
            # If no sources specified, use default configured sources
            if not sources:
                # Get sources from bulk ingestion configuration
                sources = self._get_configured_sources()
            
            total_summary = {
                'processed': 0,
                'skipped': 0,
                'failed': 0,
                'total_found': 0,
                'sources_processed': 0
            }
            
            # Process each source
            for source_path in sources:
                source_path_obj = Path(source_path)
                if source_path_obj.exists() and source_path_obj.is_dir():
                    self.logger.info(f"📁 Processing source: {source_path}")
                    summary = ingestor.ingest_folder(source_path_obj)
                    
                    # Aggregate results
                    total_summary['processed'] += summary.get('processed', 0)
                    total_summary['skipped'] += summary.get('skipped', 0)
                    total_summary['failed'] += summary.get('failed', 0)
                    total_summary['total_found'] += summary.get('total_found', 0)
                    total_summary['sources_processed'] += 1
                else:
                    self.logger.warning(f"⚠️ Source path not found or not a directory: {source_path}")
            
            self.logger.info(f"📊 Bulk ingestion summary: {total_summary}")
            return total_summary
            
        except Exception as e:
            self.logger.error(f"Bulk ingestion failed: {e}")
            raise
    
    def _execute_dream_canvas_clustering(self) -> Dict[str, Any]:
        """Execute Dream Canvas clustering process."""
        try:
            from memory.synthesis.clustering_service import ClusteringService
            from memory.memory_vectorstore import get_memory_store
            
            # Get memory store
            memory_store = get_memory_store()
            
            # Initialize clustering service
            clustering_service = ClusteringService()
            
            # Perform clustering
            self.logger.info("🧠 Starting Dream Canvas clustering...")
            clusters = clustering_service.find_concept_clusters(memory_store)
            
            clustering_summary = {
                'clusters_found': len(clusters),
                'total_memories_analyzed': len(memory_store.memory_chunks),
                'clustering_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"🔍 Clustering completed: {clustering_summary}")
            return clustering_summary
            
        except Exception as e:
            self.logger.error(f"Dream Canvas clustering failed: {e}")
            raise
    
    def _execute_dream_canvas_synthesis(self) -> Dict[str, Any]:
        """Execute Dream Canvas synthesis process."""
        try:
            from memory.synthesis.synthesis_engine import SynthesisEngine
            from memory.memory_vectorstore import get_memory_store
            
            # Get memory store
            memory_store = get_memory_store()
            
            # Initialize synthesis engine
            synthesis_engine = SynthesisEngine()
            
            # Run synthesis with visualization
            self.logger.info("💭 Starting insight synthesis...")
            synthesis_result = synthesis_engine.run_synthesis(
                memory_store=memory_store,
                visualize=True,
                save_output=True
            )
            
            synthesis_summary = {
                'insights_generated': synthesis_result.insights_generated,
                'clusters_processed': synthesis_result.clusters_found,
                'synthesis_timestamp': synthesis_result.timestamp,
                'output_file': synthesis_result.output_file
            }
            
            self.logger.info(f"💡 Synthesis completed: {synthesis_summary}")
            return synthesis_summary
            
        except Exception as e:
            self.logger.error(f"Dream Canvas synthesis failed: {e}")
            raise

    def _set_new_insights_flag(self) -> None:
        """Set the new_insights_available flag in sam_state.json."""
        try:
            # Load existing state
            state_data = {}
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)

            # Update flag
            state_data['new_insights_available'] = True
            state_data['last_insights_timestamp'] = datetime.now().isoformat()

            # Save state
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            self.logger.info(f"✨ New insights flag set in {self.state_file}")

        except Exception as e:
            self.logger.error(f"Failed to set new insights flag: {e}")
            raise

    def _get_configured_sources(self) -> List[str]:
        """Get configured bulk ingestion sources."""
        try:
            # Try to load from bulk ingestion configuration
            config_file = Path("config/bulk_ingestion_sources.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    sources = [source['path'] for source in config.get('sources', [])]
                    if sources:
                        return sources

            # Fallback to default sources
            default_sources = [
                "documents",
                "data/documents",
                "uploads"
            ]

            # Filter to existing directories
            existing_sources = []
            for source in default_sources:
                source_path = Path(source)
                if source_path.exists() and source_path.is_dir():
                    existing_sources.append(source)

            if not existing_sources:
                self.logger.warning("⚠️ No configured or default sources found")
                return []

            self.logger.info(f"📁 Using default sources: {existing_sources}")
            return existing_sources

        except Exception as e:
            self.logger.error(f"Failed to get configured sources: {e}")
            return []

    def _update_progress(self, stage: DiscoveryStage, step: str, percentage: float) -> None:
        """Update current progress tracking."""
        if self.current_progress:
            self.current_progress.stage = stage
            self.current_progress.current_step = step
            self.current_progress.progress_percentage = percentage

            if stage != DiscoveryStage.FAILED:
                self.current_progress.steps_completed.append(step)

            self.logger.debug(f"Progress update: {stage.value} - {step} ({percentage}%)")

    def _save_cycle_result(self, result: DiscoveryResult) -> None:
        """Save discovery cycle result to persistent storage."""
        try:
            # Create results directory
            results_dir = Path("logs/discovery_cycles")
            results_dir.mkdir(parents=True, exist_ok=True)

            # Save individual result
            result_file = results_dir / f"{result.cycle_id}.json"
            with open(result_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)

            # Update latest result reference
            latest_file = results_dir / "latest.json"
            with open(latest_file, 'w') as f:
                json.dump({
                    'cycle_id': result.cycle_id,
                    'status': result.status,
                    'completed_at': result.completed_at,
                    'insights_generated': result.insights_generated
                }, f, indent=2)

            self.logger.info(f"💾 Discovery cycle result saved: {result_file}")

        except Exception as e:
            self.logger.error(f"Failed to save cycle result: {e}")

    def get_current_progress(self) -> Optional[DiscoveryProgress]:
        """Get current discovery cycle progress."""
        return self.current_progress

    def is_cycle_running(self) -> bool:
        """Check if a discovery cycle is currently running."""
        return self.current_cycle_id is not None

    def get_new_insights_status(self) -> Dict[str, Any]:
        """Get the status of new insights availability."""
        try:
            if not self.state_file.exists():
                return {'new_insights_available': False}

            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            return {
                'new_insights_available': state_data.get('new_insights_available', False),
                'last_insights_timestamp': state_data.get('last_insights_timestamp'),
                'state_file': str(self.state_file)
            }

        except Exception as e:
            self.logger.error(f"Failed to get insights status: {e}")
            return {'new_insights_available': False, 'error': str(e)}

    def clear_new_insights_flag(self) -> None:
        """Clear the new_insights_available flag (called when user views insights)."""
        try:
            if not self.state_file.exists():
                return

            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            state_data['new_insights_available'] = False
            state_data['insights_viewed_at'] = datetime.now().isoformat()

            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            self.logger.info("🔄 New insights flag cleared")

        except Exception as e:
            self.logger.error(f"Failed to clear insights flag: {e}")

# Global instance for easy access
_discovery_orchestrator = None

def get_discovery_orchestrator() -> DiscoveryCycleOrchestrator:
    """Get the global discovery cycle orchestrator instance."""
    global _discovery_orchestrator
    if _discovery_orchestrator is None:
        _discovery_orchestrator = DiscoveryCycleOrchestrator()
    return _discovery_orchestrator
