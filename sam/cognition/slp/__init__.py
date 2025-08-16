"""
Scalable Latent Program (SLP) System for SAM
============================================

The SLP system enables SAM to autonomously identify, store, and reuse successful 
reasoning patterns ("programs"), leading to significant improvements in response 
speed, consistency, and personalization for recurring tasks.

Core Components:
- LatentProgram: Data structure for storing reasoning patterns
- ProgramSignature: Unique fingerprinting for task identification
- ProgramStore: Persistent storage and retrieval system
- ProgramManager: Central orchestrator for program lifecycle
- ProgramExecutor: Execution engine for latent programs

This implements the Cognitive Automation Engine as defined in task2.md.
"""

from .latent_program import LatentProgram, ExecutionResult, ValidationResult
from .program_signature import ProgramSignature, generate_signature
from .latent_program_store import LatentProgramStore
from .program_manager import ProgramManager
from .program_executor import ProgramExecutor
from .sam_slp_integration import SAMSLPIntegration, get_slp_integration, initialize_slp_integration

# Enhanced Analytics Modules (Phase 1A - DISABLED)
# SLP Analytics has been removed from the system
SLPAnalyticsEngine = None
SLPMetricsCollector = None
ENHANCED_ANALYTICS_AVAILABLE = False

# Advanced Analysis Modules (Phase 1B - DISABLED)
# SLP Analytics has been removed from the system
ProgramAnalyzer = None
CognitiveInsightsGenerator = None
ADVANCED_ANALYSIS_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "SAM Development Team"

# Module-level configuration
SLP_CONFIG = {
    "database_path": "data/latent_programs.db",
    "signature_similarity_threshold": 0.8,
    "execution_confidence_threshold": 0.7,
    "quality_threshold": 0.6,
    "max_programs_per_signature": 5,
    "program_retirement_threshold": 0.3,
    "enable_program_evolution": True,
    "enable_safety_validation": True
}

def get_slp_config():
    """Get the current SLP configuration."""
    return SLP_CONFIG.copy()

def update_slp_config(updates: dict):
    """Update SLP configuration parameters."""
    SLP_CONFIG.update(updates)

# Initialize the SLP system
def initialize_slp_system():
    """Initialize the SLP system components."""
    try:
        store = LatentProgramStore()
        manager = ProgramManager(store)
        return manager
    except Exception as e:
        print(f"Warning: Failed to initialize SLP system: {e}")
        return None

# Export main classes for easy import (preserving 100% of existing functionality)
__all__ = [
    'LatentProgram',
    'ExecutionResult',
    'ValidationResult',
    'ProgramSignature',
    'generate_signature',
    'LatentProgramStore',
    'ProgramManager',
    'ProgramExecutor',
    'SAMSLPIntegration',
    'get_slp_integration',
    'initialize_slp_integration',
    'get_slp_config',
    'update_slp_config',
    'initialize_slp_system',
    # Enhanced Analytics (Phase 1A) - REMOVED
    # 'SLPAnalyticsEngine',  # Removed
    # 'SLPMetricsCollector',  # Removed
    'ENHANCED_ANALYTICS_AVAILABLE',
    # Advanced Analysis (Phase 1B) - REMOVED
    # 'ProgramAnalyzer',  # Removed
    # 'CognitiveInsightsGenerator',  # Removed
    'ADVANCED_ANALYSIS_AVAILABLE'
]
