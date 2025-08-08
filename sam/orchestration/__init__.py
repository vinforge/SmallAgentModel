"""
SAM Orchestration Framework (SOF) v2
====================================

The SOF transforms SAM from a fixed-pipeline reasoner into a dynamic, 
context-aware agent system with intelligent skill orchestration.

Core Components:
- Universal Interface Format (UIF): Standardized data container for skill communication
- BaseSkillModule: Abstract base class for all SAM skills
- CoordinatorEngine: Orchestrates skill execution with validation and error handling
- DynamicPlanner: Generates custom execution plans using LLM-as-a-Planner
- PlanValidationEngine: Validates plans before execution
- Tool Security Framework: Sandboxed execution for external tools

Phase A: Foundational Refactoring - Universal Interface & Skill Abstraction
Phase B: Resilient Coordinator - Static planning with robust error handling  
Phase C: Secure & Performant Dynamic Planning - Intelligent plan generation

This implements the enhanced SOF v2 specification from steps8.md.
"""

__version__ = "2.0.0"
__phase__ = "Phase C - Dynamic Planning & Tool Integration"

from .uif import SAM_UIF, UIFStatus
from .skills.base import BaseSkillModule, SkillExecutionError, SkillDependencyError
from .skills.memory_retrieval import MemoryRetrievalSkill
from .skills.response_generation import ResponseGenerationSkill
from .skills.conflict_detector import ConflictDetectorSkill
from .skills.calculator_tool import CalculatorTool
from .skills.web_browser_tool import AgentZeroWebBrowserTool
from .skills.content_vetting import ContentVettingSkill
from .skills.reasoning.implicit_knowledge import ImplicitKnowledgeSkill
from .skills.financial_data_tool import FinancialDataTool
from .skills.news_api_tool import NewsApiTool
from .skills.table_to_code_expert import TableToCodeExpert
from .skills.memory_tool import MemoryTool
from .skills.autonomous.factual_correction import AutonomousFactualCorrectionSkill
from .validator import PlanValidationEngine, PlanValidationReport, ValidationResult
from .coordinator import CoordinatorEngine, ExecutionReport, ExecutionResult
from .planner import DynamicPlanner, PlanGenerationResult, PlanCacheEntry
from .security import (
    ToolSecurityManager, SecurityPolicy, RateLimitConfig,
    get_security_manager
)
from .sof_integration import (
    SOFIntegration, get_sof_integration, process_query_with_sof, is_sof_available
)
from .config import (
    SOFConfig, SOFConfigManager, get_sof_config, is_sof_enabled,
    enable_sof_framework, disable_sof_framework
)
from .discovery_cycle import (
    DiscoveryCycleOrchestrator, DiscoveryStage, DiscoveryProgress, DiscoveryResult,
    get_discovery_orchestrator
)

__all__ = [
    # Core UIF and base classes
    'SAM_UIF',
    'UIFStatus',
    'BaseSkillModule',
    'SkillExecutionError',
    'SkillDependencyError',

    # Core skills
    'MemoryRetrievalSkill',
    'ResponseGenerationSkill',
    'ConflictDetectorSkill',

    # Tool skills
    'CalculatorTool',
    'AgentZeroWebBrowserTool',
    'ContentVettingSkill',
    'FinancialDataTool',
    'NewsApiTool',
    'MemoryTool',

    # Reasoning skills
    'ImplicitKnowledgeSkill',

    # Autonomous skills
    'AutonomousFactualCorrectionSkill',

    # Phase B: Coordination and validation
    'PlanValidationEngine',
    'PlanValidationReport',
    'ValidationResult',
    'CoordinatorEngine',
    'ExecutionReport',
    'ExecutionResult',

    # Phase C: Dynamic planning and security
    'DynamicPlanner',
    'PlanGenerationResult',
    'PlanCacheEntry',
    'ToolSecurityManager',
    'SecurityPolicy',
    'RateLimitConfig',
    'get_security_manager',

    # Integration
    'SOFIntegration',
    'get_sof_integration',
    'process_query_with_sof',
    'is_sof_available',

    # Configuration
    'SOFConfig',
    'SOFConfigManager',
    'get_sof_config',
    'is_sof_enabled',
    'enable_sof_framework',
    'disable_sof_framework',

    # Task 27: Discovery Cycle Orchestration
    'DiscoveryCycleOrchestrator',
    'DiscoveryStage',
    'DiscoveryProgress',
    'DiscoveryResult',
    'get_discovery_orchestrator',
]
