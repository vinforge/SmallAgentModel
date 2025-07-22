"""
SAM Orchestration Framework Integration Module
==============================================

Main integration point for using the SOF system within SAM.
Provides high-level interface for query processing with automatic
skill registration and plan execution.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable

from .uif import SAM_UIF, UIFStatus
from .coordinator import CoordinatorEngine, ExecutionReport, ExecutionResult
from .skills.base import BaseSkillModule
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
from .config import get_sof_config, is_sof_enabled

logger = logging.getLogger(__name__)


class SOFIntegration:
    """
    Main integration class for the SAM Orchestration Framework.
    
    Provides a high-level interface for processing queries using the
    orchestration framework with automatic skill management.
    """
    
    def __init__(self, fallback_generator: Optional[Callable[[str], str]] = None):
        """
        Initialize SOF integration.

        Args:
            fallback_generator: Optional fallback function for generating responses
        """
        self.logger = logging.getLogger(f"{__name__}.SOFIntegration")

        # Initialize Goal & Motivation Engine
        self._motivation_engine = self._initialize_motivation_engine()

        # Create coordinator with motivation engine
        self._coordinator = CoordinatorEngine(
            fallback_generator=fallback_generator,
            motivation_engine=self._motivation_engine
        )
        self._initialized = False
        self._config = get_sof_config()
        
        # Initialize if SOF is enabled
        if is_sof_enabled():
            self.initialize()

    def _initialize_motivation_engine(self):
        """Initialize the Goal & Motivation Engine for autonomous goal generation."""
        try:
            from sam.autonomy import GoalStack, MotivationEngine, GoalSafetyValidator

            # Create safety validator
            safety_validator = GoalSafetyValidator()

            # Create goal stack with persistent storage
            goal_stack = GoalStack(
                db_path="memory/autonomy_goals.db",
                safety_validator=safety_validator
            )

            # Create motivation engine
            motivation_engine = MotivationEngine(
                goal_stack=goal_stack,
                safety_validator=safety_validator
            )

            self.logger.info("Goal & Motivation Engine initialized successfully")
            return motivation_engine

        except ImportError as e:
            self.logger.warning(f"Goal & Motivation Engine not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize Goal & Motivation Engine: {e}")
            return None
    
    def initialize(self) -> bool:
        """
        Initialize the SOF system with core skills.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
        
        try:
            self.logger.info("Initializing SAM Orchestration Framework")
            
            # Register core skills
            self._register_core_skills()
            
            self._initialized = True
            self.logger.info("SOF initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"SOF initialization failed: {e}")
            return False
    
    def _register_core_skills(self) -> None:
        """Register core SAM skills with the coordinator."""
        core_skills = []
        
        # Memory Retrieval Skill
        try:
            memory_skill = MemoryRetrievalSkill()
            core_skills.append(memory_skill)
            self.logger.debug("MemoryRetrievalSkill registered")
        except Exception as e:
            self.logger.warning(f"Failed to register MemoryRetrievalSkill: {e}")
        
        # Response Generation Skill
        try:
            response_skill = ResponseGenerationSkill()
            core_skills.append(response_skill)
            self.logger.debug("ResponseGenerationSkill registered")
        except Exception as e:
            self.logger.warning(f"Failed to register ResponseGenerationSkill: {e}")
        
        # Conflict Detector Skill
        try:
            conflict_skill = ConflictDetectorSkill()
            core_skills.append(conflict_skill)
            self.logger.debug("ConflictDetectorSkill registered")
        except Exception as e:
            self.logger.warning(f"Failed to register ConflictDetectorSkill: {e}")

        # Calculator Tool
        try:
            calculator_skill = CalculatorTool()
            core_skills.append(calculator_skill)
            self.logger.debug("CalculatorTool registered")
        except Exception as e:
            self.logger.warning(f"Failed to register CalculatorTool: {e}")

        # Web Browser Tool
        try:
            web_browser_skill = AgentZeroWebBrowserTool()
            core_skills.append(web_browser_skill)
            self.logger.debug("AgentZeroWebBrowserTool registered")
        except Exception as e:
            self.logger.warning(f"Failed to register AgentZeroWebBrowserTool: {e}")

        # Content Vetting Skill
        try:
            vetting_skill = ContentVettingSkill()
            core_skills.append(vetting_skill)
            self.logger.debug("ContentVettingSkill registered")
        except Exception as e:
            self.logger.warning(f"Failed to register ContentVettingSkill: {e}")

        # Implicit Knowledge Skill (Reasoning)
        try:
            implicit_knowledge_skill = ImplicitKnowledgeSkill()
            core_skills.append(implicit_knowledge_skill)
            self.logger.debug("ImplicitKnowledgeSkill registered")
        except Exception as e:
            self.logger.warning(f"Failed to register ImplicitKnowledgeSkill: {e}")

        # Financial Data Tool
        try:
            # Get Serper API key from config if available
            serper_api_key = getattr(self._config, 'serper_api_key', None)
            financial_tool = FinancialDataTool(serper_api_key=serper_api_key)
            core_skills.append(financial_tool)
            self.logger.debug("FinancialDataTool registered")
        except Exception as e:
            self.logger.warning(f"Failed to register FinancialDataTool: {e}")

        # News API Tool
        try:
            # Get NewsAPI key from config if available
            newsapi_key = getattr(self._config, 'newsapi_api_key', None)
            news_tool = NewsApiTool(newsapi_key=newsapi_key)
            core_skills.append(news_tool)
            self.logger.debug("NewsApiTool registered")
        except Exception as e:
            self.logger.warning(f"Failed to register NewsApiTool: {e}")

        # Table-to-Code Expert Tool (Phase 2)
        try:
            table_expert = TableToCodeExpert()
            core_skills.append(table_expert)
            self.logger.debug("TableToCodeExpert registered")
        except Exception as e:
            self.logger.warning(f"Failed to register TableToCodeExpert: {e}")

        # Memory Tool (Task 33, Phase 1)
        try:
            memory_tool = MemoryTool()
            core_skills.append(memory_tool)
            self.logger.debug("MemoryTool registered")
        except Exception as e:
            self.logger.warning(f"Failed to register MemoryTool: {e}")

        # Register all successfully created skills
        if core_skills:
            self._coordinator.register_skills(core_skills)
            self.logger.info(f"Registered {len(core_skills)} core skills")
        else:
            raise RuntimeError("No core skills could be registered")
    
    def process_query(self,
                     query: str,
                     user_id: Optional[str] = None,
                     session_id: Optional[str] = None,
                     user_context: Optional[Dict[str, Any]] = None,
                     custom_plan: Optional[List[str]] = None,
                     use_dynamic_planning: bool = True) -> Dict[str, Any]:
        """
        Process a query using the SOF framework.

        Args:
            query: User query to process
            user_id: Optional user identifier
            session_id: Optional session identifier
            user_context: Optional user context information
            custom_plan: Optional custom execution plan
            use_dynamic_planning: Whether to use dynamic plan generation

        Returns:
            Dictionary with processing results
        """
        if not self._initialized:
            if not self.initialize():
                return self._create_error_response("SOF initialization failed")
        
        start_time = time.time()
        
        try:
            # Create UIF for the query
            uif = SAM_UIF(
                input_query=query,
                user_id=user_id,
                session_id=session_id,
                user_context=user_context or {}
            )
            
            # Determine execution approach
            if custom_plan:
                execution_plan = custom_plan
                execution_report = self._coordinator.execute_plan(execution_plan, uif)
                self.logger.info(f"Processing query with custom plan: {execution_plan}")
            elif use_dynamic_planning:
                execution_report = self._coordinator.execute_with_dynamic_planning(uif)
                self.logger.info("Processing query with dynamic planning")
            else:
                execution_plan = self._get_default_plan()
                execution_report = self._coordinator.execute_plan(execution_plan, uif)
                self.logger.info(f"Processing query with default plan: {execution_plan}")
            
            # Create response
            response = self._create_response_from_report(execution_report, start_time)
            
            self.logger.info(f"Query processed: {execution_report.result} in {response['processing_time']:.2f}s")
            
            return response
            
        except Exception as e:
            self.logger.exception("Error processing query")
            return self._create_error_response(f"Query processing failed: {str(e)}")
    
    def _get_default_plan(self) -> List[str]:
        """
        Get the default execution plan.
        
        Returns:
            List of skill names for default execution
        """
        # Default plan: Memory -> Conflict Detection -> Response Generation
        default_plan = []
        
        registered_skills = self._coordinator.get_registered_skills()
        
        if "MemoryRetrievalSkill" in registered_skills:
            default_plan.append("MemoryRetrievalSkill")
        
        if "ConflictDetectorSkill" in registered_skills:
            default_plan.append("ConflictDetectorSkill")
        
        if "ResponseGenerationSkill" in registered_skills:
            default_plan.append("ResponseGenerationSkill")
        
        return default_plan
    
    def _create_response_from_report(self, report: ExecutionReport, start_time: float) -> Dict[str, Any]:
        """
        Create a response dictionary from an execution report.
        
        Args:
            report: Execution report
            start_time: Processing start time
            
        Returns:
            Response dictionary
        """
        processing_time = time.time() - start_time
        
        response = {
            "success": report.result in [ExecutionResult.SUCCESS, ExecutionResult.PARTIAL_SUCCESS],
            "response": report.uif.final_response or "No response generated",
            "confidence": report.uif.confidence_score,
            "processing_time": processing_time,
            "execution_result": report.result.value,
            "executed_skills": report.executed_skills,
            "failed_skills": report.failed_skills,
            "fallback_used": report.fallback_used,
            "warnings": report.uif.warnings,
            "metadata": {
                "task_id": report.uif.task_id,
                "execution_plan": report.uif.execution_plan,
                "skill_timings": report.uif.skill_timings,
                "log_trace": report.uif.log_trace if self._config.enable_execution_tracing else []
            }
        }
        
        # Add error details if execution failed
        if not response["success"]:
            response["error"] = report.error_details or report.uif.error_details
        
        # Add validation information if available
        if report.validation_report:
            response["validation"] = {
                "is_valid": report.validation_report.is_valid,
                "warnings_count": report.validation_report.warnings_count,
                "errors_count": report.validation_report.errors_count,
                "execution_estimate": report.validation_report.execution_estimate
            }
        
        return response
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "success": False,
            "response": f"I apologize, but I encountered an error: {error_message}",
            "confidence": 0.0,
            "processing_time": 0.0,
            "execution_result": "failure",
            "executed_skills": [],
            "failed_skills": [],
            "fallback_used": False,
            "warnings": [],
            "error": error_message,
            "metadata": {
                "task_id": None,
                "execution_plan": [],
                "skill_timings": {},
                "log_trace": []
            }
        }
    
    def register_custom_skill(self, skill: BaseSkillModule) -> bool:
        """
        Register a custom skill with the coordinator.
        
        Args:
            skill: Custom skill to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            self._coordinator.register_skill(skill)
            self.logger.info(f"Custom skill registered: {skill.skill_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register custom skill {skill.skill_name}: {e}")
            return False
    
    def get_available_skills(self) -> List[str]:
        """
        Get list of available skills.
        
        Returns:
            List of registered skill names
        """
        return self._coordinator.get_registered_skills()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        return self._coordinator.get_execution_stats()
    
    def is_initialized(self) -> bool:
        """Check if SOF is initialized."""
        return self._initialized
    
    def shutdown(self) -> None:
        """Shutdown the SOF system."""
        self.logger.info("Shutting down SOF integration")
        self._coordinator.clear_execution_history()
        self._initialized = False


# Global SOF integration instance
_sof_integration: Optional[SOFIntegration] = None


def get_sof_integration(fallback_generator: Optional[Callable[[str], str]] = None) -> SOFIntegration:
    """
    Get or create global SOF integration instance.
    
    Args:
        fallback_generator: Optional fallback function for generating responses
        
    Returns:
        SOFIntegration instance
    """
    global _sof_integration
    
    if _sof_integration is None:
        _sof_integration = SOFIntegration(fallback_generator)
    
    return _sof_integration


def process_query_with_sof(query: str,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None,
                          user_context: Optional[Dict[str, Any]] = None,
                          custom_plan: Optional[List[str]] = None,
                          fallback_generator: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """
    Convenience function to process a query using SOF.
    
    Args:
        query: User query to process
        user_id: Optional user identifier
        session_id: Optional session identifier
        user_context: Optional user context information
        custom_plan: Optional custom execution plan
        fallback_generator: Optional fallback function
        
    Returns:
        Dictionary with processing results
    """
    sof = get_sof_integration(fallback_generator)
    return sof.process_query(query, user_id, session_id, user_context, custom_plan)


def is_sof_available() -> bool:
    """
    Check if SOF is available and enabled.
    
    Returns:
        True if SOF is available, False otherwise
    """
    return is_sof_enabled() and get_sof_integration().is_initialized()
