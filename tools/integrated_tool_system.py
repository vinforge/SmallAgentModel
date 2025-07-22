"""
Integrated Tool-Oriented Reasoning System for SAM
Combines all Sprint 8 tool capabilities into a unified action-oriented system.

Sprint 8: Tool-Oriented Reasoning & Actionable Execution Integration
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from .tool_registry import ToolRegistry, ToolPlanner, get_tool_registry, get_tool_planner
from .secure_executor import SecureExecutor, ExecutionMode, get_secure_executor
from .tool_evaluator import ToolEvaluator, get_tool_evaluator
from .custom_tool_creator import CustomToolCreator, get_custom_tool_creator
from .action_planner import ActionPlanner, ReportFormat, get_action_planner

logger = logging.getLogger(__name__)

@dataclass
class ToolRequest:
    """Request for tool-oriented reasoning and execution."""
    request_id: str
    user_id: str
    session_id: str
    goal: str
    execution_mode: ExecutionMode
    require_approval: bool
    context: Dict[str, Any]

@dataclass
class ToolResponse:
    """Response from tool-oriented reasoning system."""
    request_id: str
    plan_id: str
    execution_results: List[Dict[str, Any]]
    success_rate: float
    total_execution_time: int
    tools_used: List[str]
    insights_generated: List[str]
    report_content: str
    created_at: str

class IntegratedToolSystem:
    """
    Unified tool-oriented reasoning system that integrates all Sprint 8 capabilities.
    """
    
    def __init__(self, approval_callback: Optional[Callable] = None):
        """
        Initialize the integrated tool system.
        
        Args:
            approval_callback: Optional callback for user approval
        """
        # Initialize all tool system components
        self.tool_registry = get_tool_registry()
        self.tool_planner = get_tool_planner(self.tool_registry)
        self.secure_executor = get_secure_executor(approval_callback=approval_callback)
        self.tool_evaluator = get_tool_evaluator()
        self.custom_tool_creator = get_custom_tool_creator()
        self.action_planner = get_action_planner()
        
        # Configuration
        self.config = {
            'auto_plan_execution': True,
            'enable_learning': True,
            'generate_reports': True,
            'max_tools_per_plan': 10
        }
        
        logger.info("Integrated tool system initialized")
    
    def process_tool_request(self, request: ToolRequest) -> ToolResponse:
        """
        Process a tool-oriented request with full planning, execution, and reporting.
        
        Args:
            request: ToolRequest to process
            
        Returns:
            ToolResponse with complete results
        """
        try:
            start_time = datetime.now()
            
            logger.info(f"Processing tool request: {request.goal[:50]}...")
            
            # Step 1: Plan tool execution
            execution_plan = self.tool_planner.create_execution_plan(
                goal=request.goal,
                context=request.context
            )
            
            # Step 2: Create action plan
            tool_sequence = []
            for step in execution_plan.steps:
                tool_sequence.append({
                    'tool_id': step['tool_id'],
                    'tool_name': step['tool_name'],
                    'title': f"Execute {step['tool_name']}",
                    'description': step['expected_output'],
                    'input_data': step['input_data'],
                    'expected_output': step['expected_output'],
                    'estimated_duration': 30,  # Default estimate
                    'requires_approval': request.require_approval,
                    'metadata': {'step_id': step['step_id']}
                })
            
            plan_id = self.action_planner.create_action_plan(
                goal=request.goal,
                user_id=request.user_id,
                session_id=request.session_id,
                tool_sequence=tool_sequence,
                title=f"Tool Execution Plan: {request.goal[:30]}..."
            )
            
            # Step 3: Execute plan (if approved or auto-approved)
            execution_results = []
            if not request.require_approval:
                # Auto-approve and execute
                self.action_planner.approve_plan(plan_id, approved=True, 
                                               approval_notes="Auto-approved for execution")
                
                execution_results = self._execute_plan_with_tools(plan_id, request.execution_mode)
            
            # Step 4: Generate insights and report
            insights = self._generate_insights(execution_plan, execution_results)
            report_content = self.action_planner.generate_execution_report(plan_id, ReportFormat.MARKDOWN)
            
            # Step 5: Update tool performance metrics
            if self.config['enable_learning'] and execution_results:
                self._update_tool_metrics(execution_results, request.goal)
            
            # Calculate metrics
            end_time = datetime.now()
            total_execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            success_rate = 0.0
            if execution_results:
                successful_executions = sum(1 for result in execution_results if result.get('success', False))
                success_rate = successful_executions / len(execution_results)
            
            tools_used = [step['tool_name'] for step in execution_plan.steps]
            
            # Create response
            tool_response = ToolResponse(
                request_id=request.request_id,
                plan_id=plan_id,
                execution_results=execution_results,
                success_rate=success_rate,
                total_execution_time=total_execution_time,
                tools_used=tools_used,
                insights_generated=insights,
                report_content=report_content,
                created_at=datetime.now().isoformat()
            )
            
            logger.info(f"Tool request completed: {success_rate:.1%} success rate, "
                       f"{len(tools_used)} tools used")
            
            return tool_response
            
        except Exception as e:
            logger.error(f"Error processing tool request: {e}")
            return self._create_error_response(request, str(e))
    
    def get_recommended_tools(self, goal: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get recommended tools for a specific goal.
        
        Args:
            goal: Goal to find tools for
            context: Additional context
            
        Returns:
            List of recommended tools with metadata
        """
        try:
            # Get recommendations from evaluator
            recommended_tool_ids = self.tool_evaluator.get_recommended_tools(goal, top_k=5)
            
            # Get tool metadata
            recommended_tools = []
            for tool_id in recommended_tool_ids:
                tool_metadata = self.tool_registry.get_tool(tool_id)
                if tool_metadata:
                    scorecard = self.tool_evaluator.get_tool_scorecard(tool_id)
                    
                    tool_info = {
                        'tool_id': tool_id,
                        'name': tool_metadata.name,
                        'description': tool_metadata.description,
                        'category': tool_metadata.category.value,
                        'success_rate': scorecard.success_rate if scorecard else 0.0,
                        'average_execution_time': scorecard.average_execution_time if scorecard else 0.0,
                        'usage_count': scorecard.total_executions if scorecard else 0,
                        'requires_approval': tool_metadata.requires_approval
                    }
                    recommended_tools.append(tool_info)
            
            return recommended_tools
            
        except Exception as e:
            logger.error(f"Error getting recommended tools: {e}")
            return []
    
    def create_custom_tool(self, template_id: str, tool_name: str, tool_description: str,
                          creator_id: str, custom_logic: str,
                          test_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create and test a custom tool.
        
        Args:
            template_id: Template to use
            tool_name: Name for the new tool
            tool_description: Description of the tool
            creator_id: User creating the tool
            custom_logic: Custom logic code
            test_inputs: Test inputs for validation
            
        Returns:
            Creation and test results
        """
        try:
            # Create custom tool
            tool_id = self.custom_tool_creator.create_tool_from_template(
                template_id=template_id,
                tool_name=tool_name,
                tool_description=tool_description,
                creator_id=creator_id,
                custom_logic=custom_logic
            )
            
            # Test the tool
            test_results = self.custom_tool_creator.test_custom_tool(
                tool_id=tool_id,
                test_inputs=test_inputs
            )
            
            # Register in tool registry if tests pass
            if test_results['success_rate'] >= 0.8:
                custom_tool = self.custom_tool_creator.get_custom_tool(tool_id)
                if custom_tool:
                    # Convert to tool metadata and register
                    from .tool_registry import ToolMetadata, ToolCategory, ToolComplexity, InputType, OutputType
                    
                    tool_metadata = ToolMetadata(
                        tool_id=tool_id,
                        name=tool_name,
                        description=tool_description,
                        category=ToolCategory.CUSTOM,
                        complexity=ToolComplexity.MODERATE,
                        input_types=[InputType.JSON],
                        output_types=[OutputType.JSON],
                        tags=['custom', 'user_created'],
                        limitations=['Custom tool - use with caution'],
                        dependencies=[],
                        execution_time_estimate=30,
                        requires_approval=True,
                        version=custom_tool.version,
                        created_at=custom_tool.created_at,
                        last_updated=custom_tool.last_updated,
                        usage_count=0,
                        success_rate=test_results['success_rate'],
                        average_execution_time=30.0,
                        metadata={'custom_tool': True, 'creator_id': creator_id}
                    )
                    
                    self.tool_registry.register_tool(tool_metadata)
                    
                    # Activate the custom tool
                    self.custom_tool_creator.activate_tool(tool_id)
            
            return {
                'tool_id': tool_id,
                'creation_success': True,
                'test_results': test_results,
                'registered': test_results['success_rate'] >= 0.8,
                'activated': test_results['success_rate'] >= 0.8
            }
            
        except Exception as e:
            logger.error(f"Error creating custom tool: {e}")
            return {
                'creation_success': False,
                'error': str(e),
                'test_results': None,
                'registered': False,
                'activated': False
            }
    
    def get_tool_performance_analytics(self, tool_id: Optional[str] = None,
                                     days_back: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive tool performance analytics.
        
        Args:
            tool_id: Optional specific tool ID
            days_back: Number of days to analyze
            
        Returns:
            Performance analytics
        """
        try:
            if tool_id:
                # Get specific tool analysis
                analysis = self.tool_evaluator.analyze_tool_performance(tool_id, days_back)
                scorecard = self.tool_evaluator.get_tool_scorecard(tool_id)
                
                return {
                    'tool_id': tool_id,
                    'analysis': analysis,
                    'scorecard': scorecard.__dict__ if scorecard else None,
                    'execution_stats': self.secure_executor.get_execution_stats(days_back=days_back)
                }
            else:
                # Get overall analytics
                rankings = self.tool_evaluator.get_tool_rankings()
                execution_stats = self.secure_executor.get_execution_stats(days_back=days_back)
                
                return {
                    'tool_rankings': rankings,
                    'execution_stats': execution_stats,
                    'total_tools': len(self.tool_registry.tools),
                    'custom_tools': len(self.custom_tool_creator.custom_tools),
                    'analysis_period_days': days_back
                }
            
        except Exception as e:
            logger.error(f"Error getting tool analytics: {e}")
            return {'error': str(e)}
    
    def _execute_plan_with_tools(self, plan_id: str, execution_mode: ExecutionMode) -> List[Dict[str, Any]]:
        """Execute action plan using secure executor."""
        try:
            execution_results = []
            
            # Define execution callback
            def tool_executor(tool_id: str, input_data: Dict[str, Any]) -> Any:
                # Submit execution request
                request_id = self.secure_executor.submit_execution_request(
                    tool_id=tool_id,
                    tool_name=tool_id.replace('_', ' ').title(),
                    input_data=input_data,
                    execution_mode=execution_mode,
                    user_id="system",
                    session_id="auto_execution",
                    requires_approval=False
                )
                
                # Get execution result
                result = self.secure_executor.get_execution_result(request_id)
                
                if result:
                    execution_results.append({
                        'tool_id': tool_id,
                        'request_id': request_id,
                        'success': result.status.value == 'completed',
                        'output': result.output,
                        'execution_time_ms': result.execution_time_ms,
                        'error': result.error_message
                    })
                    
                    return result.output
                else:
                    execution_results.append({
                        'tool_id': tool_id,
                        'request_id': request_id,
                        'success': False,
                        'output': None,
                        'execution_time_ms': 0,
                        'error': 'No execution result'
                    })
                    return None
            
            # Execute the plan
            self.action_planner.execute_plan(plan_id, executor_callback=tool_executor)
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Error executing plan with tools: {e}")
            return []
    
    def _update_tool_metrics(self, execution_results: List[Dict[str, Any]], goal: str):
        """Update tool performance metrics based on execution results."""
        try:
            for result in execution_results:
                tool_id = result['tool_id']
                success = result['success']
                execution_time_ms = result['execution_time_ms']
                
                # Update tool registry stats
                self.tool_registry.update_tool_stats(
                    tool_id=tool_id,
                    success=success,
                    execution_time=execution_time_ms / 1000.0  # Convert to seconds
                )
                
                # Create evaluation
                self.tool_evaluator.evaluate_execution(
                    tool_id=tool_id,
                    execution_id=result['request_id'],
                    request_id=result['request_id'],
                    user_id="system",
                    goal=goal,
                    input_data={},
                    output_data=result['output'],
                    success=success,
                    execution_time_ms=execution_time_ms
                )
            
        except Exception as e:
            logger.error(f"Error updating tool metrics: {e}")
    
    def _generate_insights(self, execution_plan, execution_results: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from execution plan and results."""
        insights = []
        
        try:
            if execution_results:
                success_rate = sum(1 for r in execution_results if r['success']) / len(execution_results)
                
                if success_rate == 1.0:
                    insights.append("All tools executed successfully")
                elif success_rate > 0.7:
                    insights.append("Most tools executed successfully with minor issues")
                else:
                    insights.append("Execution encountered significant challenges")
                
                # Analyze execution times
                avg_time = sum(r['execution_time_ms'] for r in execution_results) / len(execution_results)
                if avg_time > 10000:  # > 10 seconds
                    insights.append("Execution times were higher than expected")
                
                # Identify best performing tools
                successful_tools = [r['tool_id'] for r in execution_results if r['success']]
                if successful_tools:
                    insights.append(f"Most reliable tools: {', '.join(set(successful_tools))}")
            
            # Plan-level insights
            if execution_plan.confidence > 0.8:
                insights.append("High confidence in tool selection and sequencing")
            elif execution_plan.confidence < 0.5:
                insights.append("Tool selection had lower confidence - consider alternatives")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Unable to generate detailed insights due to analysis error")
        
        return insights
    
    def _create_error_response(self, request: ToolRequest, error_message: str) -> ToolResponse:
        """Create an error response."""
        return ToolResponse(
            request_id=request.request_id,
            plan_id="",
            execution_results=[],
            success_rate=0.0,
            total_execution_time=0,
            tools_used=[],
            insights_generated=[f"Error: {error_message}"],
            report_content=f"Error processing tool request: {error_message}",
            created_at=datetime.now().isoformat()
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                'tool_registry': {
                    'total_tools': len(self.tool_registry.tools),
                    'available': True
                },
                'secure_executor': {
                    'pending_requests': len(self.secure_executor.pending_requests),
                    'available': True
                },
                'tool_evaluator': {
                    'total_scorecards': len(self.tool_evaluator.scorecards),
                    'available': True
                },
                'custom_tool_creator': {
                    'total_custom_tools': len(self.custom_tool_creator.custom_tools),
                    'available_templates': len(self.custom_tool_creator.templates),
                    'available': True
                },
                'action_planner': {
                    'total_plans': len(self.action_planner.action_plans),
                    'available': True
                },
                'system_ready': True
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e), 'system_ready': False}

# Global integrated tool system instance
_integrated_tool_system = None

def get_integrated_tool_system(approval_callback: Optional[Callable] = None) -> IntegratedToolSystem:
    """Get or create a global integrated tool system instance."""
    global _integrated_tool_system
    
    if _integrated_tool_system is None:
        _integrated_tool_system = IntegratedToolSystem(approval_callback=approval_callback)
    
    return _integrated_tool_system
