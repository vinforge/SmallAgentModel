"""
Action Plan Generation & Reporting for SAM
Provides structured, explainable plans before taking action with comprehensive reporting.

Sprint 8 Task 5: Action Plan Generation & Reporting
"""

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class PlanStatus(Enum):
    """Status of action plans."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepStatus(Enum):
    """Status of individual plan steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ReportFormat(Enum):
    """Report output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    TEXT = "text"

@dataclass
class ActionStep:
    """A single step in an action plan."""
    step_id: str
    step_number: int
    title: str
    description: str
    tool_id: str
    tool_name: str
    input_data: Dict[str, Any]
    expected_output: str
    dependencies: List[str]
    estimated_duration: int
    status: StepStatus
    actual_output: Optional[Any]
    execution_time: Optional[int]
    error_message: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class ActionPlan:
    """A complete action plan."""
    plan_id: str
    title: str
    description: str
    goal: str
    user_id: str
    session_id: str
    steps: List[ActionStep]
    status: PlanStatus
    estimated_total_duration: int
    actual_total_duration: Optional[int]
    success_rate: float
    created_at: str
    approved_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    approval_notes: str
    execution_notes: str
    metadata: Dict[str, Any]

@dataclass
class ExecutionReport:
    """Report of plan execution."""
    report_id: str
    plan_id: str
    title: str
    summary: str
    steps_executed: int
    steps_successful: int
    total_execution_time: int
    key_insights: List[str]
    outputs_generated: List[Dict[str, Any]]
    issues_encountered: List[str]
    recommendations: List[str]
    created_at: str
    metadata: Dict[str, Any]

class ActionPlanner:
    """
    Generates structured action plans and comprehensive execution reports.
    """
    
    def __init__(self, reports_directory: str = "task_run_reports"):
        """
        Initialize the action planner.
        
        Args:
            reports_directory: Directory for storing execution reports
        """
        self.reports_dir = Path(reports_directory)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Storage
        self.action_plans: Dict[str, ActionPlan] = {}
        self.execution_reports: Dict[str, ExecutionReport] = {}
        
        # Configuration
        self.config = {
            'require_approval_for_critical': True,
            'auto_generate_reports': True,
            'max_plan_steps': 20,
            'default_step_timeout': 300  # 5 minutes
        }
        
        logger.info(f"Action planner initialized with reports in {reports_directory}")
    
    def create_action_plan(self, goal: str, user_id: str, session_id: str,
                          tool_sequence: List[Dict[str, Any]],
                          title: Optional[str] = None,
                          description: Optional[str] = None) -> str:
        """
        Create a new action plan.
        
        Args:
            goal: The goal to achieve
            user_id: User creating the plan
            session_id: Session ID
            tool_sequence: Sequence of tools to use
            title: Optional plan title
            description: Optional plan description
            
        Returns:
            Plan ID
        """
        try:
            plan_id = f"plan_{uuid.uuid4().hex[:12]}"
            
            # Generate title and description if not provided
            if not title:
                title = f"Action Plan: {goal[:50]}..."
            if not description:
                description = f"Automated action plan to achieve: {goal}"
            
            # Create action steps
            steps = []
            total_duration = 0
            
            for i, tool_config in enumerate(tool_sequence):
                step_id = f"{plan_id}_step_{i+1}"
                
                step = ActionStep(
                    step_id=step_id,
                    step_number=i + 1,
                    title=tool_config.get('title', f"Step {i+1}"),
                    description=tool_config.get('description', f"Execute {tool_config.get('tool_name', 'tool')}"),
                    tool_id=tool_config['tool_id'],
                    tool_name=tool_config.get('tool_name', tool_config['tool_id']),
                    input_data=tool_config.get('input_data', {}),
                    expected_output=tool_config.get('expected_output', 'Tool execution result'),
                    dependencies=tool_config.get('dependencies', []),
                    estimated_duration=tool_config.get('estimated_duration', 30),
                    status=StepStatus.PENDING,
                    actual_output=None,
                    execution_time=None,
                    error_message=None,
                    started_at=None,
                    completed_at=None,
                    metadata=tool_config.get('metadata', {})
                )
                
                steps.append(step)
                total_duration += step.estimated_duration
            
            # Determine if approval is required
            requires_approval = any(
                step.metadata.get('requires_approval', False) for step in steps
            )
            
            initial_status = PlanStatus.PENDING_APPROVAL if requires_approval else PlanStatus.APPROVED
            
            # Create action plan
            action_plan = ActionPlan(
                plan_id=plan_id,
                title=title,
                description=description,
                goal=goal,
                user_id=user_id,
                session_id=session_id,
                steps=steps,
                status=initial_status,
                estimated_total_duration=total_duration,
                actual_total_duration=None,
                success_rate=0.0,
                created_at=datetime.now().isoformat(),
                approved_at=None if requires_approval else datetime.now().isoformat(),
                started_at=None,
                completed_at=None,
                approval_notes="",
                execution_notes="",
                metadata={}
            )
            
            self.action_plans[plan_id] = action_plan
            
            logger.info(f"Created action plan: {title} ({plan_id}) with {len(steps)} steps")
            return plan_id
            
        except Exception as e:
            logger.error(f"Error creating action plan: {e}")
            raise
    
    def get_action_plan(self, plan_id: str) -> Optional[ActionPlan]:
        """Get an action plan by ID."""
        return self.action_plans.get(plan_id)
    
    def approve_plan(self, plan_id: str, approved: bool = True, 
                    approval_notes: str = "") -> bool:
        """
        Approve or reject an action plan.
        
        Args:
            plan_id: Plan ID
            approved: Whether to approve the plan
            approval_notes: Notes about the approval decision
            
        Returns:
            True if successful, False otherwise
        """
        try:
            plan = self.action_plans.get(plan_id)
            if not plan:
                logger.error(f"Action plan not found: {plan_id}")
                return False
            
            if approved:
                plan.status = PlanStatus.APPROVED
                plan.approved_at = datetime.now().isoformat()
                logger.info(f"Approved action plan: {plan_id}")
            else:
                plan.status = PlanStatus.CANCELLED
                logger.info(f"Rejected action plan: {plan_id}")
            
            plan.approval_notes = approval_notes
            
            return True
            
        except Exception as e:
            logger.error(f"Error approving plan {plan_id}: {e}")
            return False
    
    def modify_plan(self, plan_id: str, modifications: Dict[str, Any]) -> bool:
        """
        Modify an action plan before execution.
        
        Args:
            plan_id: Plan ID
            modifications: Modifications to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            plan = self.action_plans.get(plan_id)
            if not plan:
                logger.error(f"Action plan not found: {plan_id}")
                return False
            
            if plan.status not in [PlanStatus.DRAFT, PlanStatus.PENDING_APPROVAL]:
                logger.error(f"Cannot modify plan in status: {plan.status}")
                return False
            
            # Apply modifications
            if 'title' in modifications:
                plan.title = modifications['title']
            
            if 'description' in modifications:
                plan.description = modifications['description']
            
            if 'step_modifications' in modifications:
                step_mods = modifications['step_modifications']
                for step_id, step_changes in step_mods.items():
                    step = next((s for s in plan.steps if s.step_id == step_id), None)
                    if step:
                        for key, value in step_changes.items():
                            if hasattr(step, key):
                                setattr(step, key, value)
            
            if 'reorder_steps' in modifications:
                new_order = modifications['reorder_steps']
                reordered_steps = []
                for step_id in new_order:
                    step = next((s for s in plan.steps if s.step_id == step_id), None)
                    if step:
                        reordered_steps.append(step)
                
                # Update step numbers
                for i, step in enumerate(reordered_steps):
                    step.step_number = i + 1
                
                plan.steps = reordered_steps
            
            logger.info(f"Modified action plan: {plan_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying plan {plan_id}: {e}")
            return False
    
    def execute_plan(self, plan_id: str, executor_callback: Optional[callable] = None) -> bool:
        """
        Execute an action plan.
        
        Args:
            plan_id: Plan ID to execute
            executor_callback: Optional callback for tool execution
            
        Returns:
            True if execution started successfully, False otherwise
        """
        try:
            plan = self.action_plans.get(plan_id)
            if not plan:
                logger.error(f"Action plan not found: {plan_id}")
                return False
            
            if plan.status != PlanStatus.APPROVED:
                logger.error(f"Plan not approved for execution: {plan_id}")
                return False
            
            plan.status = PlanStatus.EXECUTING
            plan.started_at = datetime.now().isoformat()
            
            logger.info(f"Starting execution of action plan: {plan.title} ({plan_id})")
            
            # Execute steps in order
            successful_steps = 0
            
            for step in plan.steps:
                if self._execute_step(step, executor_callback):
                    successful_steps += 1
                else:
                    # Stop execution on failure (could be configurable)
                    break
            
            # Update plan status
            plan.completed_at = datetime.now().isoformat()
            plan.success_rate = successful_steps / len(plan.steps) if plan.steps else 0.0
            
            if plan.success_rate == 1.0:
                plan.status = PlanStatus.COMPLETED
            else:
                plan.status = PlanStatus.FAILED
            
            # Calculate total execution time
            if plan.started_at:
                start_time = datetime.fromisoformat(plan.started_at)
                end_time = datetime.now()
                plan.actual_total_duration = int((end_time - start_time).total_seconds())
            
            # Generate execution report
            if self.config['auto_generate_reports']:
                self._generate_execution_report(plan)
            
            logger.info(f"Completed execution of plan {plan_id}: {successful_steps}/{len(plan.steps)} steps successful")
            return True
            
        except Exception as e:
            logger.error(f"Error executing plan {plan_id}: {e}")
            return False
    
    def generate_plan_blueprint(self, plan_id: str) -> str:
        """
        Generate a human-readable blueprint of the action plan.
        
        Args:
            plan_id: Plan ID
            
        Returns:
            Blueprint text
        """
        try:
            plan = self.action_plans.get(plan_id)
            if not plan:
                return f"Action plan not found: {plan_id}"
            
            blueprint_parts = [
                f"# Action Plan Blueprint: {plan.title}\n",
                f"**Goal:** {plan.goal}\n",
                f"**Description:** {plan.description}\n",
                f"**Estimated Duration:** {plan.estimated_total_duration} seconds\n",
                f"**Total Steps:** {len(plan.steps)}\n\n",
                "## Execution Steps\n"
            ]
            
            for step in plan.steps:
                status_emoji = {
                    StepStatus.PENDING: "⏳",
                    StepStatus.RUNNING: "🔄",
                    StepStatus.COMPLETED: "✅",
                    StepStatus.FAILED: "❌",
                    StepStatus.SKIPPED: "⏭️"
                }.get(step.status, "❓")
                
                blueprint_parts.extend([
                    f"### {status_emoji} Step {step.step_number}: {step.title}\n",
                    f"**Tool:** {step.tool_name}\n",
                    f"**Description:** {step.description}\n",
                    f"**Expected Output:** {step.expected_output}\n",
                    f"**Estimated Duration:** {step.estimated_duration} seconds\n"
                ])
                
                if step.dependencies:
                    blueprint_parts.append(f"**Dependencies:** {', '.join(step.dependencies)}\n")
                
                blueprint_parts.append("\n")
            
            return "".join(blueprint_parts)
            
        except Exception as e:
            logger.error(f"Error generating blueprint for plan {plan_id}: {e}")
            return f"Error generating blueprint: {str(e)}"
    
    def generate_execution_report(self, plan_id: str, format: ReportFormat = ReportFormat.MARKDOWN) -> str:
        """
        Generate a comprehensive execution report.
        
        Args:
            plan_id: Plan ID
            format: Report format
            
        Returns:
            Report content
        """
        try:
            plan = self.action_plans.get(plan_id)
            if not plan:
                return f"Action plan not found: {plan_id}"
            
            if format == ReportFormat.MARKDOWN:
                return self._generate_markdown_report(plan)
            elif format == ReportFormat.HTML:
                return self._generate_html_report(plan)
            elif format == ReportFormat.JSON:
                return self._generate_json_report(plan)
            else:
                return self._generate_text_report(plan)
            
        except Exception as e:
            logger.error(f"Error generating execution report: {e}")
            return f"Error generating report: {str(e)}"
    
    def _execute_step(self, step: ActionStep, executor_callback: Optional[callable]) -> bool:
        """Execute a single step."""
        try:
            step.status = StepStatus.RUNNING
            step.started_at = datetime.now().isoformat()
            
            logger.info(f"Executing step: {step.title} ({step.step_id})")
            
            # Execute using callback or simulate
            if executor_callback:
                result = executor_callback(step.tool_id, step.input_data)
            else:
                # Simulate execution
                result = self._simulate_step_execution(step)
            
            step.actual_output = result
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now().isoformat()
            
            # Calculate execution time
            if step.started_at:
                start_time = datetime.fromisoformat(step.started_at)
                end_time = datetime.now()
                step.execution_time = int((end_time - start_time).total_seconds())
            
            logger.info(f"Step completed: {step.title}")
            return True
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            step.completed_at = datetime.now().isoformat()
            
            logger.error(f"Step failed: {step.title} - {str(e)}")
            return False
    
    def _simulate_step_execution(self, step: ActionStep) -> Dict[str, Any]:
        """Simulate step execution."""
        return {
            'simulated': True,
            'tool_id': step.tool_id,
            'input_processed': step.input_data,
            'step_title': step.title,
            'execution_time': step.estimated_duration
        }
    
    def _generate_execution_report(self, plan: ActionPlan):
        """Generate and save execution report."""
        try:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
            
            # Analyze execution
            successful_steps = sum(1 for step in plan.steps if step.status == StepStatus.COMPLETED)
            
            # Extract key insights
            key_insights = []
            if plan.success_rate == 1.0:
                key_insights.append("All steps completed successfully")
            elif plan.success_rate > 0.5:
                key_insights.append("Majority of steps completed successfully")
            else:
                key_insights.append("Execution encountered significant issues")
            
            # Collect outputs
            outputs_generated = []
            for step in plan.steps:
                if step.actual_output:
                    outputs_generated.append({
                        'step': step.title,
                        'output': step.actual_output
                    })
            
            # Identify issues
            issues_encountered = []
            for step in plan.steps:
                if step.status == StepStatus.FAILED:
                    issues_encountered.append(f"Step {step.step_number} failed: {step.error_message}")
            
            # Generate recommendations
            recommendations = []
            if plan.success_rate < 1.0:
                recommendations.append("Review failed steps and consider alternative approaches")
            if plan.actual_total_duration and plan.actual_total_duration > plan.estimated_total_duration * 1.5:
                recommendations.append("Consider optimizing execution time for future runs")
            
            report = ExecutionReport(
                report_id=report_id,
                plan_id=plan.plan_id,
                title=f"Execution Report: {plan.title}",
                summary=f"Executed {successful_steps}/{len(plan.steps)} steps successfully",
                steps_executed=len(plan.steps),
                steps_successful=successful_steps,
                total_execution_time=plan.actual_total_duration or 0,
                key_insights=key_insights,
                outputs_generated=outputs_generated,
                issues_encountered=issues_encountered,
                recommendations=recommendations,
                created_at=datetime.now().isoformat(),
                metadata={'plan_goal': plan.goal}
            )
            
            self.execution_reports[report_id] = report
            
            # Save report to file
            self._save_report(report)
            
            logger.info(f"Generated execution report: {report_id}")
            
        except Exception as e:
            logger.error(f"Error generating execution report: {e}")
    
    def _generate_markdown_report(self, plan: ActionPlan) -> str:
        """Generate markdown format report."""
        successful_steps = sum(1 for step in plan.steps if step.status == StepStatus.COMPLETED)
        
        report_parts = [
            f"# Execution Report: {plan.title}\n\n",
            f"**Plan ID:** {plan.plan_id}\n",
            f"**Goal:** {plan.goal}\n",
            f"**Status:** {plan.status.value}\n",
            f"**Success Rate:** {plan.success_rate:.1%}\n",
            f"**Steps Completed:** {successful_steps}/{len(plan.steps)}\n",
            f"**Total Execution Time:** {plan.actual_total_duration or 0} seconds\n",
            f"**Created:** {plan.created_at}\n",
            f"**Completed:** {plan.completed_at or 'Not completed'}\n\n",
            "## Step Details\n\n"
        ]
        
        for step in plan.steps:
            status_emoji = {
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.RUNNING: "🔄",
                StepStatus.PENDING: "⏳",
                StepStatus.SKIPPED: "⏭️"
            }.get(step.status, "❓")
            
            report_parts.extend([
                f"### {status_emoji} Step {step.step_number}: {step.title}\n",
                f"**Tool:** {step.tool_name}\n",
                f"**Status:** {step.status.value}\n",
                f"**Execution Time:** {step.execution_time or 0} seconds\n"
            ])
            
            if step.error_message:
                report_parts.append(f"**Error:** {step.error_message}\n")
            
            if step.actual_output:
                report_parts.append(f"**Output:** {str(step.actual_output)[:200]}...\n")
            
            report_parts.append("\n")
        
        return "".join(report_parts)
    
    def _generate_html_report(self, plan: ActionPlan) -> str:
        """Generate HTML format report."""
        # Simplified HTML generation
        markdown_content = self._generate_markdown_report(plan)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Execution Report: {plan.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        .status-completed {{ color: green; }}
        .status-failed {{ color: red; }}
        .status-pending {{ color: orange; }}
    </style>
</head>
<body>
    <pre>{markdown_content}</pre>
</body>
</html>
"""
        return html_content
    
    def _generate_json_report(self, plan: ActionPlan) -> str:
        """Generate JSON format report."""
        report_data = {
            'plan_id': plan.plan_id,
            'title': plan.title,
            'goal': plan.goal,
            'status': plan.status.value,
            'success_rate': plan.success_rate,
            'total_steps': len(plan.steps),
            'successful_steps': sum(1 for step in plan.steps if step.status == StepStatus.COMPLETED),
            'execution_time': plan.actual_total_duration,
            'created_at': plan.created_at,
            'completed_at': plan.completed_at,
            'steps': [
                {
                    'step_number': step.step_number,
                    'title': step.title,
                    'tool_name': step.tool_name,
                    'status': step.status.value,
                    'execution_time': step.execution_time,
                    'error_message': step.error_message,
                    'output': step.actual_output
                }
                for step in plan.steps
            ]
        }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def _generate_text_report(self, plan: ActionPlan) -> str:
        """Generate plain text format report."""
        successful_steps = sum(1 for step in plan.steps if step.status == StepStatus.COMPLETED)
        
        report_lines = [
            f"EXECUTION REPORT: {plan.title}",
            "=" * 50,
            f"Plan ID: {plan.plan_id}",
            f"Goal: {plan.goal}",
            f"Status: {plan.status.value}",
            f"Success Rate: {plan.success_rate:.1%}",
            f"Steps Completed: {successful_steps}/{len(plan.steps)}",
            f"Total Execution Time: {plan.actual_total_duration or 0} seconds",
            "",
            "STEP DETAILS:",
            "-" * 20
        ]
        
        for step in plan.steps:
            status_symbol = "✓" if step.status == StepStatus.COMPLETED else "✗" if step.status == StepStatus.FAILED else "○"
            
            report_lines.extend([
                f"{status_symbol} Step {step.step_number}: {step.title}",
                f"   Tool: {step.tool_name}",
                f"   Status: {step.status.value}",
                f"   Time: {step.execution_time or 0}s",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def _save_report(self, report: ExecutionReport):
        """Save execution report to file."""
        try:
            # Save as markdown
            report_file = self.reports_dir / f"{report.report_id}.md"
            
            plan = self.action_plans.get(report.plan_id)
            if plan:
                markdown_content = self._generate_markdown_report(plan)
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                logger.debug(f"Saved execution report: {report.report_id}")
            
        except Exception as e:
            logger.error(f"Error saving execution report: {e}")

# Global action planner instance
_action_planner = None

def get_action_planner(reports_directory: str = "task_run_reports") -> ActionPlanner:
    """Get or create a global action planner instance."""
    global _action_planner
    
    if _action_planner is None:
        _action_planner = ActionPlanner(reports_directory=reports_directory)
    
    return _action_planner
