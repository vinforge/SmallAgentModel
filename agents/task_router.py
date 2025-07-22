"""
Multi-Agent Task Routing Engine for SAM
Breaks complex requests into sub-tasks and delegates them to specialized agents.

Sprint 10 Task 1: Multi-Agent Task Routing Engine
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

class AgentRole(Enum):
    """Roles that agents can take."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    SPECIALIST = "specialist"

class TaskType(Enum):
    """Types of tasks that can be delegated."""
    DOCUMENT_ANALYSIS = "document_analysis"
    DATA_PROCESSING = "data_processing"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"

class TaskStatus(Enum):
    """Status of delegated tasks."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SubTask:
    """A sub-task that can be delegated to an agent."""
    task_id: str
    task_type: TaskType
    description: str
    input_data: Dict[str, Any]
    required_role: AgentRole
    priority: int
    dependencies: List[str]
    estimated_duration: int
    assigned_agent: Optional[str]
    status: TaskStatus
    result: Optional[Any]
    error_message: Optional[str]
    created_at: str
    assigned_at: Optional[str]
    completed_at: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class TaskPlan:
    """A complete task decomposition plan."""
    plan_id: str
    original_request: str
    user_id: str
    session_id: str
    sub_tasks: List[SubTask]
    execution_order: List[List[str]]  # Groups of tasks that can run in parallel
    status: TaskStatus
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    metadata: Dict[str, Any]

class TaskRouter:
    """
    Routes complex tasks to appropriate agents based on roles and capabilities.
    """
    
    def __init__(self, routing_config_file: str = "task_routing_config.json"):
        """
        Initialize the task router.
        
        Args:
            routing_config_file: Path to routing configuration file
        """
        self.routing_config_file = Path(routing_config_file)
        
        # Storage
        self.task_plans: Dict[str, TaskPlan] = {}
        self.active_tasks: Dict[str, SubTask] = {}
        
        # Configuration
        self.config = {
            'max_parallel_tasks': 5,
            'task_timeout_seconds': 300,
            'enable_dependency_checking': True,
            'auto_retry_failed_tasks': True,
            'max_retries': 3
        }
        
        # Task decomposition patterns
        self.decomposition_patterns = self._initialize_decomposition_patterns()
        
        # Role capabilities mapping
        self.role_capabilities = self._initialize_role_capabilities()
        
        logger.info("Task router initialized")
    
    def decompose_task(self, request: str, user_id: str, session_id: str,
                      context: Dict[str, Any] = None) -> TaskPlan:
        """
        Decompose a complex request into sub-tasks.
        
        Args:
            request: The original user request
            user_id: User making the request
            session_id: Session ID
            context: Additional context for decomposition
            
        Returns:
            TaskPlan with sub-tasks and execution order
        """
        try:
            plan_id = f"plan_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Decomposing task: {request[:50]}...")
            
            # Analyze request to identify task types
            task_types = self._analyze_request_types(request)
            
            # Generate sub-tasks based on patterns
            sub_tasks = self._generate_sub_tasks(request, task_types, context)
            
            # Determine execution order and dependencies
            execution_order = self._determine_execution_order(sub_tasks)
            
            # Create task plan
            task_plan = TaskPlan(
                plan_id=plan_id,
                original_request=request,
                user_id=user_id,
                session_id=session_id,
                sub_tasks=sub_tasks,
                execution_order=execution_order,
                status=TaskStatus.PENDING,
                created_at=datetime.now().isoformat(),
                started_at=None,
                completed_at=None,
                metadata=context or {}
            )
            
            self.task_plans[plan_id] = task_plan
            
            logger.info(f"Task decomposed: {len(sub_tasks)} sub-tasks, "
                       f"{len(execution_order)} execution phases")
            
            return task_plan
            
        except Exception as e:
            logger.error(f"Error decomposing task: {e}")
            raise
    
    def assign_task_to_agent(self, task_id: str, agent_id: str) -> bool:
        """
        Assign a sub-task to a specific agent.
        
        Args:
            task_id: Task ID to assign
            agent_id: Agent ID to assign to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find the task
            task = None
            for plan in self.task_plans.values():
                for sub_task in plan.sub_tasks:
                    if sub_task.task_id == task_id:
                        task = sub_task
                        break
                if task:
                    break
            
            if not task:
                logger.error(f"Task not found: {task_id}")
                return False
            
            if task.status != TaskStatus.PENDING:
                logger.error(f"Task {task_id} is not in pending status: {task.status}")
                return False
            
            # Assign task
            task.assigned_agent = agent_id
            task.status = TaskStatus.ASSIGNED
            task.assigned_at = datetime.now().isoformat()
            
            # Add to active tasks
            self.active_tasks[task_id] = task
            
            logger.info(f"Assigned task {task_id} to agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error assigning task {task_id} to agent {agent_id}: {e}")
            return False
    
    def get_available_tasks(self, agent_role: AgentRole, 
                           agent_capabilities: List[str] = None) -> List[SubTask]:
        """
        Get tasks available for assignment to an agent with specific role.
        
        Args:
            agent_role: Role of the requesting agent
            agent_capabilities: Optional list of agent capabilities
            
        Returns:
            List of available tasks
        """
        try:
            available_tasks = []
            
            for plan in self.task_plans.values():
                if plan.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                    for task in plan.sub_tasks:
                        if (task.status == TaskStatus.PENDING and 
                            task.required_role == agent_role and
                            self._check_dependencies_met(task, plan)):
                            
                            # Check capabilities if specified
                            if agent_capabilities:
                                required_caps = task.metadata.get('required_capabilities', [])
                                if required_caps and not any(cap in agent_capabilities for cap in required_caps):
                                    continue
                            
                            available_tasks.append(task)
            
            # Sort by priority (higher priority first)
            available_tasks.sort(key=lambda t: t.priority, reverse=True)
            
            return available_tasks
            
        except Exception as e:
            logger.error(f"Error getting available tasks: {e}")
            return []
    
    def update_task_status(self, task_id: str, status: TaskStatus,
                          result: Any = None, error_message: str = None) -> bool:
        """
        Update the status of a task.
        
        Args:
            task_id: Task ID to update
            status: New status
            result: Task result if completed
            error_message: Error message if failed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            task = self.active_tasks.get(task_id)
            if not task:
                logger.error(f"Active task not found: {task_id}")
                return False
            
            task.status = status
            
            if status == TaskStatus.COMPLETED:
                task.result = result
                task.completed_at = datetime.now().isoformat()
                # Remove from active tasks
                del self.active_tasks[task_id]
                
            elif status == TaskStatus.FAILED:
                task.error_message = error_message
                task.completed_at = datetime.now().isoformat()
                # Remove from active tasks
                del self.active_tasks[task_id]
            
            # Check if plan is complete
            self._check_plan_completion(task_id)
            
            logger.info(f"Updated task {task_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            return False
    
    def get_task_plan(self, plan_id: str) -> Optional[TaskPlan]:
        """Get a task plan by ID."""
        return self.task_plans.get(plan_id)
    
    def get_plan_progress(self, plan_id: str) -> Dict[str, Any]:
        """
        Get progress information for a task plan.
        
        Args:
            plan_id: Plan ID
            
        Returns:
            Progress information
        """
        try:
            plan = self.task_plans.get(plan_id)
            if not plan:
                return {'error': 'Plan not found'}
            
            total_tasks = len(plan.sub_tasks)
            completed_tasks = sum(1 for task in plan.sub_tasks if task.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for task in plan.sub_tasks if task.status == TaskStatus.FAILED)
            in_progress_tasks = sum(1 for task in plan.sub_tasks if task.status == TaskStatus.IN_PROGRESS)
            
            progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            return {
                'plan_id': plan_id,
                'status': plan.status.value,
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'in_progress_tasks': in_progress_tasks,
                'progress_percentage': progress_percentage,
                'created_at': plan.created_at,
                'started_at': plan.started_at,
                'completed_at': plan.completed_at
            }
            
        except Exception as e:
            logger.error(f"Error getting plan progress: {e}")
            return {'error': str(e)}
    
    def _analyze_request_types(self, request: str) -> List[TaskType]:
        """Analyze request to identify required task types."""
        task_types = []
        request_lower = request.lower()
        
        # Document analysis patterns
        if any(word in request_lower for word in ['pdf', 'document', 'file', 'read']):
            task_types.append(TaskType.DOCUMENT_ANALYSIS)
        
        # Summarization patterns
        if any(word in request_lower for word in ['summarize', 'summary', 'brief']):
            task_types.append(TaskType.SUMMARIZATION)
        
        # Comparison patterns
        if any(word in request_lower for word in ['compare', 'comparison', 'difference', 'contrast']):
            task_types.append(TaskType.COMPARISON)
        
        # Research patterns
        if any(word in request_lower for word in ['research', 'find', 'search', 'investigate']):
            task_types.append(TaskType.RESEARCH)
        
        # Data processing patterns
        if any(word in request_lower for word in ['analyze', 'process', 'calculate', 'data']):
            task_types.append(TaskType.DATA_PROCESSING)
        
        # Synthesis patterns
        if any(word in request_lower for word in ['combine', 'merge', 'synthesize', 'integrate']):
            task_types.append(TaskType.SYNTHESIS)
        
        return task_types if task_types else [TaskType.DATA_PROCESSING]
    
    def _generate_sub_tasks(self, request: str, task_types: List[TaskType],
                           context: Dict[str, Any] = None) -> List[SubTask]:
        """Generate sub-tasks based on request and task types."""
        sub_tasks = []
        
        try:
            # Use decomposition patterns to generate tasks
            for i, task_type in enumerate(task_types):
                pattern = self.decomposition_patterns.get(task_type, {})
                
                task_id = f"task_{uuid.uuid4().hex[:8]}"
                
                sub_task = SubTask(
                    task_id=task_id,
                    task_type=task_type,
                    description=pattern.get('description', f"Execute {task_type.value} task"),
                    input_data={'request': request, 'context': context or {}},
                    required_role=pattern.get('required_role', AgentRole.EXECUTOR),
                    priority=pattern.get('priority', 5),
                    dependencies=[],
                    estimated_duration=pattern.get('estimated_duration', 60),
                    assigned_agent=None,
                    status=TaskStatus.PENDING,
                    result=None,
                    error_message=None,
                    created_at=datetime.now().isoformat(),
                    assigned_at=None,
                    completed_at=None,
                    metadata=pattern.get('metadata', {})
                )
                
                sub_tasks.append(sub_task)
            
            # Add synthesis task if multiple tasks
            if len(sub_tasks) > 1:
                synthesis_task = SubTask(
                    task_id=f"synthesis_{uuid.uuid4().hex[:8]}",
                    task_type=TaskType.SYNTHESIS,
                    description="Synthesize results from all sub-tasks",
                    input_data={'request': request},
                    required_role=AgentRole.SYNTHESIZER,
                    priority=1,  # Lower priority, runs last
                    dependencies=[task.task_id for task in sub_tasks],
                    estimated_duration=30,
                    assigned_agent=None,
                    status=TaskStatus.PENDING,
                    result=None,
                    error_message=None,
                    created_at=datetime.now().isoformat(),
                    assigned_at=None,
                    completed_at=None,
                    metadata={'synthesis_task': True}
                )
                sub_tasks.append(synthesis_task)
            
            return sub_tasks
            
        except Exception as e:
            logger.error(f"Error generating sub-tasks: {e}")
            return []
    
    def _determine_execution_order(self, sub_tasks: List[SubTask]) -> List[List[str]]:
        """Determine execution order based on dependencies."""
        try:
            execution_order = []
            remaining_tasks = {task.task_id: task for task in sub_tasks}
            completed_tasks = set()
            
            while remaining_tasks:
                # Find tasks with no unmet dependencies
                ready_tasks = []
                
                for task_id, task in remaining_tasks.items():
                    if all(dep in completed_tasks for dep in task.dependencies):
                        ready_tasks.append(task_id)
                
                if not ready_tasks:
                    # Break circular dependencies by taking highest priority task
                    ready_tasks = [max(remaining_tasks.keys(), 
                                     key=lambda tid: remaining_tasks[tid].priority)]
                
                execution_order.append(ready_tasks)
                
                # Mark tasks as completed for dependency checking
                for task_id in ready_tasks:
                    completed_tasks.add(task_id)
                    del remaining_tasks[task_id]
            
            return execution_order
            
        except Exception as e:
            logger.error(f"Error determining execution order: {e}")
            return [[task.task_id for task in sub_tasks]]
    
    def _check_dependencies_met(self, task: SubTask, plan: TaskPlan) -> bool:
        """Check if all dependencies for a task are met."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            dep_task = next((t for t in plan.sub_tasks if t.task_id == dep_id), None)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _check_plan_completion(self, task_id: str):
        """Check if a plan is complete after task update."""
        try:
            # Find the plan containing this task
            plan = None
            for p in self.task_plans.values():
                if any(t.task_id == task_id for t in p.sub_tasks):
                    plan = p
                    break
            
            if not plan:
                return
            
            # Check if all tasks are complete
            all_complete = all(
                task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                for task in plan.sub_tasks
            )
            
            if all_complete:
                plan.status = TaskStatus.COMPLETED
                plan.completed_at = datetime.now().isoformat()
                logger.info(f"Plan {plan.plan_id} completed")
            
        except Exception as e:
            logger.error(f"Error checking plan completion: {e}")
    
    def _initialize_decomposition_patterns(self) -> Dict[TaskType, Dict[str, Any]]:
        """Initialize task decomposition patterns."""
        return {
            TaskType.DOCUMENT_ANALYSIS: {
                'description': 'Analyze and extract information from documents',
                'required_role': AgentRole.EXECUTOR,
                'priority': 8,
                'estimated_duration': 120,
                'metadata': {'required_capabilities': ['document_processing']}
            },
            TaskType.SUMMARIZATION: {
                'description': 'Create summary of content',
                'required_role': AgentRole.EXECUTOR,
                'priority': 6,
                'estimated_duration': 60,
                'metadata': {'required_capabilities': ['text_processing']}
            },
            TaskType.COMPARISON: {
                'description': 'Compare and contrast multiple sources',
                'required_role': AgentRole.SYNTHESIZER,
                'priority': 4,
                'estimated_duration': 90,
                'metadata': {'required_capabilities': ['analysis']}
            },
            TaskType.RESEARCH: {
                'description': 'Research and gather information',
                'required_role': AgentRole.EXECUTOR,
                'priority': 7,
                'estimated_duration': 180,
                'metadata': {'required_capabilities': ['search', 'research']}
            },
            TaskType.SYNTHESIS: {
                'description': 'Synthesize results from multiple sources',
                'required_role': AgentRole.SYNTHESIZER,
                'priority': 2,
                'estimated_duration': 60,
                'metadata': {'required_capabilities': ['synthesis']}
            },
            TaskType.VALIDATION: {
                'description': 'Validate and verify results',
                'required_role': AgentRole.VALIDATOR,
                'priority': 3,
                'estimated_duration': 45,
                'metadata': {'required_capabilities': ['validation']}
            }
        }
    
    def _initialize_role_capabilities(self) -> Dict[AgentRole, List[str]]:
        """Initialize role capabilities mapping."""
        return {
            AgentRole.PLANNER: ['task_decomposition', 'planning', 'coordination'],
            AgentRole.EXECUTOR: ['document_processing', 'text_processing', 'search', 'research'],
            AgentRole.VALIDATOR: ['validation', 'verification', 'quality_check'],
            AgentRole.CRITIC: ['critical_analysis', 'evaluation', 'feedback'],
            AgentRole.SYNTHESIZER: ['synthesis', 'analysis', 'integration'],
            AgentRole.SPECIALIST: ['domain_expertise', 'specialized_analysis']
        }

# Global task router instance
_task_router = None

def get_task_router(routing_config_file: str = "task_routing_config.json") -> TaskRouter:
    """Get or create a global task router instance."""
    global _task_router
    
    if _task_router is None:
        _task_router = TaskRouter(routing_config_file=routing_config_file)
    
    return _task_router
