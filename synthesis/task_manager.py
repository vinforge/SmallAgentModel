"""
Multi-Turn Task Chaining for SAM
Enables SAM to carry complex goals across multiple messages.

Sprint 7 Task 2: Multi-Turn Task Chaining
"""

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Status of a task."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"

class SubtaskStatus(Enum):
    """Status of a subtask."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"

@dataclass
class Subtask:
    """A subtask within a larger task."""
    subtask_id: str
    title: str
    description: str
    status: SubtaskStatus
    dependencies: List[str]  # IDs of subtasks that must complete first
    estimated_duration_minutes: int
    actual_duration_minutes: Optional[int]
    tools_required: List[str]
    output_expected: str
    result: Optional[Dict[str, Any]]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class Task:
    """A multi-turn task with subtasks."""
    task_id: str
    title: str
    description: str
    status: TaskStatus
    user_id: str
    session_id: str
    subtasks: List[Subtask]
    current_subtask_index: int
    progress_percentage: float
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    estimated_total_duration_minutes: int
    actual_total_duration_minutes: Optional[int]
    context: Dict[str, Any]  # Persistent context across messages
    metadata: Dict[str, Any]

class TaskManager:
    """
    Manages multi-turn tasks and subtask execution.
    """
    
    def __init__(self, tasks_file: str = "task_manager.json"):
        """
        Initialize the task manager.
        
        Args:
            tasks_file: Path to tasks storage file
        """
        self.tasks_file = Path(tasks_file)
        self.tasks: Dict[str, Task] = {}
        self.active_tasks: Dict[str, str] = {}  # session_id -> task_id
        
        # Load existing tasks
        self._load_tasks()
        
        logger.info(f"Task manager initialized with {len(self.tasks)} tasks")
    
    def create_task(self, title: str, description: str, user_id: str, 
                   session_id: str, subtask_descriptions: List[str],
                   context: Dict[str, Any] = None) -> str:
        """
        Create a new multi-turn task.
        
        Args:
            title: Task title
            description: Task description
            user_id: User ID
            session_id: Session ID
            subtask_descriptions: List of subtask descriptions
            context: Initial context
            
        Returns:
            Task ID
        """
        try:
            task_id = f"task_{uuid.uuid4().hex[:12]}"
            
            # Create subtasks
            subtasks = []
            total_estimated_duration = 0
            
            for i, subtask_desc in enumerate(subtask_descriptions):
                subtask_id = f"{task_id}_sub_{i+1}"
                
                # Estimate duration based on description complexity
                estimated_duration = self._estimate_subtask_duration(subtask_desc)
                total_estimated_duration += estimated_duration
                
                # Determine required tools
                tools_required = self._analyze_required_tools(subtask_desc)
                
                subtask = Subtask(
                    subtask_id=subtask_id,
                    title=f"Step {i+1}",
                    description=subtask_desc,
                    status=SubtaskStatus.PENDING,
                    dependencies=[],  # Simple linear dependencies for now
                    estimated_duration_minutes=estimated_duration,
                    actual_duration_minutes=None,
                    tools_required=tools_required,
                    output_expected=self._determine_expected_output(subtask_desc),
                    result=None,
                    created_at=datetime.now().isoformat(),
                    started_at=None,
                    completed_at=None,
                    metadata={}
                )
                
                # Set dependencies (each subtask depends on the previous one)
                if i > 0:
                    subtask.dependencies = [f"{task_id}_sub_{i}"]
                
                subtasks.append(subtask)
            
            # Create task
            task = Task(
                task_id=task_id,
                title=title,
                description=description,
                status=TaskStatus.PLANNED,
                user_id=user_id,
                session_id=session_id,
                subtasks=subtasks,
                current_subtask_index=0,
                progress_percentage=0.0,
                created_at=datetime.now().isoformat(),
                started_at=None,
                completed_at=None,
                estimated_total_duration_minutes=total_estimated_duration,
                actual_total_duration_minutes=None,
                context=context or {},
                metadata={}
            )
            
            # Store task
            self.tasks[task_id] = task
            self.active_tasks[session_id] = task_id
            
            self._save_tasks()
            
            logger.info(f"Created task: {title} ({task_id}) with {len(subtasks)} subtasks")
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            raise
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_active_task(self, session_id: str) -> Optional[Task]:
        """Get the active task for a session."""
        task_id = self.active_tasks.get(session_id)
        if task_id:
            return self.tasks.get(task_id)
        return None
    
    def start_task(self, task_id: str) -> bool:
        """Start a task."""
        try:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now().isoformat()
            
            # Start first subtask
            if task.subtasks:
                task.subtasks[0].status = SubtaskStatus.IN_PROGRESS
                task.subtasks[0].started_at = datetime.now().isoformat()
            
            self._save_tasks()
            
            logger.info(f"Started task: {task.title} ({task_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error starting task {task_id}: {e}")
            return False
    
    def complete_current_subtask(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Complete the current subtask and move to the next."""
        try:
            task = self.tasks.get(task_id)
            if not task or task.status != TaskStatus.IN_PROGRESS:
                return False
            
            current_subtask = task.subtasks[task.current_subtask_index]
            
            # Complete current subtask
            current_subtask.status = SubtaskStatus.COMPLETED
            current_subtask.completed_at = datetime.now().isoformat()
            current_subtask.result = result
            
            # Calculate actual duration
            if current_subtask.started_at:
                start_time = datetime.fromisoformat(current_subtask.started_at)
                end_time = datetime.now()
                duration_minutes = int((end_time - start_time).total_seconds() / 60)
                current_subtask.actual_duration_minutes = duration_minutes
            
            # Move to next subtask
            task.current_subtask_index += 1
            
            # Update progress
            task.progress_percentage = (task.current_subtask_index / len(task.subtasks)) * 100
            
            # Check if task is complete
            if task.current_subtask_index >= len(task.subtasks):
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
                
                # Calculate total duration
                if task.started_at:
                    start_time = datetime.fromisoformat(task.started_at)
                    end_time = datetime.now()
                    total_duration = int((end_time - start_time).total_seconds() / 60)
                    task.actual_total_duration_minutes = total_duration
                
                logger.info(f"Task completed: {task.title} ({task_id})")
            else:
                # Start next subtask
                next_subtask = task.subtasks[task.current_subtask_index]
                
                # Check dependencies
                if self._check_subtask_dependencies(task, next_subtask):
                    next_subtask.status = SubtaskStatus.IN_PROGRESS
                    next_subtask.started_at = datetime.now().isoformat()
                    logger.info(f"Started next subtask: {next_subtask.title}")
                else:
                    logger.warning(f"Dependencies not met for subtask: {next_subtask.title}")
            
            self._save_tasks()
            return True
            
        except Exception as e:
            logger.error(f"Error completing subtask for task {task_id}: {e}")
            return False
    
    def pause_task(self, task_id: str) -> bool:
        """Pause a task."""
        try:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            task.status = TaskStatus.PAUSED
            
            # Pause current subtask
            if (task.current_subtask_index < len(task.subtasks) and
                task.subtasks[task.current_subtask_index].status == SubtaskStatus.IN_PROGRESS):
                task.subtasks[task.current_subtask_index].status = SubtaskStatus.PENDING
            
            self._save_tasks()
            
            logger.info(f"Paused task: {task.title} ({task_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error pausing task {task_id}: {e}")
            return False
    
    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        try:
            task = self.tasks.get(task_id)
            if not task or task.status != TaskStatus.PAUSED:
                return False
            
            task.status = TaskStatus.IN_PROGRESS
            
            # Resume current subtask
            if task.current_subtask_index < len(task.subtasks):
                current_subtask = task.subtasks[task.current_subtask_index]
                if current_subtask.status == SubtaskStatus.PENDING:
                    current_subtask.status = SubtaskStatus.IN_PROGRESS
                    current_subtask.started_at = datetime.now().isoformat()
            
            self._save_tasks()
            
            logger.info(f"Resumed task: {task.title} ({task_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming task {task_id}: {e}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        try:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now().isoformat()
            
            # Cancel all pending subtasks
            for subtask in task.subtasks:
                if subtask.status in [SubtaskStatus.PENDING, SubtaskStatus.IN_PROGRESS]:
                    subtask.status = SubtaskStatus.SKIPPED
            
            # Remove from active tasks
            for session_id, active_task_id in list(self.active_tasks.items()):
                if active_task_id == task_id:
                    del self.active_tasks[session_id]
            
            self._save_tasks()
            
            logger.info(f"Cancelled task: {task.title} ({task_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False
    
    def update_task_context(self, task_id: str, context_updates: Dict[str, Any]) -> bool:
        """Update task context with new information."""
        try:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            task.context.update(context_updates)
            self._save_tasks()
            
            logger.debug(f"Updated context for task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating task context {task_id}: {e}")
            return False
    
    def get_task_status_summary(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of task status."""
        try:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            completed_subtasks = sum(1 for st in task.subtasks if st.status == SubtaskStatus.COMPLETED)
            
            current_subtask = None
            if task.current_subtask_index < len(task.subtasks):
                current_subtask = task.subtasks[task.current_subtask_index]
            
            return {
                'task_id': task.task_id,
                'title': task.title,
                'status': task.status.value,
                'progress_percentage': task.progress_percentage,
                'completed_subtasks': completed_subtasks,
                'total_subtasks': len(task.subtasks),
                'current_subtask': {
                    'title': current_subtask.title if current_subtask else None,
                    'description': current_subtask.description if current_subtask else None,
                    'status': current_subtask.status.value if current_subtask else None
                } if current_subtask else None,
                'estimated_duration': task.estimated_total_duration_minutes,
                'actual_duration': task.actual_total_duration_minutes,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at
            }
            
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {e}")
            return None
    
    def list_user_tasks(self, user_id: str, status_filter: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """List tasks for a user."""
        try:
            user_tasks = []
            
            for task in self.tasks.values():
                if task.user_id == user_id:
                    if status_filter is None or task.status == status_filter:
                        summary = self.get_task_status_summary(task.task_id)
                        if summary:
                            user_tasks.append(summary)
            
            # Sort by creation date (newest first)
            user_tasks.sort(key=lambda x: x['created_at'], reverse=True)
            
            return user_tasks
            
        except Exception as e:
            logger.error(f"Error listing tasks for user {user_id}: {e}")
            return []
    
    def _estimate_subtask_duration(self, description: str) -> int:
        """Estimate subtask duration based on description."""
        # Simple heuristic based on description length and complexity indicators
        base_duration = 5  # 5 minutes base
        
        word_count = len(description.split())
        duration = base_duration + (word_count // 10)  # +1 minute per 10 words
        
        # Complexity indicators
        complexity_keywords = [
            'analyze', 'research', 'calculate', 'generate', 'create',
            'compare', 'evaluate', 'synthesize', 'complex', 'detailed'
        ]
        
        complexity_score = sum(1 for keyword in complexity_keywords 
                             if keyword in description.lower())
        
        duration += complexity_score * 3  # +3 minutes per complexity indicator
        
        return min(duration, 60)  # Cap at 60 minutes
    
    def _analyze_required_tools(self, description: str) -> List[str]:
        """Analyze what tools might be required for a subtask."""
        tools = []
        description_lower = description.lower()
        
        tool_keywords = {
            'python_interpreter': ['calculate', 'compute', 'math', 'code', 'script', 'algorithm'],
            'table_generator': ['table', 'compare', 'list', 'organize', 'structure'],
            'multimodal_query': ['search', 'find', 'lookup', 'research', 'information'],
            'web_search': ['current', 'latest', 'recent', 'news', 'web', 'online']
        }
        
        for tool, keywords in tool_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                tools.append(tool)
        
        return tools
    
    def _determine_expected_output(self, description: str) -> str:
        """Determine what output is expected from a subtask."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['table', 'list', 'comparison']):
            return "structured_data"
        elif any(word in description_lower for word in ['calculate', 'compute', 'math']):
            return "numerical_result"
        elif any(word in description_lower for word in ['analyze', 'research', 'find']):
            return "analysis_report"
        elif any(word in description_lower for word in ['create', 'generate', 'write']):
            return "generated_content"
        else:
            return "text_response"
    
    def _check_subtask_dependencies(self, task: Task, subtask: Subtask) -> bool:
        """Check if subtask dependencies are satisfied."""
        for dep_id in subtask.dependencies:
            dep_subtask = next((st for st in task.subtasks if st.subtask_id == dep_id), None)
            if not dep_subtask or dep_subtask.status != SubtaskStatus.COMPLETED:
                return False
        return True
    
    def _load_tasks(self):
        """Load tasks from storage."""
        try:
            if self.tasks_file.exists():
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruct tasks
                for task_data in data.get('tasks', []):
                    # Reconstruct subtasks
                    subtasks = []
                    for st_data in task_data['subtasks']:
                        subtask = Subtask(
                            subtask_id=st_data['subtask_id'],
                            title=st_data['title'],
                            description=st_data['description'],
                            status=SubtaskStatus(st_data['status']),
                            dependencies=st_data['dependencies'],
                            estimated_duration_minutes=st_data['estimated_duration_minutes'],
                            actual_duration_minutes=st_data.get('actual_duration_minutes'),
                            tools_required=st_data['tools_required'],
                            output_expected=st_data['output_expected'],
                            result=st_data.get('result'),
                            created_at=st_data['created_at'],
                            started_at=st_data.get('started_at'),
                            completed_at=st_data.get('completed_at'),
                            metadata=st_data.get('metadata', {})
                        )
                        subtasks.append(subtask)
                    
                    # Reconstruct task
                    task = Task(
                        task_id=task_data['task_id'],
                        title=task_data['title'],
                        description=task_data['description'],
                        status=TaskStatus(task_data['status']),
                        user_id=task_data['user_id'],
                        session_id=task_data['session_id'],
                        subtasks=subtasks,
                        current_subtask_index=task_data['current_subtask_index'],
                        progress_percentage=task_data['progress_percentage'],
                        created_at=task_data['created_at'],
                        started_at=task_data.get('started_at'),
                        completed_at=task_data.get('completed_at'),
                        estimated_total_duration_minutes=task_data['estimated_total_duration_minutes'],
                        actual_total_duration_minutes=task_data.get('actual_total_duration_minutes'),
                        context=task_data.get('context', {}),
                        metadata=task_data.get('metadata', {})
                    )
                    
                    self.tasks[task.task_id] = task
                
                # Reconstruct active tasks
                self.active_tasks = data.get('active_tasks', {})
                
                logger.info(f"Loaded {len(self.tasks)} tasks from storage")
            
        except Exception as e:
            logger.error(f"Error loading tasks: {e}")
    
    def _save_tasks(self):
        """Save tasks to storage."""
        try:
            # Convert tasks to serializable format
            tasks_data = []
            for task in self.tasks.values():
                task_dict = asdict(task)
                # Convert enums to strings
                task_dict['status'] = task.status.value
                for st_dict in task_dict['subtasks']:
                    st_dict['status'] = SubtaskStatus(st_dict['status']).value
                tasks_data.append(task_dict)
            
            data = {
                'tasks': tasks_data,
                'active_tasks': self.active_tasks,
                'last_updated': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            self.tasks_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.tasks)} tasks to storage")
            
        except Exception as e:
            logger.error(f"Error saving tasks: {e}")

# Global task manager instance
_task_manager = None

def get_task_manager(tasks_file: str = "task_manager.json") -> TaskManager:
    """Get or create a global task manager instance."""
    global _task_manager
    
    if _task_manager is None:
        _task_manager = TaskManager(tasks_file=tasks_file)
    
    return _task_manager
