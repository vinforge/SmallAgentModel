"""
Local Swarm Mode for SAM
Supports concurrent task execution via multiple local SAM agent instances.

Sprint 10 Task 2: Local Swarm Mode
"""

import logging
import json
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import queue

from .task_router import TaskRouter, AgentRole, TaskStatus, get_task_router
from .agent_comm import AgentCommunicationManager, MessageType, MessagePriority, get_agent_comm_manager

logger = logging.getLogger(__name__)

class SwarmStatus(Enum):
    """Status of the swarm."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"

@dataclass
class SwarmAgent:
    """A SAM agent in the swarm."""
    agent_id: str
    agent_name: str
    agent_role: AgentRole
    capabilities: List[str]
    status: str
    current_task: Optional[str]
    tasks_completed: int
    tasks_failed: int
    load_score: float
    last_heartbeat: str
    metadata: Dict[str, Any]

@dataclass
class SwarmConfiguration:
    """Configuration for the swarm."""
    swarm_id: str
    max_agents: int
    load_balancing_strategy: str
    shared_context_enabled: bool
    auto_scaling_enabled: bool
    heartbeat_interval: int
    task_timeout: int
    agent_roles: Dict[str, Dict[str, Any]]

class LocalSwarmManager:
    """
    Manages a local swarm of SAM agents for concurrent task execution.
    """
    
    def __init__(self, swarm_config_file: str = "swarm_config.json"):
        """
        Initialize the local swarm manager.
        
        Args:
            swarm_config_file: Path to swarm configuration file
        """
        self.swarm_config_file = Path(swarm_config_file)
        
        # Load or create configuration
        self.config = self._load_swarm_config()
        
        # Swarm state
        self.swarm_id = self.config.swarm_id
        self.status = SwarmStatus.INITIALIZING
        self.agents: Dict[str, SwarmAgent] = {}
        self.task_queues: Dict[str, queue.Queue] = {}  # Per-agent task queues
        self.shared_context: Dict[str, Any] = {}
        
        # Components
        self.task_router = get_task_router()
        self.agent_threads: Dict[str, threading.Thread] = {}
        
        # Synchronization
        self.swarm_lock = threading.Lock()
        self.running = False
        
        logger.info(f"Local swarm manager initialized: {self.swarm_id}")
    
    def start_swarm(self) -> bool:
        """
        Start the swarm with configured agents.
        
        Returns:
            True if swarm started successfully
        """
        try:
            with self.swarm_lock:
                if self.status != SwarmStatus.INITIALIZING:
                    logger.warning(f"Swarm already started: {self.status}")
                    return False
                
                self.running = True
                self.status = SwarmStatus.ACTIVE
                
                # Create agents based on configuration
                for role_name, role_config in self.config.agent_roles.items():
                    agent_count = role_config.get('count', 1)
                    
                    for i in range(agent_count):
                        agent_id = f"{role_name}_{i+1}_{uuid.uuid4().hex[:8]}"
                        self._create_agent(agent_id, role_name, role_config)
                
                logger.info(f"Swarm started with {len(self.agents)} agents")
                return True
                
        except Exception as e:
            logger.error(f"Error starting swarm: {e}")
            self.status = SwarmStatus.STOPPED
            return False
    
    def stop_swarm(self) -> bool:
        """
        Stop the swarm and all agents.
        
        Returns:
            True if swarm stopped successfully
        """
        try:
            with self.swarm_lock:
                if self.status == SwarmStatus.STOPPED:
                    return True
                
                self.status = SwarmStatus.SHUTTING_DOWN
                self.running = False
                
                # Stop all agent threads
                for agent_id, thread in self.agent_threads.items():
                    if thread.is_alive():
                        logger.info(f"Stopping agent thread: {agent_id}")
                        thread.join(timeout=5)
                
                self.status = SwarmStatus.STOPPED
                logger.info("Swarm stopped")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping swarm: {e}")
            return False
    
    def submit_task_to_swarm(self, request: str, user_id: str, session_id: str,
                           context: Dict[str, Any] = None) -> str:
        """
        Submit a task to the swarm for distributed execution.
        
        Args:
            request: User request to process
            user_id: User ID
            session_id: Session ID
            context: Additional context
            
        Returns:
            Plan ID for tracking
        """
        try:
            if self.status != SwarmStatus.ACTIVE:
                raise ValueError(f"Swarm not active: {self.status}")
            
            # Decompose task using task router
            task_plan = self.task_router.decompose_task(request, user_id, session_id, context)
            
            # Distribute sub-tasks to appropriate agents
            self._distribute_tasks(task_plan)
            
            logger.info(f"Submitted task to swarm: {task_plan.plan_id}")
            return task_plan.plan_id
            
        except Exception as e:
            logger.error(f"Error submitting task to swarm: {e}")
            raise
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status."""
        try:
            with self.swarm_lock:
                agent_stats = {}
                for agent_id, agent in self.agents.items():
                    agent_stats[agent_id] = {
                        'name': agent.agent_name,
                        'role': agent.agent_role.value,
                        'status': agent.status,
                        'current_task': agent.current_task,
                        'tasks_completed': agent.tasks_completed,
                        'tasks_failed': agent.tasks_failed,
                        'load_score': agent.load_score
                    }
                
                return {
                    'swarm_id': self.swarm_id,
                    'status': self.status.value,
                    'total_agents': len(self.agents),
                    'active_agents': sum(1 for a in self.agents.values() if a.status == 'active'),
                    'shared_context_size': len(self.shared_context),
                    'agents': agent_stats
                }
                
        except Exception as e:
            logger.error(f"Error getting swarm status: {e}")
            return {'error': str(e)}
    
    def add_to_shared_context(self, key: str, value: Any, source_agent: str):
        """
        Add information to shared context pool.
        
        Args:
            key: Context key
            value: Context value
            source_agent: Agent that provided the context
        """
        try:
            if self.config.shared_context_enabled:
                self.shared_context[key] = {
                    'value': value,
                    'source_agent': source_agent,
                    'timestamp': datetime.now().isoformat()
                }
                logger.debug(f"Added to shared context: {key} from {source_agent}")
            
        except Exception as e:
            logger.error(f"Error adding to shared context: {e}")
    
    def get_shared_context(self, key: str = None) -> Any:
        """
        Get information from shared context pool.
        
        Args:
            key: Optional specific key to retrieve
            
        Returns:
            Context value or entire context if key is None
        """
        try:
            if key:
                context_item = self.shared_context.get(key)
                return context_item['value'] if context_item else None
            else:
                return {k: v['value'] for k, v in self.shared_context.items()}
                
        except Exception as e:
            logger.error(f"Error getting shared context: {e}")
            return None
    
    def _create_agent(self, agent_id: str, role_name: str, role_config: Dict[str, Any]):
        """Create and start a new agent."""
        try:
            agent_role = AgentRole(role_name.lower())
            capabilities = role_config.get('capabilities', [])
            
            # Create agent
            agent = SwarmAgent(
                agent_id=agent_id,
                agent_name=f"SAM-{role_name}-{agent_id[-8:]}",
                agent_role=agent_role,
                capabilities=capabilities,
                status="initializing",
                current_task=None,
                tasks_completed=0,
                tasks_failed=0,
                load_score=0.0,
                last_heartbeat=datetime.now().isoformat(),
                metadata=role_config.get('metadata', {})
            )
            
            self.agents[agent_id] = agent
            
            # Create task queue for agent
            self.task_queues[agent_id] = queue.Queue()
            
            # Create communication manager for agent
            comm_manager = get_agent_comm_manager(
                agent_id=agent_id,
                agent_name=agent.agent_name,
                agent_role=role_name,
                capabilities=capabilities
            )
            
            # Register message handlers
            comm_manager.register_message_handler(MessageType.TASK_REQUEST, self._handle_task_request)
            comm_manager.register_message_handler(MessageType.STATUS_UPDATE, self._handle_status_update)
            
            # Start agent thread
            agent_thread = threading.Thread(
                target=self._run_agent,
                args=(agent_id,),
                daemon=True
            )
            agent_thread.start()
            self.agent_threads[agent_id] = agent_thread
            
            logger.info(f"Created agent: {agent.agent_name} ({agent_id})")
            
        except Exception as e:
            logger.error(f"Error creating agent {agent_id}: {e}")
    
    def _run_agent(self, agent_id: str):
        """Run an agent in its own thread."""
        try:
            agent = self.agents[agent_id]
            agent.status = "active"
            
            logger.info(f"Agent {agent.agent_name} started")
            
            while self.running:
                try:
                    # Check for tasks in queue
                    try:
                        task = self.task_queues[agent_id].get(timeout=1.0)
                    except queue.Empty:
                        continue
                    
                    # Process task
                    self._process_agent_task(agent_id, task)
                    
                    # Mark task as done
                    self.task_queues[agent_id].task_done()
                    
                except Exception as e:
                    logger.error(f"Error in agent {agent_id}: {e}")
                    agent.tasks_failed += 1
            
            agent.status = "stopped"
            logger.info(f"Agent {agent.agent_name} stopped")
            
        except Exception as e:
            logger.error(f"Error running agent {agent_id}: {e}")
    
    def _process_agent_task(self, agent_id: str, task):
        """Process a task assigned to an agent."""
        try:
            agent = self.agents[agent_id]
            agent.current_task = task.task_id
            agent.status = "busy"
            
            logger.info(f"Agent {agent.agent_name} processing task: {task.task_id}")
            
            # Update task status
            self.task_router.update_task_status(task.task_id, TaskStatus.IN_PROGRESS)
            
            # Simulate task processing (would integrate with actual SAM capabilities)
            result = self._simulate_task_execution(agent, task)
            
            # Update task with result
            if result.get('success', False):
                self.task_router.update_task_status(
                    task.task_id, 
                    TaskStatus.COMPLETED, 
                    result=result.get('output')
                )
                agent.tasks_completed += 1
                
                # Add result to shared context if relevant
                if result.get('context_key'):
                    self.add_to_shared_context(
                        result['context_key'],
                        result.get('output'),
                        agent_id
                    )
            else:
                self.task_router.update_task_status(
                    task.task_id,
                    TaskStatus.FAILED,
                    error_message=result.get('error', 'Task execution failed')
                )
                agent.tasks_failed += 1
            
            # Update agent state
            agent.current_task = None
            agent.status = "active"
            agent.last_heartbeat = datetime.now().isoformat()
            
            # Update load score
            agent.load_score = self._calculate_agent_load(agent)
            
        except Exception as e:
            logger.error(f"Error processing task for agent {agent_id}: {e}")
            agent.tasks_failed += 1
            agent.current_task = None
            agent.status = "active"
    
    def _distribute_tasks(self, task_plan):
        """Distribute tasks from a plan to appropriate agents."""
        try:
            for execution_phase in task_plan.execution_order:
                for task_id in execution_phase:
                    # Find the task
                    task = next((t for t in task_plan.sub_tasks if t.task_id == task_id), None)
                    if not task:
                        continue
                    
                    # Find best agent for this task
                    best_agent_id = self._select_best_agent(task.required_role, task.metadata.get('required_capabilities', []))
                    
                    if best_agent_id:
                        # Assign task to agent
                        self.task_router.assign_task_to_agent(task_id, best_agent_id)
                        
                        # Add task to agent's queue
                        self.task_queues[best_agent_id].put(task)
                        
                        logger.info(f"Assigned task {task_id} to agent {best_agent_id}")
                    else:
                        logger.warning(f"No suitable agent found for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error distributing tasks: {e}")
    
    def _select_best_agent(self, required_role: AgentRole, required_capabilities: List[str]) -> Optional[str]:
        """Select the best agent for a task based on role and load."""
        try:
            # Filter agents by role
            suitable_agents = [
                agent for agent in self.agents.values()
                if (agent.agent_role == required_role and 
                    agent.status == "active" and
                    all(cap in agent.capabilities for cap in required_capabilities))
            ]
            
            if not suitable_agents:
                return None
            
            # Select agent with lowest load score
            best_agent = min(suitable_agents, key=lambda a: a.load_score)
            return best_agent.agent_id
            
        except Exception as e:
            logger.error(f"Error selecting best agent: {e}")
            return None
    
    def _calculate_agent_load(self, agent: SwarmAgent) -> float:
        """Calculate load score for an agent."""
        try:
            # Simple load calculation based on queue size and task history
            queue_size = self.task_queues[agent.agent_id].qsize()
            
            # Factor in recent performance
            total_tasks = agent.tasks_completed + agent.tasks_failed
            failure_rate = agent.tasks_failed / total_tasks if total_tasks > 0 else 0
            
            # Load score (0.0 = no load, 1.0 = maximum load)
            load_score = min(1.0, (queue_size * 0.2) + (failure_rate * 0.3))
            
            return load_score
            
        except Exception as e:
            logger.error(f"Error calculating agent load: {e}")
            return 0.5
    
    def _simulate_task_execution(self, agent: SwarmAgent, task) -> Dict[str, Any]:
        """Simulate task execution (placeholder for actual SAM integration)."""
        try:
            # Simulate processing time
            time.sleep(0.1)
            
            # Simulate different outcomes based on task type
            if task.task_type.value == "document_analysis":
                return {
                    'success': True,
                    'output': f"Document analysis completed by {agent.agent_name}",
                    'context_key': f"doc_analysis_{task.task_id}",
                    'processing_time': 0.1
                }
            elif task.task_type.value == "summarization":
                return {
                    'success': True,
                    'output': f"Summary generated by {agent.agent_name}",
                    'context_key': f"summary_{task.task_id}",
                    'processing_time': 0.1
                }
            else:
                return {
                    'success': True,
                    'output': f"Task {task.task_type.value} completed by {agent.agent_name}",
                    'processing_time': 0.1
                }
                
        except Exception as e:
            logger.error(f"Error simulating task execution: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _handle_task_request(self, message):
        """Handle task request message."""
        logger.info(f"Received task request: {message.content}")
    
    def _handle_status_update(self, message):
        """Handle status update message."""
        logger.info(f"Received status update: {message.content}")
    
    def _load_swarm_config(self) -> SwarmConfiguration:
        """Load swarm configuration from file."""
        try:
            if self.swarm_config_file.exists():
                with open(self.swarm_config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                return SwarmConfiguration(
                    swarm_id=config_data.get('swarm_id', f"swarm_{uuid.uuid4().hex[:8]}"),
                    max_agents=config_data.get('max_agents', 10),
                    load_balancing_strategy=config_data.get('load_balancing_strategy', 'round_robin'),
                    shared_context_enabled=config_data.get('shared_context_enabled', True),
                    auto_scaling_enabled=config_data.get('auto_scaling_enabled', False),
                    heartbeat_interval=config_data.get('heartbeat_interval', 30),
                    task_timeout=config_data.get('task_timeout', 300),
                    agent_roles=config_data.get('agent_roles', self._get_default_agent_roles())
                )
            else:
                # Create default configuration
                config = SwarmConfiguration(
                    swarm_id=f"swarm_{uuid.uuid4().hex[:8]}",
                    max_agents=5,
                    load_balancing_strategy='load_based',
                    shared_context_enabled=True,
                    auto_scaling_enabled=False,
                    heartbeat_interval=30,
                    task_timeout=300,
                    agent_roles=self._get_default_agent_roles()
                )
                
                self._save_swarm_config(config)
                return config
                
        except Exception as e:
            logger.error(f"Error loading swarm config: {e}")
            # Return minimal default config
            return SwarmConfiguration(
                swarm_id=f"swarm_{uuid.uuid4().hex[:8]}",
                max_agents=3,
                load_balancing_strategy='round_robin',
                shared_context_enabled=True,
                auto_scaling_enabled=False,
                heartbeat_interval=30,
                task_timeout=300,
                agent_roles=self._get_default_agent_roles()
            )
    
    def _save_swarm_config(self, config: SwarmConfiguration):
        """Save swarm configuration to file."""
        try:
            config_data = asdict(config)
            
            with open(self.swarm_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved swarm configuration: {self.swarm_config_file}")
            
        except Exception as e:
            logger.error(f"Error saving swarm config: {e}")
    
    def _get_default_agent_roles(self) -> Dict[str, Dict[str, Any]]:
        """Get default agent role configuration."""
        return {
            'executor': {
                'count': 2,
                'capabilities': ['document_processing', 'text_processing', 'analysis'],
                'metadata': {'max_concurrent_tasks': 3}
            },
            'synthesizer': {
                'count': 1,
                'capabilities': ['synthesis', 'integration', 'comparison'],
                'metadata': {'max_concurrent_tasks': 2}
            },
            'validator': {
                'count': 1,
                'capabilities': ['validation', 'verification', 'quality_check'],
                'metadata': {'max_concurrent_tasks': 2}
            }
        }

# Global swarm manager instance
_swarm_manager = None

def get_swarm_manager(swarm_config_file: str = "swarm_config.json") -> LocalSwarmManager:
    """Get or create a global swarm manager instance."""
    global _swarm_manager
    
    if _swarm_manager is None:
        _swarm_manager = LocalSwarmManager(swarm_config_file=swarm_config_file)
    
    return _swarm_manager
