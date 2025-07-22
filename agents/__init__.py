"""
SAM Agents Module
Agent Collaboration & Multi-SAM Swarm Mode

Sprint 10: Agent Collaboration & Multi-SAM Swarm Mode
"""

# Import all agent system components
from .task_router import TaskRouter, AgentRole, TaskType, TaskStatus, SubTask, TaskPlan, get_task_router
from .agent_comm import AgentCommunicationManager, MessageType, MessagePriority, AgentIdentity, AgentMessage, get_agent_comm_manager, shutdown_all_comm_managers
from .swarm_manager import LocalSwarmManager, SwarmStatus, SwarmAgent, SwarmConfiguration, get_swarm_manager
from .reasoning_chain import MultiAgentReasoningEngine, ReasoningStepType, VoteType, ReasoningStep, AgentVote, ReasoningChain, get_reasoning_engine
from .agent_manager import AgentManager, AgentCollaborationRequest, AgentCollaborationResponse, get_agent_manager

__all__ = [
    # Task Router
    'TaskRouter',
    'AgentRole',
    'TaskType',
    'TaskStatus',
    'SubTask',
    'TaskPlan',
    'get_task_router',
    
    # Agent Communication
    'AgentCommunicationManager',
    'MessageType',
    'MessagePriority',
    'AgentIdentity',
    'AgentMessage',
    'get_agent_comm_manager',
    'shutdown_all_comm_managers',
    
    # Swarm Manager
    'LocalSwarmManager',
    'SwarmStatus',
    'SwarmAgent',
    'SwarmConfiguration',
    'get_swarm_manager',
    
    # Reasoning Chain
    'MultiAgentReasoningEngine',
    'ReasoningStepType',
    'VoteType',
    'ReasoningStep',
    'AgentVote',
    'ReasoningChain',
    'get_reasoning_engine',
    
    # Agent Manager
    'AgentManager',
    'AgentCollaborationRequest',
    'AgentCollaborationResponse',
    'get_agent_manager'
]
