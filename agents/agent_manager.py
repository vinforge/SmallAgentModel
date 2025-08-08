"""
Agent Manager - Main Coordination Engine for SAM
Orchestrates multi-agent collaboration and delegation.

Sprint 10: Agent Collaboration & Multi-SAM Swarm Mode
"""

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from .task_router import TaskRouter, AgentRole, TaskStatus, get_task_router
from .agent_comm import AgentCommunicationManager, MessageType, get_agent_comm_manager
from .swarm_manager import LocalSwarmManager, get_swarm_manager
from .reasoning_chain import MultiAgentReasoningEngine, ReasoningStepType, get_reasoning_engine

logger = logging.getLogger(__name__)

@dataclass
class AgentCollaborationRequest:
    """Request for multi-agent collaboration."""
    request_id: str
    user_id: str
    session_id: str
    original_request: str
    collaboration_type: str
    required_agents: List[str]
    context: Dict[str, Any]
    priority: int
    created_at: str

@dataclass
class AgentCollaborationResponse:
    """Response from multi-agent collaboration."""
    request_id: str
    plan_id: str
    reasoning_chain_id: Optional[str]
    final_answer: str
    agent_contributions: Dict[str, Any]
    execution_trace: List[Dict[str, Any]]
    confidence_score: float
    processing_time_ms: int
    status: str
    completed_at: str

class AgentManager:
    """
    Main coordination engine for multi-agent collaboration and delegation.
    """
    
    def __init__(self, config_file: str = "agent_manager_config.json"):
        """
        Initialize the agent manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.task_router = get_task_router()
        self.swarm_manager = get_swarm_manager()
        self.reasoning_engine = get_reasoning_engine()
        
        # Storage
        self.collaboration_requests: Dict[str, AgentCollaborationRequest] = {}
        self.collaboration_responses: Dict[str, AgentCollaborationResponse] = {}
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}
        
        # Agent registry
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Agent manager initialized")
    
    def process_collaboration_request(self, request: str, user_id: str, session_id: str,
                                    collaboration_type: str = "auto",
                                    context: Dict[str, Any] = None) -> AgentCollaborationResponse:
        """
        Process a request that requires multi-agent collaboration.
        
        Args:
            request: User request
            user_id: User ID
            session_id: Session ID
            collaboration_type: Type of collaboration (auto, swarm, reasoning_chain)
            context: Additional context
            
        Returns:
            AgentCollaborationResponse with results
        """
        try:
            start_time = datetime.now()
            request_id = f"collab_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Processing collaboration request: {request[:50]}...")
            
            # Create collaboration request
            collab_request = AgentCollaborationRequest(
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                original_request=request,
                collaboration_type=collaboration_type,
                required_agents=[],
                context=context or {},
                priority=5,
                created_at=datetime.now().isoformat()
            )
            
            self.collaboration_requests[request_id] = collab_request
            
            # Determine collaboration strategy
            if collaboration_type == "auto":
                collaboration_type = self._determine_collaboration_type(request)
            
            # Execute collaboration based on type
            if collaboration_type == "swarm":
                result = self._execute_swarm_collaboration(collab_request)
            elif collaboration_type == "reasoning_chain":
                result = self._execute_reasoning_chain_collaboration(collab_request)
            elif collaboration_type == "task_delegation":
                result = self._execute_task_delegation_collaboration(collab_request)
            else:
                result = self._execute_default_collaboration(collab_request)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Create response
            response = AgentCollaborationResponse(
                request_id=request_id,
                plan_id=result.get('plan_id', ''),
                reasoning_chain_id=result.get('reasoning_chain_id'),
                final_answer=result.get('final_answer', 'Collaboration completed'),
                agent_contributions=result.get('agent_contributions', {}),
                execution_trace=result.get('execution_trace', []),
                confidence_score=result.get('confidence_score', 0.7),
                processing_time_ms=processing_time_ms,
                status="completed",
                completed_at=datetime.now().isoformat()
            )
            
            self.collaboration_responses[request_id] = response
            
            logger.info(f"Collaboration completed: {request_id} in {processing_time_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error processing collaboration request: {e}")
            return self._create_error_response(request_id, str(e))
    
    def start_swarm_mode(self) -> bool:
        """Start the local swarm for concurrent task execution."""
        try:
            return self.swarm_manager.start_swarm()
        except Exception as e:
            logger.error(f"Error starting swarm mode: {e}")
            return False
    
    def stop_swarm_mode(self) -> bool:
        """Stop the local swarm."""
        try:
            return self.swarm_manager.stop_swarm()
        except Exception as e:
            logger.error(f"Error stopping swarm mode: {e}")
            return False
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status."""
        return self.swarm_manager.get_swarm_status()
    
    def register_agent(self, agent_id: str, agent_name: str, agent_role: str,
                      capabilities: List[str], endpoint: str = None) -> bool:
        """
        Register an agent with the manager.
        
        Args:
            agent_id: Unique agent ID
            agent_name: Agent name
            agent_role: Agent role
            capabilities: List of capabilities
            endpoint: Optional endpoint for remote agents
            
        Returns:
            True if registration successful
        """
        try:
            self.registered_agents[agent_id] = {
                'agent_id': agent_id,
                'agent_name': agent_name,
                'agent_role': agent_role,
                'capabilities': capabilities,
                'endpoint': endpoint,
                'status': 'active',
                'registered_at': datetime.now().isoformat()
            }
            
            logger.info(f"Registered agent: {agent_name} ({agent_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {e}")
            return False
    
    def get_agent_status(self, agent_id: str = None) -> Dict[str, Any]:
        """
        Get status of specific agent or all agents.
        
        Args:
            agent_id: Optional specific agent ID
            
        Returns:
            Agent status information
        """
        try:
            if agent_id:
                return self.registered_agents.get(agent_id, {'error': 'Agent not found'})
            else:
                return {
                    'total_agents': len(self.registered_agents),
                    'agents': self.registered_agents
                }
                
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return {'error': str(e)}
    
    def create_reasoning_chain(self, topic: str, initial_query: str,
                             participants: List[str]) -> str:
        """
        Create a new multi-agent reasoning chain.
        
        Args:
            topic: Reasoning topic
            initial_query: Initial question
            participants: List of participating agent IDs
            
        Returns:
            Chain ID
        """
        try:
            # Use first participant as initiator
            initiator_id = participants[0] if participants else "system"
            
            return self.reasoning_engine.start_reasoning_chain(
                topic=topic,
                initial_query=initial_query,
                participants=participants,
                initiator_id=initiator_id
            )
            
        except Exception as e:
            logger.error(f"Error creating reasoning chain: {e}")
            raise
    
    def get_collaboration_history(self, user_id: str = None,
                                days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Get collaboration history.
        
        Args:
            user_id: Optional user ID filter
            days_back: Number of days to look back
            
        Returns:
            List of collaboration records
        """
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            history = []
            for response in self.collaboration_responses.values():
                completed_at = datetime.fromisoformat(response.completed_at)
                
                if completed_at >= cutoff_date:
                    request = self.collaboration_requests.get(response.request_id)
                    if request and (user_id is None or request.user_id == user_id):
                        history.append({
                            'request_id': response.request_id,
                            'original_request': request.original_request,
                            'collaboration_type': request.collaboration_type,
                            'final_answer': response.final_answer,
                            'confidence_score': response.confidence_score,
                            'processing_time_ms': response.processing_time_ms,
                            'completed_at': response.completed_at
                        })
            
            # Sort by completion time (newest first)
            history.sort(key=lambda x: x['completed_at'], reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting collaboration history: {e}")
            return []
    
    def _determine_collaboration_type(self, request: str) -> str:
        """Determine the best collaboration type for a request."""
        request_lower = request.lower()
        
        # Check for reasoning-heavy requests
        if any(word in request_lower for word in ['analyze', 'compare', 'evaluate', 'reason', 'think']):
            return "reasoning_chain"
        
        # Check for multi-document or complex processing
        if any(word in request_lower for word in ['documents', 'files', 'multiple', 'several']):
            return "swarm"
        
        # Default to task delegation
        return "task_delegation"
    
    def _execute_swarm_collaboration(self, request: AgentCollaborationRequest) -> Dict[str, Any]:
        """Execute collaboration using swarm mode."""
        try:
            # Start swarm if not already running
            if self.swarm_manager.status.value != "active":
                self.swarm_manager.start_swarm()
            
            # Submit task to swarm
            plan_id = self.swarm_manager.submit_task_to_swarm(
                request=request.original_request,
                user_id=request.user_id,
                session_id=request.session_id,
                context=request.context
            )
            
            # Wait for completion (simplified - would be more sophisticated in practice)
            import time
            time.sleep(1)  # Simulate processing time
            
            # Get plan progress
            progress = self.task_router.get_plan_progress(plan_id)
            
            return {
                'plan_id': plan_id,
                'final_answer': f"Swarm collaboration completed for: {request.original_request}",
                'agent_contributions': {'swarm': 'Distributed task execution'},
                'execution_trace': [{'step': 'swarm_execution', 'status': 'completed'}],
                'confidence_score': 0.8
            }
            
        except Exception as e:
            logger.error(f"Error in swarm collaboration: {e}")
            return {
                'final_answer': f"Swarm collaboration failed: {str(e)}",
                'confidence_score': 0.1
            }
    
    def _execute_reasoning_chain_collaboration(self, request: AgentCollaborationRequest) -> Dict[str, Any]:
        """Execute collaboration using reasoning chain."""
        try:
            # Create reasoning chain with available agents
            participants = ['planner_001', 'executor_001', 'critic_001']  # Simulated agent IDs
            
            chain_id = self.reasoning_engine.start_reasoning_chain(
                topic=f"Analysis of: {request.original_request[:50]}",
                initial_query=request.original_request,
                participants=participants,
                initiator_id=participants[0]
            )
            
            # Simulate reasoning steps
            self.reasoning_engine.add_reasoning_step(
                chain_id=chain_id,
                agent_id=participants[1],
                step_type=ReasoningStepType.HYPOTHESIS,
                content="Initial hypothesis based on request analysis",
                reasoning="Analyzing the request to form initial understanding",
                confidence=0.7
            )
            
            self.reasoning_engine.add_reasoning_step(
                chain_id=chain_id,
                agent_id=participants[2],
                step_type=ReasoningStepType.CRITIQUE,
                content="Critical evaluation of the hypothesis",
                reasoning="Reviewing the hypothesis for potential issues",
                confidence=0.8
            )
            
            # Generate visual trace
            visual_trace = self.reasoning_engine.generate_visual_trace(chain_id)
            
            return {
                'reasoning_chain_id': chain_id,
                'final_answer': f"Multi-agent reasoning completed for: {request.original_request}",
                'agent_contributions': {
                    'planner': 'Initial analysis',
                    'executor': 'Hypothesis formation',
                    'critic': 'Critical evaluation'
                },
                'execution_trace': [
                    {'step': 'reasoning_chain_creation', 'chain_id': chain_id},
                    {'step': 'collaborative_reasoning', 'participants': len(participants)},
                    {'step': 'visual_trace_generation', 'length': len(visual_trace)}
                ],
                'confidence_score': 0.75,
                'visual_trace': visual_trace
            }
            
        except Exception as e:
            logger.error(f"Error in reasoning chain collaboration: {e}")
            return {
                'final_answer': f"Reasoning chain collaboration failed: {str(e)}",
                'confidence_score': 0.1
            }
    
    def _execute_task_delegation_collaboration(self, request: AgentCollaborationRequest) -> Dict[str, Any]:
        """Execute collaboration using task delegation."""
        try:
            # Decompose task
            task_plan = self.task_router.decompose_task(
                request=request.original_request,
                user_id=request.user_id,
                session_id=request.session_id,
                context=request.context
            )
            
            # Simulate task execution
            execution_trace = []
            agent_contributions = {}
            
            for i, sub_task in enumerate(task_plan.sub_tasks):
                # Simulate agent assignment and execution
                agent_id = f"agent_{i+1}"
                
                self.task_router.assign_task_to_agent(sub_task.task_id, agent_id)
                self.task_router.update_task_status(sub_task.task_id, TaskStatus.COMPLETED, 
                                                  result=f"Task {sub_task.task_type.value} completed")
                
                execution_trace.append({
                    'task_id': sub_task.task_id,
                    'agent_id': agent_id,
                    'task_type': sub_task.task_type.value,
                    'status': 'completed'
                })
                
                agent_contributions[agent_id] = f"Completed {sub_task.task_type.value}"
            
            return {
                'plan_id': task_plan.plan_id,
                'final_answer': f"Task delegation completed for: {request.original_request}",
                'agent_contributions': agent_contributions,
                'execution_trace': execution_trace,
                'confidence_score': 0.8
            }
            
        except Exception as e:
            logger.error(f"Error in task delegation collaboration: {e}")
            return {
                'final_answer': f"Task delegation collaboration failed: {str(e)}",
                'confidence_score': 0.1
            }
    
    def _execute_default_collaboration(self, request: AgentCollaborationRequest) -> Dict[str, Any]:
        """Execute default collaboration strategy."""
        return {
            'final_answer': f"Processed request: {request.original_request}",
            'agent_contributions': {'primary_agent': 'Single agent processing'},
            'execution_trace': [{'step': 'single_agent_processing', 'status': 'completed'}],
            'confidence_score': 0.6
        }
    
    def _create_error_response(self, request_id: str, error_message: str) -> AgentCollaborationResponse:
        """Create an error response."""
        return AgentCollaborationResponse(
            request_id=request_id,
            plan_id="",
            reasoning_chain_id=None,
            final_answer=f"Error processing collaboration request: {error_message}",
            agent_contributions={},
            execution_trace=[],
            confidence_score=0.0,
            processing_time_ms=0,
            status="failed",
            completed_at=datetime.now().isoformat()
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create default configuration
                default_config = {
                    'max_concurrent_collaborations': 10,
                    'default_collaboration_timeout': 300,
                    'enable_swarm_mode': True,
                    'enable_reasoning_chains': True,
                    'auto_start_swarm': False
                }
                
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
                
                return default_config
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

# Global agent manager instance
_agent_manager = None

def get_agent_manager(config_file: str = "agent_manager_config.json") -> AgentManager:
    """Get or create a global agent manager instance."""
    global _agent_manager
    
    if _agent_manager is None:
        _agent_manager = AgentManager(config_file=config_file)
    
    return _agent_manager
