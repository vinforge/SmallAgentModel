"""
Multi-Agent Chain-of-Thought Reasoning for SAM
Enables collaborative, back-and-forth deliberation between agents.

Sprint 10 Task 4: Multi-Agent Chain-of-Thought Reasoning
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ReasoningStepType(Enum):
    """Types of reasoning steps."""
    INITIAL_ANALYSIS = "initial_analysis"
    HYPOTHESIS = "hypothesis"
    EVIDENCE_GATHERING = "evidence_gathering"
    VALIDATION = "validation"
    CRITIQUE = "critique"
    REFINEMENT = "refinement"
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"

class VoteType(Enum):
    """Types of votes in validation."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    REQUEST_REVISION = "request_revision"

@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_id: str
    step_type: ReasoningStepType
    agent_id: str
    agent_name: str
    content: str
    reasoning: str
    confidence: float
    evidence: List[str]
    timestamp: str
    parent_step_id: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class AgentVote:
    """A vote from an agent on a reasoning step or plan."""
    vote_id: str
    agent_id: str
    agent_name: str
    vote_type: VoteType
    target_step_id: str
    reasoning: str
    confidence: float
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class ReasoningChain:
    """A complete chain of reasoning between agents."""
    chain_id: str
    topic: str
    participants: List[str]
    steps: List[ReasoningStep]
    votes: List[AgentVote]
    current_phase: str
    status: str
    created_at: str
    completed_at: Optional[str]
    final_conclusion: Optional[str]
    metadata: Dict[str, Any]

class MultiAgentReasoningEngine:
    """
    Manages collaborative reasoning between multiple agents.
    """
    
    def __init__(self):
        """Initialize the multi-agent reasoning engine."""
        
        # Storage
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        self.active_deliberations: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            'max_reasoning_steps': 20,
            'min_participants': 2,
            'max_participants': 5,
            'voting_threshold': 0.6,  # 60% approval needed
            'max_deliberation_rounds': 5,
            'step_timeout_minutes': 10
        }
        
        # Reasoning patterns
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
        logger.info("Multi-agent reasoning engine initialized")
    
    def start_reasoning_chain(self, topic: str, initial_query: str,
                            participants: List[str], initiator_id: str) -> str:
        """
        Start a new reasoning chain with multiple agents.
        
        Args:
            topic: Topic of reasoning
            initial_query: Initial question or problem
            participants: List of agent IDs to participate
            initiator_id: Agent starting the reasoning
            
        Returns:
            Chain ID
        """
        try:
            if len(participants) < self.config['min_participants']:
                raise ValueError(f"Need at least {self.config['min_participants']} participants")
            
            chain_id = f"chain_{uuid.uuid4().hex[:12]}"
            
            # Create initial reasoning step
            initial_step = ReasoningStep(
                step_id=f"step_{uuid.uuid4().hex[:8]}",
                step_type=ReasoningStepType.INITIAL_ANALYSIS,
                agent_id=initiator_id,
                agent_name=f"Agent-{initiator_id[-8:]}",
                content=initial_query,
                reasoning="Initial problem statement and analysis request",
                confidence=0.8,
                evidence=[],
                timestamp=datetime.now().isoformat(),
                parent_step_id=None,
                metadata={'initiator': True}
            )
            
            # Create reasoning chain
            reasoning_chain = ReasoningChain(
                chain_id=chain_id,
                topic=topic,
                participants=participants,
                steps=[initial_step],
                votes=[],
                current_phase="analysis",
                status="active",
                created_at=datetime.now().isoformat(),
                completed_at=None,
                final_conclusion=None,
                metadata={'initial_query': initial_query}
            )
            
            self.reasoning_chains[chain_id] = reasoning_chain
            
            logger.info(f"Started reasoning chain: {topic} ({chain_id}) with {len(participants)} participants")
            return chain_id
            
        except Exception as e:
            logger.error(f"Error starting reasoning chain: {e}")
            raise
    
    def add_reasoning_step(self, chain_id: str, agent_id: str, step_type: ReasoningStepType,
                          content: str, reasoning: str, confidence: float,
                          evidence: List[str] = None, parent_step_id: str = None) -> str:
        """
        Add a reasoning step to a chain.
        
        Args:
            chain_id: Chain ID
            agent_id: Agent adding the step
            step_type: Type of reasoning step
            content: Step content
            reasoning: Reasoning behind the step
            confidence: Confidence level (0.0-1.0)
            evidence: Optional supporting evidence
            parent_step_id: Optional parent step ID
            
        Returns:
            Step ID
        """
        try:
            chain = self.reasoning_chains.get(chain_id)
            if not chain:
                raise ValueError(f"Reasoning chain not found: {chain_id}")
            
            if agent_id not in chain.participants:
                raise ValueError(f"Agent {agent_id} not a participant in chain {chain_id}")
            
            if len(chain.steps) >= self.config['max_reasoning_steps']:
                raise ValueError(f"Maximum reasoning steps reached: {self.config['max_reasoning_steps']}")
            
            step_id = f"step_{uuid.uuid4().hex[:8]}"
            
            step = ReasoningStep(
                step_id=step_id,
                step_type=step_type,
                agent_id=agent_id,
                agent_name=f"Agent-{agent_id[-8:]}",
                content=content,
                reasoning=reasoning,
                confidence=confidence,
                evidence=evidence or [],
                timestamp=datetime.now().isoformat(),
                parent_step_id=parent_step_id,
                metadata={}
            )
            
            chain.steps.append(step)
            
            # Update chain phase based on step type
            self._update_chain_phase(chain, step_type)
            
            logger.info(f"Added {step_type.value} step to chain {chain_id}: {step_id}")
            return step_id
            
        except Exception as e:
            logger.error(f"Error adding reasoning step: {e}")
            raise
    
    def submit_vote(self, chain_id: str, agent_id: str, target_step_id: str,
                   vote_type: VoteType, reasoning: str, confidence: float) -> str:
        """
        Submit a vote on a reasoning step.
        
        Args:
            chain_id: Chain ID
            agent_id: Voting agent ID
            target_step_id: Step being voted on
            vote_type: Type of vote
            reasoning: Reasoning for the vote
            confidence: Confidence in the vote
            
        Returns:
            Vote ID
        """
        try:
            chain = self.reasoning_chains.get(chain_id)
            if not chain:
                raise ValueError(f"Reasoning chain not found: {chain_id}")
            
            if agent_id not in chain.participants:
                raise ValueError(f"Agent {agent_id} not a participant in chain {chain_id}")
            
            # Check if step exists
            target_step = next((s for s in chain.steps if s.step_id == target_step_id), None)
            if not target_step:
                raise ValueError(f"Target step not found: {target_step_id}")
            
            vote_id = f"vote_{uuid.uuid4().hex[:8]}"
            
            vote = AgentVote(
                vote_id=vote_id,
                agent_id=agent_id,
                agent_name=f"Agent-{agent_id[-8:]}",
                vote_type=vote_type,
                target_step_id=target_step_id,
                reasoning=reasoning,
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                metadata={}
            )
            
            chain.votes.append(vote)
            
            # Check if voting is complete
            self._check_voting_completion(chain, target_step_id)
            
            logger.info(f"Submitted {vote_type.value} vote for step {target_step_id}: {vote_id}")
            return vote_id
            
        except Exception as e:
            logger.error(f"Error submitting vote: {e}")
            raise
    
    def get_reasoning_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Get a reasoning chain by ID."""
        return self.reasoning_chains.get(chain_id)
    
    def generate_visual_trace(self, chain_id: str) -> str:
        """
        Generate a visual trace of the reasoning chain.
        
        Args:
            chain_id: Chain ID
            
        Returns:
            Visual trace as formatted text
        """
        try:
            chain = self.reasoning_chains.get(chain_id)
            if not chain:
                return f"Reasoning chain not found: {chain_id}"
            
            trace_parts = [
                f"# Reasoning Chain: {chain.topic}",
                f"**Chain ID:** {chain_id}",
                f"**Participants:** {', '.join([f'Agent-{p[-8:]}' for p in chain.participants])}",
                f"**Status:** {chain.status}",
                f"**Phase:** {chain.current_phase}",
                f"**Created:** {chain.created_at}",
                "",
                "## Reasoning Steps",
                ""
            ]
            
            # Add steps with visual indicators
            for i, step in enumerate(chain.steps, 1):
                step_emoji = {
                    ReasoningStepType.INITIAL_ANALYSIS: "🔍",
                    ReasoningStepType.HYPOTHESIS: "💡",
                    ReasoningStepType.EVIDENCE_GATHERING: "📊",
                    ReasoningStepType.VALIDATION: "✅",
                    ReasoningStepType.CRITIQUE: "🔍",
                    ReasoningStepType.REFINEMENT: "🔧",
                    ReasoningStepType.SYNTHESIS: "🔗",
                    ReasoningStepType.CONCLUSION: "🎯"
                }.get(step.step_type, "📝")
                
                trace_parts.extend([
                    f"### {step_emoji} Step {i}: {step.step_type.value.title()}",
                    f"**Agent:** {step.agent_name}",
                    f"**Confidence:** {step.confidence:.1%}",
                    f"**Content:** {step.content}",
                    f"**Reasoning:** {step.reasoning}",
                ])
                
                if step.evidence:
                    trace_parts.append(f"**Evidence:** {', '.join(step.evidence)}")
                
                # Add votes for this step
                step_votes = [v for v in chain.votes if v.target_step_id == step.step_id]
                if step_votes:
                    trace_parts.append("**Votes:**")
                    for vote in step_votes:
                        vote_emoji = {
                            VoteType.APPROVE: "👍",
                            VoteType.REJECT: "👎",
                            VoteType.ABSTAIN: "🤷",
                            VoteType.REQUEST_REVISION: "🔄"
                        }.get(vote.vote_type, "❓")
                        
                        trace_parts.append(f"  - {vote_emoji} {vote.agent_name}: {vote.reasoning}")
                
                trace_parts.append("")
            
            # Add final conclusion if available
            if chain.final_conclusion:
                trace_parts.extend([
                    "## Final Conclusion",
                    "",
                    chain.final_conclusion
                ])
            
            return "\n".join(trace_parts)
            
        except Exception as e:
            logger.error(f"Error generating visual trace: {e}")
            return f"Error generating trace: {str(e)}"
    
    def conduct_turn_based_reasoning(self, chain_id: str, max_turns: int = 5) -> Dict[str, Any]:
        """
        Conduct turn-based reasoning between planner and executor agents.
        
        Args:
            chain_id: Chain ID
            max_turns: Maximum number of turns
            
        Returns:
            Reasoning results
        """
        try:
            chain = self.reasoning_chains.get(chain_id)
            if not chain:
                raise ValueError(f"Reasoning chain not found: {chain_id}")
            
            # Identify planner and executor agents
            planner_agents = [p for p in chain.participants if 'planner' in p.lower()]
            executor_agents = [p for p in chain.participants if 'executor' in p.lower()]
            
            if not planner_agents or not executor_agents:
                logger.warning("No clear planner/executor distinction in participants")
                return {'error': 'No planner/executor agents found'}
            
            planner_id = planner_agents[0]
            executor_id = executor_agents[0]
            
            turns = []
            current_agent = planner_id
            
            for turn in range(max_turns):
                # Simulate turn-based reasoning
                if current_agent == planner_id:
                    # Planner turn
                    step_id = self.add_reasoning_step(
                        chain_id=chain_id,
                        agent_id=planner_id,
                        step_type=ReasoningStepType.HYPOTHESIS,
                        content=f"Planning step {turn + 1}: Analyzing approach",
                        reasoning=f"Turn {turn + 1} planning analysis",
                        confidence=0.7
                    )
                    turns.append({'turn': turn + 1, 'agent': 'planner', 'step_id': step_id})
                    current_agent = executor_id
                else:
                    # Executor turn
                    step_id = self.add_reasoning_step(
                        chain_id=chain_id,
                        agent_id=executor_id,
                        step_type=ReasoningStepType.EVIDENCE_GATHERING,
                        content=f"Execution step {turn + 1}: Gathering evidence",
                        reasoning=f"Turn {turn + 1} execution analysis",
                        confidence=0.8
                    )
                    turns.append({'turn': turn + 1, 'agent': 'executor', 'step_id': step_id})
                    current_agent = planner_id
            
            return {
                'chain_id': chain_id,
                'turns_completed': len(turns),
                'turns': turns,
                'final_agent': current_agent
            }
            
        except Exception as e:
            logger.error(f"Error conducting turn-based reasoning: {e}")
            return {'error': str(e)}
    
    def _update_chain_phase(self, chain: ReasoningChain, step_type: ReasoningStepType):
        """Update the current phase of a reasoning chain."""
        phase_mapping = {
            ReasoningStepType.INITIAL_ANALYSIS: "analysis",
            ReasoningStepType.HYPOTHESIS: "hypothesis",
            ReasoningStepType.EVIDENCE_GATHERING: "evidence",
            ReasoningStepType.VALIDATION: "validation",
            ReasoningStepType.CRITIQUE: "critique",
            ReasoningStepType.REFINEMENT: "refinement",
            ReasoningStepType.SYNTHESIS: "synthesis",
            ReasoningStepType.CONCLUSION: "conclusion"
        }
        
        new_phase = phase_mapping.get(step_type, chain.current_phase)
        if new_phase != chain.current_phase:
            chain.current_phase = new_phase
            logger.info(f"Chain {chain.chain_id} moved to phase: {new_phase}")
    
    def _check_voting_completion(self, chain: ReasoningChain, step_id: str):
        """Check if voting is complete for a step."""
        try:
            # Get votes for this step
            step_votes = [v for v in chain.votes if v.target_step_id == step_id]
            
            # Check if all participants have voted
            voted_agents = {v.agent_id for v in step_votes}
            all_participants = set(chain.participants)
            
            if voted_agents == all_participants:
                # Calculate voting results
                approve_votes = sum(1 for v in step_votes if v.vote_type == VoteType.APPROVE)
                total_votes = len(step_votes)
                approval_rate = approve_votes / total_votes if total_votes > 0 else 0
                
                if approval_rate >= self.config['voting_threshold']:
                    logger.info(f"Step {step_id} approved with {approval_rate:.1%} approval")
                else:
                    logger.info(f"Step {step_id} rejected with {approval_rate:.1%} approval")
            
        except Exception as e:
            logger.error(f"Error checking voting completion: {e}")
    
    def _initialize_reasoning_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize reasoning patterns for different scenarios."""
        return {
            'problem_solving': {
                'phases': ['analysis', 'hypothesis', 'evidence', 'validation', 'conclusion'],
                'required_roles': ['planner', 'executor', 'critic'],
                'min_steps': 5
            },
            'document_analysis': {
                'phases': ['analysis', 'evidence', 'synthesis', 'validation'],
                'required_roles': ['executor', 'synthesizer', 'validator'],
                'min_steps': 4
            },
            'research_synthesis': {
                'phases': ['evidence', 'analysis', 'synthesis', 'critique', 'refinement'],
                'required_roles': ['executor', 'synthesizer', 'critic'],
                'min_steps': 5
            }
        }

# Global reasoning engine instance
_reasoning_engine = None

def get_reasoning_engine() -> MultiAgentReasoningEngine:
    """Get or create a global multi-agent reasoning engine instance."""
    global _reasoning_engine
    
    if _reasoning_engine is None:
        _reasoning_engine = MultiAgentReasoningEngine()
    
    return _reasoning_engine
