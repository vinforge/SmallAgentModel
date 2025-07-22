#!/usr/bin/env python3
"""
SELF-DISCOVER + CRITIC Framework for SAM
Implements dual-agent feedback loop for self-correction and knowledge gap identification.

This framework enables SAM to:
1. SELF-DISCOVER: Identify knowledge gaps and weak assumptions
2. CRITIC: Challenge conclusions and improve confidence calibration
3. Self-correct responses through iterative refinement
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DiscoveryPhase(Enum):
    """Phases of the SELF-DISCOVER process."""
    INITIAL_RESPONSE = "initial_response"
    KNOWLEDGE_GAP_ANALYSIS = "knowledge_gap_analysis"
    ASSUMPTION_IDENTIFICATION = "assumption_identification"
    EVIDENCE_EVALUATION = "evidence_evaluation"
    CONFIDENCE_CALIBRATION = "confidence_calibration"

class CriticPhase(Enum):
    """Phases of the CRITIC process."""
    CHALLENGE_ASSUMPTIONS = "challenge_assumptions"
    IDENTIFY_WEAKNESSES = "identify_weaknesses"
    ALTERNATIVE_PERSPECTIVES = "alternative_perspectives"
    EVIDENCE_SCRUTINY = "evidence_scrutiny"
    FINAL_VALIDATION = "final_validation"

@dataclass
class KnowledgeGap:
    """Represents an identified knowledge gap."""
    gap_id: str
    description: str
    severity: float  # 0.0 to 1.0
    evidence_needed: List[str]
    confidence_impact: float
    timestamp: str

@dataclass
class CriticalChallenge:
    """Represents a critical challenge to an assumption or conclusion."""
    challenge_id: str
    target_assumption: str
    challenge_type: str  # "logical", "evidential", "alternative"
    challenge_text: str
    strength: float  # 0.0 to 1.0
    requires_revision: bool
    timestamp: str

@dataclass
class SelfDiscoverResult:
    """Result of the SELF-DISCOVER process."""
    session_id: str
    original_query: str
    initial_response: str
    knowledge_gaps: List[KnowledgeGap]
    assumptions: List[str]
    confidence_score: float
    evidence_quality: float
    improvement_suggestions: List[str]
    timestamp: str

@dataclass
class CriticResult:
    """Result of the CRITIC process."""
    session_id: str
    challenges: List[CriticalChallenge]
    revised_response: str
    confidence_adjustment: float
    validation_score: float
    final_confidence: float
    revision_needed: bool
    timestamp: str

@dataclass
class SelfDiscoverCriticSession:
    """Complete session combining SELF-DISCOVER and CRITIC."""
    session_id: str
    query: str
    context: str
    discover_result: Optional[SelfDiscoverResult]
    critic_result: Optional[CriticResult]
    final_response: str
    final_confidence: float
    iteration_count: int
    total_duration_ms: int
    timestamp: str

class SelfDiscoverCriticFramework:
    """
    Implements the SELF-DISCOVER + CRITIC dual-agent framework.
    
    This framework enables SAM to:
    1. Generate initial responses
    2. Identify knowledge gaps and assumptions (SELF-DISCOVER)
    3. Challenge and critique its own reasoning (CRITIC)
    4. Iteratively improve responses through self-correction
    """
    
    def __init__(self, max_iterations: int = 3, confidence_threshold: float = 0.8):
        """
        Initialize the SELF-DISCOVER + CRITIC framework.
        
        Args:
            max_iterations: Maximum number of self-correction iterations
            confidence_threshold: Minimum confidence score to stop iterations
        """
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.active_sessions: Dict[str, SelfDiscoverCriticSession] = {}
        
        logger.info("SELF-DISCOVER + CRITIC Framework initialized")
        logger.info(f"Max iterations: {max_iterations}, Confidence threshold: {confidence_threshold}")
    
    def process_with_self_discover_critic(self, query: str, context: str = "", 
                                        memory_results: List = None) -> SelfDiscoverCriticSession:
        """
        Process a query using the full SELF-DISCOVER + CRITIC framework.
        
        Args:
            query: User query to process
            context: Additional context for the query
            memory_results: Relevant memory search results
            
        Returns:
            Complete session with self-corrected response
        """
        session_id = f"sdc_{int(datetime.now().timestamp())}"
        start_time = datetime.now()
        
        logger.info(f"Starting SELF-DISCOVER + CRITIC session: {session_id}")
        
        # Initialize session
        session = SelfDiscoverCriticSession(
            session_id=session_id,
            query=query,
            context=context,
            discover_result=None,
            critic_result=None,
            final_response="",
            final_confidence=0.0,
            iteration_count=0,
            total_duration_ms=0,
            timestamp=datetime.now().isoformat()
        )
        
        # Generate initial response
        initial_response = self._generate_initial_response(query, context, memory_results)
        
        current_response = initial_response
        current_confidence = 0.5  # Start with moderate confidence
        
        # Iterative self-correction loop
        for iteration in range(self.max_iterations):
            session.iteration_count = iteration + 1
            
            logger.info(f"SELF-DISCOVER + CRITIC iteration {iteration + 1}")
            
            # SELF-DISCOVER phase
            discover_result = self._self_discover_phase(
                session_id, query, current_response, context, memory_results
            )
            session.discover_result = discover_result
            
            # CRITIC phase
            critic_result = self._critic_phase(
                session_id, query, current_response, discover_result
            )
            session.critic_result = critic_result
            
            # Update response and confidence
            if critic_result.revision_needed:
                current_response = critic_result.revised_response
                current_confidence = critic_result.final_confidence
                
                logger.info(f"Response revised. New confidence: {current_confidence:.3f}")
            else:
                current_confidence = critic_result.final_confidence
                logger.info(f"No revision needed. Final confidence: {current_confidence:.3f}")
            
            # Check if we've reached sufficient confidence
            if current_confidence >= self.confidence_threshold:
                logger.info(f"Confidence threshold reached: {current_confidence:.3f}")
                break
        
        # Finalize session
        session.final_response = current_response
        session.final_confidence = current_confidence
        
        end_time = datetime.now()
        session.total_duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Store session
        self.active_sessions[session_id] = session
        
        logger.info(f"SELF-DISCOVER + CRITIC completed: {session_id} ({session.total_duration_ms}ms)")
        logger.info(f"Final confidence: {session.final_confidence:.3f}, Iterations: {session.iteration_count}")
        
        return session
    
    def _generate_initial_response(self, query: str, context: str, memory_results: List) -> str:
        """Generate initial response before self-correction."""
        try:
            # This would integrate with SAM's existing response generation
            # For now, return a placeholder that indicates the framework is working
            return f"Initial response to: {query}\n\n[This response will be improved through SELF-DISCOVER + CRITIC]"
        except Exception as e:
            logger.error(f"Error generating initial response: {e}")
            return f"Error generating response to: {query}"
    
    def _self_discover_phase(self, session_id: str, query: str, response: str, 
                           context: str, memory_results: List) -> SelfDiscoverResult:
        """
        Execute the SELF-DISCOVER phase to identify knowledge gaps and assumptions.
        
        Args:
            session_id: Current session ID
            query: Original query
            response: Current response to analyze
            context: Additional context
            memory_results: Memory search results
            
        Returns:
            SELF-DISCOVER analysis results
        """
        logger.info(f"Executing SELF-DISCOVER phase for session: {session_id}")
        
        # Identify knowledge gaps
        knowledge_gaps = self._identify_knowledge_gaps(query, response, memory_results)
        
        # Identify assumptions
        assumptions = self._identify_assumptions(response)
        
        # Evaluate evidence quality
        evidence_quality = self._evaluate_evidence_quality(response, memory_results)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(knowledge_gaps, evidence_quality)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            knowledge_gaps, assumptions, evidence_quality
        )
        
        result = SelfDiscoverResult(
            session_id=session_id,
            original_query=query,
            initial_response=response,
            knowledge_gaps=knowledge_gaps,
            assumptions=assumptions,
            confidence_score=confidence_score,
            evidence_quality=evidence_quality,
            improvement_suggestions=improvement_suggestions,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"SELF-DISCOVER identified {len(knowledge_gaps)} gaps, {len(assumptions)} assumptions")
        logger.info(f"Evidence quality: {evidence_quality:.3f}, Confidence: {confidence_score:.3f}")
        
        return result
    
    def _critic_phase(self, session_id: str, query: str, response: str, 
                     discover_result: SelfDiscoverResult) -> CriticResult:
        """
        Execute the CRITIC phase to challenge assumptions and improve the response.
        
        Args:
            session_id: Current session ID
            query: Original query
            response: Current response to critique
            discover_result: Results from SELF-DISCOVER phase
            
        Returns:
            CRITIC analysis and revised response
        """
        logger.info(f"Executing CRITIC phase for session: {session_id}")
        
        # Generate critical challenges
        challenges = self._generate_critical_challenges(response, discover_result)
        
        # Determine if revision is needed
        revision_needed = any(challenge.requires_revision for challenge in challenges)
        
        # Generate revised response if needed
        if revision_needed:
            revised_response = self._generate_revised_response(
                query, response, challenges, discover_result
            )
        else:
            revised_response = response
        
        # Calculate confidence adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(challenges)
        
        # Calculate final confidence
        final_confidence = max(0.0, min(1.0, 
            discover_result.confidence_score + confidence_adjustment
        ))
        
        # Calculate validation score
        validation_score = self._calculate_validation_score(challenges, discover_result)
        
        result = CriticResult(
            session_id=session_id,
            challenges=challenges,
            revised_response=revised_response,
            confidence_adjustment=confidence_adjustment,
            validation_score=validation_score,
            final_confidence=final_confidence,
            revision_needed=revision_needed,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"CRITIC generated {len(challenges)} challenges")
        logger.info(f"Revision needed: {revision_needed}, Final confidence: {final_confidence:.3f}")

        return result

    def _identify_knowledge_gaps(self, query: str, response: str, memory_results: List) -> List[KnowledgeGap]:
        """Identify knowledge gaps in the current response."""
        gaps = []

        # Check for vague or uncertain language
        uncertainty_indicators = ["might", "could", "possibly", "perhaps", "unclear", "unknown"]
        if any(indicator in response.lower() for indicator in uncertainty_indicators):
            gaps.append(KnowledgeGap(
                gap_id=f"gap_uncertainty_{int(datetime.now().timestamp())}",
                description="Response contains uncertainty indicators",
                severity=0.6,
                evidence_needed=["More specific information", "Concrete examples"],
                confidence_impact=0.3,
                timestamp=datetime.now().isoformat()
            ))

        # Check for lack of specific details
        if len(response.split()) < 20:  # Very short response
            gaps.append(KnowledgeGap(
                gap_id=f"gap_brevity_{int(datetime.now().timestamp())}",
                description="Response lacks sufficient detail",
                severity=0.5,
                evidence_needed=["More comprehensive information", "Additional context"],
                confidence_impact=0.2,
                timestamp=datetime.now().isoformat()
            ))

        # Check for missing memory integration
        if memory_results and len(memory_results) > 0:
            memory_content = " ".join([str(result) for result in memory_results])
            if not any(word in response.lower() for word in memory_content.lower().split()[:10]):
                gaps.append(KnowledgeGap(
                    gap_id=f"gap_memory_{int(datetime.now().timestamp())}",
                    description="Response doesn't integrate available memory content",
                    severity=0.7,
                    evidence_needed=["Integration of memory search results"],
                    confidence_impact=0.4,
                    timestamp=datetime.now().isoformat()
                ))

        return gaps

    def _identify_assumptions(self, response: str) -> List[str]:
        """Identify assumptions made in the response."""
        assumptions = []

        # Look for assumption indicators
        assumption_patterns = [
            "assuming", "given that", "if we consider", "based on the premise",
            "it's likely that", "we can assume", "presumably"
        ]

        for pattern in assumption_patterns:
            if pattern in response.lower():
                # Extract the sentence containing the assumption
                sentences = response.split('.')
                for sentence in sentences:
                    if pattern in sentence.lower():
                        assumptions.append(sentence.strip())

        # Add implicit assumptions
        if "document" in response.lower() or "pdf" in response.lower():
            assumptions.append("Assumes user has uploaded a document")

        if "recent" in response.lower() or "latest" in response.lower():
            assumptions.append("Assumes temporal relevance is important")

        return assumptions

    def _evaluate_evidence_quality(self, response: str, memory_results: List) -> float:
        """Evaluate the quality of evidence supporting the response."""
        quality_score = 0.5  # Start with neutral

        # Check for specific citations or sources
        if "source:" in response.lower() or "according to" in response.lower():
            quality_score += 0.2

        # Check for quantitative information
        import re
        if re.search(r'\d+', response):
            quality_score += 0.1

        # Check memory integration
        if memory_results and len(memory_results) > 0:
            quality_score += 0.2

        # Check for balanced perspective
        if "however" in response.lower() or "on the other hand" in response.lower():
            quality_score += 0.1

        return min(1.0, quality_score)

    def _calculate_confidence_score(self, knowledge_gaps: List[KnowledgeGap], evidence_quality: float) -> float:
        """Calculate confidence score based on gaps and evidence."""
        base_confidence = 0.7

        # Reduce confidence for each knowledge gap
        gap_penalty = sum(gap.confidence_impact for gap in knowledge_gaps)

        # Adjust for evidence quality
        evidence_bonus = (evidence_quality - 0.5) * 0.3

        final_confidence = base_confidence - gap_penalty + evidence_bonus
        return max(0.0, min(1.0, final_confidence))

    def _generate_improvement_suggestions(self, knowledge_gaps: List[KnowledgeGap],
                                        assumptions: List[str], evidence_quality: float) -> List[str]:
        """Generate suggestions for improving the response."""
        suggestions = []

        if knowledge_gaps:
            suggestions.append(f"Address {len(knowledge_gaps)} identified knowledge gaps")

        if assumptions:
            suggestions.append(f"Validate or clarify {len(assumptions)} assumptions")

        if evidence_quality < 0.6:
            suggestions.append("Strengthen evidence with more specific sources")

        suggestions.append("Consider alternative perspectives")
        suggestions.append("Provide more specific examples")

        return suggestions

    def _generate_critical_challenges(self, response: str, discover_result: SelfDiscoverResult) -> List[CriticalChallenge]:
        """Generate critical challenges to the response."""
        challenges = []

        # Challenge assumptions
        for i, assumption in enumerate(discover_result.assumptions):
            challenges.append(CriticalChallenge(
                challenge_id=f"challenge_assumption_{i}",
                target_assumption=assumption,
                challenge_type="logical",
                challenge_text=f"Is this assumption necessarily true: {assumption}?",
                strength=0.6,
                requires_revision=len(assumption.split()) > 5,  # Longer assumptions need more scrutiny
                timestamp=datetime.now().isoformat()
            ))

        # Challenge evidence quality
        if discover_result.evidence_quality < 0.6:
            challenges.append(CriticalChallenge(
                challenge_id="challenge_evidence",
                target_assumption="Evidence quality",
                challenge_type="evidential",
                challenge_text="The evidence supporting this response is insufficient",
                strength=0.8,
                requires_revision=True,
                timestamp=datetime.now().isoformat()
            ))

        # Challenge knowledge gaps
        for gap in discover_result.knowledge_gaps:
            if gap.severity > 0.6:
                challenges.append(CriticalChallenge(
                    challenge_id=f"challenge_{gap.gap_id}",
                    target_assumption=gap.description,
                    challenge_type="alternative",
                    challenge_text=f"This knowledge gap undermines the response: {gap.description}",
                    strength=gap.severity,
                    requires_revision=gap.severity > 0.7,
                    timestamp=datetime.now().isoformat()
                ))

        return challenges

    def _generate_revised_response(self, query: str, original_response: str,
                                 challenges: List[CriticalChallenge],
                                 discover_result: SelfDiscoverResult) -> str:
        """Generate a revised response addressing the critical challenges."""
        # For now, return an improved version with acknowledgment of limitations
        revision_notes = []

        for challenge in challenges:
            if challenge.requires_revision:
                revision_notes.append(f"• Addressed: {challenge.challenge_text}")

        revised = f"{original_response}\n\n**Self-Correction Applied:**\n"
        revised += "\n".join(revision_notes)
        revised += f"\n\n**Confidence Level:** {discover_result.confidence_score:.1%}"

        return revised

    def _calculate_confidence_adjustment(self, challenges: List[CriticalChallenge]) -> float:
        """Calculate confidence adjustment based on critical challenges."""
        adjustment = 0.0

        for challenge in challenges:
            if challenge.requires_revision:
                adjustment -= challenge.strength * 0.1  # Reduce confidence
            else:
                adjustment += 0.05  # Slight boost for addressing challenges

        return adjustment

    def _calculate_validation_score(self, challenges: List[CriticalChallenge],
                                  discover_result: SelfDiscoverResult) -> float:
        """Calculate overall validation score."""
        base_score = discover_result.confidence_score

        # Adjust for challenges addressed
        challenges_addressed = sum(1 for c in challenges if not c.requires_revision)
        total_challenges = len(challenges)

        if total_challenges > 0:
            challenge_score = challenges_addressed / total_challenges
            base_score = (base_score + challenge_score) / 2

        return base_score

# Global instance
_self_discover_critic_framework = None

def get_self_discover_critic_framework() -> SelfDiscoverCriticFramework:
    """Get or create the global SELF-DISCOVER + CRITIC framework instance."""
    global _self_discover_critic_framework

    if _self_discover_critic_framework is None:
        _self_discover_critic_framework = SelfDiscoverCriticFramework()

    return _self_discover_critic_framework
