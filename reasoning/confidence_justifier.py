#!/usr/bin/env python3
"""
Advanced Confidence Justifier for SAM Phase 5
Provides evidence-based justification for confidence scores with transparent reasoning.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence levels with descriptions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class EvidenceType(Enum):
    """Types of evidence for confidence assessment."""
    SOURCE_CREDIBILITY = "source_credibility"
    EVIDENCE_QUANTITY = "evidence_quantity"
    EVIDENCE_QUALITY = "evidence_quality"
    REASONING_COMPLETENESS = "reasoning_completeness"
    DIMENSION_CONSISTENCY = "dimension_consistency"
    EXPERT_VALIDATION = "expert_validation"
    PEER_REVIEW = "peer_review"
    EMPIRICAL_SUPPORT = "empirical_support"

@dataclass
class ConfidenceEvidence:
    """Evidence contributing to confidence assessment."""
    evidence_type: EvidenceType
    score: float  # 0.0 to 1.0
    weight: float  # Importance weight
    description: str
    supporting_details: List[str] = field(default_factory=list)

@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence calculation."""
    component_scores: Dict[str, float]
    component_weights: Dict[str, float]
    weighted_scores: Dict[str, float]
    final_score: float
    calculation_method: str

@dataclass
class ConfidenceJustification:
    """Comprehensive confidence justification with evidence and reasoning."""
    confidence_score: float
    confidence_level: ConfidenceLevel
    
    # Evidence and analysis
    evidence_items: List[ConfidenceEvidence]
    confidence_breakdown: ConfidenceBreakdown
    
    # Explanations
    primary_factors: List[str]
    limiting_factors: List[str]
    reliability_assessment: str
    
    # Metadata
    calculation_timestamp: str
    calculation_duration_ms: int

class AdvancedConfidenceJustifier:
    """
    Advanced confidence justifier that provides evidence-based explanations
    for confidence scores with transparent reasoning and detailed analysis.
    """
    
    def __init__(self, profile: str = "general"):
        """Initialize the confidence justifier."""
        self.profile = profile
        self.evidence_weights = self._initialize_evidence_weights()
        self.confidence_thresholds = self._initialize_confidence_thresholds()
        
        logger.info(f"Advanced Confidence Justifier initialized for {profile} profile")
    
    def _initialize_evidence_weights(self) -> Dict[str, Dict[EvidenceType, float]]:
        """Initialize evidence weights for different profiles."""
        return {
            "general": {
                EvidenceType.SOURCE_CREDIBILITY: 0.25,
                EvidenceType.EVIDENCE_QUANTITY: 0.15,
                EvidenceType.EVIDENCE_QUALITY: 0.20,
                EvidenceType.REASONING_COMPLETENESS: 0.20,
                EvidenceType.DIMENSION_CONSISTENCY: 0.15,
                EvidenceType.EXPERT_VALIDATION: 0.05
            },
            "researcher": {
                EvidenceType.SOURCE_CREDIBILITY: 0.20,
                EvidenceType.EVIDENCE_QUALITY: 0.25,
                EvidenceType.REASONING_COMPLETENESS: 0.20,
                EvidenceType.PEER_REVIEW: 0.15,
                EvidenceType.EMPIRICAL_SUPPORT: 0.15,
                EvidenceType.DIMENSION_CONSISTENCY: 0.05
            },
            "business": {
                EvidenceType.SOURCE_CREDIBILITY: 0.30,
                EvidenceType.EVIDENCE_QUANTITY: 0.20,
                EvidenceType.EXPERT_VALIDATION: 0.20,
                EvidenceType.REASONING_COMPLETENESS: 0.15,
                EvidenceType.EVIDENCE_QUALITY: 0.10,
                EvidenceType.DIMENSION_CONSISTENCY: 0.05
            },
            "legal": {
                EvidenceType.SOURCE_CREDIBILITY: 0.35,
                EvidenceType.EXPERT_VALIDATION: 0.25,
                EvidenceType.EVIDENCE_QUALITY: 0.20,
                EvidenceType.REASONING_COMPLETENESS: 0.15,
                EvidenceType.PEER_REVIEW: 0.05
            }
        }
    
    def _initialize_confidence_thresholds(self) -> Dict[ConfidenceLevel, Tuple[float, float]]:
        """Initialize confidence level thresholds."""
        return {
            ConfidenceLevel.VERY_LOW: (0.0, 0.2),
            ConfidenceLevel.LOW: (0.2, 0.4),
            ConfidenceLevel.MODERATE: (0.4, 0.6),
            ConfidenceLevel.HIGH: (0.6, 0.8),
            ConfidenceLevel.VERY_HIGH: (0.8, 1.0)
        }
    
    def justify_confidence(self, response_analysis: Dict[str, Any], 
                         context: Optional[Dict[str, Any]] = None) -> ConfidenceJustification:
        """
        Generate comprehensive confidence justification with evidence-based reasoning.
        
        Args:
            response_analysis: Analysis of the response including sources, dimensions, etc.
            context: Additional context including critiques, conflicts, etc.
            
        Returns:
            ConfidenceJustification with detailed evidence and explanations
        """
        start_time = time.time()
        
        try:
            # Collect evidence
            evidence_items = self._collect_evidence(response_analysis, context)
            
            # Calculate confidence breakdown
            confidence_breakdown = self._calculate_confidence_breakdown(evidence_items)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(confidence_breakdown.final_score)
            
            # Generate explanations
            primary_factors = self._identify_primary_factors(evidence_items)
            limiting_factors = self._identify_limiting_factors(evidence_items)
            reliability_assessment = self._generate_reliability_assessment(
                confidence_breakdown.final_score, evidence_items
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            justification = ConfidenceJustification(
                confidence_score=confidence_breakdown.final_score,
                confidence_level=confidence_level,
                evidence_items=evidence_items,
                confidence_breakdown=confidence_breakdown,
                primary_factors=primary_factors,
                limiting_factors=limiting_factors,
                reliability_assessment=reliability_assessment,
                calculation_timestamp=datetime.now().isoformat(),
                calculation_duration_ms=duration_ms
            )
            
            logger.info(f"Confidence justified: {confidence_breakdown.final_score:.2f} ({confidence_level.value})")
            return justification
            
        except Exception as e:
            logger.error(f"Error justifying confidence: {e}")
            return self._create_error_justification(str(e))
    
    def _collect_evidence(self, response_analysis: Dict[str, Any], 
                         context: Optional[Dict[str, Any]]) -> List[ConfidenceEvidence]:
        """Collect evidence for confidence assessment."""
        evidence_items = []
        
        # Source credibility evidence
        evidence_items.append(self._assess_source_credibility(response_analysis))
        
        # Evidence quantity
        evidence_items.append(self._assess_evidence_quantity(response_analysis))
        
        # Evidence quality
        evidence_items.append(self._assess_evidence_quality(response_analysis))
        
        # Reasoning completeness
        evidence_items.append(self._assess_reasoning_completeness(response_analysis))
        
        # Dimension consistency
        evidence_items.append(self._assess_dimension_consistency(response_analysis))
        
        # Context-specific evidence
        if context:
            evidence_items.extend(self._assess_context_evidence(context))
        
        return evidence_items
    
    def _assess_source_credibility(self, response_analysis: Dict[str, Any]) -> ConfidenceEvidence:
        """Assess credibility of information sources."""
        sources = response_analysis.get("evidence_sources", [])
        
        if not sources:
            return ConfidenceEvidence(
                evidence_type=EvidenceType.SOURCE_CREDIBILITY,
                score=0.3,
                weight=self._get_evidence_weight(EvidenceType.SOURCE_CREDIBILITY),
                description="No identifiable sources - limited credibility assessment",
                supporting_details=["No source information available"]
            )
        
        # Assess source credibility based on metadata
        credibility_scores = []
        details = []
        
        for source in sources[:5]:  # Limit to top 5 sources
            source_score = 0.5  # Default
            source_details = []
            
            # Check for credibility indicators
            metadata = source.get("metadata", {})
            
            if metadata.get("source_type") == "academic":
                source_score += 0.3
                source_details.append("Academic source")
            
            if metadata.get("peer_reviewed"):
                source_score += 0.2
                source_details.append("Peer reviewed")
            
            if metadata.get("publication_date"):
                # Recent sources get slight boost
                source_score += 0.1
                source_details.append("Recent publication")
            
            credibility_scores.append(min(1.0, source_score))
            details.extend(source_details)
        
        avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.3
        
        return ConfidenceEvidence(
            evidence_type=EvidenceType.SOURCE_CREDIBILITY,
            score=avg_credibility,
            weight=self._get_evidence_weight(EvidenceType.SOURCE_CREDIBILITY),
            description=f"Source credibility assessment based on {len(sources)} sources",
            supporting_details=details
        )
    
    def _assess_evidence_quantity(self, response_analysis: Dict[str, Any]) -> ConfidenceEvidence:
        """Assess quantity of supporting evidence."""
        sources = response_analysis.get("evidence_sources", [])
        tool_outputs = response_analysis.get("tool_outputs", [])
        
        total_evidence = len(sources) + len(tool_outputs)
        
        # Score based on evidence quantity
        if total_evidence >= 5:
            score = 0.9
            description = f"Strong evidence base with {total_evidence} sources"
        elif total_evidence >= 3:
            score = 0.7
            description = f"Adequate evidence with {total_evidence} sources"
        elif total_evidence >= 1:
            score = 0.5
            description = f"Limited evidence with {total_evidence} sources"
        else:
            score = 0.2
            description = "Insufficient evidence - no identifiable sources"
        
        return ConfidenceEvidence(
            evidence_type=EvidenceType.EVIDENCE_QUANTITY,
            score=score,
            weight=self._get_evidence_weight(EvidenceType.EVIDENCE_QUANTITY),
            description=description,
            supporting_details=[f"{len(sources)} memory sources", f"{len(tool_outputs)} tool outputs"]
        )

    def _assess_evidence_quality(self, response_analysis: Dict[str, Any]) -> ConfidenceEvidence:
        """Assess quality of supporting evidence."""
        dimension_scores = response_analysis.get("dimension_scores", {})

        # Use credibility and utility dimensions as quality indicators
        credibility = dimension_scores.get("credibility", 0.5)
        utility = dimension_scores.get("utility", 0.5)

        # Average quality score
        quality_score = (credibility + utility) / 2

        details = [
            f"Credibility dimension: {credibility:.2f}",
            f"Utility dimension: {utility:.2f}"
        ]

        if quality_score > 0.7:
            description = "High-quality evidence with strong credibility and utility"
        elif quality_score > 0.5:
            description = "Moderate-quality evidence with acceptable credibility"
        else:
            description = "Lower-quality evidence with limited credibility assessment"

        return ConfidenceEvidence(
            evidence_type=EvidenceType.EVIDENCE_QUALITY,
            score=quality_score,
            weight=self._get_evidence_weight(EvidenceType.EVIDENCE_QUALITY),
            description=description,
            supporting_details=details
        )

    def _assess_reasoning_completeness(self, response_analysis: Dict[str, Any]) -> ConfidenceEvidence:
        """Assess completeness of reasoning process."""
        assumptions = response_analysis.get("assumptions", [])
        uncertainty_indicators = response_analysis.get("uncertainty_indicators", [])
        confidence_indicators = response_analysis.get("confidence_indicators", [])

        # Penalize for assumptions and uncertainty, reward for confidence indicators
        assumption_penalty = len(assumptions) * 0.1
        uncertainty_penalty = len(uncertainty_indicators) * 0.05
        confidence_boost = len(confidence_indicators) * 0.1

        completeness_score = max(0.0, min(1.0, 0.7 - assumption_penalty - uncertainty_penalty + confidence_boost))

        details = [
            f"{len(assumptions)} assumptions identified",
            f"{len(uncertainty_indicators)} uncertainty indicators",
            f"{len(confidence_indicators)} confidence indicators"
        ]

        if completeness_score > 0.7:
            description = "Complete reasoning with minimal assumptions"
        elif completeness_score > 0.5:
            description = "Adequate reasoning with some limitations"
        else:
            description = "Incomplete reasoning with significant assumptions"

        return ConfidenceEvidence(
            evidence_type=EvidenceType.REASONING_COMPLETENESS,
            score=completeness_score,
            weight=self._get_evidence_weight(EvidenceType.REASONING_COMPLETENESS),
            description=description,
            supporting_details=details
        )

    def _assess_dimension_consistency(self, response_analysis: Dict[str, Any]) -> ConfidenceEvidence:
        """Assess consistency across conceptual dimensions."""
        dimension_scores = response_analysis.get("dimension_scores", {})

        if not dimension_scores:
            return ConfidenceEvidence(
                evidence_type=EvidenceType.DIMENSION_CONSISTENCY,
                score=0.5,
                weight=self._get_evidence_weight(EvidenceType.DIMENSION_CONSISTENCY),
                description="No dimension analysis available",
                supporting_details=["Dimension scoring not performed"]
            )

        # Calculate variance in dimension scores
        scores = list(dimension_scores.values())
        if len(scores) < 2:
            consistency_score = 0.7  # Default for single dimension
            variance = 0.0
            mean_score = scores[0] if scores else 0.5
        else:
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            consistency_score = max(0.0, 1.0 - variance)  # Higher consistency = lower variance

        details = [
            f"{len(dimension_scores)} dimensions analyzed",
            f"Score variance: {variance:.3f}",
            f"Mean dimension score: {mean_score:.2f}"
        ]

        if consistency_score > 0.8:
            description = "High consistency across conceptual dimensions"
        elif consistency_score > 0.6:
            description = "Moderate consistency with some variation"
        else:
            description = "Low consistency - conflicting dimension scores"

        return ConfidenceEvidence(
            evidence_type=EvidenceType.DIMENSION_CONSISTENCY,
            score=consistency_score,
            weight=self._get_evidence_weight(EvidenceType.DIMENSION_CONSISTENCY),
            description=description,
            supporting_details=details
        )

    def _assess_context_evidence(self, context: Dict[str, Any]) -> List[ConfidenceEvidence]:
        """Assess context-specific evidence."""
        evidence_items = []

        # Check for critiques
        critiques = context.get("critiques", [])
        if critiques:
            critique_penalty = sum(c.get("severity", 0.5) for c in critiques) / len(critiques)
            critique_score = max(0.0, 1.0 - critique_penalty)

            evidence_items.append(ConfidenceEvidence(
                evidence_type=EvidenceType.PEER_REVIEW,
                score=critique_score,
                weight=self._get_evidence_weight(EvidenceType.PEER_REVIEW),
                description=f"Adversarial critique assessment with {len(critiques)} critiques",
                supporting_details=[f"Average critique severity: {critique_penalty:.2f}"]
            ))

        return evidence_items

    def _get_evidence_weight(self, evidence_type: EvidenceType) -> float:
        """Get weight for evidence type based on profile."""
        profile_weights = self.evidence_weights.get(self.profile, self.evidence_weights["general"])
        return profile_weights.get(evidence_type, 0.1)

    def _calculate_confidence_breakdown(self, evidence_items: List[ConfidenceEvidence]) -> ConfidenceBreakdown:
        """Calculate detailed confidence breakdown."""
        component_scores = {}
        component_weights = {}
        weighted_scores = {}

        total_weight = 0.0
        weighted_sum = 0.0

        for evidence in evidence_items:
            component_name = evidence.evidence_type.value
            component_scores[component_name] = evidence.score
            component_weights[component_name] = evidence.weight

            weighted_score = evidence.score * evidence.weight
            weighted_scores[component_name] = weighted_score

            total_weight += evidence.weight
            weighted_sum += weighted_score

        # Normalize if total weight != 1.0
        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = 0.5  # Default

        return ConfidenceBreakdown(
            component_scores=component_scores,
            component_weights=component_weights,
            weighted_scores=weighted_scores,
            final_score=final_score,
            calculation_method="weighted_average"
        )

    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from score."""
        for level, (min_score, max_score) in self.confidence_thresholds.items():
            if min_score <= score < max_score:
                return level
        return ConfidenceLevel.VERY_HIGH  # For score = 1.0

    def _identify_primary_factors(self, evidence_items: List[ConfidenceEvidence]) -> List[str]:
        """Identify primary factors contributing to confidence."""
        # Sort by weighted contribution
        weighted_contributions = [
            (evidence.score * evidence.weight, evidence.description)
            for evidence in evidence_items
        ]
        weighted_contributions.sort(reverse=True)

        # Return top contributing factors
        return [desc for _, desc in weighted_contributions[:3]]

    def _identify_limiting_factors(self, evidence_items: List[ConfidenceEvidence]) -> List[str]:
        """Identify factors limiting confidence."""
        limiting_factors = []

        for evidence in evidence_items:
            if evidence.score < 0.5:
                limiting_factors.append(f"Low {evidence.evidence_type.value}: {evidence.description}")

        return limiting_factors

    def _generate_reliability_assessment(self, confidence_score: float,
                                       evidence_items: List[ConfidenceEvidence]) -> str:
        """Generate overall reliability assessment."""
        if confidence_score >= 0.8:
            return "High reliability - strong evidence across multiple dimensions with minimal limitations"
        elif confidence_score >= 0.6:
            return "Good reliability - adequate evidence with some minor limitations or uncertainties"
        elif confidence_score >= 0.4:
            return "Moderate reliability - mixed evidence quality with notable limitations"
        elif confidence_score >= 0.2:
            return "Low reliability - limited evidence with significant concerns or gaps"
        else:
            return "Very low reliability - insufficient or poor-quality evidence with major limitations"

    def _create_error_justification(self, error: str) -> ConfidenceJustification:
        """Create error justification when calculation fails."""
        return ConfidenceJustification(
            confidence_score=0.1,
            confidence_level=ConfidenceLevel.VERY_LOW,
            evidence_items=[],
            confidence_breakdown=ConfidenceBreakdown(
                component_scores={"error": 0.1},
                component_weights={"error": 1.0},
                weighted_scores={"error": 0.1},
                final_score=0.1,
                calculation_method="error_fallback"
            ),
            primary_factors=[f"Error in confidence calculation: {error}"],
            limiting_factors=["Confidence calculation failed"],
            reliability_assessment="Unreliable due to calculation error",
            calculation_timestamp=datetime.now().isoformat(),
            calculation_duration_ms=0
        )


# Convenience functions
def justify_response_confidence(response_analysis: Dict[str, Any],
                              context: Optional[Dict[str, Any]] = None,
                              profile: str = "general") -> ConfidenceJustification:
    """Convenience function for justifying response confidence."""
    justifier = AdvancedConfidenceJustifier(profile)
    return justifier.justify_confidence(response_analysis, context)
