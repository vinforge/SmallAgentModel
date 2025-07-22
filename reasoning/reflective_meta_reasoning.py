#!/usr/bin/env python3
"""
Phase 5: Reflective Meta-Reasoning & Self-Aware Critique System
Implements introspective, self-monitoring reasoning with meta-cognitive capabilities.

This system enables SAM to:
1. Reflect on its own reasoning processes
2. Generate adversarial critiques of its responses
3. Detect and explain dimension conflicts
4. Justify confidence scores with evidence
5. Narrate trade-offs transparently
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Import existing SAM components
try:
    from .self_discover_critic import SelfDiscoverCriticFramework, SelfDiscoverResult, CriticResult
    SELF_DISCOVER_AVAILABLE = True
except ImportError:
    SELF_DISCOVER_AVAILABLE = False
    logging.warning("Self-Discover framework not available for meta-reasoning")

# Import Phase 2 dimension components
try:
    from multimodal_processing.dimension_prober_v2 import EnhancedDimensionProberV2
    DIMENSION_PROBING_AVAILABLE = True
except ImportError:
    DIMENSION_PROBING_AVAILABLE = False
    logging.warning("Dimension probing not available for meta-reasoning")

logger = logging.getLogger(__name__)

class ReflectionType(Enum):
    """Types of reflective analysis."""
    INITIAL_ANALYSIS = "initial_analysis"
    ALTERNATIVE_PERSPECTIVES = "alternative_perspectives"
    ADVERSARIAL_CRITIQUE = "adversarial_critique"
    DIMENSION_CONFLICTS = "dimension_conflicts"
    CONFIDENCE_JUSTIFICATION = "confidence_justification"
    TRADE_OFF_ANALYSIS = "trade_off_analysis"

class CritiqueLevel(Enum):
    """Levels of critique intensity."""
    GENTLE = "gentle"
    MODERATE = "moderate"
    RIGOROUS = "rigorous"
    ADVERSARIAL = "adversarial"

@dataclass
class DimensionConflict:
    """Represents a conflict between conceptual dimensions."""
    conflict_type: str
    description: str
    dimensions: List[str]
    severity: float  # 0.0 to 1.0
    recommendation: str
    evidence: List[str] = field(default_factory=list)

@dataclass
class AlternativePerspective:
    """Alternative interpretation or approach."""
    perspective_name: str
    description: str
    reasoning: str
    confidence: float
    trade_offs: List[str]
    supporting_evidence: List[str] = field(default_factory=list)

@dataclass
class ConfidenceJustification:
    """Detailed justification for confidence scores."""
    confidence_score: float
    contributing_factors: List[str]
    confidence_breakdown: Dict[str, float]
    reliability_assessment: str
    evidence_quality: float
    reasoning_completeness: float

@dataclass
class AdversarialCritique:
    """Adversarial critique of a response."""
    critique_id: str
    critique_text: str
    counter_arguments: List[str]
    edge_cases: List[str]
    bias_warnings: List[str]
    alternative_interpretations: List[str]
    severity: float
    requires_revision: bool

@dataclass
class ReflectiveResult:
    """Complete result of reflective meta-reasoning."""
    session_id: str
    original_query: str
    initial_response: str
    
    # Reflective analysis components
    response_analysis: Dict[str, Any]
    alternative_perspectives: List[AlternativePerspective]
    adversarial_critiques: List[AdversarialCritique]
    dimension_conflicts: List[DimensionConflict]
    confidence_justification: ConfidenceJustification
    trade_off_analysis: Dict[str, Any]
    
    # Final synthesis
    final_response: str
    reasoning_chain: List[Dict[str, Any]]
    critique_summary: str
    meta_confidence: float
    
    # Metadata
    reflection_duration_ms: int
    timestamp: str

class ReflectiveMetaReasoningEngine:
    """
    Revolutionary meta-cognitive reasoning engine that enables SAM to reflect on
    its own reasoning, critique responses, and explain trade-offs transparently.
    """
    
    def __init__(self, critique_level: CritiqueLevel = CritiqueLevel.MODERATE):
        """Initialize the reflective meta-reasoning engine."""
        self.critique_level = critique_level
        
        # Initialize existing components if available
        if SELF_DISCOVER_AVAILABLE:
            self.self_discover_framework = SelfDiscoverCriticFramework()
            logger.info("Self-Discover framework integrated for meta-reasoning")
        else:
            self.self_discover_framework = None
        
        if DIMENSION_PROBING_AVAILABLE:
            self.dimension_prober = EnhancedDimensionProberV2()
            logger.info("Dimension probing integrated for meta-reasoning")
        else:
            self.dimension_prober = None
        
        # Initialize reflection components
        self._init_reflection_patterns()
        self._init_critique_templates()
        
        logger.info(f"Reflective Meta-Reasoning Engine initialized with {critique_level.value} critique level")
    
    def _init_reflection_patterns(self):
        """Initialize patterns for reflective analysis."""
        self.reflection_patterns = {
            "assumption_detection": [
                r"assumes?\s+that",
                r"presupposes?\s+",
                r"takes?\s+for\s+granted",
                r"given\s+that",
                r"if\s+we\s+assume"
            ],
            "uncertainty_indicators": [
                r"might\s+be",
                r"could\s+be",
                r"possibly",
                r"potentially",
                r"unclear",
                r"uncertain"
            ],
            "confidence_indicators": [
                r"definitely",
                r"certainly",
                r"clearly",
                r"obviously",
                r"undoubtedly"
            ]
        }
    
    def _init_critique_templates(self):
        """Initialize templates for adversarial critique."""
        self.critique_templates = {
            CritiqueLevel.GENTLE: {
                "counter_argument": "Consider an alternative perspective: {alternative}",
                "bias_warning": "This response might reflect a bias toward {bias_type}",
                "edge_case": "This approach might not work well when {edge_case}"
            },
            CritiqueLevel.MODERATE: {
                "counter_argument": "A significant counter-argument is: {alternative}",
                "bias_warning": "Warning: This response shows potential {bias_type} bias",
                "edge_case": "Critical limitation: Fails to address {edge_case}"
            },
            CritiqueLevel.RIGOROUS: {
                "counter_argument": "Strong counter-evidence suggests: {alternative}",
                "bias_warning": "Serious bias detected: {bias_type} - requires reconsideration",
                "edge_case": "Fatal flaw: Completely fails when {edge_case}"
            },
            CritiqueLevel.ADVERSARIAL: {
                "counter_argument": "This response is fundamentally flawed because: {alternative}",
                "bias_warning": "Dangerous bias: {bias_type} - response should be rejected",
                "edge_case": "Catastrophic failure mode: {edge_case} - completely unreliable"
            }
        }
    
    def reflective_reasoning_cycle(self, query: str, initial_response: str, 
                                 context: Optional[Dict[str, Any]] = None) -> ReflectiveResult:
        """
        Execute complete reflective reasoning cycle with meta-cognitive analysis.
        
        Args:
            query: Original user query
            initial_response: Initial response to analyze and improve
            context: Additional context including memory results, tool outputs, etc.
            
        Returns:
            ReflectiveResult with comprehensive meta-reasoning analysis
        """
        start_time = time.time()
        session_id = f"reflect_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"Starting reflective reasoning cycle for session: {session_id}")
            
            # Stage 1: Initial Response Analysis
            response_analysis = self._analyze_initial_response(initial_response, context)
            
            # Stage 2: Generate Alternative Perspectives
            alternatives = self._generate_alternative_perspectives(query, initial_response, context)
            
            # Stage 3: Adversarial Critique
            critiques = self._generate_adversarial_critiques(initial_response, alternatives, context)
            
            # Stage 4: Dimension Conflict Detection
            conflicts = self._detect_dimension_conflicts(response_analysis, context)
            
            # Stage 5: Confidence Justification
            confidence_justification = self._justify_confidence(response_analysis, critiques, context)
            
            # Stage 6: Trade-off Analysis
            trade_offs = self._analyze_trade_offs(alternatives, conflicts, context)
            
            # Stage 7: Synthesize Final Response
            final_response = self._synthesize_final_response(
                query, initial_response, alternatives, critiques, conflicts, trade_offs
            )
            
            # Stage 8: Build Reasoning Chain
            reasoning_chain = self._build_reasoning_chain(
                response_analysis, alternatives, critiques, conflicts, confidence_justification, trade_offs
            )
            
            # Stage 9: Generate Critique Summary
            critique_summary = self._generate_critique_summary(critiques, conflicts, trade_offs)
            
            # Stage 10: Calculate Meta-Confidence
            meta_confidence = self._calculate_meta_confidence(
                confidence_justification, critiques, conflicts
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            result = ReflectiveResult(
                session_id=session_id,
                original_query=query,
                initial_response=initial_response,
                response_analysis=response_analysis,
                alternative_perspectives=alternatives,
                adversarial_critiques=critiques,
                dimension_conflicts=conflicts,
                confidence_justification=confidence_justification,
                trade_off_analysis=trade_offs,
                final_response=final_response,
                reasoning_chain=reasoning_chain,
                critique_summary=critique_summary,
                meta_confidence=meta_confidence,
                reflection_duration_ms=duration_ms,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Reflective reasoning completed in {duration_ms}ms with {meta_confidence:.2f} meta-confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error in reflective reasoning cycle: {e}")
            # Return minimal result with error information
            return self._create_error_result(session_id, query, initial_response, str(e))
    
    def _analyze_initial_response(self, response: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the initial response for assumptions, confidence indicators, etc."""
        analysis = {
            "response_length": len(response),
            "assumptions": self._detect_assumptions(response),
            "uncertainty_indicators": self._detect_uncertainty_indicators(response),
            "confidence_indicators": self._detect_confidence_indicators(response),
            "dimension_scores": {},
            "evidence_sources": [],
            "reasoning_gaps": []
        }
        
        # Add dimension analysis if available
        if self.dimension_prober and context:
            try:
                dimension_result = self.dimension_prober.probe_chunk(response)
                analysis["dimension_scores"] = dimension_result.scores.scores
                # Handle different confidence attribute names
                if hasattr(dimension_result, 'confidence'):
                    if hasattr(dimension_result.confidence, 'confidence_scores'):
                        analysis["dimension_confidence"] = dimension_result.confidence.confidence_scores
                    else:
                        analysis["dimension_confidence"] = dimension_result.confidence
                else:
                    analysis["dimension_confidence"] = {}
            except Exception as e:
                logger.warning(f"Dimension analysis failed: {e}")
        
        # Extract evidence sources from context
        if context:
            analysis["evidence_sources"] = context.get("memory_results", [])
            analysis["tool_outputs"] = context.get("tool_outputs", [])
        
        return analysis

    def _detect_assumptions(self, response: str) -> List[str]:
        """Detect assumptions in the response text."""
        import re
        assumptions = []

        for pattern in self.reflection_patterns["assumption_detection"]:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context
                start = max(0, match.start() - 50)
                end = min(len(response), match.end() + 100)
                context = response[start:end].strip()
                assumptions.append(context)

        return assumptions

    def _detect_uncertainty_indicators(self, response: str) -> List[str]:
        """Detect uncertainty indicators in the response."""
        import re
        indicators = []

        for pattern in self.reflection_patterns["uncertainty_indicators"]:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(response), match.end() + 50)
                context = response[start:end].strip()
                indicators.append(context)

        return indicators

    def _detect_confidence_indicators(self, response: str) -> List[str]:
        """Detect confidence indicators in the response."""
        import re
        indicators = []

        for pattern in self.reflection_patterns["confidence_indicators"]:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(response), match.end() + 50)
                context = response[start:end].strip()
                indicators.append(context)

        return indicators

    def _generate_alternative_perspectives(self, query: str, response: str,
                                         context: Optional[Dict[str, Any]]) -> List[AlternativePerspective]:
        """Generate alternative perspectives and interpretations."""
        alternatives = []

        # Generate different reasoning profiles perspectives
        profiles = ["researcher", "business", "legal", "general"]

        for profile in profiles:
            try:
                alternative = self._generate_profile_perspective(query, response, profile, context)
                if alternative:
                    alternatives.append(alternative)
            except Exception as e:
                logger.warning(f"Failed to generate {profile} perspective: {e}")

        # Generate contrarian perspective
        try:
            contrarian = self._generate_contrarian_perspective(query, response, context)
            if contrarian:
                alternatives.append(contrarian)
        except Exception as e:
            logger.warning(f"Failed to generate contrarian perspective: {e}")

        return alternatives

    def _generate_profile_perspective(self, query: str, response: str, profile: str,
                                    context: Optional[Dict[str, Any]]) -> Optional[AlternativePerspective]:
        """Generate perspective from a specific reasoning profile."""

        profile_descriptions = {
            "researcher": "Focus on innovation, methodology, and evidence quality",
            "business": "Emphasize ROI, feasibility, and market impact",
            "legal": "Prioritize compliance, liability, and regulatory considerations",
            "general": "Balanced approach considering multiple stakeholder perspectives"
        }

        if profile not in profile_descriptions:
            return None

        # Analyze response from profile perspective
        perspective_reasoning = f"From a {profile} perspective: {profile_descriptions[profile]}"

        # Generate trade-offs specific to this profile
        trade_offs = self._generate_profile_trade_offs(response, profile)

        return AlternativePerspective(
            perspective_name=f"{profile.title()} Perspective",
            description=profile_descriptions[profile],
            reasoning=perspective_reasoning,
            confidence=0.7,  # Default confidence for generated perspectives
            trade_offs=trade_offs,
            supporting_evidence=[]
        )

    def _generate_profile_trade_offs(self, response: str, profile: str) -> List[str]:
        """Generate profile-specific trade-offs."""
        trade_offs = []

        if profile == "researcher":
            trade_offs = [
                "High innovation potential vs. unproven methodology",
                "Novel approach vs. lack of peer validation",
                "Theoretical soundness vs. practical applicability"
            ]
        elif profile == "business":
            trade_offs = [
                "High ROI potential vs. implementation costs",
                "Market opportunity vs. competitive risks",
                "Short-term gains vs. long-term sustainability"
            ]
        elif profile == "legal":
            trade_offs = [
                "Compliance certainty vs. operational flexibility",
                "Risk mitigation vs. innovation constraints",
                "Regulatory approval vs. time-to-market"
            ]
        else:  # general
            trade_offs = [
                "Comprehensive solution vs. complexity",
                "Broad applicability vs. specific optimization",
                "Balanced approach vs. specialized focus"
            ]

        return trade_offs

    def _generate_contrarian_perspective(self, query: str, response: str,
                                       context: Optional[Dict[str, Any]]) -> Optional[AlternativePerspective]:
        """Generate a contrarian perspective that challenges the main response."""

        contrarian_reasoning = "Challenging the primary response by considering opposite viewpoints and potential flaws"

        # Generate contrarian trade-offs
        contrarian_trade_offs = [
            "Assumed benefits may not materialize in practice",
            "Hidden costs or risks not adequately considered",
            "Alternative approaches might be more effective",
            "Context-specific factors could invalidate the approach"
        ]

        return AlternativePerspective(
            perspective_name="Contrarian Analysis",
            description="Devil's advocate perspective challenging primary assumptions",
            reasoning=contrarian_reasoning,
            confidence=0.6,  # Lower confidence for contrarian views
            trade_offs=contrarian_trade_offs,
            supporting_evidence=[]
        )

    def _generate_adversarial_critiques(self, response: str, alternatives: List[AlternativePerspective],
                                      context: Optional[Dict[str, Any]]) -> List[AdversarialCritique]:
        """Generate adversarial critiques of the response."""
        critiques = []

        # Generate counter-arguments
        counter_arguments = self._generate_counter_arguments(response, alternatives)

        # Identify edge cases
        edge_cases = self._identify_edge_cases(response, context)

        # Detect potential biases
        bias_warnings = self._detect_potential_biases(response, context)

        # Generate alternative interpretations
        alternative_interpretations = self._generate_alternative_interpretations(response, alternatives)

        # Create adversarial critique
        critique_id = f"critique_{int(time.time() * 1000)}"

        # Determine severity based on critique level
        severity = {
            CritiqueLevel.GENTLE: 0.3,
            CritiqueLevel.MODERATE: 0.5,
            CritiqueLevel.RIGOROUS: 0.7,
            CritiqueLevel.ADVERSARIAL: 0.9
        }.get(self.critique_level, 0.5)

        # Determine if revision is required
        requires_revision = (
            len(counter_arguments) > 2 or
            len(bias_warnings) > 1 or
            severity > 0.6
        )

        critique = AdversarialCritique(
            critique_id=critique_id,
            critique_text=self._generate_critique_text(counter_arguments, edge_cases, bias_warnings),
            counter_arguments=counter_arguments,
            edge_cases=edge_cases,
            bias_warnings=bias_warnings,
            alternative_interpretations=alternative_interpretations,
            severity=severity,
            requires_revision=requires_revision
        )

        critiques.append(critique)
        return critiques

    def _generate_counter_arguments(self, response: str, alternatives: List[AlternativePerspective]) -> List[str]:
        """Generate counter-arguments to the response."""
        counter_arguments = []

        # Use alternative perspectives to generate counter-arguments
        for alternative in alternatives:
            if alternative.perspective_name == "Contrarian Analysis":
                counter_arguments.extend(alternative.trade_offs)

        # Add generic counter-arguments based on response analysis
        if "definitely" in response.lower() or "certainly" in response.lower():
            counter_arguments.append("Response shows overconfidence - alternatives not adequately considered")

        if len(response.split()) < 50:
            counter_arguments.append("Response may be too brief to address complexity of the question")

        if "assume" in response.lower():
            counter_arguments.append("Response relies on assumptions that may not hold in all contexts")

        return counter_arguments

    def _identify_edge_cases(self, response: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """Identify edge cases where the response might fail."""
        edge_cases = []

        # Generic edge cases
        edge_cases.extend([
            "Extreme scale scenarios (very large or very small)",
            "Resource-constrained environments",
            "Regulatory changes or legal restrictions",
            "Cultural or contextual differences",
            "Technology limitations or failures"
        ])

        # Context-specific edge cases
        if context and context.get("domain"):
            domain = context["domain"]
            if domain == "cybersecurity":
                edge_cases.append("Advanced persistent threats or zero-day exploits")
            elif domain == "business":
                edge_cases.append("Market disruption or economic downturns")
            elif domain == "legal":
                edge_cases.append("Jurisdictional conflicts or regulatory changes")

        return edge_cases[:3]  # Limit to top 3 edge cases

    def _detect_potential_biases(self, response: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """Detect potential biases in the response."""
        biases = []

        # Confirmation bias
        if "confirms" in response.lower() or "validates" in response.lower():
            biases.append("Confirmation bias - may favor information that confirms existing beliefs")

        # Availability bias
        if "recent" in response.lower() or "latest" in response.lower():
            biases.append("Availability bias - may overweight recent or easily recalled information")

        # Authority bias
        if "expert" in response.lower() or "authority" in response.lower():
            biases.append("Authority bias - may defer too heavily to expert opinions")

        # Optimism bias
        if "will succeed" in response.lower() or "guaranteed" in response.lower():
            biases.append("Optimism bias - may underestimate risks or overestimate benefits")

        return biases

    def _generate_alternative_interpretations(self, response: str,
                                            alternatives: List[AlternativePerspective]) -> List[str]:
        """Generate alternative interpretations of the query or response."""
        interpretations = []

        for alternative in alternatives:
            if alternative.perspective_name != "Contrarian Analysis":
                interpretation = f"From {alternative.perspective_name}: {alternative.reasoning}"
                interpretations.append(interpretation)

        return interpretations

    def _generate_critique_text(self, counter_arguments: List[str], edge_cases: List[str],
                              bias_warnings: List[str]) -> str:
        """Generate comprehensive critique text."""
        critique_parts = []

        if counter_arguments:
            critique_parts.append(f"Counter-arguments identified: {'; '.join(counter_arguments[:2])}")

        if edge_cases:
            critique_parts.append(f"Edge cases to consider: {'; '.join(edge_cases[:2])}")

        if bias_warnings:
            critique_parts.append(f"Potential biases detected: {'; '.join(bias_warnings[:2])}")

        if not critique_parts:
            critique_parts.append("Response appears reasonable but should be validated against alternative perspectives")

        return " | ".join(critique_parts)

    def _detect_dimension_conflicts(self, response_analysis: Dict[str, Any],
                                  context: Optional[Dict[str, Any]]) -> List[DimensionConflict]:
        """Detect conflicts between conceptual dimensions."""
        conflicts = []

        dimension_scores = response_analysis.get("dimension_scores", {})
        if not dimension_scores:
            return conflicts

        # High utility vs High risk/danger
        utility = dimension_scores.get('utility', 0)
        danger = dimension_scores.get('danger', 0)
        if utility > 0.7 and danger > 0.7:
            conflicts.append(DimensionConflict(
                conflict_type="utility_risk_tension",
                description="High utility but also high risk - requires careful risk-benefit analysis",
                dimensions=["utility", "danger"],
                severity=0.8,
                recommendation="Implement risk mitigation strategies before proceeding",
                evidence=[f"Utility score: {utility:.2f}", f"Danger score: {danger:.2f}"]
            ))

        # Innovation vs Feasibility
        novelty = dimension_scores.get('novelty', 0)
        feasibility = dimension_scores.get('feasibility', 0)
        if novelty > 0.8 and feasibility < 0.4:
            conflicts.append(DimensionConflict(
                conflict_type="innovation_feasibility_gap",
                description="Highly innovative but potentially difficult to implement",
                dimensions=["novelty", "feasibility"],
                severity=0.7,
                recommendation="Consider phased implementation or proof-of-concept approach",
                evidence=[f"Novelty score: {novelty:.2f}", f"Feasibility score: {feasibility:.2f}"]
            ))

        # Complexity vs Clarity
        complexity = dimension_scores.get('complexity', 0)
        clarity = dimension_scores.get('clarity', 0)
        if complexity > 0.8 and clarity < 0.4:
            conflicts.append(DimensionConflict(
                conflict_type="complexity_clarity_tension",
                description="High complexity may compromise understanding and adoption",
                dimensions=["complexity", "clarity"],
                severity=0.6,
                recommendation="Simplify presentation or provide additional explanation",
                evidence=[f"Complexity score: {complexity:.2f}", f"Clarity score: {clarity:.2f}"]
            ))

        # Market Impact vs Compliance Risk
        market_impact = dimension_scores.get('market_impact', 0)
        compliance_risk = dimension_scores.get('compliance_risk', 0)
        if market_impact > 0.7 and compliance_risk > 0.6:
            conflicts.append(DimensionConflict(
                conflict_type="market_compliance_tension",
                description="High market potential but significant compliance concerns",
                dimensions=["market_impact", "compliance_risk"],
                severity=0.8,
                recommendation="Conduct thorough regulatory review before market entry",
                evidence=[f"Market impact: {market_impact:.2f}", f"Compliance risk: {compliance_risk:.2f}"]
            ))

        return conflicts

    def _justify_confidence(self, response_analysis: Dict[str, Any],
                          critiques: List[AdversarialCritique],
                          context: Optional[Dict[str, Any]]) -> ConfidenceJustification:
        """Generate detailed confidence justification."""

        # Base confidence from response analysis
        base_confidence = 0.7  # Default

        # Factors affecting confidence
        factors = []
        confidence_breakdown = {}

        # Evidence quality assessment
        evidence_sources = response_analysis.get("evidence_sources", [])
        evidence_quality = min(1.0, len(evidence_sources) * 0.2)
        confidence_breakdown["evidence_quality"] = evidence_quality
        factors.append(f"Evidence quality: {evidence_quality:.2f} based on {len(evidence_sources)} sources")

        # Dimension consistency
        dimension_scores = response_analysis.get("dimension_scores", {})
        if dimension_scores:
            dimension_variance = self._calculate_dimension_variance(dimension_scores)
            dimension_consistency = max(0.0, 1.0 - dimension_variance)
            confidence_breakdown["dimension_consistency"] = dimension_consistency
            factors.append(f"Dimension consistency: {dimension_consistency:.2f}")
        else:
            dimension_consistency = 0.5
            confidence_breakdown["dimension_consistency"] = dimension_consistency
            factors.append("Dimension consistency: 0.50 (no dimension analysis available)")

        # Critique impact
        critique_penalty = sum(critique.severity for critique in critiques) * 0.1
        critique_impact = max(0.0, 1.0 - critique_penalty)
        confidence_breakdown["critique_impact"] = critique_impact
        factors.append(f"Critique impact: {critique_impact:.2f} (penalty: {critique_penalty:.2f})")

        # Reasoning completeness
        assumptions = response_analysis.get("assumptions", [])
        uncertainty_indicators = response_analysis.get("uncertainty_indicators", [])
        reasoning_completeness = max(0.3, 1.0 - len(assumptions) * 0.1 - len(uncertainty_indicators) * 0.05)
        confidence_breakdown["reasoning_completeness"] = reasoning_completeness
        factors.append(f"Reasoning completeness: {reasoning_completeness:.2f}")

        # Calculate overall confidence
        overall_confidence = (
            evidence_quality * 0.3 +
            dimension_consistency * 0.25 +
            critique_impact * 0.25 +
            reasoning_completeness * 0.2
        )

        # Reliability assessment
        if overall_confidence > 0.8:
            reliability = "High reliability - strong evidence and consistent reasoning"
        elif overall_confidence > 0.6:
            reliability = "Moderate reliability - adequate evidence with some limitations"
        elif overall_confidence > 0.4:
            reliability = "Low reliability - limited evidence or significant concerns"
        else:
            reliability = "Very low reliability - insufficient evidence or major flaws"

        return ConfidenceJustification(
            confidence_score=overall_confidence,
            contributing_factors=factors,
            confidence_breakdown=confidence_breakdown,
            reliability_assessment=reliability,
            evidence_quality=evidence_quality,
            reasoning_completeness=reasoning_completeness
        )

    def _calculate_dimension_variance(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate variance in dimension scores to assess consistency."""
        if not dimension_scores:
            return 1.0

        scores = list(dimension_scores.values())
        if len(scores) < 2:
            return 0.0

        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        return min(1.0, variance)  # Normalize to 0-1 range

    def _analyze_trade_offs(self, alternatives: List[AlternativePerspective],
                          conflicts: List[DimensionConflict],
                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade-offs between different approaches and considerations."""

        trade_off_analysis = {
            "perspective_trade_offs": {},
            "dimension_trade_offs": [],
            "decision_factors": [],
            "recommendations": []
        }

        # Analyze perspective-specific trade-offs
        for alternative in alternatives:
            trade_off_analysis["perspective_trade_offs"][alternative.perspective_name] = {
                "benefits": alternative.trade_offs[:2] if len(alternative.trade_offs) > 2 else alternative.trade_offs,
                "limitations": alternative.trade_offs[2:] if len(alternative.trade_offs) > 2 else [],
                "confidence": alternative.confidence
            }

        # Analyze dimension conflicts as trade-offs
        for conflict in conflicts:
            trade_off_analysis["dimension_trade_offs"].append({
                "conflict_type": conflict.conflict_type,
                "description": conflict.description,
                "severity": conflict.severity,
                "recommendation": conflict.recommendation
            })

        # Generate decision factors
        trade_off_analysis["decision_factors"] = [
            "Consider stakeholder priorities and constraints",
            "Evaluate risk tolerance and mitigation capabilities",
            "Assess resource availability and timeline constraints",
            "Review regulatory and compliance requirements"
        ]

        # Generate recommendations
        if conflicts:
            high_severity_conflicts = [c for c in conflicts if c.severity > 0.7]
            if high_severity_conflicts:
                trade_off_analysis["recommendations"].append(
                    "Address high-severity dimension conflicts before proceeding"
                )

        if len(alternatives) > 2:
            trade_off_analysis["recommendations"].append(
                "Consider hybrid approach combining strengths of multiple perspectives"
            )

        return trade_off_analysis

    def _synthesize_final_response(self, query: str, initial_response: str,
                                 alternatives: List[AlternativePerspective],
                                 critiques: List[AdversarialCritique],
                                 conflicts: List[DimensionConflict],
                                 trade_offs: Dict[str, Any]) -> str:
        """Synthesize final response incorporating all reflective analysis."""

        # Start with initial response
        final_parts = [initial_response]

        # Add meta-cognitive reflection
        final_parts.append("\n\n**🧠 Meta-Cognitive Reflection:**")

        # Add critique summary if significant issues found
        significant_critiques = [c for c in critiques if c.severity > 0.6]
        if significant_critiques:
            final_parts.append(f"⚠️ **Critical Analysis:** {significant_critiques[0].critique_text}")

        # Add dimension conflicts if any
        if conflicts:
            final_parts.append(f"⚖️ **Trade-off Alert:** {conflicts[0].description}")
            final_parts.append(f"💡 **Recommendation:** {conflicts[0].recommendation}")

        # Add alternative perspectives summary
        if alternatives:
            final_parts.append("🔄 **Alternative Perspectives:**")
            for alt in alternatives[:2]:  # Limit to top 2 alternatives
                final_parts.append(f"• **{alt.perspective_name}:** {alt.reasoning}")

        # Add confidence note
        final_parts.append(f"\n📊 **Confidence Level:** Moderate (reflective analysis applied)")

        return "\n".join(final_parts)

    def _build_reasoning_chain(self, response_analysis: Dict[str, Any],
                             alternatives: List[AlternativePerspective],
                             critiques: List[AdversarialCritique],
                             conflicts: List[DimensionConflict],
                             confidence_justification: ConfidenceJustification,
                             trade_offs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build transparent reasoning chain showing meta-cognitive process."""

        chain = []

        # Step 1: Initial Analysis
        chain.append({
            "step": 1,
            "type": "initial_analysis",
            "description": "Analyzed initial response for assumptions and confidence indicators",
            "details": {
                "assumptions_found": len(response_analysis.get("assumptions", [])),
                "uncertainty_indicators": len(response_analysis.get("uncertainty_indicators", [])),
                "confidence_indicators": len(response_analysis.get("confidence_indicators", []))
            }
        })

        # Step 2: Alternative Perspectives
        chain.append({
            "step": 2,
            "type": "alternative_perspectives",
            "description": f"Generated {len(alternatives)} alternative perspectives",
            "details": {
                "perspectives": [alt.perspective_name for alt in alternatives]
            }
        })

        # Step 3: Adversarial Critique
        chain.append({
            "step": 3,
            "type": "adversarial_critique",
            "description": f"Applied {self.critique_level.value} level critique",
            "details": {
                "critiques_generated": len(critiques),
                "max_severity": max([c.severity for c in critiques]) if critiques else 0.0
            }
        })

        # Step 4: Dimension Conflicts
        chain.append({
            "step": 4,
            "type": "dimension_conflicts",
            "description": f"Detected {len(conflicts)} dimension conflicts",
            "details": {
                "conflicts": [c.conflict_type for c in conflicts]
            }
        })

        # Step 5: Confidence Justification
        chain.append({
            "step": 5,
            "type": "confidence_justification",
            "description": "Calculated evidence-based confidence score",
            "details": {
                "final_confidence": confidence_justification.confidence_score,
                "reliability": confidence_justification.reliability_assessment
            }
        })

        return chain

    def _generate_critique_summary(self, critiques: List[AdversarialCritique],
                                 conflicts: List[DimensionConflict],
                                 trade_offs: Dict[str, Any]) -> str:
        """Generate summary of all critiques and analyses."""

        summary_parts = []

        if critiques:
            max_severity = max(c.severity for c in critiques)
            summary_parts.append(f"Adversarial critique applied with {max_severity:.1f} severity")

            total_counter_args = sum(len(c.counter_arguments) for c in critiques)
            if total_counter_args > 0:
                summary_parts.append(f"{total_counter_args} counter-arguments identified")

        if conflicts:
            high_severity = len([c for c in conflicts if c.severity > 0.7])
            summary_parts.append(f"{len(conflicts)} dimension conflicts detected ({high_severity} high-severity)")

        perspective_count = len(trade_offs.get("perspective_trade_offs", {}))
        if perspective_count > 0:
            summary_parts.append(f"{perspective_count} alternative perspectives analyzed")

        if not summary_parts:
            return "No significant issues identified in reflective analysis"

        return "; ".join(summary_parts)

    def _calculate_meta_confidence(self, confidence_justification: ConfidenceJustification,
                                 critiques: List[AdversarialCritique],
                                 conflicts: List[DimensionConflict]) -> float:
        """Calculate meta-confidence in the reflective reasoning process itself."""

        base_confidence = confidence_justification.confidence_score

        # Adjust based on critique quality
        critique_quality = 0.8 if critiques else 0.5  # Higher if critiques were generated

        # Adjust based on conflict detection
        conflict_detection = 0.9 if conflicts else 0.7  # Higher if conflicts were detected

        # Adjust based on reasoning completeness
        reasoning_completeness = confidence_justification.reasoning_completeness

        # Calculate meta-confidence
        meta_confidence = (
            base_confidence * 0.4 +
            critique_quality * 0.3 +
            conflict_detection * 0.2 +
            reasoning_completeness * 0.1
        )

        return min(1.0, max(0.0, meta_confidence))

    def _create_error_result(self, session_id: str, query: str, initial_response: str, error: str) -> ReflectiveResult:
        """Create minimal result when errors occur."""
        return ReflectiveResult(
            session_id=session_id,
            original_query=query,
            initial_response=initial_response,
            response_analysis={"error": error},
            alternative_perspectives=[],
            adversarial_critiques=[],
            dimension_conflicts=[],
            confidence_justification=ConfidenceJustification(
                confidence_score=0.3,
                contributing_factors=[f"Error in reflective reasoning: {error}"],
                confidence_breakdown={"error_penalty": 0.7},
                reliability_assessment="Low reliability due to processing error",
                evidence_quality=0.0,
                reasoning_completeness=0.0
            ),
            trade_off_analysis={},
            final_response=f"{initial_response}\n\n⚠️ **Note:** Reflective analysis encountered an error: {error}",
            reasoning_chain=[],
            critique_summary=f"Error in reflective reasoning: {error}",
            meta_confidence=0.2,
            reflection_duration_ms=0,
            timestamp=datetime.now().isoformat()
        )
