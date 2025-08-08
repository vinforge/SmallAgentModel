#!/usr/bin/env python3
"""
Advanced Dimension Conflict Detector for SAM Phase 5
Detects and explains tensions between conceptual dimensions with sophisticated analysis.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ConflictSeverity(Enum):
    """Severity levels for dimension conflicts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConflictType(Enum):
    """Types of dimension conflicts."""
    UTILITY_RISK = "utility_risk"
    INNOVATION_FEASIBILITY = "innovation_feasibility"
    COMPLEXITY_CLARITY = "complexity_clarity"
    MARKET_COMPLIANCE = "market_compliance"
    SPEED_QUALITY = "speed_quality"
    COST_BENEFIT = "cost_benefit"
    SECURITY_USABILITY = "security_usability"
    SCALABILITY_SIMPLICITY = "scalability_simplicity"

@dataclass
class DimensionConflictRule:
    """Rule for detecting dimension conflicts."""
    conflict_type: ConflictType
    primary_dimension: str
    secondary_dimension: str
    primary_threshold: float
    secondary_threshold: float
    severity_calculator: str  # Formula for calculating severity
    description_template: str
    recommendation_template: str
    evidence_template: str

@dataclass
class ConflictEvidence:
    """Evidence supporting a dimension conflict."""
    dimension_name: str
    score: float
    threshold: float
    deviation: float
    supporting_text: Optional[str] = None

@dataclass
class DimensionConflict:
    """Comprehensive dimension conflict with detailed analysis."""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    severity_score: float  # 0.0 to 1.0
    
    # Core conflict information
    description: str
    recommendation: str
    dimensions_involved: List[str]
    
    # Evidence and analysis
    evidence: List[ConflictEvidence]
    risk_assessment: str
    mitigation_strategies: List[str]
    
    # Context and metadata
    context_factors: List[str]
    confidence: float
    timestamp: str

class AdvancedDimensionConflictDetector:
    """
    Advanced detector for conflicts between conceptual dimensions with
    sophisticated analysis and evidence-based recommendations.
    """
    
    def __init__(self):
        """Initialize the dimension conflict detector."""
        self.conflict_rules = self._initialize_conflict_rules()
        self.context_analyzers = self._initialize_context_analyzers()
        
        logger.info("Advanced Dimension Conflict Detector initialized")
    
    def _initialize_conflict_rules(self) -> List[DimensionConflictRule]:
        """Initialize rules for detecting dimension conflicts."""
        rules = [
            # Utility vs Risk/Danger
            DimensionConflictRule(
                conflict_type=ConflictType.UTILITY_RISK,
                primary_dimension="utility",
                secondary_dimension="danger",
                primary_threshold=0.7,
                secondary_threshold=0.6,
                severity_calculator="min(primary_score, secondary_score)",
                description_template="High utility ({primary_score:.2f}) but significant risk ({secondary_score:.2f})",
                recommendation_template="Implement comprehensive risk mitigation before proceeding",
                evidence_template="Utility score {primary_score:.2f} exceeds threshold {primary_threshold:.2f}, Risk score {secondary_score:.2f} exceeds threshold {secondary_threshold:.2f}"
            ),
            
            # Innovation vs Feasibility
            DimensionConflictRule(
                conflict_type=ConflictType.INNOVATION_FEASIBILITY,
                primary_dimension="novelty",
                secondary_dimension="feasibility",
                primary_threshold=0.8,
                secondary_threshold=0.4,
                severity_calculator="primary_score - secondary_score",
                description_template="Highly innovative ({primary_score:.2f}) but low feasibility ({secondary_score:.2f})",
                recommendation_template="Consider phased implementation or proof-of-concept approach",
                evidence_template="Innovation score {primary_score:.2f} vs feasibility score {secondary_score:.2f} creates implementation gap"
            ),
            
            # Complexity vs Clarity
            DimensionConflictRule(
                conflict_type=ConflictType.COMPLEXITY_CLARITY,
                primary_dimension="complexity",
                secondary_dimension="clarity",
                primary_threshold=0.8,
                secondary_threshold=0.4,
                severity_calculator="primary_score - secondary_score",
                description_template="High complexity ({primary_score:.2f}) compromises clarity ({secondary_score:.2f})",
                recommendation_template="Simplify presentation or provide additional explanation and training",
                evidence_template="Complexity {primary_score:.2f} may hinder understanding with clarity at {secondary_score:.2f}"
            ),
            
            # Market Impact vs Compliance Risk
            DimensionConflictRule(
                conflict_type=ConflictType.MARKET_COMPLIANCE,
                primary_dimension="market_impact",
                secondary_dimension="compliance_risk",
                primary_threshold=0.7,
                secondary_threshold=0.6,
                severity_calculator="(primary_score + secondary_score) / 2",
                description_template="Strong market potential ({primary_score:.2f}) with compliance concerns ({secondary_score:.2f})",
                recommendation_template="Conduct thorough regulatory review and compliance assessment",
                evidence_template="Market opportunity {primary_score:.2f} balanced against compliance risk {secondary_score:.2f}"
            ),
            
            # Speed vs Quality
            DimensionConflictRule(
                conflict_type=ConflictType.SPEED_QUALITY,
                primary_dimension="urgency",
                secondary_dimension="credibility",
                primary_threshold=0.8,
                secondary_threshold=0.4,
                severity_calculator="primary_score - secondary_score",
                description_template="High urgency ({primary_score:.2f}) may compromise quality ({secondary_score:.2f})",
                recommendation_template="Balance speed requirements with quality assurance processes",
                evidence_template="Urgency {primary_score:.2f} creates pressure that may affect quality {secondary_score:.2f}"
            ),
            
            # Security vs Usability
            DimensionConflictRule(
                conflict_type=ConflictType.SECURITY_USABILITY,
                primary_dimension="security_level",
                secondary_dimension="usability",
                primary_threshold=0.8,
                secondary_threshold=0.4,
                severity_calculator="primary_score - secondary_score",
                description_template="High security requirements ({primary_score:.2f}) reduce usability ({secondary_score:.2f})",
                recommendation_template="Design user-friendly security measures and provide training",
                evidence_template="Security level {primary_score:.2f} creates usability challenges at {secondary_score:.2f}"
            )
        ]
        
        return rules
    
    def _initialize_context_analyzers(self) -> Dict[str, Any]:
        """Initialize context analyzers for domain-specific conflict detection."""
        return {
            "cybersecurity": {
                "high_priority_conflicts": [ConflictType.SECURITY_USABILITY, ConflictType.UTILITY_RISK],
                "context_factors": ["threat_landscape", "compliance_requirements", "user_training"]
            },
            "business": {
                "high_priority_conflicts": [ConflictType.MARKET_COMPLIANCE, ConflictType.COST_BENEFIT],
                "context_factors": ["market_conditions", "competitive_landscape", "resource_constraints"]
            },
            "research": {
                "high_priority_conflicts": [ConflictType.INNOVATION_FEASIBILITY, ConflictType.COMPLEXITY_CLARITY],
                "context_factors": ["research_timeline", "funding_constraints", "peer_review_requirements"]
            },
            "legal": {
                "high_priority_conflicts": [ConflictType.MARKET_COMPLIANCE, ConflictType.SPEED_QUALITY],
                "context_factors": ["regulatory_changes", "jurisdictional_differences", "precedent_analysis"]
            }
        }
    
    def detect_conflicts(self, dimension_scores: Dict[str, float], 
                        context: Optional[Dict[str, Any]] = None) -> List[DimensionConflict]:
        """
        Detect dimension conflicts with comprehensive analysis.
        
        Args:
            dimension_scores: Dictionary of dimension names to scores (0.0 to 1.0)
            context: Optional context including domain, profile, etc.
            
        Returns:
            List of detected dimension conflicts with detailed analysis
        """
        try:
            conflicts = []
            
            # Apply conflict detection rules
            for rule in self.conflict_rules:
                conflict = self._apply_conflict_rule(rule, dimension_scores, context)
                if conflict:
                    conflicts.append(conflict)
            
            # Sort by severity score (highest first)
            conflicts.sort(key=lambda x: x.severity_score, reverse=True)
            
            # Add context-specific analysis
            if context:
                conflicts = self._enhance_with_context(conflicts, context)
            
            logger.info(f"Detected {len(conflicts)} dimension conflicts")
            return conflicts
            
        except Exception as e:
            logger.error(f"Error detecting dimension conflicts: {e}")
            return []
    
    def _apply_conflict_rule(self, rule: DimensionConflictRule, 
                           dimension_scores: Dict[str, float],
                           context: Optional[Dict[str, Any]]) -> Optional[DimensionConflict]:
        """Apply a single conflict detection rule."""
        
        # Check if required dimensions are present
        if rule.primary_dimension not in dimension_scores or rule.secondary_dimension not in dimension_scores:
            return None
        
        primary_score = dimension_scores[rule.primary_dimension]
        secondary_score = dimension_scores[rule.secondary_dimension]
        
        # Check if conflict conditions are met
        conflict_detected = False
        
        if rule.conflict_type in [ConflictType.UTILITY_RISK, ConflictType.MARKET_COMPLIANCE]:
            # Both dimensions high
            conflict_detected = (primary_score >= rule.primary_threshold and 
                               secondary_score >= rule.secondary_threshold)
        else:
            # Primary high, secondary low
            conflict_detected = (primary_score >= rule.primary_threshold and 
                               secondary_score <= rule.secondary_threshold)
        
        if not conflict_detected:
            return None
        
        # Calculate severity
        severity_score = self._calculate_severity(rule, primary_score, secondary_score)
        severity = self._determine_severity_level(severity_score)
        
        # Generate evidence
        evidence = self._generate_evidence(rule, primary_score, secondary_score)
        
        # Generate description and recommendation
        description = rule.description_template.format(
            primary_score=primary_score,
            secondary_score=secondary_score
        )
        
        recommendation = rule.recommendation_template
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(rule.conflict_type, context)
        
        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(rule.conflict_type, severity_score)
        
        # Generate context factors
        context_factors = self._generate_context_factors(rule.conflict_type, context)
        
        conflict_id = f"{rule.conflict_type.value}_{int(time.time() * 1000)}"
        
        return DimensionConflict(
            conflict_id=conflict_id,
            conflict_type=rule.conflict_type,
            severity=severity,
            severity_score=severity_score,
            description=description,
            recommendation=recommendation,
            dimensions_involved=[rule.primary_dimension, rule.secondary_dimension],
            evidence=evidence,
            risk_assessment=risk_assessment,
            mitigation_strategies=mitigation_strategies,
            context_factors=context_factors,
            confidence=0.8,  # Default confidence
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_severity(self, rule: DimensionConflictRule, 
                          primary_score: float, secondary_score: float) -> float:
        """Calculate severity score based on rule formula."""
        try:
            # Simple formula evaluation
            if rule.severity_calculator == "min(primary_score, secondary_score)":
                return min(primary_score, secondary_score)
            elif rule.severity_calculator == "primary_score - secondary_score":
                return abs(primary_score - secondary_score)
            elif rule.severity_calculator == "(primary_score + secondary_score) / 2":
                return (primary_score + secondary_score) / 2
            else:
                # Default calculation
                return (primary_score + secondary_score) / 2
        except Exception:
            return 0.5  # Default severity
    
    def _determine_severity_level(self, severity_score: float) -> ConflictSeverity:
        """Determine severity level from score."""
        if severity_score >= 0.8:
            return ConflictSeverity.CRITICAL
        elif severity_score >= 0.6:
            return ConflictSeverity.HIGH
        elif severity_score >= 0.4:
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW

    def _generate_evidence(self, rule: DimensionConflictRule,
                         primary_score: float, secondary_score: float) -> List[ConflictEvidence]:
        """Generate evidence for the conflict."""
        evidence = []

        # Primary dimension evidence
        primary_deviation = primary_score - rule.primary_threshold
        evidence.append(ConflictEvidence(
            dimension_name=rule.primary_dimension,
            score=primary_score,
            threshold=rule.primary_threshold,
            deviation=primary_deviation,
            supporting_text=f"{rule.primary_dimension} score {primary_score:.2f} exceeds threshold {rule.primary_threshold:.2f}"
        ))

        # Secondary dimension evidence
        secondary_deviation = abs(secondary_score - rule.secondary_threshold)
        evidence.append(ConflictEvidence(
            dimension_name=rule.secondary_dimension,
            score=secondary_score,
            threshold=rule.secondary_threshold,
            deviation=secondary_deviation,
            supporting_text=f"{rule.secondary_dimension} score {secondary_score:.2f} creates tension with {rule.primary_dimension}"
        ))

        return evidence

    def _generate_mitigation_strategies(self, conflict_type: ConflictType,
                                      context: Optional[Dict[str, Any]]) -> List[str]:
        """Generate mitigation strategies for the conflict type."""
        strategies = {
            ConflictType.UTILITY_RISK: [
                "Implement comprehensive risk assessment and mitigation plan",
                "Establish monitoring and early warning systems",
                "Develop contingency plans for high-risk scenarios",
                "Consider phased rollout to limit exposure"
            ],
            ConflictType.INNOVATION_FEASIBILITY: [
                "Conduct proof-of-concept studies",
                "Break down implementation into manageable phases",
                "Identify and address technical barriers early",
                "Seek expert consultation on implementation challenges"
            ],
            ConflictType.COMPLEXITY_CLARITY: [
                "Develop comprehensive documentation and training materials",
                "Create simplified interfaces or abstractions",
                "Implement progressive disclosure of complexity",
                "Establish user support and guidance systems"
            ],
            ConflictType.MARKET_COMPLIANCE: [
                "Engage regulatory experts early in the process",
                "Conduct thorough compliance review and gap analysis",
                "Develop compliance-first implementation strategy",
                "Establish ongoing regulatory monitoring"
            ],
            ConflictType.SPEED_QUALITY: [
                "Prioritize critical quality requirements",
                "Implement automated quality assurance processes",
                "Use iterative development with quality gates",
                "Balance speed with acceptable quality thresholds"
            ],
            ConflictType.SECURITY_USABILITY: [
                "Design security measures with user experience in mind",
                "Implement progressive security based on risk levels",
                "Provide comprehensive user training and support",
                "Use security by design principles"
            ]
        }

        return strategies.get(conflict_type, ["Develop context-specific mitigation strategies"])

    def _generate_risk_assessment(self, conflict_type: ConflictType, severity_score: float) -> str:
        """Generate risk assessment for the conflict."""
        risk_levels = {
            ConflictType.UTILITY_RISK: "High operational and safety risks",
            ConflictType.INNOVATION_FEASIBILITY: "Implementation and adoption risks",
            ConflictType.COMPLEXITY_CLARITY: "User adoption and training risks",
            ConflictType.MARKET_COMPLIANCE: "Regulatory and legal risks",
            ConflictType.SPEED_QUALITY: "Quality and reputation risks",
            ConflictType.SECURITY_USABILITY: "Security breach and user resistance risks"
        }

        base_risk = risk_levels.get(conflict_type, "Moderate operational risks")

        if severity_score >= 0.8:
            return f"CRITICAL: {base_risk} with high probability of negative impact"
        elif severity_score >= 0.6:
            return f"HIGH: {base_risk} requiring immediate attention"
        elif severity_score >= 0.4:
            return f"MODERATE: {base_risk} that should be monitored"
        else:
            return f"LOW: {base_risk} with manageable impact"

    def _generate_context_factors(self, conflict_type: ConflictType,
                                context: Optional[Dict[str, Any]]) -> List[str]:
        """Generate context factors relevant to the conflict."""
        factors = []

        # Add general context factors
        general_factors = {
            ConflictType.UTILITY_RISK: ["Risk tolerance", "Safety requirements", "Regulatory environment"],
            ConflictType.INNOVATION_FEASIBILITY: ["Technical capabilities", "Resource availability", "Timeline constraints"],
            ConflictType.COMPLEXITY_CLARITY: ["User expertise", "Training resources", "Support infrastructure"],
            ConflictType.MARKET_COMPLIANCE: ["Regulatory landscape", "Market maturity", "Competitive pressure"],
            ConflictType.SPEED_QUALITY: ["Market timing", "Quality standards", "Resource constraints"],
            ConflictType.SECURITY_USABILITY: ["Threat landscape", "User behavior", "Security culture"]
        }

        factors.extend(general_factors.get(conflict_type, []))

        # Add context-specific factors
        if context:
            domain = context.get("domain", "")
            if domain in self.context_analyzers:
                domain_factors = self.context_analyzers[domain]["context_factors"]
                factors.extend(domain_factors)

        return factors

    def _enhance_with_context(self, conflicts: List[DimensionConflict],
                            context: Dict[str, Any]) -> List[DimensionConflict]:
        """Enhance conflicts with context-specific analysis."""
        domain = context.get("domain", "")
        profile = context.get("profile", "")

        if domain in self.context_analyzers:
            domain_config = self.context_analyzers[domain]
            high_priority_conflicts = domain_config["high_priority_conflicts"]

            # Boost severity for domain-specific high-priority conflicts
            for conflict in conflicts:
                if conflict.conflict_type in high_priority_conflicts:
                    conflict.severity_score = min(1.0, conflict.severity_score * 1.2)
                    conflict.severity = self._determine_severity_level(conflict.severity_score)
                    conflict.context_factors.append(f"High priority for {domain} domain")

        return conflicts

    def analyze_conflict_trends(self, conflicts_history: List[List[DimensionConflict]]) -> Dict[str, Any]:
        """Analyze trends in dimension conflicts over time."""
        if not conflicts_history:
            return {}

        trend_analysis = {
            "most_common_conflicts": {},
            "severity_trends": {},
            "resolution_patterns": {},
            "recommendations": []
        }

        # Count conflict types
        conflict_counts = {}
        for conflicts in conflicts_history:
            for conflict in conflicts:
                conflict_type = conflict.conflict_type.value
                conflict_counts[conflict_type] = conflict_counts.get(conflict_type, 0) + 1

        # Sort by frequency
        trend_analysis["most_common_conflicts"] = dict(
            sorted(conflict_counts.items(), key=lambda x: x[1], reverse=True)
        )

        # Generate recommendations based on trends
        if conflict_counts:
            most_common = max(conflict_counts.items(), key=lambda x: x[1])
            trend_analysis["recommendations"].append(
                f"Focus on addressing {most_common[0]} conflicts (occurred {most_common[1]} times)"
            )

        return trend_analysis


# Convenience functions for easy integration
def detect_dimension_conflicts(dimension_scores: Dict[str, float],
                             context: Optional[Dict[str, Any]] = None) -> List[DimensionConflict]:
    """Convenience function for detecting dimension conflicts."""
    detector = AdvancedDimensionConflictDetector()
    return detector.detect_conflicts(dimension_scores, context)


def analyze_conflict_severity(conflicts: List[DimensionConflict]) -> Dict[str, Any]:
    """Analyze the overall severity of a set of conflicts."""
    if not conflicts:
        return {"overall_severity": "none", "critical_count": 0, "recommendations": []}

    severity_counts = {
        "critical": len([c for c in conflicts if c.severity == ConflictSeverity.CRITICAL]),
        "high": len([c for c in conflicts if c.severity == ConflictSeverity.HIGH]),
        "medium": len([c for c in conflicts if c.severity == ConflictSeverity.MEDIUM]),
        "low": len([c for c in conflicts if c.severity == ConflictSeverity.LOW])
    }

    # Determine overall severity
    if severity_counts["critical"] > 0:
        overall_severity = "critical"
    elif severity_counts["high"] > 0:
        overall_severity = "high"
    elif severity_counts["medium"] > 0:
        overall_severity = "medium"
    else:
        overall_severity = "low"

    # Generate recommendations
    recommendations = []
    if severity_counts["critical"] > 0:
        recommendations.append("Immediate action required for critical conflicts")
    if severity_counts["high"] > 0:
        recommendations.append("Prioritize resolution of high-severity conflicts")
    if len(conflicts) > 3:
        recommendations.append("Consider simplifying approach to reduce conflict complexity")

    return {
        "overall_severity": overall_severity,
        "severity_breakdown": severity_counts,
        "total_conflicts": len(conflicts),
        "recommendations": recommendations
    }
