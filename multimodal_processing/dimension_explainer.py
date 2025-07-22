#!/usr/bin/env python3
"""
Dimension Explainer for SAM's Conceptual Understanding
Provides natural language explanations for dimension scores and reasoning transparency.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class ExplanationResult:
    """Result of dimension explanation generation."""
    summary: str
    detailed_explanations: Dict[str, str]
    reasoning_chain: List[str]
    confidence_assessment: str
    profile_context: str

class DimensionExplainer:
    """
    Natural language explanation generator for conceptual dimensions.
    
    Provides transparent, human-readable explanations for why specific
    dimension scores were assigned to content chunks.
    """
    
    def __init__(self):
        """Initialize dimension explainer."""
        self.explanation_patterns = self._initialize_explanation_patterns()
        self.confidence_levels = {
            (0.8, 1.0): "very high",
            (0.6, 0.8): "high", 
            (0.4, 0.6): "moderate",
            (0.2, 0.4): "low",
            (0.0, 0.2): "very low"
        }
    
    def _initialize_explanation_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize explanation patterns for different dimensions."""
        return {
            # General profile explanations
            "utility": {
                "high": [
                    "demonstrates clear practical applications",
                    "provides actionable insights and solutions",
                    "offers real-world implementation guidance",
                    "presents valuable tools and methodologies"
                ],
                "medium": [
                    "shows some practical value",
                    "contains useful information with limited application",
                    "provides theoretical value with potential practical use"
                ],
                "low": [
                    "primarily theoretical with limited practical application",
                    "lacks clear real-world implementation guidance"
                ]
            },
            "complexity": {
                "high": [
                    "contains sophisticated technical concepts",
                    "involves advanced algorithms and methodologies",
                    "requires specialized knowledge to understand",
                    "presents intricate theoretical frameworks"
                ],
                "medium": [
                    "involves moderate technical depth",
                    "requires some specialized knowledge",
                    "contains technical concepts with clear explanations"
                ],
                "low": [
                    "presents straightforward concepts",
                    "uses accessible language and simple explanations"
                ]
            },
            "clarity": {
                "high": [
                    "presents information in a clear, well-structured manner",
                    "uses precise language and explicit definitions",
                    "provides logical flow and easy-to-follow explanations"
                ],
                "medium": [
                    "generally clear with some areas needing clarification",
                    "mostly well-structured with minor ambiguities"
                ],
                "low": [
                    "contains ambiguous or unclear explanations",
                    "lacks clear structure or logical flow"
                ]
            },
            "relevance": {
                "high": [
                    "directly addresses key topics and important concepts",
                    "provides essential information for understanding the subject",
                    "contains critical insights and significant findings"
                ],
                "medium": [
                    "relates to the topic with moderate importance",
                    "provides supporting information and context"
                ],
                "low": [
                    "tangentially related to the main topic",
                    "provides background information with limited relevance"
                ]
            },
            "credibility": {
                "high": [
                    "cites authoritative sources and peer-reviewed research",
                    "presents evidence-based conclusions",
                    "demonstrates rigorous methodology and validation"
                ],
                "medium": [
                    "provides some supporting evidence",
                    "cites credible sources with moderate validation"
                ],
                "low": [
                    "lacks supporting evidence or citations",
                    "presents unvalidated claims or opinions"
                ]
            },
            # Research profile explanations
            "novelty": {
                "high": [
                    "presents groundbreaking ideas and innovative approaches",
                    "introduces novel concepts not previously explored",
                    "demonstrates original thinking and creative solutions"
                ],
                "medium": [
                    "builds on existing ideas with some innovative elements",
                    "presents incremental advances in the field"
                ],
                "low": [
                    "largely based on established knowledge",
                    "lacks significant innovative contributions"
                ]
            },
            "technical_depth": {
                "high": [
                    "demonstrates deep technical expertise and sophisticated analysis",
                    "employs advanced methodologies and rigorous approaches",
                    "provides comprehensive technical details and specifications"
                ],
                "medium": [
                    "shows solid technical understanding",
                    "employs standard methodologies with competent execution"
                ],
                "low": [
                    "limited technical depth or superficial analysis",
                    "lacks detailed technical specifications"
                ]
            },
            # Business profile explanations
            "market_impact": {
                "high": [
                    "demonstrates significant potential for market disruption",
                    "addresses large market opportunities with clear value proposition",
                    "shows strong competitive advantages and market positioning"
                ],
                "medium": [
                    "presents moderate market opportunities",
                    "shows some competitive advantages"
                ],
                "low": [
                    "limited market impact or unclear value proposition",
                    "faces significant market challenges"
                ]
            },
            "roi_potential": {
                "high": [
                    "demonstrates strong financial returns and profitability",
                    "shows clear path to revenue generation",
                    "presents compelling cost-benefit analysis"
                ],
                "medium": [
                    "shows moderate financial potential",
                    "presents reasonable return expectations"
                ],
                "low": [
                    "unclear financial benefits or poor return potential",
                    "high costs with uncertain revenue streams"
                ]
            },
            # Legal profile explanations
            "compliance_risk": {
                "high": [
                    "presents significant regulatory compliance challenges",
                    "involves complex legal requirements and potential violations",
                    "requires careful legal review and risk mitigation"
                ],
                "medium": [
                    "involves moderate compliance considerations",
                    "requires standard regulatory review"
                ],
                "low": [
                    "minimal compliance risks or well-established regulatory framework",
                    "standard compliance requirements"
                ]
            }
        }
    
    def explain_scores(self, scores_v2, text: str, profile_config: Dict[str, Any]) -> ExplanationResult:
        """
        Generate comprehensive explanation for dimension scores.
        
        Args:
            scores_v2: DimensionScoresV2 object with scores and metadata
            text: Original text that was analyzed
            profile_config: Profile configuration used for scoring
            
        Returns:
            ExplanationResult with detailed explanations
        """
        # Generate detailed explanations for each dimension
        detailed_explanations = {}
        reasoning_chain = []
        
        for dimension, score in scores_v2.scores.items():
            explanation = self._explain_dimension_score(dimension, score, text)
            detailed_explanations[dimension] = explanation
            
            if score > 0.3:  # Only include significant dimensions in reasoning chain
                reasoning_chain.append(f"{dimension.replace('_', ' ').title()}: {explanation}")
        
        # Generate overall summary
        summary = self._generate_summary(scores_v2, profile_config)
        
        # Assess confidence
        confidence_assessment = self._assess_confidence(scores_v2.confidence)
        
        # Profile context
        profile_context = self._generate_profile_context(scores_v2.profile, profile_config)
        
        return ExplanationResult(
            summary=summary,
            detailed_explanations=detailed_explanations,
            reasoning_chain=reasoning_chain,
            confidence_assessment=confidence_assessment,
            profile_context=profile_context
        )
    
    def _explain_dimension_score(self, dimension: str, score: float, text: str) -> str:
        """Generate explanation for a specific dimension score."""
        # Determine score level
        if score > 0.6:
            level = "high"
        elif score > 0.3:
            level = "medium"
        else:
            level = "low"
        
        # Get base explanation patterns
        patterns = self.explanation_patterns.get(dimension, {}).get(level, [])
        
        if patterns:
            base_explanation = patterns[0]  # Use first pattern as base
        else:
            base_explanation = f"shows {level} {dimension.replace('_', ' ')}"
        
        # Add specific evidence from text
        evidence = self._find_evidence_in_text(dimension, text, score)
        
        # Combine explanation with evidence
        if evidence:
            return f"{base_explanation} (score: {score:.2f}) - {evidence}"
        else:
            return f"{base_explanation} (score: {score:.2f})"
    
    def _find_evidence_in_text(self, dimension: str, text: str, score: float) -> str:
        """Find specific evidence in text that supports the dimension score."""
        text_lower = text.lower()
        
        # Dimension-specific evidence patterns
        evidence_patterns = {
            "utility": [
                r'\b(?:practical|application|implementation|solution|benefit)\b',
                r'\b(?:useful|valuable|applicable|effective)\b'
            ],
            "complexity": [
                r'\b(?:algorithm|technical|sophisticated|advanced|complex)\b',
                r'\b(?:methodology|framework|architecture)\b'
            ],
            "novelty": [
                r'\b(?:novel|innovative|breakthrough|new|original)\b',
                r'\b(?:first|pioneering|cutting-edge|state-of-the-art)\b'
            ],
            "credibility": [
                r'\b(?:peer-reviewed|validated|verified|evidence|research)\b',
                r'\b(?:study|analysis|data|findings)\b'
            ]
        }
        
        patterns = evidence_patterns.get(dimension, [])
        found_terms = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            found_terms.extend(matches[:2])  # Limit to 2 matches per pattern
        
        if found_terms:
            unique_terms = list(set(found_terms))[:3]  # Limit to 3 unique terms
            return f"evidence includes: {', '.join(unique_terms)}"
        
        return ""
    
    def _generate_summary(self, scores_v2, profile_config: Dict[str, Any]) -> str:
        """Generate overall summary of dimension analysis."""
        high_dims = scores_v2.get_high_dimensions(threshold=0.6)
        medium_dims = [(dim, score) for dim, score in scores_v2.scores.items() 
                      if 0.3 < score <= 0.6]
        
        profile_name = scores_v2.profile
        
        if high_dims:
            high_list = [f"{dim.replace('_', ' ')} ({score:.2f})" for dim, score in high_dims]
            summary = f"Using {profile_name} profile, this content shows high {', '.join(high_list)}"
            
            if medium_dims:
                medium_list = [dim.replace('_', ' ') for dim, _ in medium_dims[:2]]
                summary += f" with moderate {', '.join(medium_list)}"
        elif medium_dims:
            medium_list = [f"{dim.replace('_', ' ')} ({score:.2f})" for dim, score in medium_dims[:3]]
            summary = f"Using {profile_name} profile, this content shows moderate {', '.join(medium_list)}"
        else:
            summary = f"Using {profile_name} profile, this content shows generally low dimension scores"
        
        return summary + "."
    
    def _assess_confidence(self, confidence_scores: Dict[str, float]) -> str:
        """Assess overall confidence in the dimension scoring."""
        if not confidence_scores:
            return "Confidence assessment unavailable"
        
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        
        for (min_conf, max_conf), level in self.confidence_levels.items():
            if min_conf <= avg_confidence < max_conf:
                return f"Overall confidence is {level} ({avg_confidence:.2f})"
        
        return f"Confidence: {avg_confidence:.2f}"
    
    def _generate_profile_context(self, profile_name: str, profile_config: Dict[str, Any]) -> str:
        """Generate context about the profile used for analysis."""
        description = profile_config.get('description', f'{profile_name} profile')
        target_users = profile_config.get('target_users', [])
        
        context = f"Analysis performed using {profile_name} profile: {description}"
        
        if target_users:
            user_list = ', '.join(target_users[:3])
            context += f" Optimized for: {user_list}"
        
        return context + "."
    
    def generate_comparison_explanation(self, chunks_with_scores: List[Tuple[Any, Any]]) -> str:
        """Generate explanation comparing multiple chunks' dimension scores."""
        if len(chunks_with_scores) < 2:
            return "Insufficient chunks for comparison"
        
        # Find the highest scoring chunk for each dimension
        dimension_leaders = {}
        
        for chunk, scores in chunks_with_scores:
            for dim, score in scores.scores.items():
                if dim not in dimension_leaders or score > dimension_leaders[dim][1]:
                    dimension_leaders[dim] = (chunk, score)
        
        explanations = []
        for dim, (leader_chunk, score) in dimension_leaders.items():
            if score > 0.4:  # Only mention significant scores
                explanations.append(f"Highest {dim.replace('_', ' ')}: {score:.2f}")
        
        if explanations:
            return "Comparison highlights: " + "; ".join(explanations[:3])
        else:
            return "All chunks show similar, moderate dimension scores"
    
    def explain_retrieval_ranking(self, ranked_chunks: List[Tuple[Any, float, Any]], 
                                 query_context: str) -> str:
        """Explain why chunks were ranked in a specific order."""
        if not ranked_chunks:
            return "No chunks to explain"
        
        top_chunk, top_score, top_dimensions = ranked_chunks[0]
        
        # Identify key factors in ranking
        high_dims = [(dim, score) for dim, score in top_dimensions.scores.items() if score > 0.5]
        
        if high_dims:
            dim_explanations = [f"{dim.replace('_', ' ')} ({score:.2f})" 
                              for dim, score in high_dims[:2]]
            return f"Top result ranked highly due to strong {', '.join(dim_explanations)} in context of: {query_context}"
        else:
            return f"Top result selected based on overall relevance to: {query_context}"
