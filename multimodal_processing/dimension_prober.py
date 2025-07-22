#!/usr/bin/env python3
"""
Conceptual Dimension Probing for SAM
Implements human-like conceptual understanding through dimension scoring.
Based on LLMs_core_dimensions research for enhanced document QA accuracy.
"""

import os
import sys
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Add plugins directory to path for SPoSE access
PLUGINS_DIR = Path(__file__).parent.parent / "plugins" / "dimension_probing"
sys.path.append(str(PLUGINS_DIR / "model_training" / "SPoSE"))

try:
    from models.model import SPoSE
    SPOSE_AVAILABLE = True
except ImportError:
    SPOSE_AVAILABLE = False
    logging.warning("SPoSE model not available. Dimension probing will use fallback methods.")

@dataclass
class DimensionScores:
    """Container for conceptual dimension scores."""
    danger: float = 0.0
    complexity: float = 0.0
    utility: float = 0.0
    sensitivity: float = 0.0
    moral_weight: float = 0.0
    
    # Government/Defense-specific dimensions
    classification_level: float = 0.0  # Unclassified=0.0, Confidential=0.3, Secret=0.6, Top Secret=1.0
    itar_sensitivity: float = 0.0      # Export control relevance
    operational_impact: float = 0.0    # Mission criticality
    innovation_potential: float = 0.0  # For SBIR scoring
    technical_readiness: float = 0.0   # TRL level (0.0-1.0 normalized)
    
    # Confidence scores for each dimension
    confidence: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence is None:
            self.confidence = {
                'danger': 0.5,
                'complexity': 0.5,
                'utility': 0.5,
                'sensitivity': 0.5,
                'moral_weight': 0.5,
                'classification_level': 0.5,
                'itar_sensitivity': 0.5,
                'operational_impact': 0.5,
                'innovation_potential': 0.5,
                'technical_readiness': 0.5
            }

@dataclass
class DimensionProbeResult:
    """Result of dimension probing with reasoning."""
    scores: DimensionScores
    reasoning: Dict[str, str]
    probe_method: str
    processing_time_ms: float

class DimensionProber:
    """
    Conceptual Dimension Probing System for SAM
    
    Implements human-like conceptual understanding through dimension scoring
    to enhance document QA accuracy and provide security-aware processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dimension prober with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core dimensions from research
        self.core_dimensions = [
            'danger', 'complexity', 'utility', 'sensitivity', 'moral_weight'
        ]
        
        # Government/Defense-specific dimensions
        self.defense_dimensions = [
            'classification_level', 'itar_sensitivity', 'operational_impact',
            'innovation_potential', 'technical_readiness'
        ]
        
        self.all_dimensions = self.core_dimensions + self.defense_dimensions
        
        # Pattern-based scoring rules
        self._init_pattern_rules()
        
        # Load SPoSE models if available
        self.spose_models = {}
        if SPOSE_AVAILABLE:
            self._load_spose_models()
    
    def _init_pattern_rules(self):
        """Initialize pattern-based scoring rules for fallback."""
        self.danger_patterns = [
            (r'\b(?:attack|exploit|vulnerability|breach|malware|virus|trojan)\b', 0.8),
            (r'\b(?:weapon|explosive|bomb|missile|ammunition)\b', 0.9),
            (r'\b(?:classified|secret|confidential|restricted)\b', 0.7),
            (r'\b(?:cyber|hacking|penetration|intrusion)\b', 0.6),
            (r'\b(?:threat|risk|danger|hazard|critical)\b', 0.5),
        ]
        
        self.complexity_patterns = [
            (r'\b(?:algorithm|neural|machine learning|AI|artificial intelligence)\b', 0.8),
            (r'\b(?:quantum|cryptographic|encryption|decryption)\b', 0.9),
            (r'\b(?:multi-layer|hierarchical|distributed|parallel)\b', 0.7),
            (r'\b(?:advanced|sophisticated|complex|intricate)\b', 0.6),
            (r'\b(?:technical|engineering|scientific|mathematical)\b', 0.5),
        ]
        
        self.utility_patterns = [
            (r'\b(?:capability|function|feature|tool|system)\b', 0.7),
            (r'\b(?:operational|mission|tactical|strategic)\b', 0.8),
            (r'\b(?:effective|efficient|useful|valuable)\b', 0.6),
            (r'\b(?:requirement|specification|objective|goal)\b', 0.5),
        ]
        
        self.sensitivity_patterns = [
            (r'\b(?:classified|secret|confidential|restricted|sensitive)\b', 0.9),
            (r'\b(?:ITAR|export control|controlled technology)\b', 0.8),
            (r'\b(?:proprietary|confidential|private|internal)\b', 0.6),
            (r'\b(?:security|clearance|authorization)\b', 0.7),
        ]
        
        self.innovation_patterns = [
            (r'\b(?:novel|innovative|breakthrough|cutting-edge|state-of-the-art)\b', 0.8),
            (r'\b(?:research|development|prototype|experimental)\b', 0.7),
            (r'\b(?:SBIR|Phase I|Phase II|commercialization)\b', 0.9),
            (r'\b(?:patent|intellectual property|IP)\b', 0.6),
        ]
    
    def _load_spose_models(self):
        """Load pre-trained SPoSE models if available."""
        try:
            models_dir = PLUGINS_DIR / "model_training" / "SPoSE" / "results"
            if models_dir.exists():
                self.logger.info("SPoSE models directory found, loading pre-trained models...")
                # Implementation would load specific models here
                # For now, we'll use pattern-based fallback
            else:
                self.logger.info("No pre-trained SPoSE models found, using pattern-based scoring")
        except Exception as e:
            self.logger.warning(f"Failed to load SPoSE models: {e}")
    
    def probe_chunk(self, text: str, context: Optional[Dict[str, Any]] = None) -> DimensionProbeResult:
        """
        Probe a text chunk for conceptual dimensions.
        
        Args:
            text: Text content to analyze
            context: Optional context information (source, document type, etc.)
            
        Returns:
            DimensionProbeResult with scores and reasoning
        """
        import time
        start_time = time.time()
        
        # Use pattern-based scoring for now (can be enhanced with SPoSE later)
        scores = self._pattern_based_scoring(text, context)
        reasoning = self._generate_reasoning(text, scores)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return DimensionProbeResult(
            scores=scores,
            reasoning=reasoning,
            probe_method="pattern_based",
            processing_time_ms=processing_time
        )
    
    def _pattern_based_scoring(self, text: str, context: Optional[Dict[str, Any]] = None) -> DimensionScores:
        """Score text using pattern-based rules."""
        text_lower = text.lower()
        
        # Core dimensions
        danger_score = self._score_patterns(text_lower, self.danger_patterns)
        complexity_score = self._score_patterns(text_lower, self.complexity_patterns)
        utility_score = self._score_patterns(text_lower, self.utility_patterns)
        sensitivity_score = self._score_patterns(text_lower, self.sensitivity_patterns)
        
        # Moral weight (based on danger and sensitivity)
        moral_weight_score = min(1.0, (danger_score + sensitivity_score) / 2.0)
        
        # Defense-specific dimensions
        classification_score = self._score_classification_level(text_lower)
        itar_score = self._score_itar_sensitivity(text_lower)
        operational_score = self._score_operational_impact(text_lower, context)
        innovation_score = self._score_patterns(text_lower, self.innovation_patterns)
        trl_score = self._score_technical_readiness(text_lower)
        
        # Generate confidence scores based on pattern matches
        confidence = self._calculate_confidence_scores(text_lower)
        
        return DimensionScores(
            danger=danger_score,
            complexity=complexity_score,
            utility=utility_score,
            sensitivity=sensitivity_score,
            moral_weight=moral_weight_score,
            classification_level=classification_score,
            itar_sensitivity=itar_score,
            operational_impact=operational_score,
            innovation_potential=innovation_score,
            technical_readiness=trl_score,
            confidence=confidence
        )
    
    def _score_patterns(self, text: str, patterns: List[Tuple[str, float]]) -> float:
        """Score text against a list of regex patterns."""
        max_score = 0.0
        total_weight = 0.0
        
        for pattern, weight in patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                # Logarithmic scaling for multiple matches
                score = min(1.0, weight * (1 + 0.1 * np.log(matches)))
                max_score = max(max_score, score)
                total_weight += weight * matches
        
        # Combine max score with weighted average
        if total_weight > 0:
            return min(1.0, (max_score + total_weight / len(patterns)) / 2.0)
        return 0.0
    
    def _score_classification_level(self, text: str) -> float:
        """Score classification level based on security keywords."""
        if re.search(r'\b(?:top secret|ts)\b', text):
            return 1.0
        elif re.search(r'\b(?:secret|s)\b', text):
            return 0.6
        elif re.search(r'\b(?:confidential|c)\b', text):
            return 0.3
        elif re.search(r'\b(?:unclassified|u)\b', text):
            return 0.0
        elif re.search(r'\b(?:classified|restricted|sensitive)\b', text):
            return 0.5  # Generic classification indicator
        return 0.0
    
    def _score_itar_sensitivity(self, text: str) -> float:
        """Score ITAR/export control sensitivity."""
        itar_patterns = [
            r'\bitar\b',
            r'\bexport control\b',
            r'\bcontrolled technology\b',
            r'\bmunitions list\b',
            r'\bdefense article\b',
            r'\bdefense service\b'
        ]
        
        score = 0.0
        for pattern in itar_patterns:
            if re.search(pattern, text):
                score = max(score, 0.8)
        
        # Additional scoring for defense/military context
        if re.search(r'\b(?:military|defense|weapon|missile|radar)\b', text):
            score = max(score, 0.6)
        
        return score
    
    def _score_operational_impact(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Score operational/mission impact."""
        impact_patterns = [
            (r'\b(?:mission critical|critical|essential)\b', 0.9),
            (r'\b(?:operational|mission|tactical)\b', 0.7),
            (r'\b(?:strategic|important|significant)\b', 0.6),
            (r'\b(?:capability|requirement|objective)\b', 0.5),
        ]
        
        return self._score_patterns(text, impact_patterns)
    
    def _score_technical_readiness(self, text: str) -> float:
        """Score Technical Readiness Level (TRL)."""
        trl_patterns = [
            (r'\btrl\s*[789]\b', 0.9),  # High TRL
            (r'\btrl\s*[456]\b', 0.6),  # Medium TRL
            (r'\btrl\s*[123]\b', 0.3),  # Low TRL
            (r'\b(?:prototype|demonstration|testing)\b', 0.7),
            (r'\b(?:research|concept|theoretical)\b', 0.3),
            (r'\b(?:production|deployment|operational)\b', 0.9),
        ]
        
        return self._score_patterns(text, trl_patterns)
    
    def _calculate_confidence_scores(self, text: str) -> Dict[str, float]:
        """Calculate confidence scores for each dimension."""
        # Base confidence on text length and pattern density
        text_length = len(text)
        word_count = len(text.split())
        
        # Higher confidence for longer, more detailed text
        base_confidence = min(0.9, 0.3 + (word_count / 100) * 0.6)
        
        return {dim: base_confidence for dim in self.all_dimensions}
    
    def _generate_reasoning(self, text: str, scores: DimensionScores) -> Dict[str, str]:
        """Generate human-readable reasoning for dimension scores."""
        reasoning = {}
        
        # Generate reasoning for each dimension
        if scores.danger > 0.5:
            reasoning['danger'] = f"High danger score ({scores.danger:.2f}) due to security/threat-related content"
        elif scores.danger > 0.2:
            reasoning['danger'] = f"Moderate danger score ({scores.danger:.2f}) with some risk indicators"
        else:
            reasoning['danger'] = f"Low danger score ({scores.danger:.2f}) - minimal security concerns"
        
        if scores.complexity > 0.5:
            reasoning['complexity'] = f"High complexity score ({scores.complexity:.2f}) indicating technical sophistication"
        elif scores.complexity > 0.2:
            reasoning['complexity'] = f"Moderate complexity score ({scores.complexity:.2f}) with technical elements"
        else:
            reasoning['complexity'] = f"Low complexity score ({scores.complexity:.2f}) - straightforward content"
        
        if scores.utility > 0.5:
            reasoning['utility'] = f"High utility score ({scores.utility:.2f}) showing practical value"
        else:
            reasoning['utility'] = f"Moderate utility score ({scores.utility:.2f})"
        
        if scores.sensitivity > 0.5:
            reasoning['sensitivity'] = f"High sensitivity score ({scores.sensitivity:.2f}) - contains sensitive information"
        else:
            reasoning['sensitivity'] = f"Low sensitivity score ({scores.sensitivity:.2f}) - public information"
        
        if scores.innovation_potential > 0.5:
            reasoning['innovation_potential'] = f"High innovation score ({scores.innovation_potential:.2f}) - novel/cutting-edge content"
        else:
            reasoning['innovation_potential'] = f"Standard innovation score ({scores.innovation_potential:.2f})"
        
        return reasoning
    
    def get_dimension_summary(self, scores: DimensionScores) -> str:
        """Generate a human-readable summary of dimension scores."""
        high_dims = []
        medium_dims = []
        
        for dim in self.all_dimensions:
            score = getattr(scores, dim)
            if score > 0.6:
                high_dims.append(f"{dim}({score:.2f})")
            elif score > 0.3:
                medium_dims.append(f"{dim}({score:.2f})")
        
        summary_parts = []
        if high_dims:
            summary_parts.append(f"High: {', '.join(high_dims)}")
        if medium_dims:
            summary_parts.append(f"Medium: {', '.join(medium_dims)}")
        
        return "; ".join(summary_parts) if summary_parts else "All dimensions low"
