#!/usr/bin/env python3
"""
Phase 5 Integration Component for SAM
Integrates reflective meta-reasoning with existing SAM response generation.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Import Phase 5 components
from .reflective_meta_reasoning import ReflectiveMetaReasoningEngine, ReflectiveResult, CritiqueLevel
from .dimension_conflict_detector import AdvancedDimensionConflictDetector, DimensionConflict
from .confidence_justifier import AdvancedConfidenceJustifier, ConfidenceJustification

# Import existing SAM components
try:
    from .self_discover_critic import SelfDiscoverCriticFramework
    SELF_DISCOVER_AVAILABLE = True
except ImportError:
    SELF_DISCOVER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class Phase5EnhancedResponse:
    """Enhanced response with Phase 5 meta-reasoning capabilities."""
    # Original response
    original_response: str
    
    # Phase 5 enhancements
    reflective_result: ReflectiveResult
    dimension_conflicts: List[DimensionConflict]
    confidence_justification: ConfidenceJustification
    
    # Final enhanced response
    enhanced_response: str
    meta_reasoning_summary: str
    
    # Metadata
    processing_time_ms: int
    phase5_enabled: bool
    timestamp: str

class Phase5ResponseEnhancer:
    """
    Integrates Phase 5 reflective meta-reasoning capabilities with SAM's
    existing response generation system.
    """
    
    def __init__(self, 
                 critique_level: CritiqueLevel = CritiqueLevel.MODERATE,
                 profile: str = "general",
                 enable_dimension_conflicts: bool = True,
                 enable_confidence_justification: bool = True):
        """Initialize Phase 5 response enhancer."""
        
        self.critique_level = critique_level
        self.profile = profile
        self.enable_dimension_conflicts = enable_dimension_conflicts
        self.enable_confidence_justification = enable_confidence_justification
        
        # Initialize Phase 5 components
        try:
            self.reflective_engine = ReflectiveMetaReasoningEngine(critique_level)
            self.conflict_detector = AdvancedDimensionConflictDetector() if enable_dimension_conflicts else None
            self.confidence_justifier = AdvancedConfidenceJustifier(profile) if enable_confidence_justification else None
            
            self.phase5_available = True
            logger.info(f"Phase 5 Response Enhancer initialized (profile: {profile}, critique: {critique_level.value})")
            
        except Exception as e:
            logger.warning(f"Phase 5 components not fully available: {e}")
            self.reflective_engine = None
            self.conflict_detector = None
            self.confidence_justifier = None
            self.phase5_available = False
    
    def enhance_response(self, 
                        query: str,
                        initial_response: str,
                        context: Optional[Dict[str, Any]] = None) -> Phase5EnhancedResponse:
        """
        Enhance response with Phase 5 reflective meta-reasoning.
        
        Args:
            query: Original user query
            initial_response: Initial response from SAM
            context: Context including memory results, tool outputs, etc.
            
        Returns:
            Phase5EnhancedResponse with meta-reasoning enhancements
        """
        start_time = time.time()
        
        if not self.phase5_available:
            return self._create_fallback_response(query, initial_response, "Phase 5 components not available")
        
        try:
            # Stage 1: Reflective Meta-Reasoning
            reflective_result = self.reflective_engine.reflective_reasoning_cycle(
                query, initial_response, context
            )
            
            # Stage 2: Dimension Conflict Detection (if enabled)
            dimension_conflicts = []
            if self.conflict_detector and reflective_result.response_analysis.get("dimension_scores"):
                dimension_conflicts = self.conflict_detector.detect_conflicts(
                    reflective_result.response_analysis["dimension_scores"],
                    context
                )
            
            # Stage 3: Confidence Justification (if enabled)
            confidence_justification = None
            if self.confidence_justifier:
                confidence_context = {
                    "critiques": [c.__dict__ for c in reflective_result.adversarial_critiques],
                    "conflicts": [c.__dict__ for c in dimension_conflicts]
                }
                confidence_justification = self.confidence_justifier.justify_confidence(
                    reflective_result.response_analysis,
                    confidence_context
                )
            
            # Stage 4: Synthesize Enhanced Response
            enhanced_response = self._synthesize_enhanced_response(
                initial_response, reflective_result, dimension_conflicts, confidence_justification
            )
            
            # Stage 5: Generate Meta-Reasoning Summary
            meta_reasoning_summary = self._generate_meta_reasoning_summary(
                reflective_result, dimension_conflicts, confidence_justification
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            result = Phase5EnhancedResponse(
                original_response=initial_response,
                reflective_result=reflective_result,
                dimension_conflicts=dimension_conflicts,
                confidence_justification=confidence_justification,
                enhanced_response=enhanced_response,
                meta_reasoning_summary=meta_reasoning_summary,
                processing_time_ms=processing_time,
                phase5_enabled=True,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Response enhanced with Phase 5 meta-reasoning in {processing_time}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error in Phase 5 response enhancement: {e}")
            return self._create_fallback_response(query, initial_response, str(e))
    
    def _synthesize_enhanced_response(self,
                                    initial_response: str,
                                    reflective_result: ReflectiveResult,
                                    dimension_conflicts: List[DimensionConflict],
                                    confidence_justification: Optional[ConfidenceJustification]) -> str:
        """Synthesize enhanced response with meta-reasoning insights."""
        
        # Start with the final response from reflective reasoning
        enhanced_parts = [reflective_result.final_response]
        
        # Add critical dimension conflicts if any
        critical_conflicts = [c for c in dimension_conflicts if c.severity.value in ["high", "critical"]]
        if critical_conflicts:
            enhanced_parts.append("\n\n🚨 **Critical Analysis Alert:**")
            for conflict in critical_conflicts[:2]:  # Limit to top 2
                enhanced_parts.append(f"⚠️ **{conflict.conflict_type.value.replace('_', ' ').title()}:** {conflict.description}")
                enhanced_parts.append(f"💡 **Recommendation:** {conflict.recommendation}")
        
        # Add confidence assessment if available
        if confidence_justification:
            confidence_level = confidence_justification.confidence_level.value.replace('_', ' ').title()
            enhanced_parts.append(f"\n\n📊 **Confidence Assessment:** {confidence_level} ({confidence_justification.confidence_score:.2f})")
            enhanced_parts.append(f"🔍 **Reliability:** {confidence_justification.reliability_assessment}")
            
            # Add primary factors
            if confidence_justification.primary_factors:
                enhanced_parts.append("✅ **Key Strengths:** " + "; ".join(confidence_justification.primary_factors[:2]))
            
            # Add limiting factors if any
            if confidence_justification.limiting_factors:
                enhanced_parts.append("⚠️ **Limitations:** " + "; ".join(confidence_justification.limiting_factors[:2]))
        
        # Add meta-reasoning note
        enhanced_parts.append(f"\n\n🧠 **Meta-Reasoning Applied:** {self.critique_level.value.title()} level analysis with {len(reflective_result.alternative_perspectives)} alternative perspectives considered")
        
        return "\n".join(enhanced_parts)
    
    def _generate_meta_reasoning_summary(self,
                                       reflective_result: ReflectiveResult,
                                       dimension_conflicts: List[DimensionConflict],
                                       confidence_justification: Optional[ConfidenceJustification]) -> str:
        """Generate summary of meta-reasoning process."""
        
        summary_parts = []
        
        # Reflective analysis summary
        summary_parts.append(f"Reflective Analysis: {len(reflective_result.alternative_perspectives)} perspectives, {len(reflective_result.adversarial_critiques)} critiques")
        
        # Dimension conflicts summary
        if dimension_conflicts:
            high_severity = len([c for c in dimension_conflicts if c.severity.value in ["high", "critical"]])
            summary_parts.append(f"Dimension Conflicts: {len(dimension_conflicts)} detected ({high_severity} high-severity)")
        
        # Confidence summary
        if confidence_justification:
            summary_parts.append(f"Confidence: {confidence_justification.confidence_level.value} ({confidence_justification.confidence_score:.2f})")
        
        # Processing summary
        summary_parts.append(f"Meta-Confidence: {reflective_result.meta_confidence:.2f}")
        summary_parts.append(f"Processing Time: {reflective_result.reflection_duration_ms}ms")
        
        return " | ".join(summary_parts)
    
    def _create_fallback_response(self, query: str, initial_response: str, error: str) -> Phase5EnhancedResponse:
        """Create fallback response when Phase 5 processing fails."""
        
        # Create minimal reflective result
        from .reflective_meta_reasoning import ReflectiveResult
        fallback_reflective_result = ReflectiveResult(
            session_id=f"fallback_{int(time.time() * 1000)}",
            original_query=query,
            initial_response=initial_response,
            response_analysis={"error": error},
            alternative_perspectives=[],
            adversarial_critiques=[],
            dimension_conflicts=[],
            confidence_justification=None,
            trade_off_analysis={},
            final_response=initial_response,
            reasoning_chain=[],
            critique_summary=f"Phase 5 processing failed: {error}",
            meta_confidence=0.3,
            reflection_duration_ms=0,
            timestamp=datetime.now().isoformat()
        )
        
        return Phase5EnhancedResponse(
            original_response=initial_response,
            reflective_result=fallback_reflective_result,
            dimension_conflicts=[],
            confidence_justification=None,
            enhanced_response=f"{initial_response}\n\n⚠️ **Note:** Advanced meta-reasoning temporarily unavailable.",
            meta_reasoning_summary=f"Fallback mode: {error}",
            processing_time_ms=0,
            phase5_enabled=False,
            timestamp=datetime.now().isoformat()
        )
    
    def configure_critique_level(self, critique_level: CritiqueLevel):
        """Dynamically configure critique level."""
        self.critique_level = critique_level
        if self.reflective_engine:
            self.reflective_engine.critique_level = critique_level
            logger.info(f"Critique level updated to: {critique_level.value}")
    
    def configure_profile(self, profile: str):
        """Dynamically configure reasoning profile."""
        self.profile = profile
        if self.confidence_justifier:
            self.confidence_justifier.profile = profile
            logger.info(f"Profile updated to: {profile}")
    
    def get_phase5_status(self) -> Dict[str, Any]:
        """Get status of Phase 5 components."""
        return {
            "phase5_available": self.phase5_available,
            "reflective_engine": self.reflective_engine is not None,
            "conflict_detector": self.conflict_detector is not None,
            "confidence_justifier": self.confidence_justifier is not None,
            "critique_level": self.critique_level.value,
            "profile": self.profile,
            "dimension_conflicts_enabled": self.enable_dimension_conflicts,
            "confidence_justification_enabled": self.enable_confidence_justification
        }


# Global Phase 5 enhancer instance
_phase5_enhancer = None

def get_phase5_enhancer(profile: str = "general", 
                       critique_level: CritiqueLevel = CritiqueLevel.MODERATE) -> Phase5ResponseEnhancer:
    """Get or create global Phase 5 enhancer instance."""
    global _phase5_enhancer
    
    if _phase5_enhancer is None:
        _phase5_enhancer = Phase5ResponseEnhancer(
            critique_level=critique_level,
            profile=profile
        )
    
    return _phase5_enhancer

def enhance_sam_response(query: str, 
                        initial_response: str,
                        context: Optional[Dict[str, Any]] = None,
                        profile: str = "general",
                        critique_level: CritiqueLevel = CritiqueLevel.MODERATE) -> Phase5EnhancedResponse:
    """Convenience function for enhancing SAM responses with Phase 5 meta-reasoning."""
    enhancer = get_phase5_enhancer(profile, critique_level)
    return enhancer.enhance_response(query, initial_response, context)
