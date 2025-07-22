"""
Principle Validator for Cognitive Distillation Engine
====================================================

Validates discovered cognitive principles for quality and usefulness.

Author: SAM Development Team
Version: 1.0.0
"""

import re
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of principle validation."""
    is_valid: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]
    quality_score: float

class PrincipleValidator:
    """Validates cognitive principles for quality and usefulness."""
    
    def __init__(self):
        """Initialize the principle validator."""
        self.min_length = 15
        self.max_length = 250
        self.min_confidence = 0.4

        # Enhanced quality criteria weights for Phase 1B
        self.criteria_weights = {
            'clarity': 0.25,
            'actionability': 0.25,
            'specificity': 0.20,
            'generalizability': 0.15,
            'novelty': 0.15
        }

        # Known good principle patterns for validation
        self.good_principle_patterns = [
            r'for\s+\w+\s+queries?,\s+prioritize\s+\w+',
            r'when\s+\w+,\s+emphasize\s+\w+',
            r'for\s+\w+\s+topics?,\s+\w+\s+specific\s+\w+',
            r'in\s+\w+\s+contexts?,\s+\w+\s+\w+\s+sources?'
        ]

        # Anti-patterns to avoid
        self.anti_patterns = [
            r'be\s+good',
            r'provide\s+accurate',
            r'answer\s+correctly',
            r'give\s+helpful',
            r'respond\s+appropriately'
        ]

        logger.info("Enhanced principle validator initialized for Phase 1B")
    
    def validate_principle(self, principle_text: str) -> Dict[str, Any]:
        """
        Validate a discovered cognitive principle.
        
        Args:
            principle_text: The principle to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            issues = []
            suggestions = []
            scores = {}
            
            # Basic format validation
            if not self._validate_format(principle_text, issues, suggestions):
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'issues': issues,
                    'suggestions': suggestions,
                    'quality_score': 0.0
                }
            
            # Quality criteria evaluation
            scores['clarity'] = self._evaluate_clarity(principle_text)
            scores['actionability'] = self._evaluate_actionability(principle_text)
            scores['specificity'] = self._evaluate_specificity(principle_text)
            scores['generalizability'] = self._evaluate_generalizability(principle_text)
            scores['novelty'] = self._evaluate_novelty(principle_text)
            
            # Calculate overall quality score
            quality_score = sum(
                scores[criterion] * weight 
                for criterion, weight in self.criteria_weights.items()
            )
            
            # Determine validation result
            is_valid = quality_score >= self.min_confidence
            confidence = min(1.0, quality_score)
            
            # Add quality-based suggestions
            self._add_quality_suggestions(scores, suggestions)

            # Enhanced validation for LLM-generated principles
            self._validate_llm_principle_quality(principle_text, scores, issues, suggestions)
            
            return {
                'is_valid': is_valid,
                'confidence': confidence,
                'issues': issues,
                'suggestions': suggestions,
                'quality_score': quality_score,
                'detailed_scores': scores
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'is_valid': False,
                'confidence': 0.0,
                'issues': [f"Validation error: {str(e)}"],
                'suggestions': [],
                'quality_score': 0.0
            }
    
    def _validate_format(self, principle_text: str, issues: List[str], suggestions: List[str]) -> bool:
        """Validate basic format requirements."""
        if not principle_text or not principle_text.strip():
            issues.append("Principle text is empty")
            return False
        
        principle_text = principle_text.strip()
        
        # Length validation
        if len(principle_text) < self.min_length:
            issues.append(f"Principle too short (minimum {self.min_length} characters)")
            suggestions.append("Provide more specific guidance in the principle")
            return False
        
        if len(principle_text) > self.max_length:
            issues.append(f"Principle too long (maximum {self.max_length} characters)")
            suggestions.append("Make the principle more concise and focused")
            return False
        
        # Basic content validation
        if not re.search(r'[a-zA-Z]', principle_text):
            issues.append("Principle contains no alphabetic characters")
            return False
        
        # Check for complete sentences
        if not principle_text.endswith(('.', '!', '?')):
            suggestions.append("End the principle with proper punctuation")
        
        return True
    
    def _evaluate_clarity(self, principle_text: str) -> float:
        """Evaluate how clear and understandable the principle is."""
        score = 0.5  # Base score
        
        # Positive indicators
        if len(principle_text.split()) >= 5:  # Sufficient detail
            score += 0.2
        
        if any(word in principle_text.lower() for word in ['when', 'if', 'for', 'during']):
            score += 0.1  # Conditional clarity
        
        if any(word in principle_text.lower() for word in ['should', 'must', 'always', 'never']):
            score += 0.1  # Clear directives
        
        # Negative indicators
        if len(principle_text.split()) > 30:  # Too verbose
            score -= 0.2
        
        if principle_text.count(',') > 3:  # Too many clauses
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_actionability(self, principle_text: str) -> float:
        """Evaluate how actionable the principle is."""
        score = 0.3  # Base score
        
        # Action words
        action_words = ['prioritize', 'focus', 'emphasize', 'include', 'avoid', 'use', 'apply', 'consider']
        if any(word in principle_text.lower() for word in action_words):
            score += 0.3
        
        # Specific guidance
        if any(word in principle_text.lower() for word in ['specific', 'detailed', 'recent', 'relevant']):
            score += 0.2
        
        # Context specification
        if any(word in principle_text.lower() for word in ['for', 'when', 'during', 'in case of']):
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_specificity(self, principle_text: str) -> float:
        """Evaluate how specific the principle is."""
        score = 0.4  # Base score
        
        # Domain-specific terms
        domains = ['financial', 'technical', 'medical', 'legal', 'research', 'academic']
        if any(domain in principle_text.lower() for domain in domains):
            score += 0.3
        
        # Specific methods or approaches
        methods = ['cite', 'reference', 'analyze', 'compare', 'validate', 'verify']
        if any(method in principle_text.lower() for method in methods):
            score += 0.2
        
        # Avoid overly generic terms
        generic_terms = ['good', 'better', 'best', 'appropriate', 'suitable']
        if any(term in principle_text.lower() for term in generic_terms):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_generalizability(self, principle_text: str) -> float:
        """Evaluate how generalizable the principle is."""
        score = 0.5  # Base score
        
        # Too specific (negative)
        if len([word for word in principle_text.split() if word.isupper()]) > 2:
            score -= 0.2  # Too many proper nouns
        
        # Good generalization indicators
        if any(word in principle_text.lower() for word in ['topics', 'queries', 'questions', 'situations']):
            score += 0.2
        
        if any(word in principle_text.lower() for word in ['generally', 'typically', 'usually']):
            score += 0.1
        
        # Overly specific indicators (negative)
        if any(word in principle_text.lower() for word in ['always', 'never', 'only', 'exclusively']):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_novelty(self, principle_text: str) -> float:
        """Evaluate how novel/non-obvious the principle is."""
        score = 0.6  # Base score (assume moderate novelty)
        
        # Common sense indicators (negative)
        obvious_phrases = [
            'be accurate', 'provide good', 'answer correctly', 'be helpful',
            'give information', 'respond appropriately'
        ]
        if any(phrase in principle_text.lower() for phrase in obvious_phrases):
            score -= 0.3
        
        # Novel approach indicators
        novel_indicators = [
            'prioritize recency', 'emphasize sources', 'cross-reference',
            'validate against', 'synthesize from', 'contextualize within'
        ]
        if any(indicator in principle_text.lower() for indicator in novel_indicators):
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _add_quality_suggestions(self, scores: Dict[str, float], suggestions: List[str]):
        """Add suggestions based on quality scores."""
        if scores['clarity'] < 0.5:
            suggestions.append("Make the principle clearer and more understandable")
        
        if scores['actionability'] < 0.5:
            suggestions.append("Add more specific, actionable guidance")
        
        if scores['specificity'] < 0.5:
            suggestions.append("Include more domain-specific or method-specific details")
        
        if scores['generalizability'] < 0.5:
            suggestions.append("Make the principle more broadly applicable")
        
        if scores['novelty'] < 0.4:
            suggestions.append("Ensure the principle provides non-obvious insights")

    def _validate_llm_principle_quality(self, principle_text: str, scores: Dict[str, float],
                                       issues: List[str], suggestions: List[str]):
        """Enhanced validation for LLM-generated principles."""
        import re

        # Check for good principle patterns
        has_good_pattern = any(
            re.search(pattern, principle_text, re.IGNORECASE)
            for pattern in self.good_principle_patterns
        )

        if has_good_pattern:
            scores['pattern_match'] = 0.8
        else:
            scores['pattern_match'] = 0.3
            suggestions.append("Consider using more structured principle patterns (e.g., 'For X queries, prioritize Y')")

        # Check for anti-patterns
        has_anti_pattern = any(
            re.search(pattern, principle_text, re.IGNORECASE)
            for pattern in self.anti_patterns
        )

        if has_anti_pattern:
            issues.append("Principle contains generic/obvious statements")
            suggestions.append("Avoid generic advice like 'be good' or 'provide accurate information'")
            scores['novelty'] = max(0.0, scores['novelty'] - 0.3)

        # Check for domain specificity
        domain_indicators = ['financial', 'technical', 'medical', 'legal', 'research', 'academic', 'scientific']
        has_domain = any(domain in principle_text.lower() for domain in domain_indicators)

        if has_domain:
            scores['domain_specificity'] = 0.7
        else:
            scores['domain_specificity'] = 0.4
            suggestions.append("Consider making the principle more domain-specific")

        # Check for actionable verbs
        actionable_verbs = ['prioritize', 'emphasize', 'focus', 'cite', 'reference', 'analyze', 'synthesize', 'validate']
        has_actionable_verb = any(verb in principle_text.lower() for verb in actionable_verbs)

        if has_actionable_verb:
            scores['actionable_verbs'] = 0.8
        else:
            scores['actionable_verbs'] = 0.3
            suggestions.append("Include more actionable verbs (prioritize, emphasize, focus, etc.)")

        # Update overall actionability score
        scores['actionability'] = (scores['actionability'] + scores.get('actionable_verbs', 0.5)) / 2
    
    def validate_batch(self, principles: List[str]) -> List[Dict[str, Any]]:
        """Validate a batch of principles."""
        return [self.validate_principle(principle) for principle in principles]
    
    def get_validation_summary(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics for validation results."""
        if not validation_results:
            return {}
        
        valid_count = sum(1 for result in validation_results if result['is_valid'])
        avg_confidence = sum(result['confidence'] for result in validation_results) / len(validation_results)
        avg_quality = sum(result['quality_score'] for result in validation_results) / len(validation_results)
        
        return {
            'total_principles': len(validation_results),
            'valid_principles': valid_count,
            'validation_rate': round(valid_count / len(validation_results) * 100, 1),
            'avg_confidence': round(avg_confidence, 3),
            'avg_quality_score': round(avg_quality, 3)
        }

# Global validator instance
principle_validator = PrincipleValidator()
