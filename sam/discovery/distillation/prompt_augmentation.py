"""
Principle-Augmented Prompting System
===================================

Injects discovered cognitive principles into live reasoning prompts for improved performance.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .registry import PrincipleRegistry, CognitivePrinciple

logger = logging.getLogger(__name__)

@dataclass
class AugmentedPrompt:
    """Represents a prompt augmented with cognitive principles."""
    original_prompt: str
    augmented_prompt: str
    applied_principles: List[CognitivePrinciple]
    augmentation_metadata: Dict[str, Any]
    confidence_boost: float

class PromptAugmentation:
    """Handles principle-augmented prompting for live reasoning."""
    
    def __init__(self):
        """Initialize the prompt augmentation system."""
        self.registry = PrincipleRegistry()
        self.max_principles_per_prompt = 3
        self.min_principle_confidence = 0.5
        self.semantic_similarity_threshold = 0.6
        
        # Cache for frequently used principles
        self._principle_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes
        
        logger.info("Prompt augmentation system initialized")
    
    def augment_prompt(self, original_prompt: str, context: Dict[str, Any] = None) -> AugmentedPrompt:
        """
        Augment a prompt with relevant cognitive principles.
        
        Args:
            original_prompt: The original reasoning prompt
            context: Additional context for principle selection
            
        Returns:
            AugmentedPrompt with principles injected
        """
        try:
            # Extract domain and query type from prompt
            domain_info = self._extract_domain_info(original_prompt, context)
            
            # Find relevant principles
            relevant_principles = self._find_relevant_principles(
                original_prompt, domain_info, context
            )
            
            if not relevant_principles:
                logger.info("No relevant principles found for prompt augmentation")
                return AugmentedPrompt(
                    original_prompt=original_prompt,
                    augmented_prompt=original_prompt,
                    applied_principles=[],
                    augmentation_metadata={'reason': 'no_relevant_principles'},
                    confidence_boost=0.0
                )
            
            # Create augmented prompt
            augmented_prompt = self._create_augmented_prompt(original_prompt, relevant_principles)
            
            # Calculate confidence boost
            confidence_boost = self._calculate_confidence_boost(relevant_principles)
            
            # Track principle usage
            self._track_principle_usage(relevant_principles, original_prompt)
            
            logger.info(f"Augmented prompt with {len(relevant_principles)} principles")
            
            return AugmentedPrompt(
                original_prompt=original_prompt,
                augmented_prompt=augmented_prompt,
                applied_principles=relevant_principles,
                augmentation_metadata={
                    'domain_info': domain_info,
                    'principle_count': len(relevant_principles),
                    'confidence_boost': confidence_boost,
                    'augmentation_timestamp': datetime.now().isoformat()
                },
                confidence_boost=confidence_boost
            )
            
        except Exception as e:
            logger.error(f"Prompt augmentation failed: {e}")
            return AugmentedPrompt(
                original_prompt=original_prompt,
                augmented_prompt=original_prompt,
                applied_principles=[],
                augmentation_metadata={'error': str(e)},
                confidence_boost=0.0
            )
    
    def _extract_domain_info(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract domain and query type information from prompt and context."""
        domain_info = {
            'domains': [],
            'query_type': 'general',
            'complexity': 'medium',
            'context_indicators': []
        }
        
        prompt_lower = prompt.lower()
        
        # Domain detection
        domain_keywords = {
            'financial': ['financial', 'money', 'investment', 'stock', 'market', 'trading', 'economy'],
            'technical': ['technical', 'code', 'programming', 'software', 'development', 'algorithm'],
            'medical': ['medical', 'health', 'disease', 'treatment', 'diagnosis', 'patient'],
            'legal': ['legal', 'law', 'regulation', 'compliance', 'contract', 'court'],
            'research': ['research', 'study', 'analysis', 'academic', 'scientific', 'paper'],
            'educational': ['education', 'learning', 'teaching', 'student', 'course', 'curriculum']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                domain_info['domains'].append(domain)
        
        # Query type detection
        if any(word in prompt_lower for word in ['how', 'explain', 'describe']):
            domain_info['query_type'] = 'explanatory'
        elif any(word in prompt_lower for word in ['what', 'when', 'where', 'who']):
            domain_info['query_type'] = 'factual'
        elif any(word in prompt_lower for word in ['should', 'recommend', 'suggest']):
            domain_info['query_type'] = 'advisory'
        elif any(word in prompt_lower for word in ['analyze', 'compare', 'evaluate']):
            domain_info['query_type'] = 'analytical'
        
        # Complexity assessment
        if len(prompt.split()) > 50 or any(word in prompt_lower for word in ['complex', 'detailed', 'comprehensive']):
            domain_info['complexity'] = 'high'
        elif len(prompt.split()) < 20:
            domain_info['complexity'] = 'low'
        
        # Context indicators
        if context:
            if context.get('user_expertise') == 'expert':
                domain_info['context_indicators'].append('expert_user')
            if context.get('requires_sources'):
                domain_info['context_indicators'].append('source_required')
            if context.get('time_sensitive'):
                domain_info['context_indicators'].append('time_sensitive')
        
        return domain_info
    
    def _find_relevant_principles(self, prompt: str, domain_info: Dict[str, Any], 
                                context: Dict[str, Any] = None) -> List[CognitivePrinciple]:
        """Find principles relevant to the prompt and context."""
        relevant_principles = []
        
        try:
            # Get cached principles or fetch fresh ones
            all_principles = self._get_cached_principles()
            
            # Score principles by relevance
            scored_principles = []
            
            for principle in all_principles:
                relevance_score = self._calculate_principle_relevance(
                    principle, prompt, domain_info, context
                )
                
                if relevance_score >= self.semantic_similarity_threshold:
                    scored_principles.append((principle, relevance_score))
            
            # Sort by relevance and confidence
            scored_principles.sort(key=lambda x: (x[1], x[0].confidence_score), reverse=True)
            
            # Select top principles
            relevant_principles = [
                principle for principle, score in scored_principles[:self.max_principles_per_prompt]
            ]
            
            logger.info(f"Found {len(relevant_principles)} relevant principles from {len(all_principles)} total")
            
        except Exception as e:
            logger.error(f"Failed to find relevant principles: {e}")
        
        return relevant_principles
    
    def _get_cached_principles(self) -> List[CognitivePrinciple]:
        """Get principles from cache or fetch fresh ones."""
        current_time = datetime.now()
        
        # Check if cache is valid
        if (self._cache_timestamp and 
            (current_time - self._cache_timestamp).total_seconds() < self._cache_ttl and
            self._principle_cache):
            return self._principle_cache
        
        # Fetch fresh principles
        principles = self.registry.get_active_principles(limit=50)
        
        # Filter by minimum confidence
        filtered_principles = [
            p for p in principles 
            if p.confidence_score >= self.min_principle_confidence
        ]
        
        # Update cache
        self._principle_cache = filtered_principles
        self._cache_timestamp = current_time
        
        logger.info(f"Cached {len(filtered_principles)} high-confidence principles")
        return filtered_principles
    
    def _calculate_principle_relevance(self, principle: CognitivePrinciple, prompt: str,
                                     domain_info: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """Calculate how relevant a principle is to the current prompt."""
        relevance_score = 0.0
        
        # Domain matching
        principle_domains = set(principle.domain_tags)
        prompt_domains = set(domain_info['domains'])
        
        if principle_domains & prompt_domains:  # Intersection
            relevance_score += 0.4
        elif 'general' in principle_domains or not principle_domains:
            relevance_score += 0.1
        
        # Text similarity (simple keyword matching)
        principle_words = set(principle.principle_text.lower().split())
        prompt_words = set(prompt.lower().split())
        
        common_words = principle_words & prompt_words
        if common_words:
            word_similarity = len(common_words) / max(len(principle_words), len(prompt_words))
            relevance_score += word_similarity * 0.3
        
        # Query type matching
        principle_text_lower = principle.principle_text.lower()
        query_type = domain_info['query_type']
        
        query_type_indicators = {
            'explanatory': ['explain', 'describe', 'detail'],
            'factual': ['cite', 'reference', 'source'],
            'advisory': ['recommend', 'suggest', 'prioritize'],
            'analytical': ['analyze', 'compare', 'evaluate']
        }
        
        if query_type in query_type_indicators:
            if any(indicator in principle_text_lower for indicator in query_type_indicators[query_type]):
                relevance_score += 0.2
        
        # Context indicators
        for indicator in domain_info.get('context_indicators', []):
            if indicator == 'source_required' and 'source' in principle_text_lower:
                relevance_score += 0.1
            elif indicator == 'expert_user' and 'specific' in principle_text_lower:
                relevance_score += 0.1
        
        # Principle confidence and usage success
        relevance_score *= principle.confidence_score
        
        return min(1.0, relevance_score)
    
    def _create_augmented_prompt(self, original_prompt: str, 
                               principles: List[CognitivePrinciple]) -> str:
        """Create an augmented prompt with principles injected."""
        if not principles:
            return original_prompt
        
        # Create principle guidance section
        principle_guidance = "**Cognitive Principles to Apply:**\n"
        
        for i, principle in enumerate(principles, 1):
            principle_guidance += f"{i}. {principle.principle_text}\n"
        
        principle_guidance += "\n**Instructions:** Apply the above principles while reasoning through the following query.\n\n"
        
        # Inject principles before the original prompt
        augmented_prompt = principle_guidance + original_prompt
        
        return augmented_prompt
    
    def _calculate_confidence_boost(self, principles: List[CognitivePrinciple]) -> float:
        """Calculate the confidence boost from applied principles."""
        if not principles:
            return 0.0
        
        # Average confidence of applied principles, weighted by usage success
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for principle in principles:
            weight = max(1.0, principle.usage_count * principle.success_rate)
            total_weighted_confidence += principle.confidence_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        avg_confidence = total_weighted_confidence / total_weight
        
        # Scale by number of principles (diminishing returns)
        principle_factor = min(1.0, len(principles) / self.max_principles_per_prompt)
        
        confidence_boost = avg_confidence * principle_factor * 0.2  # Max 20% boost
        
        return round(confidence_boost, 3)
    
    def _track_principle_usage(self, principles: List[CognitivePrinciple], prompt: str):
        """Track principle usage for performance monitoring."""
        try:
            for principle in principles:
                # Update usage count (actual performance will be tracked later)
                self.registry.update_principle_performance(
                    principle.principle_id, 
                    "applied",  # Neutral outcome, actual success tracked later
                    0.0
                )
                
        except Exception as e:
            logger.warning(f"Failed to track principle usage: {e}")
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get statistics about prompt augmentation usage."""
        try:
            # This would be enhanced with actual usage tracking
            cached_count = len(self._principle_cache) if self._principle_cache else 0
            
            return {
                'cached_principles': cached_count,
                'cache_age_seconds': (
                    (datetime.now() - self._cache_timestamp).total_seconds()
                    if self._cache_timestamp else 0
                ),
                'max_principles_per_prompt': self.max_principles_per_prompt,
                'min_principle_confidence': self.min_principle_confidence,
                'semantic_similarity_threshold': self.semantic_similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to get augmentation stats: {e}")
            return {}

# Global prompt augmentation instance
prompt_augmentation = PromptAugmentation()
