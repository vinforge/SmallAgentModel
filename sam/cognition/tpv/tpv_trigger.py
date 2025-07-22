"""
TPV Trigger Module for SAM
Phase 1 - Intelligent TPV Activation

This module determines when to activate TPV monitoring based on
query characteristics and user context.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """Types of TPV triggers."""
    COMPLEXITY_HIGH = "complexity_high"
    UNCERTAINTY_KEYWORDS = "uncertainty_keywords"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    TECHNICAL_DOMAIN = "technical_domain"
    USER_REQUEST = "user_request"
    ALWAYS_ON = "always_on"
    USER_DISABLED = "user_disabled"
    FALLBACK = "fallback"

class QueryIntent(Enum):
    """Types of query intents for TPV triggering."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    PROBLEM_SOLVING = "problem_solving"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"

@dataclass
class UserProfile:
    """User profile for TPV trigger decisions."""
    expertise_level: str = "general"  # beginner, general, expert
    preferred_detail: str = "medium"  # low, medium, high
    tpv_preference: str = "auto"      # always, auto, never
    domain_expertise: List[str] = None
    
    def __post_init__(self):
        if self.domain_expertise is None:
            self.domain_expertise = []

@dataclass
class TriggerResult:
    """Result of TPV trigger evaluation."""
    should_activate: bool
    trigger_type: str
    confidence: float
    reason: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'should_activate': self.should_activate,
            'trigger_type': self.trigger_type,
            'confidence': self.confidence,
            'reason': self.reason,
            'metadata': self.metadata
        }

class TPVTrigger:
    """
    Intelligent trigger system for TPV activation.
    
    Analyzes queries and context to determine when TPV monitoring
    would be most beneficial.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TPV trigger system.
        
        Args:
            config: Configuration for trigger behavior
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'complexity_threshold': 0.7,
            'confidence_threshold': 0.6,
            'enable_keyword_triggers': True,
            'enable_complexity_analysis': True,
            'enable_domain_detection': True,
            'default_activation_rate': 0.3  # 30% of queries by default
        }
        self.default_config.update(self.config)
        
        # Initialize trigger patterns
        self.uncertainty_keywords = self._initialize_uncertainty_keywords()
        self.complexity_patterns = self._initialize_complexity_patterns()
        self.technical_domains = self._initialize_technical_domains()
        
        # Statistics
        self.total_evaluations = 0
        self.activations = 0
        self.trigger_counts = {trigger.value: 0 for trigger in TriggerType}
        
        logger.info("TPVTrigger initialized")
    
    def _initialize_uncertainty_keywords(self) -> List[str]:
        """Initialize keywords that indicate uncertainty or complexity."""
        return [
            # Uncertainty indicators
            "maybe", "perhaps", "possibly", "might", "could be", "uncertain",
            "not sure", "unclear", "ambiguous", "complex", "complicated",
            
            # Reasoning indicators
            "analyze", "compare", "evaluate", "assess", "determine", "conclude",
            "reason", "logic", "because", "therefore", "however", "although",
            
            # Multi-step indicators
            "first", "second", "then", "next", "finally", "step by step",
            "process", "procedure", "method", "approach", "strategy",
            
            # Problem-solving indicators
            "solve", "problem", "issue", "challenge", "difficulty", "solution",
            "answer", "explain", "clarify", "understand", "figure out",

            # Paradox and contradiction indicators
            "paradox", "contradiction", "conflict", "inconsistent", "impossible",
            "logical", "causality", "time travel", "grandfather", "bootstrap",
            "circular", "self-referential", "recursive", "loop"
        ]
    
    def _initialize_complexity_patterns(self) -> List[str]:
        """Initialize regex patterns for complexity detection."""
        return [
            r'\b(how|why|what|when|where)\s+(exactly|specifically|precisely)\b',
            r'\b(explain|describe|analyze)\s+.{20,}\b',  # Long explanatory requests
            r'\b(compare|contrast|differentiate)\b',
            r'\b(pros\s+and\s+cons|advantages\s+and\s+disadvantages)\b',
            r'\b(step\s+by\s+step|detailed\s+process)\b',
            r'\b(multiple|several|various)\s+(ways|methods|approaches)\b',
            r'\?\s*.{10,}\?',  # Multiple questions
            r'\b(if|when|unless|provided\s+that)\b.{15,}',  # Conditional reasoning
        ]
    
    def _initialize_technical_domains(self) -> Dict[str, List[str]]:
        """Initialize technical domain keywords."""
        return {
            'programming': [
                'code', 'programming', 'algorithm', 'function', 'variable',
                'python', 'javascript', 'java', 'c++', 'sql', 'api', 'database'
            ],
            'science': [
                'hypothesis', 'experiment', 'theory', 'research', 'analysis',
                'data', 'statistics', 'correlation', 'causation', 'methodology'
            ],
            'mathematics': [
                'equation', 'formula', 'calculate', 'solve', 'proof', 'theorem',
                'derivative', 'integral', 'matrix', 'probability', 'statistics'
            ],
            'engineering': [
                'design', 'system', 'architecture', 'optimization', 'efficiency',
                'performance', 'specification', 'requirements', 'implementation'
            ],
            'medical': [
                'diagnosis', 'treatment', 'symptoms', 'patient', 'clinical',
                'medical', 'health', 'disease', 'therapy', 'medication'
            ]
        }
    
    def should_activate_tpv(self, 
                           query: str,
                           user_profile: Optional[UserProfile] = None,
                           initial_confidence: float = 0.5,
                           context: Optional[Dict[str, Any]] = None) -> TriggerResult:
        """
        Determine if TPV should be activated for a query.
        
        Args:
            query: The user query
            user_profile: User profile information
            initial_confidence: Initial confidence in the query
            context: Additional context information
            
        Returns:
            TriggerResult with activation decision
        """
        self.total_evaluations += 1
        
        # Default user profile
        if user_profile is None:
            user_profile = UserProfile()
        
        # Check user preference first
        if user_profile.tpv_preference == "never":
            return self._create_result(
                False, TriggerType.USER_DISABLED, 1.0,
                "User has disabled TPV monitoring"
            )
        
        if user_profile.tpv_preference == "always":
            return self._create_result(
                True, TriggerType.ALWAYS_ON, 1.0,
                "User has enabled always-on TPV monitoring"
            )
        
        # Analyze query for activation triggers
        trigger_scores = self._analyze_query(query, user_profile, context)
        
        # Calculate overall activation score
        activation_score = self._calculate_activation_score(trigger_scores)
        
        # Determine activation decision
        should_activate = activation_score >= self.default_config['confidence_threshold']
        
        # Find primary trigger
        primary_trigger = max(trigger_scores.items(), key=lambda x: x[1])
        trigger_type = TriggerType(primary_trigger[0])
        
        # Create result
        result = self._create_result(
            should_activate, trigger_type, activation_score,
            f"Activation score: {activation_score:.3f} (threshold: {self.default_config['confidence_threshold']})",
            {
                'trigger_scores': trigger_scores,
                'user_profile': user_profile.__dict__,
                'initial_confidence': initial_confidence
            }
        )
        
        # Update statistics
        if should_activate:
            self.activations += 1
        self.trigger_counts[trigger_type.value] += 1
        
        return result
    
    def _analyze_query(self, 
                      query: str, 
                      user_profile: UserProfile,
                      context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze query for various trigger conditions.
        
        Args:
            query: The user query
            user_profile: User profile
            context: Additional context
            
        Returns:
            Dictionary of trigger scores
        """
        scores = {}
        query_lower = query.lower()
        
        # 1. Complexity analysis
        if self.default_config['enable_complexity_analysis']:
            scores['complexity_high'] = self._analyze_complexity(query, query_lower)
        
        # 2. Uncertainty keyword detection
        if self.default_config['enable_keyword_triggers']:
            scores['uncertainty_keywords'] = self._analyze_uncertainty_keywords(query_lower)
        
        # 3. Multi-step reasoning detection
        scores['multi_step_reasoning'] = self._analyze_multi_step_reasoning(query_lower)
        
        # 4. Technical domain detection
        if self.default_config['enable_domain_detection']:
            scores['technical_domain'] = self._analyze_technical_domain(query_lower, user_profile)
        
        # 5. User request analysis
        scores['user_request'] = self._analyze_user_request(query_lower, context)
        
        return scores
    
    def _analyze_complexity(self, query: str, query_lower: str) -> float:
        """Analyze query complexity."""
        score = 0.0
        
        # Length-based complexity
        word_count = len(query.split())
        if word_count > 20:
            score += 0.3
        elif word_count > 10:
            score += 0.1
        
        # Pattern-based complexity
        for pattern in self.complexity_patterns:
            if re.search(pattern, query_lower):
                score += 0.2
        
        # Question complexity
        question_count = query.count('?')
        if question_count > 1:
            score += 0.2
        
        return min(1.0, score)
    
    def _analyze_uncertainty_keywords(self, query_lower: str) -> float:
        """Analyze uncertainty keywords in query."""
        keyword_count = 0
        for keyword in self.uncertainty_keywords:
            if keyword in query_lower:
                keyword_count += 1
        
        # Normalize by query length
        word_count = len(query_lower.split())
        keyword_density = keyword_count / max(1, word_count)
        
        return min(1.0, keyword_density * 10)  # Scale up density
    
    def _analyze_multi_step_reasoning(self, query_lower: str) -> float:
        """Analyze indicators of multi-step reasoning."""
        score = 0.0
        
        # Sequential indicators
        sequential_words = ['first', 'second', 'third', 'then', 'next', 'finally', 'last']
        for word in sequential_words:
            if word in query_lower:
                score += 0.15
        
        # Process indicators
        process_words = ['step', 'process', 'procedure', 'method', 'approach']
        for word in process_words:
            if word in query_lower:
                score += 0.2
        
        # Logical connectors
        logical_words = ['because', 'therefore', 'however', 'although', 'since', 'while']
        for word in logical_words:
            if word in query_lower:
                score += 0.1
        
        return min(1.0, score)
    
    def _analyze_technical_domain(self, query_lower: str, user_profile: UserProfile) -> float:
        """Analyze technical domain indicators."""
        domain_scores = {}
        
        for domain, keywords in self.technical_domains.items():
            domain_score = 0.0
            for keyword in keywords:
                if keyword in query_lower:
                    domain_score += 1.0
            
            # Normalize by keyword count
            domain_scores[domain] = domain_score / len(keywords)
        
        # Get maximum domain score
        max_domain_score = max(domain_scores.values()) if domain_scores else 0.0
        
        # Boost score if user has expertise in detected domain
        detected_domains = [domain for domain, score in domain_scores.items() if score > 0.1]
        if any(domain in user_profile.domain_expertise for domain in detected_domains):
            max_domain_score *= 1.2
        
        return min(1.0, max_domain_score)
    
    def _analyze_user_request(self, query_lower: str, context: Optional[Dict[str, Any]]) -> float:
        """Analyze explicit user requests for detailed reasoning."""
        score = 0.0
        
        # Explicit reasoning requests
        reasoning_requests = [
            'explain your reasoning', 'show your work', 'think step by step',
            'walk me through', 'break it down', 'detailed explanation'
        ]
        
        for request in reasoning_requests:
            if request in query_lower:
                score += 0.5
        
        # Context-based requests
        if context and context.get('request_detailed_reasoning'):
            score += 0.4
        
        return min(1.0, score)
    
    def _calculate_activation_score(self, trigger_scores: Dict[str, float]) -> float:
        """Calculate overall activation score from individual trigger scores."""
        if not trigger_scores:
            return 0.0
        
        # Weighted combination of trigger scores
        weights = {
            'complexity_high': 0.3,
            'uncertainty_keywords': 0.2,
            'multi_step_reasoning': 0.25,
            'technical_domain': 0.15,
            'user_request': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for trigger, score in trigger_scores.items():
            weight = weights.get(trigger, 0.1)  # Default weight for unknown triggers
            weighted_score += score * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_score /= total_weight
        
        return min(1.0, weighted_score)
    
    def _create_result(self, 
                      should_activate: bool,
                      trigger_type: TriggerType,
                      confidence: float,
                      reason: str,
                      metadata: Optional[Dict[str, Any]] = None) -> TriggerResult:
        """Create a TriggerResult object."""
        return TriggerResult(
            should_activate=should_activate,
            trigger_type=trigger_type.value,
            confidence=confidence,
            reason=reason,
            metadata=metadata or {}
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get trigger system status and statistics."""
        activation_rate = self.activations / max(1, self.total_evaluations)
        
        return {
            'total_evaluations': self.total_evaluations,
            'activations': self.activations,
            'activation_rate': activation_rate,
            'trigger_counts': self.trigger_counts,
            'config': self.default_config
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update trigger configuration."""
        self.default_config.update(new_config)
        logger.info(f"TPVTrigger configuration updated: {new_config}")
    
    def reset_statistics(self):
        """Reset trigger statistics."""
        self.total_evaluations = 0
        self.activations = 0
        self.trigger_counts = {trigger.value: 0 for trigger in TriggerType}
        logger.info("TPVTrigger statistics reset")
