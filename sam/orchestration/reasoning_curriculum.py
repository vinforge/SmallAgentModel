"""
PINN-Inspired Reasoning Curriculum System

Implements curriculum learning principles for SAM's dynamic planning,
inspired by Physics-Informed Neural Networks' staged training approaches.
Progressively increases reasoning complexity based on query difficulty.

Key Features:
- Staged reasoning progression from simple to complex
- Adaptive skill selection based on curriculum level
- Dynamic complexity assessment and adjustment
- Performance-based curriculum advancement

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CurriculumLevel(Enum):
    """Curriculum levels for reasoning complexity."""
    FOUNDATION = "foundation"      # Level 1: Basic retrieval + response
    INTERMEDIATE = "intermediate"  # Level 2: Add conflict detection
    ADVANCED = "advanced"         # Level 3: Add analysis skills
    EXPERT = "expert"             # Level 4: Full reasoning pipeline
    RESEARCH = "research"         # Level 5: Deep analysis + tools

class QueryComplexity(Enum):
    """Query complexity levels."""
    TRIVIAL = "trivial"           # Simple factual queries
    SIMPLE = "simple"             # Basic questions with clear answers
    MODERATE = "moderate"         # Multi-step reasoning required
    COMPLEX = "complex"           # Deep analysis and synthesis
    RESEARCH = "research"         # Requires extensive investigation

@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    level: CurriculumLevel
    required_skills: List[str]
    optional_skills: List[str]
    max_skills: int
    complexity_threshold: float
    success_threshold: float
    description: str

@dataclass
class ReasoningPlan:
    """Enhanced reasoning plan with curriculum information."""
    skills: List[str]
    curriculum_level: CurriculumLevel
    complexity_score: float
    confidence_threshold: float
    reasoning_strategy: str
    estimated_effort: float

class ReasoningCurriculum:
    """
    PINN-inspired curriculum system for progressive reasoning development.
    
    Implements staged learning where SAM starts with simple reasoning
    patterns and progressively tackles more complex queries as it
    demonstrates competence at each level.
    """
    
    def __init__(
        self,
        enable_adaptive_progression: bool = True,
        performance_window: int = 10,
        advancement_threshold: float = 0.8
    ):
        """
        Initialize the reasoning curriculum.
        
        Args:
            enable_adaptive_progression: Whether to adapt curriculum based on performance
            performance_window: Number of recent queries to consider for advancement
            advancement_threshold: Success rate needed to advance curriculum level
        """
        self.enable_adaptive_progression = enable_adaptive_progression
        self.performance_window = performance_window
        self.advancement_threshold = advancement_threshold
        
        # Performance tracking for curriculum advancement
        self.performance_history: List[Dict[str, Any]] = []
        self.current_level = CurriculumLevel.FOUNDATION
        
        # Define curriculum stages
        self.curriculum_stages = self._initialize_curriculum_stages()
        
        # Query complexity patterns
        self.complexity_patterns = self._initialize_complexity_patterns()
        
        logger.info(f"ReasoningCurriculum initialized at {self.current_level.value} level")
    
    def _initialize_curriculum_stages(self) -> Dict[CurriculumLevel, CurriculumStage]:
        """Initialize the curriculum stages with skill progressions."""
        return {
            CurriculumLevel.FOUNDATION: CurriculumStage(
                level=CurriculumLevel.FOUNDATION,
                required_skills=["MemoryRetrievalSkill", "ResponseGenerationSkill"],
                optional_skills=[],
                max_skills=2,
                complexity_threshold=0.3,
                success_threshold=0.8,
                description="Basic retrieval and response generation"
            ),
            
            CurriculumLevel.INTERMEDIATE: CurriculumStage(
                level=CurriculumLevel.INTERMEDIATE,
                required_skills=["MemoryRetrievalSkill", "ConflictDetectorSkill", "ResponseGenerationSkill"],
                optional_skills=["ContentVettingSkill"],
                max_skills=4,
                complexity_threshold=0.5,
                success_threshold=0.75,
                description="Add conflict detection and basic vetting"
            ),
            
            CurriculumLevel.ADVANCED: CurriculumStage(
                level=CurriculumLevel.ADVANCED,
                required_skills=["MemoryRetrievalSkill", "ConflictDetectorSkill", "ImplicitKnowledgeSkill", "ResponseGenerationSkill"],
                optional_skills=["ContentVettingSkill", "CalculatorTool"],
                max_skills=6,
                complexity_threshold=0.7,
                success_threshold=0.7,
                description="Add implicit knowledge reasoning and tools"
            ),
            
            CurriculumLevel.EXPERT: CurriculumStage(
                level=CurriculumLevel.EXPERT,
                required_skills=["MemoryRetrievalSkill", "ConflictDetectorSkill", "ImplicitKnowledgeSkill", "ContentVettingSkill", "ResponseGenerationSkill"],
                optional_skills=["CalculatorTool", "AgentZeroWebBrowserTool"],
                max_skills=8,
                complexity_threshold=0.85,
                success_threshold=0.65,
                description="Full reasoning pipeline with comprehensive analysis"
            ),
            
            CurriculumLevel.RESEARCH: CurriculumStage(
                level=CurriculumLevel.RESEARCH,
                required_skills=["MemoryRetrievalSkill", "ConflictDetectorSkill", "ImplicitKnowledgeSkill", "ContentVettingSkill", "AgentZeroWebBrowserTool", "ResponseGenerationSkill"],
                optional_skills=["CalculatorTool", "MEMOIR_EditSkill"],
                max_skills=10,
                complexity_threshold=1.0,
                success_threshold=0.6,
                description="Research-level analysis with web tools and memory editing"
            )
        }
    
    def _initialize_complexity_patterns(self) -> Dict[QueryComplexity, Dict[str, Any]]:
        """Initialize patterns for query complexity assessment."""
        return {
            QueryComplexity.TRIVIAL: {
                "keywords": ["what is", "who is", "when is", "where is", "define"],
                "max_words": 8,
                "complexity_score": 0.1,
                "recommended_level": CurriculumLevel.FOUNDATION
            },
            
            QueryComplexity.SIMPLE: {
                "keywords": ["how to", "explain", "describe", "list", "show me"],
                "max_words": 15,
                "complexity_score": 0.3,
                "recommended_level": CurriculumLevel.FOUNDATION
            },
            
            QueryComplexity.MODERATE: {
                "keywords": ["compare", "difference", "relationship", "why does", "how does"],
                "max_words": 25,
                "complexity_score": 0.5,
                "recommended_level": CurriculumLevel.INTERMEDIATE
            },
            
            QueryComplexity.COMPLEX: {
                "keywords": ["analyze", "evaluate", "synthesize", "implications", "pros and cons"],
                "max_words": 40,
                "complexity_score": 0.7,
                "recommended_level": CurriculumLevel.ADVANCED
            },
            
            QueryComplexity.RESEARCH: {
                "keywords": ["research", "investigate", "comprehensive analysis", "systematic review"],
                "max_words": 100,
                "complexity_score": 0.9,
                "recommended_level": CurriculumLevel.RESEARCH
            }
        }
    
    def assess_query_complexity(self, query: str) -> Tuple[QueryComplexity, float]:
        """
        Assess the complexity of a query.
        
        Args:
            query: The input query to assess
            
        Returns:
            Tuple of (complexity_level, complexity_score)
        """
        if not query:
            return QueryComplexity.SIMPLE, 0.3
        
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Score based on patterns
        complexity_scores = []
        
        for complexity, patterns in self.complexity_patterns.items():
            score = 0.0
            
            # Check for keywords
            keyword_matches = sum(1 for keyword in patterns["keywords"] if keyword in query_lower)
            if keyword_matches > 0:
                score += patterns["complexity_score"]
            
            # Adjust based on word count
            if word_count <= patterns["max_words"]:
                score += 0.1
            else:
                score -= 0.1
            
            complexity_scores.append((complexity, max(0.0, score)))
        
        # Find the complexity with highest score
        best_complexity, best_score = max(complexity_scores, key=lambda x: x[1])
        
        # If no clear match, use word count as fallback
        if best_score == 0.0:
            if word_count <= 8:
                return QueryComplexity.SIMPLE, 0.3
            elif word_count <= 20:
                return QueryComplexity.MODERATE, 0.5
            else:
                return QueryComplexity.COMPLEX, 0.7
        
        return best_complexity, best_score
    
    def generate_reasoning_plan(
        self,
        query: str,
        available_skills: List[str],
        force_level: Optional[CurriculumLevel] = None
    ) -> ReasoningPlan:
        """
        Generate a reasoning plan based on curriculum level and query complexity.
        
        Args:
            query: The input query
            available_skills: List of available skill names
            force_level: Force a specific curriculum level (for testing)
            
        Returns:
            Reasoning plan with curriculum-informed skill selection
        """
        # Assess query complexity
        query_complexity, complexity_score = self.assess_query_complexity(query)
        
        # Determine curriculum level to use
        target_level = force_level or self._determine_target_level(query_complexity, complexity_score)
        
        # Get curriculum stage configuration
        stage = self.curriculum_stages[target_level]
        
        # Build skill plan
        selected_skills = []
        
        # Add required skills that are available
        for skill in stage.required_skills:
            if skill in available_skills:
                selected_skills.append(skill)
        
        # Add optional skills based on complexity and availability
        remaining_slots = stage.max_skills - len(selected_skills)
        for skill in stage.optional_skills:
            if skill in available_skills and len(selected_skills) < stage.max_skills:
                # Add skill if complexity warrants it
                if complexity_score >= stage.complexity_threshold * 0.8:
                    selected_skills.append(skill)
                    remaining_slots -= 1
        
        # Determine reasoning strategy
        strategy = self._determine_reasoning_strategy(target_level, query_complexity)
        
        # Estimate effort based on curriculum level
        effort_multipliers = {
            CurriculumLevel.FOUNDATION: 0.5,
            CurriculumLevel.INTERMEDIATE: 0.7,
            CurriculumLevel.ADVANCED: 1.0,
            CurriculumLevel.EXPERT: 1.3,
            CurriculumLevel.RESEARCH: 1.8
        }
        estimated_effort = len(selected_skills) * effort_multipliers[target_level]
        
        plan = ReasoningPlan(
            skills=selected_skills,
            curriculum_level=target_level,
            complexity_score=complexity_score,
            confidence_threshold=stage.success_threshold,
            reasoning_strategy=strategy,
            estimated_effort=estimated_effort
        )
        
        logger.info(f"Generated {target_level.value} level plan with {len(selected_skills)} skills for {query_complexity.value} query")
        
        return plan
    
    def _determine_target_level(self, query_complexity: QueryComplexity, complexity_score: float) -> CurriculumLevel:
        """Determine the target curriculum level for a query."""
        # Get recommended level from complexity patterns
        recommended_level = self.complexity_patterns[query_complexity]["recommended_level"]
        
        # Don't exceed current curriculum level unless forced
        if self.enable_adaptive_progression:
            # Allow one level above current for challenging queries
            max_allowed_level = self._get_next_level(self.current_level) or self.current_level
        else:
            max_allowed_level = CurriculumLevel.RESEARCH  # No restrictions
        
        # Choose the minimum of recommended and allowed
        level_order = [CurriculumLevel.FOUNDATION, CurriculumLevel.INTERMEDIATE, 
                      CurriculumLevel.ADVANCED, CurriculumLevel.EXPERT, CurriculumLevel.RESEARCH]
        
        recommended_index = level_order.index(recommended_level)
        max_allowed_index = level_order.index(max_allowed_level)
        
        target_index = min(recommended_index, max_allowed_index)
        return level_order[target_index]
    
    def _determine_reasoning_strategy(self, level: CurriculumLevel, complexity: QueryComplexity) -> str:
        """Determine the reasoning strategy for a curriculum level and complexity."""
        strategies = {
            CurriculumLevel.FOUNDATION: "direct_retrieval",
            CurriculumLevel.INTERMEDIATE: "conflict_aware_retrieval",
            CurriculumLevel.ADVANCED: "analytical_reasoning",
            CurriculumLevel.EXPERT: "comprehensive_analysis",
            CurriculumLevel.RESEARCH: "systematic_investigation"
        }
        
        base_strategy = strategies[level]
        
        # Modify strategy based on complexity
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.RESEARCH]:
            if "analysis" not in base_strategy:
                base_strategy = f"enhanced_{base_strategy}"
        
        return base_strategy
    
    def record_performance(
        self,
        plan: ReasoningPlan,
        success: bool,
        confidence: float,
        execution_time: float
    ) -> None:
        """
        Record performance for curriculum advancement.
        
        Args:
            plan: The reasoning plan that was executed
            success: Whether the execution was successful
            confidence: Final confidence score
            execution_time: Time taken for execution
        """
        performance_record = {
            "timestamp": time.time(),
            "curriculum_level": plan.curriculum_level.value,
            "complexity_score": plan.complexity_score,
            "success": success,
            "confidence": confidence,
            "execution_time": execution_time,
            "num_skills": len(plan.skills)
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window * 2:
            self.performance_history = self.performance_history[-self.performance_window:]
        
        # Check for curriculum advancement
        if self.enable_adaptive_progression:
            self._check_curriculum_advancement()
    
    def _check_curriculum_advancement(self) -> None:
        """Check if curriculum should advance based on recent performance."""
        if len(self.performance_history) < self.performance_window:
            return
        
        # Get recent performance at current level
        recent_performance = [
            record for record in self.performance_history[-self.performance_window:]
            if record["curriculum_level"] == self.current_level.value
        ]
        
        if len(recent_performance) < self.performance_window // 2:
            return  # Not enough data at current level
        
        # Calculate success rate
        success_rate = sum(1 for record in recent_performance if record["success"]) / len(recent_performance)
        avg_confidence = sum(record["confidence"] for record in recent_performance) / len(recent_performance)
        
        # Check advancement criteria
        stage = self.curriculum_stages[self.current_level]
        if (success_rate >= self.advancement_threshold and 
            avg_confidence >= stage.success_threshold):
            
            next_level = self._get_next_level(self.current_level)
            if next_level:
                logger.info(f"Advancing curriculum from {self.current_level.value} to {next_level.value}")
                logger.info(f"Performance: {success_rate:.2f} success rate, {avg_confidence:.2f} avg confidence")
                self.current_level = next_level
    
    def _get_next_level(self, current: CurriculumLevel) -> Optional[CurriculumLevel]:
        """Get the next curriculum level."""
        level_progression = {
            CurriculumLevel.FOUNDATION: CurriculumLevel.INTERMEDIATE,
            CurriculumLevel.INTERMEDIATE: CurriculumLevel.ADVANCED,
            CurriculumLevel.ADVANCED: CurriculumLevel.EXPERT,
            CurriculumLevel.EXPERT: CurriculumLevel.RESEARCH,
            CurriculumLevel.RESEARCH: None  # Max level
        }
        return level_progression.get(current)
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum status and statistics."""
        recent_performance = self.performance_history[-self.performance_window:] if self.performance_history else []
        
        if recent_performance:
            success_rate = sum(1 for record in recent_performance if record["success"]) / len(recent_performance)
            avg_confidence = sum(record["confidence"] for record in recent_performance) / len(recent_performance)
            avg_execution_time = sum(record["execution_time"] for record in recent_performance) / len(recent_performance)
        else:
            success_rate = avg_confidence = avg_execution_time = 0.0
        
        return {
            "current_level": self.current_level.value,
            "total_queries": len(self.performance_history),
            "recent_success_rate": success_rate,
            "recent_avg_confidence": avg_confidence,
            "recent_avg_execution_time": avg_execution_time,
            "advancement_threshold": self.advancement_threshold,
            "adaptive_progression": self.enable_adaptive_progression
        }
