"""
Dynamic Planner for SAM Orchestration Framework
===============================================

Implements intelligent plan generation using LLM-as-a-Planner approach.
Generates custom execution plans based on query analysis and available skills.
"""

import json
import logging
import hashlib
import time
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta

from .uif import SAM_UIF
from .skills.base import BaseSkillModule
from .config import get_sof_config
from .reasoning_curriculum import ReasoningCurriculum, CurriculumLevel

logger = logging.getLogger(__name__)


@dataclass
class PlanCacheEntry:
    """Cache entry for generated plans."""
    plan: List[str]
    confidence: float
    created_at: datetime
    usage_count: int
    query_hash: str
    skill_context: str


@dataclass
class PlanGenerationResult:
    """Result of plan generation."""
    plan: List[str]
    confidence: float
    reasoning: str
    cache_hit: bool
    generation_time: float
    fallback_used: bool


class DynamicPlanner:
    """
    Enhanced graph-aware planner for SAM's Cognitive Memory Core.

    Features:
    - Intelligent plan generation based on query analysis
    - Graph-aware skill selection and retrieval mode optimization
    - Plan caching to reduce LLM overhead
    - Fallback to default plans when generation fails
    - Context-aware skill selection with memory mode intelligence
    - Performance optimization with dual-mode retrieval
    - Integration with Cognitive Memory Core (Phase B)
    """
    
    def __init__(self, enable_curriculum: bool = True, goal_stack=None):
        self.logger = logging.getLogger(f"{__name__}.DynamicPlanner")
        self._config = get_sof_config()
        self._registered_skills: Dict[str, BaseSkillModule] = {}
        self._plan_cache: Dict[str, PlanCacheEntry] = {}
        self._llm_model = None
        self._graph_database = None
        self._goal_stack = goal_stack  # Phase B: Goal-informed planning

        # PINN-inspired reasoning curriculum
        self._curriculum = ReasoningCurriculum() if enable_curriculum else None

        self._initialize_llm()
        self._initialize_graph_awareness()

        self.logger.info(f"DynamicPlanner initialized (curriculum: {enable_curriculum}, goal_aware: {goal_stack is not None})")
    
    def _initialize_llm(self) -> None:
        """Initialize the language model for plan generation."""
        try:
            # Try to import SAM's LLM configuration
            from config.llm_config import get_llm_model
            self._llm_model = get_llm_model()
            self.logger.info("LLM model initialized for plan generation")
            
        except ImportError:
            self.logger.warning("Could not import SAM LLM config, plan generation will use fallbacks")
            self._llm_model = None
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {e}")
            self._llm_model = None

    def _initialize_graph_awareness(self) -> None:
        """Initialize graph database awareness for enhanced planning."""
        try:
            from sam.memory.graph.graph_database import get_graph_database

            self._graph_database = get_graph_database()
            self.logger.info("Graph database awareness initialized for planning")

        except ImportError:
            self.logger.info("Graph database not available for planning")
            self._graph_database = None
        except Exception as e:
            self.logger.warning(f"Error initializing graph awareness: {e}")
            self._graph_database = None
    
    def register_skill(self, skill: BaseSkillModule) -> None:
        """
        Register a skill for plan generation.
        
        Args:
            skill: Skill to register
        """
        self._registered_skills[skill.skill_name] = skill
        self.logger.debug(f"Registered skill for planning: {skill.skill_name}")
    
    def register_skills(self, skills: List[BaseSkillModule]) -> None:
        """
        Register multiple skills for plan generation.
        
        Args:
            skills: List of skills to register
        """
        for skill in skills:
            self.register_skill(skill)
    
    def create_plan(self, uif: SAM_UIF, mode: str = "user_focused") -> PlanGenerationResult:
        """
        Create a custom execution plan for the given query.

        Args:
            uif: Universal Interface Format with query and context
            mode: Planning mode - "user_focused", "goal_informed", or "goal_focused"

        Returns:
            Plan generation result with plan and metadata
        """
        start_time = time.time()

        # Phase B: Handle goal-informed planning modes
        background_goal = None
        if mode in ["goal_informed", "goal_focused"] and self._goal_stack:
            background_goal = self._get_background_goal()
            if background_goal:
                self.logger.info(f"Planning with background goal: {background_goal.description[:50]}...")
                uif.add_log_entry(f"Background goal: {background_goal.goal_id}")

        # For goal_focused mode with empty query, focus entirely on the goal
        if mode == "goal_focused" and not uif.input_query.strip() and background_goal:
            uif.input_query = f"Address autonomous goal: {background_goal.description}"
            uif.add_log_entry(f"Planning focused on autonomous goal: {background_goal.goal_id}")

        self.logger.info(f"Creating plan for query: {uif.input_query[:100]}... (mode: {mode})")

        try:
            # Check for Test-Time Training (TTT) opportunity first
            ttt_plan = self._check_for_ttt_opportunity(uif)
            if ttt_plan:
                generation_time = time.time() - start_time
                return PlanGenerationResult(
                    plan=ttt_plan,
                    confidence=0.9,
                    reasoning="Detected few-shot reasoning task - using Test-Time Training adaptation",
                    cache_hit=False,
                    generation_time=generation_time,
                    fallback_used=False
                )

            # Check cache first
            if self._config.enable_plan_caching:
                cached_result = self._check_plan_cache(uif)
                if cached_result:
                    generation_time = time.time() - start_time
                    return PlanGenerationResult(
                        plan=cached_result.plan,
                        confidence=cached_result.confidence,
                        reasoning=f"Retrieved from cache (used {cached_result.usage_count} times)",
                        cache_hit=True,
                        generation_time=generation_time,
                        fallback_used=False
                    )

            # Generate new plan - prioritize LLM for enhanced tool selection
            if self._llm_model:
                result = self._generate_llm_plan(uif, background_goal)
            elif self._curriculum:
                result = self._generate_curriculum_plan(uif, background_goal)
            else:
                result = self._generate_fallback_plan(uif, background_goal)

            # Phase 5C: Apply SELF-REFLECT policy to enhance plan
            result = self._apply_self_reflect_policy(result, uif)

            # Cache the result if successful
            if self._config.enable_plan_caching and result.plan and result.confidence > 0.5:
                self._cache_plan(uif, result)

            generation_time = time.time() - start_time
            result.generation_time = generation_time

            self.logger.info(f"Plan created: {result.plan} (confidence: {result.confidence:.2f})")

            return result
            
        except Exception as e:
            self.logger.exception("Error during plan generation")
            generation_time = time.time() - start_time
            
            # Return fallback plan
            fallback_plan = self._get_default_fallback_plan()
            return PlanGenerationResult(
                plan=fallback_plan,
                confidence=0.3,
                reasoning=f"Error during generation: {str(e)}",
                cache_hit=False,
                generation_time=generation_time,
                fallback_used=True
            )
    
    def _check_plan_cache(self, uif: SAM_UIF) -> Optional[PlanCacheEntry]:
        """
        Check if a cached plan exists for the query.
        
        Returns:
            Cached plan entry if found, None otherwise
        """
        query_hash = self._generate_query_hash(uif)
        
        if query_hash in self._plan_cache:
            entry = self._plan_cache[query_hash]
            
            # Check if cache entry is still valid
            if self._is_cache_entry_valid(entry):
                entry.usage_count += 1
                self.logger.debug(f"Cache hit for query hash: {query_hash}")
                return entry
            else:
                # Remove expired entry
                del self._plan_cache[query_hash]
                self.logger.debug(f"Removed expired cache entry: {query_hash}")
        
        return None

    def _get_background_goal(self):
        """
        Get the top priority background goal for goal-informed planning.

        Returns:
            Top priority goal or None if no goals available
        """
        if not self._goal_stack:
            return None

        try:
            top_goals = self._goal_stack.get_top_priority_goals(limit=1, status="pending")
            return top_goals[0] if top_goals else None
        except Exception as e:
            self.logger.warning(f"Failed to retrieve background goal: {e}")
            return None

    def _generate_curriculum_plan(self, uif: SAM_UIF, background_goal=None) -> PlanGenerationResult:
        """
        Generate plan using PINN-inspired curriculum learning.

        Returns:
            Plan generation result with curriculum-informed skill selection
        """
        try:
            # Generate reasoning plan using curriculum
            available_skills = list(self._registered_skills.keys())
            reasoning_plan = self._curriculum.generate_reasoning_plan(
                query=uif.input_query,
                available_skills=available_skills
            )

            # Store curriculum information in UIF for coordinator
            # Note: These are dynamic attributes, not part of the UIF model
            setattr(uif, 'curriculum_level', reasoning_plan.curriculum_level.value)
            setattr(uif, 'complexity_score', reasoning_plan.complexity_score)
            setattr(uif, 'reasoning_strategy', reasoning_plan.reasoning_strategy)
            setattr(uif, 'estimated_effort', reasoning_plan.estimated_effort)

            return PlanGenerationResult(
                plan=reasoning_plan.skills,
                confidence=reasoning_plan.confidence_threshold,
                reasoning=f"Curriculum {reasoning_plan.curriculum_level.value}: {reasoning_plan.reasoning_strategy}",
                cache_hit=False,
                generation_time=0.0,  # Will be set by caller
                fallback_used=False
            )

        except Exception as e:
            self.logger.error(f"Curriculum plan generation failed: {e}")
            # Fall back to standard plan generation
            if self._llm_model:
                return self._generate_llm_plan(uif, background_goal)
            else:
                return self._generate_fallback_plan(uif, background_goal)

    def _generate_llm_plan(self, uif: SAM_UIF, background_goal=None) -> PlanGenerationResult:
        """
        Generate plan using LLM-as-a-Planner approach.
        
        Returns:
            Plan generation result
        """
        try:
            # Create planning prompt
            prompt = self._create_planning_prompt(uif, background_goal)

            # Generate plan using LLM
            response = self._llm_model.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent planning
                max_tokens=500
            )
            
            # Parse the response
            plan_data = self._parse_llm_response(response)
            
            if plan_data and plan_data.get("plan"):
                return PlanGenerationResult(
                    plan=plan_data["plan"],
                    confidence=plan_data.get("confidence", 0.8),
                    reasoning=plan_data.get("reasoning", "Generated by LLM planner"),
                    cache_hit=False,
                    generation_time=0.0,  # Will be set by caller
                    fallback_used=False
                )
            else:
                # LLM response was invalid, use fallback
                return self._generate_fallback_plan(uif, background_goal)

        except Exception as e:
            self.logger.error(f"LLM plan generation failed: {e}")
            return self._generate_fallback_plan(uif, background_goal)

    def _create_planning_prompt(self, uif: SAM_UIF, background_goal=None) -> str:
        """
        Create a specialized prompt for plan generation.

        Args:
            uif: Universal Interface Format
            background_goal: Optional background goal for goal-informed planning

        Returns:
            Formatted planning prompt
        """
        available_skills = list(self._registered_skills.keys())
        skill_descriptions = {}
        
        for skill_name, skill in self._registered_skills.items():
            skill_descriptions[skill_name] = {
                "description": skill.skill_description,
                "inputs": skill.required_inputs,
                "outputs": skill.output_keys,
                "category": skill.skill_category
            }
        
        prompt_parts = [
            "You are the planning engine for the SAM AI system. Your task is to analyze the user's query and generate a JSON plan specifying the precise skills needed to answer the query, in the correct execution order.",
            "",
            f"User Query: {uif.input_query}",
            ""
        ]

        # Phase B: Add background goal context for goal-informed planning
        if background_goal:
            prompt_parts.extend([
                f"High-Priority Background Goal: {background_goal.description}",
                f"Goal Source: {background_goal.source_skill}",
                f"Goal Priority: {background_goal.priority:.2f}",
                "",
                "Instructions: You must intelligently decide if you can safely and efficiently integrate steps to address the background goal within the same plan. If the user query is empty, you must generate a plan to address the background goal directly. Prioritize the user's query unless the background goal is directly related or critically urgent.",
                ""
            ])

        prompt_parts.append("Available Skills:")
        
        for skill_name, info in skill_descriptions.items():
            prompt_parts.append(f"- {skill_name}: {info['description']}")
            prompt_parts.append(f"  Inputs: {info['inputs']}")
            prompt_parts.append(f"  Outputs: {info['outputs']}")
            prompt_parts.append(f"  Category: {info['category']}")
            prompt_parts.append("")
        
        # Add context if available
        if uif.user_context:
            prompt_parts.extend([
                "User Context:",
                str(uif.user_context),
                ""
            ])
        
        prompt_parts.extend([
            "TOOL SELECTION GUIDE:",
            "Choose the RIGHT tool for the specific type of query:",
            "",
            "📊 FinancialDataTool - Use for SPECIFIC factual data lookups:",
            "  • Market capitalization, stock prices, company valuations",
            "  • Financial metrics, revenue, earnings, financial data",
            "  • Specific numerical facts about companies or markets",
            "  • Example: 'What is NVIDIA's market capitalization?' → FinancialDataTool",
            "",
            "📰 NewsApiTool - Use for CURRENT news and recent events:",
            "  • Breaking news, latest developments, recent headlines",
            "  • Current events, news updates, what's happening now",
            "  • Recent articles about topics or companies",
            "  • Example: 'Latest news about NVIDIA' → NewsApiTool",
            "",
            "🌐 AgentZeroWebBrowserTool - Use for GENERAL web searches:",
            "  • Broad information gathering, research topics",
            "  • When you need comprehensive web content",
            "  • General 'search the web' requests without specific data needs",
            "  • Example: 'Search the web for information about AI' → AgentZeroWebBrowserTool",
            "",
            "🧮 CalculatorTool - Use for MATHEMATICAL computations:",
            "  • Arithmetic operations, calculations, mathematical expressions",
            "  • When you need to compute values from other tool outputs",
            "  • Example: 'Calculate 1000 + 45 - 56' → CalculatorTool",
            "",
            "Instructions:",
            "1. Analyze the query to understand what information and processing is needed",
            "2. Select the minimum necessary skills to fulfill the request using the Tool Selection Guide above",
            "3. Order skills so that dependencies are satisfied (outputs of earlier skills feed inputs of later skills)",
            "4. Always include ResponseGenerationSkill as the final step to create the user response",
            "5. Consider including MemoryRetrievalSkill if the query might benefit from stored knowledge",
            "6. Include ConflictDetectorSkill if multiple information sources might conflict",
            "7. Include ImplicitKnowledgeSkill for multi-hop questions that require connecting disparate pieces of information",
            "",
            "Response Format (JSON only):",
            "{",
            '  "plan": ["SkillName1", "SkillName2", "SkillName3"],',
            '  "reasoning": "Brief explanation of why these skills were chosen and ordered this way",',
            '  "confidence": 0.85',
            "}",
            "",
            "Your JSON response:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response to extract plan data.
        
        Returns:
            Parsed plan data or None if invalid
        """
        try:
            # Clean the response
            response = response.strip()
            
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                plan_data = json.loads(json_str)
                
                # Validate the plan
                if self._validate_generated_plan(plan_data):
                    return plan_data
            
            return None
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return None
    
    def _validate_generated_plan(self, plan_data: Dict[str, Any]) -> bool:
        """
        Validate that the generated plan is valid.
        
        Returns:
            True if plan is valid, False otherwise
        """
        if not isinstance(plan_data, dict):
            return False
        
        if "plan" not in plan_data or not isinstance(plan_data["plan"], list):
            return False
        
        plan = plan_data["plan"]
        
        # Check that all skills exist
        for skill_name in plan:
            if skill_name not in self._registered_skills:
                self.logger.warning(f"Generated plan contains unknown skill: {skill_name}")
                return False
        
        # Check plan length
        if len(plan) > self._config.max_plan_length:
            self.logger.warning(f"Generated plan too long: {len(plan)} > {self._config.max_plan_length}")
            return False
        
        return True
    
    def _generate_fallback_plan(self, uif: SAM_UIF, background_goal=None) -> PlanGenerationResult:
        """
        Generate an enhanced graph-aware fallback plan using rule-based logic.

        Args:
            uif: Universal Interface Format
            background_goal: Optional background goal for goal-informed planning

        Returns:
            Fallback plan generation result
        """
        plan = []
        reasoning_parts = []

        # Analyze query for plan generation
        query_lower = uif.input_query.lower()

        # Phase B: Consider background goal in fallback planning
        if background_goal:
            goal_lower = background_goal.description.lower()
            # Merge goal keywords with query for better skill selection
            query_lower = f"{query_lower} {goal_lower}"
            reasoning_parts.append(f"Integrated background goal: {background_goal.description[:30]}...")

        # Enhanced tool selection logic (matching the Tool Selection Guide)

        # Guard: avoid financial tool for document summarization/analysis queries
        doc_summarization_context = any(t in query_lower for t in [
            "summarize", "summary of", ".pdf", "document", "paper", "report"
        ])

        # Financial data queries - use FinancialDataTool (stricter triggers)
        if (not doc_summarization_context) and any(term in query_lower for term in [
            "market cap", "market capitalization", "stock price", "share price",
            "financial data", "revenue", "earnings", "valuation", "worth",
            "cost", "price", "trading", "current price",
            "stock", "shares", "equity", "investment", "finance", "financial"
        ]):
            if "FinancialDataTool" in self._registered_skills:
                plan.append("FinancialDataTool")
                reasoning_parts.append("Added FinancialDataTool for financial data lookup (strict match)")

        # News queries - use NewsApiTool
        elif any(term in query_lower for term in [
            "news", "breaking", "latest", "recent", "current events",
            "headlines", "updates", "developments", "happening"
        ]):
            if "NewsApiTool" in self._registered_skills:
                plan.append("NewsApiTool")
                reasoning_parts.append("Added NewsApiTool for current news")

        # General web search - use AgentZeroWebBrowserTool
        elif ("search" in query_lower or "find" in query_lower or "look up" in query_lower):
            if "AgentZeroWebBrowserTool" in self._registered_skills:
                plan.append("AgentZeroWebBrowserTool")
                reasoning_parts.append("Added web search for general information")

        # Calculator for mathematical operations
        if any(op in query_lower for op in ["+", "-", "*", "/", "calculate", "compute", "math"]):
            if "CalculatorTool" in self._registered_skills:
                plan.append("CalculatorTool")
                reasoning_parts.append("Added calculator for mathematical operations")

        # Determine optimal retrieval mode and configure memory retrieval
        if "MemoryRetrievalSkill" in self._registered_skills:
            retrieval_mode = self._determine_retrieval_mode(query_lower)

            # Configure search context with retrieval mode
            search_context = uif.intermediate_data.get("search_context", {})
            search_context["retrieval_mode"] = retrieval_mode

            # Add graph-specific parameters if graph mode
            if retrieval_mode in ["GRAPH", "HYBRID"] and self._graph_database:
                search_context["graph_depth"] = self._determine_graph_depth(query_lower)
                search_context["max_results"] = 15  # Increase for graph queries

            uif.intermediate_data["search_context"] = search_context

            plan.append("MemoryRetrievalSkill")
            reasoning_parts.append(f"Added {retrieval_mode.lower()} memory retrieval for context")
        
        # Add conflict detection if query suggests multiple sources
        conflict_indicators = ["compare", "versus", "different", "conflicting", "sources"]
        if any(indicator in query_lower for indicator in conflict_indicators):
            if "ConflictDetectorSkill" in self._registered_skills:
                plan.append("ConflictDetectorSkill")
                reasoning_parts.append("Added conflict detection for comparison query")

        # Add implicit knowledge skill for multi-hop questions
        multi_hop_indicators = [
            "how does", "what is the relationship", "connect", "link", "relate",
            "why does", "what causes", "how are", "what connects", "bridge",
            "underlying", "implicit", "hidden connection", "between"
        ]
        if any(indicator in query_lower for indicator in multi_hop_indicators):
            if "ImplicitKnowledgeSkill" in self._registered_skills:
                # Insert before response generation
                plan.append("ImplicitKnowledgeSkill")
                reasoning_parts.append("Added implicit knowledge skill for multi-hop reasoning")

        # Always end with response generation
        if "ResponseGenerationSkill" in self._registered_skills:
            plan.append("ResponseGenerationSkill")
            reasoning_parts.append("Added response generation as final step")
        
        # If no skills were added, use default fallback
        if not plan:
            plan = self._get_default_fallback_plan()
            reasoning_parts = ["Used default fallback plan"]
        
        return PlanGenerationResult(
            plan=plan,
            confidence=0.6,
            reasoning="; ".join(reasoning_parts),
            cache_hit=False,
            generation_time=0.0,
            fallback_used=True
        )
    
    def _get_default_fallback_plan(self) -> List[str]:
        """
        Get the default fallback plan.
        
        Returns:
            List of skill names for default execution
        """
        default_plan = []
        
        # Add available core skills in order
        core_skills = ["MemoryRetrievalSkill", "ResponseGenerationSkill"]
        
        for skill_name in core_skills:
            if skill_name in self._registered_skills:
                default_plan.append(skill_name)
        
        return default_plan or ["ResponseGenerationSkill"]  # Ensure at least one skill
    
    def _cache_plan(self, uif: SAM_UIF, result: PlanGenerationResult) -> None:
        """
        Cache a generated plan for future use.
        
        Args:
            uif: Universal Interface Format
            result: Plan generation result to cache
        """
        query_hash = self._generate_query_hash(uif)
        skill_context = self._generate_skill_context()
        
        entry = PlanCacheEntry(
            plan=result.plan,
            confidence=result.confidence,
            created_at=datetime.now(),
            usage_count=0,
            query_hash=query_hash,
            skill_context=skill_context
        )
        
        self._plan_cache[query_hash] = entry
        self.logger.debug(f"Cached plan for query hash: {query_hash}")
        
        # Clean old cache entries if needed
        self._cleanup_cache()
    
    def _generate_query_hash(self, uif: SAM_UIF) -> str:
        """
        Generate a hash for the query to use as cache key.
        
        Returns:
            Query hash string
        """
        # Include query, user profile, and available skills in hash
        hash_input = f"{uif.input_query}|{uif.active_profile}|{sorted(self._registered_skills.keys())}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _generate_skill_context(self) -> str:
        """
        Generate a context string representing current skill configuration.
        
        Returns:
            Skill context string
        """
        skill_signatures = []
        for skill_name in sorted(self._registered_skills.keys()):
            skill = self._registered_skills[skill_name]
            signature = f"{skill_name}:{skill.skill_version}"
            skill_signatures.append(signature)
        
        return "|".join(skill_signatures)
    
    def _is_cache_entry_valid(self, entry: PlanCacheEntry) -> bool:
        """
        Check if a cache entry is still valid.
        
        Returns:
            True if entry is valid, False otherwise
        """
        # Check TTL
        age = datetime.now() - entry.created_at
        if age.total_seconds() > self._config.plan_cache_ttl:
            return False
        
        # Check if skill context has changed
        current_context = self._generate_skill_context()
        if entry.skill_context != current_context:
            return False
        
        return True
    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        expired_keys = []
        
        for key, entry in self._plan_cache.items():
            if not self._is_cache_entry_valid(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._plan_cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self._plan_cache)
        total_usage = sum(entry.usage_count for entry in self._plan_cache.values())
        
        return {
            "total_entries": total_entries,
            "total_usage": total_usage,
            "average_usage": total_usage / total_entries if total_entries > 0 else 0,
            "cache_enabled": self._config.enable_plan_caching
        }
    
    def clear_cache(self) -> None:
        """Clear the plan cache."""
        self._plan_cache.clear()
        self.logger.info("Plan cache cleared")

    def record_curriculum_performance(
        self,
        plan: List[str],
        success: bool,
        confidence: float,
        execution_time: float
    ) -> None:
        """
        Record performance for curriculum advancement.

        Args:
            plan: The executed plan
            success: Whether execution was successful
            confidence: Final confidence score
            execution_time: Time taken for execution
        """
        if not self._curriculum:
            return

        # Find the reasoning plan that matches this execution
        # For now, create a minimal plan object for recording
        from .reasoning_curriculum import ReasoningPlan, CurriculumLevel

        # Estimate curriculum level from plan complexity
        if len(plan) <= 2:
            level = CurriculumLevel.FOUNDATION
        elif len(plan) <= 4:
            level = CurriculumLevel.INTERMEDIATE
        elif len(plan) <= 6:
            level = CurriculumLevel.ADVANCED
        elif len(plan) <= 8:
            level = CurriculumLevel.EXPERT
        else:
            level = CurriculumLevel.RESEARCH

        # Create a minimal reasoning plan for recording
        reasoning_plan = ReasoningPlan(
            skills=plan,
            curriculum_level=level,
            complexity_score=0.5,  # Default
            confidence_threshold=0.7,  # Default
            reasoning_strategy="executed_plan",
            estimated_effort=len(plan)
        )

        self._curriculum.record_performance(
            plan=reasoning_plan,
            success=success,
            confidence=confidence,
            execution_time=execution_time
        )

    def get_curriculum_status(self) -> Optional[Dict[str, Any]]:
        """
        Get curriculum status and statistics.

        Returns:
            Curriculum status or None if not enabled
        """
        if self._curriculum:
            return self._curriculum.get_curriculum_status()
        return None
    
    def get_registered_skills(self) -> List[str]:
        """Get list of registered skill names."""
        return list(self._registered_skills.keys())

    # ========================================================================
    # GRAPH-AWARE PLANNING METHODS (Cognitive Memory Core Phase B)
    # ========================================================================

    def _determine_retrieval_mode(self, query_lower: str) -> str:
        """
        Determine the optimal retrieval mode based on query characteristics.

        Args:
            query_lower: Lowercase query string

        Returns:
            Optimal retrieval mode: VECTOR, GRAPH, HYBRID, or AUTO
        """
        # If graph database not available, use vector mode
        if not self._graph_database:
            return "VECTOR"

        # Graph mode indicators (relationship and connection queries)
        graph_indicators = [
            "how", "why", "what causes", "relationship", "connection", "related to",
            "because", "leads to", "results in", "depends on", "influences",
            "who works", "where is", "when did", "which company", "what technology",
            "connects", "links", "associates", "correlates", "impacts"
        ]

        # Vector mode indicators (similarity and content queries)
        vector_indicators = [
            "similar to", "like", "about", "regarding", "concerning",
            "find documents", "search for", "show me", "tell me about",
            "content", "text", "document", "file", "information"
        ]

        # Hybrid mode indicators (complex analytical queries)
        hybrid_indicators = [
            "explain", "analyze", "compare", "contrast", "overview",
            "summary", "comprehensive", "detailed", "complete picture",
            "understand", "breakdown", "elaborate", "describe fully"
        ]

        # Count indicators
        graph_score = sum(1 for indicator in graph_indicators if indicator in query_lower)
        vector_score = sum(1 for indicator in vector_indicators if indicator in query_lower)
        hybrid_score = sum(1 for indicator in hybrid_indicators if indicator in query_lower)

        # Determine mode based on scores
        if hybrid_score > 0 or (graph_score > 0 and vector_score > 0):
            return "HYBRID"
        elif graph_score > vector_score:
            return "GRAPH"
        elif vector_score > 0:
            return "VECTOR"
        else:
            return "AUTO"  # Let MemoryRetrievalSkill decide

    def _determine_graph_depth(self, query_lower: str) -> int:
        """
        Determine optimal graph traversal depth based on query complexity.

        Args:
            query_lower: Lowercase query string

        Returns:
            Graph traversal depth (1-4)
        """
        # Deep traversal indicators
        deep_indicators = [
            "comprehensive", "complete", "all", "everything", "thorough",
            "detailed", "full picture", "entire", "whole", "extensive"
        ]

        # Multi-hop indicators
        multi_hop_indicators = [
            "chain", "sequence", "path", "route", "journey", "process",
            "step by step", "how does", "what leads", "cascade", "ripple"
        ]

        # Simple relationship indicators
        simple_indicators = [
            "direct", "immediate", "first", "primary", "main", "basic"
        ]

        # Count indicators
        deep_count = sum(1 for indicator in deep_indicators if indicator in query_lower)
        multi_hop_count = sum(1 for indicator in multi_hop_indicators if indicator in query_lower)
        simple_count = sum(1 for indicator in simple_indicators if indicator in query_lower)

        # Determine depth
        if deep_count > 0:
            return 4  # Maximum depth for comprehensive queries
        elif multi_hop_count > 0:
            return 3  # Multi-hop traversal
        elif simple_count > 0:
            return 1  # Direct relationships only
        else:
            return 2  # Default moderate depth

    def is_graph_aware(self) -> bool:
        """
        Check if the planner has graph database awareness.

        Returns:
            True if graph-aware, False otherwise
        """
        return self._graph_database is not None

    def get_graph_status(self) -> Dict[str, Any]:
        """
        Get graph database status for planning.

        Returns:
            Dictionary with graph status information
        """
        return {
            "graph_available": self._graph_database is not None,
            "graph_aware_planning": True,
            "supported_modes": ["VECTOR", "GRAPH", "HYBRID", "AUTO"] if self._graph_database else ["VECTOR"],
            "default_graph_depth": 2,
            "max_graph_depth": 4
        }

    def _check_for_ttt_opportunity(self, uif: SAM_UIF) -> Optional[List[str]]:
        """
        Check if the query represents a few-shot reasoning task suitable for TTT.

        Args:
            uif: Universal Interface Format with query and context

        Returns:
            TTT-enabled plan if applicable, None otherwise
        """
        try:
            query = uif.input_query.lower()

            # Pattern 1: Explicit few-shot structure (Example: ... Problem: ...)
            example_pattern = r'example\s*\d*\s*:.*?(?=example\s*\d*\s*:|problem\s*:|$)'
            examples = re.findall(example_pattern, query, re.IGNORECASE | re.DOTALL)

            if len(examples) >= 2:
                self.logger.info(f"Detected {len(examples)} explicit examples - TTT applicable")
                return self._create_ttt_plan(examples, "explicit_examples")

            # Pattern 2: Input-Output pairs
            io_pattern = r'(?:input|in)\s*:.*?(?:output|out|answer)\s*:.*?(?=(?:input|in)\s*:|$)'
            io_pairs = re.findall(io_pattern, query, re.IGNORECASE | re.DOTALL)

            if len(io_pairs) >= 2:
                self.logger.info(f"Detected {len(io_pairs)} input-output pairs - TTT applicable")
                return self._create_ttt_plan(io_pairs, "io_pairs")

            # Pattern 3: Numbered examples (1. ... 2. ... Now solve: ...)
            numbered_pattern = r'\d+\.\s+.*?(?=\d+\.\s+|now\s+solve|solve\s+this|what\s+is|$)'
            numbered_examples = re.findall(numbered_pattern, query, re.IGNORECASE | re.DOTALL)

            if len(numbered_examples) >= 2 and any(keyword in query for keyword in ['solve', 'what is', 'find', 'determine']):
                self.logger.info(f"Detected {len(numbered_examples)} numbered examples - TTT applicable")
                return self._create_ttt_plan(numbered_examples, "numbered_examples")

            # Pattern 4: Analogical reasoning (A is to B as C is to ?)
            analogy_pattern = r'.*?\s+is\s+to\s+.*?\s+as\s+.*?\s+is\s+to\s+.*?'
            if re.search(analogy_pattern, query, re.IGNORECASE):
                self.logger.info("Detected analogical reasoning pattern - TTT applicable")
                return self._create_ttt_plan([query], "analogical_reasoning")

            # Pattern 5: Rule learning from examples
            rule_keywords = ['pattern', 'rule', 'follows', 'sequence', 'series', 'logic']
            if any(keyword in query for keyword in rule_keywords) and len(query.split()) > 20:
                # Look for multiple instances of similar structures
                sentences = query.split('.')
                if len(sentences) >= 3:
                    self.logger.info("Detected potential rule learning task - TTT applicable")
                    return self._create_ttt_plan(sentences, "rule_learning")

            return None

        except Exception as e:
            self.logger.error(f"Error in TTT detection: {e}")
            return None

    def _create_ttt_plan(self, examples: List[str], task_type: str) -> List[str]:
        """
        Create a TTT-enabled execution plan.

        Args:
            examples: Detected examples or patterns
            task_type: Type of few-shot task detected

        Returns:
            List of skill names for TTT execution
        """
        # Store TTT context for the skill
        ttt_context = {
            "examples": examples,
            "task_type": task_type,
            "example_count": len(examples),
            "detected_at": datetime.now().isoformat()
        }

        self.logger.info(f"Creating TTT plan for {task_type} with {len(examples)} examples")

        # TTT-enabled plan: adaptation first, then response generation
        return ["TestTimeAdaptation", "ResponseGeneration"]

    def _extract_few_shot_examples(self, query: str, pattern_type: str) -> List[Dict[str, str]]:
        """
        Extract structured few-shot examples from query text.

        Args:
            query: Input query text
            pattern_type: Type of pattern detected

        Returns:
            List of structured examples with input/output pairs
        """
        examples = []

        try:
            if pattern_type == "explicit_examples":
                # Extract Example: ... format
                example_blocks = re.findall(r'example\s*\d*\s*:(.*?)(?=example\s*\d*\s*:|problem\s*:|$)',
                                          query, re.IGNORECASE | re.DOTALL)

                for i, block in enumerate(example_blocks):
                    # Try to split into input/output
                    if '->' in block:
                        parts = block.split('->', 1)
                        examples.append({
                            "input": parts[0].strip(),
                            "output": parts[1].strip(),
                            "example_id": f"explicit_{i}"
                        })
                    else:
                        examples.append({
                            "input": block.strip(),
                            "output": "",
                            "example_id": f"explicit_{i}"
                        })

            elif pattern_type == "io_pairs":
                # Extract Input: ... Output: ... format
                pairs = re.findall(r'(?:input|in)\s*:(.*?)(?:output|out|answer)\s*:(.*?)(?=(?:input|in)\s*:|$)',
                                 query, re.IGNORECASE | re.DOTALL)

                for i, (inp, out) in enumerate(pairs):
                    examples.append({
                        "input": inp.strip(),
                        "output": out.strip(),
                        "example_id": f"io_pair_{i}"
                    })

            elif pattern_type == "numbered_examples":
                # Extract numbered examples
                numbered = re.findall(r'\d+\.\s+(.*?)(?=\d+\.\s+|now\s+solve|solve\s+this|what\s+is|$)',
                                    query, re.IGNORECASE | re.DOTALL)

                for i, example in enumerate(numbered):
                    if '->' in example or '=' in example:
                        if '->' in example:
                            parts = example.split('->', 1)
                        else:
                            parts = example.split('=', 1)

                        examples.append({
                            "input": parts[0].strip(),
                            "output": parts[1].strip(),
                            "example_id": f"numbered_{i}"
                        })
                    else:
                        examples.append({
                            "input": example.strip(),
                            "output": "",
                            "example_id": f"numbered_{i}"
                        })

        except Exception as e:
            self.logger.error(f"Error extracting examples: {e}")

        return examples

    # SELF-REFLECT Policy Implementation (Phase 5C)

    def _apply_self_reflect_policy(self, result: PlanGenerationResult, uif: SAM_UIF) -> PlanGenerationResult:
        """
        Apply SELF-REFLECT policy to enhance execution plan.

        Determines if AutonomousFactualCorrectionSkill should be added to the plan
        based on query characteristics, confidence levels, and user profiles.

        Args:
            result: Original plan generation result
            uif: Universal Interface Format with query context

        Returns:
            Enhanced plan generation result with SELF-REFLECT if applicable
        """
        try:
            # Check if SELF-REFLECT is enabled
            if not self._config.enable_self_reflect:
                return result

            # Check if AutonomousFactualCorrectionSkill is available
            if "AutonomousFactualCorrectionSkill" not in self._registered_skills:
                self.logger.warning("AutonomousFactualCorrectionSkill not registered, skipping SELF-REFLECT")
                return result

            should_add_self_reflect = self._should_trigger_self_reflect_policy(uif, result)

            if should_add_self_reflect:
                # Add AutonomousFactualCorrectionSkill after ResponseGenerationSkill
                enhanced_plan = self._insert_self_reflect_skill(result.plan)

                # Update result with enhanced plan
                enhanced_result = PlanGenerationResult(
                    plan=enhanced_plan,
                    confidence=result.confidence,
                    reasoning=result.reasoning + "; Added SELF-REFLECT for factual verification",
                    cache_hit=result.cache_hit,
                    generation_time=result.generation_time,
                    fallback_used=result.fallback_used
                )

                self.logger.info("Enhanced plan with SELF-REFLECT capability")
                return enhanced_result

            return result

        except Exception as e:
            self.logger.error(f"Error applying SELF-REFLECT policy: {e}")
            return result

    def _should_trigger_self_reflect_policy(self, uif: SAM_UIF, result: PlanGenerationResult) -> bool:
        """
        Determine if SELF-REFLECT should be triggered based on policy rules.

        Args:
            uif: Universal Interface Format with query context
            result: Plan generation result

        Returns:
            True if SELF-REFLECT should be triggered
        """
        try:
            query = uif.input_query.lower()

            # Rule 1: Check for factual query keywords
            factual_keywords = self._config.self_reflect_query_keywords
            if any(keyword in query for keyword in factual_keywords):
                self.logger.info("SELF-REFLECT triggered by factual query keywords")
                return True

            # Rule 2: Check confidence threshold
            if result.confidence < self._config.self_reflect_confidence_threshold:
                self.logger.info(f"SELF-REFLECT triggered by low confidence: {result.confidence}")
                return True

            # Rule 3: Check user profile (if available in UIF)
            user_profile = uif.intermediate_data.get("user_profile", "")
            if user_profile in self._config.self_reflect_profiles:
                self.logger.info(f"SELF-REFLECT triggered by user profile: {user_profile}")
                return True

            # Rule 4: Check for high dissonance indicators
            dissonance_keywords = ["conflicting", "contradictory", "uncertain", "unclear", "disputed"]
            if any(keyword in query for keyword in dissonance_keywords):
                self.logger.info("SELF-REFLECT triggered by dissonance indicators")
                return True

            # Rule 5: Long responses are more likely to contain errors
            response_text = uif.intermediate_data.get("response_text", "")
            if len(response_text.split()) > 150:
                self.logger.info("SELF-REFLECT triggered by long response length")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error in SELF-REFLECT policy evaluation: {e}")
            return False

    def _insert_self_reflect_skill(self, original_plan: List[str]) -> List[str]:
        """
        Insert AutonomousFactualCorrectionSkill into the execution plan.

        Args:
            original_plan: Original execution plan

        Returns:
            Enhanced plan with SELF-REFLECT skill
        """
        enhanced_plan = original_plan.copy()

        # Find the best insertion point (after ResponseGenerationSkill)
        insertion_index = len(enhanced_plan)  # Default to end

        for i, skill_name in enumerate(enhanced_plan):
            if skill_name == "ResponseGenerationSkill":
                insertion_index = i + 1
                break

        # Insert the SELF-REFLECT skill
        enhanced_plan.insert(insertion_index, "AutonomousFactualCorrectionSkill")

        return enhanced_plan
