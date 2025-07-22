"""
Motivation Engine for SAM Autonomy
==================================

This module implements the MotivationEngine that analyzes UIF execution results
to autonomously generate new goals based on detected patterns, conflicts, and
learning opportunities.

Author: SAM Development Team
Version: 2.0.0
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..orchestration.uif import SAM_UIF
from .goals import Goal
from .goal_stack import GoalStack
from .safety.goal_validator import GoalSafetyValidator

logger = logging.getLogger(__name__)

@dataclass
class GoalGenerationRule:
    """Rule for generating goals from UIF analysis."""
    name: str
    description: str
    trigger_condition: str  # Key to check in UIF
    goal_template: str
    priority_base: float
    enabled: bool = True

class MotivationEngine:
    """
    Autonomous goal generation engine for SAM.
    
    The MotivationEngine analyzes completed UIF executions to identify
    opportunities for autonomous goal generation. It looks for patterns
    such as conflicts, learning failures, low-confidence inferences,
    and other "goal-worthy" events.
    
    Features:
    - Rule-based goal generation
    - Context-aware goal creation
    - Priority calculation
    - Integration with GoalStack
    - Safety validation
    - Comprehensive logging
    """
    
    def __init__(self, goal_stack: GoalStack, 
                 safety_validator: Optional[GoalSafetyValidator] = None):
        """
        Initialize the MotivationEngine.
        
        Args:
            goal_stack: GoalStack instance for storing generated goals
            safety_validator: Optional safety validator
        """
        self.logger = logging.getLogger(f"{__name__}.MotivationEngine")
        self.goal_stack = goal_stack
        self.safety_validator = safety_validator or GoalSafetyValidator()
        
        # Configuration
        self.config = {
            'enable_conflict_goals': True,
            'enable_inference_goals': True,
            'enable_learning_goals': True,
            'enable_error_goals': True,
            'enable_optimization_goals': True,
            'min_confidence_threshold': 0.7,
            'max_goals_per_analysis': 5,
            'priority_boost_recent': 0.1,
            'priority_boost_critical': 0.2
        }
        
        # Statistics tracking (initialize before rules)
        self.stats = {
            'total_analyses': 0,
            'goals_generated': 0,
            'goals_rejected': 0,
            'generation_errors': 0,
            'rules_triggered': {},
            'last_analysis': None
        }

        # Initialize goal generation rules
        self._init_generation_rules()

        self.logger.info("MotivationEngine initialized")
    
    def _init_generation_rules(self) -> None:
        """Initialize the goal generation rules."""
        self.generation_rules = [
            GoalGenerationRule(
                name="conflict_detection",
                description="Generate goals to resolve detected conflicts",
                trigger_condition="conflict_detected",
                goal_template="Resolve conflict between memory chunks: {conflicting_ids}",
                priority_base=0.8
            ),
            GoalGenerationRule(
                name="low_confidence_inference",
                description="Generate goals to find supporting evidence for low-confidence inferences",
                trigger_condition="implicit_knowledge_summary",
                goal_template="Find supporting evidence for inferred relationship: {inference_summary}",
                priority_base=0.6
            ),
            GoalGenerationRule(
                name="learning_failure",
                description="Generate goals to retry or diagnose failed learning attempts",
                trigger_condition="learning_stall",
                goal_template="Re-attempt or diagnose failed knowledge edit: {edit_details}",
                priority_base=0.7
            ),
            GoalGenerationRule(
                name="factual_error",
                description="Generate goals to correct detected factual errors",
                trigger_condition="factual_error_detected",
                goal_template="Correct factual error in memory: {error_details}",
                priority_base=0.9
            ),
            GoalGenerationRule(
                name="knowledge_gap",
                description="Generate goals to fill identified knowledge gaps",
                trigger_condition="knowledge_gap_identified",
                goal_template="Research and fill knowledge gap: {gap_description}",
                priority_base=0.5
            ),
            GoalGenerationRule(
                name="optimization_opportunity",
                description="Generate goals to optimize system performance",
                trigger_condition="optimization_opportunity",
                goal_template="Optimize system component: {optimization_target}",
                priority_base=0.4
            ),
            GoalGenerationRule(
                name="web_search_failure",
                description="Generate goals to retry failed web searches with different strategies",
                trigger_condition="web_search_failed",
                goal_template="Retry web search with alternative strategy: {search_query}",
                priority_base=0.6
            ),
            GoalGenerationRule(
                name="memory_inconsistency",
                description="Generate goals to resolve memory inconsistencies",
                trigger_condition="memory_inconsistency",
                goal_template="Resolve memory inconsistency: {inconsistency_details}",
                priority_base=0.8
            )
        ]
        
        # Initialize rule statistics
        for rule in self.generation_rules:
            self.stats['rules_triggered'][rule.name] = 0
    
    def generate_goals_from_uif(self, uif: SAM_UIF) -> List[Goal]:
        """
        Analyze a completed UIF and generate autonomous goals.

        Args:
            uif: Completed UIF to analyze

        Returns:
            List of generated goals
        """
        self.stats['total_analyses'] += 1
        self.stats['last_analysis'] = datetime.now()

        generated_goals = []

        # Phase 2B: Add tracing integration
        trace_id = uif.intermediate_data.get('trace_id')
        if trace_id:
            try:
                from ..cognition.trace_logger import get_trace_logger
                trace_logger = get_trace_logger()
                trace_logger.log_event(
                    trace_id=trace_id,
                    source_module="MotivationEngine",
                    event_type="goal_generation_start",
                    message=f"Starting goal generation analysis for UIF {uif.task_id}",
                    severity="info",
                    payload={
                        "uif_task_id": uif.task_id,
                        "analysis_timestamp": datetime.now().isoformat(),
                        "total_analyses": self.stats['total_analyses']
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to log goal generation start: {e}")

        try:
            self.logger.debug(f"Analyzing UIF {uif.task_id} for goal generation")

            # Check each generation rule
            for rule in self.generation_rules:
                if not rule.enabled:
                    continue

                # Check if rule condition is met
                goals_from_rule = self._apply_generation_rule(rule, uif)
                generated_goals.extend(goals_from_rule)

                if goals_from_rule:
                    self.stats['rules_triggered'][rule.name] += len(goals_from_rule)

                    # Phase 2B: Log rule triggering
                    if trace_id:
                        try:
                            trace_logger.log_event(
                                trace_id=trace_id,
                                source_module="MotivationEngine",
                                event_type="rule_triggered",
                                message=f"Rule '{rule.name}' generated {len(goals_from_rule)} goals",
                                severity="info",
                                payload={
                                    "rule_name": rule.name,
                                    "rule_description": rule.description,
                                    "goals_generated": len(goals_from_rule),
                                    "goal_descriptions": [g.description for g in goals_from_rule]
                                }
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to log rule triggering: {e}")

            # Limit number of goals per analysis
            if len(generated_goals) > self.config['max_goals_per_analysis']:
                # Sort by priority and take top goals
                generated_goals.sort(key=lambda g: g.priority, reverse=True)
                limited_goals = generated_goals[:self.config['max_goals_per_analysis']]

                # Phase 2B: Log goal limiting
                if trace_id:
                    try:
                        trace_logger.log_event(
                            trace_id=trace_id,
                            source_module="MotivationEngine",
                            event_type="goal_limiting",
                            message=f"Limited goals from {len(generated_goals)} to {len(limited_goals)}",
                            severity="info",
                            payload={
                                "original_count": len(generated_goals),
                                "limited_count": len(limited_goals),
                                "max_goals_config": self.config['max_goals_per_analysis'],
                                "dropped_goals": [g.description for g in generated_goals[len(limited_goals):]]
                            }
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to log goal limiting: {e}")

                generated_goals = limited_goals
                self.logger.info(f"Limited goals to {self.config['max_goals_per_analysis']} highest priority")
            
            # Add generated goals to the goal stack
            successfully_added = 0
            rejected_goals = []

            for goal in generated_goals:
                if self.goal_stack.add_goal(goal):
                    successfully_added += 1
                    self.stats['goals_generated'] += 1
                else:
                    self.stats['goals_rejected'] += 1
                    rejected_goals.append(goal)

            # Phase 2B: Log final goal generation results
            if trace_id:
                try:
                    trace_logger.log_event(
                        trace_id=trace_id,
                        source_module="MotivationEngine",
                        event_type="goal_generation_complete",
                        message=f"Goal generation completed: {successfully_added} added, {len(rejected_goals)} rejected",
                        severity="info",
                        payload={
                            "total_goals_generated": len(generated_goals),
                            "successfully_added": successfully_added,
                            "rejected_goals": len(rejected_goals),
                            "added_goal_descriptions": [g.description for g in generated_goals if g not in rejected_goals],
                            "rejected_goal_descriptions": [g.description for g in rejected_goals],
                            "rules_triggered": {k: v for k, v in self.stats['rules_triggered'].items() if v > 0},
                            "generation_stats": {
                                "total_analyses": self.stats['total_analyses'],
                                "goals_generated": self.stats['goals_generated'],
                                "goals_rejected": self.stats['goals_rejected']
                            }
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to log goal generation completion: {e}")

            if successfully_added > 0:
                self.logger.info(f"Generated {successfully_added} new autonomous goals from UIF analysis")

            return generated_goals

        except Exception as e:
            self.logger.error(f"Error generating goals from UIF: {e}")
            self.stats['generation_errors'] += 1

            # Phase 2B: Log generation errors
            if trace_id:
                try:
                    trace_logger.log_event(
                        trace_id=trace_id,
                        source_module="MotivationEngine",
                        event_type="goal_generation_error",
                        message=f"Goal generation failed: {str(e)}",
                        severity="error",
                        payload={
                            "error_message": str(e),
                            "error_type": type(e).__name__,
                            "uif_task_id": uif.task_id,
                            "generation_errors": self.stats['generation_errors']
                        }
                    )
                except Exception as trace_error:
                    self.logger.warning(f"Failed to log goal generation error: {trace_error}")

            return []
    
    def _apply_generation_rule(self, rule: GoalGenerationRule, uif: SAM_UIF) -> List[Goal]:
        """Apply a specific generation rule to a UIF."""
        goals = []
        
        try:
            # Check if trigger condition exists in UIF
            trigger_data = self._extract_trigger_data(rule.trigger_condition, uif)
            if not trigger_data:
                return goals
            
            # Generate goal based on rule
            goal = self._create_goal_from_rule(rule, trigger_data, uif)
            if goal:
                goals.append(goal)
            
        except Exception as e:
            self.logger.error(f"Error applying rule {rule.name}: {e}")
        
        return goals
    
    def _extract_trigger_data(self, trigger_condition: str, uif: SAM_UIF) -> Optional[Dict[str, Any]]:
        """Extract trigger data from UIF based on condition."""
        
        # Check intermediate data
        if hasattr(uif, 'intermediate_data') and uif.intermediate_data:
            if trigger_condition in uif.intermediate_data:
                return {
                    'source': 'intermediate_data',
                    'data': uif.intermediate_data[trigger_condition],
                    'context': uif.intermediate_data
                }
        
        # Check execution results
        if hasattr(uif, 'execution_results') and uif.execution_results:
            for skill_name, result in uif.execution_results.items():
                if isinstance(result, dict) and trigger_condition in result:
                    return {
                        'source': f'execution_results.{skill_name}',
                        'data': result[trigger_condition],
                        'context': result,
                        'source_skill': skill_name
                    }
        
        # Check log entries for patterns
        if hasattr(uif, 'log_trace') and uif.log_trace:
            for log_entry in uif.log_trace:
                if trigger_condition.lower() in log_entry.lower():
                    return {
                        'source': 'log_trace',
                        'data': log_entry,
                        'context': {'log_entries': uif.log_trace}
                    }
        
        # Check error conditions
        if hasattr(uif, 'error_message') and uif.error_message:
            if trigger_condition.lower() in uif.error_message.lower():
                return {
                    'source': 'error_message',
                    'data': uif.error_message,
                    'context': {'status': uif.status}
                }
        
        return None
    
    def _create_goal_from_rule(self, rule: GoalGenerationRule, 
                              trigger_data: Dict[str, Any], uif: SAM_UIF) -> Optional[Goal]:
        """Create a goal from a rule and trigger data."""
        try:
            # Extract context for goal description
            context_vars = self._extract_context_variables(trigger_data)
            
            # Format goal description
            description = rule.goal_template.format(**context_vars)
            
            # Calculate priority
            priority = self._calculate_goal_priority(rule, trigger_data, uif)
            
            # Determine source skill
            source_skill = trigger_data.get('source_skill', 'MotivationEngine')
            
            # Create goal
            goal = Goal(
                description=description,
                priority=priority,
                source_skill=source_skill,
                source_context={
                    'rule_name': rule.name,
                    'trigger_condition': rule.trigger_condition,
                    'trigger_data': trigger_data,
                    'uif_task_id': uif.task_id,
                    'generation_timestamp': datetime.now().isoformat()
                },
                tags=[rule.name, 'autonomous', 'motivation_engine']
            )
            
            return goal
            
        except Exception as e:
            self.logger.error(f"Error creating goal from rule {rule.name}: {e}")
            return None
    
    def _extract_context_variables(self, trigger_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract variables for goal description formatting."""
        context_vars = {}
        
        data = trigger_data.get('data', {})
        
        # Handle different data types
        if isinstance(data, dict):
            # Extract common fields
            context_vars.update({
                'conflicting_ids': str(data.get('conflicting_ids', 'unknown')),
                'inference_summary': str(data.get('summary', data.get('inference', 'unknown'))),
                'edit_details': str(data.get('edit_details', data.get('error', 'unknown'))),
                'error_details': str(data.get('error_details', data.get('error', 'unknown'))),
                'gap_description': str(data.get('gap_description', data.get('description', 'unknown'))),
                'optimization_target': str(data.get('target', data.get('component', 'unknown'))),
                'search_query': str(data.get('query', data.get('search_term', 'unknown'))),
                'inconsistency_details': str(data.get('inconsistency', data.get('details', 'unknown')))
            })
        else:
            # Handle string data
            data_str = str(data)[:100]  # Limit length
            context_vars.update({
                'conflicting_ids': data_str,
                'inference_summary': data_str,
                'edit_details': data_str,
                'error_details': data_str,
                'gap_description': data_str,
                'optimization_target': data_str,
                'search_query': data_str,
                'inconsistency_details': data_str
            })
        
        return context_vars
    
    def _calculate_goal_priority(self, rule: GoalGenerationRule, 
                               trigger_data: Dict[str, Any], uif: SAM_UIF) -> float:
        """Calculate priority for a generated goal."""
        base_priority = rule.priority_base
        
        # Apply priority boosts
        priority_boost = 0.0
        
        # Recent execution boost
        if hasattr(uif, 'created_at'):
            try:
                creation_time = datetime.fromisoformat(uif.created_at)
                time_diff = datetime.now() - creation_time
                if time_diff.total_seconds() < 3600:  # Within last hour
                    priority_boost += self.config['priority_boost_recent']
            except:
                pass
        
        # Critical condition boost
        data = trigger_data.get('data', {})
        if isinstance(data, dict):
            if data.get('critical', False) or data.get('urgent', False):
                priority_boost += self.config['priority_boost_critical']
        
        # Confidence-based adjustment
        if hasattr(uif, 'confidence_score') and uif.confidence_score is not None:
            if uif.confidence_score < self.config['min_confidence_threshold']:
                priority_boost += 0.1  # Boost for low confidence
        
        final_priority = min(1.0, base_priority + priority_boost)
        return final_priority
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get motivation engine statistics."""
        return self.stats.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update engine configuration."""
        self.config.update(new_config)
        self.logger.info(f"MotivationEngine configuration updated: {new_config}")
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a specific generation rule."""
        for rule in self.generation_rules:
            if rule.name == rule_name:
                rule.enabled = True
                self.logger.info(f"Enabled generation rule: {rule_name}")
                return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a specific generation rule."""
        for rule in self.generation_rules:
            if rule.name == rule_name:
                rule.enabled = False
                self.logger.info(f"Disabled generation rule: {rule_name}")
                return True
        return False

# Global motivation engine instance
_motivation_engine = None

def get_motivation_engine() -> MotivationEngine:
    """Get the global motivation engine instance."""
    global _motivation_engine
    if _motivation_engine is None:
        from .goal_stack import GoalStack
        from .safety.goal_validator import GoalSafetyValidator

        # Initialize with default components
        goal_stack = GoalStack()
        safety_validator = GoalSafetyValidator()
        _motivation_engine = MotivationEngine(goal_stack, safety_validator)
    return _motivation_engine
