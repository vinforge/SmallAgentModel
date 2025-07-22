"""
Tool Evaluation & Feedback Loop for SAM
Evaluates tool results and learns which tools are most effective for each task type.

Sprint 8 Task 3: Tool Evaluation & Feedback Loop
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class EvaluationCriteria(Enum):
    """Criteria for evaluating tool results."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    EFFICIENCY = "efficiency"
    USER_SATISFACTION = "user_satisfaction"

class ToolPerformanceLevel(Enum):
    """Performance levels for tools."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    FAILING = "failing"

@dataclass
class ToolEvaluation:
    """Evaluation of a tool execution."""
    evaluation_id: str
    tool_id: str
    execution_id: str
    request_id: str
    user_id: str
    goal: str
    input_data: Dict[str, Any]
    output_data: Any
    success: bool
    execution_time_ms: int
    accuracy_score: float
    completeness_score: float
    relevance_score: float
    efficiency_score: float
    user_satisfaction_score: float
    overall_score: float
    feedback_notes: str
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class ToolScorecard:
    """Performance scorecard for a tool."""
    tool_id: str
    tool_name: str
    total_executions: int
    successful_executions: int
    success_rate: float
    average_execution_time: float
    average_accuracy: float
    average_completeness: float
    average_relevance: float
    average_efficiency: float
    average_user_satisfaction: float
    overall_performance_score: float
    performance_level: ToolPerformanceLevel
    best_use_cases: List[str]
    common_failures: List[str]
    improvement_suggestions: List[str]
    last_updated: str
    metadata: Dict[str, Any]

@dataclass
class TaskTypeProfile:
    """Performance profile for tools on specific task types."""
    task_type: str
    tool_rankings: List[Tuple[str, float]]  # (tool_id, score)
    recommended_tools: List[str]
    fallback_tools: List[str]
    success_patterns: List[str]
    failure_patterns: List[str]
    last_updated: str

class ToolEvaluator:
    """
    Evaluates tool performance and maintains learning feedback loops.
    """
    
    def __init__(self, scorecard_file: str = "tool_scorecard.json"):
        """
        Initialize the tool evaluator.
        
        Args:
            scorecard_file: Path to tool scorecard storage file
        """
        self.scorecard_file = Path(scorecard_file)
        
        # Storage
        self.evaluations: List[ToolEvaluation] = []
        self.scorecards: Dict[str, ToolScorecard] = {}
        self.task_profiles: Dict[str, TaskTypeProfile] = {}
        
        # Configuration
        self.config = {
            'auto_evaluate': True,
            'min_executions_for_ranking': 5,
            'evaluation_window_days': 30,
            'performance_threshold': 0.7
        }
        
        # Load existing data
        self._load_scorecards()
        
        logger.info(f"Tool evaluator initialized with {len(self.scorecards)} scorecards")
    
    def evaluate_execution(self, tool_id: str, execution_id: str, request_id: str,
                          user_id: str, goal: str, input_data: Dict[str, Any],
                          output_data: Any, success: bool, execution_time_ms: int,
                          user_feedback: Optional[Dict[str, Any]] = None) -> str:
        """
        Evaluate a tool execution.
        
        Args:
            tool_id: Tool ID
            execution_id: Execution ID
            request_id: Request ID
            user_id: User ID
            goal: Original goal/query
            input_data: Input data used
            output_data: Output produced
            success: Whether execution was successful
            execution_time_ms: Execution time in milliseconds
            user_feedback: Optional user feedback
            
        Returns:
            Evaluation ID
        """
        try:
            import uuid
            evaluation_id = f"eval_{uuid.uuid4().hex[:12]}"
            
            # Calculate evaluation scores
            scores = self._calculate_evaluation_scores(
                tool_id, goal, input_data, output_data, success, 
                execution_time_ms, user_feedback
            )
            
            evaluation = ToolEvaluation(
                evaluation_id=evaluation_id,
                tool_id=tool_id,
                execution_id=execution_id,
                request_id=request_id,
                user_id=user_id,
                goal=goal,
                input_data=input_data,
                output_data=output_data,
                success=success,
                execution_time_ms=execution_time_ms,
                accuracy_score=scores['accuracy'],
                completeness_score=scores['completeness'],
                relevance_score=scores['relevance'],
                efficiency_score=scores['efficiency'],
                user_satisfaction_score=scores['user_satisfaction'],
                overall_score=scores['overall'],
                feedback_notes=user_feedback.get('notes', '') if user_feedback else '',
                timestamp=datetime.now().isoformat(),
                metadata=user_feedback or {}
            )
            
            self.evaluations.append(evaluation)
            
            # Update tool scorecard
            self._update_tool_scorecard(evaluation)
            
            # Update task type profiles
            task_type = self._classify_task_type(goal)
            self._update_task_profile(task_type, evaluation)
            
            logger.info(f"Evaluated execution: {tool_id} - overall score: {scores['overall']:.2f}")
            return evaluation_id
            
        except Exception as e:
            logger.error(f"Error evaluating execution: {e}")
            raise
    
    def get_tool_scorecard(self, tool_id: str) -> Optional[ToolScorecard]:
        """Get scorecard for a specific tool."""
        return self.scorecards.get(tool_id)
    
    def get_tool_rankings(self, task_type: Optional[str] = None,
                         min_executions: int = None) -> List[Tuple[str, float]]:
        """
        Get tool rankings by performance.
        
        Args:
            task_type: Optional task type filter
            min_executions: Minimum executions required
            
        Returns:
            List of (tool_id, score) tuples sorted by performance
        """
        try:
            if min_executions is None:
                min_executions = self.config['min_executions_for_ranking']
            
            # Get relevant scorecards
            relevant_scorecards = []
            for scorecard in self.scorecards.values():
                if scorecard.total_executions >= min_executions:
                    relevant_scorecards.append(scorecard)
            
            # Filter by task type if specified
            if task_type and task_type in self.task_profiles:
                task_profile = self.task_profiles[task_type]
                tool_scores = dict(task_profile.tool_rankings)
                
                rankings = [(tool_id, tool_scores.get(tool_id, 0.0)) 
                           for tool_id in tool_scores.keys()]
            else:
                # Use overall performance scores
                rankings = [(scorecard.tool_id, scorecard.overall_performance_score)
                           for scorecard in relevant_scorecards]
            
            # Sort by score (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error getting tool rankings: {e}")
            return []
    
    def get_recommended_tools(self, goal: str, top_k: int = 3) -> List[str]:
        """
        Get recommended tools for a specific goal.
        
        Args:
            goal: Goal/query to find tools for
            top_k: Number of tools to recommend
            
        Returns:
            List of recommended tool IDs
        """
        try:
            task_type = self._classify_task_type(goal)
            
            # Check if we have a task profile
            if task_type in self.task_profiles:
                profile = self.task_profiles[task_type]
                return profile.recommended_tools[:top_k]
            
            # Fall back to overall rankings
            rankings = self.get_tool_rankings()
            return [tool_id for tool_id, _ in rankings[:top_k]]
            
        except Exception as e:
            logger.error(f"Error getting recommended tools: {e}")
            return []
    
    def get_fallback_tools(self, goal: str, failed_tool_id: str) -> List[str]:
        """
        Get fallback tools when a tool fails.
        
        Args:
            goal: Original goal
            failed_tool_id: Tool that failed
            
        Returns:
            List of fallback tool IDs
        """
        try:
            task_type = self._classify_task_type(goal)
            
            # Check task profile for fallbacks
            if task_type in self.task_profiles:
                profile = self.task_profiles[task_type]
                fallbacks = [tool_id for tool_id in profile.fallback_tools 
                           if tool_id != failed_tool_id]
                if fallbacks:
                    return fallbacks
            
            # Fall back to general rankings, excluding failed tool
            rankings = self.get_tool_rankings()
            return [tool_id for tool_id, _ in rankings 
                   if tool_id != failed_tool_id][:3]
            
        except Exception as e:
            logger.error(f"Error getting fallback tools: {e}")
            return []
    
    def analyze_tool_performance(self, tool_id: str, 
                               days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze detailed performance for a tool.
        
        Args:
            tool_id: Tool ID to analyze
            days_back: Number of days to analyze
            
        Returns:
            Performance analysis
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Get recent evaluations for this tool
            recent_evals = [
                eval for eval in self.evaluations
                if (eval.tool_id == tool_id and 
                    datetime.fromisoformat(eval.timestamp) >= cutoff_date)
            ]
            
            if not recent_evals:
                return {'error': 'No recent evaluations found'}
            
            # Calculate trends
            scores_by_date = {}
            for eval in recent_evals:
                date = eval.timestamp[:10]  # YYYY-MM-DD
                if date not in scores_by_date:
                    scores_by_date[date] = []
                scores_by_date[date].append(eval.overall_score)
            
            # Calculate daily averages
            daily_averages = {
                date: statistics.mean(scores)
                for date, scores in scores_by_date.items()
            }
            
            # Identify patterns
            success_patterns = []
            failure_patterns = []
            
            for eval in recent_evals:
                if eval.success and eval.overall_score > 0.8:
                    success_patterns.append(self._extract_pattern(eval))
                elif not eval.success or eval.overall_score < 0.4:
                    failure_patterns.append(self._extract_pattern(eval))
            
            return {
                'tool_id': tool_id,
                'analysis_period_days': days_back,
                'total_evaluations': len(recent_evals),
                'success_rate': sum(1 for e in recent_evals if e.success) / len(recent_evals),
                'average_score': statistics.mean([e.overall_score for e in recent_evals]),
                'score_trend': list(daily_averages.values()),
                'best_performance_date': max(daily_averages.keys(), key=lambda k: daily_averages[k]),
                'worst_performance_date': min(daily_averages.keys(), key=lambda k: daily_averages[k]),
                'success_patterns': list(set(success_patterns)),
                'failure_patterns': list(set(failure_patterns)),
                'improvement_suggestions': self._generate_improvement_suggestions(recent_evals)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tool performance: {e}")
            return {'error': str(e)}
    
    def _calculate_evaluation_scores(self, tool_id: str, goal: str, 
                                   input_data: Dict[str, Any], output_data: Any,
                                   success: bool, execution_time_ms: int,
                                   user_feedback: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate evaluation scores for different criteria."""
        scores = {}
        
        # Base success score
        base_score = 1.0 if success else 0.0
        
        # Accuracy score
        if user_feedback and 'accuracy' in user_feedback:
            scores['accuracy'] = user_feedback['accuracy']
        else:
            scores['accuracy'] = base_score
        
        # Completeness score
        if user_feedback and 'completeness' in user_feedback:
            scores['completeness'] = user_feedback['completeness']
        else:
            # Estimate based on output length/content
            if output_data and isinstance(output_data, (str, dict, list)):
                if isinstance(output_data, str):
                    scores['completeness'] = min(1.0, len(output_data) / 100)
                else:
                    scores['completeness'] = base_score
            else:
                scores['completeness'] = base_score * 0.5
        
        # Relevance score
        if user_feedback and 'relevance' in user_feedback:
            scores['relevance'] = user_feedback['relevance']
        else:
            # Simple keyword matching
            goal_words = set(goal.lower().split())
            if isinstance(output_data, str):
                output_words = set(output_data.lower().split())
                overlap = len(goal_words.intersection(output_words))
                scores['relevance'] = min(1.0, overlap / max(len(goal_words), 1))
            else:
                scores['relevance'] = base_score
        
        # Efficiency score (based on execution time)
        expected_time = self._get_expected_execution_time(tool_id)
        if execution_time_ms <= expected_time * 1000:
            scores['efficiency'] = 1.0
        else:
            scores['efficiency'] = max(0.1, expected_time * 1000 / execution_time_ms)
        
        # User satisfaction score
        if user_feedback and 'satisfaction' in user_feedback:
            scores['user_satisfaction'] = user_feedback['satisfaction']
        else:
            scores['user_satisfaction'] = base_score
        
        # Overall score (weighted average)
        weights = {
            'accuracy': 0.3,
            'completeness': 0.2,
            'relevance': 0.2,
            'efficiency': 0.1,
            'user_satisfaction': 0.2
        }
        
        scores['overall'] = sum(scores[criterion] * weight 
                               for criterion, weight in weights.items())
        
        return scores
    
    def _update_tool_scorecard(self, evaluation: ToolEvaluation):
        """Update tool scorecard with new evaluation."""
        try:
            tool_id = evaluation.tool_id
            
            if tool_id not in self.scorecards:
                # Create new scorecard
                self.scorecards[tool_id] = ToolScorecard(
                    tool_id=tool_id,
                    tool_name=tool_id.replace('_', ' ').title(),
                    total_executions=0,
                    successful_executions=0,
                    success_rate=0.0,
                    average_execution_time=0.0,
                    average_accuracy=0.0,
                    average_completeness=0.0,
                    average_relevance=0.0,
                    average_efficiency=0.0,
                    average_user_satisfaction=0.0,
                    overall_performance_score=0.0,
                    performance_level=ToolPerformanceLevel.AVERAGE,
                    best_use_cases=[],
                    common_failures=[],
                    improvement_suggestions=[],
                    last_updated=datetime.now().isoformat(),
                    metadata={}
                )
            
            scorecard = self.scorecards[tool_id]
            
            # Update counts
            scorecard.total_executions += 1
            if evaluation.success:
                scorecard.successful_executions += 1
            
            # Update averages
            n = scorecard.total_executions
            scorecard.success_rate = scorecard.successful_executions / n
            
            # Update running averages
            scorecard.average_execution_time = self._update_average(
                scorecard.average_execution_time, evaluation.execution_time_ms, n
            )
            scorecard.average_accuracy = self._update_average(
                scorecard.average_accuracy, evaluation.accuracy_score, n
            )
            scorecard.average_completeness = self._update_average(
                scorecard.average_completeness, evaluation.completeness_score, n
            )
            scorecard.average_relevance = self._update_average(
                scorecard.average_relevance, evaluation.relevance_score, n
            )
            scorecard.average_efficiency = self._update_average(
                scorecard.average_efficiency, evaluation.efficiency_score, n
            )
            scorecard.average_user_satisfaction = self._update_average(
                scorecard.average_user_satisfaction, evaluation.user_satisfaction_score, n
            )
            scorecard.overall_performance_score = self._update_average(
                scorecard.overall_performance_score, evaluation.overall_score, n
            )
            
            # Update performance level
            scorecard.performance_level = self._determine_performance_level(
                scorecard.overall_performance_score
            )
            
            # Update use cases and failures
            if evaluation.success and evaluation.overall_score > 0.8:
                pattern = self._extract_pattern(evaluation)
                if pattern not in scorecard.best_use_cases:
                    scorecard.best_use_cases.append(pattern)
            elif not evaluation.success or evaluation.overall_score < 0.4:
                pattern = self._extract_pattern(evaluation)
                if pattern not in scorecard.common_failures:
                    scorecard.common_failures.append(pattern)
            
            scorecard.last_updated = datetime.now().isoformat()
            
            # Save scorecards
            self._save_scorecards()
            
        except Exception as e:
            logger.error(f"Error updating tool scorecard: {e}")
    
    def _update_task_profile(self, task_type: str, evaluation: ToolEvaluation):
        """Update task type profile with new evaluation."""
        try:
            if task_type not in self.task_profiles:
                self.task_profiles[task_type] = TaskTypeProfile(
                    task_type=task_type,
                    tool_rankings=[],
                    recommended_tools=[],
                    fallback_tools=[],
                    success_patterns=[],
                    failure_patterns=[],
                    last_updated=datetime.now().isoformat()
                )
            
            profile = self.task_profiles[task_type]
            
            # Update tool rankings
            tool_scores = dict(profile.tool_rankings)
            current_score = tool_scores.get(evaluation.tool_id, 0.0)
            
            # Simple moving average update
            if evaluation.tool_id in tool_scores:
                new_score = (current_score + evaluation.overall_score) / 2
            else:
                new_score = evaluation.overall_score
            
            tool_scores[evaluation.tool_id] = new_score
            
            # Sort and update rankings
            profile.tool_rankings = sorted(tool_scores.items(), 
                                         key=lambda x: x[1], reverse=True)
            
            # Update recommended tools (top performers)
            profile.recommended_tools = [tool_id for tool_id, score in profile.tool_rankings[:5]
                                       if score > 0.6]
            
            # Update fallback tools (decent performers)
            profile.fallback_tools = [tool_id for tool_id, score in profile.tool_rankings[5:10]
                                    if score > 0.4]
            
            # Update patterns
            pattern = self._extract_pattern(evaluation)
            if evaluation.success and evaluation.overall_score > 0.7:
                if pattern not in profile.success_patterns:
                    profile.success_patterns.append(pattern)
            elif not evaluation.success or evaluation.overall_score < 0.4:
                if pattern not in profile.failure_patterns:
                    profile.failure_patterns.append(pattern)
            
            profile.last_updated = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error updating task profile: {e}")
    
    def _classify_task_type(self, goal: str) -> str:
        """Classify task type from goal."""
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ['calculate', 'compute', 'math']):
            return 'computation'
        elif any(word in goal_lower for word in ['analyze', 'data', 'statistics']):
            return 'data_analysis'
        elif any(word in goal_lower for word in ['chart', 'graph', 'visualize', 'table']):
            return 'visualization'
        elif any(word in goal_lower for word in ['search', 'find', 'lookup']):
            return 'search'
        elif any(word in goal_lower for word in ['web', 'internet', 'online']):
            return 'web_interaction'
        else:
            return 'general'
    
    def _extract_pattern(self, evaluation: ToolEvaluation) -> str:
        """Extract a pattern from an evaluation."""
        # Simple pattern extraction based on goal keywords
        goal_words = evaluation.goal.lower().split()[:3]
        return ' '.join(goal_words)
    
    def _get_expected_execution_time(self, tool_id: str) -> float:
        """Get expected execution time for a tool."""
        expected_times = {
            'python_interpreter': 5.0,
            'table_generator': 2.0,
            'multimodal_query': 3.0,
            'web_search': 4.0
        }
        return expected_times.get(tool_id, 3.0)
    
    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average."""
        return ((current_avg * (count - 1)) + new_value) / count
    
    def _determine_performance_level(self, score: float) -> ToolPerformanceLevel:
        """Determine performance level from score."""
        if score >= 0.9:
            return ToolPerformanceLevel.EXCELLENT
        elif score >= 0.7:
            return ToolPerformanceLevel.GOOD
        elif score >= 0.5:
            return ToolPerformanceLevel.AVERAGE
        elif score >= 0.3:
            return ToolPerformanceLevel.POOR
        else:
            return ToolPerformanceLevel.FAILING
    
    def _generate_improvement_suggestions(self, evaluations: List[ToolEvaluation]) -> List[str]:
        """Generate improvement suggestions based on evaluations."""
        suggestions = []
        
        # Analyze common failure patterns
        failed_evals = [e for e in evaluations if not e.success or e.overall_score < 0.5]
        
        if len(failed_evals) > len(evaluations) * 0.3:  # More than 30% failures
            suggestions.append("High failure rate - review input validation and error handling")
        
        # Check execution time issues
        slow_evals = [e for e in evaluations if e.execution_time_ms > 10000]  # > 10 seconds
        if len(slow_evals) > len(evaluations) * 0.2:  # More than 20% slow
            suggestions.append("Performance optimization needed - execution times are high")
        
        # Check accuracy issues
        low_accuracy = [e for e in evaluations if e.accuracy_score < 0.6]
        if len(low_accuracy) > len(evaluations) * 0.3:
            suggestions.append("Accuracy improvements needed - review output quality")
        
        return suggestions
    
    def _load_scorecards(self):
        """Load tool scorecards from storage."""
        try:
            if self.scorecard_file.exists():
                with open(self.scorecard_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load scorecards
                for scorecard_data in data.get('scorecards', []):
                    scorecard = ToolScorecard(
                        tool_id=scorecard_data['tool_id'],
                        tool_name=scorecard_data['tool_name'],
                        total_executions=scorecard_data['total_executions'],
                        successful_executions=scorecard_data['successful_executions'],
                        success_rate=scorecard_data['success_rate'],
                        average_execution_time=scorecard_data['average_execution_time'],
                        average_accuracy=scorecard_data['average_accuracy'],
                        average_completeness=scorecard_data['average_completeness'],
                        average_relevance=scorecard_data['average_relevance'],
                        average_efficiency=scorecard_data['average_efficiency'],
                        average_user_satisfaction=scorecard_data['average_user_satisfaction'],
                        overall_performance_score=scorecard_data['overall_performance_score'],
                        performance_level=ToolPerformanceLevel(scorecard_data['performance_level']),
                        best_use_cases=scorecard_data['best_use_cases'],
                        common_failures=scorecard_data['common_failures'],
                        improvement_suggestions=scorecard_data['improvement_suggestions'],
                        last_updated=scorecard_data['last_updated'],
                        metadata=scorecard_data.get('metadata', {})
                    )
                    self.scorecards[scorecard.tool_id] = scorecard
                
                # Load task profiles
                for profile_data in data.get('task_profiles', []):
                    profile = TaskTypeProfile(
                        task_type=profile_data['task_type'],
                        tool_rankings=profile_data['tool_rankings'],
                        recommended_tools=profile_data['recommended_tools'],
                        fallback_tools=profile_data['fallback_tools'],
                        success_patterns=profile_data['success_patterns'],
                        failure_patterns=profile_data['failure_patterns'],
                        last_updated=profile_data['last_updated']
                    )
                    self.task_profiles[profile.task_type] = profile
                
                logger.info(f"Loaded {len(self.scorecards)} scorecards and {len(self.task_profiles)} task profiles")
            
        except Exception as e:
            logger.error(f"Error loading scorecards: {e}")
    
    def _save_scorecards(self):
        """Save tool scorecards to storage."""
        try:
            # Convert scorecards to serializable format
            scorecards_data = []
            for scorecard in self.scorecards.values():
                scorecard_dict = asdict(scorecard)
                scorecard_dict['performance_level'] = scorecard.performance_level.value
                scorecards_data.append(scorecard_dict)
            
            # Convert task profiles to serializable format
            profiles_data = []
            for profile in self.task_profiles.values():
                profiles_data.append(asdict(profile))
            
            data = {
                'scorecards': scorecards_data,
                'task_profiles': profiles_data,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            # Ensure directory exists
            self.scorecard_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.scorecard_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.scorecards)} scorecards")
            
        except Exception as e:
            logger.error(f"Error saving scorecards: {e}")

# Global tool evaluator instance
_tool_evaluator = None

def get_tool_evaluator(scorecard_file: str = "tool_scorecard.json") -> ToolEvaluator:
    """Get or create a global tool evaluator instance."""
    global _tool_evaluator
    
    if _tool_evaluator is None:
        _tool_evaluator = ToolEvaluator(scorecard_file=scorecard_file)
    
    return _tool_evaluator
