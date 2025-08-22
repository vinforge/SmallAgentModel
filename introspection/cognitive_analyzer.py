"""
Cognitive Analyzer for SAM
==========================

Analyzes SAM's cognitive processes from introspection logs to provide
insights into reasoning patterns, performance bottlenecks, and decision-making.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter

from .introspection_logger import CognitiveEvent, EventType


@dataclass
class CognitiveInsight:
    """Represents an insight about SAM's cognitive processes."""
    insight_type: str
    title: str
    description: str
    confidence: float  # 0-1
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


@dataclass
class ReasoningPattern:
    """Represents a pattern in SAM's reasoning."""
    pattern_id: str
    description: str
    frequency: int
    avg_duration_ms: float
    success_rate: float
    complexity_score: float
    examples: List[str]  # Event IDs


@dataclass
class PerformanceBottleneck:
    """Represents a performance bottleneck."""
    component: str
    operation: str
    avg_duration_ms: float
    frequency: int
    impact_score: float
    suggestions: List[str]


class CognitiveAnalyzer:
    """
    Analyzes SAM's cognitive processes to extract insights and patterns.
    
    Features:
    - Reasoning pattern detection
    - Performance bottleneck identification
    - Decision-making analysis
    - Cognitive load assessment
    - Learning progress tracking
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the cognitive analyzer.
        
        Args:
            log_file: Path to introspection log file
        """
        self.log_file = log_file
        self.events: List[CognitiveEvent] = []
        self.insights: List[CognitiveInsight] = []
        
        if log_file:
            self.load_events_from_file(log_file)
    
    def load_events_from_file(self, log_file: str):
        """Load events from a JSONL log file."""
        self.events = []
        log_path = Path(log_file)
        
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        event_data = json.loads(line)
                        event = CognitiveEvent.from_dict(event_data)
                        self.events.append(event)
                    except Exception as e:
                        print(f"Warning: Failed to parse event: {e}")
        
        print(f"Loaded {len(self.events)} events from {log_file}")
    
    def analyze_reasoning_patterns(self) -> List[ReasoningPattern]:
        """Analyze reasoning patterns in the events."""
        patterns = []
        
        # Group reasoning sequences
        reasoning_sequences = self._extract_reasoning_sequences()
        
        # Analyze each sequence type
        sequence_groups = defaultdict(list)
        for seq in reasoning_sequences:
            # Group by similar patterns (simplified)
            pattern_key = self._get_pattern_key(seq)
            sequence_groups[pattern_key].append(seq)
        
        # Create pattern objects
        for pattern_key, sequences in sequence_groups.items():
            if len(sequences) >= 2:  # Only patterns that occur multiple times
                durations = [seq['duration_ms'] for seq in sequences if seq['duration_ms']]
                success_rates = [seq['success'] for seq in sequences]
                
                pattern = ReasoningPattern(
                    pattern_id=pattern_key,
                    description=f"Reasoning pattern: {pattern_key}",
                    frequency=len(sequences),
                    avg_duration_ms=statistics.mean(durations) if durations else 0,
                    success_rate=sum(success_rates) / len(success_rates),
                    complexity_score=self._calculate_complexity_score(sequences),
                    examples=[seq['start_event_id'] for seq in sequences[:3]]
                )
                patterns.append(pattern)
        
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)
    
    def identify_performance_bottlenecks(self) -> List[PerformanceBottleneck]:
        """Identify performance bottlenecks in SAM's operations."""
        bottlenecks = []
        
        # Group events by component and operation
        operation_stats = defaultdict(list)
        
        for event in self.events:
            if event.duration_ms and event.component:
                key = (event.component, event.event_type.value)
                operation_stats[key].append(event.duration_ms)
        
        # Analyze each operation
        for (component, operation), durations in operation_stats.items():
            if len(durations) >= 5:  # Need sufficient data
                avg_duration = statistics.mean(durations)
                frequency = len(durations)
                
                # Calculate impact score (frequency * avg_duration)
                impact_score = frequency * avg_duration / 1000  # Convert to seconds
                
                # Consider it a bottleneck if it's slow or frequent
                if avg_duration > 1000 or frequency > 20:  # 1 second or 20+ occurrences
                    suggestions = self._generate_performance_suggestions(
                        component, operation, avg_duration, frequency
                    )
                    
                    bottleneck = PerformanceBottleneck(
                        component=component,
                        operation=operation,
                        avg_duration_ms=avg_duration,
                        frequency=frequency,
                        impact_score=impact_score,
                        suggestions=suggestions
                    )
                    bottlenecks.append(bottleneck)
        
        return sorted(bottlenecks, key=lambda b: b.impact_score, reverse=True)
    
    def analyze_decision_making(self) -> Dict[str, Any]:
        """Analyze SAM's decision-making patterns."""
        decision_events = [e for e in self.events if e.event_type == EventType.DECISION_POINT]
        
        if not decision_events:
            return {"message": "No decision points found in logs"}
        
        # Analyze confidence patterns
        confidences = [e.confidence_score for e in decision_events if e.confidence_score]
        
        # Analyze decision outcomes (simplified)
        decision_contexts = Counter()
        for event in decision_events:
            context_key = event.details.get('context_type', 'unknown')
            decision_contexts[context_key] += 1
        
        return {
            "total_decisions": len(decision_events),
            "avg_confidence": statistics.mean(confidences) if confidences else None,
            "confidence_std": statistics.stdev(confidences) if len(confidences) > 1 else None,
            "decision_contexts": dict(decision_contexts),
            "high_confidence_decisions": len([c for c in confidences if c > 0.8]),
            "low_confidence_decisions": len([c for c in confidences if c < 0.5])
        }
    
    def assess_cognitive_load(self) -> Dict[str, Any]:
        """Assess SAM's cognitive load over time."""
        if not self.events:
            return {"message": "No events to analyze"}
        
        # Group events by time windows (e.g., 1-minute intervals)
        time_windows = defaultdict(list)
        start_time = min(e.timestamp for e in self.events)
        
        for event in self.events:
            window_key = int((event.timestamp - start_time).total_seconds() // 60)
            time_windows[window_key].append(event)
        
        # Calculate load metrics for each window
        load_metrics = []
        for window, events in time_windows.items():
            complexity_scores = [e.complexity_score for e in events if e.complexity_score]
            
            load_metric = {
                "window": window,
                "event_count": len(events),
                "avg_complexity": statistics.mean(complexity_scores) if complexity_scores else 0,
                "reasoning_events": len([e for e in events if e.event_type in [
                    EventType.REASONING_START, EventType.REASONING_STEP, EventType.REASONING_END
                ]]),
                "error_events": len([e for e in events if e.event_type == EventType.ERROR_OCCURRED])
            }
            load_metrics.append(load_metric)
        
        # Overall assessment
        total_events = len(self.events)
        total_errors = len([e for e in self.events if e.event_type == EventType.ERROR_OCCURRED])
        error_rate = total_errors / total_events if total_events > 0 else 0
        
        return {
            "total_events": total_events,
            "error_rate": error_rate,
            "time_windows": len(time_windows),
            "load_by_window": load_metrics,
            "peak_load_window": max(load_metrics, key=lambda x: x["event_count"])["window"] if load_metrics else None
        }
    
    def generate_insights(self) -> List[CognitiveInsight]:
        """Generate comprehensive insights about SAM's cognitive processes."""
        insights = []
        
        # Reasoning pattern insights
        patterns = self.analyze_reasoning_patterns()
        if patterns:
            most_common = patterns[0]
            insights.append(CognitiveInsight(
                insight_type="reasoning_pattern",
                title="Most Common Reasoning Pattern",
                description=f"Pattern '{most_common.pattern_id}' occurs {most_common.frequency} times with {most_common.success_rate:.1%} success rate",
                confidence=0.8,
                supporting_data={"pattern": most_common},
                recommendations=[
                    "Monitor this pattern for optimization opportunities",
                    "Consider caching results for repeated patterns"
                ],
                timestamp=datetime.now()
            ))
        
        # Performance insights
        bottlenecks = self.identify_performance_bottlenecks()
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            insights.append(CognitiveInsight(
                insight_type="performance",
                title="Primary Performance Bottleneck",
                description=f"{top_bottleneck.component} {top_bottleneck.operation} takes {top_bottleneck.avg_duration_ms:.0f}ms on average",
                confidence=0.9,
                supporting_data={"bottleneck": top_bottleneck},
                recommendations=top_bottleneck.suggestions,
                timestamp=datetime.now()
            ))
        
        # Decision-making insights
        decision_analysis = self.analyze_decision_making()
        if "avg_confidence" in decision_analysis and decision_analysis["avg_confidence"]:
            avg_conf = decision_analysis["avg_confidence"]
            insights.append(CognitiveInsight(
                insight_type="decision_making",
                title="Decision Confidence Analysis",
                description=f"Average decision confidence is {avg_conf:.1%}",
                confidence=0.7,
                supporting_data=decision_analysis,
                recommendations=[
                    "High confidence decisions could be automated",
                    "Low confidence decisions need more context"
                ] if avg_conf > 0.7 else [
                    "Consider improving decision support systems",
                    "Provide more context for decision-making"
                ],
                timestamp=datetime.now()
            ))
        
        # Cognitive load insights
        load_analysis = self.assess_cognitive_load()
        if "error_rate" in load_analysis:
            error_rate = load_analysis["error_rate"]
            insights.append(CognitiveInsight(
                insight_type="cognitive_load",
                title="Error Rate Analysis",
                description=f"Current error rate is {error_rate:.1%}",
                confidence=0.8,
                supporting_data=load_analysis,
                recommendations=[
                    "Error rate is acceptable" if error_rate < 0.05 else "Consider error reduction strategies",
                    "Monitor error patterns for systematic issues"
                ],
                timestamp=datetime.now()
            ))
        
        self.insights = insights
        return insights
    
    def _extract_reasoning_sequences(self) -> List[Dict[str, Any]]:
        """Extract reasoning sequences from events."""
        sequences = []
        current_sequence = None
        
        for event in self.events:
            if event.event_type == EventType.REASONING_START:
                current_sequence = {
                    "start_event_id": event.event_id,
                    "start_time": event.timestamp,
                    "steps": [],
                    "success": True,
                    "duration_ms": None
                }
            
            elif event.event_type == EventType.REASONING_STEP and current_sequence:
                current_sequence["steps"].append(event)
            
            elif event.event_type == EventType.REASONING_END and current_sequence:
                current_sequence["end_time"] = event.timestamp
                current_sequence["duration_ms"] = (
                    event.timestamp - current_sequence["start_time"]
                ).total_seconds() * 1000
                sequences.append(current_sequence)
                current_sequence = None
            
            elif event.event_type == EventType.ERROR_OCCURRED and current_sequence:
                current_sequence["success"] = False
        
        return sequences
    
    def _get_pattern_key(self, sequence: Dict[str, Any]) -> str:
        """Generate a pattern key for a reasoning sequence."""
        # Simplified pattern key based on number of steps and types
        step_count = len(sequence["steps"])
        step_types = [step.details.get("step_type", "unknown") for step in sequence["steps"]]
        return f"steps_{step_count}_types_{'_'.join(set(step_types))}"
    
    def _calculate_complexity_score(self, sequences: List[Dict[str, Any]]) -> float:
        """Calculate complexity score for a pattern."""
        avg_steps = statistics.mean([len(seq["steps"]) for seq in sequences])
        return min(avg_steps / 10, 1.0)  # Normalize to 0-1
    
    def _generate_performance_suggestions(self, component: str, operation: str, 
                                        avg_duration: float, frequency: int) -> List[str]:
        """Generate performance improvement suggestions."""
        suggestions = []
        
        if avg_duration > 5000:  # 5 seconds
            suggestions.append("Consider breaking down this operation into smaller steps")
            suggestions.append("Implement caching for repeated operations")
        
        if frequency > 50:
            suggestions.append("High frequency operation - consider optimization")
            suggestions.append("Batch similar operations together")
        
        if component == "memory_manager":
            suggestions.append("Consider memory indexing improvements")
        elif component == "model_interface":
            suggestions.append("Consider model optimization or quantization")
        elif component == "tool_executor":
            suggestions.append("Consider tool result caching")
        
        return suggestions or ["Monitor this operation for optimization opportunities"]
