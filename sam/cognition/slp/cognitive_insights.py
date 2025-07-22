"""
SLP Cognitive Insights Generator
===============================

Advanced cognitive insights generation for the SLP system.
Identifies learning patterns, generates automation insights, and tracks cognitive evolution.

Phase 1B.5 - Cognitive Insights Generation (preserving 100% of existing functionality)
"""

import logging
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics

from .latent_program_store import LatentProgramStore
from .program_analyzer import ProgramAnalyzer

logger = logging.getLogger(__name__)


class CognitiveInsightsGenerator:
    """
    Advanced cognitive insights generation for SLP system.
    
    Identifies learning patterns, automation opportunities, and cognitive evolution
    while preserving 100% of existing SLP functionality.
    """
    
    def __init__(self, store: Optional[LatentProgramStore] = None,
                 analyzer: Optional[ProgramAnalyzer] = None):
        """Initialize the cognitive insights generator."""
        self.store = store or LatentProgramStore()
        self.analyzer = analyzer or ProgramAnalyzer(self.store)
        
        # Insight generation configuration
        self.learning_velocity_window_days = 7
        self.pattern_significance_threshold = 3
        self.automation_confidence_threshold = 0.7
        
        # Caching for expensive operations
        self.insights_cache = {}
        self.cache_ttl = 600  # 10 minutes
        self.last_cache_update = {}
        
        logger.info("Cognitive Insights Generator initialized")
    
    def identify_most_successful_program_types(self, time_window_days: int = 30) -> Dict[str, Any]:
        """
        Identify the most successful types of programs based on various metrics.
        
        Args:
            time_window_days: Time window for analysis
            
        Returns:
            Dictionary containing successful program type analysis
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            # Get program performance data
            with sqlite3.connect(self.store.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        p.id,
                        p.signature_data,
                        p.usage_count,
                        p.success_rate,
                        p.confidence_score,
                        AVG(a.quality_score) as avg_quality,
                        AVG(a.efficiency_gain) as avg_efficiency,
                        COUNT(a.id) as recent_executions,
                        AVG(a.user_feedback) as avg_feedback
                    FROM latent_programs p
                    LEFT JOIN program_analytics_enhanced a ON p.id = a.program_id
                    WHERE p.is_active = 1 AND (a.execution_timestamp > ? OR a.execution_timestamp IS NULL)
                    GROUP BY p.id
                    HAVING recent_executions > 0
                    ORDER BY avg_quality DESC, avg_efficiency DESC
                """, (cutoff_date.isoformat(),))
                
                program_data = cursor.fetchall()
            
            if not program_data:
                return {
                    'analysis_period_days': time_window_days,
                    'successful_types': [],
                    'insights': ['No program execution data found for the specified period']
                }
            
            # Analyze program types by signature characteristics
            type_analysis = self._analyze_program_types_by_signature(program_data)
            
            # Identify success patterns
            success_patterns = self._identify_success_patterns(type_analysis)
            
            # Generate insights
            insights = self._generate_success_type_insights(success_patterns, time_window_days)
            
            return {
                'analysis_period_days': time_window_days,
                'total_programs_analyzed': len(program_data),
                'successful_types': success_patterns,
                'type_analysis': type_analysis,
                'insights': insights,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to identify successful program types: {e}")
            return {}
    
    def detect_automation_opportunities(self, user_patterns: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect automation opportunities based on usage patterns and success metrics.
        
        Args:
            user_patterns: Optional user behavior patterns for analysis
            
        Returns:
            List of automation opportunities with detailed analysis
        """
        try:
            opportunities = []
            
            # Analyze query patterns for automation potential
            query_patterns = self._analyze_query_automation_patterns()
            
            # Analyze workflow patterns
            workflow_patterns = self._analyze_workflow_automation_patterns()
            
            # Analyze user-specific patterns if provided
            user_opportunities = []
            if user_patterns:
                user_opportunities = self._analyze_user_specific_automation(user_patterns)
            
            # Combine and score opportunities
            all_opportunities = query_patterns + workflow_patterns + user_opportunities
            
            # Filter and rank by confidence and impact
            filtered_opportunities = [
                opp for opp in all_opportunities 
                if opp.get('confidence_score', 0) >= self.automation_confidence_threshold
            ]
            
            # Sort by potential impact
            filtered_opportunities.sort(
                key=lambda x: x.get('potential_impact_score', 0), 
                reverse=True
            )
            
            # Add detailed analysis for top opportunities
            for opp in filtered_opportunities[:10]:  # Top 10 opportunities
                opp['detailed_analysis'] = self._generate_opportunity_analysis(opp)
            
            logger.info(f"Detected {len(filtered_opportunities)} automation opportunities")
            return filtered_opportunities
            
        except Exception as e:
            logger.error(f"Failed to detect automation opportunities: {e}")
            return []
    
    def generate_learning_insights(self, time_window_days: int = 30) -> Dict[str, Any]:
        """
        Generate insights about the learning progress and patterns.
        
        Args:
            time_window_days: Time window for learning analysis
            
        Returns:
            Dictionary containing learning insights and progress metrics
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            # Analyze pattern discovery trends
            discovery_trends = self._analyze_pattern_discovery_trends(cutoff_date)
            
            # Analyze learning velocity
            learning_velocity = self._calculate_learning_velocity(cutoff_date)
            
            # Analyze knowledge evolution
            knowledge_evolution = self._analyze_knowledge_evolution(cutoff_date)
            
            # Analyze adaptation patterns
            adaptation_patterns = self._analyze_adaptation_patterns(cutoff_date)
            
            # Generate comprehensive learning insights
            learning_insights = self._generate_learning_insights(
                discovery_trends, learning_velocity, knowledge_evolution, adaptation_patterns
            )
            
            return {
                'analysis_period_days': time_window_days,
                'discovery_trends': discovery_trends,
                'learning_velocity': learning_velocity,
                'knowledge_evolution': knowledge_evolution,
                'adaptation_patterns': adaptation_patterns,
                'learning_insights': learning_insights,
                'learning_health_score': self._calculate_learning_health_score(
                    discovery_trends, learning_velocity, knowledge_evolution
                ),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate learning insights: {e}")
            return {}
    
    def track_cognitive_evolution(self, baseline_period_days: int = 30, 
                                current_period_days: int = 7) -> Dict[str, Any]:
        """
        Track cognitive evolution by comparing different time periods.
        
        Args:
            baseline_period_days: Days for baseline period
            current_period_days: Days for current period comparison
            
        Returns:
            Dictionary containing cognitive evolution analysis
        """
        try:
            current_end = datetime.utcnow()
            current_start = current_end - timedelta(days=current_period_days)
            baseline_end = current_start
            baseline_start = baseline_end - timedelta(days=baseline_period_days)
            
            # Analyze baseline period
            baseline_metrics = self._get_cognitive_metrics(baseline_start, baseline_end)
            
            # Analyze current period
            current_metrics = self._get_cognitive_metrics(current_start, current_end)
            
            # Calculate evolution metrics
            evolution_analysis = self._calculate_cognitive_evolution(baseline_metrics, current_metrics)
            
            # Generate evolution insights
            evolution_insights = self._generate_evolution_insights(evolution_analysis)
            
            return {
                'baseline_period': {
                    'start': baseline_start.isoformat(),
                    'end': baseline_end.isoformat(),
                    'metrics': baseline_metrics
                },
                'current_period': {
                    'start': current_start.isoformat(),
                    'end': current_end.isoformat(),
                    'metrics': current_metrics
                },
                'evolution_analysis': evolution_analysis,
                'evolution_insights': evolution_insights,
                'cognitive_growth_score': evolution_analysis.get('overall_growth_score', 0),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to track cognitive evolution: {e}")
            return {}
    
    def get_comprehensive_cognitive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive cognitive insights report.
        
        Returns:
            Dictionary containing comprehensive cognitive analysis
        """
        try:
            # Generate all major insights
            successful_types = self.identify_most_successful_program_types()
            automation_opportunities = self.detect_automation_opportunities()
            learning_insights = self.generate_learning_insights()
            cognitive_evolution = self.track_cognitive_evolution()
            
            # Get current system status
            system_status = self._get_cognitive_system_status()
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                successful_types, automation_opportunities, learning_insights, cognitive_evolution
            )
            
            return {
                'executive_summary': executive_summary,
                'system_status': system_status,
                'successful_program_types': successful_types,
                'automation_opportunities': automation_opportunities[:5],  # Top 5
                'learning_insights': learning_insights,
                'cognitive_evolution': cognitive_evolution,
                'recommendations': self._generate_comprehensive_recommendations(
                    successful_types, automation_opportunities, learning_insights, cognitive_evolution
                ),
                'report_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive cognitive report: {e}")
            return {}
    
    def predict_automation_potential(self, query_pattern: str, 
                                   context_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict automation potential for a specific query and context pattern.
        
        Args:
            query_pattern: Query pattern to analyze
            context_pattern: Context pattern characteristics
            
        Returns:
            Dictionary containing automation potential prediction
        """
        try:
            # Analyze similar historical patterns
            similar_patterns = self._find_similar_historical_patterns(query_pattern, context_pattern)
            
            # Calculate automation potential score
            automation_score = self._calculate_automation_potential_score(similar_patterns, context_pattern)
            
            # Generate prediction insights
            prediction_insights = self._generate_prediction_insights(automation_score, similar_patterns)
            
            return {
                'query_pattern': query_pattern,
                'context_pattern': context_pattern,
                'automation_potential_score': automation_score,
                'confidence_level': self._calculate_prediction_confidence(similar_patterns),
                'similar_patterns_found': len(similar_patterns),
                'prediction_insights': prediction_insights,
                'recommended_actions': self._generate_automation_recommendations(automation_score),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict automation potential: {e}")
            return {}

    # Helper Methods for Cognitive Analysis (Phase 1B.5 - preserving 100% of existing functionality)

    def _analyze_program_types_by_signature(self, program_data: List[Tuple]) -> Dict[str, Any]:
        """Analyze program types based on signature characteristics."""
        try:
            type_groups = defaultdict(list)

            for row in program_data:
                program_id, signature_data, usage_count, success_rate, confidence_score, \
                avg_quality, avg_efficiency, recent_executions, avg_feedback = row

                try:
                    signature = json.loads(signature_data) if signature_data else {}
                except json.JSONDecodeError:
                    signature = {}

                # Group by primary intent and complexity
                primary_intent = signature.get('primary_intent', 'unknown')
                complexity = signature.get('complexity_level', 'unknown')
                type_key = f"{primary_intent}_{complexity}"

                type_groups[type_key].append({
                    'program_id': program_id,
                    'usage_count': usage_count or 0,
                    'success_rate': success_rate or 0,
                    'confidence_score': confidence_score or 0,
                    'avg_quality': avg_quality or 0,
                    'avg_efficiency': avg_efficiency or 0,
                    'recent_executions': recent_executions or 0,
                    'avg_feedback': avg_feedback or 0,
                    'signature': signature
                })

            # Calculate type-level metrics
            type_analysis = {}
            for type_key, programs in type_groups.items():
                if len(programs) >= self.pattern_significance_threshold:
                    type_analysis[type_key] = {
                        'program_count': len(programs),
                        'avg_usage_count': statistics.mean(p['usage_count'] for p in programs),
                        'avg_success_rate': statistics.mean(p['success_rate'] for p in programs),
                        'avg_quality_score': statistics.mean(p['avg_quality'] for p in programs),
                        'avg_efficiency_gain': statistics.mean(p['avg_efficiency'] for p in programs),
                        'total_recent_executions': sum(p['recent_executions'] for p in programs),
                        'avg_feedback_score': statistics.mean(p['avg_feedback'] for p in programs if p['avg_feedback'] != 0),
                        'programs': programs
                    }

            return type_analysis

        except Exception as e:
            logger.error(f"Failed to analyze program types by signature: {e}")
            return {}

    def _identify_success_patterns(self, type_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify success patterns from type analysis."""
        try:
            success_patterns = []

            for type_key, metrics in type_analysis.items():
                # Calculate success score
                success_score = self._calculate_type_success_score(metrics)

                if success_score >= 0.6:  # Threshold for successful types
                    pattern = {
                        'type_identifier': type_key,
                        'success_score': round(success_score, 3),
                        'program_count': metrics['program_count'],
                        'avg_quality_score': round(metrics['avg_quality_score'], 3),
                        'avg_efficiency_gain': round(metrics['avg_efficiency_gain'], 2),
                        'avg_success_rate': round(metrics['avg_success_rate'], 3),
                        'total_executions': metrics['total_recent_executions'],
                        'characteristics': self._extract_type_characteristics(type_key, metrics),
                        'success_factors': self._identify_success_factors(metrics)
                    }
                    success_patterns.append(pattern)

            # Sort by success score
            success_patterns.sort(key=lambda x: x['success_score'], reverse=True)

            return success_patterns

        except Exception as e:
            logger.error(f"Failed to identify success patterns: {e}")
            return []

    def _calculate_type_success_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate success score for a program type."""
        try:
            # Weighted combination of success factors
            quality_factor = min(1.0, metrics.get('avg_quality_score', 0))
            efficiency_factor = min(1.0, metrics.get('avg_efficiency_gain', 0) / 50.0)  # Normalize to 50% max
            success_rate_factor = metrics.get('avg_success_rate', 0)
            usage_factor = min(1.0, metrics.get('avg_usage_count', 0) / 10.0)  # Normalize to 10 uses
            feedback_factor = (metrics.get('avg_feedback_score', 0) + 1) / 2.0 if metrics.get('avg_feedback_score', 0) != 0 else 0.5

            # Weighted average
            weights = [0.3, 0.25, 0.25, 0.1, 0.1]
            factors = [quality_factor, efficiency_factor, success_rate_factor, usage_factor, feedback_factor]

            success_score = sum(factor * weight for factor, weight in zip(factors, weights))
            return max(0.0, min(1.0, success_score))

        except Exception as e:
            logger.error(f"Failed to calculate type success score: {e}")
            return 0.0

    def _extract_type_characteristics(self, type_key: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract characteristics of a program type."""
        try:
            intent, complexity = type_key.split('_', 1) if '_' in type_key else (type_key, 'unknown')

            # Analyze common signature elements
            programs = metrics.get('programs', [])
            if programs:
                signatures = [p['signature'] for p in programs if p['signature']]

                # Find common document types
                doc_types = []
                for sig in signatures:
                    doc_types.extend(sig.get('document_types', []))
                common_doc_types = [item for item, count in Counter(doc_types).most_common(3)]

                # Find common content domains
                domains = []
                for sig in signatures:
                    domains.extend(sig.get('content_domains', []))
                common_domains = [item for item, count in Counter(domains).most_common(3)]

                return {
                    'primary_intent': intent,
                    'complexity_level': complexity,
                    'common_document_types': common_doc_types,
                    'common_content_domains': common_domains,
                    'avg_program_age_days': self._calculate_avg_program_age(programs),
                    'consistency_score': self._calculate_type_consistency(programs)
                }

            return {
                'primary_intent': intent,
                'complexity_level': complexity,
                'common_document_types': [],
                'common_content_domains': [],
                'avg_program_age_days': 0,
                'consistency_score': 0
            }

        except Exception as e:
            logger.error(f"Failed to extract type characteristics: {e}")
            return {}

    def _identify_success_factors(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify key success factors for a program type."""
        try:
            factors = []

            if metrics.get('avg_quality_score', 0) >= 0.8:
                factors.append("High quality responses")

            if metrics.get('avg_efficiency_gain', 0) >= 20:
                factors.append("Significant efficiency improvements")

            if metrics.get('avg_success_rate', 0) >= 0.9:
                factors.append("Excellent reliability")

            if metrics.get('avg_usage_count', 0) >= 5:
                factors.append("Frequent usage pattern")

            if metrics.get('avg_feedback_score', 0) > 0.5:
                factors.append("Positive user feedback")

            if metrics.get('program_count', 0) >= 5:
                factors.append("Multiple successful implementations")

            return factors if factors else ["Moderate performance across metrics"]

        except Exception as e:
            logger.error(f"Failed to identify success factors: {e}")
            return []

    def _analyze_query_automation_patterns(self) -> List[Dict[str, Any]]:
        """Analyze query patterns for automation opportunities."""
        try:
            opportunities = []

            # Get frequent query patterns from analytics
            with sqlite3.connect(self.store.db_path) as conn:
                cursor = conn.execute("""
                    SELECT
                        query_type,
                        COUNT(*) as frequency,
                        AVG(execution_time_ms) as avg_execution_time,
                        AVG(quality_score) as avg_quality,
                        AVG(efficiency_gain) as avg_efficiency,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
                    FROM program_analytics_enhanced
                    WHERE execution_timestamp > datetime('now', '-30 days')
                    GROUP BY query_type
                    HAVING frequency >= ?
                    ORDER BY frequency DESC
                """, (self.pattern_significance_threshold,))

                patterns = cursor.fetchall()

            for row in patterns:
                query_type, frequency, avg_time, avg_quality, avg_efficiency, success_rate = row

                # Calculate automation potential
                potential_score = self._calculate_query_automation_potential(
                    frequency, avg_quality, success_rate, avg_efficiency
                )

                if potential_score >= self.automation_confidence_threshold:
                    opportunity = {
                        'type': 'query_automation',
                        'pattern_identifier': query_type,
                        'frequency': frequency,
                        'confidence_score': round(potential_score, 3),
                        'potential_impact_score': round(frequency * avg_efficiency / 100, 2),
                        'avg_quality_score': round(avg_quality or 0, 3),
                        'success_rate': round(success_rate or 0, 3),
                        'avg_efficiency_gain': round(avg_efficiency or 0, 2),
                        'estimated_time_savings_ms': round(frequency * (avg_efficiency or 0) * (avg_time or 0) / 100, 2)
                    }
                    opportunities.append(opportunity)

            return opportunities

        except Exception as e:
            logger.error(f"Failed to analyze query automation patterns: {e}")
            return []

    def _calculate_query_automation_potential(self, frequency: int, avg_quality: float,
                                           success_rate: float, avg_efficiency: float) -> float:
        """Calculate automation potential for a query pattern."""
        try:
            # Normalize factors
            frequency_factor = min(1.0, frequency / 20.0)  # Normalize to 20 occurrences
            quality_factor = min(1.0, avg_quality or 0)
            success_factor = success_rate or 0
            efficiency_factor = min(1.0, (avg_efficiency or 0) / 30.0)  # Normalize to 30% efficiency

            # Weighted score
            weights = [0.3, 0.3, 0.25, 0.15]
            factors = [frequency_factor, quality_factor, success_factor, efficiency_factor]

            potential_score = sum(factor * weight for factor, weight in zip(factors, weights))
            return max(0.0, min(1.0, potential_score))

        except Exception as e:
            logger.error(f"Failed to calculate query automation potential: {e}")
            return 0.0

    def _analyze_workflow_automation_patterns(self) -> List[Dict[str, Any]]:
        """Analyze workflow patterns for automation opportunities."""
        try:
            # This would analyze sequences of program executions to identify workflow patterns
            # For now, return basic workflow analysis
            return []

        except Exception as e:
            logger.error(f"Failed to analyze workflow automation patterns: {e}")
            return []

    def _analyze_user_specific_automation(self, user_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze user-specific automation opportunities."""
        try:
            opportunities = []

            # Analyze user's frequent patterns
            if 'frequent_queries' in user_patterns:
                for pattern in user_patterns['frequent_queries']:
                    if pattern.get('frequency', 0) >= 3:
                        opportunity = {
                            'type': 'user_specific_automation',
                            'pattern_identifier': pattern.get('pattern', 'unknown'),
                            'user_profile': user_patterns.get('user_profile', 'default'),
                            'frequency': pattern['frequency'],
                            'confidence_score': min(1.0, pattern['frequency'] / 10.0),
                            'potential_impact_score': pattern.get('avg_time_saved', 0) * pattern['frequency'],
                            'personalization_benefit': True
                        }
                        opportunities.append(opportunity)

            return opportunities

        except Exception as e:
            logger.error(f"Failed to analyze user-specific automation: {e}")
            return []

    def _generate_opportunity_analysis(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis for an automation opportunity."""
        try:
            analysis = {
                'opportunity_type': opportunity.get('type', 'unknown'),
                'confidence_level': self._categorize_confidence(opportunity.get('confidence_score', 0)),
                'impact_assessment': self._assess_opportunity_impact(opportunity),
                'implementation_complexity': self._assess_implementation_complexity(opportunity),
                'risk_factors': self._identify_risk_factors(opportunity),
                'success_probability': self._calculate_success_probability(opportunity),
                'recommended_timeline': self._recommend_implementation_timeline(opportunity)
            }

            return analysis

        except Exception as e:
            logger.error(f"Failed to generate opportunity analysis: {e}")
            return {}

    def _categorize_confidence(self, confidence_score: float) -> str:
        """Categorize confidence level."""
        if confidence_score >= 0.9:
            return "very_high"
        elif confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.7:
            return "medium"
        elif confidence_score >= 0.6:
            return "low"
        else:
            return "very_low"

    def _assess_opportunity_impact(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of an automation opportunity."""
        try:
            impact_score = opportunity.get('potential_impact_score', 0)
            frequency = opportunity.get('frequency', 0)
            time_savings = opportunity.get('estimated_time_savings_ms', 0)

            return {
                'impact_level': 'high' if impact_score > 10 else 'medium' if impact_score > 5 else 'low',
                'frequency_benefit': 'high' if frequency > 10 else 'medium' if frequency > 5 else 'low',
                'time_savings_benefit': 'high' if time_savings > 5000 else 'medium' if time_savings > 1000 else 'low',
                'overall_impact': 'high' if impact_score > 10 and frequency > 10 else 'medium'
            }

        except Exception as e:
            logger.error(f"Failed to assess opportunity impact: {e}")
            return {}

    def _assess_implementation_complexity(self, opportunity: Dict[str, Any]) -> str:
        """Assess implementation complexity."""
        opp_type = opportunity.get('type', 'unknown')

        if opp_type == 'query_automation':
            return 'low'  # Query automation is typically straightforward
        elif opp_type == 'workflow_automation':
            return 'high'  # Workflow automation is more complex
        elif opp_type == 'user_specific_automation':
            return 'medium'  # User-specific requires personalization
        else:
            return 'unknown'

    def _identify_risk_factors(self, opportunity: Dict[str, Any]) -> List[str]:
        """Identify risk factors for an automation opportunity."""
        risks = []

        if opportunity.get('confidence_score', 0) < 0.8:
            risks.append("Low confidence in automation success")

        if opportunity.get('frequency', 0) < 5:
            risks.append("Low frequency may not justify automation effort")

        if opportunity.get('avg_quality_score', 0) < 0.7:
            risks.append("Quality concerns in current implementations")

        return risks

    def _calculate_success_probability(self, opportunity: Dict[str, Any]) -> float:
        """Calculate probability of successful automation implementation."""
        try:
            confidence = opportunity.get('confidence_score', 0)
            quality = opportunity.get('avg_quality_score', 0)
            success_rate = opportunity.get('success_rate', 0)

            # Weighted average of success indicators
            probability = (confidence * 0.4 + quality * 0.3 + success_rate * 0.3)
            return round(min(1.0, max(0.0, probability)), 3)

        except Exception as e:
            logger.error(f"Failed to calculate success probability: {e}")
            return 0.0

    def _recommend_implementation_timeline(self, opportunity: Dict[str, Any]) -> str:
        """Recommend implementation timeline."""
        impact = opportunity.get('potential_impact_score', 0)
        confidence = opportunity.get('confidence_score', 0)

        if impact > 15 and confidence > 0.8:
            return "immediate"  # High impact, high confidence
        elif impact > 10 or confidence > 0.8:
            return "short_term"  # Good impact or confidence
        elif impact > 5:
            return "medium_term"  # Moderate impact
        else:
            return "long_term"  # Low priority

    # Placeholder methods for learning analysis (to be implemented based on specific requirements)

    def _analyze_pattern_discovery_trends(self, cutoff_date: datetime) -> Dict[str, Any]:
        """Analyze pattern discovery trends."""
        try:
            with sqlite3.connect(self.store.db_path) as conn:
                cursor = conn.execute("""
                    SELECT
                        DATE(discovery_timestamp) as discovery_date,
                        COUNT(*) as discoveries_count,
                        SUM(CASE WHEN capture_success = 1 THEN 1 ELSE 0 END) as successful_captures,
                        AVG(similarity_score) as avg_similarity
                    FROM pattern_discovery_log
                    WHERE discovery_timestamp > ?
                    GROUP BY DATE(discovery_timestamp)
                    ORDER BY discovery_date
                """, (cutoff_date.isoformat(),))

                trends = cursor.fetchall()

            return {
                'daily_discoveries': [{'date': row[0], 'count': row[1], 'success_rate': row[2]/row[1] if row[1] > 0 else 0} for row in trends],
                'total_discoveries': sum(row[1] for row in trends),
                'avg_success_rate': sum(row[2] for row in trends) / sum(row[1] for row in trends) if trends else 0
            }

        except Exception as e:
            logger.error(f"Failed to analyze pattern discovery trends: {e}")
            return {}

    def _calculate_learning_velocity(self, cutoff_date: datetime) -> Dict[str, Any]:
        """Calculate learning velocity metrics."""
        try:
            # Calculate rate of new program creation and improvement
            with sqlite3.connect(self.store.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM latent_programs
                    WHERE created_at > ?
                """, (cutoff_date.isoformat(),))

                new_programs = cursor.fetchone()[0]

            days_analyzed = (datetime.utcnow() - cutoff_date).days
            velocity = new_programs / max(days_analyzed, 1)

            return {
                'new_programs_created': new_programs,
                'analysis_period_days': days_analyzed,
                'programs_per_day': round(velocity, 2),
                'learning_velocity_category': 'high' if velocity > 1 else 'medium' if velocity > 0.5 else 'low'
            }

        except Exception as e:
            logger.error(f"Failed to calculate learning velocity: {e}")
            return {}

    def _analyze_knowledge_evolution(self, cutoff_date: datetime) -> Dict[str, Any]:
        """Analyze knowledge evolution patterns."""
        # Placeholder implementation
        return {
            'knowledge_domains_expanded': 0,
            'complexity_progression': 'stable',
            'specialization_trends': []
        }

    def _analyze_adaptation_patterns(self, cutoff_date: datetime) -> Dict[str, Any]:
        """Analyze adaptation patterns."""
        # Placeholder implementation
        return {
            'adaptation_events': 0,
            'adaptation_success_rate': 0.0,
            'adaptation_triggers': []
        }

    def _generate_learning_insights(self, discovery_trends: Dict[str, Any],
                                  learning_velocity: Dict[str, Any],
                                  knowledge_evolution: Dict[str, Any],
                                  adaptation_patterns: Dict[str, Any]) -> List[str]:
        """Generate comprehensive learning insights."""
        try:
            insights = []

            # Discovery trends insights
            total_discoveries = discovery_trends.get('total_discoveries', 0)
            if total_discoveries > 0:
                insights.append(f"Discovered {total_discoveries} new patterns in the analysis period")

                avg_success_rate = discovery_trends.get('avg_success_rate', 0)
                if avg_success_rate > 0.8:
                    insights.append("High pattern capture success rate indicates effective learning")
                elif avg_success_rate < 0.5:
                    insights.append("Low pattern capture success rate suggests learning challenges")
            else:
                insights.append("No new pattern discoveries detected")

            # Learning velocity insights
            velocity_category = learning_velocity.get('learning_velocity_category', 'unknown')
            programs_per_day = learning_velocity.get('programs_per_day', 0)

            if velocity_category == 'high':
                insights.append(f"High learning velocity: {programs_per_day:.1f} programs per day")
            elif velocity_category == 'low':
                insights.append(f"Low learning velocity: {programs_per_day:.1f} programs per day - consider optimization")
            else:
                insights.append(f"Moderate learning velocity: {programs_per_day:.1f} programs per day")

            # Knowledge evolution insights
            complexity_progression = knowledge_evolution.get('complexity_progression', 'stable')
            if complexity_progression == 'increasing':
                insights.append("Knowledge complexity is increasing - system is handling more sophisticated tasks")
            elif complexity_progression == 'decreasing':
                insights.append("Knowledge complexity is decreasing - focus on simpler tasks")

            return insights

        except Exception as e:
            logger.error(f"Failed to generate learning insights: {e}")
            return ["Error generating learning insights"]

    def _calculate_learning_health_score(self, discovery_trends: Dict[str, Any],
                                       learning_velocity: Dict[str, Any],
                                       knowledge_evolution: Dict[str, Any]) -> float:
        """Calculate overall learning health score."""
        try:
            factors = []

            # Discovery success factor
            success_rate = discovery_trends.get('avg_success_rate', 0)
            factors.append(success_rate)

            # Velocity factor
            velocity_category = learning_velocity.get('learning_velocity_category', 'low')
            velocity_score = {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(velocity_category, 0.4)
            factors.append(velocity_score)

            # Evolution factor (placeholder)
            evolution_score = 0.7  # Default moderate evolution
            factors.append(evolution_score)

            # Calculate weighted average
            health_score = sum(factors) / len(factors) if factors else 0.0
            return round(max(0.0, min(1.0, health_score)), 3)

        except Exception as e:
            logger.error(f"Failed to calculate learning health score: {e}")
            return 0.0

    def _get_cognitive_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get cognitive metrics for a specific time period."""
        try:
            with sqlite3.connect(self.store.db_path) as conn:
                # Get program creation metrics
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM latent_programs
                    WHERE created_at BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                programs_created = cursor.fetchone()[0]

                # Get execution metrics
                cursor = conn.execute("""
                    SELECT
                        COUNT(*) as total_executions,
                        AVG(quality_score) as avg_quality,
                        AVG(efficiency_gain) as avg_efficiency,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
                    FROM program_analytics_enhanced
                    WHERE execution_timestamp BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))

                execution_data = cursor.fetchone()

                return {
                    'programs_created': programs_created,
                    'total_executions': execution_data[0] or 0,
                    'avg_quality_score': execution_data[1] or 0,
                    'avg_efficiency_gain': execution_data[2] or 0,
                    'success_rate': execution_data[3] or 0,
                    'period_start': start_date.isoformat(),
                    'period_end': end_date.isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get cognitive metrics: {e}")
            return {}

    def _calculate_cognitive_evolution(self, baseline_metrics: Dict[str, Any],
                                     current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cognitive evolution between two periods."""
        try:
            evolution = {}

            # Calculate growth in program creation
            baseline_programs = baseline_metrics.get('programs_created', 0)
            current_programs = current_metrics.get('programs_created', 0)
            program_growth = ((current_programs - baseline_programs) / max(baseline_programs, 1)) * 100
            evolution['program_creation_growth'] = round(program_growth, 2)

            # Calculate quality improvement
            baseline_quality = baseline_metrics.get('avg_quality_score', 0)
            current_quality = current_metrics.get('avg_quality_score', 0)
            quality_improvement = current_quality - baseline_quality
            evolution['quality_improvement'] = round(quality_improvement, 3)

            # Calculate efficiency improvement
            baseline_efficiency = baseline_metrics.get('avg_efficiency_gain', 0)
            current_efficiency = current_metrics.get('avg_efficiency_gain', 0)
            efficiency_improvement = current_efficiency - baseline_efficiency
            evolution['efficiency_improvement'] = round(efficiency_improvement, 2)

            # Calculate overall growth score
            growth_factors = [
                min(1.0, max(-1.0, program_growth / 100)),  # Normalize to -1 to 1
                min(1.0, max(-1.0, quality_improvement * 5)),  # Scale quality change
                min(1.0, max(-1.0, efficiency_improvement / 20))  # Scale efficiency change
            ]

            overall_growth = sum(growth_factors) / len(growth_factors)
            evolution['overall_growth_score'] = round(overall_growth, 3)

            return evolution

        except Exception as e:
            logger.error(f"Failed to calculate cognitive evolution: {e}")
            return {}

    def _generate_evolution_insights(self, evolution_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from cognitive evolution analysis."""
        try:
            insights = []

            overall_growth = evolution_analysis.get('overall_growth_score', 0)
            if overall_growth > 0.2:
                insights.append("Strong cognitive growth detected across multiple metrics")
            elif overall_growth > 0.05:
                insights.append("Moderate cognitive improvement observed")
            elif overall_growth < -0.05:
                insights.append("Cognitive performance decline detected - requires attention")
            else:
                insights.append("Stable cognitive performance with minimal change")

            # Specific metric insights
            program_growth = evolution_analysis.get('program_creation_growth', 0)
            if program_growth > 20:
                insights.append("Rapid expansion in program library")
            elif program_growth < -10:
                insights.append("Decline in new program creation")

            quality_improvement = evolution_analysis.get('quality_improvement', 0)
            if quality_improvement > 0.1:
                insights.append("Significant improvement in response quality")
            elif quality_improvement < -0.1:
                insights.append("Quality degradation detected")

            return insights

        except Exception as e:
            logger.error(f"Failed to generate evolution insights: {e}")
            return ["Error generating evolution insights"]

    def _get_cognitive_system_status(self) -> Dict[str, Any]:
        """Get current cognitive system status."""
        try:
            total_programs = len(self.store.get_all_programs())
            active_programs = len([p for p in self.store.get_all_programs() if p.is_active])

            return {
                'total_programs': total_programs,
                'active_programs': active_programs,
                'system_health': 'healthy' if active_programs > 0 else 'inactive',
                'learning_status': 'active' if total_programs > 0 else 'initializing'
            }

        except Exception as e:
            logger.error(f"Failed to get cognitive system status: {e}")
            return {}

    def _generate_executive_summary(self, successful_types: Dict[str, Any],
                                  automation_opportunities: List[Dict[str, Any]],
                                  learning_insights: Dict[str, Any],
                                  cognitive_evolution: Dict[str, Any]) -> List[str]:
        """Generate executive summary of cognitive insights."""
        try:
            summary = []

            # System overview
            total_programs = successful_types.get('total_programs_analyzed', 0)
            summary.append(f"Analyzed {total_programs} programs for cognitive insights")

            # Success patterns
            successful_patterns = len(successful_types.get('successful_types', []))
            if successful_patterns > 0:
                summary.append(f"Identified {successful_patterns} highly successful program patterns")

            # Automation opportunities
            high_confidence_opportunities = len([opp for opp in automation_opportunities
                                               if opp.get('confidence_score', 0) > 0.8])
            if high_confidence_opportunities > 0:
                summary.append(f"Found {high_confidence_opportunities} high-confidence automation opportunities")

            # Learning health
            learning_health = learning_insights.get('learning_health_score', 0)
            if learning_health > 0.8:
                summary.append("Excellent learning health with strong pattern discovery")
            elif learning_health > 0.6:
                summary.append("Good learning health with steady progress")
            else:
                summary.append("Learning health requires attention and optimization")

            # Cognitive evolution
            growth_score = cognitive_evolution.get('cognitive_growth_score', 0)
            if growth_score > 0.1:
                summary.append("Positive cognitive evolution with measurable improvements")
            elif growth_score < -0.1:
                summary.append("Cognitive performance decline requires immediate attention")

            return summary

        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return ["Error generating executive summary"]

    def _generate_comprehensive_recommendations(self, successful_types: Dict[str, Any],
                                              automation_opportunities: List[Dict[str, Any]],
                                              learning_insights: Dict[str, Any],
                                              cognitive_evolution: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations."""
        try:
            recommendations = []

            # Based on successful types
            if successful_types.get('successful_types'):
                recommendations.append("Focus development on proven successful program patterns")

            # Based on automation opportunities
            high_impact_opportunities = [opp for opp in automation_opportunities
                                       if opp.get('potential_impact_score', 0) > 10]
            if high_impact_opportunities:
                recommendations.append(f"Prioritize implementation of {len(high_impact_opportunities)} high-impact automation opportunities")

            # Based on learning health
            learning_health = learning_insights.get('learning_health_score', 0)
            if learning_health < 0.6:
                recommendations.append("Improve learning mechanisms and pattern capture processes")

            # Based on cognitive evolution
            growth_score = cognitive_evolution.get('cognitive_growth_score', 0)
            if growth_score < 0:
                recommendations.append("Investigate and address factors causing cognitive performance decline")

            return recommendations if recommendations else ["Continue current cognitive development approach"]

        except Exception as e:
            logger.error(f"Failed to generate comprehensive recommendations: {e}")
            return ["Error generating recommendations"]

    # Placeholder methods for prediction functionality

    def _find_similar_historical_patterns(self, query_pattern: str,
                                         context_pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar historical patterns for prediction."""
        # Placeholder implementation
        return []

    def _calculate_automation_potential_score(self, similar_patterns: List[Dict[str, Any]],
                                            context_pattern: Dict[str, Any]) -> float:
        """Calculate automation potential score."""
        # Placeholder implementation
        return 0.5

    def _generate_prediction_insights(self, automation_score: float,
                                    similar_patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate prediction insights."""
        # Placeholder implementation
        return ["Prediction analysis requires more historical data"]

    def _calculate_prediction_confidence(self, similar_patterns: List[Dict[str, Any]]) -> float:
        """Calculate prediction confidence."""
        # Placeholder implementation
        return 0.5

    def _generate_automation_recommendations(self, automation_score: float) -> List[str]:
        """Generate automation recommendations."""
        # Placeholder implementation
        if automation_score > 0.7:
            return ["High automation potential - recommend immediate implementation"]
        elif automation_score > 0.5:
            return ["Moderate automation potential - consider pilot implementation"]
        else:
            return ["Low automation potential - monitor for future opportunities"]
