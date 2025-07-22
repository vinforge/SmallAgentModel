"""
SLP Program Analyzer
===================

Advanced program analysis and insights generation for the SLP system.
Provides pattern similarity analysis, program effectiveness scoring, and quality trend analysis.

Phase 1B.4 - Advanced Program Analytics (preserving 100% of existing functionality)
"""

import logging
import sqlite3
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import hashlib
import re

from .latent_program_store import LatentProgramStore
from .latent_program import LatentProgram

logger = logging.getLogger(__name__)


class ProgramAnalyzer:
    """
    Advanced program analysis and insights generation.
    
    Provides comprehensive analysis of program patterns, effectiveness,
    and usage trends while preserving 100% of existing SLP functionality.
    """
    
    def __init__(self, store: Optional[LatentProgramStore] = None):
        """Initialize the program analyzer."""
        self.store = store or LatentProgramStore()
        
        # Analysis configuration
        self.similarity_threshold = 0.7
        self.effectiveness_window_days = 30
        self.trend_analysis_periods = 7  # Number of periods for trend analysis
        
        # Caching for expensive operations
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_update = {}
        
        logger.info("Program Analyzer initialized")
    
    def analyze_pattern_similarity(self, programs: Optional[List[LatentProgram]] = None) -> Dict[str, Any]:
        """
        Analyze similarity patterns between programs.
        
        Args:
            programs: Optional list of programs to analyze. If None, analyzes all active programs.
            
        Returns:
            Dictionary containing similarity analysis results
        """
        try:
            if programs is None:
                programs = [p for p in self.store.get_all_programs() if p.is_active]
            
            if len(programs) < 2:
                return {
                    'total_programs': len(programs),
                    'similarity_clusters': [],
                    'average_similarity': 0.0,
                    'similarity_matrix': [],
                    'insights': ['Insufficient programs for similarity analysis']
                }
            
            # Calculate similarity matrix
            similarity_matrix = self._calculate_similarity_matrix(programs)
            
            # Identify similarity clusters
            clusters = self._identify_similarity_clusters(programs, similarity_matrix)
            
            # Calculate statistics
            avg_similarity = np.mean(similarity_matrix) if len(similarity_matrix) > 0 else 0.0
            max_similarity = np.max(similarity_matrix) if len(similarity_matrix) > 0 else 0.0
            min_similarity = np.min(similarity_matrix) if len(similarity_matrix) > 0 else 0.0
            
            # Generate insights
            insights = self._generate_similarity_insights(clusters, avg_similarity, programs)
            
            return {
                'total_programs': len(programs),
                'similarity_clusters': clusters,
                'average_similarity': round(avg_similarity, 3),
                'max_similarity': round(max_similarity, 3),
                'min_similarity': round(min_similarity, 3),
                'similarity_matrix': similarity_matrix.tolist() if hasattr(similarity_matrix, 'tolist') else similarity_matrix,
                'cluster_count': len(clusters),
                'insights': insights,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze pattern similarity: {e}")
            return {}
    
    def calculate_program_effectiveness(self, program_id: str, 
                                      time_window_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive effectiveness score for a program.
        
        Args:
            program_id: ID of the program to analyze
            time_window_days: Optional time window for analysis (default: 30 days)
            
        Returns:
            Dictionary containing effectiveness analysis
        """
        try:
            if time_window_days is None:
                time_window_days = self.effectiveness_window_days
            
            # Get program data
            program = self.store.get_program(program_id)
            if not program:
                return {'error': f'Program {program_id} not found'}
            
            # Get execution analytics for the time window
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            with sqlite3.connect(self.store.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        execution_time_ms,
                        quality_score,
                        user_feedback,
                        efficiency_gain,
                        success,
                        token_count,
                        execution_timestamp
                    FROM program_analytics_enhanced 
                    WHERE program_id = ? AND execution_timestamp > ?
                    ORDER BY execution_timestamp DESC
                """, (program_id, cutoff_date.isoformat()))
                
                executions = cursor.fetchall()
            
            if not executions:
                return {
                    'program_id': program_id,
                    'effectiveness_score': 0.0,
                    'execution_count': 0,
                    'analysis_period_days': time_window_days,
                    'insights': ['No executions found in the specified time window']
                }
            
            # Calculate effectiveness metrics
            effectiveness_metrics = self._calculate_effectiveness_metrics(executions)
            
            # Calculate overall effectiveness score
            effectiveness_score = self._calculate_overall_effectiveness_score(effectiveness_metrics)
            
            # Generate effectiveness insights
            insights = self._generate_effectiveness_insights(effectiveness_metrics, program)
            
            return {
                'program_id': program_id,
                'effectiveness_score': round(effectiveness_score, 3),
                'execution_count': len(executions),
                'analysis_period_days': time_window_days,
                'metrics': effectiveness_metrics,
                'insights': insights,
                'program_metadata': {
                    'created_at': program.created_at,
                    'usage_count': program.usage_count,
                    'success_rate': program.success_rate,
                    'confidence_score': program.confidence_score
                },
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate program effectiveness: {e}")
            return {}
    
    def analyze_usage_frequency(self, time_window_days: int = 30) -> Dict[str, Any]:
        """
        Analyze usage frequency patterns across all programs.
        
        Args:
            time_window_days: Time window for frequency analysis
            
        Returns:
            Dictionary containing usage frequency analysis
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            with sqlite3.connect(self.store.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        program_id,
                        COUNT(*) as execution_count,
                        AVG(execution_time_ms) as avg_execution_time,
                        AVG(quality_score) as avg_quality,
                        AVG(efficiency_gain) as avg_efficiency_gain,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
                    FROM program_analytics_enhanced 
                    WHERE execution_timestamp > ?
                    GROUP BY program_id
                    ORDER BY execution_count DESC
                """, (cutoff_date.isoformat(),))
                
                frequency_data = cursor.fetchall()
            
            if not frequency_data:
                return {
                    'analysis_period_days': time_window_days,
                    'total_programs_used': 0,
                    'frequency_distribution': [],
                    'insights': ['No program executions found in the specified time window']
                }
            
            # Process frequency data
            frequency_analysis = []
            total_executions = sum(row[1] for row in frequency_data)
            
            for row in frequency_data:
                program_id, exec_count, avg_time, avg_quality, avg_efficiency, success_count = row
                
                frequency_analysis.append({
                    'program_id': program_id,
                    'execution_count': exec_count,
                    'frequency_percentage': round((exec_count / total_executions) * 100, 2),
                    'avg_execution_time_ms': round(avg_time or 0, 2),
                    'avg_quality_score': round(avg_quality or 0, 3),
                    'avg_efficiency_gain': round(avg_efficiency or 0, 2),
                    'success_rate': round((success_count / exec_count) * 100, 2) if exec_count > 0 else 0,
                    'usage_category': self._categorize_usage_frequency(exec_count, total_executions)
                })
            
            # Generate frequency insights
            insights = self._generate_frequency_insights(frequency_analysis, time_window_days)
            
            return {
                'analysis_period_days': time_window_days,
                'total_programs_used': len(frequency_data),
                'total_executions': total_executions,
                'frequency_distribution': frequency_analysis,
                'usage_categories': self._summarize_usage_categories(frequency_analysis),
                'insights': insights,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze usage frequency: {e}")
            return {}
    
    def analyze_quality_trends(self, program_id: Optional[str] = None, 
                             time_window_days: int = 30) -> Dict[str, Any]:
        """
        Analyze quality trends over time for a specific program or all programs.
        
        Args:
            program_id: Optional specific program ID to analyze
            time_window_days: Time window for trend analysis
            
        Returns:
            Dictionary containing quality trend analysis
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            # Build query based on whether analyzing specific program or all programs
            if program_id:
                query = """
                    SELECT 
                        DATE(execution_timestamp) as execution_date,
                        AVG(quality_score) as avg_quality,
                        COUNT(*) as execution_count,
                        AVG(efficiency_gain) as avg_efficiency,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
                    FROM program_analytics_enhanced 
                    WHERE program_id = ? AND execution_timestamp > ?
                    GROUP BY DATE(execution_timestamp)
                    ORDER BY execution_date
                """
                params = (program_id, cutoff_date.isoformat())
            else:
                query = """
                    SELECT 
                        DATE(execution_timestamp) as execution_date,
                        AVG(quality_score) as avg_quality,
                        COUNT(*) as execution_count,
                        AVG(efficiency_gain) as avg_efficiency,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
                    FROM program_analytics_enhanced 
                    WHERE execution_timestamp > ?
                    GROUP BY DATE(execution_timestamp)
                    ORDER BY execution_date
                """
                params = (cutoff_date.isoformat(),)
            
            with sqlite3.connect(self.store.db_path) as conn:
                cursor = conn.execute(query, params)
                trend_data = cursor.fetchall()
            
            if not trend_data:
                return {
                    'program_id': program_id,
                    'analysis_period_days': time_window_days,
                    'trend_data': [],
                    'trend_analysis': {},
                    'insights': ['No execution data found for trend analysis']
                }
            
            # Process trend data
            processed_trends = []
            for row in trend_data:
                date, avg_quality, exec_count, avg_efficiency, success_count = row
                processed_trends.append({
                    'date': date,
                    'avg_quality_score': round(avg_quality or 0, 3),
                    'execution_count': exec_count,
                    'avg_efficiency_gain': round(avg_efficiency or 0, 2),
                    'success_rate': round((success_count / exec_count) * 100, 2) if exec_count > 0 else 0
                })
            
            # Calculate trend analysis
            trend_analysis = self._calculate_trend_analysis(processed_trends)
            
            # Generate trend insights
            insights = self._generate_trend_insights(trend_analysis, program_id)
            
            return {
                'program_id': program_id,
                'analysis_period_days': time_window_days,
                'trend_data': processed_trends,
                'trend_analysis': trend_analysis,
                'insights': insights,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze quality trends: {e}")
            return {}
    
    def get_comprehensive_program_analysis(self, program_id: str) -> Dict[str, Any]:
        """
        Get comprehensive analysis for a specific program.
        
        Args:
            program_id: ID of the program to analyze
            
        Returns:
            Dictionary containing comprehensive program analysis
        """
        try:
            # Get program basic info
            program = self.store.get_program(program_id)
            if not program:
                return {'error': f'Program {program_id} not found'}
            
            # Perform all analyses
            effectiveness = self.calculate_program_effectiveness(program_id)
            quality_trends = self.analyze_quality_trends(program_id)
            
            # Get related programs through similarity analysis
            all_programs = [p for p in self.store.get_all_programs() if p.is_active]
            similarity_analysis = self.analyze_pattern_similarity(all_programs)
            
            # Find programs similar to this one
            similar_programs = self._find_similar_programs(program_id, similarity_analysis)
            
            return {
                'program_id': program_id,
                'program_info': {
                    'id': program.id,
                    'created_at': program.created_at,
                    'usage_count': program.usage_count,
                    'success_rate': program.success_rate,
                    'confidence_score': program.confidence_score,
                    'is_active': program.is_active,
                    'signature': program.signature
                },
                'effectiveness_analysis': effectiveness,
                'quality_trends': quality_trends,
                'similar_programs': similar_programs,
                'comprehensive_insights': self._generate_comprehensive_insights(
                    program, effectiveness, quality_trends, similar_programs
                ),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive program analysis: {e}")
            return {}

    # Helper Methods for Analysis (Phase 1B.4 - preserving 100% of existing functionality)

    def _calculate_similarity_matrix(self, programs: List[LatentProgram]) -> np.ndarray:
        """Calculate similarity matrix between programs."""
        try:
            n = len(programs)
            similarity_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        similarity = self._calculate_program_similarity(programs[i], programs[j])
                        similarity_matrix[i][j] = similarity
                        similarity_matrix[j][i] = similarity

            return similarity_matrix

        except Exception as e:
            logger.error(f"Failed to calculate similarity matrix: {e}")
            return np.array([])

    def _calculate_program_similarity(self, program1: LatentProgram, program2: LatentProgram) -> float:
        """Calculate similarity between two programs."""
        try:
            similarity_factors = []

            # Signature similarity
            sig1 = program1.signature
            sig2 = program2.signature

            if isinstance(sig1, dict) and isinstance(sig2, dict):
                # Compare signature components
                sig_similarity = self._compare_signatures(sig1, sig2)
                similarity_factors.append(sig_similarity)

            # Usage pattern similarity
            usage_similarity = self._compare_usage_patterns(program1, program2)
            similarity_factors.append(usage_similarity)

            # Performance similarity
            perf_similarity = self._compare_performance_metrics(program1, program2)
            similarity_factors.append(perf_similarity)

            # Calculate weighted average
            weights = [0.5, 0.3, 0.2]  # Signature, usage, performance
            if len(similarity_factors) == len(weights):
                similarity = sum(factor * weight for factor, weight in zip(similarity_factors, weights))
            else:
                similarity = sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.error(f"Failed to calculate program similarity: {e}")
            return 0.0

    def _compare_signatures(self, sig1: Dict[str, Any], sig2: Dict[str, Any]) -> float:
        """Compare signature similarity."""
        try:
            # Compare primary intent
            intent_match = 1.0 if sig1.get('primary_intent') == sig2.get('primary_intent') else 0.0

            # Compare secondary intents
            sec1 = set(sig1.get('secondary_intents', []))
            sec2 = set(sig2.get('secondary_intents', []))
            if sec1 or sec2:
                intent_overlap = len(sec1 & sec2) / len(sec1 | sec2) if (sec1 | sec2) else 0.0
            else:
                intent_overlap = 1.0

            # Compare complexity level
            complexity_match = 1.0 if sig1.get('complexity_level') == sig2.get('complexity_level') else 0.5

            # Compare document types
            doc1 = set(sig1.get('document_types', []))
            doc2 = set(sig2.get('document_types', []))
            if doc1 or doc2:
                doc_overlap = len(doc1 & doc2) / len(doc1 | doc2) if (doc1 | doc2) else 0.0
            else:
                doc_overlap = 1.0

            # Weighted average
            return (intent_match * 0.4 + intent_overlap * 0.3 + complexity_match * 0.2 + doc_overlap * 0.1)

        except Exception as e:
            logger.error(f"Failed to compare signatures: {e}")
            return 0.0

    def _compare_usage_patterns(self, program1: LatentProgram, program2: LatentProgram) -> float:
        """Compare usage pattern similarity."""
        try:
            # Compare usage counts (normalized)
            max_usage = max(program1.usage_count, program2.usage_count, 1)
            min_usage = min(program1.usage_count, program2.usage_count)
            usage_similarity = min_usage / max_usage

            # Compare success rates
            success_diff = abs(program1.success_rate - program2.success_rate)
            success_similarity = 1.0 - success_diff

            # Compare confidence scores
            conf_diff = abs(program1.confidence_score - program2.confidence_score)
            conf_similarity = 1.0 - conf_diff

            return (usage_similarity * 0.4 + success_similarity * 0.3 + conf_similarity * 0.3)

        except Exception as e:
            logger.error(f"Failed to compare usage patterns: {e}")
            return 0.0

    def _compare_performance_metrics(self, program1: LatentProgram, program2: LatentProgram) -> float:
        """Compare performance metrics similarity."""
        try:
            # Compare average latency
            if program1.avg_latency_ms > 0 and program2.avg_latency_ms > 0:
                max_latency = max(program1.avg_latency_ms, program2.avg_latency_ms)
                min_latency = min(program1.avg_latency_ms, program2.avg_latency_ms)
                latency_similarity = min_latency / max_latency
            else:
                latency_similarity = 0.5  # Default if no latency data

            # Compare token counts
            if program1.avg_token_count > 0 and program2.avg_token_count > 0:
                max_tokens = max(program1.avg_token_count, program2.avg_token_count)
                min_tokens = min(program1.avg_token_count, program2.avg_token_count)
                token_similarity = min_tokens / max_tokens
            else:
                token_similarity = 0.5  # Default if no token data

            return (latency_similarity * 0.6 + token_similarity * 0.4)

        except Exception as e:
            logger.error(f"Failed to compare performance metrics: {e}")
            return 0.0

    def _identify_similarity_clusters(self, programs: List[LatentProgram],
                                    similarity_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Identify clusters of similar programs."""
        try:
            if len(programs) < 2 or similarity_matrix.size == 0:
                return []

            clusters = []
            visited = set()

            for i, program in enumerate(programs):
                if i in visited:
                    continue

                # Find all programs similar to this one
                cluster_members = [i]
                visited.add(i)

                for j in range(len(programs)):
                    if j != i and j not in visited and similarity_matrix[i][j] >= self.similarity_threshold:
                        cluster_members.append(j)
                        visited.add(j)

                if len(cluster_members) > 1:
                    cluster = {
                        'cluster_id': len(clusters),
                        'program_ids': [programs[idx].id for idx in cluster_members],
                        'program_count': len(cluster_members),
                        'avg_similarity': round(np.mean([similarity_matrix[cluster_members[0]][j]
                                                       for j in cluster_members[1:]]), 3),
                        'cluster_characteristics': self._analyze_cluster_characteristics(
                            [programs[idx] for idx in cluster_members]
                        )
                    }
                    clusters.append(cluster)

            return clusters

        except Exception as e:
            logger.error(f"Failed to identify similarity clusters: {e}")
            return []

    def _analyze_cluster_characteristics(self, programs: List[LatentProgram]) -> Dict[str, Any]:
        """Analyze characteristics of a program cluster."""
        try:
            if not programs:
                return {}

            # Analyze common signature elements
            primary_intents = [p.signature.get('primary_intent', 'unknown') for p in programs
                             if isinstance(p.signature, dict)]
            complexity_levels = [p.signature.get('complexity_level', 'unknown') for p in programs
                               if isinstance(p.signature, dict)]

            # Performance characteristics
            avg_usage = sum(p.usage_count for p in programs) / len(programs)
            avg_success_rate = sum(p.success_rate for p in programs) / len(programs)
            avg_confidence = sum(p.confidence_score for p in programs) / len(programs)

            return {
                'common_primary_intent': Counter(primary_intents).most_common(1)[0][0] if primary_intents else 'unknown',
                'common_complexity': Counter(complexity_levels).most_common(1)[0][0] if complexity_levels else 'unknown',
                'avg_usage_count': round(avg_usage, 1),
                'avg_success_rate': round(avg_success_rate, 3),
                'avg_confidence_score': round(avg_confidence, 3),
                'program_count': len(programs)
            }

        except Exception as e:
            logger.error(f"Failed to analyze cluster characteristics: {e}")
            return {}

    def _calculate_effectiveness_metrics(self, executions: List[Tuple]) -> Dict[str, Any]:
        """Calculate effectiveness metrics from execution data."""
        try:
            if not executions:
                return {}

            # Extract metrics from executions
            execution_times = [row[0] for row in executions if row[0] is not None]
            quality_scores = [row[1] for row in executions if row[1] is not None]
            user_feedback = [row[2] for row in executions if row[2] is not None and row[2] != 0]
            efficiency_gains = [row[3] for row in executions if row[3] is not None and row[3] > 0]
            success_flags = [row[4] for row in executions if row[4] is not None]
            token_counts = [row[5] for row in executions if row[5] is not None]

            metrics = {
                'execution_count': len(executions),
                'avg_execution_time_ms': round(sum(execution_times) / len(execution_times), 2) if execution_times else 0,
                'avg_quality_score': round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0,
                'avg_efficiency_gain': round(sum(efficiency_gains) / len(efficiency_gains), 2) if efficiency_gains else 0,
                'success_rate': round(sum(success_flags) / len(success_flags) * 100, 2) if success_flags else 0,
                'avg_token_count': round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
                'user_feedback_score': round(sum(user_feedback) / len(user_feedback), 2) if user_feedback else 0,
                'feedback_count': len(user_feedback),
                'consistency_score': self._calculate_consistency_score(quality_scores, execution_times)
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate effectiveness metrics: {e}")
            return {}

    def _calculate_overall_effectiveness_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall effectiveness score from metrics."""
        try:
            if not metrics:
                return 0.0

            # Normalize and weight different factors
            factors = []

            # Quality factor (0-1)
            quality_factor = min(1.0, metrics.get('avg_quality_score', 0))
            factors.append(('quality', quality_factor, 0.3))

            # Success rate factor (0-1)
            success_factor = metrics.get('success_rate', 0) / 100.0
            factors.append(('success', success_factor, 0.25))

            # Efficiency factor (0-1, normalized)
            efficiency_gain = metrics.get('avg_efficiency_gain', 0)
            efficiency_factor = min(1.0, max(0.0, efficiency_gain / 50.0))  # Normalize to 50% max gain
            factors.append(('efficiency', efficiency_factor, 0.2))

            # User feedback factor (0-1, normalized from -1 to 1 scale)
            feedback_score = metrics.get('user_feedback_score', 0)
            feedback_factor = (feedback_score + 1) / 2.0 if feedback_score != 0 else 0.5
            factors.append(('feedback', feedback_factor, 0.15))

            # Consistency factor (0-1)
            consistency_factor = metrics.get('consistency_score', 0.5)
            factors.append(('consistency', consistency_factor, 0.1))

            # Calculate weighted score
            total_score = sum(factor * weight for _, factor, weight in factors)

            return max(0.0, min(1.0, total_score))

        except Exception as e:
            logger.error(f"Failed to calculate overall effectiveness score: {e}")
            return 0.0

    def _calculate_consistency_score(self, quality_scores: List[float], execution_times: List[float]) -> float:
        """Calculate consistency score based on variance in quality and execution time."""
        try:
            if len(quality_scores) < 2 or len(execution_times) < 2:
                return 0.5  # Default for insufficient data

            # Calculate coefficient of variation for quality scores
            quality_mean = sum(quality_scores) / len(quality_scores)
            quality_variance = sum((x - quality_mean) ** 2 for x in quality_scores) / len(quality_scores)
            quality_cv = (quality_variance ** 0.5) / quality_mean if quality_mean > 0 else 1.0

            # Calculate coefficient of variation for execution times
            time_mean = sum(execution_times) / len(execution_times)
            time_variance = sum((x - time_mean) ** 2 for x in execution_times) / len(execution_times)
            time_cv = (time_variance ** 0.5) / time_mean if time_mean > 0 else 1.0

            # Lower CV means higher consistency
            quality_consistency = max(0.0, 1.0 - quality_cv)
            time_consistency = max(0.0, 1.0 - min(time_cv, 1.0))

            # Weighted average
            consistency_score = (quality_consistency * 0.7 + time_consistency * 0.3)

            return max(0.0, min(1.0, consistency_score))

        except Exception as e:
            logger.error(f"Failed to calculate consistency score: {e}")
            return 0.5

    def _categorize_usage_frequency(self, execution_count: int, total_executions: int) -> str:
        """Categorize usage frequency."""
        try:
            frequency_percentage = (execution_count / total_executions) * 100 if total_executions > 0 else 0

            if frequency_percentage >= 20:
                return "very_high"
            elif frequency_percentage >= 10:
                return "high"
            elif frequency_percentage >= 5:
                return "medium"
            elif frequency_percentage >= 1:
                return "low"
            else:
                return "very_low"

        except Exception as e:
            logger.error(f"Failed to categorize usage frequency: {e}")
            return "unknown"

    def _summarize_usage_categories(self, frequency_analysis: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize usage categories."""
        try:
            categories = Counter(item['usage_category'] for item in frequency_analysis)
            return dict(categories)

        except Exception as e:
            logger.error(f"Failed to summarize usage categories: {e}")
            return {}

    def _calculate_trend_analysis(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trend analysis from time series data."""
        try:
            if len(trend_data) < 2:
                return {'trend_direction': 'insufficient_data', 'trend_strength': 0.0}

            # Extract quality scores over time
            quality_scores = [item['avg_quality_score'] for item in trend_data]
            efficiency_gains = [item['avg_efficiency_gain'] for item in trend_data]
            success_rates = [item['success_rate'] for item in trend_data]

            # Calculate trends using simple linear regression
            quality_trend = self._calculate_linear_trend(quality_scores)
            efficiency_trend = self._calculate_linear_trend(efficiency_gains)
            success_trend = self._calculate_linear_trend(success_rates)

            # Determine overall trend direction
            trend_scores = [quality_trend['slope'], efficiency_trend['slope'], success_trend['slope']]
            avg_slope = sum(trend_scores) / len(trend_scores)

            if avg_slope > 0.01:
                trend_direction = 'improving'
            elif avg_slope < -0.01:
                trend_direction = 'declining'
            else:
                trend_direction = 'stable'

            return {
                'trend_direction': trend_direction,
                'trend_strength': abs(avg_slope),
                'quality_trend': quality_trend,
                'efficiency_trend': efficiency_trend,
                'success_trend': success_trend,
                'data_points': len(trend_data)
            }

        except Exception as e:
            logger.error(f"Failed to calculate trend analysis: {e}")
            return {}

    def _calculate_linear_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate linear trend for a series of values."""
        try:
            if len(values) < 2:
                return {'slope': 0.0, 'correlation': 0.0}

            n = len(values)
            x = list(range(n))

            # Calculate slope using least squares
            x_mean = sum(x) / n
            y_mean = sum(values) / n

            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            slope = numerator / denominator if denominator != 0 else 0.0

            # Calculate correlation coefficient
            x_var = sum((x[i] - x_mean) ** 2 for i in range(n)) / n
            y_var = sum((values[i] - y_mean) ** 2 for i in range(n)) / n

            if x_var > 0 and y_var > 0:
                correlation = numerator / (n * (x_var * y_var) ** 0.5)
            else:
                correlation = 0.0

            return {
                'slope': round(slope, 4),
                'correlation': round(correlation, 3)
            }

        except Exception as e:
            logger.error(f"Failed to calculate linear trend: {e}")
            return {'slope': 0.0, 'correlation': 0.0}

    # Insight Generation Methods (Phase 1B.4 - preserving 100% of existing functionality)

    def _generate_similarity_insights(self, clusters: List[Dict[str, Any]],
                                    avg_similarity: float, programs: List[LatentProgram]) -> List[str]:
        """Generate insights from similarity analysis."""
        try:
            insights = []

            if len(clusters) == 0:
                insights.append("No significant similarity clusters found - programs are highly diverse")
            elif len(clusters) == 1:
                insights.append(f"Found 1 similarity cluster with {clusters[0]['program_count']} programs")
            else:
                insights.append(f"Found {len(clusters)} similarity clusters")
                largest_cluster = max(clusters, key=lambda x: x['program_count'])
                insights.append(f"Largest cluster contains {largest_cluster['program_count']} programs")

            if avg_similarity > 0.8:
                insights.append("High overall similarity suggests potential for program consolidation")
            elif avg_similarity < 0.3:
                insights.append("Low overall similarity indicates diverse program portfolio")

            # Analyze cluster characteristics
            if clusters:
                common_intents = Counter(cluster['cluster_characteristics'].get('common_primary_intent', 'unknown')
                                       for cluster in clusters)
                if common_intents:
                    most_common = common_intents.most_common(1)[0]
                    insights.append(f"Most common cluster intent: {most_common[0]} ({most_common[1]} clusters)")

            return insights

        except Exception as e:
            logger.error(f"Failed to generate similarity insights: {e}")
            return ["Error generating similarity insights"]

    def _generate_effectiveness_insights(self, metrics: Dict[str, Any],
                                       program: LatentProgram) -> List[str]:
        """Generate insights from effectiveness analysis."""
        try:
            insights = []

            effectiveness_score = self._calculate_overall_effectiveness_score(metrics)

            if effectiveness_score >= 0.8:
                insights.append("Excellent program effectiveness - high performance across all metrics")
            elif effectiveness_score >= 0.6:
                insights.append("Good program effectiveness with room for improvement")
            elif effectiveness_score >= 0.4:
                insights.append("Moderate program effectiveness - consider optimization")
            else:
                insights.append("Low program effectiveness - requires attention")

            # Specific metric insights
            if metrics.get('avg_quality_score', 0) < 0.6:
                insights.append("Quality scores below threshold - review program logic")

            if metrics.get('success_rate', 0) < 80:
                insights.append("Success rate below 80% - investigate failure patterns")

            if metrics.get('avg_efficiency_gain', 0) < 10:
                insights.append("Low efficiency gains - program may not provide significant value")

            if metrics.get('consistency_score', 0) < 0.5:
                insights.append("Inconsistent performance - program behavior varies significantly")

            if metrics.get('user_feedback_score', 0) < 0:
                insights.append("Negative user feedback - user satisfaction concerns")

            return insights

        except Exception as e:
            logger.error(f"Failed to generate effectiveness insights: {e}")
            return ["Error generating effectiveness insights"]

    def _generate_frequency_insights(self, frequency_analysis: List[Dict[str, Any]],
                                   time_window_days: int) -> List[str]:
        """Generate insights from frequency analysis."""
        try:
            insights = []

            if not frequency_analysis:
                insights.append("No program usage detected in the analysis period")
                return insights

            total_programs = len(frequency_analysis)
            high_usage_programs = len([p for p in frequency_analysis if p['usage_category'] in ['high', 'very_high']])
            low_usage_programs = len([p for p in frequency_analysis if p['usage_category'] in ['low', 'very_low']])

            insights.append(f"Analyzed {total_programs} programs over {time_window_days} days")

            if high_usage_programs > 0:
                insights.append(f"{high_usage_programs} programs show high usage patterns")

            if low_usage_programs > total_programs * 0.5:
                insights.append(f"{low_usage_programs} programs have low usage - consider cleanup")

            # Top performer insights
            top_program = frequency_analysis[0]
            insights.append(f"Most used program: {top_program['program_id']} ({top_program['execution_count']} executions)")

            # Quality insights
            high_quality_programs = len([p for p in frequency_analysis if p['avg_quality_score'] >= 0.8])
            if high_quality_programs > 0:
                insights.append(f"{high_quality_programs} programs maintain high quality scores")

            return insights

        except Exception as e:
            logger.error(f"Failed to generate frequency insights: {e}")
            return ["Error generating frequency insights"]

    def _generate_trend_insights(self, trend_analysis: Dict[str, Any],
                               program_id: Optional[str]) -> List[str]:
        """Generate insights from trend analysis."""
        try:
            insights = []

            trend_direction = trend_analysis.get('trend_direction', 'unknown')
            trend_strength = trend_analysis.get('trend_strength', 0)

            scope = f"Program {program_id}" if program_id else "Overall system"

            if trend_direction == 'improving':
                insights.append(f"{scope} shows improving performance trends")
                if trend_strength > 0.1:
                    insights.append("Strong improvement trend detected")
            elif trend_direction == 'declining':
                insights.append(f"{scope} shows declining performance trends")
                if trend_strength > 0.1:
                    insights.append("Significant decline detected - requires attention")
            else:
                insights.append(f"{scope} performance is stable")

            # Specific trend insights
            quality_trend = trend_analysis.get('quality_trend', {})
            if quality_trend.get('slope', 0) > 0.01:
                insights.append("Quality scores are improving over time")
            elif quality_trend.get('slope', 0) < -0.01:
                insights.append("Quality scores are declining - investigate causes")

            efficiency_trend = trend_analysis.get('efficiency_trend', {})
            if efficiency_trend.get('slope', 0) > 0.01:
                insights.append("Efficiency gains are increasing")

            data_points = trend_analysis.get('data_points', 0)
            if data_points < 5:
                insights.append("Limited data points - trend analysis may not be reliable")

            return insights

        except Exception as e:
            logger.error(f"Failed to generate trend insights: {e}")
            return ["Error generating trend insights"]

    def _find_similar_programs(self, program_id: str,
                             similarity_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find programs similar to the specified program."""
        try:
            similar_programs = []

            clusters = similarity_analysis.get('similarity_clusters', [])
            for cluster in clusters:
                if program_id in cluster.get('program_ids', []):
                    # Found the cluster containing this program
                    other_programs = [pid for pid in cluster['program_ids'] if pid != program_id]
                    for other_id in other_programs:
                        similar_programs.append({
                            'program_id': other_id,
                            'similarity_score': cluster.get('avg_similarity', 0),
                            'cluster_id': cluster.get('cluster_id', -1)
                        })
                    break

            return similar_programs

        except Exception as e:
            logger.error(f"Failed to find similar programs: {e}")
            return []

    def _generate_comprehensive_insights(self, program: LatentProgram,
                                       effectiveness: Dict[str, Any],
                                       quality_trends: Dict[str, Any],
                                       similar_programs: List[Dict[str, Any]]) -> List[str]:
        """Generate comprehensive insights for a program."""
        try:
            insights = []

            # Program overview
            insights.append(f"Program {program.id} analysis summary:")
            insights.append(f"Usage: {program.usage_count} executions, {program.success_rate:.1%} success rate")

            # Effectiveness insights
            effectiveness_score = effectiveness.get('effectiveness_score', 0)
            if effectiveness_score >= 0.8:
                insights.append("✅ High-performing program with excellent effectiveness")
            elif effectiveness_score >= 0.6:
                insights.append("✅ Well-performing program with good effectiveness")
            else:
                insights.append("⚠️ Program effectiveness below optimal levels")

            # Trend insights
            trend_direction = quality_trends.get('trend_analysis', {}).get('trend_direction', 'unknown')
            if trend_direction == 'improving':
                insights.append("📈 Performance trending upward")
            elif trend_direction == 'declining':
                insights.append("📉 Performance trending downward - needs attention")

            # Similarity insights
            if similar_programs:
                insights.append(f"🔗 Found {len(similar_programs)} similar programs for potential optimization")
            else:
                insights.append("🔍 Unique program with no close similarities")

            # Recommendations
            if effectiveness_score < 0.6:
                insights.append("💡 Recommendation: Review and optimize program logic")

            if len(similar_programs) > 2:
                insights.append("💡 Recommendation: Consider consolidating similar programs")

            return insights

        except Exception as e:
            logger.error(f"Failed to generate comprehensive insights: {e}")
            return ["Error generating comprehensive insights"]
