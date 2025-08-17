"""
Program Manager
==============

Central orchestrator for the SLP system. Manages program capture, retrieval,
execution, and lifecycle management.
"""

import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

from .latent_program import LatentProgram, ExecutionResult, ValidationResult
from .program_signature import ProgramSignature, generate_signature
from .latent_program_store import LatentProgramStore
from .program_executor import ProgramExecutor

# Initialize logger first
logger = logging.getLogger(__name__)

# Enhanced Analytics Integration (Phase 1A.2 - preserving 100% of existing functionality)
try:
    from .analytics_engine import SLPAnalyticsEngine
    from .metrics_collector import SLPMetricsCollector
except ImportError:
    logger.warning("Analytics modules not available, using basic functionality")
    SLPAnalyticsEngine = None
    SLPMetricsCollector = None

# Import validator with fallback
try:
    from .program_validator import ProgramValidator
except ImportError:
    logger.warning("ProgramValidator not available, using basic validation")
    ProgramValidator = None


class ProgramManager:
    """
    Central orchestrator for the SLP system.
    
    Handles program lifecycle from capture through execution and evolution.
    """
    
    def __init__(self, store: Optional[LatentProgramStore] = None,
                 executor: Optional[ProgramExecutor] = None,
                 validator: Optional['ProgramValidator'] = None,
                 analytics_engine: Optional['SLPAnalyticsEngine'] = None,
                 metrics_collector: Optional['SLPMetricsCollector'] = None):
        """Initialize the program manager with enhanced analytics support."""
        self.store = store or LatentProgramStore()
        self.executor = executor or ProgramExecutor()
        self.validator = validator or (ProgramValidator() if ProgramValidator else None)

        # Enhanced Analytics Integration (Phase 1A.2 - preserving 100% of existing functionality)
        self.analytics_engine = analytics_engine or (SLPAnalyticsEngine(self.store) if SLPAnalyticsEngine else None)
        self.metrics_collector = metrics_collector or (SLPMetricsCollector(self.store) if SLPMetricsCollector else None)

        # Configuration (preserving all existing settings)
        self.similarity_threshold = 0.8
        self.execution_threshold = 0.7
        self.quality_threshold = 0.6
        self.max_programs_per_signature = 5

        # Enhanced analytics configuration
        self.analytics_enabled = True
        self.enable_real_time_metrics = True

        # Start metrics collection if available
        if self.metrics_collector and self.enable_real_time_metrics:
            self.metrics_collector.start_collection()

        logger.info("Program manager initialized with enhanced analytics support")
    
    def find_matching_program(self, query: str, context: Dict[str, Any],
                            user_profile: Optional[str] = None) -> Tuple[Optional[LatentProgram], float]:
        """
        Find the best matching program for a query.

        Args:
            query: User's query
            context: Context information
            user_profile: Optional user profile

        Returns:
            Tuple of (best matching program or None, confidence score)
        """
        try:
            # Generate signature for the query
            signature = generate_signature(query, context, user_profile)

            # Find candidate programs
            candidates = self.find_candidate_programs(signature)

            if not candidates:
                logger.debug("No candidate programs found")
                return None, 0.0

            # Score and rank candidates
            best_match = self.score_and_rank_candidates(candidates, signature, query, context)

            if best_match:
                # Get the match confidence score
                match_confidence = getattr(best_match, 'match_score', best_match.confidence_score)

                if match_confidence >= self.execution_threshold:
                    logger.info(f"Found matching program: {best_match.id} (confidence: {match_confidence:.2f})")
                    return best_match, match_confidence
                else:
                    logger.debug(f"Best match confidence {match_confidence:.2f} below threshold {self.execution_threshold}")
                    return None, match_confidence

            return None, 0.0

        except Exception as e:
            logger.error(f"Error finding matching program: {e}")
            return None, 0.0
    
    def find_candidate_programs(self, signature: ProgramSignature) -> List[LatentProgram]:
        """Find candidate programs based on signature similarity."""
        try:
            # First try exact signature hash match
            exact_matches = self.store.get_programs_by_signature_hash(signature.signature_hash)
            
            if exact_matches:
                logger.debug(f"Found {len(exact_matches)} exact signature matches")
                return exact_matches
            
            # Fall back to similarity search
            similar_programs = self.store.find_similar_programs(
                signature, 
                self.similarity_threshold,
                self.max_programs_per_signature
            )
            
            logger.debug(f"Found {len(similar_programs)} similar programs")
            return similar_programs
            
        except Exception as e:
            logger.error(f"Error finding candidate programs: {e}")
            return []
    
    def score_and_rank_candidates(self, candidates: List[LatentProgram],
                                signature: ProgramSignature, query: str,
                                context: Dict[str, Any]) -> Optional[LatentProgram]:
        """Score and rank candidate programs."""
        try:
            if not candidates:
                return None
            
            scored_candidates = []
            
            for program in candidates:
                # Calculate composite score
                score = self._calculate_program_score(program, signature, query, context)
                scored_candidates.append((program, score))
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Return the best candidate
            best_program, best_score = scored_candidates[0]
            best_program.match_score = best_score  # Add score for reference
            
            return best_program
            
        except Exception as e:
            logger.error(f"Error scoring candidates: {e}")
            return candidates[0] if candidates else None
    
    def _calculate_program_score(self, program: LatentProgram, signature: ProgramSignature,
                               query: str, context: Dict[str, Any]) -> float:
        """Calculate a composite score for program matching."""
        try:
            factors = []
            
            # Signature similarity (if available)
            if hasattr(program, 'similarity_score'):
                factors.append(('similarity', program.similarity_score, 0.4))
            
            # Program confidence
            factors.append(('confidence', program.confidence_score, 0.3))
            
            # Success rate
            factors.append(('success_rate', program.success_rate, 0.2))
            
            # Usage frequency (normalized)
            usage_score = min(1.0, program.usage_count / 10.0)
            factors.append(('usage', usage_score, 0.1))
            
            # Calculate weighted score
            total_score = sum(score * weight for _, score, weight in factors)
            
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            logger.warning(f"Error calculating program score: {e}")
            return program.confidence_score
    
    def execute_program(self, program: LatentProgram, query: str,
                       context: Dict[str, Any]) -> ExecutionResult:
        """Execute a latent program with enhanced analytics tracking."""
        try:
            logger.info(f"Executing program {program.id}")

            # Record baseline execution time for efficiency calculation
            baseline_start = time.time()

            # Execute with monitoring (preserving 100% of existing functionality)
            result = self.executor.execute_with_monitoring(program, query, context)

            # Calculate baseline execution time (if this was executed without program)
            baseline_time_ms = context.get('baseline_execution_time_ms', result.execution_time_ms * 1.5)

            # Enhanced Analytics Collection (Phase 1A.2 - preserving 100% of existing functionality)
            if self.analytics_enabled and self.metrics_collector:
                # Collect detailed execution metrics
                execution_context = {
                    **context,
                    'baseline_execution_time_ms': baseline_time_ms,
                    'tpv_enabled': context.get('tpv_enabled', False),
                    'query_text': query
                }
                self.metrics_collector.on_program_execution(program, result, execution_context)

            # Update program performance metrics (preserving existing functionality)
            self.store.update_program_performance(
                program.id,
                result.execution_time_ms,
                result.token_count,
                result.success,
                result.quality_score
            )

            # Enhanced analytics data collection
            if self.analytics_enabled and self.analytics_engine:
                execution_data = {
                    'execution_time_ms': result.execution_time_ms,
                    'quality_score': result.quality_score,
                    'success': result.success,
                    'token_count': result.token_count,
                    'baseline_time_ms': baseline_time_ms,
                    'user_profile': program.active_profile,
                    'context_hash': self._calculate_context_hash(context),
                    'tpv_used': context.get('tpv_enabled', False)
                }
                self.analytics_engine.collect_execution_metrics(program.id, execution_data)

            return result

        except Exception as e:
            logger.error(f"Program execution failed: {e}")

            # Create error result
            error_result = ExecutionResult(
                response="",
                quality_score=0.0,
                execution_time_ms=0.0,
                program_used=program.id,
                success=False,
                error_message=str(e)
            )

            # Collect error metrics if analytics enabled
            if self.analytics_enabled and self.metrics_collector:
                self.metrics_collector.on_program_execution(program, error_result, context)

            return error_result
    
    def consider_program_capture(self, query: str, context: Dict[str, Any],
                               result: Dict[str, Any], user_profile: Optional[str] = None) -> bool:
        """Consider capturing a new program from a successful interaction."""
        try:
            # Check if this was a high-quality response
            if not self._should_capture_program(result):
                logger.debug("Response quality insufficient for program capture")
                return False
            
            # Generate signature
            signature = generate_signature(query, context, user_profile)
            
            # Check if we already have similar programs
            existing_programs = self.store.find_similar_programs(signature, 0.9, 3)
            
            if len(existing_programs) >= self.max_programs_per_signature:
                logger.debug("Maximum programs per signature reached")
                return False
            
            # Create new program
            program = self._create_program_from_interaction(
                signature, query, context, result, user_profile
            )
            
            # Validate program safety if validator is available
            if self.validator:
                validation = self.validator.validate_program_safety(program)
                if not validation.is_safe:
                    logger.warning(f"Program failed safety validation: {validation.warnings}")
                    return False
            else:
                logger.debug("Program validator not available, skipping safety validation")
            
            # Store the program
            success = self.store.store_program(program)
            
            if success:
                logger.info(f"Captured new program: {program.id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error considering program capture: {e}")
            return False
    
    def _should_capture_program(self, result: Dict[str, Any]) -> bool:
        """Determine if a result is worth capturing as a program."""
        # Check quality indicators
        quality_score = result.get('quality_score', 0.0)
        user_feedback = result.get('user_feedback', 0.0)
        meta_confidence = result.get('meta_confidence', 0.0)
        
        # Require minimum quality thresholds
        if quality_score < self.quality_threshold:
            return False
        
        if user_feedback > 0 and user_feedback < 3.0:  # Negative feedback
            return False
        
        if meta_confidence > 0 and meta_confidence < 0.6:
            return False
        
        return True
    
    def _create_program_from_interaction(self, signature: ProgramSignature,
                                       query: str, context: Dict[str, Any],
                                       result: Dict[str, Any],
                                       user_profile: Optional[str]) -> LatentProgram:
        """Create a new latent program from a successful interaction."""
        
        # Extract configuration that led to success
        tpv_config = result.get('tpv_config', {})
        prompt_template = result.get('prompt_template', '')
        reasoning_trace = result.get('reasoning_trace', {})
        
        # Create context requirements
        context_requirements = {
            'document_types': signature.document_types,
            'content_domains': signature.content_domains,
            'min_context_size': len(str(context))
        }
        
        # Create execution constraints
        execution_constraints = {
            'max_tokens': result.get('token_count', 1000),
            'timeout_seconds': 30
        }
        
        # Create the program
        program = LatentProgram(
            signature=signature.to_dict(),
            reasoning_trace=reasoning_trace,
            tpv_config=tpv_config,
            active_profile=user_profile or 'default',
            prompt_template_used=prompt_template,
            context_requirements=context_requirements,
            execution_constraints=execution_constraints,
            avg_latency_ms=result.get('execution_time_ms', 0.0),
            avg_token_count=result.get('token_count', 0),
            user_feedback_score=result.get('user_feedback', 0.0)
        )

        # Enhanced Pattern Discovery Logging (Phase 1A.2 - preserving 100% of existing functionality)
        if self.analytics_enabled and self.metrics_collector:
            pattern_data = {
                'pattern_type': signature.primary_intent,
                'signature_hash': signature.signature_hash,
                'capture_success': True,
                'similarity_score': 0.0,  # New pattern, no similarity
                'user_context': json.dumps(context),
                'query_text': query,
                'response_quality': result.get('quality_score', 0.0),
                'capture_reason': 'quality_threshold_met',
                'program_id': program.id,
                'user_profile': user_profile or 'default',
                'complexity_level': signature.complexity_level,
                'domain_category': self._classify_domain(context)
            }
            self.metrics_collector.on_pattern_capture(pattern_data, True)

        return program
    
    def record_user_feedback(self, program_id: str, feedback_score: float) -> bool:
        """Record user feedback for a program."""
        try:
            return self.store.update_program_performance(
                program_id, 0, 0, True, None, feedback_score
            )
        except Exception as e:
            logger.error(f"Error recording user feedback: {e}")
            return False
    
    def retire_poor_performers(self, min_usage: int = 5, 
                             success_threshold: float = 0.3) -> int:
        """Retire programs that perform poorly."""
        try:
            programs = self.store.get_all_programs()
            retired_count = 0
            
            for program in programs:
                if (program.usage_count >= min_usage and 
                    program.success_rate < success_threshold):
                    
                    if self.store.retire_program(program.id):
                        retired_count += 1
                        logger.info(f"Retired poor performing program: {program.id}")
            
            return retired_count
            
        except Exception as e:
            logger.error(f"Error retiring poor performers: {e}")
            return 0
    
    def get_program_statistics(self) -> Dict[str, Any]:
        """Get comprehensive program statistics."""
        try:
            stats = self.store.get_program_statistics()
            
            # Add additional computed statistics
            programs = self.store.get_all_programs()
            
            if programs:
                # Performance distribution
                confidence_scores = [p.confidence_score for p in programs]
                stats['confidence_distribution'] = {
                    'min': min(confidence_scores),
                    'max': max(confidence_scores),
                    'avg': sum(confidence_scores) / len(confidence_scores)
                }
                
                # Usage distribution
                usage_counts = [p.usage_count for p in programs]
                stats['usage_distribution'] = {
                    'total': sum(usage_counts),
                    'avg': sum(usage_counts) / len(usage_counts),
                    'max': max(usage_counts)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting program statistics: {e}")
            return {}
    
    def cleanup_old_programs(self, days_unused: int = 30) -> int:
        """Clean up old, unused programs."""
        try:
            return self.store.cleanup_old_programs(days_unused)
        except Exception as e:
            logger.error(f"Error cleaning up old programs: {e}")
            return 0

    # Enhanced Analytics Helper Methods (Phase 1A.2 - preserving 100% of existing functionality)

    def _calculate_context_hash(self, context: Dict[str, Any]) -> str:
        """Calculate hash of context for pattern analysis."""
        try:
            # Create a simplified context representation for hashing
            context_str = str(sorted(context.items()))
            return hashlib.md5(context_str.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"

    def _classify_domain(self, context: Dict[str, Any]) -> str:
        """Classify the domain category based on context."""
        try:
            if context.get('documents'):
                return "document_analysis"
            elif context.get('web_search'):
                return "web_research"
            elif context.get('calculation'):
                return "computation"
            elif context.get('code'):
                return "programming"
            else:
                return "general"
        except Exception:
            return "unknown"

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        try:
            summary = {}

            # Real-time metrics from collector
            if self.metrics_collector:
                summary['real_time'] = self.metrics_collector.get_real_time_stats()

            # Performance insights from analytics engine
            if self.analytics_engine:
                summary['insights'] = self.analytics_engine.generate_performance_insights()
                summary['automation_opportunities'] = self.analytics_engine.detect_automation_opportunities()

            # Basic program statistics
            summary['program_stats'] = self.store.get_program_statistics()

            return summary

        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}

    def enable_analytics(self, enable: bool = True):
        """Enable or disable analytics collection."""
        try:
            self.analytics_enabled = enable

            if enable and self.metrics_collector and not self.metrics_collector._collection_thread:
                self.metrics_collector.start_collection()
            elif not enable and self.metrics_collector:
                self.metrics_collector.stop_collection()

            logger.info(f"Analytics {'enabled' if enable else 'disabled'}")

        except Exception as e:
            logger.error(f"Failed to toggle analytics: {e}")

    def record_user_feedback_enhanced(self, program_id: str, feedback: int, context: Optional[Dict[str, Any]] = None):
        """Record user feedback with enhanced analytics tracking."""
        try:
            # Record in existing system (preserving 100% of functionality)
            success = self.record_user_feedback(program_id, float(feedback))

            # Enhanced analytics collection
            if self.analytics_enabled and self.metrics_collector:
                self.metrics_collector.on_user_feedback(program_id, feedback, context)

            return success

        except Exception as e:
            logger.error(f"Failed to record enhanced user feedback: {e}")
            return False
