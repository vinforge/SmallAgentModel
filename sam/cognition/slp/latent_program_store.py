"""
Latent Program Store
===================

Database layer for storing and retrieving latent programs with SQLite backend
and efficient querying capabilities.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .latent_program import LatentProgram
from .program_signature import ProgramSignature

logger = logging.getLogger(__name__)


class LatentProgramStore:
    """
    SQLite-based storage system for latent programs.
    
    Provides efficient storage, retrieval, and querying of cognitive programs
    with support for similarity matching and performance analytics.
    """
    
    def __init__(self, db_path: str = "data/latent_programs.db"):
        """Initialize the program store."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS latent_programs (
                        id TEXT PRIMARY KEY,
                        signature_hash TEXT NOT NULL,
                        signature_data TEXT NOT NULL,
                        program_data TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        last_used TIMESTAMP NOT NULL,
                        usage_count INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 1.0,
                        confidence_score REAL DEFAULT 0.5,
                        user_feedback_score REAL DEFAULT 0.0,
                        is_active BOOLEAN DEFAULT 1,
                        is_experimental BOOLEAN DEFAULT 1,
                        version INTEGER DEFAULT 1,
                        parent_program_id TEXT,
                        avg_latency_ms REAL DEFAULT 0.0,
                        avg_token_count INTEGER DEFAULT 0
                    )
                """)
                
                # Create indexes for efficient querying
                conn.execute("CREATE INDEX IF NOT EXISTS idx_signature_hash ON latent_programs(signature_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_active ON latent_programs(is_active)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON latent_programs(confidence_score)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_last_used ON latent_programs(last_used)")
                
                # Create analytics table for performance tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS program_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        program_id TEXT NOT NULL,
                        executed_at TIMESTAMP NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        token_count INTEGER NOT NULL,
                        success BOOLEAN NOT NULL,
                        quality_score REAL,
                        user_feedback REAL,
                        FOREIGN KEY (program_id) REFERENCES latent_programs(id)
                    )
                """)
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_executions_program ON program_executions(program_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_executions_time ON program_executions(executed_at)")

                # Enhanced Analytics Tables (Phase 1A.1 - preserving 100% of existing functionality)

                # Advanced program analytics for detailed performance tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS program_analytics_enhanced (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        program_id TEXT REFERENCES latent_programs(id),
                        execution_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        execution_time_ms REAL NOT NULL,
                        quality_score REAL DEFAULT 0.0,
                        user_feedback INTEGER DEFAULT 0,
                        context_hash TEXT,
                        tpv_used BOOLEAN DEFAULT 0,
                        efficiency_gain REAL DEFAULT 0.0,
                        token_count INTEGER DEFAULT 0,
                        user_profile TEXT,
                        query_type TEXT,
                        success BOOLEAN DEFAULT 1,
                        error_message TEXT,
                        baseline_time_ms REAL DEFAULT 0.0,
                        confidence_at_execution REAL DEFAULT 0.0,
                        memory_usage_mb REAL DEFAULT 0.0,
                        cpu_usage_percent REAL DEFAULT 0.0
                    )
                """)

                # Pattern discovery tracking for learning insights
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_discovery_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        discovery_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        pattern_type TEXT NOT NULL,
                        signature_hash TEXT NOT NULL,
                        capture_success BOOLEAN NOT NULL,
                        similarity_score REAL DEFAULT 0.0,
                        user_context TEXT,
                        query_text TEXT,
                        response_quality REAL DEFAULT 0.0,
                        capture_reason TEXT,
                        program_id TEXT,
                        user_profile TEXT,
                        complexity_level TEXT,
                        domain_category TEXT
                    )
                """)

                # System-wide performance metrics for trend analysis
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS slp_performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_programs INTEGER DEFAULT 0,
                        active_programs INTEGER DEFAULT 0,
                        hit_rate REAL DEFAULT 0.0,
                        avg_execution_time_ms REAL DEFAULT 0.0,
                        total_time_saved_ms REAL DEFAULT 0.0,
                        user_satisfaction_score REAL DEFAULT 0.0,
                        programs_captured_today INTEGER DEFAULT 0,
                        programs_executed_today INTEGER DEFAULT 0,
                        efficiency_improvement REAL DEFAULT 0.0,
                        system_load REAL DEFAULT 0.0,
                        memory_usage_mb REAL DEFAULT 0.0,
                        cache_hit_rate REAL DEFAULT 0.0
                    )
                """)

                # User-specific performance tracking for personalization
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_slp_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_profile TEXT NOT NULL,
                        metric_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        personal_hit_rate REAL DEFAULT 0.0,
                        personal_time_saved_ms REAL DEFAULT 0.0,
                        preferred_program_types TEXT,
                        automation_opportunities TEXT,
                        satisfaction_trend REAL DEFAULT 0.0,
                        learning_velocity REAL DEFAULT 0.0,
                        program_usage_patterns TEXT,
                        personalization_score REAL DEFAULT 0.0,
                        adaptation_rate REAL DEFAULT 0.0
                    )
                """)

                # Cross-program relationship tracking for pattern analysis
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS program_relationships (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        program_a_id TEXT REFERENCES latent_programs(id),
                        program_b_id TEXT REFERENCES latent_programs(id),
                        relationship_type TEXT NOT NULL,
                        similarity_score REAL DEFAULT 0.0,
                        usage_correlation REAL DEFAULT 0.0,
                        discovered_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        relationship_strength REAL DEFAULT 0.0,
                        co_occurrence_count INTEGER DEFAULT 0,
                        temporal_distance_avg REAL DEFAULT 0.0
                    )
                """)

                # Create indexes for enhanced analytics tables
                conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_program ON program_analytics_enhanced(program_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON program_analytics_enhanced(execution_timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_user ON program_analytics_enhanced(user_profile)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_discovery_timestamp ON pattern_discovery_log(discovery_timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_discovery_signature ON pattern_discovery_log(signature_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON slp_performance_metrics(metric_timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user_analytics_profile ON user_slp_analytics(user_profile)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user_analytics_timestamp ON user_slp_analytics(metric_timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_programs ON program_relationships(program_a_id, program_b_id)")

                conn.commit()
                logger.info("Latent program database with enhanced analytics initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def store_program(self, program: LatentProgram) -> bool:
        """Store a latent program in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert program to JSON
                program_data = json.dumps(program.to_dict())
                signature_data = json.dumps(program.signature)
                
                conn.execute("""
                    INSERT OR REPLACE INTO latent_programs (
                        id, signature_hash, signature_data, program_data,
                        created_at, last_used, usage_count, success_rate,
                        confidence_score, user_feedback_score, is_active,
                        is_experimental, version, parent_program_id,
                        avg_latency_ms, avg_token_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program.id,
                    program.signature.get('signature_hash', ''),
                    signature_data,
                    program_data,
                    program.created_at.isoformat(),
                    program.last_used.isoformat(),
                    program.usage_count,
                    program.success_rate,
                    program.confidence_score,
                    program.user_feedback_score,
                    program.is_active,
                    program.is_experimental,
                    program.version,
                    program.parent_program_id,
                    program.avg_latency_ms,
                    program.avg_token_count
                ))
                
                conn.commit()
                logger.debug(f"Stored program {program.id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store program {program.id}: {e}")
            return False
    
    def get_program(self, program_id: str) -> Optional[LatentProgram]:
        """Retrieve a specific program by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT program_data FROM latent_programs WHERE id = ? AND is_active = 1",
                    (program_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    program_data = json.loads(row[0])
                    return LatentProgram.from_dict(program_data)
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve program {program_id}: {e}")
            return None
    
    def find_similar_programs(self, signature: ProgramSignature, 
                            similarity_threshold: float = 0.8,
                            max_results: int = 5) -> List[LatentProgram]:
        """Find programs with similar signatures."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First, get all active programs
                cursor = conn.execute("""
                    SELECT program_data, signature_data FROM latent_programs 
                    WHERE is_active = 1 
                    ORDER BY confidence_score DESC, usage_count DESC
                """)
                
                similar_programs = []
                
                for row in cursor.fetchall():
                    try:
                        program_data = json.loads(row[0])
                        signature_data = json.loads(row[1])
                        
                        # Create signature object for comparison
                        stored_signature = ProgramSignature.from_dict(signature_data)
                        
                        # Calculate similarity
                        similarity = signature.calculate_similarity(stored_signature)
                        
                        if similarity >= similarity_threshold:
                            program = LatentProgram.from_dict(program_data)
                            program.similarity_score = similarity  # Add similarity for ranking
                            similar_programs.append(program)
                            
                    except Exception as e:
                        logger.warning(f"Error processing program in similarity search: {e}")
                        continue
                
                # Sort by similarity and confidence, return top results
                similar_programs.sort(
                    key=lambda p: (p.similarity_score, p.confidence_score), 
                    reverse=True
                )
                
                return similar_programs[:max_results]
                
        except Exception as e:
            logger.error(f"Failed to find similar programs: {e}")
            return []
    
    def get_programs_by_signature_hash(self, signature_hash: str) -> List[LatentProgram]:
        """Get programs with exact signature hash match."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT program_data FROM latent_programs 
                    WHERE signature_hash = ? AND is_active = 1
                    ORDER BY confidence_score DESC
                """, (signature_hash,))
                
                programs = []
                for row in cursor.fetchall():
                    program_data = json.loads(row[0])
                    programs.append(LatentProgram.from_dict(program_data))
                
                return programs
                
        except Exception as e:
            logger.error(f"Failed to get programs by signature hash: {e}")
            return []
    
    def update_program_performance(self, program_id: str, execution_time_ms: float,
                                 token_count: int, success: bool = True,
                                 quality_score: Optional[float] = None,
                                 user_feedback: Optional[float] = None) -> bool:
        """Update program performance metrics."""
        try:
            # First, record the execution
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO program_executions (
                        program_id, executed_at, execution_time_ms, token_count,
                        success, quality_score, user_feedback
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_id,
                    datetime.utcnow().isoformat(),
                    execution_time_ms,
                    token_count,
                    success,
                    quality_score,
                    user_feedback
                ))
                
                # Update the program's aggregated metrics
                program = self.get_program(program_id)
                if program:
                    program.update_performance_metrics(
                        execution_time_ms, token_count, success, user_feedback
                    )
                    return self.store_program(program)
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to update program performance: {e}")
            return False
    
    def retire_program(self, program_id: str) -> bool:
        """Mark a program as inactive (retired)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE latent_programs SET is_active = 0 WHERE id = ?",
                    (program_id,)
                )
                conn.commit()
                logger.info(f"Retired program {program_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to retire program {program_id}: {e}")
            return False
    
    def get_all_programs(self, include_inactive: bool = False) -> List[LatentProgram]:
        """Get all programs from the store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT program_data FROM latent_programs"
                if not include_inactive:
                    query += " WHERE is_active = 1"
                query += " ORDER BY last_used DESC"
                
                cursor = conn.execute(query)
                programs = []
                
                for row in cursor.fetchall():
                    program_data = json.loads(row[0])
                    programs.append(LatentProgram.from_dict(program_data))
                
                return programs
                
        except Exception as e:
            logger.error(f"Failed to get all programs: {e}")
            return []
    
    def get_program_statistics(self) -> Dict[str, Any]:
        """Get statistics about the program store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Basic counts
                cursor = conn.execute("SELECT COUNT(*) FROM latent_programs WHERE is_active = 1")
                active_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM latent_programs WHERE is_experimental = 1 AND is_active = 1")
                experimental_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT AVG(confidence_score) FROM latent_programs WHERE is_active = 1")
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                cursor = conn.execute("SELECT SUM(usage_count) FROM latent_programs WHERE is_active = 1")
                total_usage = cursor.fetchone()[0] or 0
                
                cursor = conn.execute("SELECT AVG(avg_latency_ms) FROM latent_programs WHERE is_active = 1")
                avg_latency = cursor.fetchone()[0] or 0.0
                
                return {
                    'active_programs': active_count,
                    'experimental_programs': experimental_count,
                    'proven_programs': active_count - experimental_count,
                    'average_confidence': avg_confidence,
                    'total_usage': total_usage,
                    'average_latency_ms': avg_latency
                }
                
        except Exception as e:
            logger.error(f"Failed to get program statistics: {e}")
            return {}
    
    def cleanup_old_programs(self, days_unused: int = 30) -> int:
        """Remove programs that haven't been used in specified days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_unused)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM latent_programs 
                    WHERE last_used < ? AND usage_count < 5
                """, (cutoff_date.isoformat(),))
                
                count = cursor.fetchone()[0]
                
                conn.execute("""
                    DELETE FROM latent_programs 
                    WHERE last_used < ? AND usage_count < 5
                """, (cutoff_date.isoformat(),))
                
                conn.commit()
                logger.info(f"Cleaned up {count} old programs")
                return count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old programs: {e}")
            return 0

    # Enhanced Analytics Methods (Phase 1A.1 - preserving 100% of existing functionality)

    def record_enhanced_execution(self, program_id: str, execution_data: Dict[str, Any]) -> bool:
        """Record detailed execution analytics for enhanced tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO program_analytics_enhanced (
                        program_id, execution_time_ms, quality_score, user_feedback,
                        context_hash, tpv_used, efficiency_gain, token_count,
                        user_profile, query_type, success, error_message,
                        baseline_time_ms, confidence_at_execution, memory_usage_mb, cpu_usage_percent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_id,
                    execution_data.get('execution_time_ms', 0.0),
                    execution_data.get('quality_score', 0.0),
                    execution_data.get('user_feedback', 0),
                    execution_data.get('context_hash', ''),
                    execution_data.get('tpv_used', False),
                    execution_data.get('efficiency_gain', 0.0),
                    execution_data.get('token_count', 0),
                    execution_data.get('user_profile', 'default'),
                    execution_data.get('query_type', 'general'),
                    execution_data.get('success', True),
                    execution_data.get('error_message', ''),
                    execution_data.get('baseline_time_ms', 0.0),
                    execution_data.get('confidence_at_execution', 0.0),
                    execution_data.get('memory_usage_mb', 0.0),
                    execution_data.get('cpu_usage_percent', 0.0)
                ))
                conn.commit()
                logger.debug(f"Recorded enhanced execution analytics for program {program_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to record enhanced execution analytics: {e}")
            return False

    def log_pattern_discovery(self, discovery_data: Dict[str, Any]) -> bool:
        """Log pattern discovery events for learning insights."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO pattern_discovery_log (
                        pattern_type, signature_hash, capture_success, similarity_score,
                        user_context, query_text, response_quality, capture_reason,
                        program_id, user_profile, complexity_level, domain_category
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    discovery_data.get('pattern_type', 'unknown'),
                    discovery_data.get('signature_hash', ''),
                    discovery_data.get('capture_success', False),
                    discovery_data.get('similarity_score', 0.0),
                    discovery_data.get('user_context', ''),
                    discovery_data.get('query_text', ''),
                    discovery_data.get('response_quality', 0.0),
                    discovery_data.get('capture_reason', ''),
                    discovery_data.get('program_id', ''),
                    discovery_data.get('user_profile', 'default'),
                    discovery_data.get('complexity_level', 'medium'),
                    discovery_data.get('domain_category', 'general')
                ))
                conn.commit()
                logger.debug(f"Logged pattern discovery: {discovery_data.get('pattern_type', 'unknown')}")
                return True

        except Exception as e:
            logger.error(f"Failed to log pattern discovery: {e}")
            return False

    def record_system_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """Record system-wide performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO slp_performance_metrics (
                        total_programs, active_programs, hit_rate, avg_execution_time_ms,
                        total_time_saved_ms, user_satisfaction_score, programs_captured_today,
                        programs_executed_today, efficiency_improvement, system_load,
                        memory_usage_mb, cache_hit_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics_data.get('total_programs', 0),
                    metrics_data.get('active_programs', 0),
                    metrics_data.get('hit_rate', 0.0),
                    metrics_data.get('avg_execution_time_ms', 0.0),
                    metrics_data.get('total_time_saved_ms', 0.0),
                    metrics_data.get('user_satisfaction_score', 0.0),
                    metrics_data.get('programs_captured_today', 0),
                    metrics_data.get('programs_executed_today', 0),
                    metrics_data.get('efficiency_improvement', 0.0),
                    metrics_data.get('system_load', 0.0),
                    metrics_data.get('memory_usage_mb', 0.0),
                    metrics_data.get('cache_hit_rate', 0.0)
                ))
                conn.commit()
                logger.debug("Recorded system performance metrics")
                return True

        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
            return False

    def record_user_analytics(self, user_profile: str, analytics_data: Dict[str, Any]) -> bool:
        """Record user-specific analytics for personalization."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_slp_analytics (
                        user_profile, personal_hit_rate, personal_time_saved_ms,
                        preferred_program_types, automation_opportunities, satisfaction_trend,
                        learning_velocity, program_usage_patterns, personalization_score, adaptation_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_profile,
                    analytics_data.get('personal_hit_rate', 0.0),
                    analytics_data.get('personal_time_saved_ms', 0.0),
                    json.dumps(analytics_data.get('preferred_program_types', [])),
                    json.dumps(analytics_data.get('automation_opportunities', [])),
                    analytics_data.get('satisfaction_trend', 0.0),
                    analytics_data.get('learning_velocity', 0.0),
                    json.dumps(analytics_data.get('program_usage_patterns', {})),
                    analytics_data.get('personalization_score', 0.0),
                    analytics_data.get('adaptation_rate', 0.0)
                ))
                conn.commit()
                logger.debug(f"Recorded user analytics for {user_profile}")
                return True

        except Exception as e:
            logger.error(f"Failed to record user analytics: {e}")
            return False

    def record_program_relationship(self, program_a_id: str, program_b_id: str,
                                  relationship_data: Dict[str, Any]) -> bool:
        """Record relationships between programs for pattern analysis."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO program_relationships (
                        program_a_id, program_b_id, relationship_type, similarity_score,
                        usage_correlation, relationship_strength, co_occurrence_count, temporal_distance_avg
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_a_id,
                    program_b_id,
                    relationship_data.get('relationship_type', 'similar'),
                    relationship_data.get('similarity_score', 0.0),
                    relationship_data.get('usage_correlation', 0.0),
                    relationship_data.get('relationship_strength', 0.0),
                    relationship_data.get('co_occurrence_count', 0),
                    relationship_data.get('temporal_distance_avg', 0.0)
                ))
                conn.commit()
                logger.debug(f"Recorded program relationship: {program_a_id} -> {program_b_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to record program relationship: {e}")
            return False
