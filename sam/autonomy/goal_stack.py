"""
Enhanced GoalStack Manager for SAM Autonomy
===========================================

This module implements the GoalStack class for persistent goal management
with enhanced lifecycle controls, deduplication, priority decay, and archiving.

Author: SAM Development Team
Version: 2.0.0
"""

import sqlite3
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager

from .goals import Goal
from .safety.goal_validator import GoalSafetyValidator

logger = logging.getLogger(__name__)

class GoalStack:
    """
    Enhanced persistent goal management system with lifecycle controls.
    
    Features:
    - SQLite-based persistent storage
    - Goal deduplication
    - Priority decay mechanism
    - Automatic archiving
    - Thread-safe operations
    - Caching for performance
    - Comprehensive logging
    - Safety validation integration
    """
    
    def __init__(self, db_path: str = "memory/autonomy_goals.db", 
                 safety_validator: Optional[GoalSafetyValidator] = None):
        """
        Initialize the GoalStack manager.
        
        Args:
            db_path: Path to SQLite database file
            safety_validator: Optional safety validator instance
        """
        self.logger = logging.getLogger(f"{__name__}.GoalStack")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Safety validation
        self.safety_validator = safety_validator or GoalSafetyValidator()
        
        # Configuration
        self.config = {
            'priority_decay_rate': 0.01,  # Daily decay rate
            'archive_after_days': 30,     # Archive completed/failed goals after N days
            'max_active_goals': 100,      # Maximum active goals
            'deduplication_threshold': 0.8,  # Similarity threshold for deduplication
            'cache_ttl_seconds': 60       # Cache TTL for goal retrieval
        }
        
        # Initialize database
        self._init_database()
        
        # Cache for performance
        self._cache_timestamp = datetime.now()
        
        self.logger.info(f"GoalStack initialized with database: {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        with self._get_db_connection() as conn:
            # Main goals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority REAL NOT NULL,
                    source_skill TEXT NOT NULL,
                    source_context TEXT NOT NULL,
                    creation_timestamp TEXT NOT NULL,
                    last_updated_timestamp TEXT NOT NULL,
                    estimated_effort REAL,
                    dependencies TEXT,
                    max_attempts INTEGER NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    failure_reason TEXT,
                    tags TEXT
                )
            """)
            
            # Archive table for completed/failed goals
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals_archive (
                    goal_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority REAL NOT NULL,
                    source_skill TEXT NOT NULL,
                    source_context TEXT NOT NULL,
                    creation_timestamp TEXT NOT NULL,
                    last_updated_timestamp TEXT NOT NULL,
                    estimated_effort REAL,
                    dependencies TEXT,
                    max_attempts INTEGER NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    failure_reason TEXT,
                    tags TEXT,
                    archived_timestamp TEXT NOT NULL
                )
            """)
            
            # Metadata table for tracking statistics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goal_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_timestamp TEXT NOT NULL
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_priority ON goals(priority DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_source_skill ON goals(source_skill)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_creation_time ON goals(creation_timestamp)")
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def add_goal(self, goal: Goal) -> bool:
        """
        Add a new goal to the stack with safety validation and deduplication.
        
        Args:
            goal: Goal to add
            
        Returns:
            True if goal was added successfully, False otherwise
        """
        with self._lock:
            try:
                # Safety validation
                is_valid, error_msg = self.safety_validator.validate_goal(goal)
                if not is_valid:
                    self.logger.warning(f"Goal rejected by safety validator: {error_msg}")
                    return False
                
                # Check for duplicates
                if self._is_duplicate_goal(goal):
                    self.logger.info(f"Duplicate goal detected, skipping: {goal.goal_id}")
                    return False
                
                # Check active goal limit
                active_count = self._get_active_goal_count()
                if active_count >= self.config['max_active_goals']:
                    self.logger.warning(f"Maximum active goals reached ({self.config['max_active_goals']})")
                    return False
                
                # Add to database
                with self._get_db_connection() as conn:
                    conn.execute("""
                        INSERT INTO goals (
                            goal_id, description, status, priority, source_skill,
                            source_context, creation_timestamp, last_updated_timestamp,
                            estimated_effort, dependencies, max_attempts, attempt_count,
                            failure_reason, tags
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        goal.goal_id,
                        goal.description,
                        goal.status,
                        goal.priority,
                        goal.source_skill,
                        json.dumps(goal.source_context),
                        goal.creation_timestamp.isoformat(),
                        goal.last_updated_timestamp.isoformat(),
                        goal.estimated_effort,
                        json.dumps(goal.dependencies),
                        goal.max_attempts,
                        goal.attempt_count,
                        goal.failure_reason,
                        json.dumps(goal.tags)
                    ))
                    conn.commit()
                
                # Clear cache
                self._clear_cache()
                
                self.logger.info(f"Goal added successfully: {goal.goal_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error adding goal: {e}")
                return False
    
    @lru_cache(maxsize=128)
    def get_top_priority_goals(self, limit: int = 3, status: str = "pending") -> List[Goal]:
        """
        Get the highest-priority goals with caching.
        
        Args:
            limit: Maximum number of goals to return
            status: Goal status to filter by
            
        Returns:
            List of top priority goals
        """
        # Check cache validity
        if (datetime.now() - self._cache_timestamp).total_seconds() > self.config['cache_ttl_seconds']:
            self.get_top_priority_goals.cache_clear()
            self._cache_timestamp = datetime.now()
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM goals 
                    WHERE status = ? 
                    ORDER BY priority DESC, creation_timestamp ASC 
                    LIMIT ?
                """, (status, limit))
                
                goals = []
                for row in cursor.fetchall():
                    goal = self._row_to_goal(row)
                    goals.append(goal)
                
                return goals
                
        except Exception as e:
            self.logger.error(f"Error retrieving top priority goals: {e}")
            return []
    
    def update_goal_status(self, goal_id: str, status: str, 
                          failure_reason: Optional[str] = None) -> bool:
        """
        Update a goal's status.
        
        Args:
            goal_id: ID of goal to update
            status: New status
            failure_reason: Optional failure reason
            
        Returns:
            True if update successful, False otherwise
        """
        with self._lock:
            try:
                with self._get_db_connection() as conn:
                    # Update the goal
                    cursor = conn.execute("""
                        UPDATE goals 
                        SET status = ?, last_updated_timestamp = ?, failure_reason = ?
                        WHERE goal_id = ?
                    """, (status, datetime.now().isoformat(), failure_reason, goal_id))
                    
                    if cursor.rowcount == 0:
                        self.logger.warning(f"Goal not found for status update: {goal_id}")
                        return False
                    
                    conn.commit()
                
                # Clear cache
                self._clear_cache()
                
                self.logger.info(f"Goal status updated: {goal_id} -> {status}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error updating goal status: {e}")
                return False
    
    def get_all_goals(self, include_archived: bool = False) -> List[Goal]:
        """
        Get all goals, optionally including archived ones.
        
        Args:
            include_archived: Whether to include archived goals
            
        Returns:
            List of all goals
        """
        try:
            goals = []
            
            with self._get_db_connection() as conn:
                # Get active goals
                cursor = conn.execute("SELECT * FROM goals ORDER BY priority DESC")
                for row in cursor.fetchall():
                    goals.append(self._row_to_goal(row))
                
                # Get archived goals if requested
                if include_archived:
                    cursor = conn.execute("SELECT * FROM goals_archive ORDER BY archived_timestamp DESC")
                    for row in cursor.fetchall():
                        goal = self._row_to_goal(row)
                        goal.tags.append("archived")  # Mark as archived
                        goals.append(goal)
            
            return goals
            
        except Exception as e:
            self.logger.error(f"Error retrieving all goals: {e}")
            return []
    
    def decay_priorities(self) -> int:
        """
        Apply priority decay to old, unaddressed goals.
        
        Returns:
            Number of goals that had their priority decayed
        """
        with self._lock:
            try:
                decay_rate = self.config['priority_decay_rate']
                cutoff_time = datetime.now() - timedelta(days=1)
                
                with self._get_db_connection() as conn:
                    # Get goals that need priority decay
                    cursor = conn.execute("""
                        SELECT goal_id, priority FROM goals 
                        WHERE status = 'pending' 
                        AND last_updated_timestamp < ?
                        AND priority > 0.1
                    """, (cutoff_time.isoformat(),))
                    
                    goals_to_decay = cursor.fetchall()
                    
                    # Apply decay
                    decayed_count = 0
                    for row in goals_to_decay:
                        goal_id, current_priority = row['goal_id'], row['priority']
                        new_priority = max(0.1, current_priority - decay_rate)
                        
                        conn.execute("""
                            UPDATE goals 
                            SET priority = ?, last_updated_timestamp = ?
                            WHERE goal_id = ?
                        """, (new_priority, datetime.now().isoformat(), goal_id))
                        
                        decayed_count += 1
                    
                    conn.commit()
                
                # Clear cache
                self._clear_cache()
                
                if decayed_count > 0:
                    self.logger.info(f"Applied priority decay to {decayed_count} goals")
                
                return decayed_count
                
            except Exception as e:
                self.logger.error(f"Error applying priority decay: {e}")
                return 0

    def archive_completed_goals(self) -> int:
        """
        Archive completed and failed goals older than the configured threshold.

        Returns:
            Number of goals archived
        """
        with self._lock:
            try:
                cutoff_time = datetime.now() - timedelta(days=self.config['archive_after_days'])

                with self._get_db_connection() as conn:
                    # Get goals to archive
                    cursor = conn.execute("""
                        SELECT * FROM goals
                        WHERE status IN ('completed', 'failed')
                        AND last_updated_timestamp < ?
                    """, (cutoff_time.isoformat(),))

                    goals_to_archive = cursor.fetchall()
                    archived_count = 0

                    for row in goals_to_archive:
                        # Insert into archive table
                        conn.execute("""
                            INSERT INTO goals_archive (
                                goal_id, description, status, priority, source_skill,
                                source_context, creation_timestamp, last_updated_timestamp,
                                estimated_effort, dependencies, max_attempts, attempt_count,
                                failure_reason, tags, archived_timestamp
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row['goal_id'], row['description'], row['status'], row['priority'],
                            row['source_skill'], row['source_context'], row['creation_timestamp'],
                            row['last_updated_timestamp'], row['estimated_effort'], row['dependencies'],
                            row['max_attempts'], row['attempt_count'], row['failure_reason'],
                            row['tags'], datetime.now().isoformat()
                        ))

                        # Remove from active table
                        conn.execute("DELETE FROM goals WHERE goal_id = ?", (row['goal_id'],))
                        archived_count += 1

                    conn.commit()

                # Clear cache
                self._clear_cache()

                if archived_count > 0:
                    self.logger.info(f"Archived {archived_count} completed/failed goals")

                return archived_count

            except Exception as e:
                self.logger.error(f"Error archiving goals: {e}")
                return 0

    def _is_duplicate_goal(self, goal: Goal) -> bool:
        """Check if a similar goal already exists."""
        try:
            with self._get_db_connection() as conn:
                # Check for exact description match
                cursor = conn.execute("""
                    SELECT goal_id FROM goals
                    WHERE description = ? AND status IN ('pending', 'active')
                """, (goal.description,))

                if cursor.fetchone():
                    return True

                # Check for similar goals from same source skill
                cursor = conn.execute("""
                    SELECT description FROM goals
                    WHERE source_skill = ? AND status IN ('pending', 'active')
                """, (goal.source_skill,))

                existing_descriptions = [row['description'] for row in cursor.fetchall()]

                # Simple similarity check (could be enhanced with NLP)
                for existing_desc in existing_descriptions:
                    similarity = self._calculate_similarity(goal.description, existing_desc)
                    if similarity > self.config['deduplication_threshold']:
                        return True

                return False

        except Exception as e:
            self.logger.error(f"Error checking for duplicate goals: {e}")
            return False

    def _calculate_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate similarity between two descriptions (simple implementation)."""
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _get_active_goal_count(self) -> int:
        """Get count of active (pending/active status) goals."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) as count FROM goals
                    WHERE status IN ('pending', 'active')
                """)
                return cursor.fetchone()['count']
        except Exception as e:
            self.logger.error(f"Error getting active goal count: {e}")
            return 0

    def _row_to_goal(self, row) -> Goal:
        """Convert database row to Goal object."""
        return Goal(
            goal_id=row['goal_id'],
            description=row['description'],
            status=row['status'],
            priority=row['priority'],
            source_skill=row['source_skill'],
            source_context=json.loads(row['source_context']),
            creation_timestamp=datetime.fromisoformat(row['creation_timestamp']),
            last_updated_timestamp=datetime.fromisoformat(row['last_updated_timestamp']),
            estimated_effort=row['estimated_effort'],
            dependencies=json.loads(row['dependencies']),
            max_attempts=row['max_attempts'],
            attempt_count=row['attempt_count'],
            failure_reason=row['failure_reason'],
            tags=json.loads(row['tags'])
        )

    def _clear_cache(self) -> None:
        """Clear the LRU cache."""
        self.get_top_priority_goals.cache_clear()
        self._cache_timestamp = datetime.now()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the goal stack."""
        try:
            with self._get_db_connection() as conn:
                stats = {}

                # Goal counts by status
                cursor = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM goals
                    GROUP BY status
                """)
                stats['goals_by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}

                # Goal counts by source skill
                cursor = conn.execute("""
                    SELECT source_skill, COUNT(*) as count
                    FROM goals
                    GROUP BY source_skill
                    ORDER BY count DESC
                """)
                stats['goals_by_source'] = {row['source_skill']: row['count'] for row in cursor.fetchall()}

                # Archive statistics
                cursor = conn.execute("SELECT COUNT(*) as count FROM goals_archive")
                stats['archived_goals'] = cursor.fetchone()['count']

                # Priority distribution
                cursor = conn.execute("""
                    SELECT
                        AVG(priority) as avg_priority,
                        MIN(priority) as min_priority,
                        MAX(priority) as max_priority
                    FROM goals
                    WHERE status IN ('pending', 'active')
                """)
                row = cursor.fetchone()
                stats['priority_stats'] = {
                    'average': row['avg_priority'],
                    'minimum': row['min_priority'],
                    'maximum': row['max_priority']
                }

                # Safety validator stats
                stats['safety_stats'] = self.safety_validator.get_validation_stats()

                return stats

        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}

    def cleanup_database(self) -> bool:
        """Perform database maintenance and cleanup."""
        with self._lock:
            try:
                with self._get_db_connection() as conn:
                    # Vacuum database
                    conn.execute("VACUUM")

                    # Update metadata
                    conn.execute("""
                        INSERT OR REPLACE INTO goal_metadata (key, value, updated_timestamp)
                        VALUES ('last_cleanup', ?, ?)
                    """, (datetime.now().isoformat(), datetime.now().isoformat()))

                    conn.commit()

                self.logger.info("Database cleanup completed")
                return True

            except Exception as e:
                self.logger.error(f"Error during database cleanup: {e}")
                return False
