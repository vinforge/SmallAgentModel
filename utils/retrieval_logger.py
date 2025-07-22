"""
Retrieval Logging System for SAM
Tracks queries, responses, and retrieval performance for analysis.

Sprint 2 Task 5: Retrieval Logging
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RetrievalLogEntry:
    """Represents a single retrieval log entry."""
    timestamp: str
    query: str
    response: str
    chunks_used: List[str]
    enrichment_scores: List[float]
    similarity_scores: List[float]
    processing_time: float
    mode: str
    user_feedback: Optional[str] = None

class RetrievalLogger:
    """
    Logs retrieval queries and responses for analysis and improvement.
    Stores data in SQLite database for easy querying and analysis.
    """
    
    def __init__(self, db_path: str = "logs/retrieval_logs.db"):
        """
        Initialize the retrieval logger.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"Retrieval logger initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize the database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS retrieval_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        chunks_used TEXT,  -- JSON array of chunk IDs
                        enrichment_scores TEXT,  -- JSON array of scores
                        similarity_scores TEXT,  -- JSON array of similarity scores
                        processing_time REAL,
                        mode TEXT,  -- 'semantic', 'direct', 'fallback'
                        user_feedback TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better query performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON retrieval_log(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mode ON retrieval_log(mode)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON retrieval_log(created_at)")
                
                logger.debug("Retrieval log database schema initialized")
                
        except Exception as e:
            logger.error(f"Error initializing retrieval log database: {e}")
            raise
    
    def log_retrieval(self, query: str, response: str, context_chunks: List[Dict[str, Any]] = None,
                     processing_time: float = 0.0, mode: str = "unknown") -> int:
        """
        Log a retrieval query and response.
        
        Args:
            query: User query
            response: System response
            context_chunks: List of chunks used for context
            processing_time: Time taken to process the query
            mode: Retrieval mode ('semantic', 'direct', 'fallback')
            
        Returns:
            Log entry ID
        """
        try:
            # Extract information from context chunks
            chunks_used = []
            enrichment_scores = []
            similarity_scores = []
            
            if context_chunks:
                for chunk in context_chunks:
                    chunks_used.append(chunk.get('chunk_id', 'unknown'))
                    enrichment_scores.append(chunk.get('metadata', {}).get('enrichment_score', 0.0))
                    similarity_scores.append(chunk.get('similarity_score', 0.0))
            
            # Create log entry
            entry = RetrievalLogEntry(
                timestamp=datetime.now().isoformat(),
                query=query,
                response=response,
                chunks_used=chunks_used,
                enrichment_scores=enrichment_scores,
                similarity_scores=similarity_scores,
                processing_time=processing_time,
                mode=mode
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO retrieval_log 
                    (timestamp, query, response, chunks_used, enrichment_scores, 
                     similarity_scores, processing_time, mode)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.timestamp,
                    entry.query,
                    entry.response,
                    json.dumps(entry.chunks_used),
                    json.dumps(entry.enrichment_scores),
                    json.dumps(entry.similarity_scores),
                    entry.processing_time,
                    entry.mode
                ))
                
                log_id = cursor.lastrowid
                logger.debug(f"Logged retrieval query with ID: {log_id}")
                return log_id
                
        except Exception as e:
            logger.error(f"Error logging retrieval: {e}")
            return -1
    
    def add_feedback(self, log_id: int, feedback: str):
        """
        Add user feedback to a logged retrieval.
        
        Args:
            log_id: ID of the log entry
            feedback: User feedback (e.g., "👍", "👎", or text)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE retrieval_log 
                    SET user_feedback = ? 
                    WHERE id = ?
                """, (feedback, log_id))
                
                logger.debug(f"Added feedback to log entry {log_id}: {feedback}")
                
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
    
    def get_recent_logs(self, limit: int = 10) -> List[RetrievalLogEntry]:
        """
        Get recent retrieval logs.
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of recent log entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, query, response, chunks_used, enrichment_scores,
                           similarity_scores, processing_time, mode, user_feedback
                    FROM retrieval_log 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                
                logs = []
                for row in cursor.fetchall():
                    entry = RetrievalLogEntry(
                        timestamp=row[0],
                        query=row[1],
                        response=row[2],
                        chunks_used=json.loads(row[3]) if row[3] else [],
                        enrichment_scores=json.loads(row[4]) if row[4] else [],
                        similarity_scores=json.loads(row[5]) if row[5] else [],
                        processing_time=row[6] or 0.0,
                        mode=row[7] or "unknown",
                        user_feedback=row[8]
                    )
                    logs.append(entry)
                
                return logs
                
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retrieval statistics.
        
        Returns:
            Dictionary with various statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total queries
                total_queries = conn.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0]
                
                # Queries by mode
                mode_stats = {}
                cursor = conn.execute("SELECT mode, COUNT(*) FROM retrieval_log GROUP BY mode")
                for mode, count in cursor.fetchall():
                    mode_stats[mode] = count
                
                # Average processing time
                avg_time = conn.execute("SELECT AVG(processing_time) FROM retrieval_log").fetchone()[0] or 0
                
                # Feedback stats
                feedback_stats = {}
                cursor = conn.execute("""
                    SELECT user_feedback, COUNT(*) 
                    FROM retrieval_log 
                    WHERE user_feedback IS NOT NULL 
                    GROUP BY user_feedback
                """)
                for feedback, count in cursor.fetchall():
                    feedback_stats[feedback] = count
                
                # Recent activity (last 24 hours)
                recent_count = conn.execute("""
                    SELECT COUNT(*) FROM retrieval_log 
                    WHERE created_at > datetime('now', '-1 day')
                """).fetchone()[0]
                
                return {
                    'total_queries': total_queries,
                    'mode_distribution': mode_stats,
                    'average_processing_time': avg_time,
                    'feedback_distribution': feedback_stats,
                    'recent_24h': recent_count
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def search_logs(self, query_pattern: str = None, mode: str = None, 
                   limit: int = 50) -> List[RetrievalLogEntry]:
        """
        Search retrieval logs.
        
        Args:
            query_pattern: Pattern to search in queries (SQL LIKE pattern)
            mode: Filter by retrieval mode
            limit: Maximum number of results
            
        Returns:
            List of matching log entries
        """
        try:
            conditions = []
            params = []
            
            if query_pattern:
                conditions.append("query LIKE ?")
                params.append(f"%{query_pattern}%")
            
            if mode:
                conditions.append("mode = ?")
                params.append(mode)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"""
                    SELECT timestamp, query, response, chunks_used, enrichment_scores,
                           similarity_scores, processing_time, mode, user_feedback
                    FROM retrieval_log 
                    WHERE {where_clause}
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, params)
                
                logs = []
                for row in cursor.fetchall():
                    entry = RetrievalLogEntry(
                        timestamp=row[0],
                        query=row[1],
                        response=row[2],
                        chunks_used=json.loads(row[3]) if row[3] else [],
                        enrichment_scores=json.loads(row[4]) if row[4] else [],
                        similarity_scores=json.loads(row[5]) if row[5] else [],
                        processing_time=row[6] or 0.0,
                        mode=row[7] or "unknown",
                        user_feedback=row[8]
                    )
                    logs.append(entry)
                
                return logs
                
        except Exception as e:
            logger.error(f"Error searching logs: {e}")
            return []

# Global logger instance
_retrieval_logger = None

def get_retrieval_logger() -> RetrievalLogger:
    """Get or create a global retrieval logger instance."""
    global _retrieval_logger
    
    if _retrieval_logger is None:
        _retrieval_logger = RetrievalLogger()
    
    return _retrieval_logger
