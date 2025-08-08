"""
Database Schema for Cognitive Distillation Engine
================================================

Defines the database schema for storing discovered cognitive principles
and related metadata.

Author: SAM Development Team
Version: 1.0.0
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DistillationSchema:
    """Manages database schema for cognitive distillation."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize schema manager."""
        if db_path is None:
            # Use SAM's existing episodic memory database
            db_path = "memory/episodic_store.db"
        
        self.db_path = Path(db_path)
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Ensure database and tables exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                self.create_tables(conn)
                logger.info("Distillation database schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize distillation schema: {e}")
            raise
    
    def create_tables(self, conn: sqlite3.Connection):
        """Create all required tables for cognitive distillation."""
        
        # Enhanced cognitive principles table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_principles (
                principle_id TEXT PRIMARY KEY,
                principle_text TEXT NOT NULL,
                source_strategy_id TEXT,
                date_discovered TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                confidence_score REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                domain_tags TEXT,  -- JSON array of applicable domains
                validation_status TEXT DEFAULT 'pending',  -- pending, validated, rejected
                created_by TEXT DEFAULT 'distillation_engine',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT  -- JSON for additional metadata
            )
        """)
        
        # Principle performance tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS principle_performance (
                performance_id TEXT PRIMARY KEY,
                principle_id TEXT NOT NULL,
                query_id TEXT,
                application_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                outcome TEXT,  -- success, failure, neutral
                confidence_before REAL,
                confidence_after REAL,
                user_feedback TEXT,
                response_quality_score REAL,
                FOREIGN KEY (principle_id) REFERENCES cognitive_principles (principle_id)
            )
        """)
        
        # Successful interaction data cache
        conn.execute("""
            CREATE TABLE IF NOT EXISTS successful_interactions (
                interaction_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                context_provided TEXT,
                response_text TEXT NOT NULL,
                user_feedback TEXT,
                success_metrics TEXT,  -- JSON with various success metrics
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_system TEXT DEFAULT 'sam_core',
                metadata TEXT  -- JSON for additional data
            )
        """)
        
        # Distillation runs tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS distillation_runs (
                run_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                start_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_timestamp TIMESTAMP,
                status TEXT DEFAULT 'running',  -- running, completed, failed
                interactions_analyzed INTEGER DEFAULT 0,
                principles_discovered INTEGER DEFAULT 0,
                error_message TEXT,
                configuration TEXT,  -- JSON with run configuration
                results TEXT  -- JSON with detailed results
            )
        """)
        
        # Create indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_principles_active ON cognitive_principles (is_active)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_principles_domain ON cognitive_principles (domain_tags)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_principles_confidence ON cognitive_principles (confidence_score)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_principle ON principle_performance (principle_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_strategy ON successful_interactions (strategy_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_strategy ON distillation_runs (strategy_id)")
        
        conn.commit()
        logger.info("Cognitive distillation tables created successfully")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    def validate_schema(self) -> bool:
        """Validate that all required tables exist."""
        required_tables = [
            'cognitive_principles',
            'principle_performance', 
            'successful_interactions',
            'distillation_runs'
        ]
        
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ({})
                """.format(','.join('?' * len(required_tables))), required_tables)
                
                existing_tables = {row[0] for row in cursor.fetchall()}
                missing_tables = set(required_tables) - existing_tables
                
                if missing_tables:
                    logger.error(f"Missing tables: {missing_tables}")
                    return False
                
                logger.info("Schema validation passed")
                return True
                
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

# Global schema instance
distillation_schema = DistillationSchema()
