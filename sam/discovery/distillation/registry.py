"""
Principle Registry for Cognitive Distillation Engine
===================================================

Manages storage, retrieval, and lifecycle of discovered cognitive principles.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from .schema import distillation_schema

logger = logging.getLogger(__name__)

@dataclass
class CognitivePrinciple:
    """Represents a discovered cognitive principle."""
    principle_id: str
    principle_text: str
    source_strategy_id: str
    date_discovered: datetime
    is_active: bool = True
    confidence_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    domain_tags: List[str] = None
    validation_status: str = 'pending'
    created_by: str = 'distillation_engine'
    last_updated: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.domain_tags is None:
            self.domain_tags = []
        if self.last_updated is None:
            self.last_updated = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class PrincipleRegistry:
    """Manages cognitive principles storage and retrieval."""
    
    def __init__(self):
        """Initialize the principle registry."""
        self.schema = distillation_schema
        logger.info("Principle registry initialized")
    
    def store_principle(self, principle: CognitivePrinciple) -> bool:
        """Store a new cognitive principle."""
        try:
            with self.schema.get_connection() as conn:
                conn.execute("""
                    INSERT INTO cognitive_principles (
                        principle_id, principle_text, source_strategy_id,
                        date_discovered, is_active, confidence_score,
                        usage_count, success_rate, domain_tags,
                        validation_status, created_by, last_updated, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    principle.principle_id,
                    principle.principle_text,
                    principle.source_strategy_id,
                    principle.date_discovered.isoformat(),
                    principle.is_active,
                    principle.confidence_score,
                    principle.usage_count,
                    principle.success_rate,
                    json.dumps(principle.domain_tags),
                    principle.validation_status,
                    principle.created_by,
                    principle.last_updated.isoformat(),
                    json.dumps(principle.metadata)
                ))
                
                logger.info(f"Stored principle: {principle.principle_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store principle: {e}")
            return False
    
    def get_principle(self, principle_id: str) -> Optional[CognitivePrinciple]:
        """Retrieve a specific principle by ID."""
        try:
            with self.schema.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM cognitive_principles 
                    WHERE principle_id = ?
                """, (principle_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_principle(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get principle {principle_id}: {e}")
            return None
    
    def get_active_principles(self, limit: Optional[int] = None) -> List[CognitivePrinciple]:
        """Get all active principles."""
        try:
            with self.schema.get_connection() as conn:
                query = """
                    SELECT * FROM cognitive_principles 
                    WHERE is_active = TRUE 
                    ORDER BY confidence_score DESC, usage_count DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query)
                return [self._row_to_principle(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get active principles: {e}")
            return []
    
    def search_principles_by_domain(self, domain: str) -> List[CognitivePrinciple]:
        """Search principles by domain tag."""
        try:
            with self.schema.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM cognitive_principles 
                    WHERE is_active = TRUE 
                    AND domain_tags LIKE ?
                    ORDER BY confidence_score DESC
                """, (f'%"{domain}"%',))
                
                return [self._row_to_principle(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to search principles by domain {domain}: {e}")
            return []
    
    def update_principle_performance(self, principle_id: str, 
                                   outcome: str, confidence_delta: float = 0.0) -> bool:
        """Update principle performance metrics."""
        try:
            with self.schema.get_connection() as conn:
                # Update usage count and success rate
                cursor = conn.execute("""
                    SELECT usage_count, success_rate FROM cognitive_principles
                    WHERE principle_id = ?
                """, (principle_id,))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Principle {principle_id} not found for performance update")
                    return False
                
                current_usage = row[0]
                current_success_rate = row[1]
                
                # Calculate new metrics
                new_usage = current_usage + 1
                success_weight = 1.0 if outcome == 'success' else 0.0
                new_success_rate = ((current_success_rate * current_usage) + success_weight) / new_usage
                
                # Update confidence score
                new_confidence = min(1.0, max(0.0, current_success_rate + confidence_delta))
                
                conn.execute("""
                    UPDATE cognitive_principles 
                    SET usage_count = ?, success_rate = ?, confidence_score = ?,
                        last_updated = ?
                    WHERE principle_id = ?
                """, (new_usage, new_success_rate, new_confidence, 
                      datetime.now().isoformat(), principle_id))
                
                logger.info(f"Updated performance for principle {principle_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update principle performance: {e}")
            return False
    
    def deactivate_principle(self, principle_id: str, reason: str = "") -> bool:
        """Deactivate a principle."""
        try:
            with self.schema.get_connection() as conn:
                conn.execute("""
                    UPDATE cognitive_principles 
                    SET is_active = FALSE, last_updated = ?,
                        metadata = json_set(COALESCE(metadata, '{}'), '$.deactivation_reason', ?)
                    WHERE principle_id = ?
                """, (datetime.now().isoformat(), reason, principle_id))
                
                logger.info(f"Deactivated principle {principle_id}: {reason}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to deactivate principle: {e}")
            return False
    
    def get_principle_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        try:
            with self.schema.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_principles,
                        COUNT(CASE WHEN is_active THEN 1 END) as active_principles,
                        AVG(confidence_score) as avg_confidence,
                        AVG(success_rate) as avg_success_rate,
                        SUM(usage_count) as total_usage
                    FROM cognitive_principles
                """)
                
                row = cursor.fetchone()
                return {
                    'total_principles': row[0] or 0,
                    'active_principles': row[1] or 0,
                    'avg_confidence': round(row[2] or 0.0, 3),
                    'avg_success_rate': round(row[3] or 0.0, 3),
                    'total_usage': row[4] or 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get principle stats: {e}")
            return {}
    
    def _row_to_principle(self, row) -> CognitivePrinciple:
        """Convert database row to CognitivePrinciple object."""
        return CognitivePrinciple(
            principle_id=row['principle_id'],
            principle_text=row['principle_text'],
            source_strategy_id=row['source_strategy_id'],
            date_discovered=datetime.fromisoformat(row['date_discovered']),
            is_active=bool(row['is_active']),
            confidence_score=row['confidence_score'],
            usage_count=row['usage_count'],
            success_rate=row['success_rate'],
            domain_tags=json.loads(row['domain_tags']) if row['domain_tags'] else [],
            validation_status=row['validation_status'],
            created_by=row['created_by'],
            last_updated=datetime.fromisoformat(row['last_updated']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    def create_principle(self, principle_text: str, source_strategy_id: str,
                        domain_tags: List[str] = None, confidence_score: float = 0.5,
                        metadata: Dict[str, Any] = None) -> CognitivePrinciple:
        """Create a new cognitive principle."""
        principle = CognitivePrinciple(
            principle_id=str(uuid.uuid4()),
            principle_text=principle_text,
            source_strategy_id=source_strategy_id,
            date_discovered=datetime.now(),
            domain_tags=domain_tags or [],
            confidence_score=confidence_score,
            metadata=metadata or {}
        )
        
        if self.store_principle(principle):
            logger.info(f"Created new principle: {principle.principle_id}")
            return principle
        else:
            raise Exception("Failed to store new principle")

# Global registry instance
principle_registry = PrincipleRegistry()
