"""
Interaction Collector for Cognitive Distillation Engine
======================================================

Collects successful interaction data from various SAM systems for analysis.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .schema import distillation_schema

logger = logging.getLogger(__name__)

@dataclass
class SuccessfulInteraction:
    """Represents a successful interaction for analysis."""
    interaction_id: str
    strategy_id: str
    query_text: str
    context_provided: str
    response_text: str
    user_feedback: Optional[str] = None
    success_metrics: Dict[str, Any] = None
    timestamp: datetime = None
    source_system: str = 'sam_core'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.success_metrics is None:
            self.success_metrics = {}
        if self.metadata is None:
            self.metadata = {}

class InteractionCollector:
    """Collects successful interactions for principle discovery."""
    
    def __init__(self):
        """Initialize the interaction collector."""
        self.schema = distillation_schema
        logger.info("Interaction collector initialized")
    
    def collect_successful_interactions(self, strategy_id: str, 
                                      limit: int = 20) -> List[SuccessfulInteraction]:
        """
        Collect successful interactions for a given strategy.
        
        Args:
            strategy_id: The strategy to collect interactions for
            limit: Maximum number of interactions to collect
            
        Returns:
            List of successful interactions
        """
        try:
            interactions = []
            
            # Collect from multiple sources
            interactions.extend(self._collect_from_episodic_memory(strategy_id, limit // 2))
            interactions.extend(self._collect_from_ab_testing_logs(strategy_id, limit // 2))
            interactions.extend(self._collect_from_conversation_threads(strategy_id, limit // 4))
            
            # Sort by success metrics and timestamp
            interactions.sort(key=lambda x: (
                x.success_metrics.get('quality_score', 0.0),
                x.timestamp
            ), reverse=True)
            
            # Return top interactions up to limit
            result = interactions[:limit]
            logger.info(f"Collected {len(result)} successful interactions for strategy {strategy_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to collect interactions for strategy {strategy_id}: {e}")
            return []
    
    def _collect_from_episodic_memory(self, strategy_id: str, limit: int) -> List[SuccessfulInteraction]:
        """Collect interactions from episodic memory store."""
        interactions = []

        try:
            # Try multiple possible episodic memory locations
            possible_paths = [
                Path("memory/episodic_store.db"),
                Path("data/episodic_memory.db"),
                Path("memory_store/episodic.db"),
                Path("memory/memory_manager.db")
            ]

            episodic_db_path = None
            for path in possible_paths:
                if path.exists():
                    episodic_db_path = path
                    break

            if not episodic_db_path:
                logger.warning("Episodic memory database not found in any expected location")
                return interactions

            with sqlite3.connect(episodic_db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Try multiple table schemas that might exist in SAM
                table_queries = [
                    # Standard interactions table
                    """
                    SELECT * FROM interactions
                    WHERE strategy_id = ? AND success_score > 0.7
                    ORDER BY success_score DESC, timestamp DESC LIMIT ?
                    """,
                    # Memory entries with quality scores
                    """
                    SELECT * FROM memory_entries
                    WHERE metadata LIKE '%strategy_id%' AND metadata LIKE ?
                    AND quality_score > 0.7
                    ORDER BY quality_score DESC, created_at DESC LIMIT ?
                    """,
                    # Conversation logs
                    """
                    SELECT * FROM conversation_logs
                    WHERE strategy_used = ? AND rating > 3
                    ORDER BY rating DESC, timestamp DESC LIMIT ?
                    """,
                    # Generic successful responses
                    """
                    SELECT * FROM responses
                    WHERE metadata LIKE ? AND success_indicator > 0.7
                    ORDER BY success_indicator DESC, created_at DESC LIMIT ?
                    """
                ]

                for query in table_queries:
                    try:
                        if 'metadata LIKE' in query:
                            cursor = conn.execute(query, (f'%{strategy_id}%', limit))
                        else:
                            cursor = conn.execute(query, (strategy_id, limit))

                        for row in cursor.fetchall():
                            interaction = self._parse_episodic_row(row, strategy_id)
                            if interaction:
                                interactions.append(interaction)

                        if interactions:
                            logger.info(f"Found {len(interactions)} interactions using episodic memory")
                            break

                    except sqlite3.OperationalError:
                        continue  # Try next query if table doesn't exist

        except Exception as e:
            logger.warning(f"Could not collect from episodic memory: {e}")

        return interactions[:limit]

    def _parse_episodic_row(self, row, strategy_id: str) -> Optional[SuccessfulInteraction]:
        """Parse a row from episodic memory into SuccessfulInteraction."""
        try:
            # Handle different possible column names
            interaction_id = (row.get('interaction_id') or
                            row.get('id') or
                            row.get('memory_id') or
                            f"episodic_{hash(str(row))}")

            query_text = (row.get('query_text') or
                         row.get('query') or
                         row.get('user_input') or
                         row.get('question') or '')

            response_text = (row.get('response_text') or
                           row.get('response') or
                           row.get('answer') or
                           row.get('content') or '')

            context_provided = (row.get('context_provided') or
                              row.get('context') or
                              row.get('retrieved_context') or '')

            # Extract success metrics
            success_score = (row.get('success_score') or
                           row.get('quality_score') or
                           row.get('rating') or
                           row.get('success_indicator') or 0.0)

            # Parse timestamp
            timestamp_str = (row.get('timestamp') or
                           row.get('created_at') or
                           row.get('date') or
                           datetime.now().isoformat())

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()

            # Only include if we have meaningful content
            if len(query_text) > 10 and len(response_text) > 20:
                return SuccessfulInteraction(
                    interaction_id=str(interaction_id),
                    strategy_id=strategy_id,
                    query_text=query_text,
                    context_provided=context_provided,
                    response_text=response_text,
                    user_feedback=row.get('user_feedback'),
                    success_metrics={
                        'quality_score': float(success_score),
                        'source': 'episodic_memory'
                    },
                    timestamp=timestamp,
                    source_system='episodic_memory',
                    metadata=self._extract_metadata_from_row(row)
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to parse episodic row: {e}")
            return None

    def _extract_metadata_from_row(self, row) -> Dict[str, Any]:
        """Extract metadata from database row."""
        metadata = {}

        # Try to parse JSON metadata if it exists
        for field in ['metadata', 'extra_data', 'details']:
            if hasattr(row, field) and row[field]:
                try:
                    metadata.update(json.loads(row[field]))
                except:
                    metadata[field] = str(row[field])

        # Add other relevant fields
        for field in ['user_id', 'session_id', 'model_used', 'tokens_used']:
            if hasattr(row, field) and row[field]:
                metadata[field] = row[field]

        return metadata
    
    def _collect_from_ab_testing_logs(self, strategy_id: str, limit: int) -> List[SuccessfulInteraction]:
        """Collect interactions from A/B testing logs."""
        interactions = []
        
        try:
            # Check for A/B testing results
            ab_test_dir = Path("ab_tests")
            if not ab_test_dir.exists():
                logger.warning("A/B testing directory not found")
                return interactions
            
            # Look for relevant test results
            for result_file in ab_test_dir.glob("*_results.jsonl"):
                try:
                    with open(result_file, 'r') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            
                            # Check if this result is for our strategy
                            if data.get('strategy_id') == strategy_id and data.get('outcome') == 'success':
                                interaction = SuccessfulInteraction(
                                    interaction_id=data.get('test_id', f"ab_{result_file.stem}"),
                                    strategy_id=strategy_id,
                                    query_text=data.get('query', ''),
                                    context_provided=data.get('context', ''),
                                    response_text=data.get('response', ''),
                                    success_metrics={
                                        'quality_score': data.get('quality_score', 0.8),
                                        'ab_test_score': data.get('score', 0.0),
                                        'source': 'ab_testing'
                                    },
                                    timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                                    source_system='ab_testing',
                                    metadata=data.get('metadata', {})
                                )
                                interactions.append(interaction)
                                
                                if len(interactions) >= limit:
                                    break
                                    
                except Exception as e:
                    logger.warning(f"Could not parse A/B test file {result_file}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Could not collect from A/B testing logs: {e}")
        
        return interactions[:limit]

    def _collect_from_conversation_threads(self, strategy_id: str, limit: int) -> List[SuccessfulInteraction]:
        """Collect interactions from conversation threads."""
        interactions = []

        try:
            # Check for conversation threads
            threads_dir = Path("conversation_threads")
            if not threads_dir.exists():
                logger.warning("Conversation threads directory not found")
                return interactions

            # Look for relevant conversation threads
            for thread_file in threads_dir.glob("thread_*.json"):
                try:
                    with open(thread_file, 'r') as f:
                        thread_data = json.load(f)

                    # Extract successful interactions from thread
                    messages = thread_data.get('messages', [])
                    for i, message in enumerate(messages):
                        if (message.get('role') == 'assistant' and
                            message.get('metadata', {}).get('strategy_id') == strategy_id and
                            message.get('metadata', {}).get('quality_score', 0) > 0.7):

                            # Find the corresponding user query
                            user_query = ""
                            if i > 0 and messages[i-1].get('role') == 'user':
                                user_query = messages[i-1].get('content', '')

                            interaction = SuccessfulInteraction(
                                interaction_id=f"thread_{thread_file.stem}_{i}",
                                strategy_id=strategy_id,
                                query_text=user_query,
                                context_provided=message.get('metadata', {}).get('context', ''),
                                response_text=message.get('content', ''),
                                success_metrics={
                                    'quality_score': message.get('metadata', {}).get('quality_score', 0.8),
                                    'source': 'conversation_threads'
                                },
                                timestamp=datetime.fromisoformat(
                                    message.get('timestamp', datetime.now().isoformat())
                                ),
                                source_system='conversation_threads',
                                metadata=message.get('metadata', {})
                            )
                            interactions.append(interaction)

                            if len(interactions) >= limit:
                                break

                except Exception as e:
                    logger.warning(f"Could not parse conversation thread {thread_file}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Could not collect from conversation threads: {e}")

        return interactions[:limit]

    def store_interaction(self, interaction: SuccessfulInteraction) -> bool:
        """Store a successful interaction for future analysis."""
        try:
            with self.schema.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO successful_interactions (
                        interaction_id, strategy_id, query_text, context_provided,
                        response_text, user_feedback, success_metrics, timestamp,
                        source_system, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction.interaction_id,
                    interaction.strategy_id,
                    interaction.query_text,
                    interaction.context_provided,
                    interaction.response_text,
                    interaction.user_feedback,
                    json.dumps(interaction.success_metrics),
                    interaction.timestamp.isoformat(),
                    interaction.source_system,
                    json.dumps(interaction.metadata)
                ))

                logger.info(f"Stored interaction: {interaction.interaction_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            return False

    def get_interaction_summary(self, strategy_id: str) -> Dict[str, Any]:
        """Get summary statistics for interactions of a strategy."""
        try:
            with self.schema.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT
                        COUNT(*) as total_interactions,
                        AVG(json_extract(success_metrics, '$.quality_score')) as avg_quality,
                        COUNT(DISTINCT source_system) as source_systems,
                        MIN(timestamp) as earliest_interaction,
                        MAX(timestamp) as latest_interaction
                    FROM successful_interactions
                    WHERE strategy_id = ?
                """, (strategy_id,))

                row = cursor.fetchone()
                return {
                    'strategy_id': strategy_id,
                    'total_interactions': row[0] or 0,
                    'avg_quality_score': round(row[1] or 0.0, 3),
                    'source_systems': row[2] or 0,
                    'earliest_interaction': row[3],
                    'latest_interaction': row[4]
                }

        except Exception as e:
            logger.error(f"Failed to get interaction summary: {e}")
            return {'strategy_id': strategy_id, 'error': str(e)}

# Global collector instance
interaction_collector = InteractionCollector()
