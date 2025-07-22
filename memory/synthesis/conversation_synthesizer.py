"""
Conversation Memory Synthesizer - mem0-inspired Background Processing
===================================================================

This module implements automated conversation clustering and insight synthesis
during SAM's "sleep" cycle, inspired by mem0's approach to memory consolidation.

The MemorySynthesizer automatically:
1. Fetches vector embeddings for all archived conversation threads
2. Uses clustering algorithms to group conversations into thematic clusters
3. Generates high-level insights for each cluster using LLM synthesis
4. Feeds structured insights to the Self-Improvement Engine

Part of SAM's mem0-inspired Memory Augmentation (Task 33, Phase 2)
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json

from .clustering_service import ClusteringService, ConceptCluster
from .synthesis_engine import SynthesisEngine, SynthesizedInsight
from ..memory_vectorstore import get_memory_store, MemoryChunk, MemoryType

logger = logging.getLogger(__name__)

@dataclass
class ConversationCluster:
    """Represents a cluster of related conversation threads."""
    cluster_id: str
    thread_ids: List[str]
    thread_titles: List[str]
    centroid_embedding: np.ndarray
    coherence_score: float
    dominant_topics: List[str]
    conversation_count: int
    date_range: Tuple[str, str]  # (earliest, latest)
    user_patterns: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ConversationInsight:
    """Structured insight object for Self-Improvement Engine integration."""
    insight_id: str
    synthesized_topic: str
    implication: str
    supporting_conversations: List[str]
    confidence_score: float
    insight_type: str  # 'user_pattern', 'knowledge_gap', 'recurring_theme', etc.
    actionable_recommendations: List[str]
    timestamp: str
    metadata: Dict[str, Any]

class MemorySynthesizer:
    """
    Automated conversation clustering and insight synthesis for SAM's sleep cycle.
    
    This component runs as a background process during idle periods, analyzing
    conversation patterns and generating insights about user behavior, knowledge
    gaps, and recurring themes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Memory Synthesizer."""
        self.config = config or self._get_default_config()
        
        # Initialize synthesis components
        self.clustering_service = ClusteringService(
            eps=self.config['clustering_eps'],
            min_samples=self.config['clustering_min_samples'],
            min_cluster_size=self.config['min_cluster_size'],
            max_clusters=self.config['max_clusters']
        )
        
        self.synthesis_engine = SynthesisEngine()
        
        # Storage
        self.conversation_threads = []
        self.conversation_clusters = []
        self.generated_insights = []
        
        # State tracking
        self.last_synthesis_time = None
        self.synthesis_history = []
        
        self.logger = logging.getLogger(f"{__name__}.MemorySynthesizer")
        self.logger.info("MemorySynthesizer initialized for conversation analysis")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for conversation synthesis."""
        return {
            'clustering_eps': 0.3,
            'clustering_min_samples': 2,
            'min_cluster_size': 3,
            'max_clusters': 20,
            'min_conversations_for_synthesis': 5,
            'synthesis_interval_hours': 24,
            'conversation_lookback_days': 30,
            'insight_confidence_threshold': 0.7,
            'max_insights_per_cluster': 2
        }
    
    async def run_conversation_synthesis(self) -> Dict[str, Any]:
        """
        Main entry point for conversation synthesis during sleep cycle.
        
        Returns:
            Synthesis results with insights for Self-Improvement Engine
        """
        try:
            self.logger.info("Starting conversation synthesis cycle")
            
            # Step 1: Load conversation threads
            conversation_threads = await self._load_conversation_threads()
            
            if len(conversation_threads) < self.config['min_conversations_for_synthesis']:
                self.logger.info(f"Insufficient conversations ({len(conversation_threads)}) for synthesis")
                return {'status': 'skipped', 'reason': 'insufficient_data'}
            
            # Step 2: Generate embeddings for conversation threads
            thread_embeddings = await self._generate_conversation_embeddings(conversation_threads)
            
            # Step 3: Cluster conversations by theme
            conversation_clusters = await self._cluster_conversations(conversation_threads, thread_embeddings)
            
            # Step 4: Generate insights for each cluster
            insights = await self._generate_cluster_insights(conversation_clusters)
            
            # Step 5: Store insights for Self-Improvement Engine
            stored_insights = await self._store_insights_for_self_improvement(insights)
            
            # Step 6: Update synthesis history
            synthesis_result = {
                'synthesis_id': f"conv_synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'conversations_analyzed': len(conversation_threads),
                'clusters_found': len(conversation_clusters),
                'insights_generated': len(insights),
                'insights_stored': len(stored_insights),
                'status': 'completed'
            }
            
            self.synthesis_history.append(synthesis_result)
            self.last_synthesis_time = datetime.now()
            
            self.logger.info(f"Conversation synthesis completed: {len(insights)} insights generated")
            return synthesis_result
            
        except Exception as e:
            self.logger.error(f"Conversation synthesis failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _load_conversation_threads(self) -> List[Dict[str, Any]]:
        """Load conversation threads from the conversation intelligence system."""
        try:
            from sam.conversation.contextual_relevance import ContextualRelevanceEngine
            
            # Initialize conversation engine
            conv_engine = ContextualRelevanceEngine()
            
            # Get archived threads within lookback period
            cutoff_date = datetime.now() - timedelta(days=self.config['conversation_lookback_days'])
            threads = conv_engine.get_archived_threads()
            
            # Filter by date and ensure we have embeddings
            recent_threads = []
            for thread in threads:
                thread_date = datetime.fromisoformat(thread.created_at)
                if thread_date >= cutoff_date and len(thread.messages) >= 2:
                    recent_threads.append({
                        'thread_id': thread.thread_id,
                        'title': thread.title,
                        'messages': thread.messages,
                        'topic_keywords': thread.topic_keywords,
                        'created_at': thread.created_at,
                        'message_count': thread.message_count
                    })
            
            self.logger.info(f"Loaded {len(recent_threads)} conversation threads for analysis")
            return recent_threads
            
        except Exception as e:
            self.logger.error(f"Failed to load conversation threads: {e}")
            return []
    
    async def _generate_conversation_embeddings(self, threads: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for conversation threads."""
        try:
            # Get memory store for embedding generation
            memory_store = get_memory_store()
            
            embeddings = []
            for thread in threads:
                # Combine thread title and key messages for embedding
                thread_text = f"{thread['title']}\n"
                
                # Add key messages (first, last, and middle if available)
                messages = thread['messages']
                if messages:
                    thread_text += messages[0].get('content', '')
                    if len(messages) > 2:
                        mid_idx = len(messages) // 2
                        thread_text += f"\n{messages[mid_idx].get('content', '')}"
                    if len(messages) > 1:
                        thread_text += f"\n{messages[-1].get('content', '')}"
                
                # Generate embedding
                embedding = memory_store._generate_embedding(thread_text[:1000])  # Limit length
                embeddings.append(embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            self.logger.error(f"Failed to generate conversation embeddings: {e}")
            return np.array([])
    
    async def _cluster_conversations(self, threads: List[Dict[str, Any]], 
                                   embeddings: np.ndarray) -> List[ConversationCluster]:
        """Cluster conversations using DBSCAN-based approach."""
        try:
            if len(embeddings) == 0:
                return []
            
            # Use existing clustering service with conversation-specific parameters
            from sklearn.cluster import DBSCAN
            
            clustering = DBSCAN(
                eps=self.config['clustering_eps'],
                min_samples=self.config['clustering_min_samples']
            )
            
            cluster_labels = clustering.fit_predict(embeddings)
            
            # Group conversations by cluster
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label != -1:  # Ignore noise points
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(idx)
            
            # Create ConversationCluster objects
            conversation_clusters = []
            for cluster_id, thread_indices in clusters.items():
                if len(thread_indices) >= self.config['min_cluster_size']:
                    cluster_threads = [threads[i] for i in thread_indices]
                    cluster_embeddings = embeddings[thread_indices]
                    
                    # Calculate cluster properties
                    centroid = np.mean(cluster_embeddings, axis=0)
                    coherence = self._calculate_cluster_coherence(cluster_embeddings, centroid)
                    
                    # Extract dominant topics
                    all_keywords = []
                    for thread in cluster_threads:
                        all_keywords.extend(thread.get('topic_keywords', []))
                    
                    # Get most common keywords
                    from collections import Counter
                    keyword_counts = Counter(all_keywords)
                    dominant_topics = [kw for kw, count in keyword_counts.most_common(5)]
                    
                    # Date range
                    dates = [thread['created_at'] for thread in cluster_threads]
                    date_range = (min(dates), max(dates))
                    
                    cluster = ConversationCluster(
                        cluster_id=f"conv_cluster_{cluster_id}",
                        thread_ids=[t['thread_id'] for t in cluster_threads],
                        thread_titles=[t['title'] for t in cluster_threads],
                        centroid_embedding=centroid,
                        coherence_score=coherence,
                        dominant_topics=dominant_topics,
                        conversation_count=len(cluster_threads),
                        date_range=date_range,
                        user_patterns={},  # Will be filled by analysis
                        metadata={'cluster_label': cluster_id}
                    )
                    
                    conversation_clusters.append(cluster)
            
            self.logger.info(f"Created {len(conversation_clusters)} conversation clusters")
            return conversation_clusters
            
        except Exception as e:
            self.logger.error(f"Conversation clustering failed: {e}")
            return []

    def _calculate_cluster_coherence(self, embeddings: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate coherence score for a conversation cluster."""
        try:
            # Calculate cosine similarities to centroid
            similarities = []
            for embedding in embeddings:
                # Cosine similarity
                dot_product = np.dot(embedding, centroid)
                norm_product = np.linalg.norm(embedding) * np.linalg.norm(centroid)
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    similarities.append(similarity)

            return np.mean(similarities) if similarities else 0.0

        except Exception as e:
            self.logger.warning(f"Coherence calculation failed: {e}")
            return 0.0

    async def _generate_cluster_insights(self, clusters: List[ConversationCluster]) -> List[ConversationInsight]:
        """Generate insights for conversation clusters using LLM synthesis."""
        insights = []

        for cluster in clusters:
            try:
                # Generate insight for this cluster
                cluster_insights = await self._synthesize_cluster_insight(cluster)
                insights.extend(cluster_insights)

            except Exception as e:
                self.logger.error(f"Failed to generate insight for cluster {cluster.cluster_id}: {e}")

        return insights

    async def _synthesize_cluster_insight(self, cluster: ConversationCluster) -> List[ConversationInsight]:
        """Synthesize insights for a single conversation cluster."""
        try:
            # Create synthesis prompt for conversation cluster
            prompt = self._create_conversation_synthesis_prompt(cluster)

            # Use existing insight generator
            from .insight_generator import InsightGenerator
            insight_gen = InsightGenerator(temperature=0.7)

            # Generate insight text
            insight_text = insight_gen._call_llm_for_synthesis(prompt)

            if not insight_text or len(insight_text.strip()) < 20:
                self.logger.warning(f"Generated insight too short for cluster {cluster.cluster_id}")
                return []

            # Parse insight into structured format
            parsed_insights = self._parse_conversation_insight(insight_text, cluster)

            return parsed_insights

        except Exception as e:
            self.logger.error(f"Cluster insight synthesis failed: {e}")
            return []

    def _create_conversation_synthesis_prompt(self, cluster: ConversationCluster) -> str:
        """Create synthesis prompt for conversation cluster analysis."""

        prompt_parts = []

        # System context
        prompt_parts.append("""🧠 **SAM CONVERSATION SYNTHESIS MODE** 🧠

You are SAM analyzing conversation patterns during cognitive consolidation. Your task is to identify meaningful insights about user behavior, knowledge gaps, and recurring themes from clustered conversations.

**SYNTHESIS OBJECTIVE:**
Generate actionable insights about user patterns and system improvement opportunities from the following related conversations.

**INSIGHT REQUIREMENTS:**
• Identify USER BEHAVIOR PATTERNS and preferences
• Detect KNOWLEDGE GAPS where users repeatedly seek information
• Find RECURRING THEMES that indicate user interests or pain points
• Suggest ACTIONABLE IMPROVEMENTS for SAM's responses
• Focus on insights that can improve future interactions""")

        # Cluster information
        prompt_parts.append(f"""
**CONVERSATION CLUSTER ANALYSIS:**
• Cluster ID: {cluster.cluster_id}
• Conversations: {cluster.conversation_count}
• Coherence Score: {cluster.coherence_score:.2f}/1.0
• Dominant Topics: {', '.join(cluster.dominant_topics[:5])}
• Date Range: {cluster.date_range[0]} to {cluster.date_range[1]}""")

        # Conversation summaries
        prompt_parts.append("\n**CONVERSATION SUMMARIES:**")
        for i, (thread_id, title) in enumerate(zip(cluster.thread_ids[:5], cluster.thread_titles[:5]), 1):
            prompt_parts.append(f"""
**Conversation {i}** (ID: {thread_id})
Title: {title}""")

        # Synthesis instruction
        prompt_parts.append(f"""
**SYNTHESIS TASK:**
Based on the {cluster.conversation_count} related conversations above, generate insights in this format:

SYNTHESIZED_TOPIC: [Brief topic description]
IMPLICATION: [What this pattern means for user experience]
INSIGHT_TYPE: [user_pattern|knowledge_gap|recurring_theme|system_improvement]
RECOMMENDATIONS: [Specific actionable improvements]

**CONVERSATION INSIGHT:**""")

        return "\n".join(prompt_parts)

    def _parse_conversation_insight(self, insight_text: str, cluster: ConversationCluster) -> List[ConversationInsight]:
        """Parse LLM-generated insight into structured ConversationInsight objects."""
        try:
            insights = []

            # Simple parsing - look for key sections
            lines = insight_text.strip().split('\n')

            current_insight = {
                'synthesized_topic': '',
                'implication': '',
                'insight_type': 'recurring_theme',
                'recommendations': []
            }

            for line in lines:
                line = line.strip()
                if line.startswith('SYNTHESIZED_TOPIC:'):
                    current_insight['synthesized_topic'] = line.replace('SYNTHESIZED_TOPIC:', '').strip()
                elif line.startswith('IMPLICATION:'):
                    current_insight['implication'] = line.replace('IMPLICATION:', '').strip()
                elif line.startswith('INSIGHT_TYPE:'):
                    insight_type = line.replace('INSIGHT_TYPE:', '').strip().lower()
                    if insight_type in ['user_pattern', 'knowledge_gap', 'recurring_theme', 'system_improvement']:
                        current_insight['insight_type'] = insight_type
                elif line.startswith('RECOMMENDATIONS:'):
                    rec_text = line.replace('RECOMMENDATIONS:', '').strip()
                    current_insight['recommendations'] = [rec_text] if rec_text else []
                elif line.startswith('- ') and current_insight['recommendations']:
                    current_insight['recommendations'].append(line[2:].strip())

            # Create ConversationInsight object
            if current_insight['synthesized_topic'] and current_insight['implication']:
                insight = ConversationInsight(
                    insight_id=f"conv_insight_{cluster.cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    synthesized_topic=current_insight['synthesized_topic'],
                    implication=current_insight['implication'],
                    supporting_conversations=cluster.thread_ids,
                    confidence_score=cluster.coherence_score,
                    insight_type=current_insight['insight_type'],
                    actionable_recommendations=current_insight['recommendations'],
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        'cluster_id': cluster.cluster_id,
                        'conversation_count': cluster.conversation_count,
                        'dominant_topics': cluster.dominant_topics,
                        'date_range': cluster.date_range
                    }
                )
                insights.append(insight)

            return insights

        except Exception as e:
            self.logger.error(f"Insight parsing failed: {e}")
            return []

    async def _store_insights_for_self_improvement(self, insights: List[ConversationInsight]) -> List[str]:
        """Store insights in format compatible with Self-Improvement Engine."""
        try:
            stored_insight_ids = []

            # Store insights in memory system for future retrieval
            memory_store = get_memory_store()

            for insight in insights:
                # Create structured content for storage
                insight_content = f"""CONVERSATION INSIGHT: {insight.synthesized_topic}

IMPLICATION: {insight.implication}

TYPE: {insight.insight_type}

SUPPORTING CONVERSATIONS: {len(insight.supporting_conversations)} conversations
- {', '.join(insight.supporting_conversations[:3])}

RECOMMENDATIONS:
{chr(10).join(f'• {rec}' for rec in insight.actionable_recommendations)}

CONFIDENCE: {insight.confidence_score:.2f}
GENERATED: {insight.timestamp}"""

                # Store in memory system
                chunk_id = memory_store.add_memory(
                    content=insight_content,
                    memory_type=MemoryType.INSIGHT,
                    source='conversation_synthesizer',
                    tags=['conversation_insight', insight.insight_type, 'self_improvement'],
                    importance_score=insight.confidence_score,
                    metadata={
                        'insight_id': insight.insight_id,
                        'insight_type': insight.insight_type,
                        'conversation_count': len(insight.supporting_conversations),
                        'synthesis_timestamp': insight.timestamp
                    }
                )

                stored_insight_ids.append(chunk_id)

            self.logger.info(f"Stored {len(stored_insight_ids)} insights for Self-Improvement Engine")
            return stored_insight_ids

        except Exception as e:
            self.logger.error(f"Failed to store insights: {e}")
            return []

    def should_run_synthesis(self) -> bool:
        """Determine if synthesis should run based on time and data availability."""
        if self.last_synthesis_time is None:
            return True

        time_since_last = datetime.now() - self.last_synthesis_time
        return time_since_last.total_seconds() >= (self.config['synthesis_interval_hours'] * 3600)

    def get_synthesis_status(self) -> Dict[str, Any]:
        """Get current synthesis status for monitoring."""
        return {
            'last_synthesis': self.last_synthesis_time.isoformat() if self.last_synthesis_time else None,
            'synthesis_count': len(self.synthesis_history),
            'config': self.config,
            'next_synthesis_due': self.should_run_synthesis()
        }
