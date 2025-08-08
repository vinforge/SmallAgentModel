"""
SAM Self-Awareness & Memory Queries
Exposes introspection capabilities and memory search interfaces.

Sprint 6 Task 3: SAM Self-Awareness & Memory Queries
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .memory_manager import LongTermMemoryManager, MemoryEntry, MemorySearchResult
from .user_profiles import UserProfileManager

logger = logging.getLogger(__name__)

@dataclass
class SessionSummary:
    """Summary of a conversation session."""
    session_id: str
    date: str
    duration_minutes: int
    interaction_count: int
    tools_used: List[str]
    topics_discussed: List[str]
    key_insights: List[str]
    memory_entries_created: int

@dataclass
class KnowledgeNode:
    """Node in the knowledge graph."""
    node_id: str
    concept: str
    node_type: str  # 'topic', 'tool', 'insight', 'memory'
    importance: float
    connections: List[str]  # IDs of connected nodes
    metadata: Dict[str, Any]

@dataclass
class KnowledgeEdge:
    """Edge in the knowledge graph."""
    edge_id: str
    source_node: str
    target_node: str
    relationship: str  # 'related_to', 'used_for', 'leads_to', 'contradicts'
    strength: float
    evidence: List[str]  # Memory IDs that support this connection

class SelfAwarenessManager:
    """
    Manages SAM's self-awareness capabilities and memory introspection.
    """
    
    def __init__(self, memory_manager: LongTermMemoryManager, 
                 profile_manager: UserProfileManager):
        """
        Initialize the self-awareness manager.
        
        Args:
            memory_manager: Long-term memory manager instance
            profile_manager: User profile manager instance
        """
        self.memory_manager = memory_manager
        self.profile_manager = profile_manager
        
        # Session tracking
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        self.session_interactions: List[Dict[str, Any]] = []
        
        logger.info("Self-awareness manager initialized")
    
    def start_session(self, user_id: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            user_id: User ID for the session
            
        Returns:
            Session ID
        """
        import uuid
        
        self.current_session_id = f"session_{uuid.uuid4().hex[:12]}"
        self.session_start_time = datetime.now()
        self.session_interactions = []
        
        # Store session start in memory
        session_start_content = f"Started conversation session with user {user_id or 'anonymous'}"
        self.memory_manager.store_memory(
            content=session_start_content,
            content_type='session_start',
            tags=['session', 'start'],
            metadata={'user_id': user_id, 'session_id': self.current_session_id}
        )
        
        logger.info(f"Started session: {self.current_session_id}")
        return self.current_session_id
    
    def end_session(self) -> Optional[SessionSummary]:
        """
        End the current session and create a summary.
        
        Returns:
            SessionSummary if session was active, None otherwise
        """
        if not self.current_session_id or not self.session_start_time:
            return None
        
        try:
            # Calculate session duration
            duration = datetime.now() - self.session_start_time
            duration_minutes = int(duration.total_seconds() / 60)
            
            # Analyze session interactions
            tools_used = list(set(
                interaction.get('tool_used') 
                for interaction in self.session_interactions 
                if interaction.get('tool_used')
            ))
            
            topics_discussed = list(set(
                interaction.get('topic') 
                for interaction in self.session_interactions 
                if interaction.get('topic')
            ))
            
            # Extract key insights (simplified)
            key_insights = [
                interaction.get('insight', '')
                for interaction in self.session_interactions
                if interaction.get('insight')
            ]
            
            # Count memory entries created during session
            memory_entries_created = sum(
                1 for interaction in self.session_interactions
                if interaction.get('memory_created')
            )
            
            # Create session summary
            summary = SessionSummary(
                session_id=self.current_session_id,
                date=self.session_start_time.isoformat(),
                duration_minutes=duration_minutes,
                interaction_count=len(self.session_interactions),
                tools_used=tools_used,
                topics_discussed=topics_discussed,
                key_insights=key_insights,
                memory_entries_created=memory_entries_created
            )
            
            # Store session summary in memory
            summary_content = f"Session summary: {len(self.session_interactions)} interactions, {len(tools_used)} tools used, topics: {', '.join(topics_discussed[:3])}"
            self.memory_manager.store_memory(
                content=summary_content,
                content_type='session_summary',
                tags=['session', 'summary'] + topics_discussed[:3],
                metadata={
                    'session_id': self.current_session_id,
                    'duration_minutes': duration_minutes,
                    'tools_used': tools_used,
                    'topics_discussed': topics_discussed
                }
            )
            
            logger.info(f"Ended session: {self.current_session_id} ({duration_minutes} minutes)")
            
            # Reset session state
            self.current_session_id = None
            self.session_start_time = None
            self.session_interactions = []
            
            return summary
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return None
    
    def record_interaction(self, query: str, response: str, tools_used: List[str] = None,
                          topic: Optional[str] = None, insight: Optional[str] = None):
        """
        Record an interaction in the current session.
        
        Args:
            query: User query
            response: SAM's response
            tools_used: List of tools used
            topic: Identified topic
            insight: Key insight from the interaction
        """
        if not self.current_session_id:
            return
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'tools_used': tools_used or [],
            'topic': topic,
            'insight': insight,
            'memory_created': False
        }
        
        self.session_interactions.append(interaction)
        
        # Auto-store important interactions in memory
        if insight or (tools_used and len(tools_used) > 1):
            memory_content = f"Q: {query[:100]}... A: {response[:200]}..."
            memory_id = self.memory_manager.store_memory(
                content=memory_content,
                content_type='conversation',
                tags=['interaction'] + (tools_used or []) + ([topic] if topic else []),
                metadata={
                    'session_id': self.current_session_id,
                    'tools_used': tools_used,
                    'topic': topic,
                    'insight': insight
                }
            )
            interaction['memory_created'] = True
            interaction['memory_id'] = memory_id
    
    def search_memory(self, query: str, user_id: Optional[str] = None,
                     days_back: Optional[int] = None, content_type: Optional[str] = None) -> List[MemorySearchResult]:
        """
        Search memory with user context.
        
        Args:
            query: Search query
            user_id: User ID for personalized search
            days_back: Limit search to last N days
            content_type: Filter by content type
            
        Returns:
            List of memory search results
        """
        try:
            # Get user preferences for search customization
            search_params = {'top_k': 5, 'similarity_threshold': 0.3}
            
            if user_id:
                profile = self.profile_manager.get_user_profile(user_id)
                if profile:
                    # Adjust search based on user preferences
                    if profile.preferences.verbosity.value == 'detailed':
                        search_params['top_k'] = 10
                    elif profile.preferences.verbosity.value == 'concise':
                        search_params['top_k'] = 3
            
            # Perform memory search
            results = self.memory_manager.search_memories(
                query=query,
                content_type=content_type,
                **search_params
            )
            
            # Filter by date if specified
            if days_back:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                results = [
                    result for result in results
                    if datetime.fromisoformat(result.memory_entry.created_at) >= cutoff_date
                ]
            
            logger.debug(f"Memory search for '{query}': {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []
    
    def get_recent_memories(self, user_id: Optional[str] = None, days: int = 7) -> List[MemoryEntry]:
        """
        Get recent memories with user context.
        
        Args:
            user_id: User ID for personalized results
            days: Number of days to look back
            
        Returns:
            List of recent memory entries
        """
        try:
            # Get user preferences
            limit = 20
            if user_id:
                profile = self.profile_manager.get_user_profile(user_id)
                if profile:
                    if profile.preferences.verbosity.value == 'detailed':
                        limit = 30
                    elif profile.preferences.verbosity.value == 'concise':
                        limit = 10
            
            return self.memory_manager.get_recent_memories(days=days, limit=limit)
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return []
    
    def get_session_history(self, user_id: Optional[str] = None, 
                           last_n_sessions: int = 5) -> List[SessionSummary]:
        """
        Get summaries of recent sessions.
        
        Args:
            user_id: User ID to filter sessions
            last_n_sessions: Number of recent sessions to retrieve
            
        Returns:
            List of session summaries
        """
        try:
            # Search for session summaries
            session_memories = self.memory_manager.search_memories(
                query="session summary",
                content_type='session_summary',
                top_k=last_n_sessions * 2  # Get more to filter by user
            )
            
            summaries = []
            for result in session_memories:
                memory = result.memory_entry
                metadata = memory.metadata
                
                # Filter by user if specified
                if user_id and metadata.get('user_id') != user_id:
                    continue
                
                # Extract session info from metadata
                summary = SessionSummary(
                    session_id=metadata.get('session_id', 'unknown'),
                    date=memory.created_at,
                    duration_minutes=metadata.get('duration_minutes', 0),
                    interaction_count=metadata.get('interaction_count', 0),
                    tools_used=metadata.get('tools_used', []),
                    topics_discussed=metadata.get('topics_discussed', []),
                    key_insights=[],  # Would need to extract from content
                    memory_entries_created=metadata.get('memory_entries_created', 0)
                )
                
                summaries.append(summary)
                
                if len(summaries) >= last_n_sessions:
                    break
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return []
    
    def generate_knowledge_graph(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a knowledge graph from memory and interactions.
        
        Args:
            user_id: User ID to personalize the graph
            
        Returns:
            Knowledge graph as JSON structure
        """
        try:
            # Get recent memories for graph generation
            memories = self.get_recent_memories(user_id=user_id, days=30)
            
            nodes = {}
            edges = []
            
            # Create nodes from memories
            for memory in memories:
                # Topic nodes
                for tag in memory.tags:
                    if tag not in nodes:
                        nodes[tag] = KnowledgeNode(
                            node_id=tag,
                            concept=tag,
                            node_type='topic',
                            importance=0.5,
                            connections=[],
                            metadata={'memory_count': 0}
                        )
                    nodes[tag].metadata['memory_count'] += 1
                    nodes[tag].importance = min(1.0, nodes[tag].metadata['memory_count'] * 0.1)
                
                # Tool nodes
                tools_used = memory.metadata.get('tools_used', [])
                for tool in tools_used:
                    tool_id = f"tool_{tool}"
                    if tool_id not in nodes:
                        nodes[tool_id] = KnowledgeNode(
                            node_id=tool_id,
                            concept=tool,
                            node_type='tool',
                            importance=0.3,
                            connections=[],
                            metadata={'usage_count': 0}
                        )
                    nodes[tool_id].metadata['usage_count'] += 1
                    nodes[tool_id].importance = min(1.0, nodes[tool_id].metadata['usage_count'] * 0.15)
            
            # Create edges between related concepts
            for memory in memories:
                memory_tags = memory.tags
                memory_tools = [f"tool_{tool}" for tool in memory.metadata.get('tools_used', [])]
                all_concepts = memory_tags + memory_tools
                
                # Create edges between concepts that appear together
                for i, concept1 in enumerate(all_concepts):
                    for concept2 in all_concepts[i+1:]:
                        if concept1 in nodes and concept2 in nodes:
                            edge = KnowledgeEdge(
                                edge_id=f"{concept1}_{concept2}",
                                source_node=concept1,
                                target_node=concept2,
                                relationship='related_to',
                                strength=0.5,
                                evidence=[memory.memory_id]
                            )
                            edges.append(edge)
                            
                            # Update node connections
                            if concept2 not in nodes[concept1].connections:
                                nodes[concept1].connections.append(concept2)
                            if concept1 not in nodes[concept2].connections:
                                nodes[concept2].connections.append(concept1)
            
            # Convert to JSON-serializable format
            graph_data = {
                'nodes': [
                    {
                        'id': node.node_id,
                        'concept': node.concept,
                        'type': node.node_type,
                        'importance': node.importance,
                        'connections': node.connections,
                        'metadata': node.metadata
                    }
                    for node in nodes.values()
                ],
                'edges': [
                    {
                        'id': edge.edge_id,
                        'source': edge.source_node,
                        'target': edge.target_node,
                        'relationship': edge.relationship,
                        'strength': edge.strength,
                        'evidence_count': len(edge.evidence)
                    }
                    for edge in edges
                ],
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'user_id': user_id,
                    'node_count': len(nodes),
                    'edge_count': len(edges),
                    'memory_count': len(memories)
                }
            }
            
            logger.info(f"Generated knowledge graph: {len(nodes)} nodes, {len(edges)} edges")
            return graph_data
            
        except Exception as e:
            logger.error(f"Error generating knowledge graph: {e}")
            return {'nodes': [], 'edges': [], 'metadata': {}}
    
    def answer_meta_query(self, query: str, user_id: Optional[str] = None) -> str:
        """
        Answer meta-queries about past conversations and memory.
        
        Args:
            query: Meta-query (e.g., "What did we talk about yesterday?")
            user_id: User ID for context
            
        Returns:
            Answer to the meta-query
        """
        try:
            query_lower = query.lower()
            
            # Parse temporal references
            days_back = 7  # Default
            if 'yesterday' in query_lower:
                days_back = 1
            elif 'last week' in query_lower:
                days_back = 7
            elif 'last month' in query_lower:
                days_back = 30
            
            # Determine query type and respond accordingly
            if any(phrase in query_lower for phrase in ['what did we talk about', 'what did we discuss']):
                # Get recent memories and summarize topics
                memories = self.get_recent_memories(user_id=user_id, days=days_back)
                
                if not memories:
                    return f"I don't have any conversation records from the last {days_back} day(s)."
                
                # Extract topics and tools
                topics = set()
                tools_used = set()
                
                for memory in memories:
                    topics.update(memory.tags)
                    if memory.metadata.get('tools_used'):
                        tools_used.update(memory.metadata['tools_used'])
                
                # Remove generic tags
                topics = {t for t in topics if t not in ['interaction', 'session', 'start', 'summary']}
                
                response_parts = [f"In the last {days_back} day(s), we discussed:"]
                
                if topics:
                    response_parts.append(f"**Topics**: {', '.join(list(topics)[:10])}")
                
                if tools_used:
                    response_parts.append(f"**Tools used**: {', '.join(list(tools_used))}")
                
                response_parts.append(f"**Total interactions**: {len(memories)}")
                
                return "\n".join(response_parts)
            
            elif any(phrase in query_lower for phrase in ['how many', 'how much']):
                # Quantitative queries
                if 'session' in query_lower:
                    sessions = self.get_session_history(user_id=user_id, last_n_sessions=10)
                    return f"You've had {len(sessions)} conversation sessions recently."
                
                elif 'tool' in query_lower:
                    memories = self.get_recent_memories(user_id=user_id, days=30)
                    tool_usage = {}
                    for memory in memories:
                        for tool in memory.metadata.get('tools_used', []):
                            tool_usage[tool] = tool_usage.get(tool, 0) + 1
                    
                    if tool_usage:
                        tool_summary = ', '.join([f"{tool}: {count}" for tool, count in tool_usage.items()])
                        return f"Tool usage in the last 30 days: {tool_summary}"
                    else:
                        return "No tool usage recorded in the last 30 days."
            
            else:
                # General memory search
                search_results = self.search_memory(query, user_id=user_id, days_back=days_back)
                
                if not search_results:
                    return f"I couldn't find any relevant information about '{query}' in our recent conversations."
                
                # Summarize search results
                response_parts = [f"Here's what I found about '{query}':"]
                
                for i, result in enumerate(search_results[:3], 1):
                    memory = result.memory_entry
                    response_parts.append(f"{i}. {memory.summary} (confidence: {result.similarity_score:.2f})")
                
                return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error answering meta-query: {e}")
            return f"I encountered an error while searching my memory: {str(e)}"
    
    def get_self_awareness_stats(self) -> Dict[str, Any]:
        """Get statistics about SAM's self-awareness and memory."""
        try:
            memory_stats = self.memory_manager.get_memory_stats()
            
            return {
                'current_session_id': self.current_session_id,
                'session_active': self.current_session_id is not None,
                'session_interactions': len(self.session_interactions) if self.session_interactions else 0,
                'memory_stats': memory_stats,
                'capabilities': [
                    'memory_search',
                    'session_tracking',
                    'knowledge_graph_generation',
                    'meta_query_answering',
                    'temporal_reasoning'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting self-awareness stats: {e}")
            return {}

# Global self-awareness manager instance
_self_awareness_manager = None

def get_self_awareness_manager(memory_manager: LongTermMemoryManager, 
                              profile_manager: UserProfileManager) -> SelfAwarenessManager:
    """Get or create a global self-awareness manager instance."""
    global _self_awareness_manager
    
    if _self_awareness_manager is None:
        _self_awareness_manager = SelfAwarenessManager(memory_manager, profile_manager)
    
    return _self_awareness_manager
