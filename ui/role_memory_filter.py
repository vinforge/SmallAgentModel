"""
Role-Based Memory Filtering for SAM
Agent-specific memory scoping for collaborative work with role-based access.

Sprint 12 Task 5: Role-Based Memory Filtering
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_vectorstore import MemoryVectorStore, MemoryType, MemorySearchResult, get_memory_store
from memory.memory_reasoning import MemoryDrivenReasoningEngine, get_memory_reasoning_engine
from agents.task_router import AgentRole

logger = logging.getLogger(__name__)

class MemoryAccessLevel(Enum):
    """Memory access levels for role-based filtering."""
    PUBLIC = "public"           # Accessible to all agents
    ROLE_SPECIFIC = "role"      # Accessible to specific roles
    AGENT_PRIVATE = "private"   # Accessible only to creating agent
    RESTRICTED = "restricted"   # Requires special permissions

@dataclass
class MemoryAccessRule:
    """Rule defining memory access for roles."""
    memory_types: List[MemoryType]
    allowed_roles: List[AgentRole]
    access_level: MemoryAccessLevel
    conditions: Dict[str, Any]
    description: str

@dataclass
class RoleMemoryContext:
    """Memory context filtered for a specific role."""
    role: AgentRole
    agent_id: str
    accessible_memories: List[MemorySearchResult]
    filtered_count: int
    access_summary: Dict[str, int]
    role_specific_insights: List[str]

class RoleBasedMemoryFilter:
    """
    Filters and scopes memory access based on agent roles and permissions.
    """
    
    def __init__(self):
        """Initialize the role-based memory filter."""
        self.memory_store = get_memory_store()
        self.memory_reasoning = get_memory_reasoning_engine()
        
        # Initialize access rules
        self.access_rules = self._initialize_access_rules()
        
        # Role-specific memory preferences
        self.role_preferences = self._initialize_role_preferences()
        
        logger.info("Role-based memory filter initialized")
    
    def filter_memories_for_role(self, role: AgentRole, agent_id: str,
                                query: str = None, max_results: int = 10,
                                user_id: str = None) -> RoleMemoryContext:
        """
        Filter memories based on agent role and permissions.
        
        Args:
            role: Agent role
            agent_id: Specific agent ID
            query: Optional search query
            max_results: Maximum results to return
            user_id: Optional user ID filter
            
        Returns:
            RoleMemoryContext with filtered memories
        """
        try:
            # Get all relevant memories
            if query:
                all_results = self.memory_reasoning.search_memories(
                    query=query,
                    user_id=user_id,
                    max_results=max_results * 3  # Get more to filter
                )
            else:
                # Get recent memories
                all_memories = list(self.memory_store.memory_chunks.values())
                if user_id:
                    all_memories = [m for m in all_memories if m.metadata.get('user_id') == user_id]
                
                # Sort by relevance to role and recency
                all_memories.sort(key=lambda m: (
                    self._calculate_role_relevance(m, role),
                    m.timestamp
                ), reverse=True)
                
                # Convert to search results format
                all_results = [
                    type('SearchResult', (), {
                        'chunk': memory,
                        'similarity_score': self._calculate_role_relevance(memory, role)
                    })()
                    for memory in all_memories[:max_results * 3]
                ]
            
            # Apply role-based filtering
            accessible_memories = []
            filtered_count = 0
            access_summary = {}
            
            for result in all_results:
                memory = result.chunk
                
                # Check access permissions
                if self._check_memory_access(memory, role, agent_id):
                    accessible_memories.append(result)
                    
                    # Update access summary
                    mem_type = memory.memory_type.value
                    access_summary[mem_type] = access_summary.get(mem_type, 0) + 1
                else:
                    filtered_count += 1
                
                if len(accessible_memories) >= max_results:
                    break
            
            # Generate role-specific insights
            role_insights = self._generate_role_insights(accessible_memories, role)
            
            return RoleMemoryContext(
                role=role,
                agent_id=agent_id,
                accessible_memories=accessible_memories,
                filtered_count=filtered_count,
                access_summary=access_summary,
                role_specific_insights=role_insights
            )
            
        except Exception as e:
            logger.error(f"Error filtering memories for role {role}: {e}")
            return RoleMemoryContext(
                role=role,
                agent_id=agent_id,
                accessible_memories=[],
                filtered_count=0,
                access_summary={},
                role_specific_insights=[]
            )
    
    def get_role_memory_permissions(self, role: AgentRole) -> Dict[str, Any]:
        """
        Get memory permissions for a specific role.
        
        Args:
            role: Agent role
            
        Returns:
            Dictionary of permissions and access rules
        """
        try:
            permissions = {
                'role': role.value,
                'access_levels': [],
                'allowed_memory_types': [],
                'restricted_memory_types': [],
                'special_permissions': [],
                'access_rules': []
            }
            
            # Check each access rule
            for rule in self.access_rules:
                if role in rule.allowed_roles:
                    permissions['access_levels'].append(rule.access_level.value)
                    permissions['allowed_memory_types'].extend([mt.value for mt in rule.memory_types])
                    permissions['access_rules'].append({
                        'description': rule.description,
                        'memory_types': [mt.value for mt in rule.memory_types],
                        'access_level': rule.access_level.value,
                        'conditions': rule.conditions
                    })
            
            # Remove duplicates
            permissions['access_levels'] = list(set(permissions['access_levels']))
            permissions['allowed_memory_types'] = list(set(permissions['allowed_memory_types']))
            
            # Determine restricted types
            all_types = [mt.value for mt in MemoryType]
            permissions['restricted_memory_types'] = [
                mt for mt in all_types 
                if mt not in permissions['allowed_memory_types']
            ]
            
            # Add role-specific permissions
            role_prefs = self.role_preferences.get(role, {})
            permissions['special_permissions'] = role_prefs.get('special_permissions', [])
            
            return permissions
            
        except Exception as e:
            logger.error(f"Error getting role permissions: {e}")
            return {'error': str(e)}
    
    def create_role_specific_memory(self, content: str, memory_type: MemoryType,
                                  source: str, role: AgentRole, agent_id: str,
                                  access_level: MemoryAccessLevel = MemoryAccessLevel.ROLE_SPECIFIC,
                                  allowed_roles: List[AgentRole] = None,
                                  user_id: str = None) -> str:
        """
        Create a memory with role-specific access controls.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            source: Memory source
            role: Creating agent role
            agent_id: Creating agent ID
            access_level: Access level for the memory
            allowed_roles: Optional list of roles that can access this memory
            user_id: Optional user ID
            
        Returns:
            Memory chunk ID
        """
        try:
            # Determine allowed roles
            if allowed_roles is None:
                if access_level == MemoryAccessLevel.PUBLIC:
                    allowed_roles = list(AgentRole)
                elif access_level == MemoryAccessLevel.ROLE_SPECIFIC:
                    allowed_roles = [role]
                elif access_level == MemoryAccessLevel.AGENT_PRIVATE:
                    allowed_roles = []
                else:
                    allowed_roles = []
            
            # Create memory metadata with access controls
            metadata = {
                'created_by_role': role.value,
                'created_by_agent': agent_id,
                'access_level': access_level.value,
                'allowed_roles': [r.value for r in allowed_roles],
                'created_at': datetime.now().isoformat(),
                'user_id': user_id
            }
            
            # Add role-specific tags
            tags = [f"role:{role.value}", f"access:{access_level.value}"]
            if allowed_roles:
                tags.extend([f"allowed:{r.value}" for r in allowed_roles])
            
            # Calculate importance based on role
            importance_score = self._calculate_role_importance(memory_type, role)
            
            # Create the memory
            chunk_id = self.memory_store.add_memory(
                content=content,
                memory_type=memory_type,
                source=source,
                tags=tags,
                importance_score=importance_score,
                metadata=metadata
            )
            
            logger.info(f"Created role-specific memory: {chunk_id} for role {role.value}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error creating role-specific memory: {e}")
            raise
    
    def get_collaborative_memories(self, roles: List[AgentRole], 
                                 query: str = None, max_results: int = 20) -> Dict[str, Any]:
        """
        Get memories accessible to multiple roles for collaborative work.
        
        Args:
            roles: List of agent roles
            query: Optional search query
            max_results: Maximum results per role
            
        Returns:
            Dictionary with collaborative memory context
        """
        try:
            collaborative_context = {
                'roles': [r.value for r in roles],
                'shared_memories': [],
                'role_specific_memories': {},
                'collaboration_insights': [],
                'access_matrix': {}
            }
            
            # Get memories for each role
            for role in roles:
                role_context = self.filter_memories_for_role(
                    role=role,
                    agent_id=f"collab_{role.value}",
                    query=query,
                    max_results=max_results
                )
                
                collaborative_context['role_specific_memories'][role.value] = {
                    'accessible_count': len(role_context.accessible_memories),
                    'filtered_count': role_context.filtered_count,
                    'access_summary': role_context.access_summary,
                    'insights': role_context.role_specific_insights
                }
            
            # Find shared memories (accessible to multiple roles)
            all_memories = list(self.memory_store.memory_chunks.values())
            
            for memory in all_memories:
                accessible_roles = []
                
                for role in roles:
                    if self._check_memory_access(memory, role, f"collab_{role.value}"):
                        accessible_roles.append(role.value)
                
                if len(accessible_roles) > 1:
                    collaborative_context['shared_memories'].append({
                        'chunk_id': memory.chunk_id,
                        'content': memory.content[:150] + "..." if len(memory.content) > 150 else memory.content,
                        'memory_type': memory.memory_type.value,
                        'source': memory.source,
                        'accessible_to': accessible_roles,
                        'importance_score': memory.importance_score
                    })
                
                # Build access matrix
                collaborative_context['access_matrix'][memory.chunk_id] = accessible_roles
            
            # Generate collaboration insights
            collaborative_context['collaboration_insights'] = self._generate_collaboration_insights(
                collaborative_context, roles
            )
            
            return collaborative_context
            
        except Exception as e:
            logger.error(f"Error getting collaborative memories: {e}")
            return {'error': str(e)}
    
    def _check_memory_access(self, memory, role: AgentRole, agent_id: str) -> bool:
        """Check if a role/agent has access to a specific memory."""
        try:
            # Get memory access metadata
            access_level = memory.metadata.get('access_level', 'public')
            allowed_roles = memory.metadata.get('allowed_roles', [])
            created_by_agent = memory.metadata.get('created_by_agent')
            
            # Public access
            if access_level == MemoryAccessLevel.PUBLIC.value:
                return True
            
            # Agent private access
            if access_level == MemoryAccessLevel.AGENT_PRIVATE.value:
                return created_by_agent == agent_id
            
            # Role-specific access
            if access_level == MemoryAccessLevel.ROLE_SPECIFIC.value:
                return role.value in allowed_roles
            
            # Restricted access - check special permissions
            if access_level == MemoryAccessLevel.RESTRICTED.value:
                role_prefs = self.role_preferences.get(role, {})
                special_perms = role_prefs.get('special_permissions', [])
                return 'access_restricted' in special_perms
            
            # Default to checking role-based rules
            for rule in self.access_rules:
                if (memory.memory_type in rule.memory_types and 
                    role in rule.allowed_roles):
                    
                    # Check additional conditions
                    if rule.conditions:
                        if not self._check_access_conditions(memory, rule.conditions, role, agent_id):
                            continue
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking memory access: {e}")
            return False
    
    def _check_access_conditions(self, memory, conditions: Dict[str, Any], 
                               role: AgentRole, agent_id: str) -> bool:
        """Check additional access conditions."""
        try:
            # Check importance threshold
            if 'min_importance' in conditions:
                if memory.importance_score < conditions['min_importance']:
                    return False
            
            # Check age restrictions
            if 'max_age_days' in conditions:
                memory_date = datetime.fromisoformat(memory.timestamp)
                age_days = (datetime.now() - memory_date).days
                if age_days > conditions['max_age_days']:
                    return False
            
            # Check tag requirements
            if 'required_tags' in conditions:
                required_tags = conditions['required_tags']
                if not all(tag in memory.tags for tag in required_tags):
                    return False
            
            # Check source restrictions
            if 'allowed_sources' in conditions:
                allowed_sources = conditions['allowed_sources']
                if not any(source in memory.source for source in allowed_sources):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking access conditions: {e}")
            return False
    
    def _calculate_role_relevance(self, memory, role: AgentRole) -> float:
        """Calculate how relevant a memory is to a specific role."""
        try:
            base_relevance = memory.importance_score
            
            # Role-specific relevance factors
            role_prefs = self.role_preferences.get(role, {})
            
            # Memory type preferences
            preferred_types = role_prefs.get('preferred_memory_types', [])
            if memory.memory_type in preferred_types:
                base_relevance += 0.2
            
            # Tag preferences
            preferred_tags = role_prefs.get('preferred_tags', [])
            tag_matches = sum(1 for tag in memory.tags if tag in preferred_tags)
            base_relevance += tag_matches * 0.1
            
            # Source preferences
            preferred_sources = role_prefs.get('preferred_sources', [])
            if any(source in memory.source for source in preferred_sources):
                base_relevance += 0.15
            
            # Recency factor
            memory_date = datetime.fromisoformat(memory.timestamp)
            days_old = (datetime.now() - memory_date).days
            recency_factor = max(0, 1 - (days_old / 30))  # Decay over 30 days
            base_relevance += recency_factor * 0.1
            
            return min(1.0, base_relevance)
            
        except Exception as e:
            logger.error(f"Error calculating role relevance: {e}")
            return memory.importance_score
    
    def _calculate_role_importance(self, memory_type: MemoryType, role: AgentRole) -> float:
        """Calculate importance score based on memory type and role."""
        try:
            base_importance = 0.5
            
            # Role-specific importance adjustments
            role_prefs = self.role_preferences.get(role, {})
            type_importance = role_prefs.get('memory_type_importance', {})
            
            if memory_type in type_importance:
                base_importance = type_importance[memory_type]
            
            return base_importance
            
        except Exception as e:
            logger.error(f"Error calculating role importance: {e}")
            return 0.5
    
    def _generate_role_insights(self, memories: List, role: AgentRole) -> List[str]:
        """Generate role-specific insights from accessible memories."""
        try:
            insights = []
            
            if not memories:
                return insights
            
            # Memory type distribution
            type_counts = {}
            for result in memories:
                mem_type = result.chunk.memory_type.value
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            
            if type_counts:
                most_common_type = max(type_counts, key=type_counts.get)
                insights.append(f"Most accessible memory type: {most_common_type} ({type_counts[most_common_type]} memories)")
            
            # High importance memories
            high_importance = [r for r in memories if r.chunk.importance_score >= 0.8]
            if high_importance:
                insights.append(f"Found {len(high_importance)} high-importance memories relevant to {role.value} role")
            
            # Recent activity
            recent_memories = [
                r for r in memories 
                if (datetime.now() - datetime.fromisoformat(r.chunk.timestamp)).days <= 7
            ]
            if recent_memories:
                insights.append(f"Found {len(recent_memories)} memories from the last week")
            
            # Role-specific patterns
            role_prefs = self.role_preferences.get(role, {})
            preferred_tags = role_prefs.get('preferred_tags', [])
            
            if preferred_tags:
                tag_matches = 0
                for result in memories:
                    if any(tag in result.chunk.tags for tag in preferred_tags):
                        tag_matches += 1
                
                if tag_matches > 0:
                    insights.append(f"Found {tag_matches} memories matching {role.value} preferences")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating role insights: {e}")
            return []
    
    def _generate_collaboration_insights(self, context: Dict[str, Any], roles: List[AgentRole]) -> List[str]:
        """Generate insights about collaborative memory access."""
        try:
            insights = []
            
            shared_count = len(context['shared_memories'])
            total_roles = len(roles)
            
            if shared_count > 0:
                insights.append(f"Found {shared_count} memories accessible to multiple roles")
            
            # Role coverage analysis
            role_coverage = {}
            for memory_id, accessible_roles in context['access_matrix'].items():
                coverage = len(accessible_roles) / total_roles
                if coverage >= 0.5:
                    role_coverage[memory_id] = coverage
            
            if role_coverage:
                insights.append(f"Found {len(role_coverage)} memories with broad role access (50%+ coverage)")
            
            # Role-specific insights
            for role_name, role_data in context['role_specific_memories'].items():
                accessible = role_data['accessible_count']
                filtered = role_data['filtered_count']
                
                if filtered > 0:
                    insights.append(f"{role_name}: {accessible} accessible, {filtered} filtered memories")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating collaboration insights: {e}")
            return []
    
    def _initialize_access_rules(self) -> List[MemoryAccessRule]:
        """Initialize default access rules for different roles."""
        return [
            # Planner access rules
            MemoryAccessRule(
                memory_types=[MemoryType.REASONING, MemoryType.INSIGHT, MemoryType.PROCEDURE],
                allowed_roles=[AgentRole.PLANNER],
                access_level=MemoryAccessLevel.ROLE_SPECIFIC,
                conditions={'min_importance': 0.6},
                description="Planners can access reasoning, insights, and procedures"
            ),
            
            # Executor access rules
            MemoryAccessRule(
                memory_types=[MemoryType.DOCUMENT, MemoryType.FACT, MemoryType.PROCEDURE],
                allowed_roles=[AgentRole.EXECUTOR],
                access_level=MemoryAccessLevel.ROLE_SPECIFIC,
                conditions={},
                description="Executors can access documents, facts, and procedures"
            ),
            
            # Validator access rules
            MemoryAccessRule(
                memory_types=[MemoryType.FACT, MemoryType.INSIGHT, MemoryType.REASONING],
                allowed_roles=[AgentRole.VALIDATOR],
                access_level=MemoryAccessLevel.ROLE_SPECIFIC,
                conditions={'min_importance': 0.5},
                description="Validators can access facts, insights, and reasoning for verification"
            ),
            
            # Critic access rules
            MemoryAccessRule(
                memory_types=list(MemoryType),  # Critics can access all types
                allowed_roles=[AgentRole.CRITIC],
                access_level=MemoryAccessLevel.ROLE_SPECIFIC,
                conditions={},
                description="Critics can access all memory types for comprehensive analysis"
            ),
            
            # Synthesizer access rules
            MemoryAccessRule(
                memory_types=[MemoryType.INSIGHT, MemoryType.REASONING, MemoryType.CONVERSATION],
                allowed_roles=[AgentRole.SYNTHESIZER],
                access_level=MemoryAccessLevel.ROLE_SPECIFIC,
                conditions={'min_importance': 0.4},
                description="Synthesizers can access insights, reasoning, and conversations"
            ),
            
            # Public access rule
            MemoryAccessRule(
                memory_types=[MemoryType.FACT, MemoryType.DOCUMENT],
                allowed_roles=list(AgentRole),
                access_level=MemoryAccessLevel.PUBLIC,
                conditions={},
                description="All roles can access public facts and documents"
            )
        ]
    
    def _initialize_role_preferences(self) -> Dict[AgentRole, Dict[str, Any]]:
        """Initialize role-specific memory preferences."""
        return {
            AgentRole.PLANNER: {
                'preferred_memory_types': [MemoryType.REASONING, MemoryType.INSIGHT, MemoryType.PROCEDURE],
                'preferred_tags': ['planning', 'strategy', 'coordination', 'high_level'],
                'preferred_sources': ['planning_session', 'strategy_meeting', 'coordination'],
                'memory_type_importance': {
                    MemoryType.REASONING: 0.8,
                    MemoryType.INSIGHT: 0.7,
                    MemoryType.PROCEDURE: 0.6,
                    MemoryType.FACT: 0.5,
                    MemoryType.DOCUMENT: 0.4,
                    MemoryType.CONVERSATION: 0.3
                },
                'special_permissions': ['access_strategic']
            },
            
            AgentRole.EXECUTOR: {
                'preferred_memory_types': [MemoryType.DOCUMENT, MemoryType.FACT, MemoryType.PROCEDURE],
                'preferred_tags': ['execution', 'implementation', 'task', 'action'],
                'preferred_sources': ['task_execution', 'implementation', 'processing'],
                'memory_type_importance': {
                    MemoryType.DOCUMENT: 0.8,
                    MemoryType.FACT: 0.7,
                    MemoryType.PROCEDURE: 0.8,
                    MemoryType.REASONING: 0.4,
                    MemoryType.INSIGHT: 0.3,
                    MemoryType.CONVERSATION: 0.5
                },
                'special_permissions': ['access_operational']
            },
            
            AgentRole.VALIDATOR: {
                'preferred_memory_types': [MemoryType.FACT, MemoryType.INSIGHT, MemoryType.REASONING],
                'preferred_tags': ['validation', 'verification', 'quality', 'accuracy'],
                'preferred_sources': ['validation_session', 'quality_check', 'verification'],
                'memory_type_importance': {
                    MemoryType.FACT: 0.9,
                    MemoryType.INSIGHT: 0.7,
                    MemoryType.REASONING: 0.8,
                    MemoryType.DOCUMENT: 0.6,
                    MemoryType.PROCEDURE: 0.5,
                    MemoryType.CONVERSATION: 0.4
                },
                'special_permissions': ['access_validation']
            },
            
            AgentRole.CRITIC: {
                'preferred_memory_types': list(MemoryType),
                'preferred_tags': ['critical', 'analysis', 'evaluation', 'review'],
                'preferred_sources': ['critical_analysis', 'review_session', 'evaluation'],
                'memory_type_importance': {
                    MemoryType.REASONING: 0.9,
                    MemoryType.INSIGHT: 0.8,
                    MemoryType.FACT: 0.7,
                    MemoryType.DOCUMENT: 0.6,
                    MemoryType.PROCEDURE: 0.5,
                    MemoryType.CONVERSATION: 0.7
                },
                'special_permissions': ['access_restricted', 'access_all_types']
            },
            
            AgentRole.SYNTHESIZER: {
                'preferred_memory_types': [MemoryType.INSIGHT, MemoryType.REASONING, MemoryType.CONVERSATION],
                'preferred_tags': ['synthesis', 'integration', 'combination', 'summary'],
                'preferred_sources': ['synthesis_session', 'integration', 'summary'],
                'memory_type_importance': {
                    MemoryType.INSIGHT: 0.9,
                    MemoryType.REASONING: 0.8,
                    MemoryType.CONVERSATION: 0.7,
                    MemoryType.FACT: 0.6,
                    MemoryType.DOCUMENT: 0.5,
                    MemoryType.PROCEDURE: 0.4
                },
                'special_permissions': ['access_synthesis']
            }
        }

# Global role-based memory filter instance
_role_filter = None

def get_role_filter() -> RoleBasedMemoryFilter:
    """Get or create a global role-based memory filter instance."""
    global _role_filter
    
    if _role_filter is None:
        _role_filter = RoleBasedMemoryFilter()
    
    return _role_filter
