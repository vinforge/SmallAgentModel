"""
Configurable Knowledge Capsules for SAM
Allows users to save and reuse knowledge or tool strategies.

Sprint 6 Task 5: Configurable Knowledge Capsules
"""

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from reasoning.self_decide_framework import SelfDecideSession
from reasoning.tool_executor import ToolResponse

logger = logging.getLogger(__name__)

@dataclass
class CapsuleContent:
    """Content stored in a knowledge capsule."""
    reasoning_trace: str
    tools_used: List[str]
    tool_outputs: List[Dict[str, Any]]
    documents_referenced: List[str]
    key_insights: List[str]
    methodology: str
    success_factors: List[str]
    limitations: List[str]

@dataclass
class KnowledgeCapsule:
    """A reusable knowledge capsule containing reasoning and strategies."""
    capsule_id: str
    name: str
    summary: str
    content: CapsuleContent
    created_at: str
    created_by: str
    last_used: str
    use_count: int
    tags: List[str]
    category: str
    effectiveness_score: float
    metadata: Dict[str, Any]

class KnowledgeCapsuleManager:
    """
    Manages knowledge capsules for saving and reusing reasoning strategies.
    """
    
    def __init__(self, capsules_directory: str = "capsules"):
        """
        Initialize the knowledge capsule manager.
        
        Args:
            capsules_directory: Directory to store capsule files
        """
        self.capsules_dir = Path(capsules_directory)
        self.capsules_dir.mkdir(exist_ok=True)
        
        # In-memory capsule index
        self.capsules: Dict[str, KnowledgeCapsule] = {}
        
        # Load existing capsules
        self._load_capsules()
        
        logger.info(f"Knowledge capsule manager initialized with {len(self.capsules)} capsules")
    
    def create_capsule(self, name: str, session: SelfDecideSession, 
                      tool_responses: List[ToolResponse] = None,
                      user_id: str = "anonymous", category: str = "general",
                      tags: List[str] = None) -> str:
        """
        Create a new knowledge capsule from a reasoning session.
        
        Args:
            name: Name for the capsule
            session: SELF-DECIDE session to capture
            tool_responses: Tool responses from the session
            user_id: User who created the capsule
            category: Category for organization
            tags: Tags for searchability
            
        Returns:
            Capsule ID
        """
        try:
            capsule_id = f"capsule_{uuid.uuid4().hex[:12]}"
            
            # Extract reasoning trace
            reasoning_trace = self._extract_reasoning_trace(session)
            
            # Extract tools and outputs
            tools_used = []
            tool_outputs = []
            
            if tool_responses:
                for tr in tool_responses:
                    if tr.success:
                        tools_used.append(tr.tool_name)
                        tool_outputs.append({
                            'tool_name': tr.tool_name,
                            'output': tr.output,
                            'execution_time_ms': tr.execution_time_ms,
                            'metadata': tr.metadata
                        })
            
            # Extract session tools
            session_tools = [exec_info.get('tool_name') for exec_info in session.tool_executions 
                           if exec_info.get('success')]
            tools_used.extend(session_tools)
            tools_used = list(set(tools_used))  # Remove duplicates
            
            # Extract key insights
            key_insights = self._extract_key_insights(session, tool_responses)
            
            # Determine methodology
            methodology = self._determine_methodology(session, tools_used)
            
            # Identify success factors
            success_factors = self._identify_success_factors(session, tool_responses)
            
            # Identify limitations
            limitations = self._identify_limitations(session, tool_responses)
            
            # Create capsule content
            content = CapsuleContent(
                reasoning_trace=reasoning_trace,
                tools_used=tools_used,
                tool_outputs=tool_outputs,
                documents_referenced=[],  # Would need to extract from session
                key_insights=key_insights,
                methodology=methodology,
                success_factors=success_factors,
                limitations=limitations
            )
            
            # Create summary
            summary = f"Reasoning strategy for '{session.original_query[:50]}...' using {len(tools_used)} tools"
            
            # Create capsule
            capsule = KnowledgeCapsule(
                capsule_id=capsule_id,
                name=name,
                summary=summary,
                content=content,
                created_at=datetime.now().isoformat(),
                created_by=user_id,
                last_used=datetime.now().isoformat(),
                use_count=0,
                tags=tags or [],
                category=category,
                effectiveness_score=session.confidence_score,
                metadata={
                    'original_query': session.original_query,
                    'session_id': session.session_id,
                    'reasoning_steps': len(session.reasoning_steps),
                    'total_duration_ms': session.total_duration_ms
                }
            )
            
            # Store capsule
            self.capsules[capsule_id] = capsule
            self._save_capsule(capsule)
            
            logger.info(f"Created knowledge capsule: {name} ({capsule_id})")
            return capsule_id
            
        except Exception as e:
            logger.error(f"Error creating knowledge capsule: {e}")
            raise
    
    def load_capsule(self, capsule_id: str) -> Optional[KnowledgeCapsule]:
        """
        Load a knowledge capsule by ID.
        
        Args:
            capsule_id: ID of the capsule to load
            
        Returns:
            KnowledgeCapsule if found, None otherwise
        """
        try:
            capsule = self.capsules.get(capsule_id)
            
            if capsule:
                # Update usage statistics
                capsule.last_used = datetime.now().isoformat()
                capsule.use_count += 1
                
                # Save updated capsule
                self._save_capsule(capsule)
                
                logger.debug(f"Loaded knowledge capsule: {capsule.name} ({capsule_id})")
                return capsule
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading capsule {capsule_id}: {e}")
            return None
    
    def search_capsules(self, query: str, category: Optional[str] = None,
                       tags: Optional[List[str]] = None, user_id: Optional[str] = None,
                       top_k: int = 5) -> List[KnowledgeCapsule]:
        """
        Search knowledge capsules.
        
        Args:
            query: Search query
            category: Filter by category
            tags: Filter by tags (any match)
            user_id: Filter by creator
            top_k: Maximum number of results
            
        Returns:
            List of matching capsules
        """
        try:
            results = []
            query_words = set(query.lower().split())
            
            for capsule in self.capsules.values():
                # Apply filters
                if category and capsule.category != category:
                    continue
                
                if tags and not any(tag in capsule.tags for tag in tags):
                    continue
                
                if user_id and capsule.created_by != user_id:
                    continue
                
                # Calculate relevance score
                score = 0.0
                
                # Name and summary matching
                name_words = set(capsule.name.lower().split())
                summary_words = set(capsule.summary.lower().split())
                tag_words = set(' '.join(capsule.tags).lower().split())
                
                all_capsule_words = name_words | summary_words | tag_words
                matches = len(query_words & all_capsule_words)
                score += matches / len(query_words) if query_words else 0
                
                # Boost by effectiveness and usage
                score *= (0.5 + 0.3 * capsule.effectiveness_score + 0.2 * min(1.0, capsule.use_count / 10))
                
                if score > 0:
                    results.append((capsule, score))
            
            # Sort by score and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return [capsule for capsule, _ in results[:top_k]]
            
        except Exception as e:
            logger.error(f"Error searching capsules: {e}")
            return []
    
    def export_capsule(self, capsule_id: str) -> Optional[Dict[str, Any]]:
        """
        Export a capsule as JSON.
        
        Args:
            capsule_id: ID of the capsule to export
            
        Returns:
            Capsule data as dictionary
        """
        try:
            capsule = self.capsules.get(capsule_id)
            if capsule:
                return asdict(capsule)
            return None
            
        except Exception as e:
            logger.error(f"Error exporting capsule {capsule_id}: {e}")
            return None
    
    def import_capsule(self, capsule_data: Dict[str, Any], user_id: str = "anonymous") -> Optional[str]:
        """
        Import a capsule from JSON data.
        
        Args:
            capsule_data: Capsule data dictionary
            user_id: User importing the capsule
            
        Returns:
            New capsule ID if successful, None otherwise
        """
        try:
            # Generate new ID to avoid conflicts
            new_capsule_id = f"capsule_{uuid.uuid4().hex[:12]}"
            
            # Reconstruct content
            content_data = capsule_data['content']
            content = CapsuleContent(
                reasoning_trace=content_data['reasoning_trace'],
                tools_used=content_data['tools_used'],
                tool_outputs=content_data['tool_outputs'],
                documents_referenced=content_data['documents_referenced'],
                key_insights=content_data['key_insights'],
                methodology=content_data['methodology'],
                success_factors=content_data['success_factors'],
                limitations=content_data['limitations']
            )
            
            # Create capsule with new ID
            capsule = KnowledgeCapsule(
                capsule_id=new_capsule_id,
                name=capsule_data['name'],
                summary=capsule_data['summary'],
                content=content,
                created_at=datetime.now().isoformat(),  # New creation time
                created_by=user_id,  # New creator
                last_used=datetime.now().isoformat(),
                use_count=0,  # Reset usage
                tags=capsule_data['tags'],
                category=capsule_data['category'],
                effectiveness_score=capsule_data['effectiveness_score'],
                metadata=capsule_data.get('metadata', {})
            )
            
            # Store capsule
            self.capsules[new_capsule_id] = capsule
            self._save_capsule(capsule)
            
            logger.info(f"Imported knowledge capsule: {capsule.name} ({new_capsule_id})")
            return new_capsule_id
            
        except Exception as e:
            logger.error(f"Error importing capsule: {e}")
            return None
    
    def delete_capsule(self, capsule_id: str) -> bool:
        """
        Delete a knowledge capsule.
        
        Args:
            capsule_id: ID of the capsule to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if capsule_id in self.capsules:
                capsule = self.capsules[capsule_id]
                
                # Remove from memory
                del self.capsules[capsule_id]
                
                # Remove file
                capsule_file = self.capsules_dir / f"{capsule_id}.json"
                if capsule_file.exists():
                    capsule_file.unlink()
                
                logger.info(f"Deleted knowledge capsule: {capsule.name} ({capsule_id})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting capsule {capsule_id}: {e}")
            return False
    
    def get_capsule_stats(self) -> Dict[str, Any]:
        """Get statistics about the capsule collection."""
        try:
            stats = {
                'total_capsules': len(self.capsules),
                'categories': {},
                'tags': {},
                'creators': {},
                'average_effectiveness': 0.0,
                'total_uses': 0,
                'most_used': None,
                'most_effective': None
            }
            
            if not self.capsules:
                return stats
            
            effectiveness_sum = 0
            most_used_capsule = None
            most_effective_capsule = None
            max_uses = 0
            max_effectiveness = 0
            
            for capsule in self.capsules.values():
                # Categories
                category = capsule.category
                stats['categories'][category] = stats['categories'].get(category, 0) + 1
                
                # Tags
                for tag in capsule.tags:
                    stats['tags'][tag] = stats['tags'].get(tag, 0) + 1
                
                # Creators
                creator = capsule.created_by
                stats['creators'][creator] = stats['creators'].get(creator, 0) + 1
                
                # Effectiveness
                effectiveness_sum += capsule.effectiveness_score
                if capsule.effectiveness_score > max_effectiveness:
                    max_effectiveness = capsule.effectiveness_score
                    most_effective_capsule = capsule.name
                
                # Usage
                stats['total_uses'] += capsule.use_count
                if capsule.use_count > max_uses:
                    max_uses = capsule.use_count
                    most_used_capsule = capsule.name
            
            stats['average_effectiveness'] = effectiveness_sum / len(self.capsules)
            stats['most_used'] = most_used_capsule
            stats['most_effective'] = most_effective_capsule
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting capsule stats: {e}")
            return {}
    
    def list_capsules(self, user_id: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get a list of capsules with basic info."""
        try:
            capsule_list = []
            
            for capsule in self.capsules.values():
                # Apply filters
                if user_id and capsule.created_by != user_id:
                    continue
                
                if category and capsule.category != category:
                    continue
                
                capsule_info = {
                    'capsule_id': capsule.capsule_id,
                    'name': capsule.name,
                    'summary': capsule.summary,
                    'category': capsule.category,
                    'tags': capsule.tags,
                    'created_at': capsule.created_at,
                    'created_by': capsule.created_by,
                    'use_count': capsule.use_count,
                    'effectiveness_score': capsule.effectiveness_score,
                    'tools_used': capsule.content.tools_used
                }
                
                capsule_list.append(capsule_info)
            
            # Sort by effectiveness and usage
            capsule_list.sort(key=lambda x: (x['effectiveness_score'], x['use_count']), reverse=True)
            
            return capsule_list
            
        except Exception as e:
            logger.error(f"Error listing capsules: {e}")
            return []
    
    def _extract_reasoning_trace(self, session: SelfDecideSession) -> str:
        """Extract a readable reasoning trace from the session."""
        trace_parts = [f"SELF-DECIDE Reasoning for: {session.original_query}"]
        
        for step in session.reasoning_steps:
            trace_parts.append(f"\n{step.step.value}: {step.reasoning}")
        
        trace_parts.append(f"\nFinal Answer: {session.final_answer}")
        trace_parts.append(f"Confidence: {session.confidence_score:.2f}")
        
        return "\n".join(trace_parts)
    
    def _extract_key_insights(self, session: SelfDecideSession, 
                            tool_responses: List[ToolResponse] = None) -> List[str]:
        """Extract key insights from the reasoning session."""
        insights = []
        
        # Extract from reasoning steps
        for step in session.reasoning_steps:
            if step.output_data and isinstance(step.output_data, dict):
                insight = step.output_data.get('insight')
                if insight:
                    insights.append(insight)
        
        # Extract from tool outputs
        if tool_responses:
            for tr in tool_responses:
                if tr.success and isinstance(tr.output, dict):
                    insight = tr.output.get('insight')
                    if insight:
                        insights.append(f"Tool insight: {insight}")
        
        # Add general insights
        if session.confidence_score > 0.8:
            insights.append("High confidence achieved through systematic reasoning")
        
        if len(session.tool_executions) > 2:
            insights.append("Multi-tool approach provided comprehensive analysis")
        
        return insights
    
    def _determine_methodology(self, session: SelfDecideSession, tools_used: List[str]) -> str:
        """Determine the methodology used in the reasoning."""
        if not tools_used:
            return "Pure reasoning without tools"
        
        if len(tools_used) == 1:
            return f"Single-tool approach using {tools_used[0]}"
        
        if 'python_interpreter' in tools_used and 'table_generator' in tools_used:
            return "Computational analysis with structured presentation"
        
        if 'multimodal_query' in tools_used:
            return "Knowledge-augmented reasoning with multimodal search"
        
        return f"Multi-tool approach using {', '.join(tools_used)}"
    
    def _identify_success_factors(self, session: SelfDecideSession, 
                                tool_responses: List[ToolResponse] = None) -> List[str]:
        """Identify factors that contributed to success."""
        factors = []
        
        if session.confidence_score > 0.7:
            factors.append("High confidence in final answer")
        
        if len(session.reasoning_steps) >= 8:
            factors.append("Complete SELF-DECIDE reasoning process")
        
        if tool_responses:
            success_rate = sum(1 for tr in tool_responses if tr.success) / len(tool_responses)
            if success_rate > 0.8:
                factors.append("High tool execution success rate")
        
        if session.total_duration_ms < 5000:
            factors.append("Efficient processing time")
        
        return factors
    
    def _identify_limitations(self, session: SelfDecideSession, 
                           tool_responses: List[ToolResponse] = None) -> List[str]:
        """Identify limitations or areas for improvement."""
        limitations = []
        
        if session.confidence_score < 0.5:
            limitations.append("Low confidence in final answer")
        
        if len(session.knowledge_gaps) > 3:
            limitations.append("Multiple knowledge gaps identified")
        
        if tool_responses:
            failed_tools = [tr.tool_name for tr in tool_responses if not tr.success]
            if failed_tools:
                limitations.append(f"Tool failures: {', '.join(failed_tools)}")
        
        if session.total_duration_ms > 30000:
            limitations.append("Long processing time")
        
        return limitations
    
    def _load_capsules(self):
        """Load all capsules from the capsules directory."""
        try:
            for capsule_file in self.capsules_dir.glob("*.json"):
                try:
                    with open(capsule_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Reconstruct content
                    content_data = data['content']
                    content = CapsuleContent(
                        reasoning_trace=content_data['reasoning_trace'],
                        tools_used=content_data['tools_used'],
                        tool_outputs=content_data['tool_outputs'],
                        documents_referenced=content_data['documents_referenced'],
                        key_insights=content_data['key_insights'],
                        methodology=content_data['methodology'],
                        success_factors=content_data['success_factors'],
                        limitations=content_data['limitations']
                    )
                    
                    # Reconstruct capsule
                    capsule = KnowledgeCapsule(
                        capsule_id=data['capsule_id'],
                        name=data['name'],
                        summary=data['summary'],
                        content=content,
                        created_at=data['created_at'],
                        created_by=data['created_by'],
                        last_used=data['last_used'],
                        use_count=data['use_count'],
                        tags=data['tags'],
                        category=data['category'],
                        effectiveness_score=data['effectiveness_score'],
                        metadata=data['metadata']
                    )
                    
                    self.capsules[capsule.capsule_id] = capsule
                    
                except Exception as e:
                    logger.warning(f"Failed to load capsule from {capsule_file}: {e}")
            
            logger.info(f"Loaded {len(self.capsules)} knowledge capsules")
            
        except Exception as e:
            logger.error(f"Error loading capsules: {e}")
    
    def _save_capsule(self, capsule: KnowledgeCapsule):
        """Save a capsule to disk."""
        try:
            capsule_file = self.capsules_dir / f"{capsule.capsule_id}.json"
            
            with open(capsule_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(capsule), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved capsule: {capsule.name}")
            
        except Exception as e:
            logger.error(f"Error saving capsule {capsule.capsule_id}: {e}")

# Global capsule manager instance
_capsule_manager = None

def get_capsule_manager(capsules_directory: str = "capsules") -> KnowledgeCapsuleManager:
    """Get or create a global capsule manager instance."""
    global _capsule_manager
    
    if _capsule_manager is None:
        _capsule_manager = KnowledgeCapsuleManager(capsules_directory=capsules_directory)
    
    return _capsule_manager
