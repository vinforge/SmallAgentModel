"""
Integrated Memory System for SAM
Combines all Sprint 6 memory components into a unified system.

Sprint 6: Active Memory & Personalized Learning Integration
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .memory_manager import LongTermMemoryManager, get_memory_manager
from .user_profiles import UserProfileManager, get_profile_manager, ConversationStyle, VerbosityLevel
from .self_awareness import SelfAwarenessManager, get_self_awareness_manager
from .retrospective_learning import RetrospectiveLearningManager, get_learning_manager
from .knowledge_capsules import KnowledgeCapsuleManager, get_capsule_manager

logger = logging.getLogger(__name__)

@dataclass
class MemoryCommand:
    """Represents a memory-related command."""
    command_type: str
    parameters: Dict[str, Any]
    user_id: Optional[str]
    response: str

class IntegratedMemorySystem:
    """
    Unified memory system that integrates all Sprint 6 memory capabilities.
    """
    
    def __init__(self, embedding_manager=None):
        """
        Initialize the integrated memory system.
        
        Args:
            embedding_manager: Embedding manager for vectorization
        """
        # Initialize all memory components
        self.memory_manager = get_memory_manager(embedding_manager=embedding_manager)
        self.profile_manager = get_profile_manager()
        self.self_awareness = get_self_awareness_manager(self.memory_manager, self.profile_manager)
        self.learning_manager = get_learning_manager(self.memory_manager)
        self.capsule_manager = get_capsule_manager()
        
        # Current session state
        self.current_user_id: Optional[str] = None
        self.current_session_id: Optional[str] = None
        
        logger.info("Integrated memory system initialized")
    
    def process_memory_command(self, command: str, user_id: Optional[str] = None) -> str:
        """
        Process memory-related commands.
        
        Args:
            command: Memory command string
            user_id: User ID for context
            
        Returns:
            Response to the command
        """
        try:
            command = command.strip().lower()
            
            # Pin command: /pin <content>
            if command.startswith('/pin '):
                content = command[5:].strip()
                return self._handle_pin_command(content, user_id)
            
            # Recall command: /recall <topic>
            elif command.startswith('/recall '):
                topic = command[8:].strip()
                return self._handle_recall_command(topic, user_id)
            
            # Memory search: /memory <topic> or /memory recent
            elif command.startswith('/memory '):
                query = command[8:].strip()
                return self._handle_memory_command(query, user_id)
            
            # History command: /history <n>
            elif command.startswith('/history '):
                try:
                    n = int(command[9:].strip())
                    return self._handle_history_command(n, user_id)
                except ValueError:
                    return "❓ Use '/history <number>' (e.g., '/history 5')"
            
            # Style command: /style <preset>
            elif command.startswith('/style '):
                style = command[7:].strip()
                return self._handle_style_command(style, user_id)
            
            # Tools command: /tools set <tool> on|off
            elif command.startswith('/tools set '):
                parts = command[11:].split()
                if len(parts) >= 2:
                    tool_name = parts[0]
                    setting = parts[1]
                    return self._handle_tools_command(tool_name, setting, user_id)
                else:
                    return "❓ Use '/tools set <tool> on|off'"
            
            # Capsule commands
            elif command.startswith('/capsule '):
                return self._handle_capsule_command(command[9:], user_id)
            
            # User management commands
            elif command.startswith('/user '):
                return self._handle_user_command(command[6:], user_id)
            
            else:
                return self._get_memory_help()
            
        except Exception as e:
            logger.error(f"Error processing memory command: {e}")
            return f"❌ Error processing command: {str(e)}"
    
    def start_session(self, user_id: Optional[str] = None) -> str:
        """Start a new memory-aware session."""
        self.current_user_id = user_id
        self.current_session_id = self.self_awareness.start_session(user_id)
        
        # Get user profile for personalization
        if user_id:
            profile = self.profile_manager.get_user_profile(user_id)
            if profile:
                return f"🧠 Session started for {profile.username}. Memory persistence: {'ON' if profile.preferences.memory_persistence else 'OFF'}"
        
        return f"🧠 Session started: {self.current_session_id}"
    
    def end_session(self) -> Optional[str]:
        """End the current session and create summary."""
        if self.current_session_id:
            summary = self.self_awareness.end_session()
            if summary:
                return f"📊 Session ended: {summary.interaction_count} interactions, {len(summary.tools_used)} tools used"
        
        self.current_user_id = None
        self.current_session_id = None
        return None
    
    def record_interaction(self, query: str, response: str, tools_used: List[str] = None,
                          topic: Optional[str] = None, insight: Optional[str] = None):
        """Record an interaction in the current session."""
        # Record in self-awareness
        self.self_awareness.record_interaction(query, response, tools_used, topic, insight)
        
        # Record in user profile
        if self.current_user_id:
            for tool in (tools_used or []):
                self.profile_manager.record_interaction(self.current_user_id, tool, topic)
    
    def get_personalized_prompt(self, base_prompt: str) -> str:
        """Get personalized system prompt for current user."""
        if self.current_user_id:
            return self.profile_manager.get_personalized_system_prompt(
                self.current_user_id, base_prompt
            )
        return base_prompt
    
    def should_store_in_memory(self, content: str, importance_score: float = 0.5) -> bool:
        """Determine if content should be stored in long-term memory."""
        # Check user preferences
        if self.current_user_id:
            profile = self.profile_manager.get_user_profile(self.current_user_id)
            if profile and not profile.preferences.memory_persistence:
                return False
        
        # Store if importance is above threshold
        return importance_score > 0.3
    
    def auto_store_interaction(self, query: str, response: str, tools_used: List[str] = None,
                              importance_score: float = 0.5) -> Optional[str]:
        """Automatically store important interactions in memory."""
        if not self.should_store_in_memory(response, importance_score):
            return None
        
        content = f"Q: {query}\nA: {response}"
        tags = ['interaction'] + (tools_used or [])
        
        if self.current_user_id:
            tags.append(f"user_{self.current_user_id}")
        
        memory_id = self.memory_manager.store_memory(
            content=content,
            content_type='conversation',
            tags=tags,
            importance_score=importance_score,
            metadata={
                'session_id': self.current_session_id,
                'user_id': self.current_user_id,
                'tools_used': tools_used
            }
        )
        
        return memory_id
    
    def _handle_pin_command(self, content: str, user_id: Optional[str]) -> str:
        """Handle /pin command."""
        try:
            memory_id = self.memory_manager.store_memory(
                content=content,
                content_type='user_note',
                tags=['pinned', 'user_flagged'],
                importance_score=1.0,  # Max importance for pinned content
                metadata={'user_id': user_id, 'pinned': True}
            )
            
            return f"📌 Content pinned to memory: {memory_id}"
            
        except Exception as e:
            return f"❌ Failed to pin content: {str(e)}"
    
    def _handle_recall_command(self, topic: str, user_id: Optional[str]) -> str:
        """Handle /recall command."""
        try:
            results = self.self_awareness.search_memory(topic, user_id=user_id, top_k=3)
            
            if not results:
                return f"🔍 No memories found for topic: {topic}"
            
            response_parts = [f"🧠 Recalled memories for '{topic}':"]
            
            for i, result in enumerate(results, 1):
                memory = result.memory_entry
                response_parts.append(
                    f"{i}. {memory.summary} (confidence: {result.similarity_score:.2f})"
                )
            
            return "\n".join(response_parts)
            
        except Exception as e:
            return f"❌ Failed to recall memories: {str(e)}"
    
    def _handle_memory_command(self, query: str, user_id: Optional[str]) -> str:
        """Handle /memory command."""
        try:
            if query == 'recent':
                memories = self.self_awareness.get_recent_memories(user_id=user_id)
                
                if not memories:
                    return "🔍 No recent memories found"
                
                response_parts = [f"📅 Recent memories ({len(memories)}):"]
                
                for memory in memories[:5]:
                    response_parts.append(f"• {memory.summary}")
                
                return "\n".join(response_parts)
            
            else:
                # Search memories
                results = self.self_awareness.search_memory(query, user_id=user_id)
                
                if not results:
                    return f"🔍 No memories found for: {query}"
                
                response_parts = [f"🔍 Memory search results for '{query}':"]
                
                for i, result in enumerate(results, 1):
                    memory = result.memory_entry
                    response_parts.append(
                        f"{i}. {memory.summary} (relevance: {result.similarity_score:.2f})"
                    )
                
                return "\n".join(response_parts)
            
        except Exception as e:
            return f"❌ Memory search failed: {str(e)}"
    
    def _handle_history_command(self, n: int, user_id: Optional[str]) -> str:
        """Handle /history command."""
        try:
            sessions = self.self_awareness.get_session_history(user_id=user_id, last_n_sessions=n)
            
            if not sessions:
                return "📊 No session history found"
            
            response_parts = [f"📊 Last {len(sessions)} sessions:"]
            
            for i, session in enumerate(sessions, 1):
                response_parts.append(
                    f"{i}. {session.date[:10]} - {session.interaction_count} interactions, "
                    f"{len(session.tools_used)} tools ({session.duration_minutes}min)"
                )
            
            return "\n".join(response_parts)
            
        except Exception as e:
            return f"❌ Failed to get history: {str(e)}"
    
    def _handle_style_command(self, style: str, user_id: Optional[str]) -> str:
        """Handle /style command."""
        try:
            if not user_id:
                return "❌ User profile required for style settings"
            
            style_map = {
                'technical': ConversationStyle.TECHNICAL,
                'casual': ConversationStyle.CASUAL,
                'exploratory': ConversationStyle.EXPLORATORY,
                'executive': ConversationStyle.EXECUTIVE
            }
            
            if style not in style_map:
                return f"❓ Available styles: {', '.join(style_map.keys())}"
            
            success = self.profile_manager.set_conversation_style(user_id, style_map[style])
            
            if success:
                return f"✅ Conversation style set to: {style}"
            else:
                return "❌ Failed to update conversation style"
            
        except Exception as e:
            return f"❌ Style command failed: {str(e)}"
    
    def _handle_tools_command(self, tool_name: str, setting: str, user_id: Optional[str]) -> str:
        """Handle /tools set command."""
        try:
            if not user_id:
                return "❌ User profile required for tool settings"
            
            if setting not in ['on', 'off']:
                return "❓ Use 'on' or 'off'"
            
            enabled = setting == 'on'
            success = self.profile_manager.toggle_tool(user_id, tool_name, enabled)
            
            if success:
                return f"✅ Tool {tool_name} {'enabled' if enabled else 'disabled'}"
            else:
                return f"❌ Failed to update tool {tool_name}"
            
        except Exception as e:
            return f"❌ Tools command failed: {str(e)}"
    
    def _handle_capsule_command(self, command: str, user_id: Optional[str]) -> str:
        """Handle /capsule commands."""
        try:
            parts = command.split()
            
            if not parts:
                return "❓ Use '/capsule save|load|list|export <name>'"
            
            action = parts[0]
            
            if action == 'list':
                capsules = self.capsule_manager.list_capsules(user_id=user_id)
                
                if not capsules:
                    return "📦 No knowledge capsules found"
                
                response_parts = [f"📦 Knowledge capsules ({len(capsules)}):"]
                
                for capsule in capsules[:10]:
                    response_parts.append(
                        f"• {capsule['name']} - {capsule['category']} "
                        f"(effectiveness: {capsule['effectiveness_score']:.2f})"
                    )
                
                return "\n".join(response_parts)
            
            elif action in ['save', 'load', 'export'] and len(parts) > 1:
                name = ' '.join(parts[1:])
                
                if action == 'save':
                    return "💡 Capsule saving requires active reasoning session"
                elif action == 'load':
                    capsules = self.capsule_manager.search_capsules(name, user_id=user_id)
                    if capsules:
                        capsule = self.capsule_manager.load_capsule(capsules[0].capsule_id)
                        if capsule:
                            return f"📦 Loaded capsule: {capsule.name}\n{capsule.summary}"
                    return f"❌ Capsule not found: {name}"
                elif action == 'export':
                    capsules = self.capsule_manager.search_capsules(name, user_id=user_id)
                    if capsules:
                        data = self.capsule_manager.export_capsule(capsules[0].capsule_id)
                        if data:
                            return f"📤 Capsule exported: {name} (JSON data available)"
                    return f"❌ Capsule not found: {name}"
            
            else:
                return "❓ Use '/capsule save|load|list|export <name>'"
            
        except Exception as e:
            return f"❌ Capsule command failed: {str(e)}"
    
    def _handle_user_command(self, command: str, user_id: Optional[str]) -> str:
        """Handle /user commands."""
        try:
            parts = command.split()
            
            if not parts:
                return "❓ Use '/user create|switch|stats <username>'"
            
            action = parts[0]
            
            if action == 'create' and len(parts) > 1:
                username = ' '.join(parts[1:])
                new_user_id = self.profile_manager.create_user_profile(username)
                return f"👤 Created user profile: {username} ({new_user_id})"
            
            elif action == 'switch' and len(parts) > 1:
                username = ' '.join(parts[1:])
                # Find user by username
                users = self.profile_manager.list_users()
                target_user = next((u for u in users if u['username'] == username), None)
                
                if target_user:
                    success = self.profile_manager.set_current_user(target_user['user_id'])
                    if success:
                        self.current_user_id = target_user['user_id']
                        return f"👤 Switched to user: {username}"
                
                return f"❌ User not found: {username}"
            
            elif action == 'stats':
                if user_id:
                    stats = self.profile_manager.get_user_stats(user_id)
                    if stats:
                        return (f"👤 User Stats:\n"
                               f"Username: {stats['username']}\n"
                               f"Sessions: {stats['session_count']}\n"
                               f"Interactions: {stats['total_interactions']}\n"
                               f"Style: {stats['conversation_style']}")
                
                return "❌ No user stats available"
            
            else:
                return "❓ Use '/user create|switch|stats <username>'"
            
        except Exception as e:
            return f"❌ User command failed: {str(e)}"
    
    def _get_memory_help(self) -> str:
        """Get help text for memory commands."""
        return """🧠 **SAM Memory Commands:**

**Memory Management:**
• `/pin <content>` - Pin important content to memory
• `/recall <topic>` - Recall memories about a topic
• `/memory <query>` - Search memories
• `/memory recent` - Show recent memories

**Session & History:**
• `/history <n>` - Show last n session summaries

**Personalization:**
• `/style <preset>` - Set conversation style (technical/casual/exploratory/executive)
• `/tools set <tool> on|off` - Enable/disable specific tools

**Knowledge Capsules:**
• `/capsule list` - List available capsules
• `/capsule load <name>` - Load a knowledge capsule
• `/capsule export <name>` - Export capsule as JSON

**User Management:**
• `/user create <username>` - Create new user profile
• `/user switch <username>` - Switch to different user
• `/user stats` - Show user statistics

Type any command for more specific help."""
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        try:
            return {
                'memory_stats': self.memory_manager.get_memory_stats(),
                'user_count': len(self.profile_manager.profiles),
                'current_user': self.current_user_id,
                'current_session': self.current_session_id,
                'capsule_stats': self.capsule_manager.get_capsule_stats(),
                'self_awareness_stats': self.self_awareness.get_self_awareness_stats()
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}

# Global integrated memory system instance
_integrated_memory = None

def get_integrated_memory(embedding_manager=None) -> IntegratedMemorySystem:
    """Get or create a global integrated memory system instance."""
    global _integrated_memory
    
    if _integrated_memory is None:
        _integrated_memory = IntegratedMemorySystem(embedding_manager=embedding_manager)
    
    return _integrated_memory
