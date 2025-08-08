"""
Personalized User Profiles for SAM
Adapts SAM behavior to individual users and preferences.

Sprint 6 Task 2: Personalized User Profiles
"""

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ConversationStyle(Enum):
    """Available conversation style presets."""
    TECHNICAL = "technical"
    CASUAL = "casual"
    EXPLORATORY = "exploratory"
    EXECUTIVE = "executive"

class VerbosityLevel(Enum):
    """Verbosity level options."""
    CONCISE = "concise"
    BALANCED = "balanced"
    DETAILED = "detailed"

class ExplanationLevel(Enum):
    """Explanation depth levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

@dataclass
class UserPreferences:
    """User preference settings."""
    verbosity: VerbosityLevel
    explanation_level: ExplanationLevel
    conversation_style: ConversationStyle
    tools_enabled: List[str]
    tools_disabled: List[str]
    formatting_preferences: Dict[str, Any]
    memory_persistence: bool
    show_reasoning: bool
    show_sources: bool
    auto_pin_important: bool
    preferred_response_length: int  # Target response length in words

@dataclass
class UserProfile:
    """Complete user profile with preferences and history."""
    user_id: str
    username: str
    preferences: UserPreferences
    created_at: str
    last_active: str
    session_count: int
    total_interactions: int
    favorite_tools: List[str]
    frequent_topics: List[str]
    custom_commands: Dict[str, str]
    privacy_settings: Dict[str, bool]
    learning_history: List[Dict[str, Any]]

class UserProfileManager:
    """
    Manages user profiles and personalization settings.
    """
    
    def __init__(self, profiles_path: str = "user_profiles.json"):
        """
        Initialize the user profile manager.
        
        Args:
            profiles_path: Path to user profiles storage file
        """
        self.profiles_path = Path(profiles_path)
        self.profiles: Dict[str, UserProfile] = {}
        self.current_user_id: Optional[str] = None
        
        # Default preferences
        self.default_preferences = UserPreferences(
            verbosity=VerbosityLevel.BALANCED,
            explanation_level=ExplanationLevel.INTERMEDIATE,
            conversation_style=ConversationStyle.EXPLORATORY,
            tools_enabled=["python_interpreter", "table_generator", "multimodal_query", "table_to_code_expert"],
            tools_disabled=[],
            formatting_preferences={
                "use_markdown": True,
                "show_code_blocks": True,
                "include_emojis": True,
                "table_format": "markdown"
            },
            memory_persistence=True,
            show_reasoning=False,
            show_sources=True,
            auto_pin_important=True,
            preferred_response_length=200
        )
        
        # Load existing profiles
        self._load_profiles()
        
        logger.info(f"User profile manager initialized with {len(self.profiles)} profiles")
    
    def create_user_profile(self, username: str, preferences: Optional[UserPreferences] = None) -> str:
        """
        Create a new user profile.
        
        Args:
            username: Username for the profile
            preferences: Custom preferences (uses defaults if None)
            
        Returns:
            User ID of the created profile
        """
        try:
            user_id = f"user_{uuid.uuid4().hex[:12]}"
            
            profile = UserProfile(
                user_id=user_id,
                username=username,
                preferences=preferences or self.default_preferences,
                created_at=datetime.now().isoformat(),
                last_active=datetime.now().isoformat(),
                session_count=0,
                total_interactions=0,
                favorite_tools=[],
                frequent_topics=[],
                custom_commands={},
                privacy_settings={
                    "store_conversations": True,
                    "share_usage_stats": False,
                    "allow_learning": True
                },
                learning_history=[]
            )
            
            self.profiles[user_id] = profile
            self._save_profiles()
            
            logger.info(f"Created user profile: {username} ({user_id})")
            return user_id
            
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            raise
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile by ID.
        
        Args:
            user_id: User ID to retrieve
            
        Returns:
            UserProfile if found, None otherwise
        """
        return self.profiles.get(user_id)
    
    def set_current_user(self, user_id: str) -> bool:
        """
        Set the current active user.
        
        Args:
            user_id: User ID to set as current
            
        Returns:
            True if successful, False if user not found
        """
        if user_id in self.profiles:
            self.current_user_id = user_id
            
            # Update last active time
            profile = self.profiles[user_id]
            profile.last_active = datetime.now().isoformat()
            profile.session_count += 1
            
            self._save_profiles()
            logger.info(f"Set current user: {profile.username} ({user_id})")
            return True
        
        return False
    
    def get_current_user(self) -> Optional[UserProfile]:
        """Get the current active user profile."""
        if self.current_user_id:
            return self.profiles.get(self.current_user_id)
        return None
    
    def update_preferences(self, user_id: str, **kwargs) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id: User ID to update
            **kwargs: Preference fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            profile = self.profiles.get(user_id)
            if not profile:
                return False
            
            # Update preference fields
            for key, value in kwargs.items():
                if hasattr(profile.preferences, key):
                    # Handle enum conversions
                    if key == 'verbosity' and isinstance(value, str):
                        value = VerbosityLevel(value)
                    elif key == 'explanation_level' and isinstance(value, str):
                        value = ExplanationLevel(value)
                    elif key == 'conversation_style' and isinstance(value, str):
                        value = ConversationStyle(value)
                    
                    setattr(profile.preferences, key, value)
                    logger.debug(f"Updated {key} for user {user_id}: {value}")
            
            self._save_profiles()
            return True
            
        except Exception as e:
            logger.error(f"Error updating preferences for user {user_id}: {e}")
            return False
    
    def set_conversation_style(self, user_id: str, style: ConversationStyle) -> bool:
        """
        Set conversation style for a user.
        
        Args:
            user_id: User ID
            style: ConversationStyle to set
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_preferences(user_id, conversation_style=style)
    
    def toggle_tool(self, user_id: str, tool_name: str, enabled: bool) -> bool:
        """
        Enable or disable a tool for a user.
        
        Args:
            user_id: User ID
            tool_name: Name of the tool
            enabled: True to enable, False to disable
            
        Returns:
            True if successful, False otherwise
        """
        try:
            profile = self.profiles.get(user_id)
            if not profile:
                return False
            
            if enabled:
                # Enable tool
                if tool_name not in profile.preferences.tools_enabled:
                    profile.preferences.tools_enabled.append(tool_name)
                if tool_name in profile.preferences.tools_disabled:
                    profile.preferences.tools_disabled.remove(tool_name)
            else:
                # Disable tool
                if tool_name not in profile.preferences.tools_disabled:
                    profile.preferences.tools_disabled.append(tool_name)
                if tool_name in profile.preferences.tools_enabled:
                    profile.preferences.tools_enabled.remove(tool_name)
            
            self._save_profiles()
            logger.info(f"Tool {tool_name} {'enabled' if enabled else 'disabled'} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error toggling tool {tool_name} for user {user_id}: {e}")
            return False
    
    def is_tool_enabled(self, user_id: str, tool_name: str) -> bool:
        """
        Check if a tool is enabled for a user.
        
        Args:
            user_id: User ID
            tool_name: Name of the tool
            
        Returns:
            True if enabled, False otherwise
        """
        profile = self.profiles.get(user_id)
        if not profile:
            return True  # Default to enabled if no profile
        
        # Check explicit disable list first
        if tool_name in profile.preferences.tools_disabled:
            return False
        
        # Check explicit enable list
        if tool_name in profile.preferences.tools_enabled:
            return True
        
        # Default behavior based on tool type
        default_enabled_tools = ["python_interpreter", "table_generator", "multimodal_query", "table_to_code_expert"]
        return tool_name in default_enabled_tools
    
    def add_custom_command(self, user_id: str, command: str, response: str) -> bool:
        """
        Add a custom command for a user.
        
        Args:
            user_id: User ID
            command: Command name (without /)
            response: Response template
            
        Returns:
            True if successful, False otherwise
        """
        try:
            profile = self.profiles.get(user_id)
            if not profile:
                return False
            
            profile.custom_commands[command] = response
            self._save_profiles()
            
            logger.info(f"Added custom command '{command}' for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom command for user {user_id}: {e}")
            return False
    
    def get_custom_command(self, user_id: str, command: str) -> Optional[str]:
        """
        Get a custom command response for a user.
        
        Args:
            user_id: User ID
            command: Command name
            
        Returns:
            Command response if found, None otherwise
        """
        profile = self.profiles.get(user_id)
        if profile:
            return profile.custom_commands.get(command)
        return None
    
    def record_interaction(self, user_id: str, tool_used: Optional[str] = None, 
                          topic: Optional[str] = None):
        """
        Record a user interaction for learning purposes.
        
        Args:
            user_id: User ID
            tool_used: Tool that was used (if any)
            topic: Topic of the interaction (if identified)
        """
        try:
            profile = self.profiles.get(user_id)
            if not profile:
                return
            
            profile.total_interactions += 1
            
            # Track favorite tools
            if tool_used:
                if tool_used not in profile.favorite_tools:
                    profile.favorite_tools.append(tool_used)
                else:
                    # Move to end (most recent)
                    profile.favorite_tools.remove(tool_used)
                    profile.favorite_tools.append(tool_used)
                
                # Keep only top 10 favorite tools
                profile.favorite_tools = profile.favorite_tools[-10:]
            
            # Track frequent topics
            if topic:
                if topic not in profile.frequent_topics:
                    profile.frequent_topics.append(topic)
                else:
                    # Move to end (most recent)
                    profile.frequent_topics.remove(topic)
                    profile.frequent_topics.append(topic)
                
                # Keep only top 20 frequent topics
                profile.frequent_topics = profile.frequent_topics[-20:]
            
            # Add to learning history
            interaction_record = {
                'timestamp': datetime.now().isoformat(),
                'tool_used': tool_used,
                'topic': topic
            }
            
            profile.learning_history.append(interaction_record)
            
            # Keep only last 100 interactions in history
            profile.learning_history = profile.learning_history[-100:]
            
            self._save_profiles()
            
        except Exception as e:
            logger.error(f"Error recording interaction for user {user_id}: {e}")
    
    def get_personalized_system_prompt(self, user_id: str, base_prompt: str) -> str:
        """
        Generate a personalized system prompt based on user preferences.
        
        Args:
            user_id: User ID
            base_prompt: Base system prompt
            
        Returns:
            Personalized system prompt
        """
        try:
            profile = self.profiles.get(user_id)
            if not profile:
                return base_prompt
            
            prefs = profile.preferences
            
            # Build personalization additions
            personalization_parts = [
                base_prompt,
                "\n## Personalization Settings:"
            ]
            
            # Conversation style
            style_descriptions = {
                ConversationStyle.TECHNICAL: "Use precise technical language, include implementation details, and focus on accuracy.",
                ConversationStyle.CASUAL: "Use friendly, conversational tone with analogies and examples.",
                ConversationStyle.EXPLORATORY: "Encourage curiosity, ask follow-up questions, and explore connections.",
                ConversationStyle.EXECUTIVE: "Be concise, focus on key insights, and provide actionable summaries."
            }
            
            personalization_parts.append(f"- **Conversation Style**: {style_descriptions.get(prefs.conversation_style, 'Balanced approach')}")
            
            # Verbosity
            verbosity_descriptions = {
                VerbosityLevel.CONCISE: "Keep responses brief and to the point.",
                VerbosityLevel.BALANCED: "Provide balanced detail level.",
                VerbosityLevel.DETAILED: "Provide comprehensive, detailed explanations."
            }
            
            personalization_parts.append(f"- **Verbosity**: {verbosity_descriptions.get(prefs.verbosity, 'Balanced')}")
            
            # Explanation level
            explanation_descriptions = {
                ExplanationLevel.BASIC: "Use simple explanations suitable for beginners.",
                ExplanationLevel.INTERMEDIATE: "Assume moderate background knowledge.",
                ExplanationLevel.ADVANCED: "Use technical depth appropriate for experts."
            }
            
            personalization_parts.append(f"- **Explanation Level**: {explanation_descriptions.get(prefs.explanation_level, 'Intermediate')}")
            
            # Response length preference
            personalization_parts.append(f"- **Target Response Length**: Aim for approximately {prefs.preferred_response_length} words")
            
            # Tool preferences
            if prefs.tools_disabled:
                personalization_parts.append(f"- **Disabled Tools**: Avoid using {', '.join(prefs.tools_disabled)}")
            
            # Display preferences
            display_prefs = []
            if prefs.show_reasoning:
                display_prefs.append("show reasoning traces")
            if prefs.show_sources:
                display_prefs.append("include source attributions")
            if not prefs.formatting_preferences.get("include_emojis", True):
                display_prefs.append("avoid emojis")
            
            if display_prefs:
                personalization_parts.append(f"- **Display Preferences**: {', '.join(display_prefs)}")
            
            # User context
            if profile.favorite_tools:
                personalization_parts.append(f"- **User's Favorite Tools**: {', '.join(profile.favorite_tools[-3:])}")
            
            if profile.frequent_topics:
                personalization_parts.append(f"- **User's Frequent Topics**: {', '.join(profile.frequent_topics[-3:])}")
            
            return "\n".join(personalization_parts)
            
        except Exception as e:
            logger.error(f"Error generating personalized prompt for user {user_id}: {e}")
            return base_prompt
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user."""
        try:
            profile = self.profiles.get(user_id)
            if not profile:
                return {}
            
            return {
                'username': profile.username,
                'created_at': profile.created_at,
                'last_active': profile.last_active,
                'session_count': profile.session_count,
                'total_interactions': profile.total_interactions,
                'favorite_tools': profile.favorite_tools,
                'frequent_topics': profile.frequent_topics,
                'conversation_style': profile.preferences.conversation_style.value,
                'verbosity': profile.preferences.verbosity.value,
                'explanation_level': profile.preferences.explanation_level.value,
                'tools_enabled_count': len(profile.preferences.tools_enabled),
                'tools_disabled_count': len(profile.preferences.tools_disabled),
                'custom_commands_count': len(profile.custom_commands)
            }
            
        except Exception as e:
            logger.error(f"Error getting user stats for {user_id}: {e}")
            return {}
    
    def list_users(self) -> List[Dict[str, str]]:
        """Get a list of all users."""
        return [
            {
                'user_id': profile.user_id,
                'username': profile.username,
                'last_active': profile.last_active,
                'session_count': str(profile.session_count)
            }
            for profile in self.profiles.values()
        ]
    
    def _load_profiles(self):
        """Load user profiles from persistent storage."""
        try:
            if self.profiles_path.exists():
                with open(self.profiles_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for profile_data in data.get('profiles', []):
                    # Reconstruct preferences
                    prefs_data = profile_data['preferences']
                    preferences = UserPreferences(
                        verbosity=VerbosityLevel(prefs_data.get('verbosity', 'balanced')),
                        explanation_level=ExplanationLevel(prefs_data.get('explanation_level', 'intermediate')),
                        conversation_style=ConversationStyle(prefs_data.get('conversation_style', 'exploratory')),
                        tools_enabled=prefs_data.get('tools_enabled', []),
                        tools_disabled=prefs_data.get('tools_disabled', []),
                        formatting_preferences=prefs_data.get('formatting_preferences', {}),
                        memory_persistence=prefs_data.get('memory_persistence', True),
                        show_reasoning=prefs_data.get('show_reasoning', False),
                        show_sources=prefs_data.get('show_sources', True),
                        auto_pin_important=prefs_data.get('auto_pin_important', True),
                        preferred_response_length=prefs_data.get('preferred_response_length', 200)
                    )
                    
                    # Reconstruct profile
                    profile = UserProfile(
                        user_id=profile_data['user_id'],
                        username=profile_data['username'],
                        preferences=preferences,
                        created_at=profile_data['created_at'],
                        last_active=profile_data['last_active'],
                        session_count=profile_data.get('session_count', 0),
                        total_interactions=profile_data.get('total_interactions', 0),
                        favorite_tools=profile_data.get('favorite_tools', []),
                        frequent_topics=profile_data.get('frequent_topics', []),
                        custom_commands=profile_data.get('custom_commands', {}),
                        privacy_settings=profile_data.get('privacy_settings', {}),
                        learning_history=profile_data.get('learning_history', [])
                    )
                    
                    self.profiles[profile.user_id] = profile
                
                logger.info(f"Loaded {len(self.profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Error loading user profiles: {e}")
    
    def _save_profiles(self):
        """Save user profiles to persistent storage."""
        try:
            profiles_data = []
            
            for profile in self.profiles.values():
                profile_dict = asdict(profile)
                profiles_data.append(profile_dict)
            
            data = {
                'profiles': profiles_data,
                'last_updated': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            self.profiles_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.profiles_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Error saving user profiles: {e}")

# Global profile manager instance
_profile_manager = None

def get_profile_manager(profiles_path: str = "user_profiles.json") -> UserProfileManager:
    """Get or create a global profile manager instance."""
    global _profile_manager
    
    if _profile_manager is None:
        _profile_manager = UserProfileManager(profiles_path=profiles_path)
    
    return _profile_manager
