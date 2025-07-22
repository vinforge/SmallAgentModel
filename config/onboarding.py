"""
User Onboarding System for SAM
Welcome interface, interactive tours, and first-time setup.

Sprint 13 Task 1: User Onboarding Flows
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class OnboardingState:
    """Track user onboarding progress."""
    first_launch: bool = True
    welcome_shown: bool = False
    chat_tour_completed: bool = False
    memory_ui_tour_completed: bool = False
    help_overlay_enabled: bool = True
    onboarding_version: str = "1.0"
    last_updated: str = ""
    user_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

class OnboardingManager:
    """
    Manages user onboarding flows and first-time setup.
    """
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the onboarding manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.onboarding_file = self.config_dir / "onboarding.json"
        self.state = self._load_onboarding_state()
        
        logger.info("Onboarding manager initialized")
    
    def _load_onboarding_state(self) -> OnboardingState:
        """Load onboarding state from file."""
        try:
            if self.onboarding_file.exists():
                with open(self.onboarding_file, 'r') as f:
                    data = json.load(f)
                    return OnboardingState(**data)
            else:
                # First time - create default state
                return OnboardingState()
                
        except Exception as e:
            logger.error(f"Error loading onboarding state: {e}")
            return OnboardingState()
    
    def _save_onboarding_state(self):
        """Save onboarding state to file."""
        try:
            self.state.last_updated = datetime.now().isoformat()
            
            with open(self.onboarding_file, 'w') as f:
                json.dump(asdict(self.state), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving onboarding state: {e}")
    
    def is_first_launch(self) -> bool:
        """Check if this is the user's first launch."""
        return self.state.first_launch
    
    def should_show_welcome(self) -> bool:
        """Check if welcome screen should be shown."""
        return self.state.first_launch or not self.state.welcome_shown
    
    def should_show_chat_tour(self) -> bool:
        """Check if chat tour should be offered."""
        return not self.state.chat_tour_completed
    
    def should_show_memory_ui_tour(self) -> bool:
        """Check if memory UI tour should be offered."""
        return not self.state.memory_ui_tour_completed
    
    def mark_welcome_shown(self):
        """Mark welcome screen as shown."""
        self.state.welcome_shown = True
        self.state.first_launch = False
        self._save_onboarding_state()
    
    def mark_chat_tour_completed(self):
        """Mark chat tour as completed."""
        self.state.chat_tour_completed = True
        self._save_onboarding_state()
    
    def mark_memory_ui_tour_completed(self):
        """Mark memory UI tour as completed."""
        self.state.memory_ui_tour_completed = True
        self._save_onboarding_state()
    
    def set_help_overlay_enabled(self, enabled: bool):
        """Enable or disable help overlays."""
        self.state.help_overlay_enabled = enabled
        self._save_onboarding_state()
    
    def get_welcome_message(self) -> Dict[str, Any]:
        """Get welcome message content."""
        return {
            "title": "Welcome to SAM!",
            "subtitle": "Small Agent Model - Your Intelligent Assistant",
            "features": [
                {
                    "icon": "🧠",
                    "title": "Memory-Driven Intelligence",
                    "description": "SAM remembers your conversations and learns from interactions"
                },
                {
                    "icon": "💬",
                    "title": "Interactive Chat",
                    "description": "Natural conversations with document processing and analysis"
                },
                {
                    "icon": "🎛️",
                    "title": "Memory Control Center",
                    "description": "Visual memory management with search, editing, and visualization"
                },
                {
                    "icon": "🤝",
                    "title": "Collaborative Modes",
                    "description": "Switch between solo analysis and collaborative swarm intelligence"
                },
                {
                    "icon": "🔍",
                    "title": "Advanced Search",
                    "description": "Semantic search across memories with role-based filtering"
                },
                {
                    "icon": "📊",
                    "title": "Analytics & Insights",
                    "description": "Memory statistics, performance metrics, and relationship visualization"
                }
            ],
            "quick_start": [
                "Start chatting to see SAM's intelligence in action",
                "Upload documents for analysis and processing",
                "Use memory commands like '!recall topic AI'",
                "Access the Memory Control Center for visual management",
                "Explore collaborative features with secure key sharing"
            ],
            "next_steps": {
                "chat_tour": "Take a tour of the chat interface",
                "memory_tour": "Explore the Memory Control Center",
                "documentation": "Read the complete documentation"
            }
        }
    
    def get_chat_tour_steps(self) -> List[Dict[str, Any]]:
        """Get chat interface tour steps."""
        return [
            {
                "step": 1,
                "title": "Welcome to SAM Chat",
                "content": "This is your main interface for talking with SAM. Type naturally and SAM will respond with intelligence and context.",
                "target": ".chat-container",
                "position": "center"
            },
            {
                "step": 2,
                "title": "Memory Control Center",
                "content": "Click this button to access advanced memory management, visualization, and analytics tools.",
                "target": ".memory-control-btn",
                "position": "bottom"
            },
            {
                "step": 3,
                "title": "Document Upload",
                "content": "Upload PDFs, DOCX, code files, and more for SAM to analyze and remember.",
                "target": ".upload-btn",
                "position": "bottom"
            },
            {
                "step": 4,
                "title": "Chat Input",
                "content": "Type your questions here. Try asking about AI, uploading a document, or using memory commands like '!recall topic AI'.",
                "target": ".chat-input",
                "position": "top"
            },
            {
                "step": 5,
                "title": "Memory Commands",
                "content": "Use commands starting with '!' to search and manage memories directly in chat. Type '!memhelp' to see all commands.",
                "target": ".chat-input",
                "position": "top"
            },
            {
                "step": 6,
                "title": "Source Transparency",
                "content": "SAM shows you where information comes from, including memory sources and document references.",
                "target": ".message-container",
                "position": "left"
            }
        ]
    
    def get_memory_ui_tour_steps(self) -> List[Dict[str, Any]]:
        """Get Memory UI tour steps."""
        return [
            {
                "step": 1,
                "title": "Memory Control Center",
                "content": "Welcome to SAM's Memory Control Center! This is where you can visually explore, manage, and analyze SAM's memory.",
                "target": "body",
                "position": "center"
            },
            {
                "step": 2,
                "title": "Navigation Menu",
                "content": "Use this sidebar to navigate between different memory management tools and features.",
                "target": ".sidebar",
                "position": "right"
            },
            {
                "step": 3,
                "title": "Memory Browser",
                "content": "Search, filter, and browse through all stored memories with advanced filtering options.",
                "target": "[data-testid='stSelectbox']",
                "position": "right"
            },
            {
                "step": 4,
                "title": "Memory Graph",
                "content": "Visualize connections between memories as an interactive network graph.",
                "target": ".memory-graph",
                "position": "center"
            },
            {
                "step": 5,
                "title": "Command Interface",
                "content": "Execute memory commands and see results in a structured format.",
                "target": ".command-interface",
                "position": "center"
            },
            {
                "step": 6,
                "title": "Role-Based Access",
                "content": "Configure memory access permissions for different agent roles in collaborative mode.",
                "target": ".role-access",
                "position": "center"
            }
        ]
    
    def get_help_tooltips(self) -> Dict[str, str]:
        """Get help tooltips for UI components."""
        return {
            "memory_search": "Search through memories using semantic similarity. Try queries like 'AI research' or 'user conversations'.",
            "memory_filter": "Filter memories by type, date, importance, tags, or user to narrow down results.",
            "memory_graph": "Interactive visualization showing connections between memories based on similarity and shared topics.",
            "memory_commands": "Execute memory operations using commands. Start with '!' followed by the command name.",
            "role_permissions": "Configure which agent roles can access specific types of memories in collaborative mode.",
            "memory_stats": "View statistics about memory usage, storage, and performance metrics.",
            "memory_export": "Export memories to JSON or backup files for sharing or archival purposes.",
            "collaboration_key": "Generate secure keys to enable collaborative mode and multi-agent access.",
            "agent_mode": "Switch between Solo Analyst mode and Collaborative Swarm mode for different use cases.",
            "memory_importance": "Importance scores help prioritize which memories are most relevant for recall and reasoning."
        }
    
    def reset_onboarding(self):
        """Reset onboarding state for testing or re-onboarding."""
        self.state = OnboardingState()
        self._save_onboarding_state()
        logger.info("Onboarding state reset")
    
    def get_onboarding_status(self) -> Dict[str, Any]:
        """Get current onboarding status."""
        return {
            "first_launch": self.state.first_launch,
            "welcome_shown": self.state.welcome_shown,
            "chat_tour_completed": self.state.chat_tour_completed,
            "memory_ui_tour_completed": self.state.memory_ui_tour_completed,
            "help_overlay_enabled": self.state.help_overlay_enabled,
            "onboarding_version": self.state.onboarding_version,
            "last_updated": self.state.last_updated,
            "completion_percentage": self._calculate_completion_percentage()
        }
    
    def _calculate_completion_percentage(self) -> float:
        """Calculate onboarding completion percentage."""
        total_steps = 4  # welcome, chat_tour, memory_ui_tour, help_overlay
        completed_steps = 0
        
        if self.state.welcome_shown:
            completed_steps += 1
        if self.state.chat_tour_completed:
            completed_steps += 1
        if self.state.memory_ui_tour_completed:
            completed_steps += 1
        if not self.state.help_overlay_enabled:  # User has chosen to disable
            completed_steps += 1
        
        return (completed_steps / total_steps) * 100
    
    def update_user_preference(self, key: str, value: Any):
        """Update user preference."""
        self.state.user_preferences[key] = value
        self._save_onboarding_state()
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference."""
        return self.state.user_preferences.get(key, default)

# Global onboarding manager instance
_onboarding_manager = None

def get_onboarding_manager() -> OnboardingManager:
    """Get or create a global onboarding manager instance."""
    global _onboarding_manager
    
    if _onboarding_manager is None:
        _onboarding_manager = OnboardingManager()
    
    return _onboarding_manager
