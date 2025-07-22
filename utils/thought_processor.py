#!/usr/bin/env python3
"""
Thought Processor - Sprint 16
Handles parsing, hiding, and toggling of SAM's <think> blocks for improved UX.
"""

import re
import json
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ThoughtBlock:
    """Represents a parsed thought block with metadata."""
    content: str
    start_pos: int
    end_pos: int
    token_count: int
    timestamp: str
    block_id: str

@dataclass
class ProcessedResponse:
    """Response with thoughts processed and separated."""
    visible_content: str
    thought_blocks: list[ThoughtBlock]
    has_thoughts: bool
    original_response: str

class ThoughtProcessor:
    """
    Processes SAM responses to extract and manage <think> blocks.
    
    Sprint 16 Features:
    - Parse <think> blocks from responses
    - Generate toggle-friendly output
    - Track thought visibility state
    - Support configuration options
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the thought processor."""
        self.config = config or self._load_default_config()
        self.show_thoughts = self.config.get('show_thoughts', True)
        self.thoughts_default_hidden = self.config.get('thoughts_default_hidden', True)
        self.enable_thought_toggle = self.config.get('enable_thought_toggle', True)
        
        # Regex pattern for matching <think> blocks
        self.think_pattern = re.compile(
            r'<think>(.*?)</think>',
            re.DOTALL | re.IGNORECASE
        )
        
        logger.info("Thought processor initialized")
        logger.info(f"Show thoughts: {self.show_thoughts}")
        logger.info(f"Default hidden: {self.thoughts_default_hidden}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        try:
            with open('config/sam_config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {
                'show_thoughts': True,
                'thoughts_default_hidden': True,
                'enable_thought_toggle': True
            }
    
    def process_response(self, response: str) -> ProcessedResponse:
        """
        Process a response to extract and handle thought blocks.
        
        Args:
            response: Raw response text that may contain <think> blocks
            
        Returns:
            ProcessedResponse with separated content and thought blocks
        """
        try:
            # Find all thought blocks
            thought_matches = list(self.think_pattern.finditer(response))
            
            if not thought_matches:
                # No thought blocks found
                return ProcessedResponse(
                    visible_content=response,
                    thought_blocks=[],
                    has_thoughts=False,
                    original_response=response
                )
            
            # Extract thought blocks
            thought_blocks = []
            visible_content = response
            
            # Process matches in reverse order to maintain string positions
            for i, match in enumerate(reversed(thought_matches)):
                thought_content = match.group(1).strip()
                start_pos = match.start()
                end_pos = match.end()
                
                # Create thought block
                thought_block = ThoughtBlock(
                    content=thought_content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    token_count=len(thought_content.split()),
                    timestamp=datetime.now().isoformat(),
                    block_id=f"think_{len(thought_matches)-i}_{int(datetime.now().timestamp())}"
                )
                
                thought_blocks.insert(0, thought_block)  # Maintain original order
                
                # Remove thought block from visible content
                visible_content = visible_content[:start_pos] + visible_content[end_pos:]
            
            # Clean up visible content
            visible_content = self._clean_visible_content(visible_content)
            
            return ProcessedResponse(
                visible_content=visible_content,
                thought_blocks=thought_blocks,
                has_thoughts=True,
                original_response=response
            )
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            # Return original response on error
            return ProcessedResponse(
                visible_content=response,
                thought_blocks=[],
                has_thoughts=False,
                original_response=response
            )
    
    def _clean_visible_content(self, content: str) -> str:
        """Clean up visible content after removing thought blocks."""
        try:
            # Remove extra whitespace and empty lines
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                stripped = line.strip()
                if stripped:  # Keep non-empty lines
                    cleaned_lines.append(line)
                elif cleaned_lines and cleaned_lines[-1].strip():  # Keep single empty lines between content
                    cleaned_lines.append('')
            
            # Remove trailing empty lines
            while cleaned_lines and not cleaned_lines[-1].strip():
                cleaned_lines.pop()
            
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            logger.debug(f"Error cleaning content: {e}")
            return content.strip()
    
    def generate_thought_toggle_html(self, thought_block: ThoughtBlock, 
                                   block_index: int = 0) -> str:
        """
        Generate HTML for a collapsible thought block.
        
        Args:
            thought_block: The thought block to render
            block_index: Index of this block in the response
            
        Returns:
            HTML string for the toggle interface
        """
        try:
            # Generate unique IDs
            toggle_id = f"thought_toggle_{thought_block.block_id}"
            content_id = f"thought_content_{thought_block.block_id}"
            
            # Format metadata
            metadata = f"Tokens: {thought_block.token_count} | Time: {thought_block.timestamp[:19]}"
            
            # Create HTML
            html = f"""
            <div class="thought-container" style="margin: 10px 0;">
                <button 
                    id="{toggle_id}"
                    class="thought-toggle-btn"
                    onclick="toggleThought('{content_id}', '{toggle_id}')"
                    style="
                        background: #f0f2f6;
                        border: 1px solid #d1d5db;
                        border-radius: 6px;
                        padding: 8px 12px;
                        cursor: pointer;
                        font-size: 14px;
                        color: #374151;
                        display: flex;
                        align-items: center;
                        gap: 6px;
                        transition: all 0.2s ease;
                    "
                    onmouseover="this.style.backgroundColor='#e5e7eb'"
                    onmouseout="this.style.backgroundColor='#f0f2f6'"
                >
                    <span id="{toggle_id}_arrow" style="font-size: 12px;">▶</span>
                    <span>🧠 SAM's Thoughts</span>
                    <span style="font-size: 12px; color: #6b7280;">({thought_block.token_count} tokens)</span>
                </button>
                
                <div 
                    id="{content_id}"
                    class="thought-content"
                    style="
                        display: none;
                        margin-top: 8px;
                        padding: 12px;
                        background: #f9fafb;
                        border: 1px solid #e5e7eb;
                        border-radius: 6px;
                        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                        font-size: 13px;
                        line-height: 1.5;
                        color: #374151;
                        white-space: pre-wrap;
                        overflow-x: auto;
                    "
                >
                    <div style="font-size: 11px; color: #6b7280; margin-bottom: 8px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px;">
                        {metadata}
                    </div>
                    {self._escape_html(thought_block.content)}
                </div>
            </div>
            """
            
            return html.strip()
            
        except Exception as e:
            logger.error(f"Error generating thought toggle HTML: {e}")
            return f'<div class="thought-error">Error displaying thought block</div>'
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters in thought content."""
        try:
            import html
            return html.escape(text)
        except Exception:
            # Fallback manual escaping
            return (text
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#x27;'))
    
    def generate_toggle_javascript(self) -> str:
        """Generate JavaScript for thought toggle functionality."""
        return """
        <script>
        function toggleThought(contentId, toggleId) {
            const content = document.getElementById(contentId);
            const toggle = document.getElementById(toggleId);
            const arrow = document.getElementById(toggleId + '_arrow');
            
            if (content.style.display === 'none' || content.style.display === '') {
                content.style.display = 'block';
                arrow.innerHTML = '▼';
                toggle.style.backgroundColor = '#e5e7eb';
            } else {
                content.style.display = 'none';
                arrow.innerHTML = '▶';
                toggle.style.backgroundColor = '#f0f2f6';
            }
        }
        
        // Keyboard shortcut: Alt + T to toggle most recent thought
        document.addEventListener('keydown', function(event) {
            if (event.altKey && event.key === 't') {
                event.preventDefault();
                const toggles = document.querySelectorAll('.thought-toggle-btn');
                if (toggles.length > 0) {
                    toggles[toggles.length - 1].click();
                }
            }
        });
        </script>
        """
    
    def should_show_thoughts(self, session_state: Optional[Dict] = None) -> bool:
        """
        Determine if thoughts should be shown based on configuration and session state.
        
        Args:
            session_state: Current session state (for per-session overrides)
            
        Returns:
            True if thoughts should be available, False otherwise
        """
        try:
            # Check global configuration
            if not self.show_thoughts:
                return False
            
            # Check session-level override
            if session_state and 'thoughts_enabled' in session_state:
                return session_state['thoughts_enabled']
            
            # Default to configuration
            return self.enable_thought_toggle
            
        except Exception as e:
            logger.debug(f"Error checking thought visibility: {e}")
            return True  # Default to showing thoughts on error
    
    def process_slash_command(self, command: str, session_state: Dict) -> Tuple[bool, str]:
        """
        Process /thoughts slash command.
        
        Args:
            command: The slash command (e.g., "/thoughts on")
            session_state: Current session state to modify
            
        Returns:
            Tuple of (command_handled, response_message)
        """
        try:
            parts = command.strip().split()
            
            if len(parts) >= 2 and parts[0] == '/thoughts':
                action = parts[1].lower()
                
                if action in ['on', 'enable', 'show']:
                    session_state['thoughts_enabled'] = True
                    return True, "🧠 Thought visibility enabled. SAM's thinking process will be available via toggle buttons."
                
                elif action in ['off', 'disable', 'hide']:
                    session_state['thoughts_enabled'] = False
                    return True, "🔒 Thought visibility disabled. SAM's thinking process will be hidden."
                
                elif action in ['status', 'check']:
                    current_status = session_state.get('thoughts_enabled', self.enable_thought_toggle)
                    status_text = "enabled" if current_status else "disabled"
                    return True, f"🧠 Thought visibility is currently {status_text}."
                
                else:
                    return True, "❓ Usage: /thoughts [on|off|status]"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error processing thoughts command: {e}")
            return True, "❌ Error processing thoughts command."

# Global thought processor instance
_thought_processor = None

def get_thought_processor() -> ThoughtProcessor:
    """Get or create a global thought processor instance."""
    global _thought_processor
    
    if _thought_processor is None:
        _thought_processor = ThoughtProcessor()
    
    return _thought_processor
