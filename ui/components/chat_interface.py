#!/usr/bin/env python3
"""
Chat Interface Components for SAM UI
====================================

Handles chat interface rendering and interaction functionality
extracted from the monolithic secure_streamlit_app.py.

This module provides:
- Chat message rendering
- Conversation history management
- Message input handling
- Chat UI components

Author: SAM Development Team
Version: 1.0.0 - Refactored from secure_streamlit_app.py
"""

import streamlit as st
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ChatInterface:
    """Manages the chat interface for SAM."""
    
    def __init__(self):
        self.max_history_length = 100
        self.message_display_limit = 50
    
    def initialize_chat_history(self):
        """Initialize chat history in session state."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = self._generate_conversation_id()
    
    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = st.session_state.get('session_id', 'default')
        return f"conv_{session_id}_{timestamp}"
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the chat history."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'message_id': f"msg_{int(time.time() * 1000)}",
            'metadata': metadata or {}
        }
        
        st.session_state.chat_history.append(message)
        
        # Trim history if it gets too long
        if len(st.session_state.chat_history) > self.max_history_length:
            st.session_state.chat_history = st.session_state.chat_history[-self.max_history_length:]
        
        logger.debug(f"Added {role} message: {content[:100]}...")
    
    def render_chat_messages(self):
        """Render all chat messages in the interface."""
        if not st.session_state.chat_history:
            self._render_welcome_message()
            return
        
        # Display recent messages (limit for performance)
        recent_messages = st.session_state.chat_history[-self.message_display_limit:]
        
        for i, message in enumerate(recent_messages):
            self._render_single_message(message, i)
    
    def _render_welcome_message(self):
        """Render welcome message when chat is empty."""
        st.markdown("""
        ### 👋 Welcome to SAM!
        
        I'm your AI assistant, ready to help with:
        - 📄 Document analysis and summarization
        - 🔍 Research and information gathering
        - 💡 Problem-solving and brainstorming
        - 🧠 Complex reasoning and analysis
        
        **Get started by:**
        - Uploading a document to analyze
        - Asking me a question
        - Exploring the suggested prompts below
        """)
        
        # Show suggested prompts
        self._render_suggested_prompts()
    
    def _render_suggested_prompts(self):
        """Render suggested conversation starters."""
        st.markdown("#### 💡 Suggested Prompts")
        
        prompts = [
            "🔍 Help me analyze a research paper",
            "📊 Explain complex data or statistics",
            "💭 Brainstorm solutions to a problem",
            "📝 Summarize key information",
            "🎯 Create a strategic plan",
            "🔬 Conduct a literature review"
        ]
        
        cols = st.columns(2)
        
        for i, prompt in enumerate(prompts):
            col = cols[i % 2]
            
            with col:
                if st.button(prompt, key=f"suggested_{i}"):
                    self.add_message("user", prompt)
                    st.rerun()
    
    def _render_single_message(self, message: Dict[str, Any], index: int):
        """Render a single chat message."""
        role = message['role']
        content = message['content']
        timestamp = message.get('timestamp', '')
        metadata = message.get('metadata', {})
        
        # Choose avatar and styling based on role
        if role == 'user':
            avatar = "👤"
            message_class = "user-message"
        elif role == 'assistant':
            avatar = "🤖"
            message_class = "assistant-message"
        else:
            avatar = "ℹ️"
            message_class = "system-message"
        
        # Create message container
        with st.chat_message(role, avatar=avatar):
            # Render message content
            if role == 'assistant':
                self._render_assistant_message(content, metadata, index)
            else:
                st.markdown(content)
            
            # Show timestamp and metadata
            if timestamp:
                time_str = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
                st.caption(f"⏰ {time_str}")
    
    def _render_assistant_message(self, content: str, metadata: Dict[str, Any], index: int):
        """Render an assistant message with special formatting."""
        # Check if this is a document analysis response
        if 'filename' in metadata:
            filename = metadata['filename']
            
            # Show document context
            st.info(f"📄 Analysis of: **{filename}**")
            
            # Render action buttons for document
            self._render_document_action_buttons(filename, index)
        
        # Render the main content
        st.markdown(content)
        
        # Show additional metadata if available
        if metadata.get('processing_time'):
            st.caption(f"⚡ Processed in {metadata['processing_time']:.2f}s")
        
        if metadata.get('tokens_used'):
            st.caption(f"🔤 Tokens used: {metadata['tokens_used']}")
    
    def _render_document_action_buttons(self, filename: str, index: int):
        """Render action buttons for document-related messages."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"📋 Summarize", key=f"summarize_{index}_{filename}"):
                from sam.ui.handlers.document_handler import generate_enhanced_summary_prompt
                prompt = f"Analyze the uploaded PDF file {filename}, not any CSV data. " + generate_enhanced_summary_prompt(filename)
                self.add_message("user", f"📋 Summarize: {filename}")
                # This would trigger response generation
                st.rerun()
        
        with col2:
            if st.button(f"❓ Key Questions", key=f"questions_{index}_{filename}"):
                from sam.ui.handlers.document_handler import generate_enhanced_questions_prompt
                prompt = f"Analyze the uploaded PDF file {filename}, not any CSV data. " + generate_enhanced_questions_prompt(filename)
                self.add_message("user", f"❓ Key Questions: {filename}")
                st.rerun()
        
        with col3:
            if st.button(f"🔍 Deep Analysis", key=f"analysis_{index}_{filename}"):
                from sam.ui.handlers.document_handler import generate_enhanced_analysis_prompt
                prompt = f"Analyze the uploaded PDF file {filename}, not any CSV data. " + generate_enhanced_analysis_prompt(filename)
                self.add_message("user", f"🔍 Deep Analysis: {filename}")
                st.rerun()
    
    def render_message_input(self) -> Optional[str]:
        """Render message input interface and return user input."""
        # Chat input
        user_input = st.chat_input("💬 Ask me anything...")
        
        if user_input:
            # Add user message to history
            self.add_message("user", user_input)
            return user_input
        
        return None
    
    def render_conversation_controls(self):
        """Render conversation control buttons."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                self.clear_chat_history()
                st.rerun()
        
        with col2:
            if st.button("💾 Save Chat"):
                self.save_chat_history()
        
        with col3:
            if st.button("📤 Export Chat"):
                self.export_chat_history()
        
        with col4:
            if st.button("🔄 New Conversation"):
                self.start_new_conversation()
                st.rerun()
    
    def clear_chat_history(self):
        """Clear the current chat history."""
        st.session_state.chat_history = []
        logger.info("Chat history cleared")
    
    def start_new_conversation(self):
        """Start a new conversation."""
        self.clear_chat_history()
        st.session_state.conversation_id = self._generate_conversation_id()
        logger.info(f"New conversation started: {st.session_state.conversation_id}")
    
    def save_chat_history(self):
        """Save chat history to file."""
        try:
            conversation_id = st.session_state.get('conversation_id', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{conversation_id}_{timestamp}.json"
            
            # Create conversations directory
            conversations_dir = Path("conversations")
            conversations_dir.mkdir(exist_ok=True)
            
            # Save chat history
            chat_data = {
                'conversation_id': conversation_id,
                'session_id': st.session_state.get('session_id'),
                'username': st.session_state.get('username'),
                'created_at': timestamp,
                'messages': st.session_state.chat_history
            }
            
            filepath = conversations_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ Chat saved to: {filename}")
            logger.info(f"Chat history saved: {filepath}")
            
        except Exception as e:
            st.error(f"❌ Failed to save chat: {str(e)}")
            logger.error(f"Failed to save chat history: {e}")
    
    def export_chat_history(self):
        """Export chat history as downloadable file."""
        try:
            conversation_id = st.session_state.get('conversation_id', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create export data
            export_data = {
                'conversation_id': conversation_id,
                'exported_at': timestamp,
                'message_count': len(st.session_state.chat_history),
                'messages': st.session_state.chat_history
            }
            
            # Convert to JSON
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            # Offer download
            st.download_button(
                label="📥 Download Chat History",
                data=json_data,
                file_name=f"sam_chat_{conversation_id}_{timestamp}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Failed to export chat: {str(e)}")
            logger.error(f"Failed to export chat history: {e}")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        if not st.session_state.chat_history:
            return {}
        
        user_messages = [msg for msg in st.session_state.chat_history if msg['role'] == 'user']
        assistant_messages = [msg for msg in st.session_state.chat_history if msg['role'] == 'assistant']
        
        return {
            'conversation_id': st.session_state.get('conversation_id'),
            'total_messages': len(st.session_state.chat_history),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'started_at': st.session_state.chat_history[0]['timestamp'] if st.session_state.chat_history else None,
            'last_message_at': st.session_state.chat_history[-1]['timestamp'] if st.session_state.chat_history else None
        }


# Global chat interface instance
_chat_interface = None


def get_chat_interface() -> ChatInterface:
    """Get the global chat interface instance."""
    global _chat_interface
    if _chat_interface is None:
        _chat_interface = ChatInterface()
    return _chat_interface


def render_chat_interface():
    """Render the complete chat interface."""
    chat = get_chat_interface()
    
    # Initialize chat history
    chat.initialize_chat_history()
    
    # Render conversation controls
    with st.container():
        chat.render_conversation_controls()
    
    # Render chat messages
    with st.container():
        chat.render_chat_messages()
    
    # Render message input
    user_input = chat.render_message_input()
    
    return user_input
