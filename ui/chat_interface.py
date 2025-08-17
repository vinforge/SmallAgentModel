#!/usr/bin/env python3
"""
Enhanced Chat Interface - Sprint 16
Streamlit chat interface with thought toggle functionality.
"""

import streamlit as st
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from datetime import datetime

logger = logging.getLogger(__name__)

def initialize_chat_session():
    """Initialize chat session state for Sprint 16 features."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'thoughts_enabled' not in st.session_state:
        # Load from config
        try:
            import json
            with open('config/sam_config.json', 'r') as f:
                config = json.load(f)
                st.session_state.thoughts_enabled = config.get('show_thoughts', True)
        except Exception:
            st.session_state.thoughts_enabled = True
    
    if 'thought_toggles' not in st.session_state:
        st.session_state.thought_toggles = {}

def render_message_with_thoughts(message: Dict[str, Any], message_index: int):
    """
    Render a chat message with thought toggle functionality.
    
    Args:
        message: Message dictionary with 'role', 'content', and optional 'thoughts'
        message_index: Index of the message in the conversation
    """
    try:
        from utils.thought_processor import get_thought_processor
        
        thought_processor = get_thought_processor()
        
        # Check if thoughts should be shown
        show_thoughts = thought_processor.should_show_thoughts(st.session_state)
        
        # Process the message content
        if message['role'] == 'assistant' and show_thoughts:
            processed = thought_processor.process_response(message['content'])
            
            # Display visible content
            with st.chat_message("assistant"):
                st.markdown(processed.visible_content)
                
                # Add thought toggles if thoughts exist
                if processed.has_thoughts:
                    st.markdown("---")
                    
                    for i, thought_block in enumerate(processed.thought_blocks):
                        # Create unique key for this thought toggle
                        toggle_key = f"thought_{message_index}_{i}"
                        
                        # Initialize toggle state if not exists
                        if toggle_key not in st.session_state.thought_toggles:
                            st.session_state.thought_toggles[toggle_key] = False
                        
                        # Create toggle button
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            button_label = f"🧠 SAM's Thoughts ({thought_block.token_count} tokens)"
                            if st.button(button_label, key=f"btn_{toggle_key}"):
                                st.session_state.thought_toggles[toggle_key] = not st.session_state.thought_toggles[toggle_key]
                        
                        with col2:
                            # Show toggle state
                            if st.session_state.thought_toggles[toggle_key]:
                                st.markdown("🔽 **Expanded**")
                            else:
                                st.markdown("▶️ **Collapsed**")
                        
                        # Show thought content if toggled
                        if st.session_state.thought_toggles[toggle_key]:
                            with st.expander("", expanded=True):
                                # Metadata
                                st.caption(f"Tokens: {thought_block.token_count} | Time: {thought_block.timestamp[:19]}")
                                
                                # Thought content
                                st.code(thought_block.content, language="text")
        else:
            # Regular message display
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                
    except Exception as e:
        logger.error(f"Error rendering message with thoughts: {e}")
        # Fallback to regular message display
        with st.chat_message(message['role']):
            st.markdown(message['content'])

def render_chat_interface():
    """Render the main chat interface with Sprint 16 enhancements."""
    try:
        # Header with engine status
        col1, col2 = st.columns([3, 1])

        with col1:
            st.title("💬 Chat with SAM")
            st.markdown("**Enhanced with Sprint 16: Thought Transparency Controls**")

        with col2:
            # Engine status indicator
            render_engine_status_indicator()
        
        # Initialize session
        initialize_chat_session()
        
        # Thought controls in sidebar
        with st.sidebar:
            st.subheader("🧠 Thought Controls")
            
            # Global thought toggle
            thoughts_enabled = st.checkbox(
                "Enable Thought Visibility",
                value=st.session_state.thoughts_enabled,
                help="Show/hide SAM's thinking process"
            )
            
            if thoughts_enabled != st.session_state.thoughts_enabled:
                st.session_state.thoughts_enabled = thoughts_enabled
                st.rerun()
            
            # Thought statistics
            if st.session_state.messages:
                total_thoughts = sum(
                    1 for msg in st.session_state.messages 
                    if msg['role'] == 'assistant' and '<think>' in msg['content']
                )
                st.metric("Messages with Thoughts", total_thoughts)
            
            # Quick commands
            st.subheader("⚡ Quick Commands")
            st.markdown("""
            - `/thoughts on` - Enable thoughts
            - `/thoughts off` - Disable thoughts  
            - `/thoughts status` - Check status
            - `Alt + T` - Toggle recent thought
            """)
        
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            render_message_with_thoughts(message, i)
        
        # Chat input
        if prompt := st.chat_input("Ask SAM anything..."):
            # Handle slash commands
            if prompt.startswith('/thoughts'):
                from utils.thought_processor import get_thought_processor
                
                thought_processor = get_thought_processor()
                handled, response = thought_processor.process_slash_command(prompt, st.session_state)
                
                if handled:
                    # Add command and response to chat
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
            else:
                # Regular chat message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Generate response (placeholder - would integrate with actual SAM model)
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("SAM is thinking..."):
                        # This would call the actual SAM model
                        response = generate_sam_response(prompt)

                        # Generate unique message ID for feedback tracking
                        import uuid
                        message_id = str(uuid.uuid4())

                        # Add response to messages with ID
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "message_id": message_id
                        })

                        # Process and display with thought handling
                        from utils.thought_processor import get_thought_processor

                        thought_processor = get_thought_processor()
                        show_thoughts = thought_processor.should_show_thoughts(st.session_state)

                        # Add feedback controls
                        st.markdown("---")
                        render_feedback_controls(message_id, prompt, response)
                        
                        if show_thoughts:
                            processed = thought_processor.process_response(response)
                            st.markdown(processed.visible_content)
                            
                            # Handle thoughts
                            if processed.has_thoughts:
                                st.markdown("---")
                                
                                for i, thought_block in enumerate(processed.thought_blocks):
                                    toggle_key = f"thought_{len(st.session_state.messages)-1}_{i}"
                                    st.session_state.thought_toggles[toggle_key] = False
                                    
                                    # Show toggle button
                                    button_label = f"🧠 SAM's Thoughts ({thought_block.token_count} tokens)"
                                    if st.button(button_label, key=f"new_btn_{toggle_key}"):
                                        st.session_state.thought_toggles[toggle_key] = True
                                        st.rerun()
                        else:
                            st.markdown(response)
        
        # Keyboard shortcut info
        if st.session_state.thoughts_enabled:
            st.markdown("---")
            st.caption("💡 **Tip:** Press `Alt + T` to toggle the most recent thought block")
        
    except Exception as e:
        logger.error(f"Error rendering chat interface: {e}")
        st.error("Error loading chat interface. Please refresh the page.")

def generate_sam_response(prompt: str) -> str:
    """
    Generate SAM response (placeholder for actual model integration).
    
    Args:
        prompt: User input prompt
        
    Returns:
        SAM response with potential <think> blocks
    """
    try:
        # This is a placeholder - would integrate with actual SAM model
        # For demo purposes, include some <think> blocks
        
        if "date" in prompt.lower() or "time" in prompt.lower():
            return f"""<think>
The user is asking about dates or time. I should check if they're referring to:
1. Current date/time
2. Dates from uploaded documents
3. Scheduling or calendar information

Let me provide a helpful response about the current date and mention document dates if relevant.
</think>

Today is {datetime.now().strftime('%B %d, %Y')}. 

If you're looking for specific dates from your uploaded documents, I can help you find important dates, deadlines, or milestones. Just let me know what kind of dates you're interested in!"""
        
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            return f"""<think>
This is a greeting. I should respond warmly and introduce my capabilities, especially the new Sprint 16 thought transparency features.
</think>

Hello! I'm SAM, your intelligent assistant. I'm here to help you with questions, document analysis, and more.

**New in Sprint 16:** You can now see my thinking process! Look for the "🧠 SAM's Thoughts" buttons to explore how I reason through responses."""
        
        else:
            return f"""<think>
The user asked: "{prompt}"

I need to provide a helpful response. Since this is a demo of Sprint 16 thought transparency, I'll include my reasoning process.
</think>

I understand you're asking about "{prompt}". I'm here to help! 

This response demonstrates the new Sprint 16 thought transparency feature - you can click the thought toggle above to see my reasoning process."""
        
    except Exception as e:
        logger.error(f"Error generating SAM response: {e}")
        return "I apologize, but I encountered an error generating a response. Please try again."

def render_feedback_controls(message_id: str, query: str, response: str):
    """Render feedback controls for a message."""
    try:
        st.markdown("**How was this response?**")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("👍 Helpful", key=f"helpful_{message_id}"):
                submit_message_feedback(message_id, query, response, rating=0.8, feedback_type="positive")
                st.success("Thank you for your feedback!")

        with col2:
            if st.button("👎 Needs Work", key=f"needs_work_{message_id}"):
                submit_message_feedback(message_id, query, response, rating=0.2, feedback_type="negative")
                st.warning("Thanks for letting me know. I'll work on improving!")

        with col3:
            if st.button("✏️ Suggest Improvement", key=f"suggest_{message_id}"):
                st.session_state[f"show_correction_{message_id}"] = True

        # Show correction input if requested
        if st.session_state.get(f"show_correction_{message_id}", False):
            correction_text = st.text_area(
                "How can I improve this response?",
                key=f"correction_{message_id}",
                placeholder="e.g., 'too mechanical, more natural next time' or 'needs more detail about...'"
            )

            col_submit, col_cancel = st.columns(2)
            with col_submit:
                if st.button("Submit Feedback", key=f"submit_{message_id}"):
                    if correction_text.strip():
                        submit_message_feedback(
                            message_id, query, response,
                            rating=0.4, feedback_type="correction",
                            correction_text=correction_text
                        )
                        st.success("Thank you! I'll learn from this feedback.")
                        st.session_state[f"show_correction_{message_id}"] = False
                        st.rerun()
                    else:
                        st.error("Please provide some feedback before submitting.")

            with col_cancel:
                if st.button("Cancel", key=f"cancel_{message_id}"):
                    st.session_state[f"show_correction_{message_id}"] = False
                    st.rerun()

    except Exception as e:
        logger.error(f"Error rendering feedback controls: {e}")

def submit_message_feedback(message_id: str, query: str, response: str,
                          rating: float, feedback_type: str,
                          correction_text: str = None):
    """Submit feedback for a message."""
    try:
        # Try to integrate with feedback system
        from learning.feedback_handler import get_feedback_handler
        from learning.feedback_types import FeedbackType, CorrectionType

        feedback_handler = get_feedback_handler()
        user_id = "streamlit_user"  # In production, get from session/auth

        # Determine feedback type
        if feedback_type == "correction":
            fb_type = FeedbackType.CORRECTION
            corr_type = CorrectionType.STYLE_IMPROVEMENT
        else:
            fb_type = FeedbackType.SATISFACTION_RATING
            corr_type = None

        # Submit feedback
        feedback_id = feedback_handler.submit_feedback(
            memory_id=message_id,
            user_id=user_id,
            feedback_type=fb_type,
            rating=rating,
            correction_text=correction_text,
            correction_type=corr_type
        )

        logger.info(f"Feedback submitted: {feedback_id} for message {message_id}")

        # Store in session state for immediate use
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []

        st.session_state.feedback_history.append({
            'message_id': message_id,
            'query': query,
            'response': response,
            'rating': rating,
            'feedback_type': feedback_type,
            'correction_text': correction_text,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        # Fallback: store in session state only
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []

        st.session_state.feedback_history.append({
            'message_id': message_id,
            'query': query,
            'response': response,
            'rating': rating,
            'feedback_type': feedback_type,
            'correction_text': correction_text,
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        })

def render_thought_settings():
    """Render thought visibility settings panel."""
    try:
        st.subheader("🧠 Thought Transparency Settings")
        st.markdown("**Sprint 16 Feature:** Control how SAM's thinking process is displayed")

        # Current settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Settings:**")
            st.metric("Thoughts Enabled", "✅" if st.session_state.thoughts_enabled else "❌")
            
            # Toggle count
            toggle_count = len([k for k, v in st.session_state.thought_toggles.items() if v])
            st.metric("Expanded Thoughts", toggle_count)
        
        with col2:
            st.markdown("**Actions:**")
            
            if st.button("🔒 Hide All Thoughts"):
                st.session_state.thoughts_enabled = False
                st.session_state.thought_toggles = {}
                st.success("All thoughts hidden")
                st.rerun()
            
            if st.button("👁️ Show All Thoughts"):
                st.session_state.thoughts_enabled = True
                st.success("Thoughts enabled")
                st.rerun()
            
            if st.button("📖 Expand All Thoughts"):
                for key in st.session_state.thought_toggles:
                    st.session_state.thought_toggles[key] = True
                st.success("All thoughts expanded")
                st.rerun()
            
            if st.button("📕 Collapse All Thoughts"):
                for key in st.session_state.thought_toggles:
                    st.session_state.thought_toggles[key] = False
                st.success("All thoughts collapsed")
                st.rerun()
        
        # Usage statistics
        st.subheader("📊 Usage Statistics")
        
        if st.session_state.messages:
            total_messages = len(st.session_state.messages)
            assistant_messages = len([m for m in st.session_state.messages if m['role'] == 'assistant'])
            thought_messages = len([m for m in st.session_state.messages if m['role'] == 'assistant' and '<think>' in m['content']])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Messages", total_messages)
            
            with col2:
                st.metric("Assistant Messages", assistant_messages)
            
            with col3:
                st.metric("Messages with Thoughts", thought_messages)
            
            # Thought usage percentage
            if assistant_messages > 0:
                thought_percentage = (thought_messages / assistant_messages) * 100
                st.metric("Thought Usage", f"{thought_percentage:.1f}%")
        else:
            st.info("No messages yet. Start a conversation to see statistics!")
        
    except Exception as e:
        logger.error(f"Error rendering thought settings: {e}")
        st.error("Error loading thought settings.")


def render_engine_status_indicator():
    """Render the current engine status indicator."""
    try:
        # Try to get current engine information
        try:
            from sam.core.model_interface import get_current_model_info
            from sam.config import get_config_manager

            # Get current model info
            current_info = get_current_model_info()
            current_model = current_info.get('primary_model', 'Unknown')

            # Try to get engine upgrade info from config
            try:
                config_manager = get_config_manager()
                config = config_manager.get_config()

                if hasattr(config, 'engine_upgrade') and config.engine_upgrade:
                    engine_info = config.engine_upgrade
                    engine_name = engine_info.get('engine_model_name', current_model)
                    engine_family = engine_info.get('engine_family', 'unknown')
                    last_upgrade = engine_info.get('last_upgrade', 'Never')

                    # Show enhanced engine status
                    st.markdown("### 🔧 Active Engine")
                    st.success(f"**{engine_name}**")
                    st.caption(f"Family: {engine_family.title()}")

                    if last_upgrade != 'Never':
                        from datetime import datetime
                        try:
                            upgrade_date = datetime.fromisoformat(last_upgrade.replace('Z', '+00:00'))
                            st.caption(f"Upgraded: {upgrade_date.strftime('%Y-%m-%d %H:%M')}")
                        except:
                            st.caption(f"Upgraded: {last_upgrade}")
                else:
                    # Show basic engine status
                    st.markdown("### 🤖 Active Model")
                    st.info(f"**{current_model}**")
                    st.caption("Default SAM Engine")

            except Exception as e:
                # Fallback to basic model info
                st.markdown("### 🤖 Active Model")
                st.info(f"**{current_model}**")

        except Exception as e:
            # Ultimate fallback
            st.markdown("### 🤖 SAM Engine")
            st.warning("**Status Unknown**")
            st.caption("Unable to determine engine status")

    except Exception as e:
        logger.error(f"Error rendering engine status: {e}")
        # Silent failure - don't break the chat interface
