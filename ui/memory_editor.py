"""
Memory Editing & Deletion Tools for SAM
User control over memory content, tags, and retention with safeguards.

Sprint 12 Task 2: Memory Editing & Deletion Tools
"""

import logging
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import uuid

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_vectorstore import MemoryVectorStore, MemoryType, MemoryChunk, get_memory_store
from memory.memory_reasoning import MemoryDrivenReasoningEngine, get_memory_reasoning_engine

logger = logging.getLogger(__name__)

class MemoryEditor:
    """
    Memory editing and deletion tools with safeguards and audit logging.
    """
    
    def __init__(self):
        """Initialize the memory editor."""
        self.memory_store = get_memory_store()
        self.memory_reasoning = get_memory_reasoning_engine()
        
        # Initialize session state for undo functionality
        if 'deleted_memories' not in st.session_state:
            st.session_state.deleted_memories = []
        if 'edit_history' not in st.session_state:
            st.session_state.edit_history = []
        if 'audit_log' not in st.session_state:
            st.session_state.audit_log = []
    
    def render_edit_interface(self, memory: MemoryChunk):
        """Render the memory editing interface."""
        st.subheader("✏️ Edit Memory")
        
        with st.form("edit_memory_form"):
            # Memory content
            st.markdown("**Content:**")
            new_content = st.text_area(
                "Memory Content",
                value=memory.content,
                height=150,
                help="Edit the memory content"
            )
            
            # Memory type
            st.markdown("**Type:**")
            new_type = st.selectbox(
                "Memory Type",
                options=[mt.value for mt in MemoryType],
                index=list(MemoryType).index(memory.memory_type),
                help="Change the memory type"
            )
            
            # Source
            st.markdown("**Source:**")
            new_source = st.text_input(
                "Source",
                value=memory.source,
                help="Edit the memory source"
            )
            
            # Tags
            st.markdown("**Tags:**")
            tags_text = ", ".join(memory.tags)
            new_tags_text = st.text_input(
                "Tags (comma-separated)",
                value=tags_text,
                help="Edit memory tags"
            )
            new_tags = [tag.strip() for tag in new_tags_text.split(",") if tag.strip()]
            
            # Importance score
            st.markdown("**Importance:**")
            new_importance = st.slider(
                "Importance Score",
                min_value=0.0,
                max_value=1.0,
                value=memory.importance_score,
                step=0.1,
                help="Adjust memory importance"
            )
            
            # Priority settings
            st.markdown("**Priority Settings:**")
            col1, col2 = st.columns(2)
            
            with col1:
                pin_priority = st.checkbox(
                    "📌 Pin as Priority",
                    value="priority" in memory.tags,
                    help="Mark this memory as high priority for recall"
                )
            
            with col2:
                do_not_recall = st.checkbox(
                    "🚫 Do Not Recall",
                    value="do_not_recall" in memory.tags,
                    help="Prevent this memory from being recalled in reasoning"
                )
            
            # Update tags based on priority settings
            if pin_priority and "priority" not in new_tags:
                new_tags.append("priority")
            elif not pin_priority and "priority" in new_tags:
                new_tags.remove("priority")
            
            if do_not_recall and "do_not_recall" not in new_tags:
                new_tags.append("do_not_recall")
            elif not do_not_recall and "do_not_recall" in new_tags:
                new_tags.remove("do_not_recall")
            
            # Metadata editing
            st.markdown("**Metadata:**")
            metadata_json = st.text_area(
                "Metadata (JSON)",
                value=json.dumps(memory.metadata, indent=2),
                height=100,
                help="Edit memory metadata as JSON"
            )
            
            # Form buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                save_button = st.form_submit_button("💾 Save Changes", type="primary")
            
            with col2:
                preview_button = st.form_submit_button("👁️ Preview Changes")
            
            with col3:
                cancel_button = st.form_submit_button("❌ Cancel")
            
            # Handle form submission
            if save_button:
                self._save_memory_changes(
                    memory, new_content, new_type, new_source, 
                    new_tags, new_importance, metadata_json
                )
            
            elif preview_button:
                self._preview_changes(
                    memory, new_content, new_type, new_source,
                    new_tags, new_importance, metadata_json
                )
            
            elif cancel_button:
                st.session_state.editing_memory = None
                st.rerun()
    
    def render_delete_interface(self, memory: MemoryChunk):
        """Render the memory deletion interface."""
        st.subheader("🗑️ Delete Memory")
        
        # Warning message
        st.warning("⚠️ You are about to delete this memory. This action can be undone within this session.")
        
        # Memory preview
        with st.container():
            st.markdown("**Memory to Delete:**")
            st.markdown(f"**Type:** {memory.memory_type.value}")
            st.markdown(f"**Source:** {memory.source}")
            st.markdown(f"**Created:** {memory.timestamp}")
            
            # Content preview
            content_preview = memory.content[:200]
            if len(memory.content) > 200:
                content_preview += "..."
            st.markdown(f"**Content:** {content_preview}")
            
            if memory.tags:
                st.markdown(f"**Tags:** {', '.join(memory.tags)}")
        
        # Deletion options
        st.markdown("**Deletion Options:**")
        
        delete_reason = st.selectbox(
            "Reason for deletion",
            options=[
                "No longer relevant",
                "Incorrect information",
                "Privacy concerns",
                "Duplicate content",
                "User request",
                "Other"
            ]
        )
        
        if delete_reason == "Other":
            custom_reason = st.text_input("Please specify:")
            delete_reason = custom_reason if custom_reason else "Other"
        
        # Confirmation
        st.markdown("**Confirmation:**")
        confirm_text = st.text_input(
            f"Type 'DELETE {memory.chunk_id[:8]}' to confirm:",
            help="This ensures you really want to delete this memory"
        )
        
        expected_text = f"DELETE {memory.chunk_id[:8]}"
        confirmation_valid = confirm_text == expected_text
        
        # Delete button
        col1, col2 = st.columns(2)
        
        with col1:
            delete_button = st.button(
                "🗑️ Delete Memory",
                type="secondary",
                disabled=not confirmation_valid,
                help="Delete this memory permanently"
            )
        
        with col2:
            cancel_button = st.button("❌ Cancel")
        
        if delete_button and confirmation_valid:
            self._delete_memory(memory, delete_reason)
        
        elif cancel_button:
            st.session_state.deleting_memory = None
            st.rerun()
    
    def render_undo_interface(self):
        """Render the undo interface for deleted memories."""
        if not st.session_state.deleted_memories:
            return
        
        st.subheader("↩️ Undo Deletions")
        
        st.info(f"You have {len(st.session_state.deleted_memories)} deleted memories that can be restored.")
        
        for i, deleted_memory in enumerate(st.session_state.deleted_memories):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{deleted_memory['memory'].memory_type.value}** - {deleted_memory['memory'].source}")
                    st.caption(f"Deleted: {deleted_memory['deleted_at']}")
                    st.caption(f"Reason: {deleted_memory['reason']}")
                
                with col2:
                    if st.button("↩️ Restore", key=f"restore_{i}"):
                        self._restore_memory(i)
                
                with col3:
                    if st.button("🗑️ Permanent", key=f"permanent_{i}"):
                        self._permanently_delete(i)
                
                st.divider()
    
    def render_audit_log(self):
        """Render the audit log interface."""
        st.subheader("📋 Audit Log")
        
        if not st.session_state.audit_log:
            st.info("No memory operations have been logged in this session.")
            return
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            action_filter = st.selectbox(
                "Filter by Action",
                options=["All", "Edit", "Delete", "Restore", "Create"],
                index=0
            )
        
        with col2:
            show_last = st.selectbox(
                "Show Last",
                options=[10, 25, 50, 100],
                index=0
            )
        
        # Filter audit log
        filtered_log = st.session_state.audit_log
        if action_filter != "All":
            filtered_log = [entry for entry in filtered_log if entry['action'] == action_filter.lower()]
        
        # Show recent entries
        recent_log = filtered_log[-show_last:]
        
        # Display audit entries
        for entry in reversed(recent_log):
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    action_emoji = {
                        'edit': '✏️',
                        'delete': '🗑️',
                        'restore': '↩️',
                        'create': '➕'
                    }.get(entry['action'], '📝')
                    
                    st.markdown(f"{action_emoji} **{entry['action'].title()}** - {entry['memory_id'][:8]}")
                    if entry.get('details'):
                        st.caption(entry['details'])
                
                with col2:
                    st.caption(entry['timestamp'][:19])
                
                with col3:
                    if entry.get('user_id'):
                        st.caption(f"User: {entry['user_id']}")
                
                st.divider()
    
    def _save_memory_changes(self, memory: MemoryChunk, new_content: str, 
                           new_type: str, new_source: str, new_tags: List[str],
                           new_importance: float, metadata_json: str):
        """Save changes to a memory."""
        try:
            # Parse metadata
            try:
                new_metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                st.error("Invalid JSON in metadata field")
                return
            
            # Create audit log entry
            changes = []
            if new_content != memory.content:
                changes.append("content")
            if new_type != memory.memory_type.value:
                changes.append("type")
            if new_source != memory.source:
                changes.append("source")
            if new_tags != memory.tags:
                changes.append("tags")
            if new_importance != memory.importance_score:
                changes.append("importance")
            if new_metadata != memory.metadata:
                changes.append("metadata")
            
            if not changes:
                st.info("No changes detected")
                return
            
            # Save original state for undo
            original_state = {
                'memory_id': memory.chunk_id,
                'original_memory': memory,
                'timestamp': datetime.now().isoformat(),
                'changes': changes
            }
            st.session_state.edit_history.append(original_state)
            
            # Update memory
            success = self.memory_store.update_memory(
                chunk_id=memory.chunk_id,
                content=new_content if new_content != memory.content else None,
                tags=new_tags if new_tags != memory.tags else None,
                importance_score=new_importance if new_importance != memory.importance_score else None,
                metadata=new_metadata if new_metadata != memory.metadata else None
            )
            
            if success:
                # Update memory object
                if new_content != memory.content:
                    memory.content = new_content
                if new_type != memory.memory_type.value:
                    memory.memory_type = MemoryType(new_type)
                if new_source != memory.source:
                    memory.source = new_source
                if new_tags != memory.tags:
                    memory.tags = new_tags
                if new_importance != memory.importance_score:
                    memory.importance_score = new_importance
                if new_metadata != memory.metadata:
                    memory.metadata = new_metadata
                
                # Log the change
                self._log_audit_entry(
                    action="edit",
                    memory_id=memory.chunk_id,
                    details=f"Changed: {', '.join(changes)}",
                    user_id=memory.metadata.get('user_id')
                )
                
                st.success("Memory updated successfully!")
                st.session_state.editing_memory = None
                st.rerun()
            else:
                st.error("Failed to update memory")
                
        except Exception as e:
            st.error(f"Error saving changes: {e}")
    
    def _preview_changes(self, memory: MemoryChunk, new_content: str,
                        new_type: str, new_source: str, new_tags: List[str],
                        new_importance: float, metadata_json: str):
        """Preview changes before saving."""
        st.subheader("👁️ Preview Changes")
        
        # Show changes
        changes = []
        
        if new_content != memory.content:
            changes.append(("Content", memory.content[:100] + "...", new_content[:100] + "..."))
        
        if new_type != memory.memory_type.value:
            changes.append(("Type", memory.memory_type.value, new_type))
        
        if new_source != memory.source:
            changes.append(("Source", memory.source, new_source))
        
        if new_tags != memory.tags:
            changes.append(("Tags", ", ".join(memory.tags), ", ".join(new_tags)))
        
        if new_importance != memory.importance_score:
            changes.append(("Importance", f"{memory.importance_score:.2f}", f"{new_importance:.2f}"))
        
        if changes:
            st.markdown("**Changes to be made:**")
            for field, old_value, new_value in changes:
                st.markdown(f"**{field}:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"*Before:* {old_value}")
                with col2:
                    st.markdown(f"*After:* {new_value}")
        else:
            st.info("No changes detected")
    
    def _delete_memory(self, memory: MemoryChunk, reason: str):
        """Delete a memory with undo capability."""
        try:
            # Store for undo
            deleted_memory = {
                'memory': memory,
                'deleted_at': datetime.now().isoformat(),
                'reason': reason,
                'memory_id': memory.chunk_id
            }
            st.session_state.deleted_memories.append(deleted_memory)
            
            # Delete from store
            success = self.memory_store.delete_memory(memory.chunk_id)
            
            if success:
                # Log the deletion
                self._log_audit_entry(
                    action="delete",
                    memory_id=memory.chunk_id,
                    details=f"Reason: {reason}",
                    user_id=memory.metadata.get('user_id')
                )
                
                st.success("Memory deleted successfully! You can undo this action below.")
                st.session_state.deleting_memory = None
                st.rerun()
            else:
                st.error("Failed to delete memory")
                # Remove from deleted list if deletion failed
                st.session_state.deleted_memories.pop()
                
        except Exception as e:
            st.error(f"Error deleting memory: {e}")
    
    def _restore_memory(self, index: int):
        """Restore a deleted memory."""
        try:
            deleted_memory = st.session_state.deleted_memories[index]
            memory = deleted_memory['memory']
            
            # Re-add to memory store
            chunk_id = self.memory_store.add_memory(
                content=memory.content,
                memory_type=memory.memory_type,
                source=memory.source,
                tags=memory.tags,
                importance_score=memory.importance_score,
                metadata=memory.metadata
            )
            
            if chunk_id:
                # Log the restoration
                self._log_audit_entry(
                    action="restore",
                    memory_id=memory.chunk_id,
                    details="Memory restored from deletion",
                    user_id=memory.metadata.get('user_id')
                )
                
                # Remove from deleted list
                st.session_state.deleted_memories.pop(index)
                
                st.success("Memory restored successfully!")
                st.rerun()
            else:
                st.error("Failed to restore memory")
                
        except Exception as e:
            st.error(f"Error restoring memory: {e}")
    
    def _permanently_delete(self, index: int):
        """Permanently delete a memory (remove from undo list)."""
        deleted_memory = st.session_state.deleted_memories[index]
        
        # Log permanent deletion
        self._log_audit_entry(
            action="permanent_delete",
            memory_id=deleted_memory['memory_id'],
            details="Memory permanently deleted",
            user_id=deleted_memory['memory'].metadata.get('user_id')
        )
        
        # Remove from deleted list
        st.session_state.deleted_memories.pop(index)
        
        st.success("Memory permanently deleted")
        st.rerun()
    
    def _log_audit_entry(self, action: str, memory_id: str, details: str = None, user_id: str = None):
        """Log an audit entry."""
        entry = {
            'action': action,
            'memory_id': memory_id,
            'timestamp': datetime.now().isoformat(),
            'details': details,
            'user_id': user_id,
            'session_id': st.session_state.get('session_id', 'unknown')
        }
        
        st.session_state.audit_log.append(entry)
        
        # Keep only last 1000 entries
        if len(st.session_state.audit_log) > 1000:
            st.session_state.audit_log = st.session_state.audit_log[-1000:]
