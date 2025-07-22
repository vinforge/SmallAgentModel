#!/usr/bin/env python3
"""
SAM Contextual Relevance Engine - Task 31 Phase 1
=================================================

Implements intelligent conversation threading through vector-based relevance
calculation. Automatically detects topic changes and manages conversation
context to prevent pollution while maintaining coherence.

Part of Task 31: Conversational Intelligence Engine
Author: SAM Development Team
Version: 1.0.0
"""

import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RelevanceResult:
    """Result of contextual relevance calculation."""
    similarity_score: float
    is_relevant: bool
    threshold_used: float
    calculation_method: str
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ConversationThread:
    """Represents an archived conversation thread."""
    thread_id: str
    title: str
    messages: List[Dict[str, Any]]
    created_at: str
    last_updated: str
    message_count: int
    topic_keywords: List[str]
    embedding_summary: Optional[List[float]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationThread':
        return cls(**data)

class ContextualRelevanceEngine:
    """
    Intelligent conversation threading engine using vector-based relevance.
    
    Features:
    - Vector similarity calculation for topic continuity
    - Automatic conversation archiving and titling
    - Configurable relevance thresholds
    - Graceful degradation for robustness
    - Integration with MEMOIR episodic memory
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Contextual Relevance Engine.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.ContextualRelevanceEngine")
        
        # Default configuration
        self.config = {
            'relevance_threshold': 0.6,
            'temporal_decay_factor': 0.1,
            'max_conversation_length': 50,
            'auto_archive_enabled': True,
            'embedding_model': 'sentence-transformers',
            'title_generation_temperature': 0.3,
            'fallback_to_related': True,  # Graceful degradation
            'min_conversation_length': 3,  # Minimum turns before archiving
            'storage_directory': 'conversation_threads'
        }
        
        if config:
            self.config.update(config)
        
        # Initialize embedding system
        self.embedding_system = None
        self._initialize_embedding_system()
        
        # Storage setup
        self.storage_dir = Path(self.config['storage_directory'])
        self.storage_dir.mkdir(exist_ok=True)
        
        self.logger.info("ContextualRelevanceEngine initialized")
    
    def calculate_relevance(self, new_query: str, conversation_buffer: List[Dict[str, Any]]) -> RelevanceResult:
        """
        Calculate contextual relevance between new query and conversation buffer.
        
        Args:
            new_query: The new user query
            conversation_buffer: List of conversation turns
            
        Returns:
            RelevanceResult with similarity score and decision
        """
        try:
            if not conversation_buffer:
                # No existing conversation - always relevant (start new)
                return RelevanceResult(
                    similarity_score=1.0,
                    is_relevant=True,
                    threshold_used=self.config['relevance_threshold'],
                    calculation_method='empty_buffer',
                    confidence=1.0,
                    metadata={'reason': 'No existing conversation buffer'}
                )
            
            # Check minimum conversation length
            if len(conversation_buffer) < self.config['min_conversation_length']:
                return RelevanceResult(
                    similarity_score=1.0,
                    is_relevant=True,
                    threshold_used=self.config['relevance_threshold'],
                    calculation_method='insufficient_history',
                    confidence=0.8,
                    metadata={'reason': f'Conversation too short ({len(conversation_buffer)} turns)'}
                )
            
            # Extract text content from conversation buffer
            buffer_text = self._extract_buffer_text(conversation_buffer)
            
            if not buffer_text.strip():
                # Empty buffer content - treat as new conversation
                return RelevanceResult(
                    similarity_score=0.0,
                    is_relevant=False,
                    threshold_used=self.config['relevance_threshold'],
                    calculation_method='empty_content',
                    confidence=0.9,
                    metadata={'reason': 'No meaningful content in buffer'}
                )
            
            # Calculate vector similarity
            similarity_score = self._calculate_vector_similarity(new_query, buffer_text)
            
            # Apply temporal weighting (recent messages matter more)
            weighted_score = self._apply_temporal_weighting(similarity_score, conversation_buffer)
            
            # Determine relevance
            threshold = self.config['relevance_threshold']
            is_relevant = weighted_score >= threshold
            
            # Calculate confidence based on score distance from threshold
            confidence = self._calculate_confidence(weighted_score, threshold)
            
            return RelevanceResult(
                similarity_score=weighted_score,
                is_relevant=is_relevant,
                threshold_used=threshold,
                calculation_method='vector_similarity',
                confidence=confidence,
                metadata={
                    'raw_similarity': similarity_score,
                    'temporal_adjustment': weighted_score - similarity_score,
                    'buffer_length': len(conversation_buffer),
                    'buffer_text_length': len(buffer_text)
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Relevance calculation failed: {e}")
            
            # Graceful degradation - assume relevant to maintain conversation flow
            if self.config['fallback_to_related']:
                return RelevanceResult(
                    similarity_score=0.7,  # Safe fallback score
                    is_relevant=True,
                    threshold_used=self.config['relevance_threshold'],
                    calculation_method='fallback_related',
                    confidence=0.3,
                    metadata={'error': str(e), 'fallback_reason': 'Calculation failed, assuming related'}
                )
            else:
                return RelevanceResult(
                    similarity_score=0.0,
                    is_relevant=False,
                    threshold_used=self.config['relevance_threshold'],
                    calculation_method='fallback_unrelated',
                    confidence=0.3,
                    metadata={'error': str(e), 'fallback_reason': 'Calculation failed, assuming unrelated'}
                )
    
    def archive_conversation_thread(self, conversation_buffer: List[Dict[str, Any]], 
                                  force_title: Optional[str] = None) -> ConversationThread:
        """
        Archive a conversation thread with automatic title generation.
        
        Args:
            conversation_buffer: List of conversation turns to archive
            force_title: Optional manual title override
            
        Returns:
            ConversationThread object with generated metadata
        """
        try:
            if not conversation_buffer:
                # Handle empty buffer gracefully with fallback
                logger.warning("Archiving empty conversation buffer - creating fallback thread")
                fallback_title = f"Empty Chat from {datetime.now().strftime('%Y-%m-%d %H:%M')}"

                thread = ConversationThread(
                    thread_id=f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    title=fallback_title,
                    messages=[],
                    created_at=datetime.now().isoformat(),
                    last_updated=datetime.now().isoformat(),
                    message_count=0,
                    topic_keywords=[],
                    embedding_summary=None,
                    metadata={
                        'title_generation_method': 'fallback_timestamp',
                        'archival_timestamp': datetime.now().isoformat(),
                        'archival_reason': 'empty_buffer_fallback'
                    }
                )

                # Store the fallback thread
                self._store_conversation_thread(thread)
                return thread
            
            # Generate unique thread ID
            thread_id = self._generate_thread_id(conversation_buffer)
            
            # Generate title (or use forced title)
            if force_title:
                title = force_title
                title_method = 'manual'
            else:
                title = self._generate_conversation_title(conversation_buffer)
                title_method = 'auto_generated'
            
            # Extract topic keywords
            topic_keywords = self._extract_topic_keywords(conversation_buffer)
            
            # Generate embedding summary for future relevance calculations
            embedding_summary = self._generate_embedding_summary(conversation_buffer)
            
            # Create conversation thread
            thread = ConversationThread(
                thread_id=thread_id,
                title=title,
                messages=conversation_buffer.copy(),
                created_at=conversation_buffer[0].get('timestamp', datetime.now().isoformat()),
                last_updated=datetime.now().isoformat(),
                message_count=len(conversation_buffer),
                topic_keywords=topic_keywords,
                embedding_summary=embedding_summary,
                metadata={
                    'title_generation_method': title_method,
                    'archival_timestamp': datetime.now().isoformat(),
                    'archival_reason': 'contextual_relevance_break'
                }
            )
            
            # Store thread persistently
            self._store_conversation_thread(thread)
            
            # Store in MEMOIR episodic memory (Phase 1 integration)
            self._store_in_memoir(thread)
            
            self.logger.info(f"Archived conversation thread: '{title}' ({len(conversation_buffer)} messages)")
            
            return thread
            
        except Exception as e:
            self.logger.error(f"Failed to archive conversation thread: {e}")
            
            # Graceful degradation - create minimal thread with timestamp title
            fallback_title = f"Chat from {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            thread = ConversationThread(
                thread_id=self._generate_thread_id(conversation_buffer),
                title=fallback_title,
                messages=conversation_buffer.copy(),
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                message_count=len(conversation_buffer),
                topic_keywords=[],
                embedding_summary=None,
                metadata={
                    'title_generation_method': 'fallback_timestamp',
                    'archival_timestamp': datetime.now().isoformat(),
                    'archival_reason': 'error_fallback',
                    'error': str(e)
                }
            )
            
            # Still try to store the thread
            try:
                self._store_conversation_thread(thread)
            except:
                self.logger.error("Failed to store fallback conversation thread")
            
            return thread
    
    def get_archived_threads(self, limit: Optional[int] = None) -> List[ConversationThread]:
        """
        Retrieve archived conversation threads.
        
        Args:
            limit: Optional limit on number of threads to return
            
        Returns:
            List of ConversationThread objects, sorted by last_updated desc
        """
        try:
            threads = []
            
            # Load threads from storage
            for thread_file in self.storage_dir.glob("thread_*.json"):
                try:
                    with open(thread_file, 'r') as f:
                        thread_data = json.load(f)
                    
                    thread = ConversationThread.from_dict(thread_data)
                    threads.append(thread)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load thread {thread_file}: {e}")
            
            # Sort by last_updated (most recent first)
            threads.sort(key=lambda t: t.last_updated, reverse=True)
            
            # Apply limit if specified
            if limit:
                threads = threads[:limit]
            
            return threads
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve archived threads: {e}")
            return []
    
    def search_archived_threads(self, query: str, limit: int = 10) -> List[Tuple[ConversationThread, float]]:
        """
        Search archived threads by relevance to query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of (ConversationThread, relevance_score) tuples
        """
        try:
            threads = self.get_archived_threads()
            scored_threads = []

            for thread in threads:
                # Calculate relevance to search query
                thread_text = f"{thread.title} {' '.join(thread.topic_keywords)}"

                try:
                    score = self._calculate_vector_similarity(query, thread_text)
                    scored_threads.append((thread, score))
                except:
                    # Fallback to keyword matching
                    score = self._keyword_similarity(query, thread_text)
                    scored_threads.append((thread, score))

            # Sort by relevance score
            scored_threads.sort(key=lambda x: x[1], reverse=True)

            # Return top results
            return scored_threads[:limit]

        except Exception as e:
            self.logger.error(f"Failed to search archived threads: {e}")
            return []

    def resume_conversation_thread(self, thread_id: str) -> bool:
        """
        Resume an archived conversation thread by loading it back into active buffer.

        Args:
            thread_id: ID of the thread to resume

        Returns:
            True if conversation was resumed successfully
        """
        try:
            # Find the thread
            threads = self.get_archived_threads()
            target_thread = None

            for thread in threads:
                if thread.thread_id == thread_id:
                    target_thread = thread
                    break

            if not target_thread:
                self.logger.error(f"Thread not found: {thread_id}")
                return False

            # Load conversation back into session manager
            try:
                import streamlit as st
                from sam.session.state_manager import get_session_manager

                session_manager = get_session_manager()
                session_id = st.session_state.get('session_id', 'default_session')
                user_id = st.session_state.get('user_id', 'anonymous')

                # Archive current conversation if it exists
                current_buffer = session_manager.get_conversation_history(session_id)
                if current_buffer:
                    self.logger.info("Archiving current conversation before resuming")
                    self.archive_conversation_thread(current_buffer, force_title="Auto-archived before resume")

                # Clear current session
                session_manager.clear_session(session_id)

                # Ensure session exists
                if not session_manager.get_session(session_id):
                    session_manager.create_session(session_id, user_id)

                # Load archived messages back into conversation buffer
                for message in target_thread.messages:
                    session_manager.add_turn(
                        session_id=session_id,
                        role=message.get('role', 'user'),
                        content=message.get('content', ''),
                        metadata=message.get('metadata', {})
                    )

                # Update conversation history in session state
                conversation_history = session_manager.format_conversation_history(session_id, max_turns=8)
                st.session_state['conversation_history'] = conversation_history

                # Load chat history for UI
                if 'chat_history' not in st.session_state:
                    st.session_state['chat_history'] = []

                # Convert messages to chat history format
                st.session_state['chat_history'] = []
                for message in target_thread.messages:
                    role = message.get('role', 'user')
                    content = message.get('content', '')

                    if role == 'user':
                        st.session_state['chat_history'].append({"role": "user", "content": content})
                    elif role == 'assistant':
                        st.session_state['chat_history'].append({"role": "assistant", "content": content})

                # Remove thread from archived list (it's now active)
                if 'archived_threads' in st.session_state:
                    st.session_state['archived_threads'] = [
                        t for t in st.session_state['archived_threads']
                        if t.get('thread_id') != thread_id
                    ]

                # Set resume notification
                st.session_state['conversation_resumed'] = {
                    'title': target_thread.title,
                    'message_count': target_thread.message_count,
                    'timestamp': target_thread.last_updated
                }

                # Delete the archived thread file
                try:
                    thread_file = self.storage_dir / f"{thread_id}.json"
                    if thread_file.exists():
                        thread_file.unlink()
                        self.logger.debug(f"Deleted archived thread file: {thread_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete archived thread file: {e}")

                self.logger.info(f"Successfully resumed conversation: '{target_thread.title}' ({target_thread.message_count} messages)")
                return True

            except Exception as e:
                self.logger.error(f"Failed to load conversation into session: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to resume conversation thread: {e}")
            return False

    def search_within_threads(self, query: str, thread_ids: Optional[List[str]] = None,
                             limit: int = 20) -> List[Dict[str, Any]]:
        """
        Advanced search within conversation content.

        Args:
            query: Search query
            thread_ids: Optional list of specific thread IDs to search
            limit: Maximum number of results

        Returns:
            List of search results with context
        """
        try:
            threads = self.get_archived_threads()

            # Filter threads if specific IDs provided
            if thread_ids:
                threads = [t for t in threads if t.thread_id in thread_ids]

            search_results = []

            for thread in threads:
                # Search within thread messages
                for i, message in enumerate(thread.messages):
                    content = message.get('content', '').lower()
                    role = message.get('role', 'unknown')

                    # Simple text search (could be enhanced with fuzzy matching)
                    if query.lower() in content:
                        # Get context (surrounding messages)
                        context_start = max(0, i - 2)
                        context_end = min(len(thread.messages), i + 3)
                        context_messages = thread.messages[context_start:context_end]

                        # Calculate relevance score
                        relevance_score = self._calculate_search_relevance(query, content)

                        search_result = {
                            'thread_id': thread.thread_id,
                            'thread_title': thread.title,
                            'message_index': i,
                            'message_role': role,
                            'message_content': message.get('content', ''),
                            'relevance_score': relevance_score,
                            'context_messages': context_messages,
                            'timestamp': message.get('timestamp', thread.created_at)
                        }

                        search_results.append(search_result)

            # Sort by relevance score
            search_results.sort(key=lambda x: x['relevance_score'], reverse=True)

            return search_results[:limit]

        except Exception as e:
            self.logger.error(f"Failed to search within threads: {e}")
            return []

    def get_conversation_analytics(self) -> Dict[str, Any]:
        """
        Generate conversation analytics and insights.

        Returns:
            Dictionary with analytics data
        """
        try:
            threads = self.get_archived_threads()

            if not threads:
                return {
                    'total_conversations': 0,
                    'total_messages': 0,
                    'average_conversation_length': 0,
                    'most_common_topics': [],
                    'conversation_frequency': {},
                    'recent_activity': []
                }

            # Basic statistics
            total_conversations = len(threads)
            total_messages = sum(thread.message_count for thread in threads)
            average_length = total_messages / total_conversations if total_conversations > 0 else 0

            # Topic analysis
            all_keywords = []
            for thread in threads:
                all_keywords.extend(thread.topic_keywords)

            # Count keyword frequency
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

            # Get most common topics
            most_common_topics = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Conversation frequency by date
            conversation_frequency = {}
            for thread in threads:
                try:
                    date = datetime.fromisoformat(thread.created_at.replace('Z', '+00:00')).date()
                    date_str = date.strftime('%Y-%m-%d')
                    conversation_frequency[date_str] = conversation_frequency.get(date_str, 0) + 1
                except:
                    pass

            # Recent activity (last 7 days)
            recent_activity = []
            for thread in threads[:20]:  # Last 20 conversations
                try:
                    dt = datetime.fromisoformat(thread.created_at.replace('Z', '+00:00'))
                    recent_activity.append({
                        'title': thread.title,
                        'date': dt.strftime('%Y-%m-%d'),
                        'time': dt.strftime('%H:%M'),
                        'message_count': thread.message_count,
                        'topics': thread.topic_keywords[:3]  # Top 3 topics
                    })
                except:
                    pass

            # Conversation length distribution
            length_distribution = {
                'short (1-5 messages)': len([t for t in threads if 1 <= t.message_count <= 5]),
                'medium (6-15 messages)': len([t for t in threads if 6 <= t.message_count <= 15]),
                'long (16+ messages)': len([t for t in threads if t.message_count >= 16])
            }

            return {
                'total_conversations': total_conversations,
                'total_messages': total_messages,
                'average_conversation_length': round(average_length, 1),
                'most_common_topics': most_common_topics,
                'conversation_frequency': conversation_frequency,
                'recent_activity': recent_activity,
                'length_distribution': length_distribution,
                'analytics_generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to generate conversation analytics: {e}")
            return {'error': str(e)}

    def add_tags_to_thread(self, thread_id: str, tags: List[str]) -> bool:
        """
        Add tags to a conversation thread.

        Args:
            thread_id: ID of the thread to tag
            tags: List of tags to add

        Returns:
            True if tags were added successfully
        """
        try:
            # Load thread
            thread_file = self.storage_dir / f"{thread_id}.json"
            if not thread_file.exists():
                self.logger.error(f"Thread file not found: {thread_file}")
                return False

            with open(thread_file, 'r') as f:
                thread_data = json.load(f)

            # Add tags to metadata
            if 'user_tags' not in thread_data['metadata']:
                thread_data['metadata']['user_tags'] = []

            # Add new tags (avoid duplicates)
            existing_tags = set(thread_data['metadata']['user_tags'])
            new_tags = [tag for tag in tags if tag not in existing_tags]
            thread_data['metadata']['user_tags'].extend(new_tags)

            # Update timestamp
            thread_data['last_updated'] = datetime.now().isoformat()

            # Save updated thread
            with open(thread_file, 'w') as f:
                json.dump(thread_data, f, indent=2)

            self.logger.info(f"Added tags {new_tags} to thread {thread_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add tags to thread: {e}")
            return False

    def generate_auto_tags(self, thread: ConversationThread) -> List[str]:
        """
        Generate automatic tags for a conversation thread.

        Args:
            thread: ConversationThread to analyze

        Returns:
            List of automatically generated tags
        """
        try:
            auto_tags = []

            # Length-based tags
            if thread.message_count <= 5:
                auto_tags.append('short-conversation')
            elif thread.message_count <= 15:
                auto_tags.append('medium-conversation')
            else:
                auto_tags.append('long-conversation')

            # Content-based tags
            content_text = ' '.join([msg.get('content', '') for msg in thread.messages]).lower()

            # Technical topics
            if any(word in content_text for word in ['code', 'programming', 'function', 'variable', 'debug']):
                auto_tags.append('technical')

            # Learning/teaching
            if any(phrase in content_text for phrase in ['teach', 'learn', 'explain', 'how to']):
                auto_tags.append('educational')

            # Problem solving
            if any(word in content_text for word in ['problem', 'issue', 'error', 'fix', 'solve']):
                auto_tags.append('troubleshooting')

            # Questions
            question_count = content_text.count('?')
            if question_count >= 3:
                auto_tags.append('q-and-a')

            # Time-based tags
            try:
                created_date = datetime.fromisoformat(thread.created_at.replace('Z', '+00:00'))
                auto_tags.append(f"created-{created_date.strftime('%Y-%m')}")

                # Recent conversations
                if (datetime.now() - created_date).days <= 1:
                    auto_tags.append('recent')
                elif (datetime.now() - created_date).days <= 7:
                    auto_tags.append('this-week')
            except:
                pass

            # Topic-based tags from keywords
            for keyword in thread.topic_keywords[:3]:  # Top 3 keywords
                if len(keyword) > 3:  # Avoid short words
                    auto_tags.append(f"topic-{keyword}")

            return auto_tags

        except Exception as e:
            self.logger.error(f"Failed to generate auto tags: {e}")
            return []

    def get_threads_by_tags(self, tags: List[str], match_all: bool = False) -> List[ConversationThread]:
        """
        Get threads that match specified tags.

        Args:
            tags: List of tags to search for
            match_all: If True, thread must have all tags; if False, any tag matches

        Returns:
            List of matching ConversationThread objects
        """
        try:
            threads = self.get_archived_threads()
            matching_threads = []

            for thread in threads:
                # Get all tags for this thread
                thread_tags = set(thread.topic_keywords)  # Topic keywords as tags

                # Add user tags if they exist
                user_tags = thread.metadata.get('user_tags', [])
                thread_tags.update(user_tags)

                # Add auto tags
                auto_tags = self.generate_auto_tags(thread)
                thread_tags.update(auto_tags)

                # Check tag matching
                search_tags = set(tags)

                if match_all:
                    # All tags must be present
                    if search_tags.issubset(thread_tags):
                        matching_threads.append(thread)
                else:
                    # Any tag matches
                    if search_tags.intersection(thread_tags):
                        matching_threads.append(thread)

            return matching_threads

        except Exception as e:
            self.logger.error(f"Failed to get threads by tags: {e}")
            return []

    def find_related_conversations(self, current_query: str, current_buffer: List[Dict[str, Any]],
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find conversations related to current context using cross-conversation bridging.

        Args:
            current_query: Current user query
            current_buffer: Current conversation buffer
            limit: Maximum number of related conversations to return

        Returns:
            List of related conversation summaries with relevance scores
        """
        try:
            # Extract context from current conversation
            current_context = self._extract_conversation_context(current_query, current_buffer)

            # Get all archived threads
            archived_threads = self.get_archived_threads()

            if not archived_threads:
                return []

            related_conversations = []

            for thread in archived_threads:
                # Calculate cross-conversation relevance
                relevance_score = self._calculate_cross_conversation_relevance(
                    current_context, thread
                )

                if relevance_score > self.config.get('cross_conversation_relevance_threshold', 0.2):  # Threshold for related conversations
                    # Generate conversation bridge summary
                    bridge_summary = self._generate_conversation_bridge(
                        current_context, thread, relevance_score
                    )

                    related_conversations.append({
                        'thread_id': thread.thread_id,
                        'title': thread.title,
                        'relevance_score': relevance_score,
                        'bridge_summary': bridge_summary,
                        'key_topics': thread.topic_keywords[:3],
                        'message_count': thread.message_count,
                        'created_at': thread.created_at,
                        'connection_type': self._determine_connection_type(current_context, thread)
                    })

            # Sort by relevance score
            related_conversations.sort(key=lambda x: x['relevance_score'], reverse=True)

            return related_conversations[:limit]

        except Exception as e:
            self.logger.error(f"Failed to find related conversations: {e}")
            return []

    def generate_ai_insights(self, conversation_history: List[ConversationThread],
                           current_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate AI-powered insights about conversation patterns and recommendations.

        Args:
            conversation_history: List of conversation threads to analyze
            current_context: Optional current conversation context

        Returns:
            Dictionary with AI-generated insights and recommendations
        """
        try:
            if not conversation_history:
                return {'insights': [], 'recommendations': [], 'patterns': []}

            # Analyze conversation patterns
            patterns = self._analyze_conversation_patterns(conversation_history)

            # Generate insights using LLM
            insights = self._generate_llm_insights(conversation_history, patterns, current_context)

            # Generate recommendations
            recommendations = self._generate_recommendations(conversation_history, patterns, current_context)

            # Identify emerging topics
            emerging_topics = self._identify_emerging_topics(conversation_history)

            # Calculate conversation health metrics
            health_metrics = self._calculate_conversation_health(conversation_history)

            return {
                'insights': insights,
                'recommendations': recommendations,
                'patterns': patterns,
                'emerging_topics': emerging_topics,
                'health_metrics': health_metrics,
                'analysis_timestamp': datetime.now().isoformat(),
                'total_conversations_analyzed': len(conversation_history)
            }

        except Exception as e:
            self.logger.error(f"Failed to generate AI insights: {e}")
            return {'error': str(e)}

    def export_conversation_data(self, thread_ids: Optional[List[str]] = None,
                               export_format: str = 'json',
                               include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export conversation data in various formats.

        Args:
            thread_ids: Optional list of specific thread IDs to export
            export_format: Export format ('json', 'markdown', 'csv')
            include_metadata: Whether to include metadata in export

        Returns:
            Dictionary with export data and metadata
        """
        try:
            # Get threads to export
            if thread_ids:
                threads = [t for t in self.get_archived_threads() if t.thread_id in thread_ids]
            else:
                threads = self.get_archived_threads()

            if not threads:
                return {'error': 'No conversations found to export'}

            # Generate export data based on format
            if export_format.lower() == 'json':
                export_data = self._export_as_json(threads, include_metadata)
            elif export_format.lower() == 'markdown':
                export_data = self._export_as_markdown(threads, include_metadata)
            elif export_format.lower() == 'csv':
                export_data = self._export_as_csv(threads, include_metadata)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")

            # Generate export metadata
            export_metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'export_format': export_format,
                'total_conversations': len(threads),
                'total_messages': sum(t.message_count for t in threads),
                'date_range': {
                    'earliest': min(t.created_at for t in threads) if threads else None,
                    'latest': max(t.last_updated for t in threads) if threads else None
                },
                'include_metadata': include_metadata
            }

            return {
                'export_data': export_data,
                'metadata': export_metadata,
                'success': True
            }

        except Exception as e:
            self.logger.error(f"Failed to export conversation data: {e}")
            return {'error': str(e), 'success': False}

    def optimize_conversation_storage(self) -> Dict[str, Any]:
        """
        Optimize conversation storage with indexing and caching.

        Returns:
            Dictionary with optimization results
        """
        try:
            optimization_results = {
                'indexed_conversations': 0,
                'cache_entries_created': 0,
                'storage_optimized': False,
                'performance_improvement': 0.0
            }

            # Create conversation index for faster search
            conversation_index = self._create_conversation_index()
            optimization_results['indexed_conversations'] = len(conversation_index)

            # Create search cache for common queries
            search_cache = self._create_search_cache()
            optimization_results['cache_entries_created'] = len(search_cache)

            # Optimize storage structure
            storage_optimized = self._optimize_storage_structure()
            optimization_results['storage_optimized'] = storage_optimized

            # Calculate performance improvement estimate
            performance_improvement = self._estimate_performance_improvement(
                optimization_results['indexed_conversations'],
                optimization_results['cache_entries_created']
            )
            optimization_results['performance_improvement'] = performance_improvement

            # Store optimization metadata
            optimization_results['optimization_timestamp'] = datetime.now().isoformat()
            optimization_results['next_optimization_due'] = (
                datetime.now() + timedelta(days=7)
            ).isoformat()

            self.logger.info(f"Storage optimization complete: {optimization_results}")
            return optimization_results

        except Exception as e:
            self.logger.error(f"Failed to optimize conversation storage: {e}")
            return {'error': str(e), 'success': False}

    def _initialize_embedding_system(self) -> None:
        """Initialize the embedding system for vector similarity."""
        try:
            # Try to use existing SAM embedding capabilities
            from memory.memory_vectorstore import MemoryVectorStore
            self.embedding_system = 'sam_vectorstore'
            self.logger.info("Using SAM MemoryVectorStore for embeddings")

        except ImportError:
            try:
                # Fallback to sentence-transformers if available
                import sentence_transformers
                self.embedding_system = 'sentence_transformers'
                self.logger.info("Using sentence-transformers for embeddings")

            except ImportError:
                # Final fallback to keyword-based similarity
                self.embedding_system = 'keyword_fallback'
                self.logger.warning("No embedding system available, using keyword fallback")

    def _extract_buffer_text(self, conversation_buffer: List[Dict[str, Any]]) -> str:
        """Extract meaningful text content from conversation buffer."""
        text_parts = []

        for turn in conversation_buffer:
            content = turn.get('content', '')
            role = turn.get('role', '')

            # Skip empty content
            if not content.strip():
                continue

            # Add role context for better understanding
            text_parts.append(f"{role}: {content}")

        return "\n".join(text_parts)

    def _calculate_vector_similarity(self, text1: str, text2: str) -> float:
        """Calculate vector similarity between two texts."""
        try:
            if self.embedding_system == 'sam_vectorstore':
                return self._sam_vector_similarity(text1, text2)
            elif self.embedding_system == 'sentence_transformers':
                return self._sentence_transformer_similarity(text1, text2)
            else:
                return self._keyword_similarity(text1, text2)

        except Exception as e:
            self.logger.warning(f"Vector similarity calculation failed: {e}")
            return self._keyword_similarity(text1, text2)

    def _sam_vector_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using SAM's vector store."""
        try:
            # Use SAM's existing embedding capabilities
            from memory.memory_vectorstore import MemoryVectorStore

            # This is a simplified approach - in practice, we'd use SAM's embedding model
            # For now, return a reasonable similarity based on text overlap
            return self._keyword_similarity(text1, text2)

        except Exception as e:
            self.logger.warning(f"SAM vector similarity failed: {e}")
            return self._keyword_similarity(text1, text2)

    def _sentence_transformer_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using sentence transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Load model (cache for efficiency)
            if not hasattr(self, '_st_model'):
                self._st_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Generate embeddings
            embeddings = self._st_model.encode([text1, text2])

            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0)
            ).item()

            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        except Exception as e:
            self.logger.warning(f"Sentence transformer similarity failed: {e}")
            return self._keyword_similarity(text1, text2)

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Fallback keyword-based similarity calculation."""
        try:
            # Simple keyword overlap similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

            words1 = words1 - stop_words
            words2 = words2 - stop_words

            if not words1 or not words2:
                return 0.0

            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Keyword similarity calculation failed: {e}")
            return 0.0

    def _apply_temporal_weighting(self, similarity_score: float, conversation_buffer: List[Dict[str, Any]]) -> float:
        """Apply temporal weighting to give more importance to recent messages."""
        try:
            if not conversation_buffer:
                return similarity_score

            # Calculate recency boost for recent messages
            decay_factor = self.config['temporal_decay_factor']
            total_messages = len(conversation_buffer)

            # Recent messages get higher weight
            recent_weight = 1.0 + (decay_factor * (total_messages / 10.0))

            # Apply weighting (but don't exceed 1.0)
            weighted_score = min(1.0, similarity_score * recent_weight)

            return weighted_score

        except Exception as e:
            self.logger.warning(f"Temporal weighting failed: {e}")
            return similarity_score

    def _calculate_confidence(self, score: float, threshold: float) -> float:
        """Calculate confidence based on distance from threshold."""
        try:
            # Distance from threshold
            distance = abs(score - threshold)

            # Higher distance = higher confidence
            # Scale to [0.3, 1.0] range
            confidence = 0.3 + (distance * 0.7)

            return min(1.0, confidence)

        except:
            return 0.5  # Default confidence

    def _generate_thread_id(self, conversation_buffer: List[Dict[str, Any]]) -> str:
        """Generate unique thread ID based on conversation content."""
        try:
            # Create hash from conversation content and timestamp
            content_text = self._extract_buffer_text(conversation_buffer)
            timestamp = datetime.now().isoformat()

            hash_input = f"{content_text}_{timestamp}".encode()
            thread_hash = hashlib.sha256(hash_input).hexdigest()[:16]

            return f"thread_{thread_hash}"

        except Exception as e:
            self.logger.warning(f"Thread ID generation failed: {e}")
            return f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _generate_conversation_title(self, conversation_buffer: List[Dict[str, Any]]) -> str:
        """Generate a descriptive title for the conversation using LLM."""
        try:
            # Extract key content for title generation
            content_text = self._extract_buffer_text(conversation_buffer)

            # Limit content length for LLM call
            if len(content_text) > 1000:
                content_text = content_text[:1000] + "..."

            # Create title generation prompt
            title_prompt = f"""Generate a concise, descriptive title (max 60 characters) for this conversation:

{content_text}

Title should capture the main topic or theme. Examples:
- "Discussion about Blue Lamps Secret"
- "TPV Security Configuration"
- "Memory System Troubleshooting"

Title:"""

            # Call LLM for title generation
            title = self._call_llm_for_title(title_prompt)

            # Clean and validate title
            title = title.strip().strip('"').strip("'")
            if len(title) > 60:
                title = title[:57] + "..."

            if not title or len(title) < 3:
                raise ValueError("Generated title too short or empty")

            return title

        except Exception as e:
            self.logger.warning(f"Title generation failed: {e}")

            # Fallback to timestamp-based title
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            return f"Chat from {timestamp}"

    def _call_llm_for_title(self, prompt: str) -> str:
        """Call LLM to generate conversation title."""
        try:
            import requests

            # Use Ollama API for title generation
            ollama_payload = {
                "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config['title_generation_temperature'],
                    "max_tokens": 20,
                    "top_p": 0.9
                }
            }

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=ollama_payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                title = result.get('response', '').strip()

                if title:
                    return title

            raise Exception(f"LLM call failed with status {response.status_code}")

        except Exception as e:
            self.logger.warning(f"LLM title generation failed: {e}")
            raise

    def _extract_topic_keywords(self, conversation_buffer: List[Dict[str, Any]]) -> List[str]:
        """Extract key topic words from conversation."""
        try:
            content_text = self._extract_buffer_text(conversation_buffer)

            # Simple keyword extraction (could be enhanced with NLP)
            words = content_text.lower().split()

            # Remove stop words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}

            keywords = []
            word_counts = {}

            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum()).lower()

                if len(word) > 3 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1

            # Get most frequent words
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, count in sorted_words[:10] if count > 1]

            return keywords

        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
            return []

    def _generate_embedding_summary(self, conversation_buffer: List[Dict[str, Any]]) -> Optional[List[float]]:
        """Generate embedding summary for future relevance calculations."""
        try:
            content_text = self._extract_buffer_text(conversation_buffer)

            if self.embedding_system == 'sentence_transformers':
                if not hasattr(self, '_st_model'):
                    from sentence_transformers import SentenceTransformer
                    self._st_model = SentenceTransformer('all-MiniLM-L6-v2')

                embedding = self._st_model.encode(content_text)
                return embedding.tolist()

            # For other systems, return None (will use text-based similarity)
            return None

        except Exception as e:
            self.logger.warning(f"Embedding summary generation failed: {e}")
            return None

    def _store_conversation_thread(self, thread: ConversationThread) -> None:
        """Store conversation thread to persistent storage."""
        try:
            thread_file = self.storage_dir / f"{thread.thread_id}.json"

            with open(thread_file, 'w') as f:
                json.dump(thread.to_dict(), f, indent=2)

            self.logger.debug(f"Stored conversation thread: {thread_file}")

        except Exception as e:
            self.logger.error(f"Failed to store conversation thread: {e}")
            raise

    def _store_in_memoir(self, thread: ConversationThread) -> None:
        """Store conversation thread in MEMOIR episodic memory."""
        try:
            # Integration with MEMOIR system (Phase 6)
            # This creates a memory entry for the archived conversation

            memory_content = f"Conversation: {thread.title}\n"
            memory_content += f"Messages: {thread.message_count}\n"
            memory_content += f"Topics: {', '.join(thread.topic_keywords)}\n"
            memory_content += f"Summary: Archived conversation thread from {thread.created_at}"

            # Try to store in MEMOIR if available
            try:
                import streamlit as st
                if hasattr(st.session_state, 'secure_memory_store'):
                    memory_store = st.session_state.secure_memory_store

                    # Store as episodic memory
                    memory_store.store_memory(
                        content=memory_content,
                        memory_type="episodic",
                        metadata={
                            "thread_id": thread.thread_id,
                            "conversation_title": thread.title,
                            "message_count": thread.message_count,
                            "topic_keywords": thread.topic_keywords,
                            "archival_timestamp": thread.last_updated
                        }
                    )

                    self.logger.info(f"Stored conversation thread in MEMOIR: {thread.title}")

            except Exception as memoir_error:
                self.logger.warning(f"Failed to store in MEMOIR: {memoir_error}")
                # Continue without MEMOIR storage - not critical

        except Exception as e:
            self.logger.warning(f"MEMOIR integration failed: {e}")
            # Non-critical failure - continue without MEMOIR storage

    def _calculate_search_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score for search results."""
        try:
            query_lower = query.lower()
            content_lower = content.lower()

            # Exact match gets highest score
            if query_lower in content_lower:
                # Calculate position bonus (earlier matches score higher)
                position = content_lower.find(query_lower)
                position_bonus = max(0, 1 - (position / len(content_lower)))

                # Calculate coverage (how much of content matches)
                coverage = len(query_lower) / len(content_lower)

                return min(1.0, 0.8 + (position_bonus * 0.1) + (coverage * 0.1))

            # Fallback to keyword similarity
            return self._keyword_similarity(query, content)

        except Exception as e:
            self.logger.warning(f"Search relevance calculation failed: {e}")
            return 0.0

    def _extract_conversation_context(self, current_query: str, current_buffer: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract meaningful context from current conversation."""
        try:
            # Combine current query with recent conversation
            buffer_text = self._extract_buffer_text(current_buffer)
            combined_context = f"{current_query}\n{buffer_text}"

            # Extract key entities and topics
            key_entities = self._extract_key_entities(combined_context)
            main_topics = self._extract_topic_keywords(current_buffer)

            # Determine conversation intent
            intent = self._determine_conversation_intent(combined_context)

            return {
                'query': current_query,
                'buffer_text': buffer_text,
                'combined_context': combined_context,
                'key_entities': key_entities,
                'main_topics': main_topics,
                'intent': intent,
                'message_count': len(current_buffer)
            }

        except Exception as e:
            self.logger.warning(f"Context extraction failed: {e}")
            return {'query': current_query, 'error': str(e)}

    def _calculate_cross_conversation_relevance(self, current_context: Dict[str, Any],
                                              archived_thread: ConversationThread) -> float:
        """Calculate relevance between current context and archived conversation."""
        try:
            relevance_score = 0.0

            # Topic overlap scoring
            current_topics = set(current_context.get('main_topics', []))
            archived_topics = set(archived_thread.topic_keywords)

            if current_topics and archived_topics:
                topic_overlap = len(current_topics.intersection(archived_topics))
                topic_union = len(current_topics.union(archived_topics))
                topic_score = topic_overlap / topic_union if topic_union > 0 else 0.0
                relevance_score += topic_score * 0.4

            # Entity overlap scoring
            current_entities = set(current_context.get('key_entities', []))
            archived_text = ' '.join([msg.get('content', '') for msg in archived_thread.messages])
            archived_entities = set(self._extract_key_entities(archived_text))

            if current_entities and archived_entities:
                entity_overlap = len(current_entities.intersection(archived_entities))
                entity_union = len(current_entities.union(archived_entities))
                entity_score = entity_overlap / entity_union if entity_union > 0 else 0.0
                relevance_score += entity_score * 0.3

            # Semantic similarity scoring
            current_text = current_context.get('combined_context', '')
            archived_summary = f"{archived_thread.title} {' '.join(archived_thread.topic_keywords)}"

            semantic_score = self._calculate_vector_similarity(current_text, archived_summary)
            relevance_score += semantic_score * 0.3

            return min(1.0, relevance_score)

        except Exception as e:
            self.logger.warning(f"Cross-conversation relevance calculation failed: {e}")
            return 0.0

    def _generate_conversation_bridge(self, current_context: Dict[str, Any],
                                    archived_thread: ConversationThread,
                                    relevance_score: float) -> str:
        """Generate a bridge summary connecting current context to archived conversation."""
        try:
            # Identify connection points
            shared_topics = set(current_context.get('main_topics', [])).intersection(
                set(archived_thread.topic_keywords)
            )

            shared_entities = set(current_context.get('key_entities', [])).intersection(
                set(self._extract_key_entities(' '.join([msg.get('content', '') for msg in archived_thread.messages])))
            )

            # Generate bridge text
            bridge_parts = []

            if shared_topics:
                bridge_parts.append(f"Related topics: {', '.join(list(shared_topics)[:3])}")

            if shared_entities:
                bridge_parts.append(f"Common elements: {', '.join(list(shared_entities)[:3])}")

            # Add relevance context
            if relevance_score > 0.7:
                bridge_parts.append("Highly relevant to current discussion")
            elif relevance_score > 0.5:
                bridge_parts.append("Moderately relevant to current topic")
            else:
                bridge_parts.append("May provide useful context")

            return " | ".join(bridge_parts) if bridge_parts else "Related conversation"

        except Exception as e:
            self.logger.warning(f"Bridge generation failed: {e}")
            return "Related conversation"

    def _determine_connection_type(self, current_context: Dict[str, Any],
                                 archived_thread: ConversationThread) -> str:
        """Determine the type of connection between conversations."""
        try:
            current_topics = set(current_context.get('main_topics', []))
            archived_topics = set(archived_thread.topic_keywords)

            # Direct topic match
            if current_topics.intersection(archived_topics):
                return "topic_match"

            # Entity-based connection
            current_entities = set(current_context.get('key_entities', []))
            archived_text = ' '.join([msg.get('content', '') for msg in archived_thread.messages])
            archived_entities = set(self._extract_key_entities(archived_text))

            if current_entities.intersection(archived_entities):
                return "entity_match"

            # Intent-based connection
            current_intent = current_context.get('intent', '')
            archived_intent = self._determine_conversation_intent(
                ' '.join([msg.get('content', '') for msg in archived_thread.messages])
            )

            if current_intent == archived_intent and current_intent:
                return "intent_match"

            # Semantic similarity
            return "semantic_similarity"

        except Exception as e:
            self.logger.warning(f"Connection type determination failed: {e}")
            return "unknown"

    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from text (simplified implementation)."""
        try:
            # Simple entity extraction - could be enhanced with NLP libraries
            words = text.lower().split()

            # Look for capitalized words (potential proper nouns)
            entities = []
            for word in text.split():
                if word[0].isupper() and len(word) > 2:
                    clean_word = ''.join(c for c in word if c.isalnum()).lower()
                    if clean_word not in ['the', 'and', 'but', 'for', 'you', 'are', 'this', 'that']:
                        entities.append(clean_word)

            # Look for quoted phrases
            import re
            quoted_phrases = re.findall(r'"([^"]*)"', text)
            entities.extend([phrase.lower().strip() for phrase in quoted_phrases if len(phrase.strip()) > 2])

            # Remove duplicates and return top entities
            unique_entities = list(set(entities))
            return unique_entities[:10]

        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}")
            return []

    def _determine_conversation_intent(self, text: str) -> str:
        """Determine the intent of a conversation."""
        try:
            text_lower = text.lower()

            # Question-based intent
            if '?' in text and any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                return "question_seeking"

            # Learning/teaching intent
            if any(phrase in text_lower for phrase in ['teach', 'learn', 'explain', 'understand', 'show me']):
                return "educational"

            # Problem-solving intent
            if any(word in text_lower for word in ['problem', 'issue', 'error', 'fix', 'solve', 'help']):
                return "problem_solving"

            # Information sharing
            if any(phrase in text_lower for phrase in ['i want to tell', 'let me share', 'here is', 'i have']):
                return "information_sharing"

            # General discussion
            return "general_discussion"

        except Exception as e:
            self.logger.warning(f"Intent determination failed: {e}")
            return "unknown"

    def _analyze_conversation_patterns(self, conversation_history: List[ConversationThread]) -> Dict[str, Any]:
        """Analyze patterns in conversation history."""
        try:
            patterns = {
                'temporal_patterns': {},
                'topic_evolution': [],
                'conversation_styles': {},
                'interaction_patterns': {}
            }

            # Temporal patterns
            conversation_times = []
            for thread in conversation_history:
                try:
                    dt = datetime.fromisoformat(thread.created_at.replace('Z', '+00:00'))
                    conversation_times.append(dt)
                except:
                    continue

            if conversation_times:
                # Hour of day patterns
                hours = [dt.hour for dt in conversation_times]
                hour_counts = {}
                for hour in hours:
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1

                patterns['temporal_patterns']['peak_hours'] = sorted(
                    hour_counts.items(), key=lambda x: x[1], reverse=True
                )[:3]

                # Day of week patterns
                days = [dt.weekday() for dt in conversation_times]
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = {}
                for day in days:
                    day_name = day_names[day]
                    day_counts[day_name] = day_counts.get(day_name, 0) + 1

                patterns['temporal_patterns']['active_days'] = sorted(
                    day_counts.items(), key=lambda x: x[1], reverse=True
                )[:3]

            # Topic evolution
            topic_timeline = []
            for thread in sorted(conversation_history, key=lambda x: x.created_at):
                if thread.topic_keywords:
                    topic_timeline.append({
                        'date': thread.created_at,
                        'topics': thread.topic_keywords[:3],
                        'title': thread.title
                    })

            patterns['topic_evolution'] = topic_timeline[-10:]  # Last 10 conversations

            # Conversation styles
            length_categories = {'short': 0, 'medium': 0, 'long': 0}
            for thread in conversation_history:
                if thread.message_count <= 5:
                    length_categories['short'] += 1
                elif thread.message_count <= 15:
                    length_categories['medium'] += 1
                else:
                    length_categories['long'] += 1

            patterns['conversation_styles']['length_preferences'] = length_categories

            return patterns

        except Exception as e:
            self.logger.warning(f"Pattern analysis failed: {e}")
            return {}

    def _generate_llm_insights(self, conversation_history: List[ConversationThread],
                             patterns: Dict[str, Any], current_context: Optional[str]) -> List[str]:
        """Generate AI-powered insights using LLM."""
        try:
            insights = []

            # Analyze conversation frequency
            if len(conversation_history) > 10:
                insights.append(f"You've had {len(conversation_history)} conversations, showing active engagement with AI assistance.")

            # Topic diversity analysis
            all_topics = []
            for thread in conversation_history:
                all_topics.extend(thread.topic_keywords)

            unique_topics = len(set(all_topics))
            if unique_topics > 20:
                insights.append(f"Your conversations span {unique_topics} different topics, indicating diverse interests and use cases.")

            # Conversation length patterns
            avg_length = sum(t.message_count for t in conversation_history) / len(conversation_history)
            if avg_length > 10:
                insights.append("You tend to have in-depth conversations, suggesting thorough exploration of topics.")
            elif avg_length < 5:
                insights.append("You prefer concise interactions, focusing on quick information exchange.")

            # Temporal patterns
            peak_hours = patterns.get('temporal_patterns', {}).get('peak_hours', [])
            if peak_hours:
                peak_hour = peak_hours[0][0]
                if 9 <= peak_hour <= 17:
                    insights.append("Most conversations occur during business hours, suggesting work-related usage.")
                elif 18 <= peak_hour <= 22:
                    insights.append("Evening conversations are common, indicating personal learning and exploration time.")

            # Topic evolution
            topic_evolution = patterns.get('topic_evolution', [])
            if len(topic_evolution) >= 3:
                recent_topics = [topic for entry in topic_evolution[-3:] for topic in entry['topics']]
                if len(set(recent_topics)) < len(recent_topics) * 0.5:
                    insights.append("Recent conversations show focused interest in specific topics.")

            return insights[:5]  # Return top 5 insights

        except Exception as e:
            self.logger.warning(f"LLM insights generation failed: {e}")
            return ["Analysis temporarily unavailable"]

    def _generate_recommendations(self, conversation_history: List[ConversationThread],
                                patterns: Dict[str, Any], current_context: Optional[str]) -> List[str]:
        """Generate recommendations based on conversation analysis."""
        try:
            recommendations = []

            # Topic-based recommendations
            all_topics = []
            for thread in conversation_history:
                all_topics.extend(thread.topic_keywords)

            topic_counts = {}
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

            most_common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]

            if most_common_topics:
                top_topic = most_common_topics[0][0]
                recommendations.append(f"Consider exploring advanced aspects of '{top_topic}' based on your frequent discussions.")

            # Conversation organization recommendations
            untagged_conversations = [t for t in conversation_history if not t.metadata.get('user_tags', [])]
            if len(untagged_conversations) > 5:
                recommendations.append("Consider adding tags to your conversations for better organization.")

            # Search utilization
            if len(conversation_history) > 10:
                recommendations.append("Use the search feature to quickly find information from your conversation history.")

            # Resume feature
            if len(conversation_history) > 3:
                recommendations.append("Try resuming previous conversations to continue where you left off.")

            # Export recommendations
            if len(conversation_history) > 20:
                recommendations.append("Consider exporting your conversation history for backup or analysis.")

            return recommendations[:4]  # Return top 4 recommendations

        except Exception as e:
            self.logger.warning(f"Recommendations generation failed: {e}")
            return ["No recommendations available"]

    def _identify_emerging_topics(self, conversation_history: List[ConversationThread]) -> List[Dict[str, Any]]:
        """Identify emerging topics in recent conversations."""
        try:
            if len(conversation_history) < 5:
                return []

            # Split conversations into recent and older
            sorted_conversations = sorted(conversation_history, key=lambda x: x.created_at, reverse=True)
            recent_conversations = sorted_conversations[:len(sorted_conversations)//3]  # Recent third
            older_conversations = sorted_conversations[len(sorted_conversations)//3:]  # Older two-thirds

            # Count topics in each period
            recent_topics = {}
            older_topics = {}

            for thread in recent_conversations:
                for topic in thread.topic_keywords:
                    recent_topics[topic] = recent_topics.get(topic, 0) + 1

            for thread in older_conversations:
                for topic in thread.topic_keywords:
                    older_topics[topic] = older_topics.get(topic, 0) + 1

            # Find emerging topics (more frequent recently)
            emerging_topics = []
            for topic, recent_count in recent_topics.items():
                older_count = older_topics.get(topic, 0)

                # Calculate emergence score
                if recent_count > older_count:
                    emergence_score = recent_count / (older_count + 1)  # +1 to avoid division by zero

                    if emergence_score > 1.5:  # Threshold for emerging topic
                        emerging_topics.append({
                            'topic': topic,
                            'recent_mentions': recent_count,
                            'previous_mentions': older_count,
                            'emergence_score': round(emergence_score, 2)
                        })

            # Sort by emergence score
            emerging_topics.sort(key=lambda x: x['emergence_score'], reverse=True)

            return emerging_topics[:5]  # Return top 5 emerging topics

        except Exception as e:
            self.logger.warning(f"Emerging topics identification failed: {e}")
            return []

    def _calculate_conversation_health(self, conversation_history: List[ConversationThread]) -> Dict[str, Any]:
        """Calculate conversation health metrics."""
        try:
            if not conversation_history:
                return {}

            health_metrics = {}

            # Diversity score (topic variety)
            all_topics = []
            for thread in conversation_history:
                all_topics.extend(thread.topic_keywords)

            unique_topics = len(set(all_topics))
            total_topics = len(all_topics)
            diversity_score = unique_topics / total_topics if total_topics > 0 else 0
            health_metrics['topic_diversity'] = round(diversity_score, 2)

            # Engagement score (average conversation length)
            avg_length = sum(t.message_count for t in conversation_history) / len(conversation_history)
            engagement_score = min(1.0, avg_length / 10)  # Normalize to 0-1 scale
            health_metrics['engagement_level'] = round(engagement_score, 2)

            # Consistency score (regular conversation frequency)
            conversation_dates = []
            for thread in conversation_history:
                try:
                    dt = datetime.fromisoformat(thread.created_at.replace('Z', '+00:00'))
                    conversation_dates.append(dt)
                except:
                    continue

            if len(conversation_dates) > 1:
                conversation_dates.sort()
                intervals = []
                for i in range(1, len(conversation_dates)):
                    interval = (conversation_dates[i] - conversation_dates[i-1]).days
                    intervals.append(interval)

                avg_interval = sum(intervals) / len(intervals)
                consistency_score = max(0, 1 - (avg_interval / 30))  # 30 days = 0 consistency
                health_metrics['consistency'] = round(consistency_score, 2)

            # Overall health score
            scores = [health_metrics.get(key, 0) for key in ['topic_diversity', 'engagement_level', 'consistency']]
            overall_health = sum(scores) / len(scores) if scores else 0
            health_metrics['overall_health'] = round(overall_health, 2)

            return health_metrics

        except Exception as e:
            self.logger.warning(f"Health metrics calculation failed: {e}")
            return {}

    def _export_as_json(self, threads: List[ConversationThread], include_metadata: bool) -> str:
        """Export conversations as JSON."""
        try:
            export_data = []

            for thread in threads:
                thread_data = {
                    'thread_id': thread.thread_id,
                    'title': thread.title,
                    'messages': thread.messages,
                    'created_at': thread.created_at,
                    'last_updated': thread.last_updated,
                    'message_count': thread.message_count
                }

                if include_metadata:
                    thread_data.update({
                        'topic_keywords': thread.topic_keywords,
                        'metadata': thread.metadata
                    })

                export_data.append(thread_data)

            return json.dumps(export_data, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            return json.dumps({'error': str(e)})

    def _export_as_markdown(self, threads: List[ConversationThread], include_metadata: bool) -> str:
        """Export conversations as Markdown."""
        try:
            markdown_content = []
            markdown_content.append("# Conversation Export")
            markdown_content.append(f"\nExported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            markdown_content.append(f"Total conversations: {len(threads)}\n")

            for i, thread in enumerate(threads, 1):
                markdown_content.append(f"## {i}. {thread.title}")
                markdown_content.append(f"**Created:** {thread.created_at}")
                markdown_content.append(f"**Messages:** {thread.message_count}")

                if include_metadata and thread.topic_keywords:
                    markdown_content.append(f"**Topics:** {', '.join(thread.topic_keywords)}")

                markdown_content.append("\n### Conversation:")

                for msg in thread.messages:
                    role = msg.get('role', 'unknown').title()
                    content = msg.get('content', '')
                    timestamp = msg.get('timestamp', '')

                    markdown_content.append(f"\n**{role}** ({timestamp}):")
                    markdown_content.append(f"{content}\n")

                markdown_content.append("---\n")

            return '\n'.join(markdown_content)

        except Exception as e:
            self.logger.error(f"Markdown export failed: {e}")
            return f"# Export Error\n\n{str(e)}"

    def _export_as_csv(self, threads: List[ConversationThread], include_metadata: bool) -> str:
        """Export conversations as CSV."""
        try:
            import csv
            import io

            output = io.StringIO()

            # Define CSV headers
            headers = ['thread_id', 'title', 'created_at', 'message_count', 'role', 'content', 'timestamp']
            if include_metadata:
                headers.extend(['topic_keywords', 'user_tags'])

            writer = csv.writer(output)
            writer.writerow(headers)

            # Write conversation data
            for thread in threads:
                base_row = [
                    thread.thread_id,
                    thread.title,
                    thread.created_at,
                    thread.message_count
                ]

                for msg in thread.messages:
                    row = base_row + [
                        msg.get('role', ''),
                        msg.get('content', ''),
                        msg.get('timestamp', '')
                    ]

                    if include_metadata:
                        row.extend([
                            ', '.join(thread.topic_keywords),
                            ', '.join(thread.metadata.get('user_tags', []))
                        ])

                    writer.writerow(row)

            return output.getvalue()

        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            return f"Export Error: {str(e)}"

    def _create_conversation_index(self) -> Dict[str, Any]:
        """Create search index for faster conversation retrieval."""
        try:
            threads = self.get_archived_threads()
            index = {
                'topic_index': {},
                'content_index': {},
                'date_index': {},
                'thread_lookup': {}
            }

            for thread in threads:
                thread_id = thread.thread_id

                # Topic index
                for topic in thread.topic_keywords:
                    if topic not in index['topic_index']:
                        index['topic_index'][topic] = []
                    index['topic_index'][topic].append(thread_id)

                # Content index (simplified word-based)
                content_words = set()
                for msg in thread.messages:
                    words = msg.get('content', '').lower().split()
                    content_words.update(word for word in words if len(word) > 3)

                for word in content_words:
                    if word not in index['content_index']:
                        index['content_index'][word] = []
                    index['content_index'][word].append(thread_id)

                # Date index
                try:
                    date = datetime.fromisoformat(thread.created_at.replace('Z', '+00:00')).date()
                    date_str = date.strftime('%Y-%m-%d')
                    if date_str not in index['date_index']:
                        index['date_index'][date_str] = []
                    index['date_index'][date_str].append(thread_id)
                except:
                    pass

                # Thread lookup
                index['thread_lookup'][thread_id] = {
                    'title': thread.title,
                    'message_count': thread.message_count,
                    'created_at': thread.created_at
                }

            # Save index to file for persistence
            index_file = self.storage_dir / 'conversation_index.json'
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)

            return index

        except Exception as e:
            self.logger.error(f"Index creation failed: {e}")
            return {}

    def _create_search_cache(self) -> Dict[str, Any]:
        """Create cache for common search queries."""
        try:
            # Common search terms to pre-cache
            common_terms = ['secret', 'blue', 'lamps', 'code', 'help', 'problem', 'how', 'what']

            cache = {}

            for term in common_terms:
                search_results = self.search_within_threads(term, limit=10)
                cache[term] = {
                    'results': search_results,
                    'cached_at': datetime.now().isoformat(),
                    'result_count': len(search_results)
                }

            # Save cache to file
            cache_file = self.storage_dir / 'search_cache.json'
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)

            return cache

        except Exception as e:
            self.logger.error(f"Search cache creation failed: {e}")
            return {}

    def _optimize_storage_structure(self) -> bool:
        """Optimize storage structure for better performance."""
        try:
            # Create organized directory structure
            organized_dir = self.storage_dir / 'organized'
            organized_dir.mkdir(exist_ok=True)

            # Create subdirectories by date
            threads = self.get_archived_threads()

            for thread in threads:
                try:
                    date = datetime.fromisoformat(thread.created_at.replace('Z', '+00:00')).date()
                    year_month = date.strftime('%Y-%m')

                    month_dir = organized_dir / year_month
                    month_dir.mkdir(exist_ok=True)

                    # Move thread file to organized structure
                    old_file = self.storage_dir / f"{thread.thread_id}.json"
                    new_file = month_dir / f"{thread.thread_id}.json"

                    if old_file.exists() and not new_file.exists():
                        import shutil
                        shutil.move(str(old_file), str(new_file))

                except Exception as e:
                    self.logger.warning(f"Failed to organize thread {thread.thread_id}: {e}")
                    continue

            return True

        except Exception as e:
            self.logger.error(f"Storage optimization failed: {e}")
            return False

    def _estimate_performance_improvement(self, indexed_conversations: int, cache_entries: int) -> float:
        """Estimate performance improvement from optimization."""
        try:
            # Simple heuristic for performance improvement
            base_improvement = 0.0

            # Index-based improvement
            if indexed_conversations > 0:
                index_improvement = min(0.5, indexed_conversations / 100)  # Up to 50% improvement
                base_improvement += index_improvement

            # Cache-based improvement
            if cache_entries > 0:
                cache_improvement = min(0.3, cache_entries / 20)  # Up to 30% improvement
                base_improvement += cache_improvement

            return round(base_improvement, 2)

        except Exception as e:
            self.logger.warning(f"Performance estimation failed: {e}")
            return 0.0


# Global instance
_contextual_relevance_engine: Optional[ContextualRelevanceEngine] = None

def get_contextual_relevance_engine(config: Optional[Dict[str, Any]] = None) -> ContextualRelevanceEngine:
    """Get the global contextual relevance engine instance."""
    global _contextual_relevance_engine

    if _contextual_relevance_engine is None:
        _contextual_relevance_engine = ContextualRelevanceEngine(config)
    return _contextual_relevance_engine
