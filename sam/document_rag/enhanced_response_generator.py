"""
Enhanced Response Generator
==========================

Generates natural, conversational responses for document queries with feedback integration.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class ResponseStyle:
    """Response style configuration."""
    tone: str = "conversational"  # conversational, professional, academic
    detail_level: str = "medium"  # brief, medium, detailed
    include_sources: bool = True
    natural_language: bool = True
    avoid_technical_jargon: bool = True

@dataclass
class FeedbackContext:
    """Context from previous user feedback."""
    previous_feedback: List[Dict[str, Any]]
    common_complaints: List[str]
    preferred_style: Dict[str, Any]
    improvement_areas: List[str]

class EnhancedResponseGenerator:
    """
    Enhanced response generator that creates natural, conversational responses
    for document queries while learning from user feedback.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced response generator."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Response style configuration
        self.default_style = ResponseStyle(
            tone=self.config.get('tone', 'conversational'),
            detail_level=self.config.get('detail_level', 'medium'),
            include_sources=self.config.get('include_sources', True),
            natural_language=self.config.get('natural_language', True),
            avoid_technical_jargon=self.config.get('avoid_technical_jargon', True)
        )
        
        # Feedback integration
        self.feedback_handler = None
        self._initialize_feedback_integration()
        
        self.logger.info("Enhanced response generator initialized")
    
    def _initialize_feedback_integration(self):
        """Initialize feedback integration."""
        try:
            from learning.feedback_handler import get_feedback_handler
            self.feedback_handler = get_feedback_handler()
            self.logger.info("Feedback integration initialized")
        except Exception as e:
            self.logger.warning(f"Feedback integration not available: {e}")
    
    def generate_enhanced_response(self, 
                                 query: str,
                                 document_context: str,
                                 metadata: Dict[str, Any],
                                 user_id: str = "default",
                                 style_override: Optional[ResponseStyle] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate enhanced, natural response for document query.
        
        Args:
            query: User's question
            document_context: Retrieved document content
            metadata: Document metadata
            user_id: User identifier for feedback retrieval
            style_override: Override default response style
            
        Returns:
            Tuple of (enhanced_response, response_metadata)
        """
        try:
            self.logger.info(f"Generating enhanced response for: '{query[:50]}...'")
            
            # Get feedback context for this user
            feedback_context = self._get_feedback_context(user_id, query)
            
            # Determine response style
            response_style = style_override or self._determine_response_style(feedback_context)
            
            # Extract key information from document context
            key_info = self._extract_key_information(document_context, query)
            
            # Generate natural response
            enhanced_response = self._generate_natural_response(
                query, key_info, response_style, feedback_context
            )
            
            # Add sources if requested
            if response_style.include_sources:
                enhanced_response = self._add_source_attribution(enhanced_response, metadata)
            
            # Create response metadata
            response_metadata = {
                'generation_method': 'enhanced_natural',
                'style_used': response_style.__dict__,
                'feedback_applied': len(feedback_context.previous_feedback) > 0,
                'key_info_extracted': len(key_info),
                'generation_timestamp': datetime.now().isoformat(),
                'original_context_length': len(document_context),
                'enhanced_response_length': len(enhanced_response)
            }
            
            self.logger.info(f"Enhanced response generated: {len(enhanced_response)} characters")
            return enhanced_response, response_metadata
            
        except Exception as e:
            self.logger.error(f"Enhanced response generation failed: {e}")
            # Fallback to basic response
            return self._generate_fallback_response(query, document_context), {'error': str(e)}
    
    def _get_feedback_context(self, user_id: str, query: str) -> FeedbackContext:
        """Get feedback context for the user."""
        try:
            if not self.feedback_handler:
                return FeedbackContext([], [], {}, [])
            
            # Get recent feedback for this user
            recent_feedback = self.feedback_handler.get_user_feedback_history(user_id, limit=20)
            
            # Analyze feedback patterns
            common_complaints = self._analyze_feedback_patterns(recent_feedback)
            preferred_style = self._extract_style_preferences(recent_feedback)
            improvement_areas = self._identify_improvement_areas(recent_feedback)
            
            return FeedbackContext(
                previous_feedback=recent_feedback,
                common_complaints=common_complaints,
                preferred_style=preferred_style,
                improvement_areas=improvement_areas
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to get feedback context: {e}")
            return FeedbackContext([], [], {}, [])
    
    def _analyze_feedback_patterns(self, feedback_list: List[Dict[str, Any]]) -> List[str]:
        """Analyze feedback to identify common complaints."""
        complaints = []
        
        for feedback in feedback_list:
            correction_text = feedback.get('correction_text', '').lower()
            if correction_text:
                if 'mechanical' in correction_text or 'robotic' in correction_text:
                    complaints.append('too_mechanical')
                if 'natural' in correction_text or 'conversational' in correction_text:
                    complaints.append('needs_more_natural')
                if 'technical' in correction_text or 'jargon' in correction_text:
                    complaints.append('too_technical')
                if 'brief' in correction_text or 'concise' in correction_text:
                    complaints.append('too_verbose')
                if 'detail' in correction_text or 'explain' in correction_text:
                    complaints.append('needs_more_detail')
        
        return list(set(complaints))
    
    def _extract_style_preferences(self, feedback_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract style preferences from feedback."""
        preferences = {
            'tone': 'conversational',
            'detail_level': 'medium',
            'technical_level': 'low'
        }
        
        for feedback in feedback_list:
            correction_text = feedback.get('correction_text', '').lower()
            rating = feedback.get('rating', 0.5)
            
            # High-rated responses indicate good style
            if rating > 0.7:
                if 'professional' in correction_text:
                    preferences['tone'] = 'professional'
                elif 'casual' in correction_text or 'friendly' in correction_text:
                    preferences['tone'] = 'conversational'
        
        return preferences
    
    def _identify_improvement_areas(self, feedback_list: List[Dict[str, Any]]) -> List[str]:
        """Identify areas needing improvement."""
        areas = []
        
        low_rated_feedback = [f for f in feedback_list if f.get('rating', 0.5) < 0.4]
        
        for feedback in low_rated_feedback:
            correction_text = feedback.get('correction_text', '').lower()
            if 'explain' in correction_text or 'understand' in correction_text:
                areas.append('clarity')
            if 'relevant' in correction_text or 'topic' in correction_text:
                areas.append('relevance')
            if 'complete' in correction_text or 'missing' in correction_text:
                areas.append('completeness')
        
        return list(set(areas))
    
    def _determine_response_style(self, feedback_context: FeedbackContext) -> ResponseStyle:
        """Determine response style based on feedback."""
        style = ResponseStyle()
        
        # Adjust based on feedback
        if 'too_mechanical' in feedback_context.common_complaints:
            style.natural_language = True
            style.tone = 'conversational'
            style.avoid_technical_jargon = True
        
        if 'needs_more_natural' in feedback_context.common_complaints:
            style.tone = 'conversational'
            style.natural_language = True
        
        if 'too_technical' in feedback_context.common_complaints:
            style.avoid_technical_jargon = True
        
        if 'too_verbose' in feedback_context.common_complaints:
            style.detail_level = 'brief'
        elif 'needs_more_detail' in feedback_context.common_complaints:
            style.detail_level = 'detailed'
        
        # Apply user preferences
        if feedback_context.preferred_style:
            style.tone = feedback_context.preferred_style.get('tone', style.tone)
            style.detail_level = feedback_context.preferred_style.get('detail_level', style.detail_level)
        
        return style
    
    def _extract_key_information(self, document_context: str, query: str) -> List[Dict[str, Any]]:
        """Extract key information from document context."""
        key_info = []
        
        # Split context into sections
        sections = document_context.split('\n\n')
        
        for section in sections:
            if len(section.strip()) < 50:  # Skip very short sections
                continue
            
            # Extract source information
            source_match = re.search(r'\[Source: ([^\]]+)\]', section)
            source = source_match.group(1) if source_match else "Unknown"
            
            # Clean content
            content = re.sub(r'\[Source: [^\]]+\]', '', section).strip()
            content = re.sub(r'Block: \d+', '', content).strip()
            content = re.sub(r'Relevance: \d+\.\d+', '', content).strip()
            
            if content:
                key_info.append({
                    'content': content,
                    'source': source,
                    'relevance': self._calculate_relevance_to_query(content, query)
                })
        
        # Sort by relevance
        key_info.sort(key=lambda x: x['relevance'], reverse=True)
        
        return key_info[:5]  # Top 5 most relevant pieces
    
    def _calculate_relevance_to_query(self, content: str, query: str) -> float:
        """Calculate relevance of content to query."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Simple word overlap calculation
        overlap = len(query_words.intersection(content_words))
        total_query_words = len(query_words)
        
        return overlap / total_query_words if total_query_words > 0 else 0.0
    
    def _generate_natural_response(self, 
                                  query: str,
                                  key_info: List[Dict[str, Any]],
                                  style: ResponseStyle,
                                  feedback_context: FeedbackContext) -> str:
        """Generate natural, conversational response."""
        if not key_info:
            return "I couldn't find specific information about that in your uploaded documents. Could you try rephrasing your question or check if the relevant document has been uploaded?"
        
        # Start with a natural introduction
        intro_phrases = [
            f"Based on your uploaded document, here's what I found about {self._extract_topic_from_query(query)}:",
            f"Looking at your document, I can tell you that",
            f"From what I can see in your uploaded file",
            f"According to the document you've shared"
        ]
        
        # Choose intro based on style
        if style.tone == 'conversational':
            intro = intro_phrases[0]
        elif style.tone == 'professional':
            intro = "Based on the uploaded document analysis:"
        else:
            intro = intro_phrases[1]
        
        response_parts = [intro]
        
        # Add main content naturally
        for i, info in enumerate(key_info):
            content = info['content']
            
            # Make content more natural
            if style.natural_language:
                content = self._make_content_natural(content, style)
            
            if i == 0:
                response_parts.append(f"\n{content}")
            else:
                response_parts.append(f"\nAdditionally, {content.lower()}")
        
        # Add a natural conclusion
        if style.detail_level != 'brief':
            conclusion_phrases = [
                "\nIs there anything specific about this topic you'd like me to explain further?",
                "\nLet me know if you need clarification on any of these points!",
                "\nWould you like me to dive deeper into any particular aspect?"
            ]
            
            if 'clarity' not in feedback_context.improvement_areas:
                response_parts.append(conclusion_phrases[0])
        
        return "".join(response_parts)
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract the main topic from the query."""
        # Simple topic extraction
        query_lower = query.lower()
        
        # Remove question words
        topic = re.sub(r'\b(what|how|why|when|where|who|is|are|about|the)\b', '', query_lower)
        topic = topic.strip()
        
        # Take first few meaningful words
        words = [w for w in topic.split() if len(w) > 2]
        return " ".join(words[:3]) if words else "this topic"
    
    def _make_content_natural(self, content: str, style: ResponseStyle) -> str:
        """Make content more natural and conversational."""
        # Remove technical formatting
        natural_content = content.replace('_', ' ')
        natural_content = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group().title(), natural_content)
        
        # Add conversational elements
        if style.tone == 'conversational':
            # Add natural transitions
            if not natural_content.endswith('.'):
                natural_content += '.'
        
        # Avoid technical jargon if requested
        if style.avoid_technical_jargon:
            # Replace common technical terms with simpler alternatives
            replacements = {
                'utilize': 'use',
                'implement': 'put in place',
                'methodology': 'method',
                'facilitate': 'help',
                'optimize': 'improve'
            }
            
            for technical, simple in replacements.items():
                natural_content = re.sub(r'\b' + technical + r'\b', simple, natural_content, flags=re.IGNORECASE)
        
        return natural_content
    
    def _add_source_attribution(self, response: str, metadata: Dict[str, Any]) -> str:
        """Add source attribution to response."""
        sources = metadata.get('source_documents', [])
        if not sources:
            return response
        
        # Add sources naturally
        if len(sources) == 1:
            source_text = f"\n\n*Source: {sources[0]}*"
        else:
            source_list = "\n".join([f"• {source}" for source in sources[:3]])
            source_text = f"\n\n*Sources:*\n{source_list}"
            
            if len(sources) > 3:
                source_text += f"\n*...and {len(sources) - 3} more documents*"
        
        return response + source_text
    
    def _generate_fallback_response(self, query: str, document_context: str) -> str:
        """Generate fallback response when enhanced generation fails."""
        return f"I found some information about '{query}' in your documents, but I'm having trouble presenting it clearly right now. Here's what I found:\n\n{document_context[:500]}..."

# Global instance
_enhanced_generator = None

def get_enhanced_response_generator() -> EnhancedResponseGenerator:
    """Get the global enhanced response generator."""
    global _enhanced_generator
    if _enhanced_generator is None:
        _enhanced_generator = EnhancedResponseGenerator()
    return _enhanced_generator

def generate_enhanced_document_response(query: str,
                                      document_context: str,
                                      metadata: Dict[str, Any],
                                      user_id: str = "default") -> Tuple[str, Dict[str, Any]]:
    """Convenience function for enhanced document response generation."""
    generator = get_enhanced_response_generator()
    return generator.generate_enhanced_response(query, document_context, metadata, user_id)

__all__ = [
    'EnhancedResponseGenerator',
    'ResponseStyle',
    'FeedbackContext',
    'get_enhanced_response_generator',
    'generate_enhanced_document_response'
]
