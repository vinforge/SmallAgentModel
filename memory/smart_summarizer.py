#!/usr/bin/env python3
"""
Smart Summary Generator
Provides intelligent summarization of memory blocks and documents.

Sprint 15 Deliverable #3: Smart Summary Generator
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class SummaryType(Enum):
    """Types of summaries that can be generated."""
    TOPIC_SUMMARY = "topic"           # Summary of memories about a specific topic
    DOCUMENT_SUMMARY = "document"     # Summary of a complete document
    MULTI_DOC_SUMMARY = "multi_doc"   # Summary across multiple documents
    TIMELINE_SUMMARY = "timeline"     # Chronological summary
    COMPARATIVE_SUMMARY = "comparative" # Comparison between sources

class SummaryFormat(Enum):
    """Output formats for summaries."""
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    BULLET_POINTS = "bullet_points"
    JSON = "json"
    HTML = "html"

@dataclass
class SummaryRequest:
    """Request for summary generation."""
    topic_keyword: str
    summary_type: SummaryType
    output_format: SummaryFormat
    max_length: int
    include_sources: bool
    memory_filters: Optional[Dict[str, Any]] = None

@dataclass
class GeneratedSummary:
    """Generated summary with metadata."""
    summary_text: str
    summary_type: SummaryType
    output_format: SummaryFormat
    source_count: int
    memory_ids: List[str]
    confidence_score: float
    key_topics: List[str]
    generated_at: str
    word_count: int
    summary_id: str

class SmartSummarizer:
    """
    Advanced summarization system that creates intelligent summaries from memory blocks.
    
    Features:
    - Topic-based summarization
    - Multi-document synthesis
    - Timeline extraction
    - Source attribution
    - Multiple output formats
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the smart summarizer."""
        self.config = config or self._get_default_config()
        self.max_summary_length = self.config.get('max_summary_length', 1000)
        self.min_memory_threshold = self.config.get('min_memory_threshold', 1)
        
        # Initialize LLM for summarization
        self.llm_model = None
        self._initialize_llm()
        
        logger.info("Smart summarizer initialized")
        logger.info(f"Max summary length: {self.max_summary_length}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for summarizer."""
        return {
            'max_summary_length': 1000,
            'min_memory_threshold': 1,
            'default_format': 'markdown',
            'include_sources_by_default': True,
            'enable_key_topic_extraction': True,
            'summarization_model': 'ollama',
            'chunk_size': 2000,  # Max content size per summarization call
            'overlap_size': 200   # Overlap between chunks
        }
    
    def _initialize_llm(self):
        """Initialize the LLM model for summarization."""
        try:
            from models.ollama_model import OllamaModel
            self.llm_model = OllamaModel()
            logger.info("LLM model initialized for summarization")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            self.llm_model = None
    
    def generate_summary(self, request: SummaryRequest, 
                        memory_store: Any) -> GeneratedSummary:
        """
        Generate a smart summary based on the request.
        
        Args:
            request: Summary generation request
            memory_store: Memory store to search for relevant memories
            
        Returns:
            GeneratedSummary object with the generated content
        """
        try:
            logger.info(f"Generating {request.summary_type.value} summary for: {request.topic_keyword}")
            
            # Search for relevant memories
            relevant_memories = self._find_relevant_memories(request, memory_store)
            
            if len(relevant_memories) < self.min_memory_threshold:
                logger.warning(f"Insufficient memories found: {len(relevant_memories)}")
                return self._create_empty_summary(request, "Insufficient relevant memories found")
            
            # Generate summary based on type
            summary_text = self._generate_summary_text(request, relevant_memories)
            
            # Extract key topics
            key_topics = self._extract_key_topics(summary_text, relevant_memories)
            
            # Calculate confidence score
            confidence_score = self._calculate_summary_confidence(relevant_memories, summary_text)
            
            # Create summary object
            from datetime import datetime
            import uuid
            
            summary = GeneratedSummary(
                summary_text=summary_text,
                summary_type=request.summary_type,
                output_format=request.output_format,
                source_count=len(relevant_memories),
                memory_ids=[self._get_memory_id(m) for m in relevant_memories],
                confidence_score=confidence_score,
                key_topics=key_topics,
                generated_at=datetime.now().isoformat(),
                word_count=len(summary_text.split()),
                summary_id=str(uuid.uuid4())[:8]
            )
            
            logger.info(f"Summary generated: {summary.word_count} words, {summary.source_count} sources")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._create_empty_summary(request, f"Error: {e}")
    
    def _find_relevant_memories(self, request: SummaryRequest, 
                               memory_store: Any) -> List[Any]:
        """Find memories relevant to the summary request."""
        try:
            # Search for memories matching the topic
            search_results = memory_store.search_memories(
                request.topic_keyword,
                max_results=20  # Get more results for better summarization
            )
            
            # Apply additional filters if specified
            if request.memory_filters:
                filtered_results = []
                for result in search_results:
                    if self._matches_filters(result, request.memory_filters):
                        filtered_results.append(result)
                search_results = filtered_results
            
            logger.info(f"Found {len(search_results)} relevant memories")
            return search_results
            
        except Exception as e:
            logger.error(f"Error finding relevant memories: {e}")
            return []
    
    def _matches_filters(self, memory_result: Any, filters: Dict[str, Any]) -> bool:
        """Check if memory matches the specified filters."""
        try:
            memory = memory_result.chunk if hasattr(memory_result, 'chunk') else memory_result
            
            # Check memory type filter
            if 'memory_types' in filters:
                memory_type = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
                if memory_type not in filters['memory_types']:
                    return False
            
            # Check tags filter
            if 'tags' in filters and hasattr(memory, 'tags'):
                if not any(tag in memory.tags for tag in filters['tags']):
                    return False
            
            # Check source filter
            if 'sources' in filters and hasattr(memory, 'source'):
                if not any(source in memory.source for source in filters['sources']):
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking filters: {e}")
            return True  # Include by default if filter check fails
    
    def _generate_summary_text(self, request: SummaryRequest, 
                              memories: List[Any]) -> str:
        """Generate the actual summary text."""
        try:
            if not self.llm_model:
                return self._generate_simple_summary(memories)
            
            # Prepare content for summarization
            content_chunks = self._prepare_content_for_summarization(memories)
            
            # Generate summary based on type
            if request.summary_type == SummaryType.TOPIC_SUMMARY:
                return self._generate_topic_summary(request, content_chunks)
            elif request.summary_type == SummaryType.DOCUMENT_SUMMARY:
                return self._generate_document_summary(request, content_chunks)
            elif request.summary_type == SummaryType.TIMELINE_SUMMARY:
                return self._generate_timeline_summary(request, content_chunks)
            else:
                return self._generate_topic_summary(request, content_chunks)
            
        except Exception as e:
            logger.error(f"Error generating summary text: {e}")
            return self._generate_simple_summary(memories)
    
    def _prepare_content_for_summarization(self, memories: List[Any]) -> List[str]:
        """Prepare memory content for summarization."""
        try:
            content_chunks = []
            current_chunk = ""
            chunk_size = self.config.get('chunk_size', 2000)
            
            for memory in memories:
                content = self._get_memory_content(memory)
                source = self._get_memory_source(memory)
                
                # Add source attribution
                attributed_content = f"[Source: {source}]\n{content}\n\n"
                
                # Check if adding this content would exceed chunk size
                if len(current_chunk + attributed_content) > chunk_size:
                    if current_chunk:
                        content_chunks.append(current_chunk)
                        current_chunk = attributed_content
                    else:
                        # Single content is too large, add it anyway
                        content_chunks.append(attributed_content)
                else:
                    current_chunk += attributed_content
            
            # Add remaining content
            if current_chunk:
                content_chunks.append(current_chunk)
            
            return content_chunks
            
        except Exception as e:
            logger.error(f"Error preparing content: {e}")
            return []
    
    def _generate_topic_summary(self, request: SummaryRequest, 
                               content_chunks: List[str]) -> str:
        """Generate a topic-based summary."""
        try:
            # Combine all content
            combined_content = "\n".join(content_chunks)
            
            # Create summarization prompt
            prompt = f"""Please create a comprehensive summary about "{request.topic_keyword}" based on the following information:

{combined_content}

Requirements:
- Focus specifically on information related to "{request.topic_keyword}"
- Maximum length: {request.max_length} words
- Format: {request.output_format.value}
- Include key points and important details
- Be concise but comprehensive

Summary:"""

            # Generate summary using LLM
            response = self.llm_model.generate_response(prompt)
            
            # Format the response
            return self._format_summary(response, request.output_format)
            
        except Exception as e:
            logger.error(f"Error generating topic summary: {e}")
            return f"Error generating summary: {e}"
    
    def _generate_document_summary(self, request: SummaryRequest, 
                                  content_chunks: List[str]) -> str:
        """Generate a document-based summary."""
        try:
            combined_content = "\n".join(content_chunks)
            
            prompt = f"""Please create a comprehensive summary of the following document content:

{combined_content}

Requirements:
- Summarize the main points and key information
- Maximum length: {request.max_length} words
- Format: {request.output_format.value}
- Maintain the logical structure of the original content
- Include important dates, numbers, and specific details

Summary:"""

            response = self.llm_model.generate_response(prompt)
            return self._format_summary(response, request.output_format)
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return f"Error generating summary: {e}"
    
    def _generate_timeline_summary(self, request: SummaryRequest, 
                                  content_chunks: List[str]) -> str:
        """Generate a timeline-based summary."""
        try:
            combined_content = "\n".join(content_chunks)
            
            prompt = f"""Please create a chronological timeline summary based on the following information:

{combined_content}

Requirements:
- Extract and organize information by dates and time periods
- Focus on "{request.topic_keyword}" if specified
- Maximum length: {request.max_length} words
- Format: {request.output_format.value}
- Include specific dates, deadlines, and milestones
- Order events chronologically

Timeline Summary:"""

            response = self.llm_model.generate_response(prompt)
            return self._format_summary(response, request.output_format)
            
        except Exception as e:
            logger.error(f"Error generating timeline summary: {e}")
            return f"Error generating summary: {e}"
    
    def _format_summary(self, summary_text: str, output_format: SummaryFormat) -> str:
        """Format summary according to the requested format."""
        try:
            if output_format == SummaryFormat.MARKDOWN:
                return summary_text  # Assume LLM already provides markdown
            elif output_format == SummaryFormat.BULLET_POINTS:
                # Convert to bullet points if not already
                lines = summary_text.split('\n')
                bullet_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('- ') and not line.startswith('* '):
                        bullet_lines.append(f"- {line}")
                    elif line:
                        bullet_lines.append(line)
                return '\n'.join(bullet_lines)
            elif output_format == SummaryFormat.PLAIN_TEXT:
                # Remove markdown formatting
                import re
                plain_text = re.sub(r'[*_#`]', '', summary_text)
                return plain_text
            else:
                return summary_text
                
        except Exception as e:
            logger.debug(f"Error formatting summary: {e}")
            return summary_text
    
    def _generate_simple_summary(self, memories: List[Any]) -> str:
        """Generate a simple summary without LLM (fallback)."""
        try:
            summary_parts = []
            summary_parts.append("# Summary\n")
            
            for i, memory in enumerate(memories[:5]):  # Limit to top 5
                content = self._get_memory_content(memory)
                source = self._get_memory_source(memory)
                
                # Extract first sentence or first 100 characters
                if content:
                    first_sentence = content.split('.')[0]
                    if len(first_sentence) > 100:
                        first_sentence = content[:100] + "..."
                    
                    summary_parts.append(f"**{i+1}.** {first_sentence} *(Source: {source})*\n")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating simple summary: {e}")
            return "Unable to generate summary."
    
    def _get_memory_content(self, memory: Any) -> str:
        """Extract content from memory object."""
        try:
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'content'):
                return memory.chunk.content
            elif hasattr(memory, 'content'):
                return memory.content
            return ""
        except Exception:
            return ""
    
    def _get_memory_source(self, memory: Any) -> str:
        """Extract source from memory object."""
        try:
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'source'):
                source = memory.chunk.source
            elif hasattr(memory, 'source'):
                source = memory.source
            else:
                return "unknown"
            
            # Simplify source name
            if source.startswith('document:'):
                parts = source.split(':')
                if len(parts) >= 2:
                    filename = parts[1].split('/')[-1]  # Get just filename
                    return filename
            
            return source
            
        except Exception:
            return "unknown"
    
    def _get_memory_id(self, memory: Any) -> str:
        """Extract memory ID from memory object."""
        try:
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'chunk_id'):
                return memory.chunk.chunk_id
            elif hasattr(memory, 'chunk_id'):
                return memory.chunk_id
            elif hasattr(memory, 'id'):
                return memory.id
            return "unknown"
        except Exception:
            return "unknown"
    
    def _extract_key_topics(self, summary_text: str, memories: List[Any]) -> List[str]:
        """Extract key topics from the summary and memories."""
        try:
            # Simple keyword extraction
            import re
            
            # Extract words that appear frequently
            words = re.findall(r'\b[A-Za-z]{4,}\b', summary_text.lower())
            word_freq = {}
            
            for word in words:
                if word not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'will']:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            key_topics = [word for word, freq in top_words[:5] if freq > 1]
            
            return key_topics
            
        except Exception as e:
            logger.debug(f"Error extracting key topics: {e}")
            return []
    
    def _calculate_summary_confidence(self, memories: List[Any], summary_text: str) -> float:
        """Calculate confidence score for the generated summary."""
        try:
            # Base confidence on number of sources and content quality
            source_score = min(1.0, len(memories) / 5.0)  # Normalize to 5 sources
            
            # Content quality score
            content_score = 0.5
            if len(summary_text) > 100:
                content_score = 0.7
            if len(summary_text) > 300:
                content_score = 0.9
            
            # Average memory confidence
            memory_confidences = []
            for memory in memories:
                if hasattr(memory, 'similarity_score'):
                    memory_confidences.append(memory.similarity_score)
            
            avg_memory_confidence = sum(memory_confidences) / len(memory_confidences) if memory_confidences else 0.5
            
            # Combined confidence
            confidence = (source_score * 0.3 + content_score * 0.3 + avg_memory_confidence * 0.4)
            return min(1.0, confidence)
            
        except Exception as e:
            logger.debug(f"Error calculating confidence: {e}")
            return 0.5
    
    def _create_empty_summary(self, request: SummaryRequest, reason: str) -> GeneratedSummary:
        """Create an empty summary with error information."""
        from datetime import datetime
        import uuid
        
        return GeneratedSummary(
            summary_text=f"Unable to generate summary: {reason}",
            summary_type=request.summary_type,
            output_format=request.output_format,
            source_count=0,
            memory_ids=[],
            confidence_score=0.0,
            key_topics=[],
            generated_at=datetime.now().isoformat(),
            word_count=0,
            summary_id=str(uuid.uuid4())[:8]
        )

# Global summarizer instance
_smart_summarizer = None

def get_smart_summarizer() -> SmartSummarizer:
    """Get or create a global smart summarizer instance."""
    global _smart_summarizer
    
    if _smart_summarizer is None:
        _smart_summarizer = SmartSummarizer()
    
    return _smart_summarizer
