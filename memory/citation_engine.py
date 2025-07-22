#!/usr/bin/env python3
"""
Citation Engine - Inline Quote Injection
Provides transparent citations and memory quotes in SAM's responses.

Sprint 15 Deliverable #2: Inline Quote Injection (Citation Engine)
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CitationStyle(Enum):
    """Different citation formatting styles."""
    INLINE = "inline"           # [source: doc.pdf:Block 2]
    FOOTNOTE = "footnote"       # [1] with footnotes at end
    ACADEMIC = "academic"       # (Author, 2025)
    SIMPLE = "simple"          # (doc.pdf)

@dataclass
class Citation:
    """Represents a single citation with enhanced granular source information."""
    source_id: str
    source_name: str
    block_id: Optional[str]
    quote_text: str
    confidence_score: float
    citation_label: str
    full_source_path: str
    # Enhanced metadata tracking (LongBioBench-inspired)
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    paragraph_number: Optional[int] = None
    section_title: Optional[str] = None
    document_position: Optional[float] = None  # 0.0-1.0 position in document

@dataclass
class CitedResponse:
    """Response with embedded citations and source transparency."""
    response_text: str
    citations: List[Citation]
    source_count: int
    citation_style: CitationStyle
    transparency_score: float

class CitationEngine:
    """
    Advanced citation system that injects relevant memory quotes and source references
    into SAM's responses for improved transparency and verifiability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the citation engine."""
        self.config = config or self._get_default_config()
        self.citation_style = CitationStyle(self.config.get('citation_style', 'inline'))
        self.enable_citations = self.config.get('enable_citations', True)
        self.max_quote_length = self.config.get('max_quote_length', 150)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.3)
        
        logger.info("Citation engine initialized")
        logger.info(f"Citation style: {self.citation_style.value}")
        logger.info(f"Citations enabled: {self.enable_citations}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for citation engine."""
        return {
            'citation_style': 'inline',
            'enable_citations': True,
            'max_quote_length': 150,
            # CRITICAL FIX: Lower confidence threshold to allow more citations
            # The original 0.3 was too high and filtered out all valid citations
            'min_confidence_threshold': 0.05,
            'max_citations_per_response': 5,
            'quote_extraction_method': 'smart',
            'citation_placement': 'end',  # 'inline' or 'end' - use 'end' for better visibility
            'include_confidence_scores': False
        }
    
    def inject_citations(self, response_text: str, source_memories: List[Any], 
                        query: str = "") -> CitedResponse:
        """
        Inject citations and quotes into a response based on source memories.
        
        Args:
            response_text: The original response text
            source_memories: List of memory chunks that were used
            query: Original query for context
            
        Returns:
            CitedResponse with embedded citations
        """
        try:
            if not self.enable_citations or not source_memories:
                return CitedResponse(
                    response_text=response_text,
                    citations=[],
                    source_count=0,
                    citation_style=self.citation_style,
                    transparency_score=0.0
                )
            
            logger.info(f"Injecting citations for {len(source_memories)} source memories")
            
            # Extract relevant quotes and create citations
            citations = self._extract_citations(source_memories, query)
            
            # Filter citations by confidence
            filtered_citations = [
                c for c in citations 
                if c.confidence_score >= self.min_confidence_threshold
            ]
            
            # Limit number of citations
            max_citations = self.config.get('max_citations_per_response', 5)
            filtered_citations = filtered_citations[:max_citations]
            
            # Inject citations into response
            cited_response_text = self._inject_citations_into_text(
                response_text, filtered_citations
            )
            
            # Calculate transparency score
            transparency_score = self._calculate_transparency_score(
                cited_response_text, filtered_citations
            )
            
            logger.info(f"Citations injected: {len(filtered_citations)} citations")
            
            return CitedResponse(
                response_text=cited_response_text,
                citations=filtered_citations,
                source_count=len(source_memories),
                citation_style=self.citation_style,
                transparency_score=transparency_score
            )
            
        except Exception as e:
            logger.error(f"Error injecting citations: {e}")
            return CitedResponse(
                response_text=response_text,
                citations=[],
                source_count=len(source_memories) if source_memories else 0,
                citation_style=self.citation_style,
                transparency_score=0.0
            )
    
    def _extract_citations(self, source_memories: List[Any], query: str) -> List[Citation]:
        """Extract relevant citations from source memories."""
        citations = []
        
        try:
            for i, memory in enumerate(source_memories):
                # Get memory content and metadata
                content = self._get_memory_content(memory)
                source_info = self._get_source_info(memory)
                confidence = self._get_confidence_score(memory)
                
                if not content or not source_info:
                    continue
                
                # Extract relevant quote
                quote = self._extract_relevant_quote(content, query)
                
                if quote:
                    # Phase 3.2: Direct metadata access (no secondary lookups)
                    enhanced_metadata = self._get_direct_metadata(memory)

                    # Create citation with enhanced tracking
                    citation = Citation(
                        source_id=f"src_{i+1}",
                        source_name=source_info['name'],
                        block_id=source_info.get('block_id'),
                        quote_text=quote,
                        confidence_score=confidence,
                        citation_label=self._generate_citation_label(source_info, i+1),
                        full_source_path=source_info['full_path'],
                        # Enhanced metadata from direct access
                        page_number=enhanced_metadata.get('page_number'),
                        chunk_index=enhanced_metadata.get('chunk_index'),
                        paragraph_number=enhanced_metadata.get('paragraph_number'),
                        section_title=enhanced_metadata.get('section_title'),
                        document_position=enhanced_metadata.get('document_position')
                    )
                    
                    citations.append(citation)
            
            # Sort by confidence score
            citations.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return citations
            
        except Exception as e:
            logger.error(f"Error extracting citations: {e}")
            return []
    
    def _get_memory_content(self, memory: Any) -> str:
        """Extract content from memory object (Phase 3.2 enhanced)."""
        try:
            # Phase 3: RankedMemoryResult with direct content access
            if hasattr(memory, 'content') and hasattr(memory, 'final_score'):
                return memory.content

            # Legacy: MemorySearchResult with chunk
            elif hasattr(memory, 'chunk') and hasattr(memory.chunk, 'content'):
                return memory.chunk.content

            # Fallback: direct content attribute
            elif hasattr(memory, 'content'):
                return memory.content

            return ""
        except Exception:
            return ""
    
    def _get_source_info(self, memory: Any) -> Dict[str, str]:
        """Extract source information from memory object (Phase 3.2 enhanced)."""
        try:
            # Phase 3: RankedMemoryResult with rich metadata
            if hasattr(memory, 'metadata') and hasattr(memory, 'final_score'):
                metadata = memory.metadata
                return {
                    'name': metadata.get('source_name', 'Unknown'),
                    'full_path': metadata.get('source_path', ''),
                    'block_id': str(metadata.get('chunk_index', 0)),
                    'page_number': metadata.get('page_number', 1),
                    'section_title': metadata.get('section_title', ''),
                    'confidence_indicator': metadata.get('confidence_indicator', '?')
                }

            # Legacy: MemorySearchResult - parse source string
            source = ""
            if hasattr(memory, 'chunk') and hasattr(memory.chunk, 'source'):
                source = memory.chunk.source
            elif hasattr(memory, 'source'):
                source = memory.source

            source_info = {'full_path': source}

            # Parse legacy source format
            if source.startswith('document:'):
                # Format: document:filename:block_id or document:filename
                parts = source.split(':')
                if len(parts) >= 2:
                    filename = parts[1]
                    # Extract just the filename without path
                    if '/' in filename:
                        filename = filename.split('/')[-1]
                    source_info['name'] = filename

                    if len(parts) >= 3:
                        source_info['block_id'] = parts[2]
            else:
                source_info['name'] = source

            return source_info

        except Exception as e:
            logger.debug(f"Error getting source info: {e}")
            return {'name': 'unknown', 'full_path': ''}
    
    def _get_confidence_score(self, memory: Any) -> float:
        """Get confidence score from memory object (Phase 3.2 enhanced)."""
        try:
            # Phase 3: RankedMemoryResult with multiple score types
            if hasattr(memory, 'final_score') and hasattr(memory, 'confidence_score'):
                # Use final hybrid score for overall confidence
                return memory.final_score
            elif hasattr(memory, 'confidence_score'):
                return memory.confidence_score
            elif hasattr(memory, 'semantic_score'):
                return memory.semantic_score

            # Legacy: MemorySearchResult
            elif hasattr(memory, 'similarity_score'):
                return memory.similarity_score

            # Legacy: MemoryChunk importance score
            elif hasattr(memory, 'chunk') and hasattr(memory.chunk, 'importance_score'):
                return memory.chunk.importance_score
            elif hasattr(memory, 'importance_score'):
                return memory.importance_score

            return 0.5  # Default confidence

        except Exception:
            return 0.5
    
    def _extract_relevant_quote(self, content: str, query: str) -> str:
        """
        Extract the most relevant quote from content.

        CRITICAL FIX: Enhanced quote extraction that works for summary queries
        and doesn't require exact word overlap.
        """
        try:
            if not content:
                return ""

            # Clean content and remove metadata prefixes
            content = content.strip()

            # Remove document metadata prefixes that aren't useful for quotes
            if content.startswith('Document:'):
                lines = content.split('\n')
                # Skip metadata lines and get to actual content
                content_lines = []
                skip_metadata = True
                for line in lines:
                    if skip_metadata and (line.startswith('Document:') or
                                        line.startswith('Content Type:') or
                                        line.strip() == ''):
                        continue
                    skip_metadata = False
                    content_lines.append(line)
                content = '\n'.join(content_lines).strip()

            if not content:
                return ""

            # If content is short enough, use it as quote
            if len(content) <= self.max_quote_length:
                return content

            # ENHANCED APPROACH: For summary queries, extract meaningful content
            if query and ("summary" in query.lower() or "summarize" in query.lower()):
                # For summary queries, extract the most informative sentences
                sentences = self._split_into_sentences(content)

                # Score sentences by informativeness (length, keywords, position)
                scored_sentences = []
                for i, sentence in enumerate(sentences):
                    if len(sentence.strip()) < 20:  # Skip very short sentences
                        continue

                    # Score based on multiple factors
                    score = 0

                    # Length score (prefer medium-length sentences)
                    length_score = min(1.0, len(sentence) / 100.0)
                    score += length_score * 0.3

                    # Keyword score (look for important terms)
                    important_keywords = ['framework', 'evaluation', 'decompiler', 'analysis',
                                        'approach', 'method', 'system', 'tool', 'benchmark',
                                        'performance', 'results', 'findings', 'conclusion']
                    keyword_count = sum(1 for keyword in important_keywords
                                      if keyword in sentence.lower())
                    score += keyword_count * 0.4

                    # Position score (prefer earlier sentences)
                    position_score = 1.0 - (i / len(sentences)) * 0.3
                    score += position_score * 0.3

                    scored_sentences.append((sentence, score))

                # Get best sentence
                if scored_sentences:
                    scored_sentences.sort(key=lambda x: x[1], reverse=True)
                    best_sentence = scored_sentences[0][0].strip()

                    # If sentence is too long, truncate intelligently
                    if len(best_sentence) > self.max_quote_length:
                        # Try to truncate at word boundary
                        truncated = best_sentence[:self.max_quote_length-3]
                        last_space = truncated.rfind(' ')
                        if last_space > self.max_quote_length * 0.7:  # If we can keep most of it
                            truncated = truncated[:last_space]
                        return truncated + "..."

                    return best_sentence

            # FALLBACK: Try query word overlap (original approach)
            if query:
                query_words = set(query.lower().split())
                sentences = self._split_into_sentences(content)

                # Score sentences by query word overlap
                scored_sentences = []
                for sentence in sentences:
                    if len(sentence.strip()) < 20:
                        continue
                    sentence_words = set(sentence.lower().split())
                    overlap = len(query_words.intersection(sentence_words))
                    scored_sentences.append((sentence, overlap))

                # Get best sentence(s)
                scored_sentences.sort(key=lambda x: x[1], reverse=True)

                if scored_sentences and scored_sentences[0][1] > 0:
                    best_sentence = scored_sentences[0][0].strip()

                    # If sentence is too long, truncate
                    if len(best_sentence) > self.max_quote_length:
                        return best_sentence[:self.max_quote_length-3] + "..."

                    return best_sentence

            # FINAL FALLBACK: Extract first meaningful content
            sentences = self._split_into_sentences(content)
            for sentence in sentences:
                if len(sentence.strip()) >= 20:  # Get first substantial sentence
                    if len(sentence) > self.max_quote_length:
                        return sentence[:self.max_quote_length-3] + "..."
                    return sentence.strip()

            # Last resort: use first part of content
            if len(content) > self.max_quote_length:
                return content[:self.max_quote_length-3] + "..."

            return content
            
        except Exception as e:
            logger.debug(f"Error extracting quote: {e}")
            return content[:self.max_quote_length] if content else ""
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            return [text]
    
    def _generate_citation_label(self, source_info: Dict[str, str], index: int) -> str:
        """Generate citation label based on style."""
        try:
            source_name = source_info.get('name', 'unknown')
            block_id = source_info.get('block_id', '')
            
            if self.citation_style == CitationStyle.INLINE:
                if block_id:
                    return f"[source: {source_name}:{block_id}]"
                else:
                    return f"[source: {source_name}]"
            
            elif self.citation_style == CitationStyle.FOOTNOTE:
                return f"[{index}]"
            
            elif self.citation_style == CitationStyle.ACADEMIC:
                # Extract year if possible
                year = "2025"  # Default
                if "2024" in source_name or "2025" in source_name:
                    year = "2025"
                return f"({source_name}, {year})"
            
            elif self.citation_style == CitationStyle.SIMPLE:
                return f"({source_name})"
            
            return f"[{index}]"
            
        except Exception as e:
            logger.debug(f"Error generating citation label: {e}")
            return f"[{index}]"
    
    def _inject_citations_into_text(self, response_text: str, 
                                   citations: List[Citation]) -> str:
        """Inject citations into response text."""
        try:
            if not citations:
                return response_text
            
            cited_text = response_text
            
            # Phase 3.2: Enhanced citation formatting
            if self.config.get('citation_placement', 'end') == 'end':
                cited_text += "\n\n**📚 Sources:**\n"
                for i, citation in enumerate(citations, 1):
                    # Enhanced citation with rich metadata and confidence indicators
                    citation_detail = self._format_enhanced_citation(citation)
                    cited_text += f"{i}. {citation_detail}\n"
            
            else:
                # Inline injection (more complex - would need NLP to find best insertion points)
                # For now, add at end of first paragraph
                paragraphs = cited_text.split('\n\n')
                if paragraphs and citations:
                    first_citation = citations[0]
                    paragraphs[0] += f" {first_citation.citation_label}"
                    cited_text = '\n\n'.join(paragraphs)
                    
                    # Add sources section
                    cited_text += "\n\n**Sources:**\n"
                    for citation in citations:
                        cited_text += f"- {citation.citation_label}: \"{citation.quote_text}\"\n"
            
            return cited_text
            
        except Exception as e:
            logger.error(f"Error injecting citations into text: {e}")
            return response_text
    
    def _calculate_transparency_score(self, cited_text: str, 
                                    citations: List[Citation]) -> float:
        """Calculate transparency score based on citation coverage."""
        try:
            if not citations:
                return 0.0
            
            # Basic transparency metrics
            citation_count = len(citations)
            avg_confidence = sum(c.confidence_score for c in citations) / citation_count
            
            # Coverage score (how much of response is backed by sources)
            coverage_score = min(1.0, citation_count / 3.0)  # Normalize to 3 citations
            
            # Quality score (average confidence)
            quality_score = avg_confidence
            
            # Combined transparency score
            transparency_score = (coverage_score * 0.6 + quality_score * 0.4)
            
            return min(1.0, transparency_score)
            
        except Exception as e:
            logger.debug(f"Error calculating transparency score: {e}")
            return 0.5

    def _get_direct_metadata(self, memory) -> Dict[str, Any]:
        """
        Get metadata directly from memory object (Phase 3.2 - no secondary lookups).

        This eliminates legacy JSON file lookups and uses direct metadata access
        from RankedMemoryResult objects or legacy MemorySearchResult objects.
        """
        metadata = {}

        try:
            # Phase 3: RankedMemoryResult with rich metadata
            if hasattr(memory, 'metadata') and hasattr(memory, 'final_score'):
                memory_metadata = memory.metadata

                # Direct metadata access - no parsing needed
                metadata = {
                    'page_number': memory_metadata.get('page_number', 1),
                    'chunk_index': memory_metadata.get('chunk_index', 0),
                    'paragraph_number': memory_metadata.get('paragraph_number', 1),
                    'section_title': memory_metadata.get('section_title', ''),
                    'document_position': memory_metadata.get('document_position', 0.0),
                    'confidence_indicator': memory_metadata.get('confidence_indicator', '?'),
                    'source_name': memory_metadata.get('source_name', 'Unknown'),
                    'content_type': memory_metadata.get('content_type', 'text')
                }

                logger.debug(f"Direct metadata access: {len(metadata)} fields extracted")
                return metadata

            # Legacy: MemorySearchResult - extract from chunk metadata
            elif hasattr(memory, 'chunk') and hasattr(memory.chunk, 'metadata'):
                memory_metadata = memory.chunk.metadata

                # Extract available metadata fields
                metadata = {
                    'page_number': memory_metadata.get('page_number', memory_metadata.get('page', 1)),
                    'chunk_index': memory_metadata.get('chunk_index', memory_metadata.get('block_id', 0)),
                    'paragraph_number': memory_metadata.get('paragraph_number', memory_metadata.get('paragraph', 1)),
                    'section_title': memory_metadata.get('section_title', memory_metadata.get('section', '')),
                    'document_position': memory_metadata.get('document_position', memory_metadata.get('position', 0.0))
                }

                # Calculate document position if not available
                if not metadata['document_position'] and metadata['page_number'] and memory_metadata.get('total_pages'):
                    metadata['document_position'] = metadata['page_number'] / memory_metadata['total_pages']

                logger.debug(f"Legacy metadata extraction: {len(metadata)} fields extracted")
                return metadata

            # Fallback: minimal metadata
            else:
                logger.debug("Using fallback metadata")
                return {
                    'page_number': 1,
                    'chunk_index': 0,
                    'paragraph_number': 1,
                    'section_title': '',
                    'document_position': 0.0
                }

        except Exception as e:
            logger.debug(f"Error getting direct metadata: {e}")
            return {
                'page_number': 1,
                'chunk_index': 0,
                'paragraph_number': 1,
                'section_title': '',
                'document_position': 0.0
            }

    # Legacy method removed in Phase 3.2 - replaced with _get_direct_metadata()
    # This eliminates secondary lookups and content parsing

    def _format_enhanced_citation(self, citation: Citation) -> str:
        """
        Format citation with enhanced granular metadata (Phase 3.2 enhanced).

        Uses rich metadata from RankedMemoryResult for detailed source attribution
        with confidence indicators and granular location information.
        """
        try:
            # Phase 3.2: Enhanced citation format with rich metadata
            citation_parts = []

            # Start with source name and confidence visualization
            source_name = citation.source_name or "Unknown"
            confidence_score = citation.confidence_score

            # Visual confidence indicator (5-dot scale)
            filled_dots = min(5, max(1, int(confidence_score * 5)))
            confidence_dots = "●" * filled_dots + "○" * (5 - filled_dots)

            # Format: 📚 **Source Name** ●●●○○ (85.2%)
            citation_parts.append(f"📚 **{source_name}** {confidence_dots} ({confidence_score:.1%})")

            # Add granular location information
            location_parts = []

            if citation.page_number and citation.page_number > 1:
                location_parts.append(f"p.{citation.page_number}")

            if citation.section_title:
                # Truncate long section titles
                section = citation.section_title
                if len(section) > 30:
                    section = section[:27] + "..."
                location_parts.append(f"§{section}")

            if citation.chunk_index is not None and citation.chunk_index > 0:
                location_parts.append(f"chunk {citation.chunk_index}")

            # Add document position indicator if available
            if hasattr(citation, 'document_position') and citation.document_position:
                position_percent = int(citation.document_position * 100)
                if position_percent > 0:
                    location_parts.append(f"{position_percent}% through doc")

            # Format location information
            if location_parts:
                location_str = f"[{', '.join(location_parts)}]"
                citation_parts.append(location_str)

            # Add relevant quote with smart truncation
            quote_text = citation.quote_text.strip()
            if quote_text:
                # Smart truncation at sentence boundaries
                if len(quote_text) > 120:
                    # Try to truncate at sentence boundary
                    sentences = quote_text.split('. ')
                    if len(sentences) > 1 and len(sentences[0]) < 100:
                        quote_text = sentences[0] + "..."
                    else:
                        quote_text = quote_text[:117] + "..."

                # Clean up quote formatting
                quote_text = quote_text.replace('\n', ' ').replace('  ', ' ')
                citation_parts.append(f"\n   _{quote_text}_")

            return "\n".join(citation_parts) if len(citation_parts) > 1 else citation_parts[0]

        except Exception as e:
            logger.debug(f"Error formatting enhanced citation: {e}")
            # Fallback to basic format
            confidence_dots = "●○○○○"  # Default low confidence
            return f"📚 **{citation.source_name or 'Unknown'}** {confidence_dots} (0.0%)\n   _{citation.quote_text or 'No quote available'}_"

# Global citation engine instance
_citation_engine = None

def get_citation_engine() -> CitationEngine:
    """Get or create a global citation engine instance."""
    global _citation_engine
    
    if _citation_engine is None:
        _citation_engine = CitationEngine()
    
    return _citation_engine
