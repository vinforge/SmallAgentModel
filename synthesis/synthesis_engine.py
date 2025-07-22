"""
Cross-Document Synthesis Engine for SAM
Combines insights from multiple files, memory entries, and capsules into unified outputs.

Sprint 7 Task 1: Cross-Document Synthesis Engine
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class SourceType(Enum):
    """Types of information sources."""
    DOCUMENT = "document"
    MEMORY = "memory"
    CAPSULE = "capsule"
    TOOL_OUTPUT = "tool_output"
    WEB_SEARCH = "web_search"

class ConflictType(Enum):
    """Types of conflicts between sources."""
    CONTRADICTION = "contradiction"
    INCONSISTENCY = "inconsistency"
    OUTDATED = "outdated"
    UNCERTAINTY = "uncertainty"

@dataclass
class SourceInfo:
    """Information about a source."""
    source_id: str
    source_type: SourceType
    title: str
    content: str
    confidence: float
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class ConflictDetection:
    """Detected conflict between sources."""
    conflict_type: ConflictType
    source1: SourceInfo
    source2: SourceInfo
    description: str
    confidence: float
    resolution_suggestion: str

@dataclass
class SynthesisResult:
    """Result of cross-document synthesis."""
    synthesized_content: str
    sources_used: List[SourceInfo]
    provenance_map: Dict[str, List[str]]  # content_section -> source_ids
    conflicts_detected: List[ConflictDetection]
    synthesis_confidence: float
    synthesis_metadata: Dict[str, Any]

class CrossDocumentSynthesizer:
    """
    Engine for synthesizing information from multiple sources into unified outputs.
    """
    
    def __init__(self, memory_manager=None, capsule_manager=None, 
                 multimodal_pipeline=None):
        """
        Initialize the cross-document synthesizer.
        
        Args:
            memory_manager: Long-term memory manager
            capsule_manager: Knowledge capsule manager
            multimodal_pipeline: Multimodal content pipeline
        """
        self.memory_manager = memory_manager
        self.capsule_manager = capsule_manager
        self.multimodal_pipeline = multimodal_pipeline
        
        # Synthesis configuration
        self.config = {
            'max_sources': 10,
            'min_confidence_threshold': 0.3,
            'conflict_detection_enabled': True,
            'provenance_tracking': True,
            'consistency_checking': True
        }
        
        logger.info("Cross-document synthesizer initialized")
    
    def synthesize_multi_source_answer(self, query: str, 
                                     document_sources: List[Dict[str, Any]] = None,
                                     memory_sources: List[Dict[str, Any]] = None,
                                     capsule_sources: List[Dict[str, Any]] = None,
                                     tool_outputs: List[Dict[str, Any]] = None) -> SynthesisResult:
        """
        Synthesize an answer from multiple sources.
        
        Args:
            query: The question or topic to synthesize information about
            document_sources: List of document sources
            memory_sources: List of memory sources
            capsule_sources: List of capsule sources
            tool_outputs: List of tool outputs
            
        Returns:
            SynthesisResult with unified answer and provenance
        """
        try:
            logger.info(f"Starting multi-source synthesis for: {query[:50]}...")
            
            # Gather all sources
            all_sources = self._gather_sources(
                query, document_sources, memory_sources, capsule_sources, tool_outputs
            )
            
            if not all_sources:
                return SynthesisResult(
                    synthesized_content="No relevant sources found for synthesis.",
                    sources_used=[],
                    provenance_map={},
                    conflicts_detected=[],
                    synthesis_confidence=0.0,
                    synthesis_metadata={'query': query, 'sources_found': 0}
                )
            
            # Filter and rank sources
            filtered_sources = self._filter_and_rank_sources(all_sources, query)
            
            # Detect conflicts
            conflicts = []
            if self.config['conflict_detection_enabled']:
                conflicts = self._detect_conflicts(filtered_sources)
            
            # Synthesize content
            synthesized_content, provenance_map = self._synthesize_content(
                query, filtered_sources, conflicts
            )
            
            # Calculate synthesis confidence
            synthesis_confidence = self._calculate_synthesis_confidence(
                filtered_sources, conflicts
            )
            
            # Create metadata
            synthesis_metadata = {
                'query': query,
                'sources_count': len(filtered_sources),
                'conflicts_count': len(conflicts),
                'synthesis_timestamp': datetime.now().isoformat(),
                'source_types': [s.source_type.value for s in filtered_sources]
            }
            
            result = SynthesisResult(
                synthesized_content=synthesized_content,
                sources_used=filtered_sources,
                provenance_map=provenance_map,
                conflicts_detected=conflicts,
                synthesis_confidence=synthesis_confidence,
                synthesis_metadata=synthesis_metadata
            )
            
            logger.info(f"Synthesis completed: {len(filtered_sources)} sources, "
                       f"{len(conflicts)} conflicts, confidence: {synthesis_confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-source synthesis: {e}")
            return SynthesisResult(
                synthesized_content=f"Error during synthesis: {str(e)}",
                sources_used=[],
                provenance_map={},
                conflicts_detected=[],
                synthesis_confidence=0.0,
                synthesis_metadata={'error': str(e)}
            )
    
    def auto_synthesize_from_query(self, query: str, max_sources: int = 5) -> SynthesisResult:
        """
        Automatically gather and synthesize sources for a query.
        
        Args:
            query: Query to synthesize information for
            max_sources: Maximum number of sources to use
            
        Returns:
            SynthesisResult with auto-gathered sources
        """
        try:
            # Auto-gather from available sources
            document_sources = []
            memory_sources = []
            capsule_sources = []
            
            # Search multimodal documents
            if self.multimodal_pipeline:
                try:
                    doc_results = self.multimodal_pipeline.search_multimodal_content(
                        query=query, top_k=max_sources//2
                    )
                    document_sources = [
                        {
                            'content': result.get('content', ''),
                            'metadata': result.get('metadata', {}),
                            'confidence': result.get('similarity_score', 0.5)
                        }
                        for result in doc_results
                    ]
                except Exception as e:
                    logger.warning(f"Document search failed: {e}")
            
            # Search memory
            if self.memory_manager:
                try:
                    memory_results = self.memory_manager.search_memories(
                        query=query, top_k=max_sources//2
                    )
                    memory_sources = [
                        {
                            'content': result.memory_entry.content,
                            'metadata': result.memory_entry.metadata,
                            'confidence': result.similarity_score
                        }
                        for result in memory_results
                    ]
                except Exception as e:
                    logger.warning(f"Memory search failed: {e}")
            
            # Search capsules
            if self.capsule_manager:
                try:
                    capsule_results = self.capsule_manager.search_capsules(
                        query=query, top_k=max_sources//3
                    )
                    capsule_sources = [
                        {
                            'content': capsule.content.reasoning_trace,
                            'metadata': {
                                'capsule_name': capsule.name,
                                'effectiveness': capsule.effectiveness_score,
                                'tools_used': capsule.content.tools_used
                            },
                            'confidence': capsule.effectiveness_score
                        }
                        for capsule in capsule_results
                    ]
                except Exception as e:
                    logger.warning(f"Capsule search failed: {e}")
            
            return self.synthesize_multi_source_answer(
                query=query,
                document_sources=document_sources,
                memory_sources=memory_sources,
                capsule_sources=capsule_sources
            )
            
        except Exception as e:
            logger.error(f"Error in auto-synthesis: {e}")
            return SynthesisResult(
                synthesized_content=f"Auto-synthesis failed: {str(e)}",
                sources_used=[],
                provenance_map={},
                conflicts_detected=[],
                synthesis_confidence=0.0,
                synthesis_metadata={'error': str(e)}
            )
    
    def _gather_sources(self, query: str, document_sources: List[Dict[str, Any]] = None,
                       memory_sources: List[Dict[str, Any]] = None,
                       capsule_sources: List[Dict[str, Any]] = None,
                       tool_outputs: List[Dict[str, Any]] = None) -> List[SourceInfo]:
        """Gather and convert all sources to SourceInfo objects."""
        sources = []
        
        # Process document sources
        if document_sources:
            for i, doc in enumerate(document_sources):
                source = SourceInfo(
                    source_id=f"doc_{i}",
                    source_type=SourceType.DOCUMENT,
                    title=doc.get('metadata', {}).get('title', f"Document {i+1}"),
                    content=doc.get('content', ''),
                    confidence=doc.get('confidence', 0.5),
                    timestamp=doc.get('metadata', {}).get('timestamp', datetime.now().isoformat()),
                    metadata=doc.get('metadata', {})
                )
                sources.append(source)
        
        # Process memory sources
        if memory_sources:
            for i, mem in enumerate(memory_sources):
                source = SourceInfo(
                    source_id=f"mem_{i}",
                    source_type=SourceType.MEMORY,
                    title=f"Memory: {mem.get('content', '')[:30]}...",
                    content=mem.get('content', ''),
                    confidence=mem.get('confidence', 0.5),
                    timestamp=mem.get('metadata', {}).get('created_at', datetime.now().isoformat()),
                    metadata=mem.get('metadata', {})
                )
                sources.append(source)
        
        # Process capsule sources
        if capsule_sources:
            for i, cap in enumerate(capsule_sources):
                source = SourceInfo(
                    source_id=f"cap_{i}",
                    source_type=SourceType.CAPSULE,
                    title=cap.get('metadata', {}).get('capsule_name', f"Capsule {i+1}"),
                    content=cap.get('content', ''),
                    confidence=cap.get('confidence', 0.5),
                    timestamp=cap.get('metadata', {}).get('created_at', datetime.now().isoformat()),
                    metadata=cap.get('metadata', {})
                )
                sources.append(source)
        
        # Process tool outputs
        if tool_outputs:
            for i, tool in enumerate(tool_outputs):
                source = SourceInfo(
                    source_id=f"tool_{i}",
                    source_type=SourceType.TOOL_OUTPUT,
                    title=f"Tool: {tool.get('tool_name', 'Unknown')}",
                    content=str(tool.get('output', '')),
                    confidence=0.8 if tool.get('success', False) else 0.3,
                    timestamp=tool.get('timestamp', datetime.now().isoformat()),
                    metadata=tool.get('metadata', {})
                )
                sources.append(source)
        
        return sources
    
    def _filter_and_rank_sources(self, sources: List[SourceInfo], query: str) -> List[SourceInfo]:
        """Filter and rank sources by relevance and confidence."""
        # Filter by confidence threshold
        filtered = [s for s in sources if s.confidence >= self.config['min_confidence_threshold']]
        
        # Calculate relevance scores
        query_words = set(query.lower().split())
        
        for source in filtered:
            content_words = set(source.content.lower().split())
            title_words = set(source.title.lower().split())
            
            # Calculate word overlap
            content_overlap = len(query_words & content_words) / len(query_words) if query_words else 0
            title_overlap = len(query_words & title_words) / len(query_words) if query_words else 0
            
            # Combined relevance score
            relevance = (content_overlap * 0.7 + title_overlap * 0.3)
            
            # Boost by source confidence and recency
            try:
                source_time = datetime.fromisoformat(source.timestamp.replace('Z', '+00:00'))
                age_days = (datetime.now() - source_time.replace(tzinfo=None)).days
                recency_factor = max(0.1, 1.0 - (age_days / 365))  # Decay over a year
            except:
                recency_factor = 0.5
            
            # Final score
            source.metadata['relevance_score'] = relevance * source.confidence * recency_factor
        
        # Sort by relevance score and limit
        filtered.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        
        return filtered[:self.config['max_sources']]
    
    def _detect_conflicts(self, sources: List[SourceInfo]) -> List[ConflictDetection]:
        """Detect conflicts between sources."""
        conflicts = []
        
        # Simple conflict detection based on contradictory keywords
        contradiction_patterns = [
            (r'\bis\s+not\b', r'\bis\b'),
            (r'\bfalse\b', r'\btrue\b'),
            (r'\bincorrect\b', r'\bcorrect\b'),
            (r'\bwrong\b', r'\bright\b'),
            (r'\bdisagree\b', r'\bagree\b')
        ]
        
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources[i+1:], i+1):
                # Check for contradictory patterns
                for neg_pattern, pos_pattern in contradiction_patterns:
                    if (re.search(neg_pattern, source1.content, re.IGNORECASE) and
                        re.search(pos_pattern, source2.content, re.IGNORECASE)):
                        
                        conflict = ConflictDetection(
                            conflict_type=ConflictType.CONTRADICTION,
                            source1=source1,
                            source2=source2,
                            description=f"Contradictory statements detected between {source1.title} and {source2.title}",
                            confidence=0.6,
                            resolution_suggestion="Review both sources and determine which is more reliable"
                        )
                        conflicts.append(conflict)
                
                # Check for date-based conflicts (newer vs older information)
                try:
                    time1 = datetime.fromisoformat(source1.timestamp.replace('Z', '+00:00'))
                    time2 = datetime.fromisoformat(source2.timestamp.replace('Z', '+00:00'))
                    
                    time_diff = abs((time1 - time2).days)
                    
                    if time_diff > 365:  # More than a year apart
                        older_source = source1 if time1 < time2 else source2
                        newer_source = source2 if time1 < time2 else source1
                        
                        conflict = ConflictDetection(
                            conflict_type=ConflictType.OUTDATED,
                            source1=older_source,
                            source2=newer_source,
                            description=f"Potentially outdated information: {older_source.title} is {time_diff} days older",
                            confidence=0.4,
                            resolution_suggestion="Prefer newer information unless older source is more authoritative"
                        )
                        conflicts.append(conflict)
                        
                except:
                    pass  # Skip if timestamp parsing fails
        
        return conflicts
    
    def _synthesize_content(self, query: str, sources: List[SourceInfo], 
                          conflicts: List[ConflictDetection]) -> Tuple[str, Dict[str, List[str]]]:
        """Synthesize content from sources with provenance tracking."""
        synthesis_parts = []
        provenance_map = {}
        
        # Create introduction
        intro = f"## Synthesis for: {query}\n\n"
        synthesis_parts.append(intro)
        provenance_map['introduction'] = ['synthesis_engine']
        
        # Group sources by type
        source_groups = {}
        for source in sources:
            source_type = source.source_type.value
            if source_type not in source_groups:
                source_groups[source_type] = []
            source_groups[source_type].append(source)
        
        # Synthesize by source type
        for source_type, type_sources in source_groups.items():
            section_title = f"### From {source_type.title()} Sources\n\n"
            synthesis_parts.append(section_title)
            
            section_content = []
            section_source_ids = []
            
            for source in type_sources:
                # Extract key points from source
                content_summary = self._extract_key_points(source.content, query)
                
                if content_summary:
                    source_line = f"**{source.title}** (confidence: {source.confidence:.2f}): {content_summary}\n\n"
                    section_content.append(source_line)
                    section_source_ids.append(source.source_id)
            
            if section_content:
                synthesis_parts.extend(section_content)
                provenance_map[f'{source_type}_section'] = section_source_ids
        
        # Add conflicts section if any
        if conflicts:
            conflicts_section = "### ⚠️ Detected Conflicts\n\n"
            synthesis_parts.append(conflicts_section)
            
            for conflict in conflicts:
                conflict_desc = f"- **{conflict.conflict_type.value.title()}**: {conflict.description}\n"
                conflict_desc += f"  - Resolution: {conflict.resolution_suggestion}\n\n"
                synthesis_parts.append(conflict_desc)
            
            provenance_map['conflicts'] = [c.source1.source_id for c in conflicts] + [c.source2.source_id for c in conflicts]
        
        # Add summary
        summary_section = "### 📋 Summary\n\n"
        summary_content = self._generate_summary(sources, query)
        synthesis_parts.extend([summary_section, summary_content])
        provenance_map['summary'] = [s.source_id for s in sources]
        
        return ''.join(synthesis_parts), provenance_map
    
    def _extract_key_points(self, content: str, query: str) -> str:
        """Extract key points from content relevant to the query."""
        # Simple extraction: find sentences containing query keywords
        query_words = set(query.lower().split())
        sentences = content.split('.')
        
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words & sentence_words)
            
            if overlap > 0:
                relevant_sentences.append(sentence)
        
        # Return first few relevant sentences
        if relevant_sentences:
            return '. '.join(relevant_sentences[:2]) + '.'
        else:
            # Fallback: return first part of content
            return content[:200] + '...' if len(content) > 200 else content
    
    def _generate_summary(self, sources: List[SourceInfo], query: str) -> str:
        """Generate a summary of the synthesis."""
        summary_parts = []
        
        # Count sources by type
        source_counts = {}
        for source in sources:
            source_type = source.source_type.value
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        # Create summary
        summary_parts.append(f"Based on analysis of {len(sources)} sources")
        
        if source_counts:
            type_descriptions = []
            for source_type, count in source_counts.items():
                type_descriptions.append(f"{count} {source_type}")
            summary_parts.append(f" ({', '.join(type_descriptions)})")
        
        summary_parts.append(f" regarding '{query}', ")
        
        # Add confidence assessment
        avg_confidence = sum(s.confidence for s in sources) / len(sources) if sources else 0
        if avg_confidence > 0.7:
            summary_parts.append("the information appears highly reliable.")
        elif avg_confidence > 0.5:
            summary_parts.append("the information appears moderately reliable.")
        else:
            summary_parts.append("the information should be verified from additional sources.")
        
        return ''.join(summary_parts) + '\n\n'
    
    def _calculate_synthesis_confidence(self, sources: List[SourceInfo], 
                                      conflicts: List[ConflictDetection]) -> float:
        """Calculate overall confidence in the synthesis."""
        if not sources:
            return 0.0
        
        # Base confidence from sources
        avg_source_confidence = sum(s.confidence for s in sources) / len(sources)
        
        # Reduce confidence based on conflicts
        conflict_penalty = len(conflicts) * 0.1
        
        # Boost confidence based on source diversity
        source_types = set(s.source_type for s in sources)
        diversity_bonus = min(0.2, len(source_types) * 0.05)
        
        # Final confidence
        final_confidence = avg_source_confidence - conflict_penalty + diversity_bonus
        
        return max(0.0, min(1.0, final_confidence))

# Global synthesizer instance
_synthesizer = None

def get_synthesizer(memory_manager=None, capsule_manager=None, 
                   multimodal_pipeline=None) -> CrossDocumentSynthesizer:
    """Get or create a global synthesizer instance."""
    global _synthesizer
    
    if _synthesizer is None:
        _synthesizer = CrossDocumentSynthesizer(
            memory_manager=memory_manager,
            capsule_manager=capsule_manager,
            multimodal_pipeline=multimodal_pipeline
        )
    
    return _synthesizer
