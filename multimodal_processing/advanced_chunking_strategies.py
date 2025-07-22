#!/usr/bin/env python3
"""
Advanced Chunking Strategies Implementation
Implements all strategies from steps2.md for improved document QA accuracy:
- Semantic-Aware Chunking
- Title + Body Chunk Fusion  
- Hierarchical Chunking (Multi-Level)
- Table, List, and Bullet-Aware Extraction
- Overlapping Window Strategy with Semantic Boundary Control
- Contextual Labeling for Chunk Enrichment
- Prompt-Optimized Chunk Embedding (RAG-Aligned)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AdvancedChunkMetadata:
    """Rich metadata for advanced chunking strategies."""
    chunk_id: str
    parent_section: str
    section_name: str
    page_number: int
    hierarchy_level: int
    chunk_type: str
    tags: List[str]
    role: str  # Abstract, Requirement, Objective, etc.
    enrichment_tags: List[str]  # Cyber Capabilities, Evaluation Criteria, etc.
    embedding_prefix: str
    overlap_info: Dict[str, Any]
    source_boundaries: Dict[str, int]  # start_char, end_char, start_line, end_line

class AdvancedChunkingStrategies:
    """Implementation of advanced chunking strategies from steps2.md."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Recursive separators for semantic-aware chunking
        self.semantic_separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ".",     # Sentence ends
            "!",     # Exclamation
            "?",     # Question
            ",",     # Comma
            " "      # Word breaks
        ]
        
        # Section detection patterns
        self.section_patterns = {
            'abstract': r'(?i)^(?:abstract|executive\s+summary)[:.]?\s*$',
            'introduction': r'(?i)^(?:introduction|background|overview)[:.]?\s*$',
            'objective': r'(?i)^(?:objective|goals?|purpose)[:.]?\s*$',
            'requirements': r'(?i)^(?:requirements?|specifications?|criteria)[:.]?\s*$',
            'capabilities': r'(?i)^(?:capabilities?|features?|functions?)[:.]?\s*$',
            'methodology': r'(?i)^(?:methodology|approach|methods?)[:.]?\s*$',
            'results': r'(?i)^(?:results?|findings?|outcomes?)[:.]?\s*$',
            'conclusion': r'(?i)^(?:conclusion|summary|recommendations?)[:.]?\s*$',
            'references': r'(?i)^(?:references?|bibliography|citations?)[:.]?\s*$',
        }
        
        # Enrichment tag patterns
        self.enrichment_patterns = {
            'cyber_capability': [
                r'(?i)\b(?:cyber|security|attack|defense|threat|vulnerability)\b',
                r'(?i)\b(?:encryption|authentication|firewall|intrusion)\b',
            ],
            'technical_requirement': [
                r'(?i)\b(?:shall|must|will|should|required|mandatory)\b',
                r'(?i)\b(?:performance|throughput|latency|accuracy)\b',
            ],
            'evaluation_criteria': [
                r'(?i)\b(?:criteria|metric|measure|evaluation|assessment)\b',
                r'(?i)\b(?:score|rating|grade|benchmark|standard)\b',
            ],
            'sbir_specific': [
                r'(?i)\b(?:phase\s+[i123]|sbir|sttr|innovation|commercialization)\b',
                r'(?i)\b(?:deliverable|milestone|timeline|budget)\b',
            ]
        }

    def process_document_with_advanced_strategies(self, text: str, source_location: str, 
                                                page_number: int = None) -> List[Dict[str, Any]]:
        """
        Process document using all advanced chunking strategies from steps2.md.
        
        Returns list of chunks with rich metadata for improved QA accuracy.
        """
        
        # Strategy 1: Semantic-Aware Chunking
        semantic_chunks = self._semantic_aware_chunking(text)
        
        # Strategy 2: Title + Body Chunk Fusion
        fused_chunks = self._title_body_fusion(semantic_chunks)
        
        # Strategy 3: Hierarchical Chunking (Multi-Level)
        hierarchical_chunks = self._hierarchical_chunking(fused_chunks, source_location)
        
        # Strategy 4: Table, List, and Bullet-Aware Extraction
        structure_aware_chunks = self._structure_aware_extraction(hierarchical_chunks)
        
        # Strategy 5: Overlapping Window Strategy
        overlapped_chunks = self._overlapping_window_strategy(structure_aware_chunks)
        
        # Strategy 6: Contextual Labeling for Chunk Enrichment
        enriched_chunks = self._contextual_labeling(overlapped_chunks, page_number)
        
        # Strategy 7: Prompt-Optimized Chunk Embedding
        final_chunks = self._prompt_optimized_embedding(enriched_chunks)
        
        return final_chunks

    def _semantic_aware_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Strategy 1: Semantic-aware chunking using logical boundaries."""
        chunks = []
        
        # Split by semantic separators in order of preference
        current_chunks = [text]
        
        for separator in self.semantic_separators:
            new_chunks = []
            for chunk in current_chunks:
                if len(chunk) <= self.chunk_size:
                    new_chunks.append(chunk)
                else:
                    # Split by current separator
                    parts = chunk.split(separator)
                    current_part = ""
                    
                    for part in parts:
                        potential_chunk = current_part + separator + part if current_part else part
                        
                        if len(potential_chunk) <= self.chunk_size:
                            current_part = potential_chunk
                        else:
                            if current_part:
                                new_chunks.append(current_part)
                            current_part = part
                    
                    if current_part:
                        new_chunks.append(current_part)
            
            current_chunks = new_chunks
        
        # Convert to chunk dictionaries
        for i, chunk_text in enumerate(current_chunks):
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text.strip(),
                    'chunk_id': f"semantic_{i}",
                    'strategy': 'semantic_aware',
                    'boundaries': 'logical'
                })
        
        return chunks

    def _title_body_fusion(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strategy 2: Title + Body Chunk Fusion to maintain topic continuity."""
        fused_chunks = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            text = chunk['text']
            
            # Check if this looks like a title/header
            if self._is_title_line(text):
                title = text
                body_parts = []
                
                # Collect following body paragraphs
                j = i + 1
                while j < len(chunks) and not self._is_title_line(chunks[j]['text']):
                    body_parts.append(chunks[j]['text'])
                    j += 1
                
                # Fuse title with body
                if body_parts:
                    body = '\n\n'.join(body_parts)
                    fused_text = f"{title}\n\n{body}"
                    
                    fused_chunks.append({
                        'text': fused_text,
                        'chunk_id': f"fused_{len(fused_chunks)}",
                        'strategy': 'title_body_fusion',
                        'title': title,
                        'body': body,
                        'original_chunks': j - i
                    })
                    
                    i = j  # Skip processed chunks
                else:
                    # Title without body
                    fused_chunks.append({
                        'text': title,
                        'chunk_id': f"title_only_{len(fused_chunks)}",
                        'strategy': 'title_only',
                        'title': title
                    })
                    i += 1
            else:
                # Regular content chunk
                fused_chunks.append({
                    'text': text,
                    'chunk_id': f"content_{len(fused_chunks)}",
                    'strategy': 'content',
                    'original_chunk_id': chunk['chunk_id']
                })
                i += 1
        
        return fused_chunks

    def _hierarchical_chunking(self, chunks: List[Dict[str, Any]], source_location: str) -> List[Dict[str, Any]]:
        """Strategy 3: Hierarchical chunking with multi-level structure."""
        hierarchical_chunks = []
        
        current_section = None
        section_counter = 0
        
        for chunk in chunks:
            text = chunk['text']
            
            # Detect section boundaries
            section_info = self._detect_section_type(text)
            
            if section_info['is_section_header']:
                # Start new section
                section_counter += 1
                current_section = {
                    'section_id': f"section_{section_counter}",
                    'section_name': section_info['section_name'],
                    'section_type': section_info['section_type'],
                    'level': section_info['level']
                }
            
            # Add hierarchical metadata
            hierarchical_chunk = {
                **chunk,
                'hierarchy_level': section_info['level'],
                'parent_section': current_section['section_id'] if current_section else None,
                'section_name': current_section['section_name'] if current_section else 'unknown',
                'section_type': current_section['section_type'] if current_section else 'content',
                'doc_id': source_location,
                'paragraph_id': f"{source_location}_{chunk['chunk_id']}",
            }
            
            hierarchical_chunks.append(hierarchical_chunk)
        
        return hierarchical_chunks

    def _structure_aware_extraction(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strategy 4: Table, List, and Bullet-Aware Extraction."""
        structure_aware_chunks = []
        
        for chunk in chunks:
            text = chunk['text']
            
            # Detect structural elements
            structure_info = self._analyze_text_structure(text)
            
            # Enhanced chunk with structure metadata
            enhanced_chunk = {
                **chunk,
                'structure_type': structure_info['type'],
                'has_lists': structure_info['has_lists'],
                'has_tables': structure_info['has_tables'],
                'list_items': structure_info['list_items'],
                'table_data': structure_info['table_data'],
                'structure_metadata': structure_info
            }
            
            structure_aware_chunks.append(enhanced_chunk)
        
        return structure_aware_chunks

    def _overlapping_window_strategy(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strategy 5: Overlapping Window Strategy with Semantic Boundary Control."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Calculate overlap content
            overlap_info = {
                'has_overlap': False,
                'previous_context': None,
                'next_context': None
            }
            
            # Add previous context
            if i > 0 and self.chunk_overlap > 0:
                prev_chunk = chunks[i - 1]
                prev_words = prev_chunk['text'].split()
                if len(prev_words) > 0:
                    overlap_words = min(self.chunk_overlap // 4, len(prev_words))
                    overlap_info['previous_context'] = " ".join(prev_words[-overlap_words:])
                    overlap_info['has_overlap'] = True
            
            # Add next context
            if i < len(chunks) - 1 and self.chunk_overlap > 0:
                next_chunk = chunks[i + 1]
                next_words = next_chunk['text'].split()
                if len(next_words) > 0:
                    overlap_words = min(self.chunk_overlap // 4, len(next_words))
                    overlap_info['next_context'] = " ".join(next_words[:overlap_words])
                    overlap_info['has_overlap'] = True
            
            # Enhanced chunk with overlap
            overlapped_chunk = {
                **chunk,
                'overlap_info': overlap_info,
                'chunk_overlap_size': self.chunk_overlap
            }
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks

    def _contextual_labeling(self, chunks: List[Dict[str, Any]], page_number: int = None) -> List[Dict[str, Any]]:
        """Strategy 6: Contextual Labeling for Chunk Enrichment."""
        enriched_chunks = []
        
        for chunk in chunks:
            text = chunk['text']
            
            # Generate enrichment tags
            enrichment_tags = []
            for tag_type, patterns in self.enrichment_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text):
                        enrichment_tags.append(tag_type)
                        break
            
            # Create rich metadata
            metadata = AdvancedChunkMetadata(
                chunk_id=chunk['chunk_id'],
                parent_section=chunk.get('parent_section', 'unknown'),
                section_name=chunk.get('section_name', 'unknown'),
                page_number=page_number or 1,
                hierarchy_level=chunk.get('hierarchy_level', 0),
                chunk_type=chunk.get('structure_type', 'text'),
                tags=list(set(enrichment_tags)),
                role=chunk.get('section_type', 'content'),
                enrichment_tags=enrichment_tags,
                embedding_prefix="",  # Will be set in next step
                overlap_info=chunk.get('overlap_info', {}),
                source_boundaries={}  # Could be enhanced with character positions
            )
            
            enriched_chunk = {
                **chunk,
                'metadata': metadata,
                'enrichment_tags': enrichment_tags,
                'page_number': page_number
            }
            
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks

    def _prompt_optimized_embedding(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strategy 7: Prompt-Optimized Chunk Embedding (RAG-Aligned)."""
        optimized_chunks = []
        
        for chunk in chunks:
            text = chunk['text']
            chunk_type = chunk.get('structure_type', 'text')
            section_type = chunk.get('section_type', 'content')
            
            # Generate task-specific embedding prefix
            embedding_prefix = self._generate_embedding_prefix(chunk_type, section_type, chunk)
            
            # Update metadata
            if hasattr(chunk.get('metadata'), 'embedding_prefix'):
                chunk['metadata'].embedding_prefix = embedding_prefix
            
            optimized_chunk = {
                **chunk,
                'embedding_prefix': embedding_prefix,
                'embedding_input': f"{embedding_prefix}{text}",
                'rag_optimized': True
            }
            
            optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks

    def _is_title_line(self, text: str) -> bool:
        """Check if text line is a title/header."""
        text = text.strip()
        
        # Check for section patterns
        for pattern in self.section_patterns.values():
            if re.match(pattern, text):
                return True
        
        # Check for other title indicators
        title_indicators = [
            r'^\d+\.\s+[A-Z]',  # "1. Title"
            r'^[A-Z][A-Z\s]{2,}:?\s*$',  # "ALL CAPS TITLE"
            r'^#{1,6}\s+',  # Markdown headers
            r'^\d+\.\d+\s+',  # "1.1 Section"
        ]
        
        for pattern in title_indicators:
            if re.match(pattern, text):
                return True
        
        return False

    def _detect_section_type(self, text: str) -> Dict[str, Any]:
        """Detect section type and hierarchy level."""
        text_lower = text.lower().strip()
        
        for section_type, pattern in self.section_patterns.items():
            if re.match(pattern, text):
                return {
                    'is_section_header': True,
                    'section_type': section_type,
                    'section_name': text.strip(),
                    'level': self._get_hierarchy_level(text)
                }
        
        return {
            'is_section_header': False,
            'section_type': 'content',
            'section_name': 'content',
            'level': 0
        }

    def _get_hierarchy_level(self, text: str) -> int:
        """Get hierarchy level from text."""
        # Markdown headers
        if text.strip().startswith('#'):
            return len(re.match(r'^#+', text.strip()).group())
        
        # Numbered sections
        section_match = re.match(r'^\s*(\d+(?:\.\d+)*)', text)
        if section_match:
            return len(section_match.group(1).split('.'))
        
        return 1

    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure for lists, tables, etc."""
        lines = text.split('\n')
        
        structure_info = {
            'type': 'text',
            'has_lists': False,
            'has_tables': False,
            'list_items': [],
            'table_data': [],
        }
        
        # Detect lists
        list_patterns = [
            r'^\s*[•·▪▫‣⁃\-*+]\s+',  # Bullet points
            r'^\s*\d+\.\s+',          # Numbered lists
            r'^\s*[a-z]\.\s+',        # Lettered lists
        ]
        
        for line in lines:
            for pattern in list_patterns:
                if re.match(pattern, line):
                    structure_info['has_lists'] = True
                    structure_info['list_items'].append(line.strip())
                    structure_info['type'] = 'list'
                    break
        
        # Detect tables (simple heuristic)
        tab_lines = [line for line in lines if '\t' in line or '|' in line]
        if len(tab_lines) >= 2:
            structure_info['has_tables'] = True
            structure_info['table_data'] = tab_lines
            structure_info['type'] = 'table'
        
        return structure_info

    def _generate_embedding_prefix(self, chunk_type: str, section_type: str, chunk: Dict[str, Any]) -> str:
        """Generate task-specific embedding prefix."""
        
        # Base prefixes by chunk type
        type_prefixes = {
            'list': "Instruction: This is a structured list of requirements or capabilities.\nContent: ",
            'table': "Instruction: This is tabular data with structured information.\nContent: ",
            'text': "Instruction: This is descriptive text content.\nContent: ",
        }
        
        # Enhanced prefixes by section type
        section_prefixes = {
            'abstract': "Instruction: This is an abstract or executive summary section.\nContent: ",
            'objective': "Instruction: This is an objective or goal statement section.\nContent: ",
            'requirements': "Instruction: This is a requirements or specifications section.\nContent: ",
            'capabilities': "Instruction: This is a capabilities or features section.\nContent: ",
            'methodology': "Instruction: This is a methodology or approach section.\nContent: ",
        }
        
        # Check for enrichment tags
        enrichment_tags = chunk.get('enrichment_tags', [])
        if 'cyber_capability' in enrichment_tags:
            return "Instruction: This is a cybersecurity capability or requirement chunk.\nContent: "
        elif 'technical_requirement' in enrichment_tags:
            return "Instruction: This is a technical requirement or specification chunk.\nContent: "
        elif 'sbir_specific' in enrichment_tags:
            return "Instruction: This is SBIR-specific content with innovation focus.\nContent: "
        
        # Use section-specific prefix if available
        if section_type in section_prefixes:
            return section_prefixes[section_type]
        
        # Fall back to chunk type prefix
        return type_prefixes.get(chunk_type, "Instruction: This is document content.\nContent: ")
