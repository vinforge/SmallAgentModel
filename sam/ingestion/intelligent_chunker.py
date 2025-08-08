#!/usr/bin/env python3
"""
Intelligent Chunker
Advanced content-aware chunking that respects document structure and semantic boundaries.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .document_classifier import DocumentClassification, DocumentType
from .structure_mapper import DocumentMap, StructuralElement, DocumentElement

logger = logging.getLogger(__name__)

@dataclass
class IntelligentChunk:
    """Enhanced chunk with rich metadata and structural awareness."""
    content: str
    chunk_id: str
    start_pos: int
    end_pos: int
    
    # Structural metadata
    section_title: Optional[str] = None
    section_level: int = 0
    chunk_type: str = "content"  # content, table, formula, code, header
    
    # Content metadata
    contains_formulas: bool = False
    contains_tables: bool = False
    contains_code: bool = False
    contains_numerical_data: bool = False
    contains_definitions: bool = False
    
    # Processing metadata
    atomic_block: bool = False  # Should not be split further
    priority_level: int = 1     # 1=normal, 2=important, 3=critical
    processing_hints: Dict[str, Any] = None
    
    # Relationship metadata
    parent_chunk_id: Optional[str] = None
    related_chunk_ids: List[str] = None
    element_ids: List[str] = None  # Source structural elements
    
    def __post_init__(self):
        if self.processing_hints is None:
            self.processing_hints = {}
        if self.related_chunk_ids is None:
            self.related_chunk_ids = []
        if self.element_ids is None:
            self.element_ids = []

class IntelligentChunker:
    """
    Advanced chunker that uses document structure and classification for optimal chunking.
    """
    
    def __init__(self):
        self.chunking_strategies = {
            DocumentType.FINANCIAL_REPORT: self._chunk_financial_document,
            DocumentType.RESEARCH_PAPER: self._chunk_academic_document,
            DocumentType.TECHNICAL_MANUAL: self._chunk_technical_document,
            DocumentType.LEGAL_DOCUMENT: self._chunk_legal_document,
            DocumentType.GENERAL_TEXT: self._chunk_general_document
        }
        
        # Default chunking parameters
        self.default_chunk_size = 800
        self.min_chunk_size = 200
        self.max_chunk_size = 2000
        self.overlap_size = 100
        
        logger.info("IntelligentChunker initialized")
    
    def chunk_document(self, content: str, classification: DocumentClassification, 
                      document_map: DocumentMap) -> List[IntelligentChunk]:
        """
        Perform intelligent chunking based on document analysis.
        
        Args:
            content: Full document text
            classification: Document classification results
            document_map: Structural map of the document
            
        Returns:
            List of intelligently created chunks
        """
        try:
            # Select chunking strategy based on document type
            strategy = self.chunking_strategies.get(
                classification.document_type, 
                self._chunk_general_document
            )
            
            # Apply the appropriate chunking strategy
            chunks = strategy(content, classification, document_map)
            
            # Post-process chunks for optimization
            chunks = self._optimize_chunks(chunks, classification, document_map)
            
            # Add relationship metadata
            chunks = self._build_chunk_relationships(chunks, document_map)
            
            logger.info(f"📄 Document chunked into {len(chunks)} intelligent chunks")
            logger.info(f"🎯 Chunk types: {self._get_chunk_type_summary(chunks)}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Intelligent chunking failed: {e}")
            # Fallback to basic chunking
            return self._fallback_chunking(content)
    
    def _chunk_financial_document(self, content: str, classification: DocumentClassification,
                                document_map: DocumentMap) -> List[IntelligentChunk]:
        """Specialized chunking for financial documents."""
        chunks = []
        
        # Priority sections for financial documents
        priority_sections = ['executive summary', 'financial highlights', 'income statement', 
                           'balance sheet', 'cash flow', 'key metrics']
        
        # Process sections with financial awareness
        for section in document_map.sections:
            section_title = section['title'].lower()
            is_priority = any(priority in section_title for priority in priority_sections)
            
            section_content = content[section['start_pos']:section['end_pos']]
            
            # Handle tables specially in financial documents
            section_chunks = self._chunk_section_with_tables(
                section_content, section, document_map, 
                preserve_financial_data=True
            )
            
            # Mark priority chunks
            for chunk in section_chunks:
                if is_priority:
                    chunk.priority_level = 3  # Critical
                    chunk.processing_hints['financial_priority'] = True
                
                chunk.section_title = section['title']
                chunk.section_level = section['level']
                
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_academic_document(self, content: str, classification: DocumentClassification,
                               document_map: DocumentMap) -> List[IntelligentChunk]:
        """Specialized chunking for research papers."""
        chunks = []
        
        # Academic sections have different importance
        section_priorities = {
            'abstract': 3, 'conclusion': 3, 'results': 3,
            'introduction': 2, 'methodology': 2, 'discussion': 2,
            'literature review': 1, 'references': 1
        }
        
        for section in document_map.sections:
            section_title = section['title'].lower()
            priority = 1
            
            # Determine section priority
            for key_section, section_priority in section_priorities.items():
                if key_section in section_title:
                    priority = section_priority
                    break
            
            section_content = content[section['start_pos']:section['end_pos']]
            
            # Academic documents often have formulas and citations
            section_chunks = self._chunk_section_with_formulas(
                section_content, section, document_map
            )
            
            for chunk in section_chunks:
                chunk.priority_level = priority
                chunk.section_title = section['title']
                chunk.section_level = section['level']
                chunk.processing_hints['academic_section'] = True
                
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_technical_document(self, content: str, classification: DocumentClassification,
                                document_map: DocumentMap) -> List[IntelligentChunk]:
        """Specialized chunking for technical manuals."""
        chunks = []
        
        # Technical documents need to preserve procedures
        for section in document_map.sections:
            section_content = content[section['start_pos']:section['end_pos']]
            
            # Preserve code blocks and procedures
            section_chunks = self._chunk_section_with_code(
                section_content, section, document_map
            )
            
            for chunk in section_chunks:
                chunk.section_title = section['title']
                chunk.section_level = section['level']
                chunk.processing_hints['technical_content'] = True
                
                # Mark procedural content as important
                if any(word in chunk.content.lower() for word in 
                      ['step', 'procedure', 'install', 'configure', 'setup']):
                    chunk.priority_level = 2
                
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_legal_document(self, content: str, classification: DocumentClassification,
                            document_map: DocumentMap) -> List[IntelligentChunk]:
        """Specialized chunking for legal documents."""
        chunks = []
        
        # Legal documents have clauses and definitions
        for section in document_map.sections:
            section_content = content[section['start_pos']:section['end_pos']]
            
            # Preserve legal clauses and definitions
            section_chunks = self._chunk_section_with_definitions(
                section_content, section, document_map
            )
            
            for chunk in section_chunks:
                chunk.section_title = section['title']
                chunk.section_level = section['level']
                chunk.processing_hints['legal_content'] = True
                
                # Mark definitions and key clauses as important
                if any(word in chunk.content.lower() for word in 
                      ['whereas', 'definition', 'clause', 'agreement', 'liability']):
                    chunk.priority_level = 2
                
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_general_document(self, content: str, classification: DocumentClassification,
                              document_map: DocumentMap) -> List[IntelligentChunk]:
        """General chunking strategy for unclassified documents."""
        chunks = []
        
        if document_map.sections:
            # Use sections if available
            for section in document_map.sections:
                section_content = content[section['start_pos']:section['end_pos']]
                section_chunks = self._chunk_section_semantically(
                    section_content, section, document_map
                )
                
                for chunk in section_chunks:
                    chunk.section_title = section['title']
                    chunk.section_level = section['level']
                    
                chunks.extend(section_chunks)
        else:
            # Fallback to paragraph-based chunking
            chunks = self._chunk_by_paragraphs(content, document_map)
        
        return chunks
    
    def _chunk_section_with_tables(self, section_content: str, section: Dict[str, Any],
                                 document_map: DocumentMap, preserve_financial_data: bool = False) -> List[IntelligentChunk]:
        """Chunk a section while preserving table integrity."""
        chunks = []
        
        # Find atomic blocks (tables) in this section
        section_atomic_blocks = [
            block for block in document_map.atomic_blocks
            if section['start_pos'] <= block['start_pos'] < section['end_pos']
        ]
        
        current_pos = 0
        chunk_id_counter = 0
        
        for atomic_block in section_atomic_blocks:
            # Chunk content before the atomic block
            if current_pos < atomic_block['start_pos'] - section['start_pos']:
                pre_content = section_content[current_pos:atomic_block['start_pos'] - section['start_pos']]
                if pre_content.strip():
                    pre_chunks = self._create_semantic_chunks(pre_content, f"section_{chunk_id_counter}")
                    chunks.extend(pre_chunks)
                    chunk_id_counter += len(pre_chunks)
            
            # Create atomic chunk for table
            atomic_content = section_content[
                atomic_block['start_pos'] - section['start_pos']:
                atomic_block['end_pos'] - section['start_pos']
            ]
            
            table_chunk = IntelligentChunk(
                content=atomic_content,
                chunk_id=f"table_{chunk_id_counter}",
                start_pos=atomic_block['start_pos'],
                end_pos=atomic_block['end_pos'],
                chunk_type="table",
                contains_tables=True,
                contains_numerical_data=preserve_financial_data,
                atomic_block=True,
                priority_level=2 if preserve_financial_data else 1
            )
            
            chunks.append(table_chunk)
            chunk_id_counter += 1
            current_pos = atomic_block['end_pos'] - section['start_pos']
        
        # Chunk remaining content
        if current_pos < len(section_content):
            remaining_content = section_content[current_pos:]
            if remaining_content.strip():
                remaining_chunks = self._create_semantic_chunks(remaining_content, f"section_{chunk_id_counter}")
                chunks.extend(remaining_chunks)
        
        return chunks
    
    def _chunk_section_with_formulas(self, section_content: str, section: Dict[str, Any],
                                   document_map: DocumentMap) -> List[IntelligentChunk]:
        """Chunk a section while preserving formula integrity."""
        # Similar to table chunking but for formulas
        return self._chunk_section_with_tables(section_content, section, document_map)
    
    def _chunk_section_with_code(self, section_content: str, section: Dict[str, Any],
                               document_map: DocumentMap) -> List[IntelligentChunk]:
        """Chunk a section while preserving code block integrity."""
        # Similar to table chunking but for code blocks
        return self._chunk_section_with_tables(section_content, section, document_map)
    
    def _chunk_section_with_definitions(self, section_content: str, section: Dict[str, Any],
                                      document_map: DocumentMap) -> List[IntelligentChunk]:
        """Chunk a section while preserving definition integrity."""
        chunks = []
        
        # Look for definition patterns
        definition_patterns = [
            r'([A-Z][a-z\s]+):\s*([A-Z][^.]+\.)',  # Term: Definition.
            r'([A-Z][a-z\s]+)\s*-\s*([A-Z][^.]+\.)',  # Term - Definition.
        ]
        
        current_pos = 0
        chunk_id_counter = 0
        
        for pattern in definition_patterns:
            for match in re.finditer(pattern, section_content):
                # Chunk content before definition
                if current_pos < match.start():
                    pre_content = section_content[current_pos:match.start()]
                    if pre_content.strip():
                        pre_chunks = self._create_semantic_chunks(pre_content, f"section_{chunk_id_counter}")
                        chunks.extend(pre_chunks)
                        chunk_id_counter += len(pre_chunks)
                
                # Create definition chunk
                definition_chunk = IntelligentChunk(
                    content=match.group(0),
                    chunk_id=f"definition_{chunk_id_counter}",
                    start_pos=section['start_pos'] + match.start(),
                    end_pos=section['start_pos'] + match.end(),
                    chunk_type="definition",
                    contains_definitions=True,
                    priority_level=2
                )
                
                chunks.append(definition_chunk)
                chunk_id_counter += 1
                current_pos = match.end()
        
        # Chunk remaining content
        if current_pos < len(section_content):
            remaining_content = section_content[current_pos:]
            if remaining_content.strip():
                remaining_chunks = self._create_semantic_chunks(remaining_content, f"section_{chunk_id_counter}")
                chunks.extend(remaining_chunks)
        
        return chunks
    
    def _chunk_section_semantically(self, section_content: str, section: Dict[str, Any],
                                  document_map: DocumentMap) -> List[IntelligentChunk]:
        """Chunk a section using semantic boundaries."""
        return self._create_semantic_chunks(section_content, "semantic")
    
    def _chunk_by_paragraphs(self, content: str, document_map: DocumentMap) -> List[IntelligentChunk]:
        """Fallback chunking by paragraphs."""
        chunks = []
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        current_pos = 0
        chunk_id_counter = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.default_chunk_size and current_chunk:
                # Create chunk
                chunk = IntelligentChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"paragraph_{chunk_id_counter}",
                    start_pos=current_pos - len(current_chunk),
                    end_pos=current_pos,
                    chunk_type="content"
                )
                chunks.append(chunk)
                chunk_id_counter += 1
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            
            current_pos += len(paragraph) + 2  # +2 for \n\n
        
        # Add final chunk
        if current_chunk.strip():
            chunk = IntelligentChunk(
                content=current_chunk.strip(),
                chunk_id=f"paragraph_{chunk_id_counter}",
                start_pos=current_pos - len(current_chunk),
                end_pos=current_pos,
                chunk_type="content"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_semantic_chunks(self, content: str, base_id: str) -> List[IntelligentChunk]:
        """Create chunks using semantic boundaries."""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = ""
        chunk_id_counter = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.default_chunk_size and current_chunk:
                # Create chunk
                chunk = IntelligentChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"{base_id}_{chunk_id_counter}",
                    start_pos=0,  # Will be adjusted later
                    end_pos=len(current_chunk),
                    chunk_type="content",
                    contains_numerical_data=bool(re.search(r'\d+', current_chunk)),
                    contains_formulas=bool(re.search(r'[=]\s*[A-Za-z0-9\+\-\*\/\(\)]+', current_chunk))
                )
                chunks.append(chunk)
                chunk_id_counter += 1
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk = IntelligentChunk(
                content=current_chunk.strip(),
                chunk_id=f"{base_id}_{chunk_id_counter}",
                start_pos=0,
                end_pos=len(current_chunk),
                chunk_type="content",
                contains_numerical_data=bool(re.search(r'\d+', current_chunk)),
                contains_formulas=bool(re.search(r'[=]\s*[A-Za-z0-9\+\-\*\/\(\)]+', current_chunk))
            )
            chunks.append(chunk)
        
        return chunks
    
    def _optimize_chunks(self, chunks: List[IntelligentChunk], classification: DocumentClassification,
                        document_map: DocumentMap) -> List[IntelligentChunk]:
        """Optimize chunks for better processing."""
        optimized_chunks = []
        
        for chunk in chunks:
            # Merge very small chunks
            if len(chunk.content) < self.min_chunk_size and not chunk.atomic_block:
                if optimized_chunks and len(optimized_chunks[-1].content) < self.default_chunk_size:
                    # Merge with previous chunk
                    prev_chunk = optimized_chunks[-1]
                    prev_chunk.content += "\n\n" + chunk.content
                    prev_chunk.end_pos = chunk.end_pos
                    continue
            
            # Split very large chunks (unless atomic)
            if len(chunk.content) > self.max_chunk_size and not chunk.atomic_block:
                sub_chunks = self._split_large_chunk(chunk)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _split_large_chunk(self, chunk: IntelligentChunk) -> List[IntelligentChunk]:
        """Split a large chunk into smaller ones."""
        sub_chunks = []
        content = chunk.content
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        current_content = ""
        sub_chunk_counter = 0
        
        for paragraph in paragraphs:
            if len(current_content) + len(paragraph) > self.default_chunk_size and current_content:
                # Create sub-chunk
                sub_chunk = IntelligentChunk(
                    content=current_content.strip(),
                    chunk_id=f"{chunk.chunk_id}_sub_{sub_chunk_counter}",
                    start_pos=chunk.start_pos,  # Approximate
                    end_pos=chunk.start_pos + len(current_content),
                    chunk_type=chunk.chunk_type,
                    section_title=chunk.section_title,
                    section_level=chunk.section_level,
                    priority_level=chunk.priority_level,
                    processing_hints=chunk.processing_hints.copy()
                )
                sub_chunks.append(sub_chunk)
                sub_chunk_counter += 1
                current_content = paragraph
            else:
                current_content += "\n\n" + paragraph if current_content else paragraph
        
        # Add final sub-chunk
        if current_content.strip():
            sub_chunk = IntelligentChunk(
                content=current_content.strip(),
                chunk_id=f"{chunk.chunk_id}_sub_{sub_chunk_counter}",
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                chunk_type=chunk.chunk_type,
                section_title=chunk.section_title,
                section_level=chunk.section_level,
                priority_level=chunk.priority_level,
                processing_hints=chunk.processing_hints.copy()
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _build_chunk_relationships(self, chunks: List[IntelligentChunk], 
                                 document_map: DocumentMap) -> List[IntelligentChunk]:
        """Build relationships between chunks."""
        # Add parent-child relationships for hierarchical sections
        for i, chunk in enumerate(chunks):
            # Find related chunks in same section
            related_chunks = [
                other_chunk.chunk_id for other_chunk in chunks
                if (other_chunk.section_title == chunk.section_title and 
                    other_chunk.chunk_id != chunk.chunk_id)
            ]
            chunk.related_chunk_ids = related_chunks[:5]  # Limit to 5 related chunks
        
        return chunks
    
    def _get_chunk_type_summary(self, chunks: List[IntelligentChunk]) -> Dict[str, int]:
        """Get summary of chunk types for logging."""
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        return type_counts
    
    def _fallback_chunking(self, content: str) -> List[IntelligentChunk]:
        """Fallback to basic chunking if intelligent chunking fails."""
        chunks = []
        words = content.split()
        
        current_chunk = ""
        chunk_id_counter = 0
        
        for word in words:
            if len(current_chunk) + len(word) > self.default_chunk_size and current_chunk:
                chunk = IntelligentChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"fallback_{chunk_id_counter}",
                    start_pos=0,
                    end_pos=len(current_chunk),
                    chunk_type="content"
                )
                chunks.append(chunk)
                chunk_id_counter += 1
                current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word
        
        # Add final chunk
        if current_chunk.strip():
            chunk = IntelligentChunk(
                content=current_chunk.strip(),
                chunk_id=f"fallback_{chunk_id_counter}",
                start_pos=0,
                end_pos=len(current_chunk),
                chunk_type="content"
            )
            chunks.append(chunk)
        
        return chunks

# Global instance
_intelligent_chunker = None

def get_intelligent_chunker() -> IntelligentChunker:
    """Get or create the global intelligent chunker instance."""
    global _intelligent_chunker
    if _intelligent_chunker is None:
        _intelligent_chunker = IntelligentChunker()
    return _intelligent_chunker
