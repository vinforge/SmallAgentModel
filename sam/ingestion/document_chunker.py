#!/usr/bin/env python3
"""
V2 Document Chunker for SAM MUVERA Retrieval Pipeline
Advanced chunking strategies optimized for multi-vector embeddings.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Global chunker instance
_v2_chunker = None

class ChunkingStrategy(Enum):
    """Chunking strategies for different document types."""
    SEMANTIC = "semantic"           # Semantic boundary-aware chunking
    FIXED_SIZE = "fixed_size"      # Fixed size with overlap
    SENTENCE = "sentence"          # Sentence-based chunking
    PARAGRAPH = "paragraph"        # Paragraph-based chunking
    SLIDING_WINDOW = "sliding_window"  # Sliding window approach
    ADAPTIVE = "adaptive"          # Adaptive based on content

@dataclass
class ChunkResult:
    """Result from document chunking."""
    chunks: List[str]              # Text chunks
    chunk_metadata: List[Dict[str, Any]]  # Metadata for each chunk
    chunking_strategy: ChunkingStrategy   # Strategy used
    total_chunks: int              # Number of chunks created
    avg_chunk_size: float          # Average chunk size
    processing_time: float         # Time taken for chunking
    metadata: Dict[str, Any]       # Additional metadata

class V2DocumentChunker:
    """
    Advanced document chunker optimized for v2 multi-vector embeddings.
    
    Provides multiple chunking strategies to optimize for different document types
    and embedding models.
    """
    
    def __init__(self,
                 default_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 50,
                 max_chunk_size: int = 1024):
        """
        Initialize the v2 document chunker.
        
        Args:
            default_strategy: Default chunking strategy
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.default_strategy = default_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        
        logger.info(f"📄 V2DocumentChunker initialized: {default_strategy.value}")
        logger.info(f"📊 Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    
    def chunk_document(self, 
                      text: str,
                      strategy: Optional[ChunkingStrategy] = None,
                      document_type: Optional[str] = None) -> ChunkResult:
        """
        Chunk a document using the specified strategy.
        
        Args:
            text: Document text to chunk
            strategy: Chunking strategy (uses default if None)
            document_type: Document type hint for adaptive chunking
            
        Returns:
            ChunkResult with chunks and metadata
        """
        import time
        start_time = time.time()
        
        try:
            strategy = strategy or self.default_strategy
            
            logger.debug(f"🔄 Chunking document: {len(text)} chars, strategy: {strategy.value}")
            
            # Select chunking method
            if strategy == ChunkingStrategy.SEMANTIC:
                chunks, metadata = self._chunk_semantic(text)
            elif strategy == ChunkingStrategy.FIXED_SIZE:
                chunks, metadata = self._chunk_fixed_size(text)
            elif strategy == ChunkingStrategy.SENTENCE:
                chunks, metadata = self._chunk_sentence(text)
            elif strategy == ChunkingStrategy.PARAGRAPH:
                chunks, metadata = self._chunk_paragraph(text)
            elif strategy == ChunkingStrategy.SLIDING_WINDOW:
                chunks, metadata = self._chunk_sliding_window(text)
            elif strategy == ChunkingStrategy.ADAPTIVE:
                chunks, metadata = self._chunk_adaptive(text, document_type)
            else:
                logger.warning(f"⚠️  Unknown strategy: {strategy}, using semantic")
                chunks, metadata = self._chunk_semantic(text)
            
            # Filter chunks by size
            filtered_chunks = []
            filtered_metadata = []
            
            for i, chunk in enumerate(chunks):
                if self.min_chunk_size <= len(chunk) <= self.max_chunk_size:
                    filtered_chunks.append(chunk)
                    filtered_metadata.append(metadata[i] if i < len(metadata) else {})
                elif len(chunk) > self.max_chunk_size:
                    # Split oversized chunks
                    sub_chunks = self._split_oversized_chunk(chunk)
                    filtered_chunks.extend(sub_chunks)
                    for j, sub_chunk in enumerate(sub_chunks):
                        sub_metadata = metadata[i].copy() if i < len(metadata) else {}
                        sub_metadata['sub_chunk_index'] = j
                        sub_metadata['parent_chunk_size'] = len(chunk)
                        filtered_metadata.append(sub_metadata)
            
            processing_time = time.time() - start_time
            avg_chunk_size = sum(len(chunk) for chunk in filtered_chunks) / len(filtered_chunks) if filtered_chunks else 0
            
            result = ChunkResult(
                chunks=filtered_chunks,
                chunk_metadata=filtered_metadata,
                chunking_strategy=strategy,
                total_chunks=len(filtered_chunks),
                avg_chunk_size=avg_chunk_size,
                processing_time=processing_time,
                metadata={
                    'original_text_length': len(text),
                    'original_chunks': len(chunks),
                    'filtered_chunks': len(filtered_chunks),
                    'document_type': document_type,
                    'chunk_size_target': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                }
            )
            
            logger.debug(f"✅ Chunking completed: {len(filtered_chunks)} chunks, "
                        f"avg size: {avg_chunk_size:.0f}, {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Chunking failed: {e}")
            return ChunkResult(
                chunks=[text],  # Fallback to single chunk
                chunk_metadata=[{'error': str(e)}],
                chunking_strategy=strategy or self.default_strategy,
                total_chunks=1,
                avg_chunk_size=len(text),
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _chunk_semantic(self, text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Semantic boundary-aware chunking."""
        chunks = []
        metadata = []
        
        # Split into paragraphs first
        paragraphs = self.paragraph_breaks.split(text)
        
        current_chunk = ""
        current_sentences = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Split paragraph into sentences
            sentences = self.sentence_endings.split(paragraph)
            
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence would exceed chunk size
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(potential_chunk) > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(current_chunk.strip())
                    metadata.append({
                        'chunk_type': 'semantic',
                        'paragraph_range': (max(0, para_idx - len(current_sentences)), para_idx),
                        'sentence_count': len(current_sentences),
                        'chunk_size': len(current_chunk)
                    })
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and current_sentences:
                        overlap_sentences = current_sentences[-2:] if len(current_sentences) > 1 else current_sentences
                        current_chunk = " ".join(overlap_sentences) + " " + sentence
                        current_sentences = overlap_sentences + [sentence]
                    else:
                        current_chunk = sentence
                        current_sentences = [sentence]
                else:
                    current_chunk = potential_chunk
                    current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            metadata.append({
                'chunk_type': 'semantic',
                'paragraph_range': (len(paragraphs) - 1, len(paragraphs)),
                'sentence_count': len(current_sentences),
                'chunk_size': len(current_chunk)
            })
        
        return chunks, metadata
    
    def _chunk_fixed_size(self, text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Fixed size chunking with overlap."""
        chunks = []
        metadata = []
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundaries
            if end < len(text):
                # Look for space within last 50 characters
                space_pos = text.rfind(' ', max(start, end - 50), end)
                if space_pos > start:
                    end = space_pos
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({
                    'chunk_type': 'fixed_size',
                    'chunk_index': chunk_index,
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_size': len(chunk)
                })
                chunk_index += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks, metadata
    
    def _chunk_sentence(self, text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Sentence-based chunking."""
        sentences = self.sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        metadata = []
        
        current_chunk = ""
        sentence_count = 0
        
        for i, sentence in enumerate(sentences):
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                metadata.append({
                    'chunk_type': 'sentence',
                    'sentence_count': sentence_count,
                    'sentence_range': (i - sentence_count, i),
                    'chunk_size': len(current_chunk)
                })
                
                current_chunk = sentence
                sentence_count = 1
            else:
                current_chunk = potential_chunk
                sentence_count += 1
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            metadata.append({
                'chunk_type': 'sentence',
                'sentence_count': sentence_count,
                'sentence_range': (len(sentences) - sentence_count, len(sentences)),
                'chunk_size': len(current_chunk)
            })
        
        return chunks, metadata
    
    def _chunk_paragraph(self, text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Paragraph-based chunking."""
        paragraphs = self.paragraph_breaks.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        metadata = []
        
        current_chunk = ""
        paragraph_count = 0
        
        for i, paragraph in enumerate(paragraphs):
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                metadata.append({
                    'chunk_type': 'paragraph',
                    'paragraph_count': paragraph_count,
                    'paragraph_range': (i - paragraph_count, i),
                    'chunk_size': len(current_chunk)
                })
                
                current_chunk = paragraph
                paragraph_count = 1
            else:
                current_chunk = potential_chunk
                paragraph_count += 1
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            metadata.append({
                'chunk_type': 'paragraph',
                'paragraph_count': paragraph_count,
                'paragraph_range': (len(paragraphs) - paragraph_count, len(paragraphs)),
                'chunk_size': len(current_chunk)
            })
        
        return chunks, metadata
    
    def _chunk_sliding_window(self, text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Sliding window chunking."""
        chunks = []
        metadata = []
        
        step_size = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
                metadata.append({
                    'chunk_type': 'sliding_window',
                    'chunk_index': chunk_index,
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_size': len(chunk),
                    'overlap_size': self.chunk_overlap if start > 0 else 0
                })
                chunk_index += 1
            
            start += step_size
            
            # Stop if we've reached the end
            if end >= len(text):
                break
        
        return chunks, metadata
    
    def _chunk_adaptive(self, text: str, document_type: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Adaptive chunking based on document type."""
        # Choose strategy based on document type
        if document_type in ['research_paper', 'academic']:
            return self._chunk_semantic(text)
        elif document_type in ['financial_report', 'legal']:
            return self._chunk_paragraph(text)
        elif document_type in ['technical_manual', 'code']:
            return self._chunk_sentence(text)
        else:
            return self._chunk_semantic(text)  # Default to semantic
    
    def _split_oversized_chunk(self, chunk: str) -> List[str]:
        """Split an oversized chunk into smaller pieces."""
        if len(chunk) <= self.max_chunk_size:
            return [chunk]
        
        sub_chunks = []
        start = 0
        
        while start < len(chunk):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(chunk):
                sentence_end = chunk.rfind('.', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            sub_chunk = chunk[start:end].strip()
            if sub_chunk:
                sub_chunks.append(sub_chunk)
            
            start = end
        
        return sub_chunks

def get_v2_chunker(default_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
                  chunk_size: int = 512,
                  chunk_overlap: int = 50,
                  min_chunk_size: int = 50,
                  max_chunk_size: int = 1024) -> V2DocumentChunker:
    """
    Get or create a v2 document chunker instance.
    
    Args:
        default_strategy: Default chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        
    Returns:
        V2DocumentChunker instance
    """
    global _v2_chunker
    
    if _v2_chunker is None:
        _v2_chunker = V2DocumentChunker(
            default_strategy=default_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )
    
    return _v2_chunker
