#!/usr/bin/env python3
"""
Intelligent Document Pre-Processing Pipeline (IDPP) Coordinator
Main orchestrator for the complete document processing pipeline.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from .document_classifier import get_document_classifier, DocumentClassification
from .structure_mapper import get_structure_mapper, DocumentMap
from .intelligent_chunker import get_intelligent_chunker, IntelligentChunk
from .analyzers import get_analyzer_pipeline, ContentSignature

logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    """Complete processed document with all IDPP enhancements."""
    # Original document info
    filename: str
    content: str
    file_size: int
    
    # IDPP Analysis Results
    classification: DocumentClassification
    document_map: DocumentMap
    chunks: List[IntelligentChunk]
    
    # Processing metadata
    processing_time_ms: float
    total_chunks: int
    chunk_types: Dict[str, int]
    processing_stats: Dict[str, Any]
    
    # Quality metrics
    processing_quality_score: float
    optimization_applied: List[str]

class IDPPCoordinator:
    """
    Main coordinator for the Intelligent Document Pre-Processing Pipeline.
    Orchestrates the complete document analysis and processing workflow.
    """
    
    def __init__(self):
        # Initialize all components
        self.document_classifier = get_document_classifier()
        self.structure_mapper = get_structure_mapper()
        self.intelligent_chunker = get_intelligent_chunker()
        self.analyzer_pipeline = get_analyzer_pipeline()
        
        # Processing statistics
        self.documents_processed = 0
        self.total_processing_time = 0.0
        self.processing_stats = {
            'classification_time': 0.0,
            'structure_mapping_time': 0.0,
            'chunking_time': 0.0,
            'analysis_time': 0.0
        }
        
        logger.info("🚀 IDPP Coordinator initialized - Ready for intelligent document processing")
    
    def process_document(self, content: str, filename: str = "") -> ProcessedDocument:
        """
        Execute the complete IDPP pipeline on a document.
        
        Args:
            content: Full document text content
            filename: Original filename for additional context
            
        Returns:
            ProcessedDocument with complete analysis and intelligent chunks
        """
        start_time = time.time()
        
        try:
            logger.info(f"📄 Starting IDPP processing for '{filename}' ({len(content)} chars)")
            
            # Phase 1: Document Classification
            phase1_start = time.time()
            classification = self.document_classifier.classify_document(content, filename)
            phase1_time = (time.time() - phase1_start) * 1000
            
            logger.info(f"🏷️ Phase 1 Complete: {classification.document_type.value} "
                       f"(confidence: {classification.confidence:.2f}) in {phase1_time:.1f}ms")
            
            # Phase 2: Structure Mapping
            phase2_start = time.time()
            document_map = self.structure_mapper.map_document_structure(content)
            phase2_time = (time.time() - phase2_start) * 1000
            
            logger.info(f"🗺️ Phase 2 Complete: {len(document_map.elements)} elements, "
                       f"{len(document_map.sections)} sections in {phase2_time:.1f}ms")
            
            # Phase 3: Intelligent Chunking
            phase3_start = time.time()
            chunks = self.intelligent_chunker.chunk_document(content, classification, document_map)
            phase3_time = (time.time() - phase3_start) * 1000
            
            logger.info(f"🔪 Phase 3 Complete: {len(chunks)} intelligent chunks in {phase3_time:.1f}ms")
            
            # Phase 4: Content Analysis & Enrichment
            phase4_start = time.time()
            enriched_chunks = self._enrich_chunks_with_analysis(chunks)
            phase4_time = (time.time() - phase4_start) * 1000
            
            logger.info(f"🧠 Phase 4 Complete: Chunks enriched with metadata in {phase4_time:.1f}ms")
            
            # Calculate processing statistics
            total_time = (time.time() - start_time) * 1000
            
            processing_stats = {
                'classification_time_ms': phase1_time,
                'structure_mapping_time_ms': phase2_time,
                'chunking_time_ms': phase3_time,
                'analysis_time_ms': phase4_time,
                'total_time_ms': total_time
            }
            
            # Create processed document
            processed_doc = ProcessedDocument(
                filename=filename,
                content=content,
                file_size=len(content),
                classification=classification,
                document_map=document_map,
                chunks=enriched_chunks,
                processing_time_ms=total_time,
                total_chunks=len(enriched_chunks),
                chunk_types=self._get_chunk_type_distribution(enriched_chunks),
                processing_stats=processing_stats,
                processing_quality_score=self._calculate_processing_quality(
                    classification, document_map, enriched_chunks
                ),
                optimization_applied=self._get_optimizations_applied(
                    classification, document_map, enriched_chunks
                )
            )
            
            # Update global statistics
            self._update_global_stats(processing_stats)
            
            logger.info(f"✅ IDPP Complete: '{filename}' processed in {total_time:.1f}ms "
                       f"(Quality: {processed_doc.processing_quality_score:.2f})")
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"❌ IDPP processing failed for '{filename}': {e}")
            # Return minimal processed document for graceful degradation
            return self._create_fallback_processed_document(content, filename, str(e))
    
    def _enrich_chunks_with_analysis(self, chunks: List[IntelligentChunk]) -> List[IntelligentChunk]:
        """Enrich chunks with content analysis metadata."""
        enriched_chunks = []
        
        for chunk in chunks:
            try:
                # Run content analysis pipeline
                content_signature = self.analyzer_pipeline.analyze_chunk(chunk.content)
                
                # Update chunk with analysis results
                chunk.contains_formulas = content_signature.contains_formulas
                chunk.contains_tables = content_signature.contains_tables
                chunk.contains_code = content_signature.contains_code
                chunk.contains_numerical_data = (
                    content_signature.contains_financial_data or 
                    bool(content_signature.numerical_values)
                )
                chunk.contains_definitions = content_signature.contains_definitions
                
                # Add analysis metadata to processing hints
                chunk.processing_hints.update({
                    'content_signature': content_signature,
                    'complexity_score': content_signature.complexity_score,
                    'technical_level': content_signature.technical_level,
                    'domain_indicators': content_signature.domain_indicators,
                    'query_keywords': list(content_signature.query_keywords),
                    'semantic_tags': content_signature.semantic_tags
                })
                
                # Adjust priority based on content analysis
                if content_signature.complexity_score > 0.7:
                    chunk.priority_level = max(chunk.priority_level, 2)
                
                if (content_signature.contains_formulas or 
                    content_signature.contains_financial_data):
                    chunk.priority_level = max(chunk.priority_level, 2)
                
                enriched_chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"Chunk analysis failed for chunk {chunk.chunk_id}: {e}")
                # Add chunk without enrichment
                enriched_chunks.append(chunk)
        
        return enriched_chunks
    
    def _get_chunk_type_distribution(self, chunks: List[IntelligentChunk]) -> Dict[str, int]:
        """Get distribution of chunk types."""
        distribution = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            distribution[chunk_type] = distribution.get(chunk_type, 0) + 1
        return distribution
    
    def _calculate_processing_quality(self, classification: DocumentClassification,
                                    document_map: DocumentMap, 
                                    chunks: List[IntelligentChunk]) -> float:
        """Calculate overall processing quality score."""
        quality_factors = []
        
        # Classification quality
        quality_factors.append(classification.confidence)
        
        # Structure mapping quality
        structure_quality = min(len(document_map.elements) / 10, 1.0)  # More elements = better structure
        quality_factors.append(structure_quality)
        
        # Chunking quality
        if chunks:
            # Check for appropriate chunk sizes
            avg_chunk_size = sum(len(chunk.content) for chunk in chunks) / len(chunks)
            size_quality = 1.0 if 200 <= avg_chunk_size <= 1500 else 0.7
            quality_factors.append(size_quality)
            
            # Check for content type detection
            specialized_chunks = sum(1 for chunk in chunks if chunk.chunk_type != "content")
            specialization_quality = min(specialized_chunks / len(chunks), 0.5) + 0.5
            quality_factors.append(specialization_quality)
        else:
            quality_factors.extend([0.5, 0.5])
        
        return sum(quality_factors) / len(quality_factors)
    
    def _get_optimizations_applied(self, classification: DocumentClassification,
                                 document_map: DocumentMap,
                                 chunks: List[IntelligentChunk]) -> List[str]:
        """Get list of optimizations that were applied."""
        optimizations = []
        
        # Document type specific optimizations
        if classification.document_type.value != 'general_text':
            optimizations.append(f"specialized_{classification.document_type.value}_processing")
        
        # Structure-based optimizations
        if document_map.atomic_blocks:
            optimizations.append("atomic_block_preservation")
        
        if document_map.sections:
            optimizations.append("section_aware_chunking")
        
        # Content-based optimizations
        formula_chunks = sum(1 for chunk in chunks if chunk.contains_formulas)
        if formula_chunks > 0:
            optimizations.append("formula_preservation")
        
        table_chunks = sum(1 for chunk in chunks if chunk.contains_tables)
        if table_chunks > 0:
            optimizations.append("table_integrity_preservation")
        
        code_chunks = sum(1 for chunk in chunks if chunk.contains_code)
        if code_chunks > 0:
            optimizations.append("code_block_preservation")
        
        return optimizations
    
    def _update_global_stats(self, processing_stats: Dict[str, float]):
        """Update global processing statistics."""
        self.documents_processed += 1
        self.total_processing_time += processing_stats['total_time_ms']
        
        for key, value in processing_stats.items():
            if key in self.processing_stats:
                # Running average
                current_avg = self.processing_stats[key]
                new_avg = (current_avg * (self.documents_processed - 1) + value) / self.documents_processed
                self.processing_stats[key] = new_avg
    
    def _create_fallback_processed_document(self, content: str, filename: str, 
                                          error: str) -> ProcessedDocument:
        """Create a minimal processed document when processing fails."""
        from .document_classifier import DocumentType, DocumentClassification
        from .structure_mapper import DocumentMap
        
        # Create minimal classification
        fallback_classification = DocumentClassification(
            document_type=DocumentType.GENERAL_TEXT,
            confidence=0.5,
            content_patterns=[],
            structure=None,
            processing_hints={'error': error},
            metadata={'processing_error': error}
        )
        
        # Create minimal document map
        fallback_map = DocumentMap([], {}, [], [], {})
        
        # Create basic chunks (fallback chunking)
        words = content.split()
        chunk_size = 800
        fallback_chunks = []
        
        current_chunk = ""
        chunk_id = 0
        
        for word in words:
            if len(current_chunk) + len(word) > chunk_size and current_chunk:
                fallback_chunks.append(IntelligentChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"fallback_{chunk_id}",
                    start_pos=0,
                    end_pos=len(current_chunk),
                    chunk_type="content"
                ))
                chunk_id += 1
                current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word
        
        if current_chunk.strip():
            fallback_chunks.append(IntelligentChunk(
                content=current_chunk.strip(),
                chunk_id=f"fallback_{chunk_id}",
                start_pos=0,
                end_pos=len(current_chunk),
                chunk_type="content"
            ))
        
        return ProcessedDocument(
            filename=filename,
            content=content,
            file_size=len(content),
            classification=fallback_classification,
            document_map=fallback_map,
            chunks=fallback_chunks,
            processing_time_ms=0.0,
            total_chunks=len(fallback_chunks),
            chunk_types={'content': len(fallback_chunks)},
            processing_stats={'error': error},
            processing_quality_score=0.3,  # Low quality due to fallback
            optimization_applied=['fallback_processing']
        )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            'documents_processed': self.documents_processed,
            'total_processing_time_ms': self.total_processing_time,
            'average_processing_time_ms': (
                self.total_processing_time / self.documents_processed 
                if self.documents_processed > 0 else 0
            ),
            'phase_averages': self.processing_stats.copy(),
            'throughput_docs_per_second': (
                self.documents_processed / (self.total_processing_time / 1000)
                if self.total_processing_time > 0 else 0
            )
        }
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.documents_processed = 0
        self.total_processing_time = 0.0
        self.processing_stats = {
            'classification_time': 0.0,
            'structure_mapping_time': 0.0,
            'chunking_time': 0.0,
            'analysis_time': 0.0
        }
        logger.info("📊 Processing statistics reset")

# Global instance
_idpp_coordinator = None

def get_idpp_coordinator() -> IDPPCoordinator:
    """Get or create the global IDPP coordinator instance."""
    global _idpp_coordinator
    if _idpp_coordinator is None:
        _idpp_coordinator = IDPPCoordinator()
    return _idpp_coordinator
