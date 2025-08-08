#!/usr/bin/env python3
"""
SAM Intelligent Document Pre-Processing Pipeline (IDPP)
Advanced document ingestion with content-aware analysis and intelligent chunking.
"""

from .document_classifier import (
    DocumentClassifier, 
    DocumentType, 
    DocumentClassification,
    get_document_classifier
)

from .structure_mapper import (
    StructureMapper,
    DocumentMap,
    StructuralElement,
    DocumentElement,
    get_structure_mapper
)

from .intelligent_chunker import (
    IntelligentChunker,
    IntelligentChunk,
    get_intelligent_chunker
)

from .analyzers import (
    ContentAnalyzerPipeline,
    ContentSignature,
    FormulaAnalyzer,
    TableAnalyzer,
    FinancialAnalyzer,
    CodeAnalyzer,
    DefinitionAnalyzer,
    get_analyzer_pipeline
)

from .idpp_coordinator import (
    IDPPCoordinator,
    ProcessedDocument,
    get_idpp_coordinator
)

# v2 Ingestion Pipeline Components
from .v2_document_processor import (
    V2DocumentProcessor,
    V2ProcessingResult,
    get_v2_document_processor,
    process_document_v2
)

from .v2_ingestion_pipeline import (
    V2IngestionPipeline,
    V2IngestionConfig,
    get_v2_ingestion_pipeline,
    ingest_document_v2
)

from .document_chunker import (
    V2DocumentChunker,
    ChunkingStrategy,
    ChunkResult,
    get_v2_chunker
)

__all__ = [
    # Document Classification
    'DocumentClassifier',
    'DocumentType', 
    'DocumentClassification',
    'get_document_classifier',
    
    # Structure Mapping
    'StructureMapper',
    'DocumentMap',
    'StructuralElement',
    'DocumentElement',
    'get_structure_mapper',
    
    # Intelligent Chunking
    'IntelligentChunker',
    'IntelligentChunk',
    'get_intelligent_chunker',
    
    # Content Analysis
    'ContentAnalyzerPipeline',
    'ContentSignature',
    'FormulaAnalyzer',
    'TableAnalyzer',
    'FinancialAnalyzer',
    'CodeAnalyzer',
    'DefinitionAnalyzer',
    'get_analyzer_pipeline',
    
    # Main Coordinator
    'IDPPCoordinator',
    'ProcessedDocument',
    'get_idpp_coordinator',

    # v2 Pipeline Components
    'V2DocumentProcessor',
    'V2ProcessingResult',
    'get_v2_document_processor',
    'process_document_v2',
    'V2IngestionPipeline',
    'V2IngestionConfig',
    'get_v2_ingestion_pipeline',
    'ingest_document_v2',
    'V2DocumentChunker',
    'ChunkingStrategy',
    'ChunkResult',
    'get_v2_chunker'
]

# Version info
__version__ = "1.0.0"
__author__ = "SAM Development Team"
__description__ = "Intelligent Document Pre-Processing Pipeline for SAM"
