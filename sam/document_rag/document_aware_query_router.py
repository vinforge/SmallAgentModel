#!/usr/bin/env python3
"""
Document-Aware Query Router
The "first brain" that determines whether to prioritize document context or fallback to general knowledge.
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .semantic_document_search import SemanticDocumentSearchEngine, DocumentSearchResult
from .document_context_assembler import DocumentContextAssembler, AssembledContext

logger = logging.getLogger(__name__)

class QueryStrategy(Enum):
    """Query routing strategies."""
    DOCUMENT_ONLY = "document_only"
    DOCUMENT_PRIORITY = "document_priority"
    HYBRID = "hybrid"
    GENERAL_KNOWLEDGE = "general_knowledge"

@dataclass
class RoutingDecision:
    """Decision made by the query router."""
    strategy: QueryStrategy
    confidence_level: str
    document_context: Optional[AssembledContext]
    reasoning: str
    should_search_web: bool
    fallback_strategy: Optional[QueryStrategy]
    routing_metadata: Dict[str, Any]

class DocumentAwareQueryRouter:
    """
    Routes queries to appropriate knowledge sources with document-first strategy.
    This is the "first brain" that decides where to look for information.
    """
    
    def __init__(self,
                 search_engine: SemanticDocumentSearchEngine,
                 context_assembler: DocumentContextAssembler,
                 high_confidence_threshold: float = 0.75,  # Lowered from 0.85
                 medium_confidence_threshold: float = 0.50,  # Lowered from 0.65
                 low_confidence_threshold: float = 0.30):   # Lowered from 0.45
        """
        Initialize the document-aware query router.
        
        Args:
            search_engine: Semantic document search engine
            context_assembler: Document context assembler
            high_confidence_threshold: Threshold for high confidence document context
            medium_confidence_threshold: Threshold for medium confidence
            low_confidence_threshold: Threshold for low confidence
        """
        self.search_engine = search_engine
        self.context_assembler = context_assembler
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Query pattern detection
        self.document_reference_patterns = [
            r'\b(?:document|file|paper|pdf|doc|report)\b',
            r'\b(?:uploaded|attached|provided)\b',
            r'\b(?:according to|based on|from)\s+(?:the|my|our)\s+(?:document|file|paper)\b',
            r'\.(?:pdf|docx?|txt|md)\b',
        ]
        
        self.explicit_document_patterns = [
            r'\bwhat does (?:the|my|our) (?:document|file|paper) say\b',
            r'\bin (?:the|my|our) (?:document|file|paper)\b',
            r'\bfrom (?:the|my|our) uploaded (?:document|file|paper)\b',
        ]
    
    def route_query(self, 
                   query: str, 
                   max_document_results: int = 5,
                   user_context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Route a query to the appropriate knowledge source with document-first strategy.
        
        Args:
            query: User's query
            max_document_results: Maximum document chunks to retrieve
            user_context: Additional user context
            
        Returns:
            RoutingDecision with strategy and context
        """
        try:
            self.logger.info(f"🧠 Routing query: '{query[:50]}...'")
            
            # Step 1: Always search documents first (document-first strategy)
            document_search_result = self.search_engine.search_uploaded_documents(
                query=query,
                max_results=max_document_results
            )
            
            # Step 2: Assemble document context if found
            document_context = None
            if document_search_result.chunks:
                document_context = self.context_assembler.assemble_context(document_search_result)
            
            # Step 3: Analyze query characteristics
            query_analysis = self._analyze_query(query)
            
            # Step 4: Make routing decision based on document relevance and query type
            routing_decision = self._make_routing_decision(
                query=query,
                document_search_result=document_search_result,
                document_context=document_context,
                query_analysis=query_analysis,
                user_context=user_context
            )
            
            self.logger.info(f"📍 Routing decision: {routing_decision.strategy.value} ({routing_decision.confidence_level} confidence)")
            self.logger.info(f"💭 Reasoning: {routing_decision.reasoning}")
            
            return routing_decision
            
        except Exception as e:
            self.logger.error(f"❌ Query routing failed: {e}")
            return self._create_fallback_decision(query, str(e))
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics to inform routing decisions."""
        query_lower = query.lower()
        
        # Check for explicit document references
        has_explicit_doc_ref = any(
            re.search(pattern, query_lower) 
            for pattern in self.explicit_document_patterns
        )
        
        # Check for implicit document references
        has_implicit_doc_ref = any(
            re.search(pattern, query_lower) 
            for pattern in self.document_reference_patterns
        )
        
        # Analyze query type
        query_type = self._classify_query_type(query_lower)
        
        # Check for specific document names or filenames
        has_filename_ref = bool(re.search(r'\b\w+\.(?:pdf|docx?|txt|md)\b', query_lower))
        
        return {
            'has_explicit_document_reference': has_explicit_doc_ref,
            'has_implicit_document_reference': has_implicit_doc_ref,
            'has_filename_reference': has_filename_ref,
            'query_type': query_type,
            'query_length': len(query.split()),
            'is_question': query.strip().endswith('?'),
            'has_comparison_words': any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']),
            'has_summary_words': any(word in query_lower for word in ['summarize', 'summary', 'overview', 'explain'])
        }
    
    def _classify_query_type(self, query_lower: str) -> str:
        """Classify the type of query."""
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'factual'
        elif any(word in query_lower for word in ['compare', 'analyze', 'evaluate', 'assess']):
            return 'analytical'
        elif any(word in query_lower for word in ['summarize', 'overview', 'explain']):
            return 'summary'
        elif any(word in query_lower for word in ['create', 'generate', 'write', 'make']):
            return 'creative'
        else:
            return 'general'
    
    def _make_routing_decision(self,
                              query: str,
                              document_search_result: DocumentSearchResult,
                              document_context: Optional[AssembledContext],
                              query_analysis: Dict[str, Any],
                              user_context: Optional[Dict[str, Any]]) -> RoutingDecision:
        """Make the routing decision based on all available information."""

        # Get document relevance score
        doc_relevance = document_search_result.highest_relevance_score

        # SESSION-BASED PRIORITIZATION: Check for recent uploads
        has_recent_upload = any(
            chunk.metadata.get('upload_method', '') in [
                'streamlit', 'terminal', 'content_recall_test', 'search_prioritization_fix',
                'final_proof', 'comprehensive_fix', 'terminal_proof', 'complete_prioritization_fix'
            ]
            for chunk in document_search_result.chunks
        )

        logger.info(f"📤 Recent upload detected: {has_recent_upload}")

        # SESSION-BASED ROUTING: Recent uploads get absolute priority
        if has_recent_upload and document_search_result.chunks:
            # Force document-only strategy for recent uploads
            strategy = QueryStrategy.DOCUMENT_ONLY
            reasoning = f"Recent upload detected - forcing document-only strategy (relevance: {doc_relevance:.2f})"
            should_search_web = False
            fallback = None
            confidence = "HIGH"
            logger.info(f"🚀 Forcing document-only strategy for recent upload")

        # Standard decision logic based on document relevance and query characteristics
        elif doc_relevance >= self.high_confidence_threshold:
            # High confidence in document relevance
            if query_analysis['has_explicit_document_reference']:
                strategy = QueryStrategy.DOCUMENT_ONLY
                reasoning = f"Explicit document reference with high relevance ({doc_relevance:.2f})"
                should_search_web = False
                fallback = None
            else:
                strategy = QueryStrategy.DOCUMENT_PRIORITY
                reasoning = f"High document relevance ({doc_relevance:.2f}), prioritizing documents"
                should_search_web = False
                fallback = QueryStrategy.GENERAL_KNOWLEDGE
            confidence = "HIGH"
            
        elif doc_relevance >= self.medium_confidence_threshold:
            # Medium confidence - hybrid approach
            strategy = QueryStrategy.HYBRID
            reasoning = f"Medium document relevance ({doc_relevance:.2f}), using hybrid approach"
            should_search_web = query_analysis['query_type'] in ['factual', 'analytical']
            fallback = QueryStrategy.GENERAL_KNOWLEDGE
            confidence = "MEDIUM"
            
        elif doc_relevance >= self.low_confidence_threshold:
            # Low confidence - supplement with general knowledge
            if query_analysis['has_explicit_document_reference']:
                strategy = QueryStrategy.DOCUMENT_PRIORITY
                reasoning = f"Explicit document reference but low relevance ({doc_relevance:.2f})"
                should_search_web = True
                fallback = QueryStrategy.GENERAL_KNOWLEDGE
            else:
                strategy = QueryStrategy.GENERAL_KNOWLEDGE
                reasoning = f"Low document relevance ({doc_relevance:.2f}), using general knowledge"
                should_search_web = True
                fallback = None
            confidence = "LOW"
            
        else:
            # Very low or no document relevance
            strategy = QueryStrategy.GENERAL_KNOWLEDGE
            reasoning = f"Very low document relevance ({doc_relevance:.2f}), using general knowledge"
            should_search_web = True
            fallback = None
            confidence = "VERY_LOW"
        
        return RoutingDecision(
            strategy=strategy,
            confidence_level=confidence,
            document_context=document_context,
            reasoning=reasoning,
            should_search_web=should_search_web,
            fallback_strategy=fallback,
            routing_metadata={
                'document_relevance_score': doc_relevance,
                'documents_found': document_search_result.total_chunks_found,
                'documents_searched': document_search_result.documents_searched,
                'query_analysis': query_analysis,
                'routing_timestamp': self._get_timestamp()
            }
        )
    
    def _create_fallback_decision(self, query: str, error: str) -> RoutingDecision:
        """Create a fallback decision when routing fails."""
        return RoutingDecision(
            strategy=QueryStrategy.GENERAL_KNOWLEDGE,
            confidence_level="ERROR",
            document_context=None,
            reasoning=f"Routing failed: {error}",
            should_search_web=True,
            fallback_strategy=None,
            routing_metadata={
                'error': error,
                'routing_timestamp': self._get_timestamp()
            }
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def should_use_document_context(self, routing_decision: RoutingDecision) -> bool:
        """Check if document context should be used based on routing decision."""
        return routing_decision.strategy in [
            QueryStrategy.DOCUMENT_ONLY,
            QueryStrategy.DOCUMENT_PRIORITY,
            QueryStrategy.HYBRID
        ] and routing_decision.document_context is not None
    
    def get_routing_summary(self, routing_decision: RoutingDecision) -> str:
        """Get a human-readable summary of the routing decision."""
        if routing_decision.document_context:
            doc_info = f" (found {routing_decision.document_context.source_count} relevant sources)"
        else:
            doc_info = " (no relevant documents found)"
        
        return f"Strategy: {routing_decision.strategy.value}{doc_info} - {routing_decision.reasoning}"
