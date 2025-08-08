"""
Query Router for SAM
Intelligent routing of user queries to appropriate knowledge sources.

Sprint 14 Task 4: Query Routing Enhancement
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries for routing."""
    DOCUMENT_SPECIFIC = "document_specific"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MEMORY_SEARCH = "memory_search"
    TECHNICAL_HELP = "technical_help"
    CONVERSATIONAL = "conversational"

@dataclass
class QueryAnalysis:
    """Analysis result for a user query."""
    query_type: QueryType
    confidence: float
    document_indicators: List[str]
    memory_indicators: List[str]
    technical_indicators: List[str]
    suggested_sources: List[str]
    search_strategy: str
    reasoning: str

class QueryRouter:
    """
    Routes user queries to appropriate knowledge sources based on content analysis.
    """
    
    def __init__(self):
        """Initialize the query router."""
        self.document_patterns = [
            # Direct document references
            r'\b(?:document|file|pdf|upload|uploaded|attachment)\b',
            r'\b(?:according to|based on|from the|in the)\b.*\b(?:document|file|paper)\b',

            # Content-specific patterns
            r'\b(?:table|chart|graph|figure|image)\b.*\b(?:shows|contains|displays)\b',
            r'\b(?:section|chapter|page|paragraph)\b.*\b(?:says|mentions|states)\b',

            # Question patterns about documents
            r'\b(?:what does|what is|how does)\b.*\b(?:document|file|paper)\b',
            r'\b(?:summarize|summary of|explain)\b.*\b(?:document|content|file)\b',

            # Enhanced document analysis patterns
            r'\b(?:give me a|can you give|provide a)\b.*\b(?:summary|overview|brief)\b',
            r'\b(?:what.*contains|what.*document.*contains|document.*content)\b',
            r'\b(?:brief summary|summary of what|what this.*contains)\b',
            r'\b(?:analyze|tell me about|explain)\b.*\b(?:this|the)\b',
            r'\b(?:uploaded.*file|just uploaded|i.*uploaded)\b',

            # Specific document reference patterns
            r'\b(?:this document|the document|the file|the pdf|the paper)\b',
            r'\b(?:what i uploaded|what i shared|the attachment)\b',
            r'\b(?:document analysis|file analysis|content analysis)\b',

            # Code and technical content
            r'\b(?:code|function|class|method|variable)\b.*\b(?:in the|from the)\b',
            r'\b(?:implementation|algorithm|approach)\b.*\b(?:shown|described|used)\b'
        ]
        
        self.memory_patterns = [
            # Memory-specific queries
            r'\b(?:remember|recall|previous|earlier|before)\b',
            r'\b(?:we discussed|you mentioned|last time|previously)\b',
            r'\b(?:conversation|chat|talked about|spoke about)\b',
            
            # Context references
            r'\b(?:context|background|history|past)\b',
            r'\b(?:what did|when did|where did)\b.*\b(?:we|I|you)\b'
        ]
        
        self.technical_patterns = [
            # Programming and technical terms
            r'\b(?:python|javascript|java|c\+\+|sql|html|css|api|database)\b',
            r'\b(?:function|class|method|variable|array|object|string)\b',
            r'\b(?:algorithm|data structure|framework|library|package)\b',
            r'\b(?:debug|error|exception|bug|issue|problem)\b',
            
            # Technical questions
            r'\b(?:how to|how do I|how can I)\b.*\b(?:code|program|implement|build)\b',
            r'\b(?:what is|explain|define)\b.*\b(?:programming|technical|computer)\b'
        ]
        
        self.general_patterns = [
            # General knowledge questions
            r'\b(?:what is|who is|when is|where is|why is|how is)\b',
            r'\b(?:explain|define|describe|tell me about)\b',
            r'\b(?:history|science|mathematics|physics|chemistry|biology)\b',
            r'\b(?:world|country|city|culture|language|religion)\b'
        ]
        
        logger.info("Query router initialized")
    
    def analyze_query(self, query: str, available_documents: Optional[List[str]] = None) -> QueryAnalysis:
        """
        Analyze a user query to determine routing strategy.
        
        Args:
            query: User's query text
            available_documents: List of available document names/sources
            
        Returns:
            QueryAnalysis with routing recommendations
        """
        try:
            query_lower = query.lower()
            
            # Check for document indicators
            document_indicators = []
            for pattern in self.document_patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                document_indicators.extend(matches)
            
            # Check for memory indicators
            memory_indicators = []
            for pattern in self.memory_patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                memory_indicators.extend(matches)
            
            # Check for technical indicators
            technical_indicators = []
            for pattern in self.technical_patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                technical_indicators.extend(matches)
            
            # Check for specific document references
            document_name_matches = []
            if available_documents:
                for doc in available_documents:
                    doc_name = doc.lower()
                    if doc_name in query_lower or any(part in query_lower for part in doc_name.split('.')):
                        document_name_matches.append(doc)
            
            # Determine query type and confidence
            query_type, confidence, reasoning = self._classify_query(
                query_lower, document_indicators, memory_indicators, 
                technical_indicators, document_name_matches
            )
            
            # Determine suggested sources and search strategy
            suggested_sources, search_strategy = self._determine_search_strategy(
                query_type, document_indicators, memory_indicators, technical_indicators
            )
            
            return QueryAnalysis(
                query_type=query_type,
                confidence=confidence,
                document_indicators=document_indicators,
                memory_indicators=memory_indicators,
                technical_indicators=technical_indicators,
                suggested_sources=suggested_sources,
                search_strategy=search_strategy,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return safe default
            return QueryAnalysis(
                query_type=QueryType.GENERAL_KNOWLEDGE,
                confidence=0.5,
                document_indicators=[],
                memory_indicators=[],
                technical_indicators=[],
                suggested_sources=["memory", "general"],
                search_strategy="broad_search",
                reasoning="Error in analysis, using default routing"
            )
    
    def _classify_query(self, query_lower: str, document_indicators: List[str], 
                       memory_indicators: List[str], technical_indicators: List[str],
                       document_name_matches: List[str]) -> Tuple[QueryType, float, str]:
        """Classify the query type with confidence score."""
        
        # Score different aspects
        document_score = len(document_indicators) + len(document_name_matches) * 2
        memory_score = len(memory_indicators)
        technical_score = len(technical_indicators)
        
        # Check for explicit document references
        if document_name_matches:
            return QueryType.DOCUMENT_SPECIFIC, 0.9, f"Direct reference to documents: {document_name_matches}"
        
        # Strong document indicators
        if document_score >= 2:
            return QueryType.DOCUMENT_SPECIFIC, 0.8, f"Multiple document indicators: {document_indicators}"
        
        # Strong memory indicators
        if memory_score >= 2:
            return QueryType.MEMORY_SEARCH, 0.8, f"Multiple memory indicators: {memory_indicators}"
        
        # Technical content
        if technical_score >= 2:
            return QueryType.TECHNICAL_HELP, 0.7, f"Technical indicators: {technical_indicators}"
        
        # Single strong indicator
        if document_score == 1:
            return QueryType.DOCUMENT_SPECIFIC, 0.6, f"Single document indicator: {document_indicators}"
        
        if memory_score == 1:
            return QueryType.MEMORY_SEARCH, 0.6, f"Single memory indicator: {memory_indicators}"
        
        if technical_score == 1:
            return QueryType.TECHNICAL_HELP, 0.6, f"Single technical indicator: {technical_indicators}"
        
        # Check for conversational patterns
        conversational_patterns = [
            r'\b(?:hello|hi|hey|thanks|thank you|please|sorry)\b',
            r'\b(?:how are you|what\'s up|good morning|good afternoon)\b'
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, query_lower):
                return QueryType.CONVERSATIONAL, 0.7, "Conversational greeting or politeness"
        
        # Default to general knowledge
        return QueryType.GENERAL_KNOWLEDGE, 0.5, "No specific indicators found, defaulting to general knowledge"
    
    def _determine_search_strategy(self, query_type: QueryType, document_indicators: List[str],
                                 memory_indicators: List[str], technical_indicators: List[str]) -> Tuple[List[str], str]:
        """Determine search sources and strategy based on query type."""
        
        if query_type == QueryType.DOCUMENT_SPECIFIC:
            return ["memory_documents", "vector_store"], "document_focused"
        
        elif query_type == QueryType.MEMORY_SEARCH:
            return ["memory_conversations", "memory_documents"], "memory_focused"
        
        elif query_type == QueryType.TECHNICAL_HELP:
            return ["memory_documents", "vector_store", "general"], "technical_focused"
        
        elif query_type == QueryType.CONVERSATIONAL:
            return ["memory_conversations"], "conversational"
        
        else:  # GENERAL_KNOWLEDGE
            return ["memory", "vector_store", "general"], "broad_search"
    
    def get_search_filters(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Get search filters based on query analysis."""
        from memory.memory_vectorstore import MemoryType

        filters = {}

        if analysis.query_type == QueryType.DOCUMENT_SPECIFIC:
            filters["tags"] = ["document", "uploaded"]
            filters["memory_types"] = [MemoryType.DOCUMENT]  # Use enum instead of string

        elif analysis.query_type == QueryType.MEMORY_SEARCH:
            filters["tags"] = ["conversation", "user_interaction"]
            filters["memory_types"] = [MemoryType.CONVERSATION]  # Use enum instead of string

        elif analysis.query_type == QueryType.TECHNICAL_HELP:
            filters["tags"] = ["code", "technical", "programming"]
            filters["content_types"] = ["code", "technical"]

        return filters
    
    def should_search_documents(self, analysis: QueryAnalysis) -> bool:
        """Determine if document search should be prioritized."""
        return analysis.query_type in [
            QueryType.DOCUMENT_SPECIFIC,
            QueryType.TECHNICAL_HELP
        ] or analysis.confidence > 0.7
    
    def should_search_memory(self, analysis: QueryAnalysis) -> bool:
        """Determine if memory search should be prioritized."""
        return analysis.query_type in [
            QueryType.MEMORY_SEARCH,
            QueryType.DOCUMENT_SPECIFIC,
            QueryType.CONVERSATIONAL
        ]
    
    def get_routing_explanation(self, analysis: QueryAnalysis) -> str:
        """Get human-readable explanation of routing decision."""
        explanations = {
            QueryType.DOCUMENT_SPECIFIC: "🔍 Searching uploaded documents and files",
            QueryType.MEMORY_SEARCH: "💭 Searching conversation history and memory",
            QueryType.TECHNICAL_HELP: "⚙️ Searching technical documentation and code",
            QueryType.CONVERSATIONAL: "💬 Using conversational context",
            QueryType.GENERAL_KNOWLEDGE: "🌐 Using general knowledge and available sources"
        }
        
        base_explanation = explanations.get(analysis.query_type, "🔍 Using available knowledge sources")
        
        if analysis.confidence > 0.8:
            confidence_text = "High confidence"
        elif analysis.confidence > 0.6:
            confidence_text = "Medium confidence"
        else:
            confidence_text = "Low confidence"
        
        return f"{base_explanation} ({confidence_text}: {analysis.confidence:.2f})"

# Global router instance
_query_router = None

def get_query_router() -> QueryRouter:
    """Get or create a global query router instance."""
    global _query_router
    
    if _query_router is None:
        _query_router = QueryRouter()
    
    return _query_router
