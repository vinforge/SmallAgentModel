#!/usr/bin/env python3
"""
Enhanced Smart Query Router Service with STEF Integration
Intelligently routes all query types including search, calculations, tools, conversations, and structured tasks.
Evolved from SearchRouter to handle comprehensive intent classification, action routing, and STEF delegation.
"""

import logging
import re
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Types of search queries (legacy - maintained for compatibility)."""
    DOCUMENT = "document"
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    CORRECTION = "correction"
    GENERAL = "general"

class QueryIntent(Enum):
    """Enhanced query intent classification for smart routing."""
    # Search intents (existing functionality)
    DOCUMENT_SEARCH = "document_search"
    CONVERSATION_SEARCH = "conversation_search"
    KNOWLEDGE_SEARCH = "knowledge_search"
    CORRECTION_SEARCH = "correction_search"
    GENERAL_SEARCH = "general_search"

    # Action intents (new functionality)
    CALCULATION = "calculation"
    TOOL_REQUEST = "tool_request"
    GENERATION = "generation"
    CONVERSION = "conversion"

    # Hybrid intents
    HYBRID_DOC_CALC = "hybrid_doc_calc"  # "What's 15% of revenue in my document?"
    HYBRID_SEARCH_TOOL = "hybrid_search_tool"

    # Conversational intents
    CHAT = "chat"
    GREETING = "greeting"
    CLARIFICATION = "clarification"

class RouteType(Enum):
    """Types of routing decisions."""
    SEARCH = "search"           # Route to search services
    CALCULATE = "calculate"     # Route to calculator tool
    GENERATE = "generate"       # Route to generation tools
    CHAT = "chat"              # Route to conversational AI
    HYBRID = "hybrid"          # Route to multiple services
    CLARIFY = "clarify"        # Ask for clarification
    STEF = "stef"              # Route to STEF structured task execution

@dataclass
class IntentSignals:
    """Detected signals for intent classification."""
    math_signals: Dict[str, float] = field(default_factory=dict)
    document_signals: Dict[str, float] = field(default_factory=dict)
    tool_signals: Dict[str, float] = field(default_factory=dict)
    conversation_signals: Dict[str, float] = field(default_factory=dict)

    @property
    def strongest_signal_type(self) -> str:
        """Get the signal type with highest confidence."""
        all_signals = {
            'math': max(self.math_signals.values()) if self.math_signals else 0,
            'document': max(self.document_signals.values()) if self.document_signals else 0,
            'tool': max(self.tool_signals.values()) if self.tool_signals else 0,
            'conversation': max(self.conversation_signals.values()) if self.conversation_signals else 0
        }
        return max(all_signals.items(), key=lambda x: x[1])[0]

@dataclass
class SmartRoute:
    """Enhanced routing decision with multiple handlers and fallbacks."""
    route_type: RouteType
    primary_intent: QueryIntent
    confidence: float
    primary_handler: str
    fallback_handlers: List[str] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Dict[str, Any] = field(default_factory=dict)  # For calculations, document refs, etc.

    # Legacy compatibility
    search_type: Optional[SearchType] = None

class SmartQueryRouter:
    """
    Enhanced intelligent router that handles all query types including search, calculations,
    tools, and conversations. Evolved from SearchRouter with comprehensive intent classification.
    """

    def __init__(self):
        # Initialize STEF integration
        self._stef_enabled = True
        self._stef_executor = None
        self._program_registry = None

        # Legacy search keywords (maintained for compatibility)
        self.document_keywords = [
            'document', 'file', '.docx', '.pdf', '.txt', 'upload', 'uploaded',
            'sam story', 'chroma', 'ethan hayes', 'neural network', 'university lab',
            'project chroma', 'synthesis', 'summarize', 'summary', 'resume', 'continue'
        ]

        self.conversation_keywords = [
            'conversation', 'chat', 'discussion', 'talked about', 'mentioned',
            'said', 'previous', 'earlier', 'thread', 'session'
        ]

        self.correction_keywords = [
            'correction', 'fix', 'wrong', 'incorrect', 'mistake', 'error',
            'actually', 'should be', 'correct answer'
        ]

        self.knowledge_keywords = [
            'explain', 'what is', 'how does', 'definition', 'concept',
            'theory', 'principle', 'algorithm', 'method'
        ]

        # New: Mathematical operation keywords
        self.math_keywords = [
            'calculate', 'compute', 'what\'s', 'what is', 'how much', 'total',
            'sum', 'add', 'subtract', 'multiply', 'divide', 'percentage', 'percent',
            'interest', 'compound', 'simple', 'rate', 'return', 'profit', 'loss'
        ]

        # New: Tool request keywords
        self.tool_keywords = [
            'create', 'generate', 'make', 'build', 'design', 'draw', 'chart',
            'graph', 'image', 'picture', 'convert', 'transform', 'export'
        ]

        # New: Conversational keywords
        self.chat_keywords = [
            'hello', 'hi', 'hey', 'how are you', 'good morning', 'good afternoon',
            'thanks', 'thank you', 'please', 'help', 'joke', 'story', 'tell me'
        ]

        # Mathematical patterns for regex detection
        self.math_patterns = {
            'basic_arithmetic': r'\b\d+\s*[\+\-\*\/]\s*\d+',
            'percentage': r'\b\d+%\s+of\s+\d+|\b\d+\s+percent\s+of\s+\d+',
            'percentage_question': r'what\'s\s+\d+%\s+of|what\s+is\s+\d+%\s+of',
            'calculation_request': r'\b(calculate|compute|what\'s|what\s+is)\s+[\d\+\-\*\/\%\s]+',
            'financial_calc': r'\b(interest|loan|mortgage|investment|return|profit)\b.*\b\d+',
            'conversion': r'\b(convert|change|transform)\s+\d+.*\bto\b'
        }

        # ULTRA-AGGRESSIVE: Pure mathematical expression patterns
        self.pure_math_patterns = [
            r'^\s*\d+\s*[\+\-\*\/]\s*\d+\s*$',  # 99+1, 5*8, 100/4
            r'^\s*\d+\s*[\+\-\*\/]\s*\d+\s*[\+\-\*\/]\s*\d+\s*$',  # 5+3*2, 10-5+2
            r'^\s*\d+\s*[\+\-\*\/]\s*\d+\s*[\+\-\*\/]\s*\d+\s*[\+\-\*\/]\s*\d+\s*$',  # 1+2+3+4
            r'^\s*\(\s*\d+\s*[\+\-\*\/]\s*\d+\s*\)\s*[\+\-\*\/]\s*\d+\s*$',  # (5+3)*2
            r'^\s*\d+\s*[\+\-\*\/]\s*\(\s*\d+\s*[\+\-\*\/]\s*\d+\s*\)\s*$',  # 2*(5+3)
            r'^\s*what\'s\s+\d+\s*[\+\-\*\/]\s*\d+\s*\??\s*$',  # what's 99+1?
            r'^\s*what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\s*\??\s*$',  # what is 99+1?
            r'^\s*\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\??\s*$',  # 99+1=?
            r'^\s*\d+\s*[\+\-\*\/]\s*\d+\s*\?\s*$',  # 99+1?
            r'^\s*calculate\s+\d+\s*[\+\-\*\/]\s*\d+\s*$',  # calculate 99+1
            r'^\s*solve\s+\d+\s*[\+\-\*\/]\s*\d+\s*$',  # solve 99+1
        ]

        # Tool registry for available tools
        self.available_tools = {
            'calculator': 'calculator_tool',
            'document_search': 'document_search_service',
            'web_search': 'web_search_tool',
            'image_generator': 'image_generation_tool',
            'chart_creator': 'chart_creation_tool'
        }

        # Initialize STEF components lazily
        self._initialize_stef_components()

    def is_pure_math_expression(self, query: str) -> bool:
        """
        Ultra-aggressive detection for pure mathematical expressions.
        Returns True if query is ONLY a mathematical expression that should be calculated immediately.
        """
        if not query or len(query.strip()) == 0:
            return False

        # Clean the query
        cleaned = query.strip().lower()

        # Remove common question words and punctuation
        cleaned = re.sub(r'^(what\'s|what is|calculate|solve|compute)\s+', '', cleaned)
        cleaned = re.sub(r'\?+$', '', cleaned)  # Remove question marks
        cleaned = re.sub(r'=\s*\?*$', '', cleaned)  # Remove = and =?
        cleaned = cleaned.strip()

        # Check against pure math patterns
        for pattern in self.pure_math_patterns:
            if re.match(pattern, query.strip(), re.IGNORECASE):
                logger.info(f"🧮 PURE MATH DETECTED: '{query}' matches pattern")
                return True

        # Additional check: if cleaned query is just numbers and operators
        if re.match(r'^\d+(?:\s*[\+\-\*\/]\s*\d+)+$', cleaned):
            logger.info(f"🧮 PURE MATH DETECTED: '{query}' is clean arithmetic")
            return True

        return False

    def detect_pure_math_signals(self, query: str) -> Dict[str, float]:
        """
        Detect pure mathematical expressions with maximum confidence.
        This bypasses all other analysis for instant math routing.
        """
        signals = {}

        if self.is_pure_math_expression(query):
            signals['pure_math_expression'] = 1.0  # Maximum confidence
            signals['instant_calculation'] = 1.0   # Flag for immediate processing
            logger.info(f"🧮 PURE MATH SIGNALS: Maximum confidence for '{query}'")

        return signals

    def _initialize_stef_components(self):
        """Initialize STEF components lazily to avoid circular imports."""
        try:
            if self._stef_enabled:
                from sam.agents.stef import get_program_registry, STEF_Executor
                self._program_registry = get_program_registry()
                logger.info("✅ STEF components initialized successfully")
            else:
                logger.info("STEF integration disabled")
        except ImportError as e:
            logger.warning(f"STEF components not available: {e}")
            self._stef_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize STEF components: {e}")
            self._stef_enabled = False
    
    def detect_math_signals(self, query: str) -> Dict[str, float]:
        """
        Enhanced mathematical operation detection with ultra-aggressive pure math detection.
        Prioritizes calculator tool whenever numbers and calculation signs are present.
        """
        signals = {}
        query_lower = query.lower()

        # ULTRA-AGGRESSIVE: Check for pure mathematical expressions FIRST
        pure_math_signals = self.detect_pure_math_signals(query)
        if pure_math_signals:
            # Pure math gets maximum priority - skip all other analysis
            signals.update(pure_math_signals)
            logger.info(f"🧮 PURE MATH OVERRIDE: '{query}' gets maximum confidence")
            return signals

        # ENHANCED: Aggressive numerical + operator detection
        # Look for any combination of numbers with mathematical operators
        numbers_with_operators = re.findall(r'\d+(?:\.\d+)?\s*[\+\-\*\/\%]\s*\d+(?:\.\d+)?', query)
        if numbers_with_operators:
            signals['explicit_calculation'] = 1.0  # Highest priority
            logger.info(f"🧮 Explicit calculation detected: {numbers_with_operators}")

        # ENHANCED: Numbers with percentage signs
        percentage_with_numbers = re.findall(r'\d+(?:\.\d+)?%', query)
        if percentage_with_numbers:
            signals['percentage_notation'] = 0.9
            logger.info(f"🧮 Percentage notation detected: {percentage_with_numbers}")

        # ENHANCED: Mathematical operators near numbers (within 5 characters)
        operators = ['+', '-', '*', '/', '%', '=', '×', '÷']
        for i, char in enumerate(query):
            if char in operators:
                # Look for numbers within 5 characters before or after
                start = max(0, i - 5)
                end = min(len(query), i + 6)
                context = query[start:end]
                if re.search(r'\d', context):
                    signals['operator_near_numbers'] = 0.8
                    logger.info(f"🧮 Operator near numbers detected: '{context.strip()}'")
                    break

        # ENHANCED: Common calculation phrases with numbers
        calc_phrases_with_numbers = [
            r'\d+\s*(?:plus|add|added to)\s*\d+',
            r'\d+\s*(?:minus|subtract|less)\s*\d+',
            r'\d+\s*(?:times|multiplied by|x)\s*\d+',
            r'\d+\s*(?:divided by|over)\s*\d+',
            r'\d+\s*percent\s*of\s*\d+',
            r'\d+%\s*of\s*\d+',
            r'what\'s\s*\d+.*\d+',
            r'calculate\s*\d+',
            r'compute\s*\d+'
        ]

        for pattern in calc_phrases_with_numbers:
            if re.search(pattern, query_lower):
                signals['calculation_phrases'] = 0.9
                logger.info(f"🧮 Calculation phrase detected with pattern: {pattern}")
                break

        # Original pattern-based detection (enhanced)
        for pattern_name, pattern in self.math_patterns.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                signals[pattern_name] = len(matches) * 0.5  # Increased weight

        # Enhanced keyword-based detection
        math_keyword_count = sum(1 for keyword in self.math_keywords if keyword in query_lower)
        if math_keyword_count > 0:
            signals['math_keywords'] = math_keyword_count * 0.4  # Increased weight

        # ENHANCED: Number density with operator bonus
        digit_count = len(re.findall(r'\d', query))
        operator_count = sum(1 for op in operators if op in query)

        if digit_count > 1 and operator_count > 0:
            # Strong signal when both numbers and operators present
            signals['numbers_with_operators'] = min((digit_count * operator_count) * 0.2, 0.9)
            logger.info(f"🧮 Numbers + operators detected: {digit_count} digits, {operator_count} operators")
        elif digit_count > 3:
            # Moderate signal for high number density
            signals['number_density'] = min(digit_count * 0.1, 0.6)

        # ENHANCED: Standalone mathematical operators (strong signal)
        standalone_operators = sum(1 for op in ['+', '-', '*', '/', '%', '='] if op in query)
        if standalone_operators > 0:
            signals['standalone_operators'] = standalone_operators * 0.4

        # ENHANCED: Mathematical symbols and Unicode operators
        unicode_operators = ['×', '÷', '±', '√', '^', '²', '³']
        unicode_count = sum(1 for op in unicode_operators if op in query)
        if unicode_count > 0:
            signals['unicode_math'] = unicode_count * 0.6

        return signals

    def detect_document_signals(self, query: str) -> Dict[str, float]:
        """Detect document-related signals in the query."""
        signals = {}
        query_lower = query.lower()

        # Document keyword detection
        doc_keyword_count = sum(1 for keyword in self.document_keywords if keyword in query_lower)
        if doc_keyword_count > 0:
            signals['document_keywords'] = doc_keyword_count * 0.4

        # File extension detection
        file_extensions = ['.pdf', '.docx', '.txt', '.doc', '.md']
        for ext in file_extensions:
            if ext in query_lower:
                signals['file_extension'] = 0.6
                break

        # Document action words
        doc_actions = ['summarize', 'resume', 'continue', 'read', 'show', 'find in']
        action_count = sum(1 for action in doc_actions if action in query_lower)
        if action_count > 0:
            signals['document_actions'] = action_count * 0.3

        # Specific document references
        if 'sam story' in query_lower or 'my document' in query_lower:
            signals['specific_document'] = 0.7

        return signals

    def detect_tool_signals(self, query: str) -> Dict[str, float]:
        """Detect tool request signals in the query."""
        signals = {}
        query_lower = query.lower()

        # Tool keyword detection
        tool_keyword_count = sum(1 for keyword in self.tool_keywords if keyword in query_lower)
        if tool_keyword_count > 0:
            signals['tool_keywords'] = tool_keyword_count * 0.4

        # Creation/generation verbs
        creation_verbs = ['create', 'generate', 'make', 'build', 'design', 'draw']
        creation_count = sum(1 for verb in creation_verbs if verb in query_lower)
        if creation_count > 0:
            signals['creation_verbs'] = creation_count * 0.3

        # Output format requests
        output_formats = ['chart', 'graph', 'image', 'picture', 'table', 'list']
        format_count = sum(1 for fmt in output_formats if fmt in query_lower)
        if format_count > 0:
            signals['output_formats'] = format_count * 0.4

        return signals

    def detect_conversation_signals(self, query: str) -> Dict[str, float]:
        """Detect conversational signals in the query."""
        signals = {}
        query_lower = query.lower()

        # Chat keyword detection
        chat_keyword_count = sum(1 for keyword in self.chat_keywords if keyword in query_lower)
        if chat_keyword_count > 0:
            signals['chat_keywords'] = chat_keyword_count * 0.4

        # Question patterns
        question_patterns = [
            r'^(how are you|how\'s it going|what\'s up)',
            r'\b(tell me a joke|make me laugh)',
            r'\b(thank you|thanks|please)',
            r'^(hello|hi|hey|good morning|good afternoon)'
        ]

        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                signals['question_patterns'] = 0.5
                break

        # Short conversational queries (likely chat)
        if len(query.split()) <= 3 and any(word in query_lower for word in ['hi', 'hello', 'thanks', 'help']):
            signals['short_conversational'] = 0.6

        return signals

    def analyze_query_intent(self, query: str) -> Tuple[QueryIntent, float, IntentSignals]:
        """
        Enhanced query analysis that determines intent with confidence scoring.

        Returns:
            Tuple of (primary_intent, confidence, detected_signals)
        """
        # Detect all signal types
        math_signals = self.detect_math_signals(query)
        document_signals = self.detect_document_signals(query)
        tool_signals = self.detect_tool_signals(query)
        conversation_signals = self.detect_conversation_signals(query)

        signals = IntentSignals(
            math_signals=math_signals,
            document_signals=document_signals,
            tool_signals=tool_signals,
            conversation_signals=conversation_signals
        )

        # Calculate confidence scores for each intent with ENHANCED MATH PRIORITY
        intent_scores = {}

        # ENHANCED: Mathematical intent gets highest priority
        math_confidence = sum(math_signals.values())
        if math_confidence > 0:
            # Boost math confidence if explicit calculation detected
            if 'explicit_calculation' in math_signals or 'operator_near_numbers' in math_signals:
                math_confidence = min(math_confidence * 1.5, 1.0)  # Boost but cap at 1.0
                logger.info(f"🧮 BOOSTED math confidence to {math_confidence:.2f} due to explicit calculation")

            intent_scores[QueryIntent.CALCULATION] = math_confidence

        # Document intent (reduced priority when math is present)
        doc_confidence = sum(document_signals.values())
        if doc_confidence > 0:
            # Reduce document confidence if strong math signals present
            if math_confidence > 0.7:
                doc_confidence = doc_confidence * 0.7  # Reduce document priority
                logger.info(f"📄 REDUCED document confidence to {doc_confidence:.2f} due to strong math signals")

            intent_scores[QueryIntent.DOCUMENT_SEARCH] = doc_confidence

        # Tool intent
        tool_confidence = sum(tool_signals.values())
        if tool_confidence > 0:
            intent_scores[QueryIntent.TOOL_REQUEST] = tool_confidence

        # Conversation intent (reduced priority when math is present)
        chat_confidence = sum(conversation_signals.values())
        if chat_confidence > 0:
            # Reduce chat confidence if math signals present
            if math_confidence > 0.5:
                chat_confidence = chat_confidence * 0.6
                logger.info(f"💬 REDUCED chat confidence to {chat_confidence:.2f} due to math signals")

            intent_scores[QueryIntent.CHAT] = chat_confidence

        # ENHANCED: Hybrid intent detection with better thresholds
        if math_confidence > 0.4 and doc_confidence > 0.3:
            # Hybrid gets high priority when both math and document signals are strong
            hybrid_confidence = (math_confidence + doc_confidence) * 0.9  # Increased multiplier
            intent_scores[QueryIntent.HYBRID_DOC_CALC] = hybrid_confidence
            logger.info(f"🔗 HYBRID intent detected with confidence {hybrid_confidence:.2f}")

        # ENHANCED: Pure calculation override - if very strong math signals, force calculation
        if math_confidence >= 0.8:
            # Override other intents if math confidence is very high
            intent_scores = {QueryIntent.CALCULATION: math_confidence}
            logger.info(f"🧮 PURE CALCULATION OVERRIDE: Math confidence {math_confidence:.2f} - forcing calculator route")

        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])
            return primary_intent[0], min(primary_intent[1], 1.0), signals
        else:
            # Fallback to legacy search analysis
            search_type = self.analyze_query_legacy(query)
            legacy_intent = self._convert_search_type_to_intent(search_type)
            return legacy_intent, 0.5, signals

    def analyze_query_legacy(self, query: str) -> SearchType:
        """
        Analyze the query to determine the most appropriate search type.
        
        Args:
            query: The search query to analyze
            
        Returns:
            SearchType enum indicating the best search strategy
        """
        query_lower = query.lower()
        
        # Check for correction queries (highest priority)
        if any(keyword in query_lower for keyword in self.correction_keywords):
            return SearchType.CORRECTION
        
        # Check for document queries
        if any(keyword in query_lower for keyword in self.document_keywords):
            return SearchType.DOCUMENT
        
        # Check for conversation queries
        if any(keyword in query_lower for keyword in self.conversation_keywords):
            return SearchType.CONVERSATION
        
        # Check for knowledge queries
        if any(keyword in query_lower for keyword in self.knowledge_keywords):
            return SearchType.KNOWLEDGE
        
        # Default to general search
        return SearchType.GENERAL
    
    def route_search(self, query: str, max_results: int = 5, **kwargs) -> List[Any]:
        """
        Route the search query to the most appropriate search service.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            **kwargs: Additional parameters for specific search services
            
        Returns:
            List of search results from the appropriate service
        """
        search_type = self.analyze_query(query)
        
        logger.info(f"🎯 Search routing: '{query[:50]}...' -> {search_type.value}")
        
        try:
            if search_type == SearchType.DOCUMENT:
                return self._route_document_search(query, max_results, **kwargs)
            
            elif search_type == SearchType.CORRECTION:
                return self._route_correction_search(query, max_results, **kwargs)
            
            elif search_type == SearchType.CONVERSATION:
                return self._route_conversation_search(query, max_results, **kwargs)
            
            elif search_type == SearchType.KNOWLEDGE:
                return self._route_knowledge_search(query, max_results, **kwargs)
            
            else:  # GENERAL
                return self._route_general_search(query, max_results, **kwargs)
                
        except Exception as e:
            logger.error(f"❌ Search routing failed for {search_type.value}: {e}")
            # Fallback to general search
            return self._route_general_search(query, max_results, **kwargs)
    
    def _route_document_search(self, query: str, max_results: int, **kwargs) -> List[Any]:
        """Route to DocumentSearchService for document-related queries."""
        try:
            from services.document_search_service import get_document_search_service
            search_service = get_document_search_service()
            
            # Get standardized search results
            search_results = search_service.search_documents(query, max_results)
            
            # Convert to compatible format using ResultProcessor
            try:
                from services.result_processor_service import get_result_processor_service
                processor = get_result_processor_service()

                # Convert SearchResult objects to StandardizedResult format
                standardized_results = []
                for result in search_results:
                    from services.result_processor_service import StandardizedResult, ResultType
                    std_result = StandardizedResult(
                        content=result.content,
                        source=result.source,
                        similarity_score=result.similarity_score,
                        result_type=ResultType.DIRECT_CONTENT,
                        chunk_id=result.chunk_id,
                        source_type=result.source_type
                    )
                    standardized_results.append(std_result)

                # Create backward-compatible results
                compatible_results = processor.create_backward_compatible_results(standardized_results)

                logger.info(f"📄 DocumentSearchService returned {len(compatible_results)} results")
                return compatible_results

            except Exception as e:
                logger.warning(f"⚠️ ResultProcessor failed, using fallback: {e}")
                # Fallback to legacy compatible format
                compatible_results = []
                for result in search_results:
                    class CompatibleResult:
                        def __init__(self, search_result):
                            self.content = search_result.content
                            self.source = search_result.source
                            self.similarity_score = search_result.similarity_score
                            self.source_type = search_result.source_type
                            self.chunk_id = search_result.chunk_id
                            self.chunk = self  # For backward compatibility

                    compatible_results.append(CompatibleResult(result))

                logger.info(f"📄 DocumentSearchService returned {len(compatible_results)} results (fallback)")
                return compatible_results
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return self._route_general_search(query, max_results, **kwargs)
    
    def _route_correction_search(self, query: str, max_results: int, **kwargs) -> List[Any]:
        """Route to correction-specific search for user corrections."""
        try:
            # Import the legacy search for now - can be refactored later
            from secure_streamlit_app import search_unified_memory
            
            # Enhance query for correction search
            correction_query = f"{query} user_correction authoritative"
            results = search_unified_memory(correction_query, max_results)
            
            # Boost correction results
            for result in results:
                if hasattr(result, 'source_type'):
                    result.source_type = 'user_corrections'
                if hasattr(result, 'similarity_score'):
                    result.similarity_score = min(1.0, result.similarity_score * 1.5)
            
            logger.info(f"🔧 Correction search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Correction search failed: {e}")
            return self._route_general_search(query, max_results, **kwargs)
    
    def _route_conversation_search(self, query: str, max_results: int, **kwargs) -> List[Any]:
        """Route to conversation-specific search."""
        try:
            # For now, use the legacy search - can be enhanced later
            from secure_streamlit_app import search_unified_memory
            
            # Enhance query for conversation search
            conversation_query = f"{query} conversation chat discussion"
            results = search_unified_memory(conversation_query, max_results)
            
            logger.info(f"💬 Conversation search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Conversation search failed: {e}")
            return self._route_general_search(query, max_results, **kwargs)
    
    def _route_knowledge_search(self, query: str, max_results: int, **kwargs) -> List[Any]:
        """Route to knowledge-specific search."""
        try:
            # For now, use the legacy search - can be enhanced later
            from secure_streamlit_app import search_unified_memory
            
            # Enhance query for knowledge search
            knowledge_query = f"{query} explanation definition concept"
            results = search_unified_memory(knowledge_query, max_results)
            
            logger.info(f"🧠 Knowledge search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Knowledge search failed: {e}")
            return self._route_general_search(query, max_results, **kwargs)
    
    def _route_general_search(self, query: str, max_results: int, **kwargs) -> List[Any]:
        """Route to general search as fallback."""
        try:
            # Use the legacy search for general queries
            from secure_streamlit_app import search_unified_memory
            results = search_unified_memory(query, max_results)
            
            logger.info(f"🔍 General search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ General search failed: {e}")
            return []

    def _convert_search_type_to_intent(self, search_type: SearchType) -> QueryIntent:
        """Convert legacy SearchType to new QueryIntent."""
        mapping = {
            SearchType.DOCUMENT: QueryIntent.DOCUMENT_SEARCH,
            SearchType.CONVERSATION: QueryIntent.CONVERSATION_SEARCH,
            SearchType.KNOWLEDGE: QueryIntent.KNOWLEDGE_SEARCH,
            SearchType.CORRECTION: QueryIntent.CORRECTION_SEARCH,
            SearchType.GENERAL: QueryIntent.GENERAL_SEARCH
        }
        return mapping.get(search_type, QueryIntent.GENERAL_SEARCH)

    def route_smart_query(self, query: str, context: Dict[str, Any] = None) -> SmartRoute:
        """
        Enhanced smart routing with ultra-aggressive math detection and STEF integration.

        Priority Order:
        1. PURE MATH: Instant detection and routing for mathematical expressions
        2. STEF: Structured task execution for complex workflows
        3. EXISTING LOGIC: Standard intent classification and routing

        Args:
            query: The user query to route
            context: Optional context including conversation history, user preferences, etc.

        Returns:
            SmartRoute object with routing decision and metadata
        """
        context = context or {}

        # PRIORITY 1: PURE MATH CHECK - Instant routing for mathematical expressions
        if self.is_pure_math_expression(query):
            logger.info(f"🧮 PURE MATH ROUTE: '{query}' -> Instant calculator")
            return self._route_pure_math(query, context)

        # PRIORITY 2: STEF CHECK - Structured tasks for complex workflows
        if self._stef_enabled and self._program_registry:
            stef_program = self._check_stef_match(query, context)
            if stef_program:
                return self._route_stef_program(query, stef_program, context)

        # PRIORITY 3: EXISTING SMART ROUTING - Standard intent classification
        # Analyze query intent
        primary_intent, confidence, signals = self.analyze_query_intent(query)

        logger.info(f"🧠 Smart routing: '{query[:50]}...' -> {primary_intent.value} (confidence: {confidence:.2f})")

        # Route based on intent
        if primary_intent == QueryIntent.CALCULATION:
            return self._route_calculation(query, confidence, signals, context)

        elif primary_intent == QueryIntent.HYBRID_DOC_CALC:
            return self._route_hybrid_doc_calc(query, confidence, signals, context)

        elif primary_intent == QueryIntent.TOOL_REQUEST:
            return self._route_tool_request(query, confidence, signals, context)

        elif primary_intent == QueryIntent.CHAT:
            return self._route_conversation(query, confidence, signals, context)

        elif primary_intent in [QueryIntent.DOCUMENT_SEARCH, QueryIntent.CONVERSATION_SEARCH,
                               QueryIntent.KNOWLEDGE_SEARCH, QueryIntent.CORRECTION_SEARCH,
                               QueryIntent.GENERAL_SEARCH]:
            return self._route_search_query(query, confidence, signals, context, primary_intent)

        else:
            # Fallback to general search
            return self._route_fallback(query, confidence, signals, context)

    def _route_calculation(self, query: str, confidence: float, signals: IntentSignals,
                          context: Dict[str, Any]) -> SmartRoute:
        """Route pure calculation queries to calculator tool."""
        # Extract mathematical expression
        extracted_math = self._extract_mathematical_expression(query)

        return SmartRoute(
            route_type=RouteType.CALCULATE,
            primary_intent=QueryIntent.CALCULATION,
            confidence=confidence,
            primary_handler='calculator',
            fallback_handlers=['general_chat'],
            extracted_data={'expression': extracted_math, 'original_query': query},
            context_data=context
        )

    def _route_hybrid_doc_calc(self, query: str, confidence: float, signals: IntentSignals,
                              context: Dict[str, Any]) -> SmartRoute:
        """Route hybrid document + calculation queries."""
        # Extract both document reference and mathematical operation
        doc_reference = self._extract_document_reference(query)
        math_operation = self._extract_mathematical_expression(query)

        return SmartRoute(
            route_type=RouteType.HYBRID,
            primary_intent=QueryIntent.HYBRID_DOC_CALC,
            confidence=confidence,
            primary_handler='document_search',
            fallback_handlers=['calculator', 'general_chat'],
            extracted_data={
                'document_reference': doc_reference,
                'math_operation': math_operation,
                'original_query': query
            },
            context_data=context
        )

    def _route_tool_request(self, query: str, confidence: float, signals: IntentSignals,
                           context: Dict[str, Any]) -> SmartRoute:
        """Route tool requests to appropriate generation tools."""
        # Determine which tool is needed
        requested_tool = self._identify_requested_tool(query, signals)

        return SmartRoute(
            route_type=RouteType.GENERATE,
            primary_intent=QueryIntent.TOOL_REQUEST,
            confidence=confidence,
            primary_handler=requested_tool,
            fallback_handlers=['general_chat'],
            extracted_data={'tool_type': requested_tool, 'original_query': query},
            context_data=context
        )

    def _route_conversation(self, query: str, confidence: float, signals: IntentSignals,
                           context: Dict[str, Any]) -> SmartRoute:
        """Route conversational queries to chat handler."""
        return SmartRoute(
            route_type=RouteType.CHAT,
            primary_intent=QueryIntent.CHAT,
            confidence=confidence,
            primary_handler='general_chat',
            fallback_handlers=[],
            extracted_data={'original_query': query},
            context_data=context
        )

    def _route_search_query(self, query: str, confidence: float, signals: IntentSignals,
                           context: Dict[str, Any], intent: QueryIntent) -> SmartRoute:
        """Route search queries using existing search logic."""
        # Convert intent back to legacy search type for compatibility
        search_type_mapping = {
            QueryIntent.DOCUMENT_SEARCH: SearchType.DOCUMENT,
            QueryIntent.CONVERSATION_SEARCH: SearchType.CONVERSATION,
            QueryIntent.KNOWLEDGE_SEARCH: SearchType.KNOWLEDGE,
            QueryIntent.CORRECTION_SEARCH: SearchType.CORRECTION,
            QueryIntent.GENERAL_SEARCH: SearchType.GENERAL
        }

        search_type = search_type_mapping.get(intent, SearchType.GENERAL)

        return SmartRoute(
            route_type=RouteType.SEARCH,
            primary_intent=intent,
            confidence=confidence,
            primary_handler='search_service',
            fallback_handlers=['general_chat'],
            extracted_data={'search_type': search_type, 'original_query': query},
            context_data=context,
            search_type=search_type  # Legacy compatibility
        )

    def _route_fallback(self, query: str, confidence: float, signals: IntentSignals,
                       context: Dict[str, Any]) -> SmartRoute:
        """Fallback routing for unclear queries."""
        return SmartRoute(
            route_type=RouteType.CLARIFY,
            primary_intent=QueryIntent.GENERAL_SEARCH,
            confidence=confidence,
            primary_handler='general_chat',
            fallback_handlers=[],
            extracted_data={'original_query': query, 'needs_clarification': True},
            context_data=context
        )

    def _route_pure_math(self, query: str, context: Dict[str, Any]) -> SmartRoute:
        """
        Route pure mathematical expressions directly to STEF SimpleCalculation with maximum priority.
        This bypasses all other analysis for instant mathematical processing.

        Args:
            query: The pure mathematical expression
            context: Query context

        Returns:
            SmartRoute configured for instant mathematical calculation
        """
        # Get SimpleCalculation program from registry
        simple_calc_program = None
        if self._program_registry:
            simple_calc_program = self._program_registry.get_program('SimpleCalculation')

        return SmartRoute(
            route_type=RouteType.STEF,
            primary_intent=QueryIntent.CALCULATION,
            confidence=1.0,  # Maximum confidence for pure math
            primary_handler='stef_executor',
            fallback_handlers=[],  # No fallbacks needed for pure math
            extracted_data={
                'stef_program': simple_calc_program,
                'program_name': 'SimpleCalculation',
                'original_query': query,
                'pure_math': True,  # Flag for instant processing
                'math_expression': query.strip()
            },
            context_data=context
        )

    def _check_stef_match(self, query: str, context: Dict[str, Any]) -> Optional[Any]:
        """
        Check if any STEF program can handle this query.

        Args:
            query: The user query
            context: Query context

        Returns:
            Matching TaskProgram or None
        """
        try:
            if not self._program_registry:
                return None

            # Phase 1: Simple keyword matching
            matching_program = self._program_registry.find_matching_program(query)

            if matching_program:
                logger.info(f"🎯 STEF match found: {matching_program.program_name}")
                return matching_program

            return None

        except Exception as e:
            logger.error(f"STEF matching failed: {e}")
            return None

    def _route_stef_program(self, query: str, program: Any, context: Dict[str, Any]) -> SmartRoute:
        """
        Route query to STEF program execution.

        Args:
            query: The user query
            program: The matched TaskProgram
            context: Query context

        Returns:
            SmartRoute configured for STEF execution
        """
        return SmartRoute(
            route_type=RouteType.STEF,
            primary_intent=QueryIntent.TOOL_REQUEST,  # STEF programs are tool-based
            confidence=0.9,  # High confidence for matched STEF programs
            primary_handler='stef_executor',
            fallback_handlers=['general_chat'],
            extracted_data={
                'stef_program': program,
                'program_name': program.program_name,
                'original_query': query
            },
            context_data=context
        )

    def _extract_mathematical_expression(self, query: str) -> str:
        """
        Enhanced mathematical expression extraction with comprehensive pattern matching.
        Handles complex mathematical expressions and natural language math.
        """
        # ENHANCED: Priority patterns for explicit calculations
        priority_patterns = [
            r'\d+(?:\.\d+)?\s*[\+\-\*\/\%]\s*\d+(?:\.\d+)?',  # Basic arithmetic with decimals
            r'\d+(?:\.\d+)?%\s+of\s+\d+(?:\.\d+)?',           # Percentage calculations
            r'\d+(?:\.\d+)?\s+percent\s+of\s+\d+(?:\.\d+)?',  # Percentage (word form)
            r'\d+(?:\.\d+)?\s*×\s*\d+(?:\.\d+)?',             # Unicode multiplication
            r'\d+(?:\.\d+)?\s*÷\s*\d+(?:\.\d+)?',             # Unicode division
        ]

        # Try priority patterns first
        for pattern in priority_patterns:
            match = re.search(pattern, query)
            if match:
                extracted = match.group()
                logger.info(f"🧮 Extracted priority math expression: '{extracted}'")
                return extracted

        # ENHANCED: Natural language math patterns
        natural_patterns = [
            (r'what\'s\s+(\d+(?:\.\d+)?)\s*[\+\-\*\/\%]\s*(\d+(?:\.\d+)?)', r'\1 \2'),
            (r'what\s+is\s+(\d+(?:\.\d+)?)\s*[\+\-\*\/\%]\s*(\d+(?:\.\d+)?)', r'\1 \2'),
            (r'calculate\s+(\d+(?:\.\d+)?)\s*[\+\-\*\/\%]\s*(\d+(?:\.\d+)?)', r'\1 \2'),
            (r'compute\s+(\d+(?:\.\d+)?)\s*[\+\-\*\/\%]\s*(\d+(?:\.\d+)?)', r'\1 \2'),
            (r'(\d+(?:\.\d+)?)\s+plus\s+(\d+(?:\.\d+)?)', r'\1 + \2'),
            (r'(\d+(?:\.\d+)?)\s+minus\s+(\d+(?:\.\d+)?)', r'\1 - \2'),
            (r'(\d+(?:\.\d+)?)\s+times\s+(\d+(?:\.\d+)?)', r'\1 * \2'),
            (r'(\d+(?:\.\d+)?)\s+divided\s+by\s+(\d+(?:\.\d+)?)', r'\1 / \2'),
        ]

        for pattern, replacement in natural_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                # Clean up the extracted expression
                extracted = re.search(r'[\d\+\-\*\/\%\.\s]+', extracted)
                if extracted:
                    result = extracted.group().strip()
                    logger.info(f"🧮 Extracted natural language math: '{result}'")
                    return result

        # ENHANCED: Percentage extraction with better patterns
        percentage_patterns = [
            r'(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s+percent\s+of\s+(\d+(?:\.\d+)?)',
            r'what\'s\s+(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)',
            r'calculate\s+(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)',
        ]

        for pattern in percentage_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                percent, number = match.groups()
                result = f"{percent}% of {number}"
                logger.info(f"🧮 Extracted percentage calculation: '{result}'")
                return result

        # ENHANCED: Complex expression extraction
        complex_patterns = [
            r'what\'s\s+([\d\+\-\*\/\%\.\s\(\)]+)',
            r'what\s+is\s+([\d\+\-\*\/\%\.\s\(\)]+)',
            r'calculate\s+([\d\+\-\*\/\%\.\s\(\)]+)',
            r'compute\s+([\d\+\-\*\/\%\.\s\(\)]+)',
            r'solve\s+([\d\+\-\*\/\%\.\s\(\)]+)',
        ]

        for pattern in complex_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Validate it contains mathematical content
                if re.search(r'\d.*[\+\-\*\/\%].*\d', extracted):
                    logger.info(f"🧮 Extracted complex math expression: '{extracted}'")
                    return extracted

        # ENHANCED: Fallback - extract any mathematical-looking content
        math_content = re.findall(r'\d+(?:\.\d+)?(?:\s*[\+\-\*\/\%]\s*\d+(?:\.\d+)?)+', query)
        if math_content:
            result = math_content[0]
            logger.info(f"🧮 Extracted fallback math content: '{result}'")
            return result

        # Final fallback - return original query
        logger.info(f"🧮 No specific math pattern found, returning original query")
        return query

    def _extract_document_reference(self, query: str) -> str:
        """Extract document reference from query."""
        doc_patterns = [
            r'\b(sam story|my document|the file|the document)\b',
            r'\b[\w\s]+\.(?:pdf|docx|txt|doc)\b',
            r'\bin\s+(?:my\s+)?document\b',
            r'\bfrom\s+(?:the\s+)?(?:file|document)\b'
        ]

        for pattern in doc_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group()

        return "document"  # Generic fallback

    def _identify_requested_tool(self, query: str, signals: IntentSignals) -> str:
        """Identify which tool is being requested."""
        query_lower = query.lower()

        # Chart/graph creation
        if any(word in query_lower for word in ['chart', 'graph', 'plot']):
            return 'chart_creator'

        # Image generation
        if any(word in query_lower for word in ['image', 'picture', 'draw', 'generate image']):
            return 'image_generator'

        # Web search
        if any(word in query_lower for word in ['search web', 'google', 'find online']):
            return 'web_search'

        # Default to general tool
        return 'general_tool'

    # Legacy compatibility methods
    def analyze_query(self, query: str) -> SearchType:
        """Legacy method - routes to analyze_query_legacy for compatibility."""
        return self.analyze_query_legacy(query)

# Global instance for easy access
_search_router = None

def get_search_router() -> SmartQueryRouter:
    """Get or create the global smart query router instance."""
    global _search_router
    if _search_router is None:
        _search_router = SmartQueryRouter()
    return _search_router

# Legacy compatibility
SearchRouter = SmartQueryRouter  # Alias for backward compatibility

def smart_search(query: str, max_results: int = 5, **kwargs) -> List[Any]:
    """
    Convenience function for intelligent search routing.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        **kwargs: Additional parameters for specific search services
        
    Returns:
        List of search results from the most appropriate service
    """
    router = get_search_router()
    return router.route_search(query, max_results, **kwargs)
