#!/usr/bin/env python3
"""
Smart Query Handler Service
Integrates the Enhanced Smart Query Router with all available tools and services.
Handles the complete query processing pipeline from intent detection to response generation.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QueryResponse:
    """Response from smart query processing."""
    content: str
    confidence: float
    route_type: str
    primary_handler: str
    execution_time_ms: float
    success: bool
    metadata: Dict[str, Any]
    fallback_used: bool = False

class SmartQueryHandler:
    """
    Central handler that integrates Smart Query Router with all available tools and services.
    Provides a unified interface for processing any type of user query.
    """
    
    def __init__(self):
        self.router = None
        self.calculator_tool = None
        self.document_search_service = None
        self.stef_executor = None
        self.response_presenter = None

        # Initialize services lazily to avoid circular imports
        self._initialize_services()

        # Track performance metrics
        self.query_count = 0
        self.success_count = 0
        self.route_type_counts = {}

        logger.info("SmartQueryHandler initialized")
    
    def _initialize_services(self):
        """Initialize all required services."""
        try:
            from services.search_router import get_search_router
            self.router = get_search_router()

            from services.calculator_tool import get_calculator_tool
            self.calculator_tool = get_calculator_tool()

            from services.document_search_service import get_document_search_service
            self.document_search_service = get_document_search_service()

            # Initialize STEF executor
            try:
                from sam.agents.stef import STEF_Executor
                # Create tool registry for STEF
                tool_registry = {
                    'calculator': self.calculator_tool.calculate,
                    'web_search': self._web_search_wrapper,
                    'document_search': self._document_search_wrapper
                }
                self.stef_executor = STEF_Executor(tool_registry, None)  # LLM interface will be added later
                logger.info("✅ STEF executor initialized successfully")
            except ImportError:
                logger.warning("STEF components not available, structured tasks disabled")
                self.stef_executor = None

            # Initialize ResponsePresenter
            try:
                from services.response_presenter import get_response_presenter
                self.response_presenter = get_response_presenter()
                logger.info("✅ ResponsePresenter initialized successfully")
            except ImportError:
                logger.warning("ResponsePresenter not available, using fallback formatting")
                self.response_presenter = None

            logger.info("✅ All services initialized successfully")

        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> QueryResponse:
        """
        Process a user query using smart routing and appropriate tools.
        
        Args:
            query: The user's query
            context: Optional context including conversation history, user preferences, etc.
            
        Returns:
            QueryResponse with the processed result
        """
        import time
        start_time = time.time()
        
        self.query_count += 1
        context = context or {}
        
        try:
            # Route the query using smart router
            route = self.router.route_smart_query(query, context)
            
            # Track route type usage
            route_type = route.route_type.value
            self.route_type_counts[route_type] = self.route_type_counts.get(route_type, 0) + 1
            
            logger.info(f"🎯 Processing query via {route_type}: '{query[:50]}...'")
            
            # Execute based on route type
            if route.route_type.value == 'stef':
                response = self._handle_stef_query(query, route, context)
            elif route.route_type.value == 'calculate':
                response = self._handle_calculation(query, route, context)
            elif route.route_type.value == 'hybrid':
                response = self._handle_hybrid_query(query, route, context)
            elif route.route_type.value == 'search':
                response = self._handle_search_query(query, route, context)
            elif route.route_type.value == 'chat':
                response = self._handle_chat_query(query, route, context)
            elif route.route_type.value == 'generate':
                response = self._handle_generation_query(query, route, context)
            elif route.route_type.value == 'clarify':
                response = self._handle_clarification_query(query, route, context)
            else:
                response = self._handle_fallback_query(query, route, context)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000
            response.execution_time_ms = execution_time
            
            if response.success:
                self.success_count += 1
            
            logger.info(f"✅ Query processed successfully in {execution_time:.1f}ms")
            return response
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Query processing failed: {e}")
            
            return QueryResponse(
                content=f"I apologize, but I encountered an error processing your request: {str(e)}",
                confidence=0.0,
                route_type="error",
                primary_handler="error_handler",
                execution_time_ms=execution_time,
                success=False,
                metadata={'error': str(e), 'query': query}
            )
    
    def _handle_calculation(self, query: str, route, context: Dict[str, Any]) -> QueryResponse:
        """Handle pure calculation queries."""
        try:
            # Extract the mathematical expression
            expression = route.extracted_data.get('expression', query)
            
            # Perform calculation
            calc_result = self.calculator_tool.calculate(expression, context)
            
            if calc_result.success:
                content = f"**Calculation Result:**\n\n{calc_result.explanation}\n\n**Answer: {calc_result.result}**"
                
                return QueryResponse(
                    content=content,
                    confidence=route.confidence,
                    route_type=route.route_type.value,
                    primary_handler=route.primary_handler,
                    execution_time_ms=0,  # Will be set by caller
                    success=True,
                    metadata={
                        'calculation_result': calc_result.result,
                        'expression': calc_result.expression,
                        'explanation': calc_result.explanation
                    }
                )
            else:
                # Try fallback
                return self._try_fallback(query, route, context, calc_result.error_message)
                
        except Exception as e:
            return self._try_fallback(query, route, context, str(e))
    
    def _handle_hybrid_query(self, query: str, route, context: Dict[str, Any]) -> QueryResponse:
        """Handle hybrid document + calculation queries."""
        try:
            # First, search for document content
            doc_reference = route.extracted_data.get('document_reference', 'document')
            math_operation = route.extracted_data.get('math_operation', '')
            
            # Search documents for relevant information
            search_results = self.document_search_service.search_documents(
                query=f"{doc_reference} {math_operation}",
                max_results=3
            )
            
            if search_results:
                # Extract numerical values from document content
                extracted_values = self._extract_numbers_from_results(search_results)
                
                if extracted_values and math_operation:
                    # Perform calculation with extracted values
                    calc_context = {'extracted_values': extracted_values}
                    calc_result = self.calculator_tool.calculate(math_operation, calc_context)
                    
                    if calc_result.success:
                        content = f"**Based on your document:**\n\n"
                        content += f"Found relevant information: {extracted_values}\n\n"
                        content += f"**Calculation:** {calc_result.explanation}\n\n"
                        content += f"**Answer: {calc_result.result}**"
                        
                        return QueryResponse(
                            content=content,
                            confidence=route.confidence,
                            route_type=route.route_type.value,
                            primary_handler=route.primary_handler,
                            execution_time_ms=0,
                            success=True,
                            metadata={
                                'document_results': len(search_results),
                                'extracted_values': extracted_values,
                                'calculation_result': calc_result.result
                            }
                        )
                
                # If calculation fails, just return document content
                content = f"**From your document:**\n\n"
                for i, result in enumerate(search_results[:2], 1):
                    content += f"{i}. {result.content[:200]}...\n\n"
                content += f"I found this information but couldn't perform the calculation. Could you clarify what you'd like me to calculate?"
                
                return QueryResponse(
                    content=content,
                    confidence=route.confidence * 0.7,
                    route_type=route.route_type.value,
                    primary_handler=route.primary_handler,
                    execution_time_ms=0,
                    success=True,
                    metadata={'document_results': len(search_results), 'calculation_attempted': True}
                )
            
            else:
                # No document results, try pure calculation
                return self._handle_calculation(query, route, context)
                
        except Exception as e:
            return self._try_fallback(query, route, context, str(e))
    
    def _handle_search_query(self, query: str, route, context: Dict[str, Any]) -> QueryResponse:
        """Handle search queries using existing search services."""
        try:
            # Use existing document search service
            search_results = self.document_search_service.search_documents(query, max_results=5)
            
            if search_results:
                content = f"**Found {len(search_results)} relevant results:**\n\n"
                
                for i, result in enumerate(search_results, 1):
                    content += f"**{i}. {result.source}** (Score: {result.similarity_score:.3f})\n"
                    content += f"{result.content[:300]}...\n\n"
                
                return QueryResponse(
                    content=content,
                    confidence=route.confidence,
                    route_type=route.route_type.value,
                    primary_handler=route.primary_handler,
                    execution_time_ms=0,
                    success=True,
                    metadata={'search_results': len(search_results)}
                )
            else:
                return QueryResponse(
                    content="I couldn't find any relevant information for your query. Could you try rephrasing or providing more details?",
                    confidence=route.confidence * 0.5,
                    route_type=route.route_type.value,
                    primary_handler=route.primary_handler,
                    execution_time_ms=0,
                    success=False,
                    metadata={'search_results': 0}
                )
                
        except Exception as e:
            return self._try_fallback(query, route, context, str(e))
    
    def _handle_chat_query(self, query: str, route, context: Dict[str, Any]) -> QueryResponse:
        """Handle conversational queries."""
        # Simple conversational responses
        chat_responses = {
            'hello': "Hello! I'm SAM, your intelligent assistant. I can help you with documents, calculations, and more. What would you like to know?",
            'hi': "Hi there! How can I help you today?",
            'how are you': "I'm doing well, thank you! I'm ready to help you with any questions or tasks you have.",
            'thank you': "You're welcome! Is there anything else I can help you with?",
            'thanks': "You're welcome! Feel free to ask me anything else.",
            'help': "I can help you with:\n• Document analysis and search\n• Mathematical calculations\n• General questions\n• And much more! What would you like to do?"
        }
        
        query_lower = query.lower().strip()
        response_content = chat_responses.get(query_lower, 
            "I'm here to help! You can ask me about documents, calculations, or general questions. What would you like to know?")
        
        return QueryResponse(
            content=response_content,
            confidence=route.confidence,
            route_type=route.route_type.value,
            primary_handler=route.primary_handler,
            execution_time_ms=0,
            success=True,
            metadata={'chat_type': 'conversational'}
        )
    
    def _handle_generation_query(self, query: str, route, context: Dict[str, Any]) -> QueryResponse:
        """Handle tool/generation requests."""
        return QueryResponse(
            content="I understand you'd like me to generate or create something. This feature is coming soon! For now, I can help with document analysis and calculations.",
            confidence=route.confidence * 0.5,
            route_type=route.route_type.value,
            primary_handler=route.primary_handler,
            execution_time_ms=0,
            success=False,
            metadata={'feature_status': 'coming_soon'}
        )
    
    def _handle_clarification_query(self, query: str, route, context: Dict[str, Any]) -> QueryResponse:
        """Handle queries that need clarification."""
        return QueryResponse(
            content="I'm not quite sure what you're looking for. Could you please clarify? I can help with:\n• Document questions\n• Mathematical calculations\n• General information\n\nWhat specifically would you like me to help with?",
            confidence=route.confidence,
            route_type=route.route_type.value,
            primary_handler=route.primary_handler,
            execution_time_ms=0,
            success=True,
            metadata={'needs_clarification': True}
        )
    
    def _handle_fallback_query(self, query: str, route, context: Dict[str, Any]) -> QueryResponse:
        """Handle fallback queries."""
        return QueryResponse(
            content="I'm not sure how to handle that request. Could you try rephrasing it? I'm best at helping with document analysis and mathematical calculations.",
            confidence=route.confidence,
            route_type=route.route_type.value,
            primary_handler=route.primary_handler,
            execution_time_ms=0,
            success=False,
            metadata={'fallback_reason': 'unknown_intent'}
        )

    def _handle_stef_query(self, query: str, route, context: Dict[str, Any]) -> QueryResponse:
        """Handle STEF structured task execution with optimized pure math processing."""
        try:
            if not self.stef_executor:
                return QueryResponse(
                    content="Structured task execution is not available. Please try a simpler approach.",
                    confidence=0.0,
                    route_type=route.route_type.value,
                    primary_handler=route.primary_handler,
                    execution_time_ms=0,
                    success=False,
                    metadata={'error': 'stef_not_available'}
                )

            # Extract STEF program from route
            stef_program = route.extracted_data.get('stef_program')
            if not stef_program:
                return self._try_fallback(query, route, context, "No STEF program found in route")

            # Check if this is pure math for optimized processing
            is_pure_math = route.extracted_data.get('pure_math', False)

            if is_pure_math:
                logger.info(f"🧮 PURE MATH: Instant processing for '{query}'")
                # For pure math, try direct calculator and generate RichAnswer
                try:
                    calc_result = self.calculator_tool.calculate(query)
                    if calc_result.success:
                        # Generate RichAnswer JSON for pure math
                        rich_answer_json = f'''{{
                            "direct_answer": "{calc_result.result}",
                            "query_type": "pure_math",
                            "presentation_type": "minimal",
                            "confidence": 1.0,
                            "supporting_evidence": ["calculator"],
                            "reasoning_summary": "Direct mathematical calculation",
                            "detailed_reasoning": "Calculated: {query} = {calc_result.result}"
                        }}'''

                        logger.info("🎯 Generated RichAnswer for pure math")
                        return self._process_rich_answer(
                            rich_answer_json,
                            route,
                            0.5,  # Instant execution time
                            context.get('user_preferences', {})
                        )
                except Exception as e:
                    logger.warning(f"Direct calculator failed for pure math, falling back to STEF: {e}")

            # Execute STEF program (either non-pure-math or fallback from direct calculator)
            logger.info(f"🚀 Executing STEF program: {stef_program.program_name}")
            execution_result = self.stef_executor.execute(stef_program, query, context)

            if execution_result.success:
                # Check if the response is a RichAnswer JSON
                response_content = execution_result.final_response

                # Try to process as RichAnswer first
                if (response_content.strip().startswith('{') and
                    '"direct_answer"' in response_content):
                    logger.info("🎯 Processing STEF response as RichAnswer")
                    return self._process_rich_answer(
                        response_content,
                        route,
                        execution_result.execution_time_ms,
                        context.get('user_preferences', {})
                    )
                else:
                    # Fallback to traditional response
                    logger.info("📝 Processing STEF response as traditional text")
                    return QueryResponse(
                        content=response_content,
                        confidence=route.confidence,
                        route_type=route.route_type.value,
                        primary_handler=route.primary_handler,
                        execution_time_ms=execution_result.execution_time_ms,
                        success=True,
                        metadata={
                            'stef_program': stef_program.program_name,
                            'steps_completed': execution_result.steps_completed,
                            'steps_failed': execution_result.steps_failed,
                            'completion_rate': execution_result.completion_rate,
                            'pure_math': is_pure_math,
                            'response_type': 'traditional'
                        }
                    )
            else:
                # STEF execution failed, try fallback
                return self._try_fallback(query, route, context, execution_result.error_message or "STEF execution failed")

        except Exception as e:
            return self._try_fallback(query, route, context, f"STEF execution error: {str(e)}")

    def _web_search_wrapper(self, query: str) -> str:
        """Wrapper for web search tool to match STEF interface."""
        try:
            # This would integrate with actual web search service
            # For now, return a placeholder
            return f"Web search results for: {query} (placeholder - integrate with actual web search)"
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Web search failed: {str(e)}"

    def _document_search_wrapper(self, query: str) -> str:
        """Wrapper for document search tool to match STEF interface."""
        try:
            if self.document_search_service:
                results = self.document_search_service.search_documents(query, max_results=3)
                if results:
                    content = "\n".join([f"- {result.content[:200]}..." for result in results])
                    return f"Document search results:\n{content}"
                else:
                    return "No relevant documents found."
            else:
                return "Document search service not available."
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return f"Document search failed: {str(e)}"

    def _process_rich_answer(self, rich_answer_json: str, route, execution_time_ms: float,
                           user_preferences: Dict[str, Any] = None) -> QueryResponse:
        """
        Process a RichAnswer JSON response and format it for presentation.

        Args:
            rich_answer_json: JSON string containing RichAnswer data
            route: The routing decision that led to this response
            execution_time_ms: Time taken to execute the query
            user_preferences: Optional user preferences for formatting

        Returns:
            QueryResponse with formatted content
        """
        try:
            # Parse the RichAnswer JSON
            from services.response_types import parse_llm_response_to_rich_answer
            rich_answer = parse_llm_response_to_rich_answer(rich_answer_json)

            # Update execution metadata
            rich_answer.set_execution_metadata(
                route_type=route.route_type.value,
                execution_time_ms=execution_time_ms,
                tools_used=rich_answer.supporting_evidence
            )

            # Format the response using ResponsePresenter
            if self.response_presenter:
                formatted_content = self.response_presenter.present(rich_answer, user_preferences)
            else:
                # Fallback formatting
                formatted_content = rich_answer.direct_answer

            return QueryResponse(
                content=formatted_content,
                confidence=rich_answer.confidence,
                route_type=route.route_type.value,
                primary_handler=route.primary_handler,
                execution_time_ms=execution_time_ms,
                success=rich_answer.is_successful(),
                metadata={
                    'rich_answer': True,
                    'query_type': rich_answer.query_type.value,
                    'presentation_type': rich_answer.presentation_type.value,
                    'sources_count': len(rich_answer.sources),
                    'has_detailed_reasoning': bool(rich_answer.detailed_reasoning)
                }
            )

        except Exception as e:
            logger.error(f"RichAnswer processing failed: {e}")
            # Fallback to treating the JSON as plain text
            return QueryResponse(
                content=rich_answer_json,
                confidence=0.5,
                route_type=route.route_type.value,
                primary_handler=route.primary_handler,
                execution_time_ms=execution_time_ms,
                success=True,
                metadata={'rich_answer_error': str(e)}
            )

    def _try_fallback(self, query: str, route, context: Dict[str, Any], error_msg: str) -> QueryResponse:
        """Try fallback handlers when primary handler fails."""
        for fallback_handler in route.fallback_handlers:
            try:
                if fallback_handler == 'general_chat':
                    return QueryResponse(
                        content=f"I had trouble processing that request ({error_msg}). Could you try rephrasing it?",
                        confidence=route.confidence * 0.3,
                        route_type=route.route_type.value,
                        primary_handler=fallback_handler,
                        execution_time_ms=0,
                        success=False,
                        metadata={'fallback_used': True, 'error': error_msg},
                        fallback_used=True
                    )
            except Exception as e:
                logger.warning(f"Fallback handler {fallback_handler} also failed: {e}")
        
        # Final fallback
        return QueryResponse(
            content="I apologize, but I'm having trouble processing your request. Please try rephrasing it or ask me something else.",
            confidence=0.1,
            route_type="error",
            primary_handler="final_fallback",
            execution_time_ms=0,
            success=False,
            metadata={'all_fallbacks_failed': True, 'original_error': error_msg}
        )
    
    def _extract_numbers_from_results(self, search_results) -> Dict[str, float]:
        """Extract numerical values from search results."""
        import re
        
        extracted = {}
        for result in search_results:
            # Look for numbers with context
            content = result.content.lower()
            
            # Revenue patterns
            revenue_match = re.search(r'revenue[:\s]+\$?(\d+(?:,\d{3})*(?:\.\d+)?)', content)
            if revenue_match:
                extracted['revenue'] = float(revenue_match.group(1).replace(',', ''))
            
            # Profit patterns
            profit_match = re.search(r'profit[:\s]+\$?(\d+(?:,\d{3})*(?:\.\d+)?)', content)
            if profit_match:
                extracted['profit'] = float(profit_match.group(1).replace(',', ''))
            
            # General number extraction
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', content)
            if numbers:
                extracted['numbers'] = [float(n.replace(',', '')) for n in numbers[:3]]  # First 3 numbers
        
        return extracted
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the smart query handler."""
        success_rate = (self.success_count / self.query_count * 100) if self.query_count > 0 else 0
        
        return {
            'total_queries': self.query_count,
            'successful_queries': self.success_count,
            'success_rate': success_rate,
            'route_type_distribution': self.route_type_counts,
            'most_used_route': max(self.route_type_counts.items(), key=lambda x: x[1])[0] if self.route_type_counts else None
        }

# Global instance for easy access
_smart_query_handler = None

def get_smart_query_handler() -> SmartQueryHandler:
    """Get or create the global smart query handler instance."""
    global _smart_query_handler
    if _smart_query_handler is None:
        _smart_query_handler = SmartQueryHandler()
    return _smart_query_handler
