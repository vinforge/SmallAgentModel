"""
Integrated Synthesis System for SAM
Combines all Sprint 7 synthesis capabilities into a unified system.

Sprint 7: Knowledge Fusion & Multi-Turn Reasoning Integration
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .synthesis_engine import CrossDocumentSynthesizer, get_synthesizer
from .task_manager import TaskManager, get_task_manager, TaskStatus
from .answer_structuring import AdvancedAnswerStructurer, get_answer_structurer
from .capsule_chains import CapsuleChainManager, get_chain_manager
from .guided_learning import UserGuidedLearningManager, get_guided_learning_manager, FeedbackType

logger = logging.getLogger(__name__)

@dataclass
class SynthesisRequest:
    """Request for synthesis processing."""
    request_id: str
    user_id: str
    session_id: str
    query: str
    synthesis_type: str  # 'multi_source', 'multi_turn', 'structured', 'chained'
    parameters: Dict[str, Any]
    context: Dict[str, Any]

@dataclass
class SynthesisResponse:
    """Response from synthesis processing."""
    request_id: str
    response_content: str
    synthesis_metadata: Dict[str, Any]
    confidence_score: float
    sources_used: List[str]
    tools_used: List[str]
    processing_time_ms: int
    created_at: str

class IntegratedSynthesisSystem:
    """
    Unified synthesis system that integrates all Sprint 7 capabilities.
    """
    
    def __init__(self, memory_manager=None, capsule_manager=None, 
                 multimodal_pipeline=None, profile_manager=None):
        """
        Initialize the integrated synthesis system.
        
        Args:
            memory_manager: Long-term memory manager
            capsule_manager: Knowledge capsule manager
            multimodal_pipeline: Multimodal content pipeline
            profile_manager: User profile manager
        """
        # Initialize all synthesis components
        self.synthesizer = get_synthesizer(
            memory_manager=memory_manager,
            capsule_manager=capsule_manager,
            multimodal_pipeline=multimodal_pipeline
        )
        
        self.task_manager = get_task_manager()
        self.answer_structurer = get_answer_structurer()
        self.chain_manager = get_chain_manager(capsule_manager=capsule_manager)
        self.guided_learning = get_guided_learning_manager()
        
        # Store references to other managers
        self.memory_manager = memory_manager
        self.capsule_manager = capsule_manager
        self.multimodal_pipeline = multimodal_pipeline
        self.profile_manager = profile_manager
        
        logger.info("Integrated synthesis system initialized")
    
    def process_synthesis_request(self, request: SynthesisRequest) -> SynthesisResponse:
        """
        Process a synthesis request using appropriate synthesis method.
        
        Args:
            request: SynthesisRequest to process
            
        Returns:
            SynthesisResponse with synthesized content
        """
        try:
            start_time = datetime.now()
            
            logger.info(f"Processing synthesis request: {request.synthesis_type} for {request.query[:50]}...")
            
            # Route to appropriate synthesis method
            if request.synthesis_type == 'multi_source':
                response_content, metadata = self._process_multi_source_synthesis(request)
            elif request.synthesis_type == 'multi_turn':
                response_content, metadata = self._process_multi_turn_synthesis(request)
            elif request.synthesis_type == 'structured':
                response_content, metadata = self._process_structured_synthesis(request)
            elif request.synthesis_type == 'chained':
                response_content, metadata = self._process_chained_synthesis(request)
            else:
                # Default to multi-source synthesis
                response_content, metadata = self._process_multi_source_synthesis(request)
            
            # Apply user-guided learning if available
            learning_guidance = self._apply_learning_guidance(request)
            if learning_guidance:
                response_content = self._incorporate_learning_guidance(response_content, learning_guidance)
                metadata['learning_applied'] = True
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Create response
            synthesis_response = SynthesisResponse(
                request_id=request.request_id,
                response_content=response_content,
                synthesis_metadata=metadata,
                confidence_score=metadata.get('confidence', 0.5),
                sources_used=metadata.get('sources_used', []),
                tools_used=metadata.get('tools_used', []),
                processing_time_ms=processing_time_ms,
                created_at=datetime.now().isoformat()
            )
            
            logger.info(f"Synthesis completed: {processing_time_ms}ms, "
                       f"confidence: {synthesis_response.confidence_score:.2f}")
            
            return synthesis_response
            
        except Exception as e:
            logger.error(f"Error processing synthesis request: {e}")
            return self._create_error_response(request, str(e))
    
    def create_multi_turn_task(self, user_id: str, session_id: str, 
                              task_description: str, subtask_descriptions: List[str],
                              context: Dict[str, Any] = None) -> str:
        """
        Create a multi-turn task for complex reasoning.
        
        Args:
            user_id: User ID
            session_id: Session ID
            task_description: Overall task description
            subtask_descriptions: List of subtask descriptions
            context: Task context
            
        Returns:
            Task ID
        """
        try:
            task_id = self.task_manager.create_task(
                title=f"Multi-turn task: {task_description[:50]}...",
                description=task_description,
                user_id=user_id,
                session_id=session_id,
                subtask_descriptions=subtask_descriptions,
                context=context
            )
            
            # Start the task
            self.task_manager.start_task(task_id)
            
            logger.info(f"Created multi-turn task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating multi-turn task: {e}")
            raise
    
    def create_capsule_chain(self, user_id: str, chain_name: str, 
                           chain_description: str, capsule_sequence: List[Dict[str, Any]],
                           context: Dict[str, Any] = None) -> str:
        """
        Create a capsule chain for strategy replay.
        
        Args:
            user_id: User ID
            chain_name: Chain name
            chain_description: Chain description
            capsule_sequence: Sequence of capsules to chain
            context: Chain context
            
        Returns:
            Chain ID
        """
        try:
            chain_id = self.chain_manager.create_chain(
                name=chain_name,
                description=chain_description,
                capsule_sequence=capsule_sequence,
                user_id=user_id,
                context=context
            )
            
            logger.info(f"Created capsule chain: {chain_id}")
            return chain_id
            
        except Exception as e:
            logger.error(f"Error creating capsule chain: {e}")
            raise
    
    def collect_user_feedback(self, user_id: str, session_id: str,
                             query: str, response: str, feedback_text: str,
                             feedback_type: FeedbackType = FeedbackType.IMPROVEMENT) -> str:
        """
        Collect user feedback for guided learning.
        
        Args:
            user_id: User ID
            session_id: Session ID
            query: Original query
            response: Original response
            feedback_text: User feedback
            feedback_type: Type of feedback
            
        Returns:
            Feedback ID
        """
        try:
            feedback_id = self.guided_learning.collect_feedback(
                user_id=user_id,
                session_id=session_id,
                original_query=query,
                original_response=response,
                feedback_text=feedback_text,
                feedback_type=feedback_type
            )
            
            logger.info(f"Collected user feedback: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error collecting user feedback: {e}")
            raise
    
    def _process_multi_source_synthesis(self, request: SynthesisRequest) -> Tuple[str, Dict[str, Any]]:
        """Process multi-source synthesis request."""
        try:
            # Auto-gather sources or use provided sources
            if 'sources' in request.parameters:
                # Use provided sources
                document_sources = request.parameters.get('document_sources', [])
                memory_sources = request.parameters.get('memory_sources', [])
                capsule_sources = request.parameters.get('capsule_sources', [])
                tool_outputs = request.parameters.get('tool_outputs', [])
                
                synthesis_result = self.synthesizer.synthesize_multi_source_answer(
                    query=request.query,
                    document_sources=document_sources,
                    memory_sources=memory_sources,
                    capsule_sources=capsule_sources,
                    tool_outputs=tool_outputs
                )
            else:
                # Auto-gather sources
                synthesis_result = self.synthesizer.auto_synthesize_from_query(
                    query=request.query,
                    max_sources=request.parameters.get('max_sources', 5)
                )
            
            # Structure the response
            structured_response = self.answer_structurer.structure_response(
                query=request.query,
                raw_response=synthesis_result.synthesized_content,
                sources=[{'title': s.title, 'content': s.content} for s in synthesis_result.sources_used]
            )
            
            # Format as markdown
            formatted_response = self.answer_structurer.format_structured_response(
                structured_response,
                include_confidence=True,
                include_metadata=True
            )
            
            metadata = {
                'synthesis_type': 'multi_source',
                'sources_count': len(synthesis_result.sources_used),
                'conflicts_detected': len(synthesis_result.conflicts_detected),
                'confidence': synthesis_result.synthesis_confidence,
                'sources_used': [s.title for s in synthesis_result.sources_used],
                'tools_used': [],
                'structured_sections': len(structured_response.sections)
            }
            
            return formatted_response, metadata
            
        except Exception as e:
            logger.error(f"Error in multi-source synthesis: {e}")
            return f"Error in multi-source synthesis: {str(e)}", {'error': str(e)}
    
    def _process_multi_turn_synthesis(self, request: SynthesisRequest) -> Tuple[str, Dict[str, Any]]:
        """Process multi-turn synthesis request."""
        try:
            # Get or create active task
            task_id = request.parameters.get('task_id')
            
            if not task_id:
                # Create new task if none specified
                subtasks = request.parameters.get('subtasks', [request.query])
                task_id = self.create_multi_turn_task(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    task_description=request.query,
                    subtask_descriptions=subtasks,
                    context=request.context
                )
            
            # Get task status
            task_status = self.task_manager.get_task_status_summary(task_id)
            
            if not task_status:
                return "Task not found", {'error': 'Task not found'}
            
            # Process current subtask
            task = self.task_manager.get_task(task_id)
            if task and task.current_subtask_index < len(task.subtasks):
                current_subtask = task.subtasks[task.current_subtask_index]
                
                # Generate response for current subtask
                subtask_response = f"**Current Step ({task.current_subtask_index + 1}/{len(task.subtasks)}):** {current_subtask.description}\n\n"
                
                # Use multi-source synthesis for the subtask
                subtask_synthesis = self.synthesizer.auto_synthesize_from_query(current_subtask.description)
                subtask_response += subtask_synthesis.synthesized_content
                
                # Complete the subtask
                self.task_manager.complete_current_subtask(task_id, {
                    'response': subtask_response,
                    'confidence': subtask_synthesis.synthesis_confidence
                })
                
                # Add task progress information
                updated_status = self.task_manager.get_task_status_summary(task_id)
                progress_info = f"\n\n---\n**Task Progress:** {updated_status['progress']['percentage']:.1f}% complete ({updated_status['progress']['completed']}/{updated_status['progress']['total']} steps)"
                
                if updated_status['status'] == 'completed':
                    progress_info += "\n🎉 **Task completed!**"
                elif updated_status['current_subtask']:
                    next_step = updated_status['current_subtask']['description']
                    progress_info += f"\n**Next Step:** {next_step}"
                
                response_content = subtask_response + progress_info
            else:
                response_content = "Task is complete or no active subtask found."
            
            metadata = {
                'synthesis_type': 'multi_turn',
                'task_id': task_id,
                'task_status': task_status['status'],
                'progress_percentage': task_status['progress']['percentage'],
                'confidence': 0.7,
                'sources_used': [],
                'tools_used': ['task_manager']
            }
            
            return response_content, metadata
            
        except Exception as e:
            logger.error(f"Error in multi-turn synthesis: {e}")
            return f"Error in multi-turn synthesis: {str(e)}", {'error': str(e)}
    
    def _process_structured_synthesis(self, request: SynthesisRequest) -> Tuple[str, Dict[str, Any]]:
        """Process structured synthesis request."""
        try:
            # First get a basic response
            basic_synthesis = self.synthesizer.auto_synthesize_from_query(request.query)
            
            # Structure the response
            structured_response = self.answer_structurer.structure_response(
                query=request.query,
                raw_response=basic_synthesis.synthesized_content,
                sources=[{'title': s.title, 'content': s.content} for s in basic_synthesis.sources_used],
                reasoning_trace=request.parameters.get('reasoning_trace', []),
                tool_outputs=request.parameters.get('tool_outputs', [])
            )
            
            # Format with enhanced options
            include_decision_tree = request.parameters.get('include_decision_tree', False)
            include_metadata = request.parameters.get('include_metadata', True)
            
            formatted_response = self.answer_structurer.format_structured_response(
                structured_response,
                include_confidence=True,
                include_decision_tree=include_decision_tree,
                include_metadata=include_metadata
            )
            
            metadata = {
                'synthesis_type': 'structured',
                'sections_count': len(structured_response.sections),
                'decision_nodes': len(structured_response.decision_tree),
                'confidence': structured_response.overall_confidence,
                'sources_used': [s.title for s in basic_synthesis.sources_used],
                'tools_used': ['answer_structurer']
            }
            
            return formatted_response, metadata
            
        except Exception as e:
            logger.error(f"Error in structured synthesis: {e}")
            return f"Error in structured synthesis: {str(e)}", {'error': str(e)}
    
    def _process_chained_synthesis(self, request: SynthesisRequest) -> Tuple[str, Dict[str, Any]]:
        """Process chained synthesis request."""
        try:
            chain_id = request.parameters.get('chain_id')
            
            if not chain_id:
                return "No chain ID provided for chained synthesis", {'error': 'No chain ID'}
            
            # Execute the chain
            execution_success = self.chain_manager.execute_chain(chain_id, request.context)
            
            if not execution_success:
                return "Chain execution failed", {'error': 'Chain execution failed'}
            
            # Get chain status
            chain_status = self.chain_manager.get_chain_status(chain_id)
            
            if not chain_status:
                return "Chain status not available", {'error': 'Chain status unavailable'}
            
            # Format chain results
            response_parts = [
                f"# Capsule Chain Results: {chain_status['name']}\n",
                f"**Status:** {chain_status['status']}\n",
                f"**Success Rate:** {chain_status['success_rate']:.1%}\n",
                f"**Execution Time:** {chain_status.get('total_execution_time_ms', 0)}ms\n\n"
            ]
            
            # Add execution details
            response_parts.append("## Execution Summary\n")
            for status, count in chain_status['execution_counts'].items():
                response_parts.append(f"- {status.title()}: {count}\n")
            
            response_content = "".join(response_parts)
            
            metadata = {
                'synthesis_type': 'chained',
                'chain_id': chain_id,
                'chain_status': chain_status['status'],
                'success_rate': chain_status['success_rate'],
                'confidence': chain_status['success_rate'],
                'sources_used': [],
                'tools_used': ['capsule_chain_manager']
            }
            
            return response_content, metadata
            
        except Exception as e:
            logger.error(f"Error in chained synthesis: {e}")
            return f"Error in chained synthesis: {str(e)}", {'error': str(e)}
    
    def _apply_learning_guidance(self, request: SynthesisRequest) -> Optional[str]:
        """Apply user-guided learning to the request."""
        try:
            # Get applicable learning rules
            applicable_rules = self.guided_learning.get_applicable_rules(
                query=request.query,
                context=request.context
            )
            
            if applicable_rules:
                # Apply the most relevant rule
                best_rule = applicable_rules[0]
                guidance = self.guided_learning.apply_learning_rule(
                    rule=best_rule,
                    query=request.query,
                    context=request.context
                )
                
                return guidance
            
            return None
            
        except Exception as e:
            logger.error(f"Error applying learning guidance: {e}")
            return None
    
    def _incorporate_learning_guidance(self, response_content: str, guidance: str) -> str:
        """Incorporate learning guidance into the response."""
        try:
            # Add guidance as a note at the end
            guidance_section = f"\n\n---\n**💡 Learning Guidance Applied:** {guidance}"
            return response_content + guidance_section
            
        except Exception as e:
            logger.error(f"Error incorporating learning guidance: {e}")
            return response_content
    
    def _create_error_response(self, request: SynthesisRequest, error_message: str) -> SynthesisResponse:
        """Create an error response."""
        return SynthesisResponse(
            request_id=request.request_id,
            response_content=f"Error processing synthesis request: {error_message}",
            synthesis_metadata={'error': error_message},
            confidence_score=0.0,
            sources_used=[],
            tools_used=[],
            processing_time_ms=0,
            created_at=datetime.now().isoformat()
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                'synthesizer_available': self.synthesizer is not None,
                'task_manager_available': self.task_manager is not None,
                'answer_structurer_available': self.answer_structurer is not None,
                'chain_manager_available': self.chain_manager is not None,
                'guided_learning_available': self.guided_learning is not None,
                'active_tasks': len([t for t in self.task_manager.tasks.values() 
                                   if t.status == TaskStatus.IN_PROGRESS]) if self.task_manager else 0,
                'total_chains': len(self.chain_manager.chains) if self.chain_manager else 0,
                'feedback_entries': len(self.guided_learning.feedback_entries) if self.guided_learning else 0,
                'learning_rules': len(self.guided_learning.learning_rules) if self.guided_learning else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

# Global integrated synthesis system instance
_integrated_synthesis = None

def get_integrated_synthesis(memory_manager=None, capsule_manager=None, 
                           multimodal_pipeline=None, profile_manager=None) -> IntegratedSynthesisSystem:
    """Get or create a global integrated synthesis system instance."""
    global _integrated_synthesis
    
    if _integrated_synthesis is None:
        _integrated_synthesis = IntegratedSynthesisSystem(
            memory_manager=memory_manager,
            capsule_manager=capsule_manager,
            multimodal_pipeline=multimodal_pipeline,
            profile_manager=profile_manager
        )
    
    return _integrated_synthesis
