"""
Program Executor
===============

Execution engine for latent programs with monitoring, quality assessment,
and adaptive execution capabilities.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .latent_program import LatentProgram, ExecutionResult, ProgramExecutionError

logger = logging.getLogger(__name__)


class ProgramExecutor:
    """
    Execution engine for latent programs.
    
    Handles program execution with monitoring, quality assessment,
    and integration with SAM's existing systems.
    """
    
    def __init__(self, tpv_controller=None, reasoning_engine=None):
        """Initialize the program executor."""
        self.tpv_controller = tpv_controller
        self.reasoning_engine = reasoning_engine
        self.execution_timer = ExecutionTimer()

        # Configuration thresholds
        self.min_quality_threshold = 0.6
        self.max_execution_time_ms = 30000  # 30 seconds

    def execute(self, program: LatentProgram, query: str, context: Dict[str, Any]) -> str:
        """
        Core execute method as specified in task3.md.

        Performs streamlined response generation using the captured program's
        exact parameters and configuration.

        Args:
            program: The latent program to execute
            query: The user's query
            context: Context information

        Returns:
            Generated response string

        Raises:
            ProgramExecutionError: If execution fails
        """
        try:
            logger.info(f"🚀 Executing latent program {program.id[:8]}... for streamlined response generation")

            # Apply program's TPV configuration if available
            original_tpv_state = None
            if self.tpv_controller and program.tpv_config:
                original_tpv_state = self._get_tpv_state()
                self._apply_program_tpv_config(program)
                logger.debug(f"Applied TPV config: {program.tpv_config}")

            # Use program's proven prompt template and parameters
            if program.prompt_template_used:
                response = self._execute_with_proven_template(program, query, context)
            else:
                response = self._execute_with_proven_parameters(program, query, context)

            # Validate response quality
            if not response or len(response.strip()) < 10:
                raise ProgramExecutionError("Generated response is too short or empty")

            logger.info(f"✅ Program {program.id[:8]}... executed successfully, response length: {len(response)}")
            return response

        except Exception as e:
            logger.error(f"❌ Program execution failed: {e}")
            raise ProgramExecutionError(f"Failed to execute program {program.id}: {e}")

        finally:
            # Restore original TPV state
            if self.tpv_controller and original_tpv_state:
                self._restore_tpv_state(original_tpv_state)

    def _execute_with_proven_template(self, program: LatentProgram, query: str, context: Dict[str, Any]) -> str:
        """Execute using the program's proven prompt template."""
        try:
            # Use the exact template that was successful before
            template = program.prompt_template_used

            # Format with current query and context
            formatted_prompt = self._format_prompt_template(template, query, context)

            # Apply program's execution constraints
            execution_config = {
                'max_tokens': program.execution_constraints.get('max_tokens', 1000),
                'timeout_seconds': program.execution_constraints.get('timeout_seconds', 30),
                'temperature': program.tpv_config.get('temperature', 0.7) if program.tpv_config else 0.7
            }

            # Execute with the proven configuration
            if self.reasoning_engine:
                return self.reasoning_engine.generate_response(formatted_prompt, context, execution_config)
            else:
                # Fallback to Ollama API call with proven parameters
                return self._execute_with_ollama(formatted_prompt, execution_config)

        except Exception as e:
            raise ProgramExecutionError(f"Proven template execution failed: {e}")

    def _execute_with_proven_parameters(self, program: LatentProgram, query: str, context: Dict[str, Any]) -> str:
        """Execute using the program's proven parameters without a specific template."""
        try:
            # Build prompt using program's proven approach
            system_prompt = self._build_system_prompt_from_program(program)
            user_prompt = self._build_user_prompt_from_program(program, query, context)

            # Apply program's execution constraints
            execution_config = {
                'max_tokens': program.execution_constraints.get('max_tokens', 1000),
                'timeout_seconds': program.execution_constraints.get('timeout_seconds', 30),
                'temperature': program.tpv_config.get('temperature', 0.7) if program.tpv_config else 0.7,
                'top_p': program.tpv_config.get('top_p', 0.9) if program.tpv_config else 0.9
            }

            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

            # Execute with proven parameters
            if self.reasoning_engine:
                return self.reasoning_engine.generate_response(full_prompt, context, execution_config)
            else:
                # Fallback to Ollama API call
                return self._execute_with_ollama(full_prompt, execution_config)

        except Exception as e:
            raise ProgramExecutionError(f"Proven parameters execution failed: {e}")

    def _execute_with_ollama(self, prompt: str, config: Dict[str, Any]) -> str:
        """Execute using Ollama API with the program's proven configuration."""
        try:
            import requests

            ollama_payload = {
                "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.get('temperature', 0.7),
                    "top_p": config.get('top_p', 0.9),
                    "max_tokens": config.get('max_tokens', 1000)
                }
            }

            timeout = config.get('timeout_seconds', 30)

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=ollama_payload,
                timeout=timeout
            )

            if response.status_code == 200:
                response_data = response.json()
                ai_response = response_data.get('response', '').strip()

                if ai_response:
                    return ai_response
                else:
                    raise ProgramExecutionError("Empty response from Ollama")
            else:
                raise ProgramExecutionError(f"Ollama API error: {response.status_code}")

        except Exception as e:
            raise ProgramExecutionError(f"Ollama execution failed: {e}")

    def _build_system_prompt_from_program(self, program: LatentProgram) -> str:
        """Build system prompt based on program's proven approach."""
        # Extract system prompt characteristics from program metadata
        base_prompt = "You are SAM, a secure AI assistant."

        # Add domain-specific instructions based on program signature
        if hasattr(program, 'signature') and isinstance(program.signature, dict):
            domain_tags = program.signature.get('domain_tags', [])
            if 'cybersecurity' in domain_tags:
                base_prompt += " You specialize in cybersecurity analysis and risk assessment."
            elif 'document_analysis' in domain_tags:
                base_prompt += " You excel at document analysis and summarization."
            elif 'technical' in domain_tags:
                base_prompt += " You provide detailed technical explanations and analysis."

        # Add reasoning instructions if available
        if program.reasoning_trace:
            base_prompt += " Use structured reasoning to analyze the query step by step."

        return base_prompt

    def _build_user_prompt_from_program(self, program: LatentProgram, query: str, context: Dict[str, Any]) -> str:
        """Build user prompt based on program's proven approach."""
        # Start with the query
        user_prompt = f"Question: {query}"

        # Add context if the program typically uses it
        if context.get('memory_results') and program.context_requirements.get('min_context_size', 0) > 0:
            context_parts = []
            for result in context['memory_results'][:3]:  # Limit to top 3 results
                source_type = getattr(result, 'source_type', 'unknown')
                source_label = "📄 Document" if source_type == 'secure_documents' else "🌐 Web Knowledge"
                content_preview = result.chunk.content[:500]  # Use program's typical context size
                context_parts.append(f"{source_label} - {result.chunk.source}\nContent: {content_preview}")

            if context_parts:
                user_prompt += f"\n\nAvailable Information:\n" + "\n\n".join(context_parts)

        user_prompt += "\n\nPlease provide a helpful answer based on the available information."

        return user_prompt
    
    def execute_with_monitoring(self, program: LatentProgram, 
                              query: str, context: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a latent program with comprehensive monitoring.
        
        Args:
            program: The latent program to execute
            query: The user's query
            context: Context information
            
        Returns:
            ExecutionResult with response and metrics
        """
        self.execution_timer.start()
        original_tpv_state = None
        
        try:
            logger.info(f"Executing latent program {program.id}")
            
            # Store original TPV state if available
            if self.tpv_controller:
                original_tpv_state = self._get_tpv_state()
                self._apply_program_tpv_config(program)
            
            # Execute the program steps
            response = self._execute_program_steps(program, query, context)
            
            # Assess execution quality
            quality_score = self._assess_execution_quality(response, program, query, context)
            
            execution_time = self.execution_timer.elapsed()
            
            # Count tokens (basic implementation)
            token_count = self._estimate_token_count(response)
            
            result = ExecutionResult(
                response=response,
                quality_score=quality_score,
                execution_time_ms=execution_time,
                program_used=program.id,
                success=True,
                token_count=token_count
            )
            
            logger.info(f"Program {program.id} executed successfully in {execution_time:.1f}ms")
            return result
            
        except Exception as e:
            execution_time = self.execution_timer.elapsed()
            logger.error(f"Program execution failed: {e}")
            
            return ExecutionResult(
                response="",
                quality_score=0.0,
                execution_time_ms=execution_time,
                program_used=program.id,
                success=False,
                error_message=str(e),
                token_count=0
            )
            
        finally:
            # Restore original TPV state
            if self.tpv_controller and original_tpv_state:
                self._restore_tpv_state(original_tpv_state)
    
    def _execute_program_steps(self, program: LatentProgram, 
                             query: str, context: Dict[str, Any]) -> str:
        """Execute the core program steps."""
        try:
            # Apply program configuration
            config = self._prepare_execution_config(program, query, context)
            
            # Execute based on program type and configuration
            if program.prompt_template_used:
                response = self._execute_with_template(program, query, context, config)
            else:
                response = self._execute_standard_flow(program, query, context, config)
            
            return response
            
        except Exception as e:
            raise ProgramExecutionError(f"Failed to execute program steps: {e}")
    
    def _prepare_execution_config(self, program: LatentProgram, 
                                query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare execution configuration from program settings."""
        config = {
            'user_profile': program.active_profile,
            'tpv_config': program.tpv_config,
            'context_requirements': program.context_requirements,
            'execution_constraints': program.execution_constraints,
            'reasoning_trace': program.reasoning_trace
        }
        
        # Apply execution constraints
        if 'max_tokens' in program.execution_constraints:
            config['max_tokens'] = program.execution_constraints['max_tokens']
        
        if 'timeout_seconds' in program.execution_constraints:
            config['timeout'] = program.execution_constraints['timeout_seconds']
        
        return config
    
    def _execute_with_template(self, program: LatentProgram, query: str, 
                             context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Execute using a specific prompt template."""
        try:
            # Load the prompt template
            template = program.prompt_template_used
            
            # Format the template with query and context
            formatted_prompt = self._format_prompt_template(template, query, context)
            
            # Execute with the reasoning engine if available
            if self.reasoning_engine:
                return self.reasoning_engine.generate_response(
                    formatted_prompt, 
                    context, 
                    config
                )
            else:
                # Fallback to basic response generation
                return self._generate_basic_response(formatted_prompt, context)
                
        except Exception as e:
            raise ProgramExecutionError(f"Template execution failed: {e}")
    
    def _execute_standard_flow(self, program: LatentProgram, query: str,
                             context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Execute using standard reasoning flow with program optimizations."""
        try:
            # Apply program's proven reasoning trace if available
            if program.reasoning_trace:
                return self._execute_with_reasoning_trace(program, query, context, config)
            
            # Fallback to standard execution
            if self.reasoning_engine:
                return self.reasoning_engine.generate_response(query, context, config)
            else:
                return self._generate_basic_response(query, context)
                
        except Exception as e:
            raise ProgramExecutionError(f"Standard flow execution failed: {e}")
    
    def _execute_with_reasoning_trace(self, program: LatentProgram, query: str,
                                    context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Execute using the program's stored reasoning trace."""
        try:
            # Apply the reasoning steps from the trace
            reasoning_steps = program.reasoning_trace.get('steps', [])
            
            # Execute each step in sequence
            current_context = context.copy()
            intermediate_results = []
            
            for step in reasoning_steps:
                step_result = self._execute_reasoning_step(step, query, current_context)
                intermediate_results.append(step_result)
                
                # Update context with step result
                current_context['previous_steps'] = intermediate_results
            
            # Generate final response
            final_response = self._synthesize_final_response(
                query, context, intermediate_results, config
            )
            
            return final_response
            
        except Exception as e:
            raise ProgramExecutionError(f"Reasoning trace execution failed: {e}")
    
    def _execute_reasoning_step(self, step: Dict[str, Any], query: str, 
                              context: Dict[str, Any]) -> str:
        """Execute a single reasoning step."""
        step_type = step.get('type', 'analysis')
        step_prompt = step.get('prompt', '')
        
        # Basic step execution - would integrate with actual reasoning components
        if step_type == 'analysis':
            return f"Analysis step: {step_prompt}"
        elif step_type == 'synthesis':
            return f"Synthesis step: {step_prompt}"
        else:
            return f"Generic step: {step_prompt}"
    
    def _synthesize_final_response(self, query: str, context: Dict[str, Any],
                                 intermediate_results: list, config: Dict[str, Any]) -> str:
        """Synthesize final response from intermediate results."""
        # Basic synthesis - would integrate with actual synthesis logic
        synthesis = "Based on the analysis:\n\n"
        for i, result in enumerate(intermediate_results, 1):
            synthesis += f"{i}. {result}\n"
        
        synthesis += f"\nIn response to your query: {query}"
        return synthesis
    
    def _format_prompt_template(self, template: str, query: str, 
                              context: Dict[str, Any]) -> str:
        """Format a prompt template with query and context."""
        try:
            # Basic template formatting
            formatted = template.replace("{query}", query)
            formatted = formatted.replace("{context}", str(context))
            
            # Add more sophisticated template variables as needed
            return formatted
            
        except Exception as e:
            logger.warning(f"Template formatting failed: {e}")
            return f"{template}\n\nQuery: {query}"
    
    def _generate_basic_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate a basic response when no reasoning engine is available."""
        # Placeholder implementation
        return f"Response to: {prompt[:100]}..."
    
    def _assess_execution_quality(self, response: str, program: LatentProgram,
                                query: str, context: Dict[str, Any]) -> float:
        """Assess the quality of the execution result."""
        try:
            quality_factors = []
            
            # Response length factor (not too short, not too long)
            response_length = len(response)
            if 50 <= response_length <= 2000:
                length_score = 1.0
            elif response_length < 50:
                length_score = response_length / 50.0
            else:
                length_score = max(0.5, 2000 / response_length)
            quality_factors.append(length_score)
            
            # Relevance factor (basic keyword matching)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            relevance_score = len(query_words & response_words) / max(len(query_words), 1)
            quality_factors.append(min(1.0, relevance_score * 2))
            
            # Coherence factor (basic sentence structure check)
            sentences = response.split('.')
            coherence_score = min(1.0, len([s for s in sentences if len(s.strip()) > 10]) / max(len(sentences), 1))
            quality_factors.append(coherence_score)
            
            # Program confidence factor
            quality_factors.append(program.confidence_score)
            
            # Calculate weighted average
            weights = [0.3, 0.4, 0.2, 0.1]
            quality_score = sum(factor * weight for factor, weight in zip(quality_factors, weights))
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default moderate quality
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for the response."""
        # Basic estimation: ~4 characters per token
        return len(text) // 4
    
    def _get_tpv_state(self) -> Optional[Dict[str, Any]]:
        """Get current TPV controller state."""
        try:
            if hasattr(self.tpv_controller, 'get_current_state'):
                return self.tpv_controller.get_current_state()
        except Exception as e:
            logger.warning(f"Failed to get TPV state: {e}")
        return None
    
    def _apply_program_tpv_config(self, program: LatentProgram):
        """Apply program's TPV configuration."""
        try:
            if self.tpv_controller and program.tpv_config:
                if hasattr(self.tpv_controller, 'apply_configuration'):
                    self.tpv_controller.apply_configuration(program.tpv_config)
        except Exception as e:
            logger.warning(f"Failed to apply TPV config: {e}")
    
    def _restore_tpv_state(self, state: Dict[str, Any]):
        """Restore TPV controller state."""
        try:
            if self.tpv_controller and hasattr(self.tpv_controller, 'restore_state'):
                self.tpv_controller.restore_state(state)
        except Exception as e:
            logger.warning(f"Failed to restore TPV state: {e}")


class ExecutionTimer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000
