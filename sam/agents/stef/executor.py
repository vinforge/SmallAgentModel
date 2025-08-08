#!/usr/bin/env python3
"""
STEF Executor
Core execution engine for structured task programs with RichAnswer support.
"""

import logging
import time
from typing import Dict, Any, Optional
from .task_definitions import TaskProgram, TaskStep, ExecutionContext, StepStatus, ProgramExecutionResult

logger = logging.getLogger(__name__)

class STEF_Executor:
    """
    The core execution engine for STEF TaskPrograms.
    
    This class takes a TaskProgram and executes it step-by-step, managing
    context, error handling, and result synthesis.
    """
    
    def __init__(self, tool_registry: Dict[str, Any], llm_interface: Any):
        """
        Initialize the STEF executor.
        
        Args:
            tool_registry: Dictionary mapping tool names to tool functions
            llm_interface: LLM interface for final synthesis
        """
        self.tool_registry = tool_registry
        self.llm = llm_interface
        
        # Execution statistics
        self.programs_executed = 0
        self.programs_succeeded = 0
        self.total_execution_time = 0.0
        
        logger.info("STEF_Executor initialized")
    
    def execute(self, program: TaskProgram, initial_query: str, 
                user_context: Optional[Dict[str, Any]] = None) -> ProgramExecutionResult:
        """
        Execute a complete TaskProgram from start to finish.
        
        Args:
            program: The TaskProgram to execute
            initial_query: The original user query that triggered this program
            user_context: Optional additional context (user profile, conversation history, etc.)
            
        Returns:
            ProgramExecutionResult with execution details and final response
        """
        start_time = time.time()
        self.programs_executed += 1
        
        logger.info(f"🚀 Executing STEF Program: {program.program_name}")
        logger.info(f"📝 Query: {initial_query}")
        
        # Validate program before execution
        validation_errors = program.validate_program()
        if validation_errors:
            error_msg = f"Program validation failed: {'; '.join(validation_errors)}"
            logger.error(error_msg)
            return self._create_error_result(program, initial_query, error_msg, start_time)
        
        # Initialize execution context
        execution_context = ExecutionContext(initial_query=initial_query)
        if user_context:
            execution_context.execution_metadata.update(user_context)
        
        steps_completed = 0
        steps_failed = 0
        
        # Execute each step in sequence
        for i, step in enumerate(program.steps):
            step_number = i + 1
            total_steps = len(program.steps)
            
            logger.info(f"--- Step {step_number}/{total_steps}: {step.step_name} ---")
            
            # Set step status to running
            execution_context.set_step_status(step.step_name, StepStatus.RUNNING)
            
            try:
                # Execute the step
                step_success = self._execute_step(step, execution_context)
                
                if step_success:
                    execution_context.set_step_status(step.step_name, StepStatus.COMPLETED)
                    steps_completed += 1
                    logger.info(f"✅ Step '{step.step_name}' completed successfully")
                else:
                    execution_context.set_step_status(step.step_name, StepStatus.FAILED)
                    steps_failed += 1
                    
                    # Handle step failure
                    if step.required and not program.allow_partial_execution:
                        error_msg = f"Required step '{step.step_name}' failed"
                        logger.error(error_msg)
                        return self._create_error_result(
                            program, initial_query, error_msg, start_time,
                            execution_context, steps_completed, steps_failed
                        )
                    else:
                        logger.warning(f"⚠️ Optional step '{step.step_name}' failed, continuing...")
                        execution_context.set_step_status(step.step_name, StepStatus.SKIPPED)
                
            except Exception as e:
                execution_context.set_step_status(step.step_name, StepStatus.FAILED)
                steps_failed += 1
                error_msg = f"Step '{step.step_name}' failed with exception: {str(e)}"
                logger.error(error_msg)
                
                # Handle critical failure
                if step.required and not program.allow_partial_execution:
                    return self._create_error_result(
                        program, initial_query, error_msg, start_time,
                        execution_context, steps_completed, steps_failed
                    )
        
        # All steps completed, perform final synthesis
        try:
            logger.info("--- Final Synthesis Step ---")
            final_response = self._synthesize_final_response(program, execution_context)
            
            execution_time = (time.time() - start_time) * 1000
            self.total_execution_time += execution_time
            self.programs_succeeded += 1
            
            logger.info(f"🎉 STEF Program '{program.program_name}' completed successfully in {execution_time:.1f}ms")
            
            return ProgramExecutionResult(
                program_name=program.program_name,
                success=True,
                final_response=final_response,
                execution_context=execution_context,
                execution_time_ms=execution_time,
                steps_completed=steps_completed,
                steps_failed=steps_failed
            )
            
        except Exception as e:
            error_msg = f"Final synthesis failed: {str(e)}"
            logger.error(error_msg)
            return self._create_error_result(
                program, initial_query, error_msg, start_time,
                execution_context, steps_completed, steps_failed
            )
    
    def _execute_step(self, step: TaskStep, context: ExecutionContext) -> bool:
        """
        Execute a single step of the program.
        
        Args:
            step: The TaskStep to execute
            context: Current execution context
            
        Returns:
            True if step succeeded, False otherwise
        """
        try:
            # Check if tool exists
            if step.tool_name not in self.tool_registry:
                logger.error(f"Tool '{step.tool_name}' not found in registry")
                return False
            
            # Format the input template using context
            try:
                tool_input = context.format_template(step.input_template)
                logger.info(f"🔧 Tool: {step.tool_name}")
                logger.info(f"📥 Input: '{tool_input}'")
            except ValueError as e:
                logger.error(f"Input template formatting failed: {e}")
                return False
            
            # Execute the tool
            tool_function = self.tool_registry[step.tool_name]
            tool_output = tool_function(tool_input)
            
            # Validate output if validation function provided
            if step.validation_function and not step.validation_function(tool_output):
                logger.warning(f"Step output failed validation: {tool_output}")
                if step.required:
                    return False
            
            # Store the output in context
            context.set_step_result(step.output_key, tool_output)
            logger.info(f"📤 Output saved to context key: '{step.output_key}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            
            # Try alternative tools if available
            if step.alternative_tools:
                logger.info(f"Trying alternative tools: {step.alternative_tools}")
                for alt_tool in step.alternative_tools:
                    if alt_tool in self.tool_registry:
                        try:
                            tool_input = context.format_template(step.input_template)
                            alt_tool_function = self.tool_registry[alt_tool]
                            tool_output = alt_tool_function(tool_input)
                            context.set_step_result(step.output_key, tool_output)
                            logger.info(f"✅ Alternative tool '{alt_tool}' succeeded")
                            return True
                        except Exception as alt_e:
                            logger.warning(f"Alternative tool '{alt_tool}' also failed: {alt_e}")
                            continue
            
            return False
    
    def _synthesize_final_response(self, program: TaskProgram, context: ExecutionContext) -> str:
        """
        Generate the final response using the synthesis template.
        Now supports both RichAnswer JSON generation and fallback text responses.

        Args:
            program: The TaskProgram being executed
            context: Execution context with all step results

        Returns:
            Final synthesized response (JSON for RichAnswer or plain text)
        """
        try:
            # Format the synthesis prompt using all context data
            synthesis_prompt = context.format_template(program.synthesis_prompt_template)
            logger.info(f"🧠 Synthesis prompt: {synthesis_prompt[:200]}...")

            # Generate final response using LLM
            if self.llm:
                final_response = self.llm.generate(synthesis_prompt)
            else:
                # Fallback: Generate a simple RichAnswer JSON structure
                final_response = self._generate_fallback_rich_answer(program, context)

            return final_response

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Return a fallback RichAnswer JSON
            return self._generate_error_rich_answer(program, context, str(e))

    def _generate_fallback_rich_answer(self, program: TaskProgram, context: ExecutionContext) -> str:
        """Generate a fallback RichAnswer JSON when LLM is not available."""
        try:
            # Extract the main result from context
            main_result = ""
            if 'calculation_result' in context.step_results:
                main_result = str(context.step_results['calculation_result'])
            elif context.step_results:
                # Use the last step result
                main_result = str(list(context.step_results.values())[-1])
            else:
                main_result = "Task completed successfully"

            # Determine query type based on program name
            query_type = "pure_math" if "SimpleCalculation" in program.program_name else "complex_calculation"

            fallback_json = {
                "direct_answer": main_result,
                "query_type": query_type,
                "presentation_type": "minimal" if query_type == "pure_math" else "detailed",
                "confidence": 0.8,
                "supporting_evidence": list(context.execution_metadata.get('tools_used', [])),
                "reasoning_summary": f"Completed {program.program_name} program",
                "detailed_reasoning": f"Executed {len(context.step_results)} steps successfully"
            }

            import json
            return json.dumps(fallback_json, indent=2)

        except Exception as e:
            logger.error(f"Fallback RichAnswer generation failed: {e}")
            return f'{{"direct_answer": "Task completed", "query_type": "error", "confidence": 0.5}}'

    def _generate_error_rich_answer(self, program: TaskProgram, context: ExecutionContext, error: str) -> str:
        """Generate an error RichAnswer JSON."""
        try:
            error_json = {
                "direct_answer": f"Task execution failed: {error}",
                "query_type": "error",
                "presentation_type": "error_friendly",
                "confidence": 0.0,
                "error_details": error
            }

            import json
            return json.dumps(error_json, indent=2)

        except Exception as e:
            logger.error(f"Error RichAnswer generation failed: {e}")
            return f'{{"direct_answer": "Execution failed", "query_type": "error", "confidence": 0.0}}'
    
    def _create_error_result(self, program: TaskProgram, initial_query: str, 
                           error_msg: str, start_time: float,
                           context: Optional[ExecutionContext] = None,
                           steps_completed: int = 0, steps_failed: int = 0) -> ProgramExecutionResult:
        """Create a ProgramExecutionResult for error cases."""
        execution_time = (time.time() - start_time) * 1000
        
        # Format error message using program template if available
        try:
            formatted_error = program.error_message_template.format(
                program_name=program.program_name,
                error_details=error_msg
            )
        except:
            formatted_error = f"Error in {program.program_name}: {error_msg}"
        
        return ProgramExecutionResult(
            program_name=program.program_name,
            success=False,
            final_response=formatted_error,
            execution_context=context or ExecutionContext(initial_query),
            execution_time_ms=execution_time,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            error_message=error_msg
        )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for monitoring and debugging."""
        success_rate = (self.programs_succeeded / self.programs_executed * 100) if self.programs_executed > 0 else 0
        avg_execution_time = (self.total_execution_time / self.programs_executed) if self.programs_executed > 0 else 0
        
        return {
            'programs_executed': self.programs_executed,
            'programs_succeeded': self.programs_succeeded,
            'success_rate_percent': success_rate,
            'average_execution_time_ms': avg_execution_time,
            'total_execution_time_ms': self.total_execution_time
        }
