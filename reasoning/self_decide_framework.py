"""
SELF-DECIDE Framework for SAM
Structured reasoning framework for autonomous problem-solving with tool integration.

Sprint 5 Task 1: SELF-DECIDE Framework Integration
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ReasoningStep(Enum):
    """Enumeration of SELF-DECIDE reasoning steps."""
    STATE_QUERY = "state_query"
    EXPLORE_RETRIEVALS = "explore_retrievals"
    LABEL_GAPS = "label_knowledge_gaps"
    FORMULATE_PLAN = "formulate_plan"
    DECIDE_TOOLS = "decide_tools"
    EXECUTE_TOOLS = "execute_tools"
    CONNECT_RESULTS = "connect_results"
    INFER_ANSWER = "infer_answer"
    DOCUMENT_PROCESS = "document_thought_process"
    EVALUATE_QUALITY = "evaluate_response_quality"

@dataclass
class ReasoningStepResult:
    """Result of a single reasoning step."""
    step: ReasoningStep
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reasoning: str
    confidence: float
    timestamp: str
    duration_ms: int

@dataclass
class KnowledgeGap:
    """Represents an identified knowledge gap."""
    gap_type: str  # 'factual', 'procedural', 'contextual', 'computational'
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    suggested_tools: List[str]

@dataclass
class ReasoningPlan:
    """Structured plan for addressing the query."""
    objective: str
    approach: str
    required_tools: List[str]
    expected_steps: List[str]
    success_criteria: str
    fallback_strategy: str

@dataclass
class SelfDecideSession:
    """Complete SELF-DECIDE reasoning session."""
    session_id: str
    original_query: str
    reasoning_steps: List[ReasoningStepResult]
    knowledge_gaps: List[KnowledgeGap]
    reasoning_plan: Optional[ReasoningPlan]
    tool_executions: List[Dict[str, Any]]
    final_answer: str
    confidence_score: float
    total_duration_ms: int
    created_at: str

class SelfDecideFramework:
    """
    Implementation of the SELF-DECIDE structured reasoning framework.

    SELF-DECIDE Steps:
    1. State the Query - Understand and clarify the user's request
    2. Explore Retrievals - Search existing knowledge base
    3. Label Knowledge Gaps - Identify what information is missing
    4. Formulate a Plan - Create structured approach to solve the problem
    5. Decide on Tools - Select appropriate tools for execution
    6. Execute Tool(s) - Run selected tools and gather results
    7. Connect Results to Query - Link tool outputs to original question
    8. Infer an Answer - Synthesize final response from all information
    9. Document Thought Process - Create transparent reasoning log
    10. Evaluate Response Quality - Assess answer completeness and accuracy
    """

    def __init__(self, model=None, vector_manager=None, tool_selector=None):
        """
        Initialize the SELF-DECIDE framework.

        Args:
            model: Language model for reasoning
            vector_manager: Vector store for knowledge retrieval
            tool_selector: Tool selection module
        """
        self.model = model
        self.vector_manager = vector_manager
        self.tool_selector = tool_selector

        # Session tracking
        self.active_sessions: Dict[str, SelfDecideSession] = {}

        logger.info("SELF-DECIDE framework initialized")

    def reason(self, query: str, context: Optional[Dict[str, Any]] = None) -> SelfDecideSession:
        """
        Execute the complete SELF-DECIDE reasoning process.

        Args:
            query: User's query to reason about
            context: Optional context information

        Returns:
            Complete reasoning session with all steps and results
        """
        start_time = datetime.now()
        session_id = f"decide_{int(start_time.timestamp())}_{hash(query) % 10000}"

        logger.info(f"Starting SELF-DECIDE reasoning session: {session_id}")

        # Initialize session
        session = SelfDecideSession(
            session_id=session_id,
            original_query=query,
            reasoning_steps=[],
            knowledge_gaps=[],
            reasoning_plan=None,
            tool_executions=[],
            final_answer="",
            confidence_score=0.0,
            total_duration_ms=0,
            created_at=start_time.isoformat()
        )

        try:
            # Execute SELF-DECIDE steps
            self._step_1_state_query(session, query, context)
            self._step_2_explore_retrievals(session)
            self._step_3_label_knowledge_gaps(session)
            self._step_4_formulate_plan(session)
            self._step_5_decide_tools(session)
            self._step_6_execute_tools(session)
            self._step_7_connect_results(session)
            self._step_8_infer_answer(session)
            self._step_9_document_process(session)
            self._step_10_evaluate_quality(session)

            # Calculate total duration
            end_time = datetime.now()
            session.total_duration_ms = int((end_time - start_time).total_seconds() * 1000)

            # Store session
            self.active_sessions[session_id] = session

            logger.info(f"SELF-DECIDE reasoning completed: {session_id} ({session.total_duration_ms}ms)")
            return session

        except Exception as e:
            logger.error(f"Error in SELF-DECIDE reasoning: {e}")
            session.final_answer = f"Reasoning process encountered an error: {str(e)}"
            session.confidence_score = 0.0
            return session

    def _step_1_state_query(self, session: SelfDecideSession, query: str, context: Optional[Dict[str, Any]]):
        """Step 1: State the Query - Understand and clarify the user's request."""
        step_start = datetime.now()

        try:
            # Analyze the query to understand intent, complexity, and requirements
            analysis_prompt = f"""Analyze this user query to understand its intent, complexity, and requirements:

Query: "{query}"
Context: {json.dumps(context) if context else "None"}

Provide analysis in this format:
- Intent: [What the user wants to achieve]
- Query Type: [factual/analytical/computational/creative/procedural]
- Complexity: [simple/moderate/complex/multi-step]
- Domain: [subject area or field]
- Key Entities: [important nouns, concepts, or objects]
- Required Information: [what information is needed to answer]
- Expected Output Format: [text/data/code/visualization/etc.]

Analysis:"""

            if self.model:
                analysis_result = self.model.generate(analysis_prompt, temperature=0.3, max_tokens=400)
            else:
                analysis_result = "Query analysis not available (no model provided)"

            # Extract structured information from analysis
            query_understanding = self._parse_query_analysis(analysis_result)

            step_result = ReasoningStepResult(
                step=ReasoningStep.STATE_QUERY,
                input_data={"query": query, "context": context},
                output_data={"analysis": analysis_result, "understanding": query_understanding},
                reasoning="Analyzed user query to understand intent, complexity, and requirements",
                confidence=0.9,
                timestamp=datetime.now().isoformat(),
                duration_ms=int((datetime.now() - step_start).total_seconds() * 1000)
            )

            session.reasoning_steps.append(step_result)
            logger.debug(f"Step 1 completed: Query understanding - {query_understanding.get('intent', 'unknown')}")

        except Exception as e:
            logger.error(f"Error in step 1 (state query): {e}")
            self._add_error_step(session, ReasoningStep.STATE_QUERY, str(e), step_start)

    def _step_2_explore_retrievals(self, session: SelfDecideSession):
        """Step 2: Explore Retrievals - Search existing knowledge base."""
        step_start = datetime.now()

        try:
            query = session.original_query
            retrievals = []

            # Search vector store if available
            if self.vector_manager:
                try:
                    from utils.embedding_utils import get_embedding_manager
                    embedding_manager = get_embedding_manager()

                    # Generate query embedding
                    query_embedding = embedding_manager.embed_query(query)

                    # Set query context for memory adapter if available
                    if hasattr(self.vector_manager, 'set_query_context'):
                        self.vector_manager.set_query_context(query)

                    # Search for relevant chunks
                    search_results = self.vector_manager.search(query_embedding, top_k=5, score_threshold=0.1)

                    for result in search_results:
                        retrievals.append({
                            'source': 'vector_store',
                            'content': result.get('text', '')[:300],
                            'similarity': result.get('similarity_score', 0),
                            'metadata': result.get('metadata', {})
                        })

                except Exception as e:
                    logger.warning(f"Vector search failed: {e}")

            # Analyze retrieval quality and relevance
            retrieval_analysis = self._analyze_retrievals(retrievals, query)

            step_result = ReasoningStepResult(
                step=ReasoningStep.EXPLORE_RETRIEVALS,
                input_data={"query": query},
                output_data={"retrievals": retrievals, "analysis": retrieval_analysis},
                reasoning=f"Searched knowledge base and found {len(retrievals)} relevant items",
                confidence=0.8 if retrievals else 0.3,
                timestamp=datetime.now().isoformat(),
                duration_ms=int((datetime.now() - step_start).total_seconds() * 1000)
            )

            session.reasoning_steps.append(step_result)
            logger.debug(f"Step 2 completed: Found {len(retrievals)} retrievals")

        except Exception as e:
            logger.error(f"Error in step 2 (explore retrievals): {e}")
            self._add_error_step(session, ReasoningStep.EXPLORE_RETRIEVALS, str(e), step_start)

    def _parse_query_analysis(self, analysis_text: str) -> Dict[str, str]:
        """Parse structured query analysis from model output."""
        understanding = {}

        lines = analysis_text.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip('- ').lower().replace(' ', '_')
                value = value.strip()
                understanding[key] = value

        return understanding

    def _analyze_retrievals(self, retrievals: List[Dict], query: str) -> Dict[str, Any]:
        """Analyze the quality and relevance of retrieved information."""
        if not retrievals:
            return {
                'quality': 'poor',
                'coverage': 'none',
                'relevance': 'low',
                'gaps': ['No relevant information found in knowledge base']
            }

        # Calculate average similarity
        similarities = [r.get('similarity', 0) for r in retrievals]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # Assess coverage
        total_content_length = sum(len(r.get('content', '')) for r in retrievals)

        analysis = {
            'quality': 'good' if avg_similarity > 0.7 else 'moderate' if avg_similarity > 0.4 else 'poor',
            'coverage': 'comprehensive' if total_content_length > 1000 else 'partial' if total_content_length > 300 else 'limited',
            'relevance': 'high' if avg_similarity > 0.6 else 'medium' if avg_similarity > 0.3 else 'low',
            'avg_similarity': avg_similarity,
            'total_content': total_content_length,
            'source_count': len(retrievals)
        }

        return analysis

    def _add_error_step(self, session: SelfDecideSession, step: ReasoningStep, error: str, start_time: datetime):
        """Add an error step result to the session."""
        step_result = ReasoningStepResult(
            step=step,
            input_data={},
            output_data={"error": error},
            reasoning=f"Step failed with error: {error}",
            confidence=0.0,
            timestamp=datetime.now().isoformat(),
            duration_ms=int((datetime.now() - start_time).total_seconds() * 1000)
        )
        session.reasoning_steps.append(step_result)

    def _step_3_label_knowledge_gaps(self, session: SelfDecideSession):
        """Step 3: Label Knowledge Gaps - Identify what information is missing."""
        step_start = datetime.now()

        try:
            # Get previous step results
            query_step = next((s for s in session.reasoning_steps if s.step == ReasoningStep.STATE_QUERY), None)
            retrieval_step = next((s for s in session.reasoning_steps if s.step == ReasoningStep.EXPLORE_RETRIEVALS), None)

            query_understanding = query_step.output_data.get('understanding', {}) if query_step else {}
            retrievals = retrieval_step.output_data.get('retrievals', []) if retrieval_step else []
            retrieval_analysis = retrieval_step.output_data.get('analysis', {}) if retrieval_step else {}

            # Identify knowledge gaps
            gaps = []

            # Check if we have sufficient information
            if not retrievals or retrieval_analysis.get('relevance') == 'low':
                gaps.append(KnowledgeGap(
                    gap_type='factual',
                    description='Insufficient relevant information in knowledge base',
                    severity='high',
                    suggested_tools=['web_search', 'multimodal_query']
                ))

            # Check for computational needs
            query_type = query_understanding.get('query_type', '').lower()
            if 'computational' in query_type or any(word in session.original_query.lower()
                                                  for word in ['calculate', 'compute', 'analyze data', 'plot', 'graph']):
                gaps.append(KnowledgeGap(
                    gap_type='computational',
                    description='Query requires computational analysis or data processing',
                    severity='medium',
                    suggested_tools=['python_interpreter', 'table_generator']
                ))

            # Check for procedural knowledge
            if any(word in session.original_query.lower()
                   for word in ['how to', 'steps', 'process', 'procedure', 'method']):
                if retrieval_analysis.get('coverage') != 'comprehensive':
                    gaps.append(KnowledgeGap(
                        gap_type='procedural',
                        description='Incomplete procedural knowledge for step-by-step guidance',
                        severity='medium',
                        suggested_tools=['multimodal_query', 'web_search']
                    ))

            # Check for contextual understanding
            if query_understanding.get('complexity') in ['complex', 'multi-step']:
                gaps.append(KnowledgeGap(
                    gap_type='contextual',
                    description='Complex query may require additional context and reasoning',
                    severity='low',
                    suggested_tools=['python_interpreter', 'multimodal_query']
                ))

            session.knowledge_gaps = gaps

            step_result = ReasoningStepResult(
                step=ReasoningStep.LABEL_GAPS,
                input_data={"query_understanding": query_understanding, "retrieval_analysis": retrieval_analysis},
                output_data={"knowledge_gaps": [asdict(gap) for gap in gaps]},
                reasoning=f"Identified {len(gaps)} knowledge gaps that need to be addressed",
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                duration_ms=int((datetime.now() - step_start).total_seconds() * 1000)
            )

            session.reasoning_steps.append(step_result)
            logger.debug(f"Step 3 completed: Identified {len(gaps)} knowledge gaps")

        except Exception as e:
            logger.error(f"Error in step 3 (label knowledge gaps): {e}")
            self._add_error_step(session, ReasoningStep.LABEL_GAPS, str(e), step_start)

    def _step_4_formulate_plan(self, session: SelfDecideSession):
        """Step 4: Formulate a Plan - Create structured approach to solve the problem."""
        step_start = datetime.now()

        try:
            # Get previous step results
            query_step = next((s for s in session.reasoning_steps if s.step == ReasoningStep.STATE_QUERY), None)
            query_understanding = query_step.output_data.get('understanding', {}) if query_step else {}

            # Create reasoning plan based on query analysis and knowledge gaps
            objective = query_understanding.get('intent', 'Answer the user query')
            complexity = query_understanding.get('complexity', 'moderate')

            # Determine approach based on complexity and gaps
            if complexity == 'simple' and not session.knowledge_gaps:
                approach = "Direct response using existing knowledge"
                required_tools = []
            elif session.knowledge_gaps:
                approach = "Multi-step approach: fill knowledge gaps, then synthesize answer"
                required_tools = list(set(tool for gap in session.knowledge_gaps for tool in gap.suggested_tools))
            else:
                approach = "Analytical approach with tool assistance"
                required_tools = ['multimodal_query']

            # Define expected steps
            expected_steps = [
                "Gather additional information using selected tools",
                "Analyze and validate tool outputs",
                "Synthesize comprehensive response",
                "Verify answer completeness and accuracy"
            ]

            # Success criteria
            success_criteria = "Response addresses all aspects of the query with appropriate detail and accuracy"

            # Fallback strategy
            fallback_strategy = "If tools fail, provide best possible answer using available knowledge with clear limitations noted"

            plan = ReasoningPlan(
                objective=objective,
                approach=approach,
                required_tools=required_tools,
                expected_steps=expected_steps,
                success_criteria=success_criteria,
                fallback_strategy=fallback_strategy
            )

            session.reasoning_plan = plan

            step_result = ReasoningStepResult(
                step=ReasoningStep.FORMULATE_PLAN,
                input_data={"query_understanding": query_understanding, "knowledge_gaps": len(session.knowledge_gaps)},
                output_data={"reasoning_plan": asdict(plan)},
                reasoning=f"Created {approach.lower()} with {len(required_tools)} tools",
                confidence=0.9,
                timestamp=datetime.now().isoformat(),
                duration_ms=int((datetime.now() - step_start).total_seconds() * 1000)
            )

            session.reasoning_steps.append(step_result)
            logger.debug(f"Step 4 completed: Plan created with {len(required_tools)} tools")

        except Exception as e:
            logger.error(f"Error in step 4 (formulate plan): {e}")
            self._add_error_step(session, ReasoningStep.FORMULATE_PLAN, str(e), step_start)

    def _step_5_decide_tools(self, session: SelfDecideSession):
        """Step 5: Decide on Tools - Select appropriate tools for execution."""
        step_start = datetime.now()

        try:
            # Get required tools from plan
            required_tools = session.reasoning_plan.required_tools if session.reasoning_plan else []

            # Use tool selector if available
            selected_tools = []
            tool_rationales = {}

            if self.tool_selector and required_tools:
                for tool_name in required_tools:
                    try:
                        tool_decision = self.tool_selector.select_tool(
                            query=session.original_query,
                            context={'knowledge_gaps': session.knowledge_gaps, 'plan': session.reasoning_plan},
                            preferred_tool=tool_name
                        )

                        if tool_decision:
                            selected_tools.append(tool_decision)
                            tool_rationales[tool_name] = tool_decision.get('rationale', 'Tool selected for query requirements')

                    except Exception as e:
                        logger.warning(f"Tool selection failed for {tool_name}: {e}")

            # Fallback: create basic tool selections
            if not selected_tools and required_tools:
                for tool_name in required_tools:
                    selected_tools.append({
                        'tool_name': tool_name,
                        'input_params': {'query': session.original_query},
                        'rationale': f'Selected {tool_name} to address identified knowledge gaps'
                    })
                    tool_rationales[tool_name] = f'Required for {tool_name} functionality'

            step_result = ReasoningStepResult(
                step=ReasoningStep.DECIDE_TOOLS,
                input_data={"required_tools": required_tools},
                output_data={"selected_tools": selected_tools, "rationales": tool_rationales},
                reasoning=f"Selected {len(selected_tools)} tools for execution",
                confidence=0.8 if selected_tools else 0.4,
                timestamp=datetime.now().isoformat(),
                duration_ms=int((datetime.now() - step_start).total_seconds() * 1000)
            )

            session.reasoning_steps.append(step_result)
            logger.debug(f"Step 5 completed: Selected {len(selected_tools)} tools")

        except Exception as e:
            logger.error(f"Error in step 5 (decide tools): {e}")
            self._add_error_step(session, ReasoningStep.DECIDE_TOOLS, str(e), step_start)

    def _step_6_execute_tools(self, session: SelfDecideSession):
        """Step 6: Execute Tool(s) - Run selected tools and gather results."""
        step_start = datetime.now()

        try:
            # Get selected tools from previous step
            tools_step = next((s for s in session.reasoning_steps if s.step == ReasoningStep.DECIDE_TOOLS), None)
            selected_tools = tools_step.output_data.get('selected_tools', []) if tools_step else []

            tool_results = []

            # Execute each selected tool
            for tool_config in selected_tools:
                tool_name = tool_config.get('tool_name')
                input_params = tool_config.get('input_params', {})

                try:
                    # Execute tool (placeholder - will be implemented with actual tool executor)
                    result = self._execute_single_tool(tool_name, input_params)

                    tool_execution = {
                        'tool_name': tool_name,
                        'input_params': input_params,
                        'result': result,
                        'success': result.get('success', False),
                        'execution_time_ms': result.get('execution_time_ms', 0),
                        'timestamp': datetime.now().isoformat()
                    }

                    tool_results.append(tool_execution)
                    session.tool_executions.append(tool_execution)

                except Exception as e:
                    logger.error(f"Tool execution failed for {tool_name}: {e}")
                    tool_execution = {
                        'tool_name': tool_name,
                        'input_params': input_params,
                        'result': {'success': False, 'error': str(e)},
                        'success': False,
                        'execution_time_ms': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                    tool_results.append(tool_execution)
                    session.tool_executions.append(tool_execution)

            successful_executions = sum(1 for r in tool_results if r['success'])

            step_result = ReasoningStepResult(
                step=ReasoningStep.EXECUTE_TOOLS,
                input_data={"selected_tools": selected_tools},
                output_data={"tool_results": tool_results, "successful_executions": successful_executions},
                reasoning=f"Executed {len(tool_results)} tools, {successful_executions} successful",
                confidence=0.9 if successful_executions == len(tool_results) else 0.6 if successful_executions > 0 else 0.2,
                timestamp=datetime.now().isoformat(),
                duration_ms=int((datetime.now() - step_start).total_seconds() * 1000)
            )

            session.reasoning_steps.append(step_result)
            logger.debug(f"Step 6 completed: {successful_executions}/{len(tool_results)} tools executed successfully")

        except Exception as e:
            logger.error(f"Error in step 6 (execute tools): {e}")
            self._add_error_step(session, ReasoningStep.EXECUTE_TOOLS, str(e), step_start)

    def _step_7_connect_results(self, session: SelfDecideSession):
        """Step 7: Connect Results to Query - Link tool outputs to original question."""
        step_start = datetime.now()

        try:
            # Gather all available information
            original_query = session.original_query
            retrievals = []
            tool_outputs = []

            # Get retrieval results
            retrieval_step = next((s for s in session.reasoning_steps if s.step == ReasoningStep.EXPLORE_RETRIEVALS), None)
            if retrieval_step:
                retrievals = retrieval_step.output_data.get('retrievals', [])

            # Get tool execution results
            execution_step = next((s for s in session.reasoning_steps if s.step == ReasoningStep.EXECUTE_TOOLS), None)
            if execution_step:
                tool_results = execution_step.output_data.get('tool_results', [])
                for result in tool_results:
                    if result['success']:
                        tool_outputs.append({
                            'tool': result['tool_name'],
                            'output': result['result'].get('output', ''),
                            'metadata': result['result'].get('metadata', {})
                        })

            # Analyze connections between results and query
            connections = self._analyze_result_connections(original_query, retrievals, tool_outputs)

            step_result = ReasoningStepResult(
                step=ReasoningStep.CONNECT_RESULTS,
                input_data={"retrievals_count": len(retrievals), "tool_outputs_count": len(tool_outputs)},
                output_data={"connections": connections, "retrievals": retrievals, "tool_outputs": tool_outputs},
                reasoning=f"Connected {len(retrievals)} retrievals and {len(tool_outputs)} tool outputs to query",
                confidence=0.8 if (retrievals or tool_outputs) else 0.3,
                timestamp=datetime.now().isoformat(),
                duration_ms=int((datetime.now() - step_start).total_seconds() * 1000)
            )

            session.reasoning_steps.append(step_result)
            logger.debug(f"Step 7 completed: Connected {len(retrievals + tool_outputs)} information sources")

        except Exception as e:
            logger.error(f"Error in step 7 (connect results): {e}")
            self._add_error_step(session, ReasoningStep.CONNECT_RESULTS, str(e), step_start)

    def _step_8_infer_answer(self, session: SelfDecideSession):
        """Step 8: Infer an Answer - Synthesize final response from all information."""
        step_start = datetime.now()

        try:
            # Get all available information
            connect_step = next((s for s in session.reasoning_steps if s.step == ReasoningStep.CONNECT_RESULTS), None)
            connections = connect_step.output_data.get('connections', {}) if connect_step else {}
            retrievals = connect_step.output_data.get('retrievals', []) if connect_step else []
            tool_outputs = connect_step.output_data.get('tool_outputs', []) if connect_step else []

            # Synthesize final answer
            if self.model:
                synthesis_prompt = self._create_synthesis_prompt(session.original_query, retrievals, tool_outputs, connections)
                final_answer = self.model.generate(synthesis_prompt, temperature=0.4, max_tokens=800)
            else:
                final_answer = self._create_fallback_answer(session.original_query, retrievals, tool_outputs)

            # Clean up the answer
            final_answer = self._clean_answer(final_answer)
            session.final_answer = final_answer

            # Calculate confidence based on available information
            confidence = self._calculate_answer_confidence(retrievals, tool_outputs, connections)
            session.confidence_score = confidence

            step_result = ReasoningStepResult(
                step=ReasoningStep.INFER_ANSWER,
                input_data={"sources_count": len(retrievals + tool_outputs)},
                output_data={"final_answer": final_answer, "confidence": confidence},
                reasoning="Synthesized final answer from all available information sources",
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                duration_ms=int((datetime.now() - step_start).total_seconds() * 1000)
            )

            session.reasoning_steps.append(step_result)
            logger.debug(f"Step 8 completed: Generated answer with {confidence:.2f} confidence")

        except Exception as e:
            logger.error(f"Error in step 8 (infer answer): {e}")
            self._add_error_step(session, ReasoningStep.INFER_ANSWER, str(e), step_start)

    def _step_9_document_process(self, session: SelfDecideSession):
        """Step 9: Document Thought Process - Create transparent reasoning log."""
        step_start = datetime.now()

        try:
            # Create comprehensive reasoning documentation
            reasoning_log = self._create_reasoning_log(session)

            step_result = ReasoningStepResult(
                step=ReasoningStep.DOCUMENT_PROCESS,
                input_data={"session_steps": len(session.reasoning_steps)},
                output_data={"reasoning_log": reasoning_log},
                reasoning="Documented complete reasoning process for transparency",
                confidence=1.0,
                timestamp=datetime.now().isoformat(),
                duration_ms=int((datetime.now() - step_start).total_seconds() * 1000)
            )

            session.reasoning_steps.append(step_result)
            logger.debug("Step 9 completed: Reasoning process documented")

        except Exception as e:
            logger.error(f"Error in step 9 (document process): {e}")
            self._add_error_step(session, ReasoningStep.DOCUMENT_PROCESS, str(e), step_start)

    def _step_10_evaluate_quality(self, session: SelfDecideSession):
        """Step 10: Evaluate Response Quality - Assess answer completeness and accuracy."""
        step_start = datetime.now()

        try:
            # Evaluate response quality across multiple dimensions
            quality_assessment = self._evaluate_response_quality(session)

            step_result = ReasoningStepResult(
                step=ReasoningStep.EVALUATE_QUALITY,
                input_data={"final_answer_length": len(session.final_answer)},
                output_data={"quality_assessment": quality_assessment},
                reasoning="Evaluated response quality across completeness, accuracy, and clarity dimensions",
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                duration_ms=int((datetime.now() - step_start).total_seconds() * 1000)
            )

            session.reasoning_steps.append(step_result)
            logger.debug(f"Step 10 completed: Quality score {quality_assessment.get('overall_score', 0):.2f}")

        except Exception as e:
            logger.error(f"Error in step 10 (evaluate quality): {e}")
            self._add_error_step(session, ReasoningStep.EVALUATE_QUALITY, str(e), step_start)

    # Helper methods for SELF-DECIDE steps

    def _execute_single_tool(self, tool_name: str, input_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool and return results."""
        # Placeholder implementation - will be replaced with actual tool executor
        execution_start = datetime.now()

        try:
            if tool_name == 'python_interpreter':
                # Placeholder for Python code execution
                result = {
                    'success': True,
                    'output': 'Python execution placeholder result',
                    'metadata': {'tool': 'python_interpreter'}
                }
            elif tool_name == 'table_generator':
                # Placeholder for table generation
                result = {
                    'success': True,
                    'output': 'Generated table placeholder',
                    'metadata': {'tool': 'table_generator'}
                }
            elif tool_name == 'multimodal_query':
                # Placeholder for multimodal query
                result = {
                    'success': True,
                    'output': 'Multimodal query result placeholder',
                    'metadata': {'tool': 'multimodal_query'}
                }
            elif tool_name == 'web_search':
                # Placeholder for web search
                result = {
                    'success': True,
                    'output': 'Web search results placeholder',
                    'metadata': {'tool': 'web_search'}
                }
            else:
                result = {
                    'success': False,
                    'error': f'Unknown tool: {tool_name}'
                }

            execution_time = int((datetime.now() - execution_start).total_seconds() * 1000)
            result['execution_time_ms'] = execution_time

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': int((datetime.now() - execution_start).total_seconds() * 1000)
            }

    def _analyze_result_connections(self, query: str, retrievals: List[Dict], tool_outputs: List[Dict]) -> Dict[str, Any]:
        """Analyze how results connect to the original query."""
        connections = {
            'direct_answers': [],
            'supporting_evidence': [],
            'computational_results': [],
            'gaps_filled': []
        }

        # Analyze retrievals
        for retrieval in retrievals:
            if retrieval.get('similarity', 0) > 0.7:
                connections['direct_answers'].append({
                    'source': 'knowledge_base',
                    'content': retrieval.get('content', ''),
                    'relevance': 'high'
                })
            elif retrieval.get('similarity', 0) > 0.4:
                connections['supporting_evidence'].append({
                    'source': 'knowledge_base',
                    'content': retrieval.get('content', ''),
                    'relevance': 'medium'
                })

        # Analyze tool outputs
        for tool_output in tool_outputs:
            tool_name = tool_output.get('tool', '')
            if tool_name in ['python_interpreter', 'table_generator']:
                connections['computational_results'].append({
                    'tool': tool_name,
                    'output': tool_output.get('output', ''),
                    'type': 'computational'
                })
            else:
                connections['gaps_filled'].append({
                    'tool': tool_name,
                    'output': tool_output.get('output', ''),
                    'type': 'informational'
                })

        return connections

    def _create_synthesis_prompt(self, query: str, retrievals: List[Dict], tool_outputs: List[Dict], connections: Dict[str, Any]) -> str:
        """Create prompt for synthesizing final answer."""
        prompt_parts = [
            f"You are SAM, an intelligent assistant using the SELF-DECIDE reasoning framework.",
            f"Synthesize a comprehensive answer to this query using all available information:",
            f"",
            f"**Original Query:** {query}",
            f""
        ]

        # Add knowledge base information
        if retrievals:
            prompt_parts.append("**Knowledge Base Information:**")
            for i, retrieval in enumerate(retrievals[:3], 1):
                content = retrieval.get('content', '')[:200]
                similarity = retrieval.get('similarity', 0)
                prompt_parts.append(f"{i}. (Similarity: {similarity:.2f}) {content}...")
            prompt_parts.append("")

        # Add tool results
        if tool_outputs:
            prompt_parts.append("**Tool Execution Results:**")
            for i, tool_output in enumerate(tool_outputs, 1):
                tool_name = tool_output.get('tool', 'unknown')
                output = tool_output.get('output', '')[:300]
                prompt_parts.append(f"{i}. {tool_name}: {output}")
            prompt_parts.append("")

        # Add synthesis instructions
        prompt_parts.extend([
            "**Instructions:**",
            "1. Provide a comprehensive answer that addresses all aspects of the query",
            "2. Integrate information from both knowledge base and tool results",
            "3. Be clear about the sources of information",
            "4. If information is incomplete, acknowledge limitations",
            "5. Use markdown formatting for clarity",
            "",
            "**Answer:**"
        ])

        return "\n".join(prompt_parts)

    def _create_fallback_answer(self, query: str, retrievals: List[Dict], tool_outputs: List[Dict]) -> str:
        """Create fallback answer when model is not available."""
        answer_parts = [
            f"Based on available information for your query: '{query}'",
            ""
        ]

        if retrievals:
            answer_parts.append("**From Knowledge Base:**")
            for retrieval in retrievals[:2]:
                content = retrieval.get('content', '')[:150]
                answer_parts.append(f"- {content}...")
            answer_parts.append("")

        if tool_outputs:
            answer_parts.append("**From Tool Execution:**")
            for tool_output in tool_outputs:
                tool_name = tool_output.get('tool', 'unknown')
                output = tool_output.get('output', '')[:150]
                answer_parts.append(f"- {tool_name}: {output}")
            answer_parts.append("")

        if not retrievals and not tool_outputs:
            answer_parts.append("I don't have sufficient information to provide a comprehensive answer to your query.")

        return "\n".join(answer_parts)

    def _clean_answer(self, answer: str) -> str:
        """Clean up the generated answer."""
        # Remove thinking tags if present
        if '<think>' in answer:
            parts = answer.split('</think>')
            if len(parts) > 1:
                answer = parts[-1].strip()

        # Remove leading "Answer:" if present
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()

        return answer.strip()

    def _calculate_answer_confidence(self, retrievals: List[Dict], tool_outputs: List[Dict], connections: Dict[str, Any]) -> float:
        """Calculate confidence score for the final answer."""
        confidence = 0.5  # Base confidence

        # Boost confidence based on retrievals
        if retrievals:
            avg_similarity = sum(r.get('similarity', 0) for r in retrievals) / len(retrievals)
            confidence += avg_similarity * 0.3

        # Boost confidence based on successful tool executions
        if tool_outputs:
            confidence += min(len(tool_outputs) * 0.1, 0.3)

        # Boost confidence based on connection quality
        direct_answers = len(connections.get('direct_answers', []))
        if direct_answers > 0:
            confidence += min(direct_answers * 0.1, 0.2)

        return min(confidence, 1.0)

    def _create_reasoning_log(self, session: SelfDecideSession) -> str:
        """Create comprehensive reasoning log for transparency."""
        log_parts = [
            f"# SELF-DECIDE Reasoning Log",
            f"**Session ID:** {session.session_id}",
            f"**Query:** {session.original_query}",
            f"**Duration:** {session.total_duration_ms}ms",
            f"**Confidence:** {session.confidence_score:.2f}",
            f""
        ]

        # Add each reasoning step
        for i, step in enumerate(session.reasoning_steps, 1):
            step_name = step.step.value.replace('_', ' ').title()
            log_parts.extend([
                f"## Step {i}: {step_name}",
                f"**Duration:** {step.duration_ms}ms",
                f"**Confidence:** {step.confidence:.2f}",
                f"**Reasoning:** {step.reasoning}",
                f""
            ])

            # Add key outputs
            if step.output_data:
                key_outputs = []
                for key, value in step.output_data.items():
                    if isinstance(value, (str, int, float)):
                        key_outputs.append(f"- {key}: {value}")
                    elif isinstance(value, list):
                        key_outputs.append(f"- {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        key_outputs.append(f"- {key}: {len(value)} entries")

                if key_outputs:
                    log_parts.append("**Key Outputs:**")
                    log_parts.extend(key_outputs)
                    log_parts.append("")

        # Add tool executions summary
        if session.tool_executions:
            log_parts.extend([
                f"## Tool Executions",
                f"**Total Tools Used:** {len(session.tool_executions)}",
                f""
            ])

            for tool_exec in session.tool_executions:
                tool_name = tool_exec.get('tool_name', 'unknown')
                success = tool_exec.get('success', False)
                status = "✅ Success" if success else "❌ Failed"
                log_parts.append(f"- **{tool_name}:** {status}")

            log_parts.append("")

        return "\n".join(log_parts)

    def _evaluate_response_quality(self, session: SelfDecideSession) -> Dict[str, Any]:
        """Evaluate the quality of the final response."""
        assessment = {
            'overall_score': 0.0,
            'completeness': 0.0,
            'accuracy': 0.0,
            'clarity': 0.0,
            'source_usage': 0.0
        }

        final_answer = session.final_answer

        # Evaluate completeness (based on answer length and structure)
        if len(final_answer) > 100:
            assessment['completeness'] = 0.8
        elif len(final_answer) > 50:
            assessment['completeness'] = 0.6
        else:
            assessment['completeness'] = 0.3

        # Evaluate clarity (based on structure and formatting)
        clarity_indicators = ['**', '##', '-', '1.', '2.', '\n\n']
        clarity_score = sum(1 for indicator in clarity_indicators if indicator in final_answer)
        assessment['clarity'] = min(clarity_score * 0.15, 1.0)

        # Evaluate source usage
        if session.tool_executions:
            assessment['source_usage'] += 0.4

        retrieval_step = next((s for s in session.reasoning_steps if s.step == ReasoningStep.EXPLORE_RETRIEVALS), None)
        if retrieval_step and retrieval_step.output_data.get('retrievals'):
            assessment['source_usage'] += 0.4

        # Evaluate accuracy (based on confidence and successful tool executions)
        assessment['accuracy'] = session.confidence_score

        # Calculate overall score
        weights = {'completeness': 0.3, 'accuracy': 0.3, 'clarity': 0.2, 'source_usage': 0.2}
        assessment['overall_score'] = sum(assessment[key] * weight for key, weight in weights.items())

        return assessment

    def get_session(self, session_id: str) -> Optional[SelfDecideSession]:
        """Get a reasoning session by ID."""
        return self.active_sessions.get(session_id)

    def get_reasoning_log(self, session_id: str) -> Optional[str]:
        """Get the reasoning log for a session."""
        session = self.get_session(session_id)
        if session:
            return self._create_reasoning_log(session)
        return None

# Global framework instance
_self_decide_framework = None

def get_self_decide_framework() -> SelfDecideFramework:
    """Get or create a global SELF-DECIDE framework instance."""
    global _self_decide_framework

    if _self_decide_framework is None:
        _self_decide_framework = SelfDecideFramework()

    return _self_decide_framework