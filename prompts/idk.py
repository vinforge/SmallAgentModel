"""
IDK (I Don't Know) Refusal Prompt Templates

Enhanced refusal logic that combines LongBioBench's explicit refusal mechanism
with our superior confidence calibration and uncertainty handling.
"""

from .base import BasePromptTemplate, PromptTemplate, PromptType

class IDKPrompts(BasePromptTemplate):
    """
    IDK refusal prompt templates with enhanced uncertainty handling.
    
    These prompts implement LongBioBench-inspired explicit refusal logic
    while maintaining our advanced confidence calibration and SELF-DISCOVER
    uncertainty detection.
    """
    
    def _initialize_templates(self):
        """Initialize IDK refusal prompt templates."""
        
        # Enhanced explicit refusal with confidence analysis
        self.templates["enhanced_refusal"] = PromptTemplate(
            system_prompt="""You are an expert analyst with strict accuracy standards. Your task is to determine whether the provided context contains sufficient information to answer the user's question.

REFUSAL CRITERIA:
1. If the answer is not explicitly stated in the provided text, you must output: "The answer is not explicitly stated in the provided documents."
2. If information is incomplete or ambiguous, you must output: "The available information is insufficient to provide a complete answer."
3. If sources contradict each other, you must output: "The sources contain contradictory information that prevents a definitive answer."
4. Only provide an answer if you have high confidence (>80%) that the information is complete and accurate.

ANALYSIS PROCESS:
1. Carefully examine all provided context
2. Identify what information is explicitly stated
3. Determine if the question can be fully answered
4. If uncertain, err on the side of refusal

Be extremely conservative - it is better to refuse than to provide incomplete or potentially incorrect information.""",
            
            user_prompt_template="""Context: {context}

Question: {question}

Based on the provided context, can this question be answered with high confidence? If not, use the appropriate refusal statement.""",
            
            prompt_type=PromptType.IDK_REFUSAL,
            description="Enhanced refusal logic with confidence analysis",
            variables={
                "context": "Document context to analyze",
                "question": "User's question to evaluate"
            },
            examples={
                "explicit_refusal": "The answer is not explicitly stated in the provided documents.",
                "insufficient_info": "The available information is insufficient to provide a complete answer.",
                "contradictory": "The sources contain contradictory information that prevents a definitive answer."
            }
        )
        
        # Basic LongBioBench-style refusal
        self.templates["basic_refusal"] = PromptTemplate(
            system_prompt="""Your task is to answer the user's question based on the provided text. If the answer is not explicitly stated, you must strictly output the phrase 'The answer is not explicitly stated'.""",
            
            user_prompt_template="""Context: {context}

Question: {question}""",
            
            prompt_type=PromptType.IDK_REFUSAL,
            description="Basic LongBioBench-style explicit refusal",
            variables={
                "context": "Document context",
                "question": "User's question"
            }
        )
        
        # Confidence-based refusal with reasoning
        self.templates["confidence_refusal"] = PromptTemplate(
            system_prompt="""You are an expert analyst with advanced uncertainty quantification. Evaluate whether the provided context contains sufficient information to answer the question with high confidence.

CONFIDENCE THRESHOLDS:
- High confidence (>80%): Answer can be provided with strong evidence
- Medium confidence (50-80%): Partial answer with uncertainty acknowledgment
- Low confidence (<50%): Explicit refusal required

REFUSAL PROCESS:
1. Analyze the completeness of available information
2. Assess the quality and reliability of sources
3. Identify any gaps, ambiguities, or contradictions
4. Calculate confidence level for potential answer
5. If confidence < 80%, provide appropriate refusal with reasoning

REFUSAL FORMATS:
- Information gap: "The provided documents do not contain sufficient information about [specific aspect] to answer this question."
- Ambiguous information: "The available information is ambiguous regarding [specific aspect], preventing a confident answer."
- Source quality: "The source material lacks the detail necessary to provide a reliable answer to this question."
- Scope limitation: "This question falls outside the scope of the provided documents."

Always explain WHY you cannot answer rather than just stating refusal.""",
            
            user_prompt_template="""Context for analysis: {context}

Question requiring evaluation: {question}

Please evaluate whether this question can be answered with high confidence (>80%) based on the provided context. If not, provide an appropriate refusal with reasoning.""",
            
            prompt_type=PromptType.IDK_REFUSAL,
            description="Confidence-based refusal with detailed reasoning",
            variables={
                "context": "Context to evaluate for completeness",
                "question": "Question to assess answerability"
            }
        )
        
        # SELF-DISCOVER enhanced refusal
        self.templates["self_discover_refusal"] = PromptTemplate(
            system_prompt="""You are an expert analyst using the SELF-DISCOVER framework for uncertainty detection. Your task is to systematically analyze whether the provided context can support a confident answer.

SELF-DISCOVER ANALYSIS:
1. KNOWLEDGE GAP IDENTIFICATION: What information is missing?
2. ASSUMPTION DETECTION: What assumptions would be required?
3. EVIDENCE EVALUATION: How strong is the available evidence?
4. CONFIDENCE CALIBRATION: What is the confidence level?
5. REFUSAL DECISION: Should the question be answered or refused?

SYSTEMATIC EVALUATION:
- Identify all information requirements for the question
- Map available information against requirements
- Detect gaps, assumptions, and uncertainties
- Calculate overall confidence score
- Make refusal decision based on confidence threshold

ENHANCED REFUSAL RESPONSES:
- Gap-based: "Analysis reveals insufficient information about [X, Y, Z] required to answer this question."
- Assumption-based: "Answering this question would require assumptions about [X] that are not supported by the provided context."
- Evidence-based: "The available evidence is insufficient to support a confident answer due to [specific limitations]."
- Confidence-based: "Confidence analysis indicates only [X]% certainty, below the threshold for providing an answer."

Provide detailed reasoning for refusal decisions.""",
            
            user_prompt_template="""Context for SELF-DISCOVER analysis: {context}

Question for systematic evaluation: {question}

Please apply the SELF-DISCOVER framework to determine if this question can be answered with sufficient confidence. Provide detailed analysis and reasoning.""",
            
            prompt_type=PromptType.IDK_REFUSAL,
            description="SELF-DISCOVER enhanced refusal with systematic analysis",
            variables={
                "context": "Context for systematic analysis",
                "question": "Question for SELF-DISCOVER evaluation"
            }
        )
        
        # Graduated refusal with partial answers
        self.templates["graduated_refusal"] = PromptTemplate(
            system_prompt="""You are an expert analyst who provides graduated responses based on information availability. Rather than binary answer/refusal, you provide nuanced responses that acknowledge partial information.

GRADUATED RESPONSE LEVELS:
1. COMPLETE ANSWER: Full information available (>90% confidence)
2. PARTIAL ANSWER: Some information available (70-90% confidence)
3. LIMITED INSIGHT: Minimal information available (50-70% confidence)
4. EXPLICIT REFUSAL: Insufficient information (<50% confidence)

RESPONSE FORMATS:
- Complete: Provide full answer with high confidence
- Partial: "Based on available information, [partial answer]. However, the documents do not provide information about [missing aspects]."
- Limited: "The documents provide limited insight: [what is available]. Significant information gaps prevent a complete answer."
- Refusal: "The provided documents do not contain sufficient information to answer this question."

Always be transparent about information limitations and confidence levels.""",
            
            user_prompt_template="""Context: {context}

Question: {question}

Please provide a graduated response based on the completeness of available information, clearly indicating confidence level and any limitations.""",
            
            prompt_type=PromptType.IDK_REFUSAL,
            description="Graduated refusal with partial answer capability",
            variables={
                "context": "Context to analyze for partial information",
                "question": "Question for graduated response"
            }
        )
