"""
SELF-REFLECT Prompt Templates for Autonomous Factual Rectification
================================================================

High-fidelity prompts for the SELF-REFLECT methodology:
- Critique: Meticulous fact-checking of generated responses
- Revise: Response refinement incorporating factual corrections

These prompts are used by the enhanced AutonomousFactualCorrectionSkill
to implement the "Generate, Critique, Revise" loop.

Author: SAM Development Team
Version: 1.0.0
"""

# Critique prompt template for factual error detection
CRITIQUE_PROMPT_TEMPLATE = """You are a meticulous, impartial Fact-Checker AI. Your sole task is to review a generated response against the original query for factual inaccuracies.

INSTRUCTIONS:
- Identify every specific claim in the response that is verifiably false or unsubstantiated
- Do NOT comment on style, tone, or grammar. Focus exclusively on factual correctness
- If no errors are found, respond with "No factual errors detected."
- If errors are found, provide a bulleted list of "Revision Notes". Each note must be a specific, actionable correction

Original Query: "{original_query}"
Response to Review: "{initial_response}"

Your Factual Critique:"""

# Revise prompt template for response refinement
REVISE_PROMPT_TEMPLATE = """You are a Response Refinement AI. Your task is to rewrite an initial response to incorporate a set of revision notes, ensuring the final answer is factually accurate.

INSTRUCTIONS:
- The final response must address the original query directly
- Seamlessly integrate all corrections from the revision notes
- Preserve the original tone and intent of the response where possible, unless the revision notes state otherwise
- Maintain the same level of detail and helpfulness as the original response

Original Query: "{original_query}"
Initial Response: "{initial_response}"
Revision Notes: "{revision_notes}"

Your Final, Corrected Response:"""

# Confidence analysis prompt for determining if self-reflection is needed
CONFIDENCE_ANALYSIS_PROMPT = """Analyze the confidence level of this response for factual accuracy.

Response: "{response_text}"
Query: "{original_query}"

Rate the factual confidence on a scale of 0.0 to 1.0 where:
- 1.0 = Completely confident, all facts are verifiable and accurate
- 0.8 = High confidence, minor uncertainty in some details
- 0.6 = Moderate confidence, some claims may need verification
- 0.4 = Low confidence, several questionable claims
- 0.2 = Very low confidence, many likely errors
- 0.0 = No confidence, response likely contains significant errors

Provide only the numerical score (e.g., 0.7):"""

# Error severity assessment prompt
ERROR_SEVERITY_PROMPT = """Assess the severity of this potential factual error:

Error Type: {error_type}
Error Text: "{error_text}"
Context: "{context}"

Rate severity as:
- HIGH: Critical factual error that significantly misleads
- MEDIUM: Notable error that could cause confusion
- LOW: Minor inaccuracy with limited impact

Provide only the severity level:"""
