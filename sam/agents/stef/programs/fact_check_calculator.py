#!/usr/bin/env python3
"""
Fact Check and Calculate Program
A STEF program that verifies facts through web search and performs calculations.
"""

from ..task_definitions import TaskProgram, TaskStep

def validate_numeric_result(result: str) -> bool:
    """Validation function to ensure we got a numeric result."""
    try:
        # Try to extract a number from the result
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', str(result))
        return len(numbers) > 0
    except:
        return False

def validate_search_result(result: str) -> bool:
    """Validation function to ensure search returned meaningful content."""
    if not result or len(str(result).strip()) < 10:
        return False
    # Check if result contains common "no results" indicators
    no_results_indicators = ['no results', 'not found', 'error', 'failed']
    result_lower = str(result).lower()
    return not any(indicator in result_lower for indicator in no_results_indicators)

# Define the FactCheckAndCalculate program
fact_check_and_calculate_program = TaskProgram(
    program_name="FactCheckAndCalculate",
    description="Verifies key numbers from a query using web search and then performs calculations with verified data.",
    
    # Trigger keywords for Phase 1 simple matching
    trigger_keywords=[
        "calculate", "density", "what is the ratio", "how many per",
        "population density", "ratio of", "percentage of population",
        "calculate the", "what's the density", "density of",
        "per capita", "per square", "ratio between"
    ],
    
    steps=[
        TaskStep(
            step_name="Extract_Primary_Subject",
            tool_name="web_search",
            input_template="Find current data for the main subject mentioned in this query: {initial_query}. Focus on getting specific numerical values.",
            output_key="primary_data",
            description="Search for data about the primary subject of the calculation",
            validation_function=validate_search_result,
            retry_on_failure=True,
            max_retries=2,
            alternative_tools=["document_search"],  # Fallback to document search if web search fails
            required=True
        ),
        
        TaskStep(
            step_name="Extract_Secondary_Subject", 
            tool_name="web_search",
            input_template="Find current data for the secondary subject mentioned in this query: {initial_query}. Focus on getting specific numerical values that can be used for comparison or calculation.",
            output_key="secondary_data",
            description="Search for data about the secondary subject of the calculation",
            validation_function=validate_search_result,
            retry_on_failure=True,
            max_retries=2,
            alternative_tools=["document_search"],
            required=True
        ),
        
        TaskStep(
            step_name="Extract_Numbers",
            tool_name="calculator",
            input_template="Extract the key numbers from this data for calculation: Primary data: {primary_data}. Secondary data: {secondary_data}. Original query: {initial_query}",
            output_key="extracted_numbers",
            description="Extract and identify the specific numbers needed for calculation",
            validation_function=validate_numeric_result,
            retry_on_failure=True,
            max_retries=1,
            required=True
        ),
        
        TaskStep(
            step_name="Perform_Calculation",
            tool_name="calculator", 
            input_template="Based on the original query '{initial_query}' and the extracted data, perform the requested calculation using: {extracted_numbers}",
            output_key="calculation_result",
            description="Perform the final calculation with verified data",
            validation_function=validate_numeric_result,
            retry_on_failure=True,
            max_retries=2,
            required=True
        )
    ],
    
    synthesis_prompt_template="""Generate a JSON response for this fact-checking and calculation process:

Original Question: {initial_query}
Research Results:
- Primary Subject Data: {primary_data}
- Secondary Subject Data: {secondary_data}
- Extracted Numbers: {extracted_numbers}
- Calculation Result: {calculation_result}

Provide your response as a JSON object with this exact structure:
{{
  "direct_answer": "[Clear, direct answer to the question]",
  "query_type": "complex_calculation",
  "presentation_type": "detailed",
  "confidence": 0.9,
  "supporting_evidence": ["web_search", "calculator"],
  "reasoning_summary": "Verified facts through web search and performed calculation",
  "detailed_reasoning": "1. Searched for {initial_query} data\\n2. Found: {primary_data}\\n3. Found: {secondary_data}\\n4. Extracted numbers: {extracted_numbers}\\n5. Calculated: {calculation_result}",
  "sources": [
    {{"source_id": "web_search_1", "source_type": "web", "source_name": "Web search results", "relevance_score": 0.8}}
  ]
}}

JSON Response:""",
    
    # Phase 2 enhancements (not used in Phase 1)
    required_confidence=0.7,
    compatible_intents=["calculation", "hybrid_doc_calc", "knowledge_search"],
    allow_partial_execution=False,  # All steps required for accurate fact-checking
    
    error_message_template="I encountered an error while fact-checking and calculating your request: {error_details}. This type of query requires verifying information from multiple sources, and one of the verification steps failed. Please try rephrasing your question or providing more specific details."
)

# Enhanced SimpleCalculation program optimized for instant pure math processing
simple_calculation_program = TaskProgram(
    program_name="SimpleCalculation",
    description="Performs instant mathematical calculations with optimized processing for pure math expressions.",

    trigger_keywords=[
        "what's", "calculate", "compute", "solve", "what is",
        "+", "-", "*", "/", "%", "percent of", "percentage",
        # Ultra-aggressive triggers for pure math
        "99+1", "5*8", "100/4", "50-25"  # Example patterns
    ],

    steps=[
        TaskStep(
            step_name="Instant_Calculation",
            tool_name="calculator",
            input_template="{initial_query}",  # Direct input for pure math
            output_key="calculation_result",
            description="Instantly calculate the mathematical expression",
            validation_function=validate_numeric_result,
            retry_on_failure=True,
            max_retries=1,  # Reduced retries for speed
            required=True
        )
    ],

    synthesis_prompt_template="""Generate a JSON response for this mathematical calculation:

Query: {initial_query}
Calculation Result: {calculation_result}

Provide your response as a JSON object with this exact structure:
{{
  "direct_answer": "{calculation_result}",
  "query_type": "pure_math",
  "presentation_type": "minimal",
  "confidence": 1.0,
  "supporting_evidence": ["calculator"],
  "reasoning_summary": "Direct mathematical calculation",
  "detailed_reasoning": "Calculated: {initial_query} = {calculation_result}"
}}

JSON Response:""",

    required_confidence=0.8,
    compatible_intents=["calculation"],
    allow_partial_execution=False,

    error_message_template="Calculation error: {error_details}"
)

# Document analysis with calculation program
document_calculation_program = TaskProgram(
    program_name="DocumentCalculation", 
    description="Extracts numerical data from documents and performs calculations.",
    
    trigger_keywords=[
        "calculate from document", "what's the", "percentage in document",
        "ratio in", "from my document", "in the uploaded file",
        "document shows", "according to document"
    ],
    
    steps=[
        TaskStep(
            step_name="Search_Document",
            tool_name="document_search",
            input_template="Search for numerical data related to this query in uploaded documents: {initial_query}",
            output_key="document_data",
            description="Search documents for relevant numerical information",
            validation_function=validate_search_result,
            retry_on_failure=True,
            max_retries=2,
            required=True
        ),
        
        TaskStep(
            step_name="Extract_Document_Numbers",
            tool_name="calculator",
            input_template="Extract the specific numbers needed for calculation from this document data: {document_data}. Original query: {initial_query}",
            output_key="extracted_numbers",
            description="Extract numerical values from document content",
            validation_function=validate_numeric_result,
            retry_on_failure=True,
            max_retries=1,
            required=True
        ),
        
        TaskStep(
            step_name="Calculate_From_Document",
            tool_name="calculator",
            input_template="Perform the calculation requested in '{initial_query}' using the numbers extracted from the document: {extracted_numbers}",
            output_key="calculation_result", 
            description="Perform calculation using document-extracted data",
            validation_function=validate_numeric_result,
            retry_on_failure=True,
            max_retries=2,
            required=True
        )
    ],
    
    synthesis_prompt_template="""Based on the analysis of your document and the calculation performed, here is the answer to your question.

Question: {initial_query}
Document Data Found: {document_data}
Numbers Extracted: {extracted_numbers}
Calculation Result: {calculation_result}

Please provide a comprehensive answer that:
1. References the specific document data used
2. Shows the calculation performed
3. Provides the final result with context

Answer:""",
    
    required_confidence=0.75,
    compatible_intents=["hybrid_doc_calc", "document_search"],
    allow_partial_execution=False,
    
    error_message_template="I couldn't complete the document-based calculation: {error_details}. Please ensure your document contains the necessary numerical data and try rephrasing your question."
)
