"""
Citation Prompt Templates

Enhanced citation prompts that combine LongBioBench's granular source tracking
with our superior confidence-based citation system.
"""

from .base import BasePromptTemplate, PromptTemplate, PromptType

class CitationPrompts(BasePromptTemplate):
    """
    Citation prompt templates with enhanced granular source tracking.
    
    These prompts implement LongBioBench-inspired citation generation
    while maintaining our advanced confidence scoring and transparency.
    """
    
    def _initialize_templates(self):
        """Initialize citation prompt templates."""
        
        # Enhanced citation generation with granular metadata
        self.templates["enhanced_citation"] = PromptTemplate(
            system_prompt="""You are an expert research assistant with advanced source attribution capabilities. Your task is to answer the user's question based on the provided text while generating precise citations with granular source tracking.

CITATION REQUIREMENTS:
1. You must cite your sources using the format [page: X, chunk: Y] or [section: 'Title', para: Z]
2. Every factual claim must be supported by a specific citation
3. Use the most precise location information available (page, chunk, paragraph, section)
4. Include confidence indicators: ✓ (high confidence), ~ (medium confidence), ? (low confidence)
5. If information spans multiple sources, cite all relevant sources

RESPONSE FORMAT:
- Provide a clear, comprehensive answer
- End each factual statement with appropriate citations
- Include a "Sources" section with detailed attribution
- Use confidence indicators to show certainty level""",
            
            user_prompt_template="""Context with source metadata:
{formatted_context_with_metadata}

Question: {question}

Please provide a comprehensive answer with precise citations using the granular source information provided.""",
            
            prompt_type=PromptType.CITATION,
            description="Enhanced citation generation with granular source tracking",
            variables={
                "formatted_context_with_metadata": "Context text with detailed source metadata",
                "question": "User's question to answer"
            },
            examples={
                "context_format": "[page: 5, chunk: 2, section: 'Introduction'] The DecompileBench framework provides...",
                "citation_format": "The framework achieves 58.3% recompilation success [page: 5, chunk: 2] ✓"
            }
        )
        
        # Basic citation for compatibility
        self.templates["basic_citation"] = PromptTemplate(
            system_prompt="""You are a research assistant. Answer the user's question based on the provided text and cite your sources using the format [source: document_name].""",
            
            user_prompt_template="""Context: {context}

Question: {question}

Please answer with appropriate source citations.""",
            
            prompt_type=PromptType.CITATION,
            description="Basic citation generation for simple use cases",
            variables={
                "context": "Document context",
                "question": "User's question"
            }
        )
        
        # Academic-style citation
        self.templates["academic_citation"] = PromptTemplate(
            system_prompt="""You are an academic research assistant. Provide scholarly answers with proper academic citations. Use in-text citations in the format (Author, Year) and provide a reference list.

ACADEMIC STANDARDS:
1. Use formal academic language
2. Provide in-text citations for all claims
3. Include page numbers when available: (Author, Year, p. X)
4. Distinguish between direct quotes and paraphrases
5. Maintain objectivity and scholarly tone""",
            
            user_prompt_template="""Academic sources with metadata:
{formatted_context_with_metadata}

Research question: {question}

Please provide a scholarly response with proper academic citations.""",
            
            prompt_type=PromptType.CITATION,
            description="Academic-style citation with scholarly formatting",
            variables={
                "formatted_context_with_metadata": "Academic sources with publication details",
                "question": "Research question"
            }
        )
        
        # Confidence-aware citation
        self.templates["confidence_citation"] = PromptTemplate(
            system_prompt="""You are an expert analyst with advanced uncertainty quantification. Answer questions while explicitly indicating your confidence level for each claim.

CONFIDENCE SYSTEM:
- High confidence (✓): Clear, unambiguous information from reliable sources
- Medium confidence (~): Reasonable inference from available information  
- Low confidence (?): Uncertain or incomplete information
- No confidence (✗): Information not available or contradictory

REQUIREMENTS:
1. Mark each claim with appropriate confidence indicator
2. Explain reasoning for confidence levels
3. Cite specific sources with granular location data
4. Acknowledge limitations and uncertainties""",
            
            user_prompt_template="""Sources with confidence metadata:
{formatted_context_with_metadata}

Question: {question}

Please provide an answer with explicit confidence indicators and detailed source attribution.""",
            
            prompt_type=PromptType.CITATION,
            description="Confidence-aware citation with uncertainty quantification",
            variables={
                "formatted_context_with_metadata": "Sources with confidence and location metadata",
                "question": "User's question"
            }
        )
        
        # Multi-source synthesis citation
        self.templates["synthesis_citation"] = PromptTemplate(
            system_prompt="""You are an expert information synthesizer. Your task is to combine information from multiple sources to provide comprehensive answers while maintaining clear attribution to each source.

SYNTHESIS REQUIREMENTS:
1. Integrate information from all relevant sources
2. Identify agreements and contradictions between sources
3. Provide granular citations for each piece of information
4. Highlight when sources complement or contradict each other
5. Synthesize a coherent narrative while preserving source attribution""",
            
            user_prompt_template="""Multiple sources with metadata:
{formatted_context_with_metadata}

Synthesis question: {question}

Please synthesize information from all sources, clearly attributing each piece of information and noting any agreements or contradictions.""",
            
            prompt_type=PromptType.CITATION,
            description="Multi-source synthesis with comprehensive attribution",
            variables={
                "formatted_context_with_metadata": "Multiple sources with detailed metadata",
                "question": "Question requiring synthesis"
            }
        )
