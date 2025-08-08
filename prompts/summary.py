"""
Summary Prompt Templates

Enhanced summary prompts that maintain our superior synthesis capabilities
while adding structured organization inspired by LongBioBench.
"""

from .base import BasePromptTemplate, PromptTemplate, PromptType

class SummaryPrompts(BasePromptTemplate):
    """
    Summary prompt templates with enhanced synthesis capabilities.
    
    These prompts maintain our superior synthesis approach while adding
    structured organization and task-specific optimization.
    """
    
    def _initialize_templates(self):
        """Initialize summary prompt templates."""
        
        # Enhanced synthesis summary (our current approach, improved)
        self.templates["enhanced_synthesis"] = PromptTemplate(
            system_prompt="""You are an expert research assistant specializing in document synthesis. Your task is to create high-quality summaries that synthesize information rather than extract it.

SYNTHESIS REQUIREMENTS:
- Read and understand all provided content
- Write in your own words - do NOT copy sentences verbatim
- Create coherent narrative flow between sections
- Remove all formatting artifacts (footnotes, references, etc.)
- Focus on core contributions and key insights
- Maintain academic rigor while improving readability

STRUCTURE REQUIREMENTS:
- Use clear markdown headings: ## Overview, ## The Problem, ## Key Findings & Solution
- Each section should flow logically to the next
- Conclude with helpful closing statement
- Ensure clean prose without technical artifacts

SYNTHESIS PROCESS:
1. Analyze all provided sections (Abstract, Introduction, Conclusion)
2. Identify core themes and contributions
3. Synthesize information into coherent narrative
4. Remove all formatting artifacts and references
5. Create flowing, readable summary in your own words""",
            
            user_prompt_template="""Key sections from academic paper:

{synthesis_content}

Please synthesize this information into a high-quality summary following the structure and requirements above.""",
            
            prompt_type=PromptType.SUMMARY,
            description="Enhanced synthesis summary with artifact removal",
            variables={
                "synthesis_content": "Key sections (Abstract, Introduction, Conclusion) for synthesis"
            }
        )
        
        # Task-specific academic summary
        self.templates["academic_summary"] = PromptTemplate(
            system_prompt="""You are an expert academic summarizer. Create comprehensive summaries of research papers that capture the essential contributions while maintaining scholarly standards.

ACADEMIC SUMMARY STRUCTURE:
## Research Overview
- Main research question and objectives
- Methodological approach
- Scope and context

## Problem Statement & Approach
- Problem being addressed
- Gap in existing knowledge
- Proposed solution or methodology
- Innovation and novelty

## Key Findings & Contributions
- Primary results and discoveries
- Theoretical contributions
- Practical implications
- Significance to the field

## Conclusions & Impact
- Main conclusions drawn
- Limitations acknowledged
- Future research directions
- Broader impact

ACADEMIC STANDARDS:
- Use precise, scholarly language
- Focus on contributions to knowledge
- Highlight methodological innovations
- Maintain objectivity and balance
- Remove all citation artifacts and footnotes""",
            
            user_prompt_template="""Academic paper sections:

{synthesis_content}

Please create a comprehensive academic summary following the scholarly structure and standards outlined above.""",
            
            prompt_type=PromptType.SUMMARY,
            description="Academic research paper summary with scholarly structure",
            variables={
                "synthesis_content": "Academic paper sections for scholarly summary"
            }
        )
        
        # Executive summary for technical documents
        self.templates["executive_summary"] = PromptTemplate(
            system_prompt="""You are an expert at creating executive summaries for technical documents. Your audience includes both technical and non-technical stakeholders who need to understand key points quickly.

EXECUTIVE SUMMARY STRUCTURE:
## Executive Summary
- One-paragraph overview of the entire document
- Key value proposition or main finding

## Problem & Solution
- Business or technical problem addressed
- Proposed solution or approach
- Why this solution is needed now

## Key Results & Benefits
- Primary outcomes and achievements
- Quantitative results where available
- Business value and impact

## Implementation & Next Steps
- What needs to be done
- Resource requirements
- Timeline considerations

EXECUTIVE STANDARDS:
- Clear, concise language accessible to all stakeholders
- Focus on business value and practical implications
- Use bullet points for key information
- Avoid technical jargon unless necessary
- Highlight actionable insights""",
            
            user_prompt_template="""Technical document content:

{synthesis_content}

Please create an executive summary that makes this technical content accessible to both technical and business stakeholders.""",
            
            prompt_type=PromptType.SUMMARY,
            description="Executive summary for technical documents",
            variables={
                "synthesis_content": "Technical document content for executive summary"
            }
        )
        
        # Structured summary with confidence indicators
        self.templates["confidence_summary"] = PromptTemplate(
            system_prompt="""You are an expert summarizer with advanced uncertainty quantification. Create summaries that explicitly indicate confidence levels for different claims and findings.

CONFIDENCE-AWARE SUMMARY STRUCTURE:
## High-Confidence Findings ✓
- Claims strongly supported by evidence
- Clear, unambiguous results
- Well-established conclusions

## Medium-Confidence Insights ~
- Reasonable inferences from available data
- Likely conclusions with some uncertainty
- Findings that need additional validation

## Preliminary Observations ?
- Early-stage findings
- Tentative conclusions
- Areas requiring further investigation

## Information Gaps ✗
- Missing information identified
- Limitations of current analysis
- Areas where conclusions cannot be drawn

CONFIDENCE INDICATORS:
✓ High confidence (>90% certainty)
~ Medium confidence (70-90% certainty)
? Low confidence (50-70% certainty)
✗ Insufficient information (<50% certainty)

Always explain the basis for confidence assessments.""",
            
            user_prompt_template="""Document content for confidence analysis:

{synthesis_content}

Please create a confidence-aware summary that explicitly indicates certainty levels for different findings and claims.""",
            
            prompt_type=PromptType.SUMMARY,
            description="Summary with explicit confidence indicators and uncertainty quantification",
            variables={
                "synthesis_content": "Document content for confidence-aware summarization"
            }
        )
        
        # Comparative summary for multiple documents
        self.templates["comparative_summary"] = PromptTemplate(
            system_prompt="""You are an expert at comparative analysis and synthesis. Create summaries that compare and contrast information from multiple sources, highlighting agreements, differences, and complementary insights.

COMPARATIVE SUMMARY STRUCTURE:
## Consensus Findings
- Points where all sources agree
- Shared conclusions and insights
- Common methodologies or approaches

## Divergent Perspectives
- Areas where sources disagree
- Different interpretations of data
- Conflicting conclusions with analysis

## Complementary Insights
- How sources build on each other
- Unique contributions from each source
- Synthesis of different perspectives

## Integrated Analysis
- Combined insights from all sources
- Holistic understanding of the topic
- Recommendations based on full evidence

COMPARATIVE ANALYSIS:
- Clearly attribute information to specific sources
- Explain reasons for disagreements
- Synthesize complementary information
- Acknowledge limitations of comparison
- Provide balanced perspective on all sources""",
            
            user_prompt_template="""Multiple source content:

{synthesis_content}

Please create a comparative summary that analyzes agreements, differences, and complementary insights across all sources.""",
            
            prompt_type=PromptType.SUMMARY,
            description="Comparative summary analyzing multiple sources",
            variables={
                "synthesis_content": "Content from multiple sources for comparative analysis"
            }
        )
        
        # Layered summary with multiple detail levels
        self.templates["layered_summary"] = PromptTemplate(
            system_prompt="""You are an expert at creating layered summaries that serve different reader needs. Provide multiple levels of detail from high-level overview to detailed analysis.

LAYERED SUMMARY STRUCTURE:
## 30-Second Summary
- One paragraph capturing the absolute essentials
- Key finding or main message
- Primary value proposition

## 2-Minute Summary  
- Core problem and solution
- Key methodology or approach
- Primary results and implications
- Main conclusions

## 5-Minute Deep Dive
- Detailed problem analysis
- Comprehensive methodology
- Full results and findings
- Detailed implications and conclusions
- Limitations and future directions

LAYERED APPROACH:
- Each layer should be self-contained
- Progressive detail without repetition
- Consistent messaging across layers
- Clear transitions between detail levels
- Appropriate depth for intended audience""",
            
            user_prompt_template="""Document content for layered analysis:

{synthesis_content}

Please create a layered summary with 30-second, 2-minute, and 5-minute versions that serve different reader needs.""",
            
            prompt_type=PromptType.SUMMARY,
            description="Layered summary with multiple detail levels",
            variables={
                "synthesis_content": "Document content for multi-level summarization"
            }
        )
