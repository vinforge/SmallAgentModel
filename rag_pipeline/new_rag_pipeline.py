"""
New RAG Pipeline - Clean Implementation
Phase 1: Perfect Summary Vertical Slice

This is a complete rewrite of the RAG pipeline, starting with summary functionality.
No legacy code, no patches, just clean, purpose-built logic.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import structured prompt system
try:
    from prompts.base import get_prompt_manager, PromptType
    STRUCTURED_PROMPTS_AVAILABLE = True
except ImportError:
    STRUCTURED_PROMPTS_AVAILABLE = False
    logging.warning("Structured prompts not available, using fallback prompts")

logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Represents a section of a document with its content and metadata."""
    title: str
    content: str
    section_type: str  # 'abstract', 'introduction', 'conclusion', 'body', 'other'
    confidence: float  # How confident we are this is the right section

@dataclass
class SummaryContext:
    """Clean context for summary generation."""
    abstract: Optional[str] = None
    introduction: Optional[str] = None
    conclusion: Optional[str] = None
    document_title: Optional[str] = None
    document_type: str = "document"

class NewRAGPipeline:
    """
    Clean RAG Pipeline Implementation
    
    Phase 1: Perfect Summary Generation
    - Task-specific retrieval for summaries
    - Clean content extraction
    - Metadata-free output
    """
    
    def __init__(self, memory_store=None):
        self.memory_store = memory_store
        logger.info("New RAG Pipeline initialized - Phase 1: Summary Focus")
    
    def generate_summary(self, document_name: str, user_query: str) -> str:
        """
        Generate a perfect summary using task-specific retrieval.
        
        This is the core of Phase 1 - a complete vertical slice that:
        1. Retrieves ONLY Abstract/Introduction/Conclusion
        2. Synthesizes (not extracts) content
        3. Returns metadata-free output
        """
        logger.info(f"NEW PIPELINE: Generating summary for {document_name}")
        
        # Step 1: Task-Specific Retrieval
        summary_context = self._retrieve_summary_sections(document_name)
        
        # Step 2: Explicit Refusal Check (LongBioBench-inspired)
        refusal_response = self._check_explicit_refusal(summary_context, user_query)
        if refusal_response:
            return refusal_response

        # Step 3: Validate we have meaningful content
        if not self._has_sufficient_content(summary_context):
            return self._generate_fallback_response(document_name)
        
        # Step 4: Enhanced Synthesis with Citation Integration
        raw_summary = self._synthesize_summary_with_citations(summary_context, document_name, user_query)

        # Step 5: Preserve Enhanced Features (NO aggressive stripping)
        final_summary = self._clean_summary_preserving_enhancements(raw_summary)

        logger.info(f"NEW PIPELINE: Summary generated successfully ({len(final_summary)} chars)")
        return final_summary
    
    def _retrieve_summary_sections(self, document_name: str) -> SummaryContext:
        """
        TASK-SPECIFIC RETRIEVAL: Get Abstract, Introduction, Conclusion ONLY.
        
        This is the core fix - we force-retrieve the right sections.
        """
        logger.info(f"NEW PIPELINE: Retrieving summary sections for {document_name}")
        
        if not self.memory_store:
            logger.warning("No memory store available")
            return SummaryContext()
        
        # Find all chunks from the target document
        target_chunks = self._find_document_chunks(document_name)
        logger.info(f"NEW PIPELINE: Found {len(target_chunks)} chunks for {document_name}")
        
        # Extract sections using multiple strategies
        sections = self._extract_key_sections(target_chunks)
        
        # Build clean summary context
        context = SummaryContext(
            abstract=sections.get('abstract'),
            introduction=sections.get('introduction'), 
            conclusion=sections.get('conclusion'),
            document_title=self._clean_document_name(document_name),
            document_type=self._detect_document_type(sections)
        )
        
        logger.info(f"NEW PIPELINE: Extracted sections - Abstract: {bool(context.abstract)}, "
                   f"Introduction: {bool(context.introduction)}, Conclusion: {bool(context.conclusion)}")
        
        return context
    
    def _find_document_chunks(self, document_name: str) -> List[any]:
        """Find all memory chunks belonging to the target document."""
        target_chunks = []
        
        # Create flexible search patterns
        search_patterns = self._create_search_patterns(document_name)
        
        for chunk_id, chunk in self.memory_store.memory_chunks.items():
            if chunk.memory_type.value == 'document' and chunk.content:
                source_str = str(chunk.source).lower() if chunk.source else ""
                
                # Check if this chunk belongs to our target document
                for pattern in search_patterns:
                    if pattern.lower() in source_str:
                        target_chunks.append(chunk)
                        logger.debug(f"NEW PIPELINE: Found chunk {chunk_id} for pattern '{pattern}'")
                        break
        
        return target_chunks
    
    def _create_search_patterns(self, document_name: str) -> List[str]:
        """Create flexible patterns to find document chunks."""
        patterns = []
        
        # Extract base name without extensions and timestamps
        clean_name = document_name
        clean_name = re.sub(r'^\d{8}_\d{6}_', '', clean_name)  # Remove timestamp
        clean_name = re.sub(r'\.(pdf|docx?|txt|md)$', '', clean_name, flags=re.IGNORECASE)
        
        patterns.extend([
            document_name,  # Exact match
            clean_name,     # Without timestamp/extension
            clean_name.replace('_', ' '),  # With spaces
            clean_name.replace('-', ' '),  # With spaces instead of dashes
        ])
        
        # Add partial matches for complex names
        if len(clean_name) > 10:
            patterns.append(clean_name[:10])  # First 10 chars
        
        return patterns
    
    def _extract_key_sections(self, chunks: List[any]) -> Dict[str, str]:
        """
        Extract Abstract, Introduction, and Conclusion from document chunks.
        
        This uses multiple strategies to find the right content.
        """
        sections = {}
        
        # Strategy 1: Look for explicit section headers
        sections.update(self._find_sections_by_headers(chunks))
        
        # Strategy 2: Look for content patterns (first substantial paragraph = intro, etc.)
        if not sections.get('introduction'):
            sections.update(self._find_sections_by_patterns(chunks))
        
        # Strategy 3: Use position-based heuristics
        if not sections.get('abstract'):
            sections.update(self._find_sections_by_position(chunks))
        
        return sections
    
    def _find_sections_by_headers(self, chunks: List[any]) -> Dict[str, str]:
        """Find sections by looking for explicit headers like 'Abstract', 'Introduction'."""
        sections = {}
        
        for chunk in chunks:
            content = chunk.content.strip()
            content_lower = content.lower()
            
            # Look for Abstract
            if 'abstract' in content_lower and not sections.get('abstract'):
                abstract_text = self._extract_section_content(content, 'abstract')
                if abstract_text and len(abstract_text) > 100:
                    sections['abstract'] = abstract_text
                    logger.info(f"NEW PIPELINE: Found Abstract section ({len(abstract_text)} chars)")
            
            # Look for Introduction
            if any(term in content_lower for term in ['introduction', '1. introduction', '1 introduction']) and not sections.get('introduction'):
                intro_text = self._extract_section_content(content, 'introduction')
                if intro_text and len(intro_text) > 200:
                    sections['introduction'] = intro_text
                    logger.info(f"NEW PIPELINE: Found Introduction section ({len(intro_text)} chars)")
            
            # Look for Conclusion
            if any(term in content_lower for term in ['conclusion', 'conclusions', 'concluding remarks']) and not sections.get('conclusion'):
                conclusion_text = self._extract_section_content(content, 'conclusion')
                if conclusion_text and len(conclusion_text) > 100:
                    sections['conclusion'] = conclusion_text
                    logger.info(f"NEW PIPELINE: Found Conclusion section ({len(conclusion_text)} chars)")
        
        return sections
    
    def _extract_section_content(self, content: str, section_type: str) -> Optional[str]:
        """Extract the actual content of a section, removing headers and metadata."""
        lines = content.split('\n')
        section_lines = []
        in_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Start capturing when we hit the section header
            if section_type.lower() in line_lower and len(line.strip()) < 50:  # Likely a header
                in_section = True
                continue
            
            # Stop capturing when we hit another section header
            if in_section and any(term in line_lower for term in ['references', 'bibliography', 'acknowledgments', 'appendix']):
                break
            
            # Stop capturing when we hit a numbered section (like "2. Methodology")
            if in_section and re.match(r'^\d+\.?\s+[A-Z]', line.strip()):
                break
            
            # Capture content lines
            if in_section and line.strip():
                # Skip metadata lines
                if not any(term in line_lower for term in ['block_', 'document:', 'content type:', 'source:']):
                    section_lines.append(line.strip())
        
        if section_lines:
            return '\n'.join(section_lines)
        
        return None

    def _find_sections_by_patterns(self, chunks: List[any]) -> Dict[str, str]:
        """Find sections using content patterns when headers aren't clear."""
        sections = {}

        # Sort chunks by source to get document order
        sorted_chunks = sorted(chunks, key=lambda x: str(x.source) if x.source else "")

        # Look for introduction patterns in early chunks
        for i, chunk in enumerate(sorted_chunks[:3]):  # First 3 chunks
            content = chunk.content.strip()
            if len(content) > 300 and not sections.get('introduction'):
                # Check if this looks like an introduction
                if any(term in content.lower() for term in ['this paper', 'we present', 'we propose', 'in this work']):
                    sections['introduction'] = self._clean_content(content)
                    logger.info(f"NEW PIPELINE: Found Introduction by pattern ({len(content)} chars)")
                    break

        return sections

    def _find_sections_by_position(self, chunks: List[any]) -> Dict[str, str]:
        """Find sections using position-based heuristics."""
        sections = {}

        if not chunks:
            return sections

        # Sort chunks by source to get document order
        sorted_chunks = sorted(chunks, key=lambda x: str(x.source) if x.source else "")

        # First substantial chunk might be abstract or introduction
        for chunk in sorted_chunks[:2]:
            content = self._clean_content(chunk.content)
            if len(content) > 200 and not sections.get('abstract'):
                sections['abstract'] = content[:800]  # First 800 chars as abstract
                logger.info(f"NEW PIPELINE: Using position-based abstract ({len(content)} chars)")
                break

        return sections

    def _clean_content(self, content: str) -> str:
        """Clean content by removing metadata and formatting artifacts."""
        if not content:
            return ""

        # Remove metadata lines
        lines = content.split('\n')
        clean_lines = []

        for line in lines:
            line = line.strip()
            # Skip metadata and system lines
            if not any(term in line.lower() for term in [
                'document:', 'block_', 'content type:', 'source:', 'web_ui/uploads/',
                'processing complete', 'metadata:', 'chunk_id:'
            ]):
                clean_lines.append(line)

        return '\n'.join(clean_lines).strip()

    def _clean_document_name(self, document_name: str) -> str:
        """Clean document name for display."""
        clean_name = document_name
        clean_name = re.sub(r'web_ui/uploads/', '', clean_name)
        clean_name = re.sub(r'^\d{8}_\d{6}_', '', clean_name)  # Remove timestamp
        return clean_name

    def _detect_document_type(self, sections: Dict[str, str]) -> str:
        """Detect document type based on content."""
        all_content = ' '.join(sections.values()).lower()

        if any(term in all_content for term in ['arxiv', 'doi', 'abstract', 'references']):
            return "research paper"
        elif any(term in all_content for term in ['whitepaper', 'technical report']):
            return "technical document"
        else:
            return "document"

    def _has_sufficient_content(self, context: SummaryContext) -> bool:
        """Check if we have enough content to generate a meaningful summary."""
        total_content = 0

        if context.abstract:
            total_content += len(context.abstract)
        if context.introduction:
            total_content += len(context.introduction)
        if context.conclusion:
            total_content += len(context.conclusion)

        has_content = total_content > 300  # Need at least 300 chars
        logger.info(f"NEW PIPELINE: Content check - {total_content} chars, sufficient: {has_content}")

        return has_content

    def _check_explicit_refusal(self, context: SummaryContext, user_query: str) -> Optional[str]:
        """
        Check if we should explicitly refuse to answer based on LongBioBench-inspired logic.

        This implements enhanced refusal logic that combines LongBioBench's explicit
        refusal mechanism with our superior confidence calibration.
        """
        logger.info("NEW PIPELINE: Checking explicit refusal conditions")

        # Build context for refusal analysis
        context_content = self._build_synthesis_content(context)

        # Use structured prompts if available
        if STRUCTURED_PROMPTS_AVAILABLE:
            try:
                prompt_manager = get_prompt_manager()
                system_prompt, user_prompt = prompt_manager.get_prompt(
                    PromptType.IDK_REFUSAL,
                    "enhanced_refusal",
                    context=context_content,
                    question=user_query
                )

                # Call LLM for refusal analysis
                refusal_response = self._call_llm_for_refusal_analysis(system_prompt, user_prompt)

                # Check for explicit refusal phrases
                if refusal_response and self._is_explicit_refusal(refusal_response):
                    logger.info("NEW PIPELINE: Explicit refusal triggered by enhanced analysis")
                    return self._format_refusal_response(refusal_response, context)

            except Exception as e:
                logger.warning(f"NEW PIPELINE: Structured refusal analysis failed: {e}")

        # Fallback to rule-based refusal analysis
        return self._rule_based_refusal_check(context, user_query)

    def _call_llm_for_refusal_analysis(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call LLM for refusal analysis using structured prompts."""
        try:
            import requests
            import json

            # Combine prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Use Ollama API for refusal analysis
            ollama_url = "http://localhost:11434/api/generate"

            payload = {
                "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Very low temperature for consistent refusal logic
                    "top_p": 0.8,
                    "max_tokens": 200
                }
            }

            response = requests.post(ollama_url, json=payload, timeout=15)

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.warning(f"NEW PIPELINE: Refusal analysis API error: {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"NEW PIPELINE: Refusal analysis failed: {e}")
            return None

    def _is_explicit_refusal(self, response: str) -> bool:
        """Check if response contains explicit refusal phrases."""
        refusal_phrases = [
            "the answer is not explicitly stated",
            "the available information is insufficient",
            "the sources contain contradictory information",
            "the provided documents do not contain sufficient information",
            "analysis reveals insufficient information",
            "confidence analysis indicates",
            "the documents provide limited insight"
        ]

        response_lower = response.lower()
        return any(phrase in response_lower for phrase in refusal_phrases)

    def _rule_based_refusal_check(self, context: SummaryContext, user_query: str) -> Optional[str]:
        """Rule-based refusal check as fallback."""
        # Check content quality and completeness
        total_content_length = 0
        sections_available = 0

        if context.abstract:
            total_content_length += len(context.abstract)
            sections_available += 1

        if context.introduction:
            total_content_length += len(context.introduction)
            sections_available += 1

        if context.conclusion:
            total_content_length += len(context.conclusion)
            sections_available += 1

        # Refusal conditions
        if total_content_length < 200:
            return "The available information is insufficient to provide a complete summary. The document content is too limited."

        if sections_available == 0:
            return "The answer is not explicitly stated in the provided documents. No key sections (Abstract, Introduction, or Conclusion) could be identified."

        # Check for summary-specific requirements
        if "summary" in user_query.lower() and sections_available < 2:
            return "The provided documents do not contain sufficient information to generate a comprehensive summary. Multiple key sections are required."

        # No refusal needed
        return None

    def _format_refusal_response(self, refusal_response: str, context: SummaryContext) -> str:
        """Format refusal response with helpful context."""
        formatted_response = f"""## Summary Request - Insufficient Information

{refusal_response}

**Available Information:**
- Document: {context.document_title or 'Unknown document'}
- Sections found: {', '.join([s for s in ['Abstract', 'Introduction', 'Conclusion'] if getattr(context, s.lower())])}

**Suggestions:**
• Try asking about specific aspects of the document that are available
• Upload additional documents if more comprehensive analysis is needed
• Ask more targeted questions about particular topics you know are in the document

I'm here to help with any specific questions about the available document content!"""

        return formatted_response

    def _synthesize_summary_with_citations(self, context: SummaryContext, document_name: str, user_query: str) -> str:
        """
        ENHANCED SYNTHESIS: Generate summary with integrated citation engine.

        This is the FIXED version that actually uses our enhanced features.
        """
        logger.info("NEW PIPELINE: Enhanced synthesis with citation integration")

        # Step 1: Get relevant memories for citation
        relevant_memories = self._get_memories_for_citation(context, document_name)

        # Step 2: Generate base summary using structured prompts
        base_summary = self._synthesize_summary(context, document_name, user_query)

        # Step 3: CRITICAL FIX - ALWAYS apply citation engine regardless of LLM success
        if relevant_memories:
            try:
                from memory.citation_engine import get_citation_engine
                citation_engine = get_citation_engine()

                logger.info(f"NEW PIPELINE: Applying citation engine to {len(relevant_memories)} memories")

                # If LLM synthesis failed, create a basic summary for citation enhancement
                if not base_summary or len(base_summary.strip()) < 100:
                    logger.info("NEW PIPELINE: LLM synthesis failed, creating fallback summary for citation")
                    base_summary = self._create_fallback_summary_for_citation(context, document_name)

                # Generate citations with enhanced metadata
                cited_response = citation_engine.inject_citations(base_summary, relevant_memories, user_query)

                logger.info(f"NEW PIPELINE: Generated {len(cited_response.citations)} enhanced citations")
                logger.info(f"NEW PIPELINE: Transparency score: {cited_response.transparency_score:.3f}")

                # Return the enhanced response with citations
                return cited_response.response_text

            except Exception as e:
                logger.warning(f"NEW PIPELINE: Citation engine failed: {e}, using base summary")
                return base_summary
        else:
            logger.info("NEW PIPELINE: No memories found for citation, using base summary")
            return base_summary

    def _get_memories_for_citation(self, context: SummaryContext, document_name: str) -> List:
        """
        Get relevant memories for citation generation with STRICT document filtering.

        CRITICAL FIX: This method now ensures ZERO data contamination by filtering
        memories to only include those from the specific target document.
        """
        if not self.memory_store:
            return []

        try:
            logger.info(f"NEW PIPELINE: Searching for memories from document: '{document_name}'")

            # CRITICAL FIX: Get ALL memories and filter by document source
            all_memories = self.memory_store.get_all_memories()
            document_specific_memories = []

            # Filter memories to only include those from the target document
            for memory in all_memories:
                memory_source = self._extract_memory_source(memory)

                # Check if this memory belongs to our target document
                if self._is_memory_from_document(memory_source, document_name):
                    document_specific_memories.append(memory)
                    logger.debug(f"NEW PIPELINE: Found matching memory from: {memory_source}")
                else:
                    logger.debug(f"NEW PIPELINE: Skipping memory from: {memory_source} (not {document_name})")

            logger.info(f"NEW PIPELINE: Found {len(document_specific_memories)} memories from target document")

            # If we have document-specific memories, return them
            if document_specific_memories:
                # Limit to reasonable number for citation
                return document_specific_memories[:10]

            # FALLBACK: If no document-specific memories found, try content-based search
            # but still filter results by document
            logger.warning(f"NEW PIPELINE: No direct document memories found, trying content search")

            search_memories = self.memory_store.search_memories(document_name, max_results=20)
            filtered_search_memories = []

            for memory in search_memories:
                memory_source = self._extract_memory_source(memory)
                if self._is_memory_from_document(memory_source, document_name):
                    filtered_search_memories.append(memory)

            logger.info(f"NEW PIPELINE: Found {len(filtered_search_memories)} filtered search memories")
            return filtered_search_memories[:10]

        except Exception as e:
            logger.warning(f"NEW PIPELINE: Error getting memories for citation: {e}")
            return []

    def _clean_summary_preserving_enhancements(self, content: str) -> str:
        """
        SELECTIVE CLEANING: Remove only harmful metadata while preserving enhancements.

        This replaces the aggressive _strip_all_metadata method.
        """
        if not content:
            return content

        # Remove ONLY harmful internal references (preserve citations and confidence indicators)
        cleaned = re.sub(r'web_ui/uploads/[^\s]+', '', content)
        cleaned = re.sub(r'mem_[a-f0-9]+', '', cleaned)
        cleaned = re.sub(r'Processing complete', '', cleaned)
        cleaned = re.sub(r'Processing Status:[^\n]+', '', cleaned)

        # Remove timestamps but preserve document references
        cleaned = re.sub(r'\d{8}_\d{6}_', '', cleaned)

        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()

        logger.info(f"NEW PIPELINE: Selective cleaning - preserved enhanced features")
        return cleaned

    def _create_fallback_summary_for_citation(self, context: SummaryContext, document_name: str) -> str:
        """
        Create a fallback summary when LLM synthesis fails, specifically for citation enhancement.

        This ensures we always have content to enhance with citations.
        """
        logger.info("NEW PIPELINE: Creating fallback summary for citation enhancement")

        # Build a structured summary from available sections
        summary_parts = [f"## Summary of {document_name}"]

        if context.abstract:
            summary_parts.append("## Overview")
            # Take first 500 chars of abstract
            abstract_preview = context.abstract[:500]
            if len(context.abstract) > 500:
                abstract_preview += "..."
            summary_parts.append(abstract_preview)

        if context.introduction:
            summary_parts.append("## The Problem")
            # Take first 300 chars of introduction
            intro_preview = context.introduction[:300]
            if len(context.introduction) > 300:
                intro_preview += "..."
            summary_parts.append(intro_preview)

        if context.conclusion:
            summary_parts.append("## Key Findings & Solution")
            # Take first 300 chars of conclusion
            conclusion_preview = context.conclusion[:300]
            if len(context.conclusion) > 300:
                conclusion_preview += "..."
            summary_parts.append(conclusion_preview)

        # Add helpful note
        summary_parts.append("\nThis summary is based on the key sections of the document. I can provide more specific information about methodology, findings, or other aspects if you have follow-up questions.")

        fallback_summary = "\n\n".join(summary_parts)
        logger.info(f"NEW PIPELINE: Created fallback summary ({len(fallback_summary)} chars)")

        return fallback_summary

    def _extract_memory_source(self, memory) -> str:
        """
        Extract the source identifier from a memory object.

        This handles different memory formats and extracts the document source.
        """
        try:
            # Try different ways to get the source
            if hasattr(memory, 'source'):
                return memory.source or ''

            # If it's a search result, get the chunk source
            if hasattr(memory, 'chunk'):
                chunk = memory.chunk
                if hasattr(chunk, 'source'):
                    return chunk.source or ''
                if hasattr(chunk, 'metadata'):
                    metadata = chunk.metadata or {}
                    return metadata.get('source', '')

            # Try to extract from content if it contains source info
            content = getattr(memory, 'content', '') or str(memory)
            if 'Document:' in content:
                # Extract document name from content like "Document: filename.pdf"
                import re
                match = re.search(r'Document:\s*([^\s\n]+)', content)
                if match:
                    return match.group(1)

            # Check for upload path patterns
            if 'web_ui/uploads/' in content:
                import re
                match = re.search(r'web_ui/uploads/[^/]+/([^:\s]+)', content)
                if match:
                    return match.group(1)

            return 'unknown'

        except Exception as e:
            logger.debug(f"NEW PIPELINE: Error extracting memory source: {e}")
            return 'unknown'

    def _is_memory_from_document(self, memory_source: str, target_document: str) -> bool:
        """
        Check if a memory source belongs to the target document.

        This implements strict document filtering to prevent contamination.
        """
        if not memory_source or memory_source == 'unknown':
            return False

        # Direct match
        if target_document in memory_source:
            return True

        # Check for different upload formats
        # e.g., "20250605_180152_2505.11340v1.pdf" should match "2505.11340v1.pdf"
        if target_document.replace('.pdf', '') in memory_source:
            return True

        # Check for document patterns in paths
        # e.g., "web_ui/uploads/20250605_180152_2505.11340v1.pdf:block_5"
        import re

        # Extract base filename without timestamp
        base_target = re.sub(r'^\d{8}_\d{6}_', '', target_document)
        if base_target in memory_source:
            return True

        # Extract filename from memory source
        source_filename = re.search(r'([^/]+\.pdf)', memory_source)
        if source_filename:
            source_file = source_filename.group(1)
            # Remove timestamp prefix if present
            clean_source = re.sub(r'^\d{8}_\d{6}_', '', source_file)
            if clean_source == target_document:
                return True

        return False

    def _synthesize_summary(self, context: SummaryContext, document_name: str, user_query: str) -> str:
        """
        TRUE SYNTHESIS: Generate actual summary using LLM synthesis, not extraction.

        This uses the curated context to create a proper synthesis with LLM.
        """
        logger.info("NEW PIPELINE: Synthesizing summary from curated content using LLM")

        # Build synthesis content with clean sections
        synthesis_content = self._build_synthesis_content(context)

        # Use structured prompts if available, fallback to hardcoded prompt
        if STRUCTURED_PROMPTS_AVAILABLE:
            try:
                prompt_manager = get_prompt_manager()
                system_prompt, user_prompt = prompt_manager.get_prompt(
                    PromptType.SUMMARY,
                    "enhanced_synthesis",
                    synthesis_content=synthesis_content
                )
                synthesis_prompt = f"{system_prompt}\n\n{user_prompt}"
                logger.info("NEW PIPELINE: Using structured synthesis prompt")
            except Exception as e:
                logger.warning(f"NEW PIPELINE: Structured prompt failed, using fallback: {e}")
                synthesis_prompt = self._get_fallback_synthesis_prompt(synthesis_content)
        else:
            synthesis_prompt = self._get_fallback_synthesis_prompt(synthesis_content)

        # Use LLM for true synthesis (fallback to rule-based if LLM unavailable)
        try:
            synthesized_summary = self._call_llm_for_synthesis(synthesis_prompt)
            if synthesized_summary and len(synthesized_summary) > 200:
                logger.info(f"NEW PIPELINE: LLM synthesis successful ({len(synthesized_summary)} chars)")
                return f"""## Summary of {context.document_title}

{synthesized_summary}

This summary is based on the key sections of the document. I can provide more specific information about methodology, findings, or other aspects if you have follow-up questions."""
            else:
                logger.warning("NEW PIPELINE: LLM synthesis failed or too short, falling back to rule-based")
        except Exception as e:
            logger.warning(f"NEW PIPELINE: LLM synthesis error: {e}, falling back to rule-based")

        # Fallback to improved rule-based synthesis
        return self._rule_based_synthesis(context)

    def _build_synthesis_content(self, context: SummaryContext) -> str:
        """Build clean content for synthesis."""
        content_parts = []

        if context.abstract:
            content_parts.append(f"ABSTRACT:\n{context.abstract}")

        if context.introduction:
            content_parts.append(f"INTRODUCTION:\n{context.introduction}")

        if context.conclusion:
            content_parts.append(f"CONCLUSION:\n{context.conclusion}")

        return '\n\n'.join(content_parts)

    def _get_fallback_synthesis_prompt(self, synthesis_content: str) -> str:
        """Get fallback synthesis prompt when structured prompts aren't available."""
        return f"""You are an expert research assistant. You will be given the key sections (Abstract, Introduction, and Conclusion) of an academic paper. Your task is to synthesize a high-quality summary.

Instructions:
- Read and understand all the provided text
- Write a summary in your own words. Do NOT copy sentences or paragraphs verbatim from the source text
- Structure the summary with the following markdown headings: ## Overview, ## The Problem, and ## Key Findings & Solution
- Crucially, omit any footnotes, reference markers, asterisks, or other formatting artifacts from the original text
- The final output must be clean prose without technical artifacts
- Focus on the core contributions and findings of the research

Here are the key sections from the paper:

{synthesis_content}

Generate a synthesized summary following the instructions above."""

    def _call_llm_for_synthesis(self, prompt: str) -> Optional[str]:
        """Call LLM for true synthesis of the content."""
        try:
            import requests
            import json

            # Use Ollama API for synthesis
            ollama_url = "http://localhost:11434/api/generate"

            payload = {
                "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused synthesis
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }

            # Extended timeout for complex document queries
            response = requests.post(ollama_url, json=payload, timeout=180)

            if response.status_code == 200:
                result = response.json()
                synthesized_text = result.get('response', '').strip()

                # Clean up any remaining artifacts
                synthesized_text = self._clean_llm_output(synthesized_text)

                return synthesized_text
            else:
                logger.warning(f"NEW PIPELINE: LLM API error: {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"NEW PIPELINE: LLM synthesis failed: {e}")
            return None

    def _clean_llm_output(self, text: str) -> str:
        """Clean LLM output to remove any remaining artifacts."""
        import re

        # Remove common artifacts that might slip through
        cleaned = re.sub(r'\*+[^*]*\*+', '', text)  # Remove asterisk annotations
        cleaned = re.sub(r'†[^†]*†?', '', cleaned)  # Remove dagger symbols
        cleaned = re.sub(r'\d+https?://[^\s]+', '', cleaned)  # Remove numbered URLs
        cleaned = re.sub(r'Equal contributions?[^\n]*', '', cleaned)  # Remove contribution notes
        cleaned = re.sub(r'Corresponding author[^\n]*', '', cleaned)  # Remove author notes
        cleaned = re.sub(r'\[\d+\]', '', cleaned)  # Remove citation numbers

        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()

        return cleaned

    def _rule_based_synthesis(self, context: SummaryContext) -> str:
        """Improved rule-based synthesis as fallback."""
        summary_parts = []

        # Document header
        summary_parts.append(f"## Summary of {context.document_title}")
        summary_parts.append("")

        # Core content synthesis with improved extraction
        if context.abstract:
            summary_parts.append("## Overview")
            clean_abstract = self._extract_key_insights(context.abstract, "overview")
            summary_parts.append(clean_abstract)
            summary_parts.append("")

        if context.introduction:
            summary_parts.append("## The Problem")
            clean_intro = self._extract_key_insights(context.introduction, "problem")
            summary_parts.append(clean_intro)
            summary_parts.append("")

        if context.conclusion:
            summary_parts.append("## Key Findings & Solution")
            clean_conclusion = self._extract_key_insights(context.conclusion, "findings")
            summary_parts.append(clean_conclusion)
            summary_parts.append("")

        # Closing
        summary_parts.append("This summary is based on the key sections of the document. "
                           "I can provide more specific information about methodology, findings, "
                           "or other aspects if you have follow-up questions.")

        return '\n'.join(summary_parts)

    def _extract_key_insights(self, content: str, section_type: str) -> str:
        """Extract key insights and rewrite them more naturally."""
        # Clean the content first
        clean_content = self._clean_content(content)

        # Remove artifacts more aggressively
        import re
        clean_content = re.sub(r'\*+[^*]*\*+', '', clean_content)  # Remove asterisk annotations
        clean_content = re.sub(r'†[^†]*†?', '', clean_content)  # Remove dagger symbols
        clean_content = re.sub(r'\d+https?://[^\s]+', '', clean_content)  # Remove numbered URLs
        clean_content = re.sub(r'Equal contributions?[^\n]*', '', clean_content)
        clean_content = re.sub(r'Corresponding author[^\n]*', '', clean_content)

        # Extract the most substantial sentences
        sentences = [s.strip() for s in clean_content.split('.') if len(s.strip()) > 50]

        # Filter out sentences with artifacts
        clean_sentences = []
        for sentence in sentences:
            if not any(artifact in sentence.lower() for artifact in [
                'equal contribution', 'corresponding author', 'https://', 'github',
                'arxiv', 'doi:', 'proceedings', 'conference'
            ]):
                clean_sentences.append(sentence)

        # Take the most informative sentences based on section type
        if section_type == "overview":
            # For overview, focus on the main contribution
            key_sentences = [s for s in clean_sentences if any(word in s.lower() for word in [
                'present', 'propose', 'introduce', 'framework', 'approach', 'method'
            ])][:2]
        elif section_type == "problem":
            # For problem, focus on the challenge being addressed
            key_sentences = [s for s in clean_sentences if any(word in s.lower() for word in [
                'challenge', 'problem', 'issue', 'gap', 'limitation', 'need'
            ])][:2]
        else:  # findings
            # For findings, focus on results and conclusions
            key_sentences = [s for s in clean_sentences if any(word in s.lower() for word in [
                'result', 'finding', 'conclude', 'demonstrate', 'show', 'achieve'
            ])][:2]

        # If no specific sentences found, use the first substantial ones
        if not key_sentences:
            key_sentences = clean_sentences[:2]

        # Join and clean up
        result = '. '.join(key_sentences)
        if result and not result.endswith('.'):
            result += '.'

        return result

    def _synthesize_section(self, content: str, section_type: str) -> str:
        """Synthesize a section into clean, readable summary text."""
        # Clean the content
        clean_content = self._clean_content(content)

        # Extract key sentences (simple approach for now)
        sentences = [s.strip() for s in clean_content.split('.') if len(s.strip()) > 50]

        # Take the most substantial sentences
        key_sentences = sentences[:3] if len(sentences) >= 3 else sentences

        # Join and clean up
        synthesis = '. '.join(key_sentences)
        if synthesis and not synthesis.endswith('.'):
            synthesis += '.'

        return synthesis

    def _strip_all_metadata(self, content: str) -> str:
        """
        STRICT OUTPUT FILTERING: Remove ALL metadata and system artifacts.

        This is the final firewall - nothing internal should pass through.
        """
        if not content:
            return content

        # Remove file paths and internal references
        cleaned = re.sub(r'web_ui/uploads/[^\s]+', '', content)
        cleaned = re.sub(r'document:[^\s]+', '', cleaned)
        cleaned = re.sub(r'block_\d+', '', cleaned)
        cleaned = re.sub(r'mem_[a-f0-9]+', '', cleaned)
        cleaned = re.sub(r'chunk_[a-f0-9]+', '', cleaned)

        # Remove internal metrics and processing info
        cleaned = re.sub(r'Content Quality:\s*[\d.]+%', '', cleaned)
        cleaned = re.sub(r'Memory Blocks:\s*\d+', '', cleaned)
        cleaned = re.sub(r'Processing Status:[^\n]+', '', cleaned)
        cleaned = re.sub(r'Content Type:[^\n]+', '', cleaned)
        cleaned = re.sub(r'Processing complete', '', cleaned)
        cleaned = re.sub(r'Source:[^\n]+', '', cleaned)
        cleaned = re.sub(r'Score:\s*[\d.]+', '', cleaned)

        # Remove timestamps and IDs
        cleaned = re.sub(r'\d{8}_\d{6}_', '', cleaned)
        cleaned = re.sub(r'ID:\s*[a-f0-9]+', '', cleaned)

        # Clean up extra whitespace and formatting
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = re.sub(r'\*\*\s*\*\*', '', cleaned)  # Empty bold tags
        cleaned = cleaned.strip()

        logger.info(f"NEW PIPELINE: Metadata stripped - {len(content)} -> {len(cleaned)} chars")
        return cleaned

    def _generate_fallback_response(self, document_name: str) -> str:
        """Generate a clean fallback when we can't find sufficient content."""
        clean_name = self._clean_document_name(document_name)

        return f"""## Summary of {clean_name}

I apologize, but I couldn't generate a comprehensive summary of this document. The content may be primarily non-textual (images, tables, or complex formatting) or the document structure may not follow standard academic paper conventions.

**Suggestions:**
- Try asking about specific aspects of the document
- Upload the document again if processing may have failed
- Ask more targeted questions about particular topics you know are in the document

I'm here to help with any specific questions about the document content!"""
