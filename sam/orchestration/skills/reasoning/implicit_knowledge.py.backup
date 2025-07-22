"""
ImplicitKnowledgeSkill - PEI Framework Implementation
====================================================

Implements the core principles of the PEI (Prompt-based Explicit-to-Implicit) framework
as a reusable reasoning service within SAM. This skill generates implicit knowledge
connections between explicit text chunks using transformer models.

Key Features:
- Generates implicit knowledge embeddings that connect text chunks
- Produces both machine-usable context and human-readable summaries
- Integrates with BART model for knowledge prompting
- Supports both Q&A enhancement and Dream Canvas visualization
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from ..base import BaseSkillModule, SkillExecutionError
from ...uif import SAM_UIF

logger = logging.getLogger(__name__)


@dataclass
class ImplicitKnowledgeResult:
    """Result container for implicit knowledge generation."""
    implicit_knowledge_summary: str
    unified_context: str
    confidence_score: float
    processing_time: float
    source_chunks_count: int


class ImplicitKnowledgeSkill(BaseSkillModule):
    """
    Skill that generates implicit knowledge connections between explicit text chunks.
    
    This skill implements the PEI framework to discover unstated logical connections
    between multiple pieces of text, producing both machine-usable context for
    reasoning and human-readable summaries for UI display.
    
    Integrates with:
    - BART model for knowledge prompting
    - SAM's core LLM for summary generation
    - Dream Canvas for visualization
    - Q&A workflow for enhanced reasoning
    """
    
    skill_name = "ImplicitKnowledgeSkill"
    skill_version = "1.0.0"
    skill_description = "Generates the unstated logical connection between multiple pieces of text"
    skill_category = "reasoning"
    
    # Dependency declarations
    required_inputs = ["explicit_knowledge_chunks"]
    optional_inputs = ["context", "user_profile", "confidence_threshold"]
    output_keys = ["implicit_knowledge_summary", "unified_context"]
    
    # Skill capabilities
    requires_external_access = False
    requires_vetting = False
    can_run_parallel = True
    estimated_execution_time = 3.0
    max_execution_time = 30.0
    
    def __init__(self):
        """Initialize the ImplicitKnowledgeSkill."""
        super().__init__()
        self._bart_model = None
        self._sam_llm = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize BART model and SAM LLM for knowledge processing."""
        try:
            # Initialize BART model for implicit knowledge generation
            # For Phase 0, we'll use SAM's existing Ollama model as a substitute
            # In future phases, this would be replaced with actual BART model
            self._bart_model = self._get_ollama_model()
            self._sam_llm = self._get_ollama_model()
            
            self.logger.info("✅ ImplicitKnowledgeSkill models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize models: {e}")
            # Continue with degraded functionality
            self._bart_model = None
            self._sam_llm = None
    
    def _get_ollama_model(self):
        """Get Ollama model interface for knowledge processing."""
        try:
            import requests
            
            class OllamaInterface:
                def __init__(self):
                    self.base_url = "http://localhost:11434"
                    self.model_name = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
                
                def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
                    """Generate response using Ollama API."""
                    try:
                        response = requests.post(
                            f"{self.base_url}/api/generate",
                            json={
                                "model": self.model_name,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": temperature,
                                    "top_p": 0.9,
                                    "max_tokens": max_tokens
                                }
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            return response.json().get('response', '').strip()
                        else:
                            raise Exception(f"Ollama API error: {response.status_code}")
                    
                    except Exception as e:
                        logger.warning(f"Ollama generation failed: {e}")
                        return ""
            
            return OllamaInterface()
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Ollama interface: {e}")
            return None
    
    def execute(self, uif: SAM_UIF) -> bool:
        """
        Execute implicit knowledge generation.
        
        Args:
            uif: Universal Interface Format containing execution context
            
        Returns:
            bool: True if execution successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self._validate_inputs(uif):
                return False
            
            # Extract explicit knowledge chunks
            explicit_chunks = uif.intermediate_data.get("explicit_knowledge_chunks", [])
            
            if len(explicit_chunks) < 2:
                raise SkillExecutionError("ImplicitKnowledgeSkill requires at least 2 explicit knowledge chunks")
            
            self.logger.info(f"🧠 Processing {len(explicit_chunks)} knowledge chunks for implicit connections")
            
            # Generate implicit knowledge
            result = self._generate_implicit_knowledge(explicit_chunks, uif)
            
            # Store outputs in UIF
            uif.intermediate_data["implicit_knowledge_summary"] = result.implicit_knowledge_summary
            uif.intermediate_data["unified_context"] = result.unified_context
            uif.intermediate_data["implicit_knowledge_confidence"] = result.confidence_score
            uif.intermediate_data["implicit_knowledge_metadata"] = {
                "processing_time": result.processing_time,
                "source_chunks_count": result.source_chunks_count,
                "skill_version": self.skill_version
            }
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ ImplicitKnowledgeSkill completed in {processing_time:.2f}s")

            return True

        except Exception as e:
            self.logger.exception(f"❌ ImplicitKnowledgeSkill execution failed: {e}")
            uif.add_log_entry(f"ImplicitKnowledgeSkill error: {str(e)}")
            return False

    def _validate_inputs(self, uif: SAM_UIF) -> bool:
        """Validate required inputs are present and valid."""
        try:
            explicit_chunks = uif.intermediate_data.get("explicit_knowledge_chunks")

            if not explicit_chunks:
                raise SkillExecutionError("Missing required input: explicit_knowledge_chunks")

            if not isinstance(explicit_chunks, list):
                raise SkillExecutionError("explicit_knowledge_chunks must be a list")

            if len(explicit_chunks) < 2:
                raise SkillExecutionError("At least 2 knowledge chunks required for implicit connection generation")

            # Validate each chunk is a non-empty string
            for i, chunk in enumerate(explicit_chunks):
                if not isinstance(chunk, str) or not chunk.strip():
                    raise SkillExecutionError(f"Knowledge chunk {i} must be a non-empty string")

            return True

        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False

    def _generate_implicit_knowledge(self, explicit_chunks: List[str], uif: SAM_UIF) -> ImplicitKnowledgeResult:
        """
        Generate implicit knowledge connections between explicit chunks.

        Args:
            explicit_chunks: List of text chunks to find connections between
            uif: Universal Interface Format for context

        Returns:
            ImplicitKnowledgeResult with generated connections
        """
        start_time = time.time()

        try:
            # Step 1: Generate implicit knowledge embeddings using BART-like approach
            implicit_connections = self._generate_implicit_connections(explicit_chunks)

            # Step 2: Create unified context by combining original text with implicit knowledge
            unified_context = self._create_unified_context(explicit_chunks, implicit_connections)

            # Step 3: Generate human-readable summary using SAM's LLM
            summary = self._generate_human_readable_summary(implicit_connections, explicit_chunks)

            # Step 4: Calculate confidence score
            confidence = self._calculate_confidence_score(explicit_chunks, implicit_connections)

            processing_time = time.time() - start_time

            return ImplicitKnowledgeResult(
                implicit_knowledge_summary=summary,
                unified_context=unified_context,
                confidence_score=confidence,
                processing_time=processing_time,
                source_chunks_count=len(explicit_chunks)
            )

        except Exception as e:
            self.logger.error(f"Failed to generate implicit knowledge: {e}")
            # Return fallback result
            return self._create_fallback_result(explicit_chunks, time.time() - start_time)

    def _generate_implicit_connections(self, chunks: List[str]) -> str:
        """Generate implicit connections between text chunks using BART-like approach."""
        try:
            if not self._bart_model:
                return self._generate_fallback_connections(chunks)

            # Create prompt for implicit knowledge generation
            prompt = self._create_implicit_knowledge_prompt(chunks)

            # Generate implicit connections
            implicit_knowledge = self._bart_model.generate(
                prompt=prompt,
                temperature=0.5,  # Lower temperature for more focused connections
                max_tokens=300
            )

            if not implicit_knowledge:
                return self._generate_fallback_connections(chunks)

            return implicit_knowledge

        except Exception as e:
            self.logger.warning(f"BART model generation failed: {e}")
            return self._generate_fallback_connections(chunks)

    def _create_implicit_knowledge_prompt(self, chunks: List[str]) -> str:
        """Create prompt for implicit knowledge generation."""
        chunks_text = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks)])

        prompt = f"""You are an expert at finding hidden connections between pieces of information.

Given these separate pieces of knowledge:

{chunks_text}

Identify the implicit, unstated logical connections that link these pieces together. Focus on:
- Underlying relationships not explicitly mentioned
- Causal connections between concepts
- Shared principles or mechanisms
- Bridging concepts that connect the information

Generate a concise explanation of the implicit knowledge that connects these pieces:"""

        return prompt

    def _generate_fallback_connections(self, chunks: List[str]) -> str:
        """Generate basic connections when BART model is unavailable."""
        try:
            # Simple pattern-based connection generation
            common_terms = self._extract_common_terms(chunks)

            if common_terms:
                return f"These pieces of information are connected through shared concepts: {', '.join(common_terms)}. They likely represent different aspects of the same underlying domain or process."
            else:
                return "These pieces of information may be connected through implicit relationships that require deeper analysis to uncover."

        except Exception:
            return "Implicit connections exist between these knowledge pieces but require further analysis."

    def _extract_common_terms(self, chunks: List[str]) -> List[str]:
        """Extract common terms between chunks for fallback connection generation."""
        try:
            import re
            from collections import Counter

            # Simple term extraction (in production, would use more sophisticated NLP)
            all_words = []
            for chunk in chunks:
                # Extract meaningful words (length > 3, not common stop words)
                words = re.findall(r'\b[a-zA-Z]{4,}\b', chunk.lower())
                all_words.extend(words)

            # Find words that appear in multiple chunks
            word_counts = Counter(all_words)
            common_words = [word for word, count in word_counts.items() if count > 1]

            # Filter out very common words
            stop_words = {'that', 'this', 'with', 'from', 'they', 'have', 'been', 'were', 'will', 'would', 'could', 'should'}
            common_words = [word for word in common_words if word not in stop_words]

            return common_words[:5]  # Return top 5 common terms

        except Exception:
            return []

    def _create_unified_context(self, original_chunks: List[str], implicit_connections: str) -> str:
        """Create unified context by combining original text with implicit knowledge."""
        try:
            # Combine original chunks with implicit connections
            context_parts = []

            # Add original chunks
            for i, chunk in enumerate(original_chunks):
                context_parts.append(f"Source {i+1}: {chunk}")

            # Add implicit connections
            context_parts.append(f"Implicit Connections: {implicit_connections}")

            # Create unified context
            unified_context = "\n\n".join(context_parts)

            return unified_context

        except Exception as e:
            self.logger.warning(f"Failed to create unified context: {e}")
            # Fallback: just concatenate original chunks
            return "\n\n".join(original_chunks)

    def _generate_human_readable_summary(self, implicit_connections: str, original_chunks: List[str]) -> str:
        """Generate human-readable summary of implicit connections."""
        try:
            if not self._sam_llm:
                return self._create_fallback_summary(implicit_connections)

            # Create prompt for summary generation
            prompt = f"""Convert the following implicit knowledge analysis into a concise, human-readable sentence that explains the connection:

Implicit Knowledge Analysis:
{implicit_connections}

Original Knowledge Pieces:
{chr(10).join([f"- {chunk[:100]}..." if len(chunk) > 100 else f"- {chunk}" for chunk in original_chunks])}

Generate a single, clear sentence that starts with "Inferred Connection:" and explains how these pieces are related:"""

            summary = self._sam_llm.generate(
                prompt=prompt,
                temperature=0.3,  # Low temperature for consistent summaries
                max_tokens=100
            )

            if summary and "Inferred Connection:" in summary:
                return summary.strip()
            elif summary:
                return f"Inferred Connection: {summary.strip()}"
            else:
                return self._create_fallback_summary(implicit_connections)

        except Exception as e:
            self.logger.warning(f"Summary generation failed: {e}")
            return self._create_fallback_summary(implicit_connections)

    def _create_fallback_summary(self, implicit_connections: str) -> str:
        """Create fallback summary when LLM is unavailable."""
        try:
            # Extract key phrases from implicit connections
            if len(implicit_connections) > 100:
                summary_text = implicit_connections[:97] + "..."
            else:
                summary_text = implicit_connections

            return f"Inferred Connection: {summary_text}"

        except Exception:
            return "Inferred Connection: These knowledge pieces share underlying relationships that connect them conceptually."

    def _calculate_confidence_score(self, chunks: List[str], implicit_connections: str) -> float:
        """Calculate confidence score for the implicit knowledge generation."""
        try:
            confidence = 0.5  # Base confidence

            # Increase confidence based on chunk quality
            if len(chunks) >= 3:
                confidence += 0.1

            # Increase confidence based on connection quality
            if len(implicit_connections) > 50:
                confidence += 0.1

            if any(keyword in implicit_connections.lower() for keyword in ['because', 'therefore', 'thus', 'since', 'due to']):
                confidence += 0.1

            # Increase confidence if models are available
            if self._bart_model and self._sam_llm:
                confidence += 0.2

            return min(confidence, 1.0)

        except Exception:
            return 0.5

    def _create_fallback_result(self, chunks: List[str], processing_time: float) -> ImplicitKnowledgeResult:
        """Create fallback result when generation fails."""
        return ImplicitKnowledgeResult(
            implicit_knowledge_summary="Inferred Connection: These knowledge pieces are related through implicit connections that require further analysis.",
            unified_context="\n\n".join(chunks),
            confidence_score=0.3,
            processing_time=processing_time,
            source_chunks_count=len(chunks)
        )
