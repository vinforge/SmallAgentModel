"""
Knowledge Consolidation and Summarization for SAM
Uses SAM's LLM to summarize parsed multimodal content into clean, human-readable text with attribution.

Sprint 4 Task 2: Knowledge Consolidation and Summarization
"""

import logging
import json
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .document_parser import ParsedDocument, MultimodalContent
import requests

logger = logging.getLogger(__name__)

class OllamaModel:
    """Enhanced Ollama model implementation with knowledge injection for true learning."""

    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model_name = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
        self.learned_knowledge = []  # Store learned facts for context injection

    def inject_learned_knowledge(self, knowledge_summary: str, key_concepts: List[str]):
        """Inject learned knowledge into the model's context for future queries."""
        knowledge_entry = {
            'summary': knowledge_summary,
            'concepts': key_concepts,
            'timestamp': datetime.now().isoformat(),
            'source': 'document_learning'
        }
        self.learned_knowledge.append(knowledge_entry)

        # Keep only the most recent 10 knowledge entries to avoid context overflow
        if len(self.learned_knowledge) > 10:
            self.learned_knowledge = self.learned_knowledge[-10:]

        logger.info(f"🧠 Injected new knowledge into model context: {len(key_concepts)} concepts")

    def _build_enhanced_context(self, prompt: str) -> str:
        """Build enhanced context with learned knowledge for better responses."""
        if not self.learned_knowledge:
            return prompt

        # Create knowledge context
        knowledge_context = "LEARNED KNOWLEDGE FROM DOCUMENTS:\n"
        for i, knowledge in enumerate(self.learned_knowledge[-5:], 1):  # Use last 5 entries
            knowledge_context += f"\n{i}. {knowledge['summary'][:200]}...\n"
            knowledge_context += f"   Key concepts: {', '.join(knowledge['concepts'][:5])}\n"

        # Inject knowledge before the actual prompt
        enhanced_prompt = f"{knowledge_context}\n\nBased on the above learned knowledge and your training, respond to:\n\n{prompt}"
        return enhanced_prompt

    def generate(self, prompt, temperature=0.7, max_tokens=500, use_learned_knowledge=True):
        """Generate text using Ollama API with optional learned knowledge injection."""
        try:
            # Enhance prompt with learned knowledge if requested
            if use_learned_knowledge:
                enhanced_prompt = self._build_enhanced_context(prompt)
                logger.info(f"🧠 Enhanced prompt with {len(self.learned_knowledge)} learned knowledge entries")
            else:
                enhanced_prompt = prompt

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=180  # Extended timeout for complex document processing
            )
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Knowledge consolidation unavailable: {str(e)}"

@dataclass
class ConsolidatedKnowledge:
    """Represents consolidated and summarized knowledge from multimodal content."""
    consolidation_id: str
    source_document: str
    summary: str
    key_concepts: List[str]
    content_attribution: Dict[str, List[str]]  # Maps content types to source locations
    enriched_metadata: Dict[str, Any]
    consolidation_timestamp: str

class KnowledgeConsolidator:
    """
    Consolidates and summarizes multimodal content using SAM's LLM.
    """
    
    def __init__(self, model: Optional[OllamaModel] = None):
        """
        Initialize the knowledge consolidator.
        
        Args:
            model: Ollama model instance for summarization
        """
        self.model = model or self._init_model()
        
        logger.info("Knowledge consolidator initialized")
    
    def _init_model(self) -> OllamaModel:
        """Initialize the Ollama model."""
        try:
            return OllamaModel()
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model: {e}")
            raise
    
    def consolidate_document(self, parsed_doc: ParsedDocument) -> Optional[ConsolidatedKnowledge]:
        """
        Consolidate a parsed multimodal document into enriched knowledge.
        
        Args:
            parsed_doc: Parsed document with multimodal content
            
        Returns:
            ConsolidatedKnowledge object or None if consolidation failed
        """
        try:
            logger.info(f"Consolidating document: {parsed_doc.source_file}")
            
            # Group content by type for better processing
            content_groups = self._group_content_by_type(parsed_doc.content_blocks)
            
            # Generate comprehensive summary
            summary = self._generate_comprehensive_summary(content_groups, parsed_doc)
            
            # Extract key concepts
            key_concepts = self._extract_key_concepts(content_groups, summary)
            
            # Create content attribution mapping
            content_attribution = self._create_content_attribution(content_groups)
            
            # Create enriched metadata
            enriched_metadata = self._create_enriched_metadata(parsed_doc, content_groups)
            
            # Generate consolidation ID
            consolidation_id = f"consol_{datetime.now().timestamp()}_{hash(parsed_doc.document_id) % 10000}"
            
            consolidated = ConsolidatedKnowledge(
                consolidation_id=consolidation_id,
                source_document=parsed_doc.source_file,
                summary=summary,
                key_concepts=key_concepts,
                content_attribution=content_attribution,
                enriched_metadata=enriched_metadata,
                consolidation_timestamp=datetime.now().isoformat()
            )

            # CRITICAL: Inject learned knowledge into the model for true learning
            self.model.inject_learned_knowledge(summary, key_concepts)
            logger.info(f"🎓 MODEL LEARNING: Injected knowledge from {parsed_doc.source_file} into Ollama model context")

            logger.info(f"Successfully consolidated document: {len(summary)} chars summary, {len(key_concepts)} key concepts")
            return consolidated
            
        except Exception as e:
            logger.error(f"Error consolidating document {parsed_doc.source_file}: {e}")
            return None
    
    def _group_content_by_type(self, content_blocks: List[MultimodalContent]) -> Dict[str, List[MultimodalContent]]:
        """Group content blocks by their type."""
        groups = {
            'text': [],
            'code': [],
            'table': [],
            'image': []
        }
        
        for block in content_blocks:
            content_type = block.content_type
            if content_type in groups:
                groups[content_type].append(block)
        
        return groups
    
    def _generate_comprehensive_summary(self, content_groups: Dict[str, List[MultimodalContent]], 
                                      parsed_doc: ParsedDocument) -> str:
        """Generate a comprehensive summary of all multimodal content."""
        try:
            # Prepare content overview for the LLM
            content_overview = self._prepare_content_overview(content_groups)
            
            # Create summarization prompt
            prompt = self._create_summarization_prompt(content_overview, parsed_doc)
            
            # Generate summary using Ollama
            summary = self.model.generate(prompt, temperature=0.3, max_tokens=1000)
            
            # Clean up summary
            summary = self._clean_summary(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary generation failed due to processing error."
    
    def _prepare_content_overview(self, content_groups: Dict[str, List[MultimodalContent]]) -> str:
        """Prepare a structured overview of all content for the LLM."""
        overview_parts = []
        
        # Text content
        if content_groups['text']:
            text_content = []
            for i, text_block in enumerate(content_groups['text'][:5], 1):  # Limit to first 5
                content = str(text_block.content)[:300]  # Limit length
                text_content.append(f"Text Block {i}: {content}...")
            
            overview_parts.append(f"TEXT CONTENT ({len(content_groups['text'])} blocks):\n" + 
                                "\n".join(text_content))
        
        # Code content
        if content_groups['code']:
            code_content = []
            for i, code_block in enumerate(content_groups['code'][:3], 1):  # Limit to first 3
                language = code_block.metadata.get('language', 'unknown')
                code = str(code_block.content)[:200]  # Limit length
                code_content.append(f"Code Block {i} ({language}): {code}...")
            
            overview_parts.append(f"CODE CONTENT ({len(content_groups['code'])} blocks):\n" + 
                                "\n".join(code_content))
        
        # Table content
        if content_groups['table']:
            table_content = []
            for i, table_block in enumerate(content_groups['table'][:3], 1):  # Limit to first 3
                table_data = table_block.content
                rows = len(table_data) if isinstance(table_data, list) else 0
                cols = len(table_data[0]) if table_data and isinstance(table_data[0], list) else 0
                
                # Show first few rows
                preview = ""
                if isinstance(table_data, list) and table_data:
                    preview = str(table_data[:2])  # First 2 rows
                
                table_content.append(f"Table {i} ({rows}x{cols}): {preview}...")
            
            overview_parts.append(f"TABLE CONTENT ({len(content_groups['table'])} tables):\n" + 
                                "\n".join(table_content))
        
        # Image content
        if content_groups['image']:
            image_content = []
            for i, image_block in enumerate(content_groups['image'][:3], 1):  # Limit to first 3
                if isinstance(image_block.content, dict):
                    description = image_block.content.get('description', 'No description')
                    alt_text = image_block.content.get('alt', '')
                else:
                    description = str(image_block.content)[:100]
                    alt_text = ''
                
                image_content.append(f"Image {i}: {description} (Alt: {alt_text})")
            
            overview_parts.append(f"IMAGE CONTENT ({len(content_groups['image'])} images):\n" + 
                                "\n".join(image_content))
        
        return "\n\n".join(overview_parts)
    
    def _create_summarization_prompt(self, content_overview: str, parsed_doc: ParsedDocument) -> str:
        """Create a prompt for comprehensive summarization."""
        
        prompt = f"""You are SAM, analyzing and consolidating multimodal document content. Your task is to create a comprehensive, well-structured summary that captures the essence and key information from all content types.

**Document:** {parsed_doc.source_file}
**Content Types Found:** {', '.join(parsed_doc.parsing_stats.get('content_types', {}).keys())}
**Total Content Blocks:** {parsed_doc.parsing_stats.get('total_blocks', 0)}

**MULTIMODAL CONTENT OVERVIEW:**
{content_overview}

**Your Task:**
Create a comprehensive summary that:

1. **Synthesizes Information**: Combine insights from text, code, tables, and images into a coherent narrative
2. **Identifies Key Themes**: Highlight the main topics and concepts covered
3. **Explains Technical Content**: Describe what code blocks and tables demonstrate or illustrate
4. **Contextualizes Visual Elements**: Explain how images relate to the textual content
5. **Maintains Attribution**: Reference different content types when relevant

**Output Format:**
Provide a well-structured summary (3-5 paragraphs) that reads naturally while incorporating insights from all content modalities. Focus on:
- Main concepts and themes
- Technical implementations or examples (from code)
- Data insights (from tables)
- Visual context (from images)
- Practical applications or implications

**Summary:**"""

        return prompt
    
    def _extract_key_concepts(self, content_groups: Dict[str, List[MultimodalContent]], 
                            summary: str) -> List[str]:
        """Extract key concepts from the content and summary."""
        try:
            # Create concept extraction prompt
            prompt = f"""Based on the following summary and content analysis, extract 5-10 key concepts or terms that represent the most important ideas in this document.

**Summary:**
{summary[:500]}...

**Content Statistics:**
- Text blocks: {len(content_groups['text'])}
- Code blocks: {len(content_groups['code'])}
- Tables: {len(content_groups['table'])}
- Images: {len(content_groups['image'])}

**Instructions:**
Extract key concepts as single words or short phrases (2-3 words max). Focus on:
- Technical terms and concepts
- Main topics and themes
- Important methodologies or approaches
- Key technologies or tools mentioned

**Format:** Return exactly one concept per line, no numbering or bullets.

**Key Concepts:**"""

            # Generate concepts using Ollama
            response = self.model.generate(prompt, temperature=0.2, max_tokens=300)
            
            # Parse concepts from response
            concepts = []
            for line in response.strip().split('\n'):
                concept = line.strip()
                if concept and len(concept) > 2 and len(concept) < 50:
                    concepts.append(concept)
            
            # Limit to 10 concepts
            return concepts[:10]
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {e}")
            return []
    
    def _create_content_attribution(self, content_groups: Dict[str, List[MultimodalContent]]) -> Dict[str, List[str]]:
        """Create attribution mapping for different content types."""
        attribution = {}
        
        for content_type, blocks in content_groups.items():
            if blocks:
                locations = []
                for block in blocks:
                    if block.source_location:
                        locations.append(block.source_location)
                
                if locations:
                    attribution[content_type] = locations
        
        return attribution
    
    def _create_enriched_metadata(self, parsed_doc: ParsedDocument, 
                                content_groups: Dict[str, List[MultimodalContent]]) -> Dict[str, Any]:
        """Create enriched metadata for the consolidated knowledge."""
        metadata = {
            'source_metadata': parsed_doc.document_metadata,
            'parsing_stats': parsed_doc.parsing_stats,
            'content_diversity_score': self._calculate_content_diversity(content_groups),
            'technical_content_ratio': self._calculate_technical_ratio(content_groups),
            'multimodal_richness': self._calculate_multimodal_richness(content_groups),
            'consolidation_quality': 'high'  # Could be calculated based on various factors
        }
        
        # Add language distribution for code blocks
        if content_groups['code']:
            languages = {}
            for code_block in content_groups['code']:
                lang = code_block.metadata.get('language', 'unknown')
                languages[lang] = languages.get(lang, 0) + 1
            metadata['programming_languages'] = languages
        
        # Add table statistics
        if content_groups['table']:
            table_stats = {
                'total_tables': len(content_groups['table']),
                'total_rows': sum(len(table.content) if isinstance(table.content, list) else 0 
                                for table in content_groups['table']),
                'avg_columns': 0
            }
            
            total_cols = 0
            table_count = 0
            for table in content_groups['table']:
                if isinstance(table.content, list) and table.content:
                    total_cols += len(table.content[0]) if isinstance(table.content[0], list) else 0
                    table_count += 1
            
            if table_count > 0:
                table_stats['avg_columns'] = total_cols / table_count
            
            metadata['table_statistics'] = table_stats
        
        return metadata
    
    def _calculate_content_diversity(self, content_groups: Dict[str, List[MultimodalContent]]) -> float:
        """Calculate content diversity score (0-1)."""
        total_blocks = sum(len(blocks) for blocks in content_groups.values())
        if total_blocks == 0:
            return 0.0
        
        # Calculate entropy-like measure
        diversity = 0.0
        for blocks in content_groups.values():
            if blocks:
                ratio = len(blocks) / total_blocks
                diversity -= ratio * math.log2(ratio) if ratio > 0 else 0
        
        # Normalize to 0-1 range
        max_diversity = 2.0  # log2(4) for 4 content types
        return min(diversity / max_diversity, 1.0) if max_diversity > 0 else 0.0
    
    def _calculate_technical_ratio(self, content_groups: Dict[str, List[MultimodalContent]]) -> float:
        """Calculate ratio of technical content (code + tables) to total content."""
        total_blocks = sum(len(blocks) for blocks in content_groups.values())
        if total_blocks == 0:
            return 0.0
        
        technical_blocks = len(content_groups['code']) + len(content_groups['table'])
        return technical_blocks / total_blocks
    
    def _calculate_multimodal_richness(self, content_groups: Dict[str, List[MultimodalContent]]) -> float:
        """Calculate multimodal richness score based on content type variety."""
        content_types_present = sum(1 for blocks in content_groups.values() if blocks)
        max_types = len(content_groups)  # 4 types: text, code, table, image
        
        return content_types_present / max_types if max_types > 0 else 0.0
    
    def _clean_summary(self, summary: str) -> str:
        """Clean up the generated summary."""
        # Remove thinking tags if present
        if '<think>' in summary:
            parts = summary.split('</think>')
            if len(parts) > 1:
                summary = parts[-1].strip()
        
        # Remove any remaining artifacts
        summary = summary.strip()
        
        # Remove leading "Summary:" if present
        if summary.startswith("Summary:"):
            summary = summary[8:].strip()
        
        return summary

# Global consolidator instance
_knowledge_consolidator = None

def get_knowledge_consolidator() -> KnowledgeConsolidator:
    """Get or create a global knowledge consolidator instance."""
    global _knowledge_consolidator
    
    if _knowledge_consolidator is None:
        _knowledge_consolidator = KnowledgeConsolidator()
    
    return _knowledge_consolidator
