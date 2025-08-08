"""
Multimodal Enrichment Scoring for SAM
Scores knowledge chunks by content richness and diversity to prioritize learning value.

Sprint 4 Task 3: Multimodal Enrichment Scoring
"""

import logging
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .knowledge_consolidator import ConsolidatedKnowledge
from .document_parser import ParsedDocument, MultimodalContent

logger = logging.getLogger(__name__)

@dataclass
class EnrichmentScore:
    """Represents the enrichment score for a piece of content."""
    overall_score: float
    component_scores: Dict[str, float]
    score_explanation: str
    priority_level: str  # 'high', 'medium', 'low'

class MultimodalEnrichmentScorer:
    """
    Scores multimodal content based on richness, diversity, and learning value.
    """
    
    def __init__(self):
        """Initialize the enrichment scorer."""
        self.scoring_weights = {
            'content_diversity': 0.25,      # Variety of content types
            'technical_depth': 0.20,        # Presence of code, tables, technical content
            'information_density': 0.20,    # Amount of information per unit
            'structural_quality': 0.15,     # Organization and structure
            'multimodal_integration': 0.10, # How well different modalities work together
            'novelty_potential': 0.10       # Potential for new learning
        }
        
        logger.info("Multimodal enrichment scorer initialized")
        logger.info(f"Scoring weights: {self.scoring_weights}")
    
    def score_consolidated_knowledge(self, consolidated: ConsolidatedKnowledge) -> EnrichmentScore:
        """
        Score consolidated knowledge for enrichment value.
        
        Args:
            consolidated: ConsolidatedKnowledge object to score
            
        Returns:
            EnrichmentScore with detailed scoring breakdown
        """
        try:
            logger.debug(f"Scoring consolidated knowledge: {consolidated.consolidation_id}")
            
            # Calculate component scores
            component_scores = {}
            
            # Content diversity score
            component_scores['content_diversity'] = self._score_content_diversity(consolidated)
            
            # Technical depth score
            component_scores['technical_depth'] = self._score_technical_depth(consolidated)
            
            # Information density score
            component_scores['information_density'] = self._score_information_density(consolidated)
            
            # Structural quality score
            component_scores['structural_quality'] = self._score_structural_quality(consolidated)
            
            # Multimodal integration score
            component_scores['multimodal_integration'] = self._score_multimodal_integration(consolidated)
            
            # Novelty potential score
            component_scores['novelty_potential'] = self._score_novelty_potential(consolidated)
            
            # Calculate weighted overall score
            overall_score = sum(
                score * self.scoring_weights[component]
                for component, score in component_scores.items()
            )
            
            # Determine priority level
            priority_level = self._determine_priority_level(overall_score)
            
            # Generate explanation
            score_explanation = self._generate_score_explanation(component_scores, overall_score)
            
            enrichment_score = EnrichmentScore(
                overall_score=overall_score,
                component_scores=component_scores,
                score_explanation=score_explanation,
                priority_level=priority_level
            )
            
            logger.debug(f"Enrichment score calculated: {overall_score:.2f} ({priority_level})")
            return enrichment_score
            
        except Exception as e:
            logger.error(f"Error scoring consolidated knowledge: {e}")
            return self._create_default_score()
    
    def score_parsed_document(self, parsed_doc: ParsedDocument) -> EnrichmentScore:
        """
        Score a parsed document for enrichment value.
        
        Args:
            parsed_doc: ParsedDocument to score
            
        Returns:
            EnrichmentScore with detailed scoring breakdown
        """
        try:
            logger.debug(f"Scoring parsed document: {parsed_doc.document_id}")
            
            # Group content by type
            content_groups = self._group_content_by_type(parsed_doc.content_blocks)
            
            # Calculate component scores
            component_scores = {}
            
            component_scores['content_diversity'] = self._score_content_diversity_from_groups(content_groups)
            component_scores['technical_depth'] = self._score_technical_depth_from_groups(content_groups)
            component_scores['information_density'] = self._score_information_density_from_doc(parsed_doc)
            component_scores['structural_quality'] = self._score_structural_quality_from_doc(parsed_doc)
            component_scores['multimodal_integration'] = self._score_multimodal_integration_from_groups(content_groups)
            component_scores['novelty_potential'] = self._score_novelty_potential_from_doc(parsed_doc)
            
            # Calculate weighted overall score
            overall_score = sum(
                score * self.scoring_weights[component]
                for component, score in component_scores.items()
            )
            
            # Determine priority level
            priority_level = self._determine_priority_level(overall_score)
            
            # Generate explanation
            score_explanation = self._generate_score_explanation(component_scores, overall_score)
            
            enrichment_score = EnrichmentScore(
                overall_score=overall_score,
                component_scores=component_scores,
                score_explanation=score_explanation,
                priority_level=priority_level
            )
            
            logger.debug(f"Document enrichment score: {overall_score:.2f} ({priority_level})")
            return enrichment_score
            
        except Exception as e:
            logger.error(f"Error scoring parsed document: {e}")
            return self._create_default_score()
    
    def _score_content_diversity(self, consolidated: ConsolidatedKnowledge) -> float:
        """Score content diversity from consolidated knowledge."""
        metadata = consolidated.enriched_metadata
        
        # Use pre-calculated diversity score if available
        if 'content_diversity_score' in metadata:
            return metadata['content_diversity_score']
        
        # Fallback: calculate from content attribution
        content_types = len(consolidated.content_attribution)
        max_types = 4  # text, code, table, image
        
        return content_types / max_types
    
    def _score_content_diversity_from_groups(self, content_groups: Dict[str, List[MultimodalContent]]) -> float:
        """Score content diversity from content groups."""
        present_types = sum(1 for blocks in content_groups.values() if blocks)
        max_types = len(content_groups)
        
        diversity_ratio = present_types / max_types
        
        # Bonus for balanced distribution
        if present_types > 1:
            total_blocks = sum(len(blocks) for blocks in content_groups.values())
            if total_blocks > 0:
                # Calculate distribution balance (lower std dev = more balanced)
                type_ratios = [len(blocks) / total_blocks for blocks in content_groups.values() if blocks]
                if len(type_ratios) > 1:
                    mean_ratio = sum(type_ratios) / len(type_ratios)
                    variance = sum((ratio - mean_ratio) ** 2 for ratio in type_ratios) / len(type_ratios)
                    balance_bonus = 1.0 - min(math.sqrt(variance) * 2, 0.5)  # Cap bonus at 0.5
                    diversity_ratio += balance_bonus * 0.2  # 20% bonus for balance
        
        return min(diversity_ratio, 1.0)
    
    def _score_technical_depth(self, consolidated: ConsolidatedKnowledge) -> float:
        """Score technical depth from consolidated knowledge."""
        metadata = consolidated.enriched_metadata
        
        # Use pre-calculated technical ratio if available
        technical_ratio = metadata.get('technical_content_ratio', 0.0)
        
        # Bonus for programming language diversity
        lang_bonus = 0.0
        if 'programming_languages' in metadata:
            languages = metadata['programming_languages']
            lang_count = len(languages)
            lang_bonus = min(lang_count * 0.1, 0.3)  # Up to 30% bonus for language diversity
        
        # Bonus for table complexity
        table_bonus = 0.0
        if 'table_statistics' in metadata:
            table_stats = metadata['table_statistics']
            total_tables = table_stats.get('total_tables', 0)
            total_rows = table_stats.get('total_rows', 0)
            
            if total_tables > 0:
                avg_rows = total_rows / total_tables
                table_bonus = min(avg_rows * 0.02, 0.2)  # Up to 20% bonus for table complexity
        
        return min(technical_ratio + lang_bonus + table_bonus, 1.0)
    
    def _score_technical_depth_from_groups(self, content_groups: Dict[str, List[MultimodalContent]]) -> float:
        """Score technical depth from content groups."""
        total_blocks = sum(len(blocks) for blocks in content_groups.values())
        if total_blocks == 0:
            return 0.0
        
        technical_blocks = len(content_groups['code']) + len(content_groups['table'])
        technical_ratio = technical_blocks / total_blocks
        
        # Bonus for code language diversity
        lang_bonus = 0.0
        if content_groups['code']:
            languages = set()
            for code_block in content_groups['code']:
                lang = code_block.metadata.get('language', 'unknown')
                if lang != 'unknown':
                    languages.add(lang)
            lang_bonus = min(len(languages) * 0.1, 0.3)
        
        # Bonus for table complexity
        table_bonus = 0.0
        if content_groups['table']:
            total_rows = 0
            for table_block in content_groups['table']:
                if isinstance(table_block.content, list):
                    total_rows += len(table_block.content)
            
            avg_rows = total_rows / len(content_groups['table'])
            table_bonus = min(avg_rows * 0.02, 0.2)
        
        return min(technical_ratio + lang_bonus + table_bonus, 1.0)
    
    def _score_information_density(self, consolidated: ConsolidatedKnowledge) -> float:
        """Score information density from consolidated knowledge."""
        summary_length = len(consolidated.summary)
        concept_count = len(consolidated.key_concepts)
        
        # Base score from summary length (normalized)
        length_score = min(summary_length / 2000, 1.0)  # Normalize to 2000 chars
        
        # Bonus for key concepts
        concept_score = min(concept_count / 10, 1.0)  # Normalize to 10 concepts
        
        # Weighted combination
        return (length_score * 0.7) + (concept_score * 0.3)
    
    def _score_information_density_from_doc(self, parsed_doc: ParsedDocument) -> float:
        """Score information density from parsed document."""
        stats = parsed_doc.parsing_stats
        
        total_blocks = stats.get('total_blocks', 0)
        text_length = stats.get('total_text_length', 0)
        
        if total_blocks == 0:
            return 0.0
        
        # Average content per block
        avg_content = text_length / total_blocks if total_blocks > 0 else 0
        
        # Normalize to reasonable range
        density_score = min(avg_content / 500, 1.0)  # Normalize to 500 chars per block
        
        # Bonus for block variety
        content_types = len(stats.get('content_types', {}))
        variety_bonus = min(content_types * 0.1, 0.3)
        
        return min(density_score + variety_bonus, 1.0)
    
    def _score_structural_quality(self, consolidated: ConsolidatedKnowledge) -> float:
        """Score structural quality from consolidated knowledge."""
        # Check if summary is well-structured
        summary = consolidated.summary
        
        # Look for structure indicators
        structure_indicators = [
            '\n\n',  # Paragraph breaks
            '1.', '2.', '3.',  # Numbered lists
            '- ', '* ',  # Bullet points
            ':',  # Colons for definitions
        ]
        
        structure_score = 0.0
        for indicator in structure_indicators:
            if indicator in summary:
                structure_score += 0.2
        
        structure_score = min(structure_score, 1.0)
        
        # Bonus for good concept extraction
        concept_quality = min(len(consolidated.key_concepts) / 8, 1.0)
        
        return (structure_score * 0.7) + (concept_quality * 0.3)
    
    def _score_structural_quality_from_doc(self, parsed_doc: ParsedDocument) -> float:
        """Score structural quality from parsed document."""
        stats = parsed_doc.parsing_stats
        content_types = stats.get('content_types', {})
        
        # Good structure has balanced content types
        if len(content_types) <= 1:
            return 0.3  # Low score for single content type
        
        # Calculate balance
        total_blocks = sum(content_types.values())
        if total_blocks == 0:
            return 0.0
        
        # Prefer documents with good mix of content types
        text_ratio = content_types.get('text', 0) / total_blocks
        
        # Ideal text ratio is 40-70%
        if 0.4 <= text_ratio <= 0.7:
            balance_score = 1.0
        else:
            balance_score = 1.0 - abs(text_ratio - 0.55) * 2  # Penalty for imbalance
        
        return max(balance_score, 0.1)  # Minimum score of 0.1
    
    def _score_multimodal_integration(self, consolidated: ConsolidatedKnowledge) -> float:
        """Score how well different modalities are integrated."""
        metadata = consolidated.enriched_metadata
        
        # Use pre-calculated multimodal richness
        richness = metadata.get('multimodal_richness', 0.0)
        
        # Bonus for content attribution diversity
        attribution_types = len(consolidated.content_attribution)
        attribution_bonus = min(attribution_types * 0.2, 0.6)
        
        return min(richness + attribution_bonus, 1.0)
    
    def _score_multimodal_integration_from_groups(self, content_groups: Dict[str, List[MultimodalContent]]) -> float:
        """Score multimodal integration from content groups."""
        present_types = [content_type for content_type, blocks in content_groups.items() if blocks]
        
        if len(present_types) <= 1:
            return 0.2  # Low score for single modality
        
        # Higher score for more modalities
        integration_score = len(present_types) / 4  # 4 possible types
        
        # Bonus for complementary content (e.g., code with text explanation)
        complementary_bonus = 0.0
        if 'code' in present_types and 'text' in present_types:
            complementary_bonus += 0.2
        if 'table' in present_types and 'text' in present_types:
            complementary_bonus += 0.2
        if 'image' in present_types and 'text' in present_types:
            complementary_bonus += 0.1
        
        return min(integration_score + complementary_bonus, 1.0)
    
    def _score_novelty_potential(self, consolidated: ConsolidatedKnowledge) -> float:
        """Score potential for novel learning."""
        # This is a simplified heuristic - could be enhanced with more sophisticated analysis
        
        # More key concepts suggest more learning potential
        concept_score = min(len(consolidated.key_concepts) / 10, 1.0)
        
        # Longer, more detailed summaries suggest more learning potential
        summary_score = min(len(consolidated.summary) / 1500, 1.0)
        
        # Technical content has higher novelty potential
        metadata = consolidated.enriched_metadata
        technical_ratio = metadata.get('technical_content_ratio', 0.0)
        
        return (concept_score * 0.4) + (summary_score * 0.3) + (technical_ratio * 0.3)
    
    def _score_novelty_potential_from_doc(self, parsed_doc: ParsedDocument) -> float:
        """Score novelty potential from parsed document."""
        stats = parsed_doc.parsing_stats
        
        # More content blocks suggest more learning potential
        block_score = min(stats.get('total_blocks', 0) / 20, 1.0)
        
        # Technical content has higher novelty potential
        code_blocks = stats.get('total_code_blocks', 0)
        tables = stats.get('total_tables', 0)
        technical_score = min((code_blocks + tables) / 10, 1.0)
        
        # File size as proxy for content richness
        file_size = parsed_doc.document_metadata.get('file_size', 0)
        size_score = min(file_size / 100000, 1.0)  # Normalize to 100KB
        
        return (block_score * 0.4) + (technical_score * 0.4) + (size_score * 0.2)
    
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
    
    def _determine_priority_level(self, overall_score: float) -> str:
        """Determine priority level based on overall score."""
        if overall_score >= 0.75:
            return 'high'
        elif overall_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _generate_score_explanation(self, component_scores: Dict[str, float], overall_score: float) -> str:
        """Generate a human-readable explanation of the score."""
        explanations = []
        
        # Find strongest and weakest components
        sorted_components = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_components[0]
        weakest = sorted_components[-1]
        
        explanations.append(f"Overall enrichment score: {overall_score:.2f}")
        explanations.append(f"Strongest aspect: {strongest[0]} ({strongest[1]:.2f})")
        explanations.append(f"Weakest aspect: {weakest[0]} ({weakest[1]:.2f})")
        
        # Add specific insights
        if component_scores['content_diversity'] > 0.8:
            explanations.append("Excellent content diversity across multiple modalities")
        elif component_scores['content_diversity'] < 0.3:
            explanations.append("Limited content diversity - mostly single modality")
        
        if component_scores['technical_depth'] > 0.7:
            explanations.append("High technical depth with code and/or complex tables")
        elif component_scores['technical_depth'] < 0.2:
            explanations.append("Low technical content - primarily textual")
        
        return "; ".join(explanations)
    
    def _create_default_score(self) -> EnrichmentScore:
        """Create a default score for error cases."""
        return EnrichmentScore(
            overall_score=0.5,
            component_scores={component: 0.5 for component in self.scoring_weights.keys()},
            score_explanation="Default score due to processing error",
            priority_level='medium'
        )

# Global scorer instance
_enrichment_scorer = None

def get_enrichment_scorer() -> MultimodalEnrichmentScorer:
    """Get or create a global enrichment scorer instance."""
    global _enrichment_scorer
    
    if _enrichment_scorer is None:
        _enrichment_scorer = MultimodalEnrichmentScorer()
    
    return _enrichment_scorer
