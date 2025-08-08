"""
Multimodal Reasoning Engine for SAM
Blends multiple inputs (text, images, docs, web) for deeper reasoning with cross-modal fusion.

Sprint 9 Task 4: Multimodal Reasoning Engine
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class SourceType(Enum):
    """Types of information sources."""
    TEXT_INPUT = "text_input"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_CAPSULE = "knowledge_capsule"
    LOCAL_FILE = "local_file"

class ConfidenceLevel(Enum):
    """Confidence levels for reasoning."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

@dataclass
class SourceEvidence:
    """Evidence from a specific source."""
    source_id: str
    source_type: SourceType
    content: str
    confidence: float
    relevance_score: float
    attribution: str
    metadata: Dict[str, Any]

@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_id: str
    step_type: str
    description: str
    input_sources: List[str]
    output: str
    confidence: float
    reasoning_trace: str
    metadata: Dict[str, Any]

@dataclass
class MultimodalResponse:
    """Complete multimodal reasoning response."""
    response_id: str
    query: str
    answer: str
    source_summary: Dict[str, Any]
    reasoning_steps: List[ReasoningStep]
    confidence_level: ConfidenceLevel
    overall_confidence: float
    source_attributions: List[str]
    created_at: str
    metadata: Dict[str, Any]

class MultimodalReasoningEngine:
    """
    Advanced reasoning engine that combines multiple input modalities.
    """
    
    def __init__(self, ingestion_engine=None, local_search_engine=None, 
                 web_search_engine=None):
        """
        Initialize the multimodal reasoning engine.
        
        Args:
            ingestion_engine: Multimodal ingestion engine
            local_search_engine: Local file search engine
            web_search_engine: Web search engine
        """
        self.ingestion_engine = ingestion_engine
        self.local_search_engine = local_search_engine
        self.web_search_engine = web_search_engine
        
        # Storage
        self.reasoning_sessions: Dict[str, MultimodalResponse] = {}
        
        # Configuration
        self.config = {
            'max_sources_per_type': 5,
            'min_confidence_threshold': 0.3,
            'enable_cross_modal_fusion': True,
            'enable_source_attribution': True,
            'reasoning_depth': 3,  # Number of reasoning steps
            'confidence_weights': {
                SourceType.TEXT_INPUT: 0.8,
                SourceType.DOCUMENT: 0.9,
                SourceType.WEB_SEARCH: 0.7,
                SourceType.IMAGE: 0.6,
                SourceType.AUDIO: 0.6,
                SourceType.KNOWLEDGE_CAPSULE: 0.85,
                SourceType.LOCAL_FILE: 0.8
            }
        }
        
        logger.info("Multimodal reasoning engine initialized")
    
    def reason_multimodal(self, query: str, user_id: str, session_id: str,
                         text_inputs: List[str] = None,
                         image_ids: List[str] = None,
                         audio_ids: List[str] = None,
                         document_ids: List[str] = None,
                         enable_web_search: bool = False,
                         enable_local_search: bool = True) -> MultimodalResponse:
        """
        Perform multimodal reasoning across all available sources.
        
        Args:
            query: The question or task to reason about
            user_id: User making the request
            session_id: Session ID
            text_inputs: Additional text inputs
            image_ids: IDs of images to include
            audio_ids: IDs of audio files to include
            document_ids: IDs of documents to include
            enable_web_search: Whether to perform web search
            enable_local_search: Whether to search local files
            
        Returns:
            MultimodalResponse with reasoning results
        """
        try:
            response_id = f"reasoning_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Starting multimodal reasoning for: {query[:50]}...")
            
            # Step 1: Gather evidence from all sources
            evidence_sources = self._gather_evidence(
                query=query,
                text_inputs=text_inputs or [],
                image_ids=image_ids or [],
                audio_ids=audio_ids or [],
                document_ids=document_ids or [],
                enable_web_search=enable_web_search,
                enable_local_search=enable_local_search,
                user_id=user_id,
                session_id=session_id
            )
            
            # Step 2: Perform cross-modal fusion
            fused_evidence = self._perform_cross_modal_fusion(evidence_sources)
            
            # Step 3: Generate reasoning steps
            reasoning_steps = self._generate_reasoning_steps(query, fused_evidence)
            
            # Step 4: Synthesize final answer
            answer = self._synthesize_answer(query, fused_evidence, reasoning_steps)
            
            # Step 5: Calculate confidence and attribution
            overall_confidence = self._calculate_overall_confidence(evidence_sources, reasoning_steps)
            confidence_level = self._determine_confidence_level(overall_confidence)
            source_attributions = self._generate_source_attributions(evidence_sources)
            
            # Step 6: Create source summary
            source_summary = self._create_source_summary(evidence_sources)
            
            # Create multimodal response
            response = MultimodalResponse(
                response_id=response_id,
                query=query,
                answer=answer,
                source_summary=source_summary,
                reasoning_steps=reasoning_steps,
                confidence_level=confidence_level,
                overall_confidence=overall_confidence,
                source_attributions=source_attributions,
                created_at=datetime.now().isoformat(),
                metadata={
                    'user_id': user_id,
                    'session_id': session_id,
                    'sources_used': len(evidence_sources),
                    'reasoning_steps': len(reasoning_steps)
                }
            )
            
            self.reasoning_sessions[response_id] = response
            
            logger.info(f"Multimodal reasoning completed: {len(evidence_sources)} sources, "
                       f"confidence: {overall_confidence:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in multimodal reasoning: {e}")
            return self._create_error_response(query, str(e))
    
    def format_response(self, response: MultimodalResponse, 
                       include_reasoning_trace: bool = True,
                       include_source_details: bool = True) -> str:
        """
        Format multimodal response for display.
        
        Args:
            response: MultimodalResponse to format
            include_reasoning_trace: Whether to include reasoning steps
            include_source_details: Whether to include source details
            
        Returns:
            Formatted response string
        """
        try:
            output_parts = []
            
            # Main answer
            output_parts.append(f"# Answer\n\n{response.answer}\n")
            
            # Confidence indicator
            confidence_emoji = {
                ConfidenceLevel.VERY_HIGH: "🟢",
                ConfidenceLevel.HIGH: "🔵", 
                ConfidenceLevel.MEDIUM: "🟡",
                ConfidenceLevel.LOW: "🟠",
                ConfidenceLevel.VERY_LOW: "🔴"
            }.get(response.confidence_level, "⚪")
            
            output_parts.append(f"**Confidence:** {confidence_emoji} {response.confidence_level.value.title()} "
                              f"({response.overall_confidence:.1%})\n")
            
            # Source summary
            if include_source_details and response.source_summary:
                output_parts.append("## Source Summary\n")
                
                for source_type, details in response.source_summary.items():
                    if details['count'] > 0:
                        output_parts.append(f"- **{source_type.title()}**: {details['count']} sources "
                                          f"(avg confidence: {details['avg_confidence']:.1%})\n")
                
                output_parts.append("")
            
            # Source attributions
            if response.source_attributions:
                output_parts.append("## Sources\n")
                for i, attribution in enumerate(response.source_attributions, 1):
                    output_parts.append(f"{i}. {attribution}\n")
                output_parts.append("")
            
            # Reasoning trace
            if include_reasoning_trace and response.reasoning_steps:
                output_parts.append("## Reasoning Process\n")
                
                for i, step in enumerate(response.reasoning_steps, 1):
                    output_parts.append(f"### Step {i}: {step.step_type.title()}\n")
                    output_parts.append(f"{step.description}\n")
                    
                    if step.reasoning_trace:
                        output_parts.append(f"*Reasoning:* {step.reasoning_trace}\n")
                    
                    output_parts.append(f"*Confidence:* {step.confidence:.1%}\n\n")
            
            return "".join(output_parts)
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return f"Error formatting response: {str(e)}"
    
    def _gather_evidence(self, query: str, text_inputs: List[str],
                        image_ids: List[str], audio_ids: List[str],
                        document_ids: List[str], enable_web_search: bool,
                        enable_local_search: bool, user_id: str,
                        session_id: str) -> List[SourceEvidence]:
        """Gather evidence from all available sources."""
        evidence_sources = []
        
        try:
            # Text inputs
            for i, text_input in enumerate(text_inputs):
                if text_input.strip():
                    evidence = SourceEvidence(
                        source_id=f"text_{i}",
                        source_type=SourceType.TEXT_INPUT,
                        content=text_input,
                        confidence=0.9,
                        relevance_score=self._calculate_text_relevance(query, text_input),
                        attribution=f"Text input {i+1}",
                        metadata={'length': len(text_input)}
                    )
                    evidence_sources.append(evidence)
            
            # Multimodal content (images, audio, documents)
            if self.ingestion_engine:
                # Process images
                for image_id in image_ids:
                    result = self.ingestion_engine.get_processing_result(image_id)
                    if result:
                        content_parts = []
                        if result.caption:
                            content_parts.append(f"Caption: {result.caption}")
                        if result.extracted_text:
                            content_parts.append(f"Text: {result.extracted_text}")
                        
                        if content_parts:
                            evidence = SourceEvidence(
                                source_id=image_id,
                                source_type=SourceType.IMAGE,
                                content='\n'.join(content_parts),
                                confidence=max(result.confidence_scores.values()) if result.confidence_scores else 0.5,
                                relevance_score=self._calculate_text_relevance(query, '\n'.join(content_parts)),
                                attribution=f"Image: {self.ingestion_engine.get_media_metadata(image_id).original_filename}",
                                metadata=result.metadata
                            )
                            evidence_sources.append(evidence)
                
                # Process audio
                for audio_id in audio_ids:
                    result = self.ingestion_engine.get_processing_result(audio_id)
                    if result and result.transcription:
                        evidence = SourceEvidence(
                            source_id=audio_id,
                            source_type=SourceType.AUDIO,
                            content=result.transcription,
                            confidence=result.confidence_scores.get('transcription', 0.5),
                            relevance_score=self._calculate_text_relevance(query, result.transcription),
                            attribution=f"Audio: {self.ingestion_engine.get_media_metadata(audio_id).original_filename}",
                            metadata=result.metadata
                        )
                        evidence_sources.append(evidence)
                
                # Process documents
                for doc_id in document_ids:
                    result = self.ingestion_engine.get_processing_result(doc_id)
                    if result and result.extracted_text:
                        evidence = SourceEvidence(
                            source_id=doc_id,
                            source_type=SourceType.DOCUMENT,
                            content=result.extracted_text,
                            confidence=result.confidence_scores.get('extraction', 0.8),
                            relevance_score=self._calculate_text_relevance(query, result.extracted_text),
                            attribution=f"Document: {self.ingestion_engine.get_media_metadata(doc_id).original_filename}",
                            metadata=result.metadata
                        )
                        evidence_sources.append(evidence)
            
            # Local file search
            if enable_local_search and self.local_search_engine:
                local_results = self.local_search_engine.search(query, max_results=5)
                
                for result in local_results:
                    evidence = SourceEvidence(
                        source_id=result.result_id,
                        source_type=SourceType.LOCAL_FILE,
                        content=result.content_preview,
                        confidence=result.confidence_score,
                        relevance_score=result.confidence_score,
                        attribution=self.local_search_engine.create_citation(result),
                        metadata={'filename': result.filename, 'section': result.section}
                    )
                    evidence_sources.append(evidence)
            
            # Web search
            if enable_web_search and self.web_search_engine and self.web_search_engine.enable_web_access:
                web_results, query_id = self.web_search_engine.search(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    max_results=5
                )
                
                for result in web_results:
                    evidence = SourceEvidence(
                        source_id=result.result_id,
                        source_type=SourceType.WEB_SEARCH,
                        content=result.snippet,
                        confidence=result.confidence_score,
                        relevance_score=result.confidence_score,
                        attribution=f"🌐[{result.title}]({result.url})",
                        metadata={'url': result.url, 'title': result.title}
                    )
                    evidence_sources.append(evidence)
            
            # Filter by relevance and confidence
            filtered_evidence = [
                evidence for evidence in evidence_sources
                if evidence.relevance_score >= self.config['min_confidence_threshold']
            ]
            
            # Sort by relevance score
            filtered_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
            
            return filtered_evidence
            
        except Exception as e:
            logger.error(f"Error gathering evidence: {e}")
            return evidence_sources
    
    def _perform_cross_modal_fusion(self, evidence_sources: List[SourceEvidence]) -> List[SourceEvidence]:
        """Perform cross-modal fusion to enhance evidence."""
        try:
            if not self.config['enable_cross_modal_fusion']:
                return evidence_sources
            
            # Group evidence by source type
            evidence_by_type = {}
            for evidence in evidence_sources:
                source_type = evidence.source_type
                if source_type not in evidence_by_type:
                    evidence_by_type[source_type] = []
                evidence_by_type[source_type].append(evidence)
            
            # Apply cross-modal enhancement
            enhanced_evidence = []
            
            for evidence in evidence_sources:
                # Apply source type confidence weighting
                type_weight = self.config['confidence_weights'].get(evidence.source_type, 0.7)
                enhanced_confidence = evidence.confidence * type_weight
                
                # Create enhanced evidence
                enhanced_evidence.append(SourceEvidence(
                    source_id=evidence.source_id,
                    source_type=evidence.source_type,
                    content=evidence.content,
                    confidence=enhanced_confidence,
                    relevance_score=evidence.relevance_score,
                    attribution=evidence.attribution,
                    metadata=evidence.metadata
                ))
            
            return enhanced_evidence
            
        except Exception as e:
            logger.error(f"Error in cross-modal fusion: {e}")
            return evidence_sources
    
    def _generate_reasoning_steps(self, query: str, evidence_sources: List[SourceEvidence]) -> List[ReasoningStep]:
        """Generate reasoning steps based on evidence."""
        try:
            reasoning_steps = []
            
            # Step 1: Information gathering
            step1 = ReasoningStep(
                step_id="step_1",
                step_type="information_gathering",
                description="Gathered information from multiple sources",
                input_sources=[e.source_id for e in evidence_sources],
                output=f"Collected {len(evidence_sources)} pieces of evidence from various sources",
                confidence=0.9,
                reasoning_trace=f"Analyzed {len(evidence_sources)} sources including text, documents, and search results",
                metadata={'sources_count': len(evidence_sources)}
            )
            reasoning_steps.append(step1)
            
            # Step 2: Evidence analysis
            high_confidence_sources = [e for e in evidence_sources if e.confidence > 0.7]
            step2 = ReasoningStep(
                step_id="step_2",
                step_type="evidence_analysis",
                description="Analyzed evidence quality and relevance",
                input_sources=[e.source_id for e in high_confidence_sources],
                output=f"Identified {len(high_confidence_sources)} high-confidence sources",
                confidence=0.8,
                reasoning_trace=f"Filtered evidence based on confidence thresholds and relevance scores",
                metadata={'high_confidence_count': len(high_confidence_sources)}
            )
            reasoning_steps.append(step2)
            
            # Step 3: Synthesis
            step3 = ReasoningStep(
                step_id="step_3",
                step_type="synthesis",
                description="Synthesized information to answer the query",
                input_sources=[e.source_id for e in evidence_sources],
                output="Combined evidence from multiple modalities to form comprehensive answer",
                confidence=0.75,
                reasoning_trace="Integrated text, visual, and document evidence using cross-modal reasoning",
                metadata={'synthesis_method': 'cross_modal_fusion'}
            )
            reasoning_steps.append(step3)
            
            return reasoning_steps
            
        except Exception as e:
            logger.error(f"Error generating reasoning steps: {e}")
            return []
    
    def _synthesize_answer(self, query: str, evidence_sources: List[SourceEvidence],
                          reasoning_steps: List[ReasoningStep]) -> str:
        """Synthesize final answer from evidence and reasoning."""
        try:
            if not evidence_sources:
                return f"I don't have enough information to answer '{query}'. Please provide more context or sources."
            
            # Extract key information from evidence
            key_points = []
            for evidence in evidence_sources[:5]:  # Top 5 sources
                if evidence.confidence > 0.5:
                    # Extract key sentences from content
                    sentences = evidence.content.split('.')
                    for sentence in sentences:
                        if len(sentence.strip()) > 20 and any(word in sentence.lower() for word in query.lower().split()):
                            key_points.append(sentence.strip())
                            break
            
            # Create synthesized answer
            if key_points:
                answer_parts = [
                    f"Based on the available evidence from {len(evidence_sources)} sources, here's what I found:",
                    "",
                    *key_points[:3],  # Top 3 key points
                    "",
                    f"This information is drawn from multiple sources including documents, search results, and other inputs with an overall confidence of {self._calculate_overall_confidence(evidence_sources, reasoning_steps):.1%}."
                ]
                
                return '\n'.join(answer_parts)
            else:
                return f"While I found {len(evidence_sources)} relevant sources, I couldn't extract specific information to directly answer '{query}'. The sources may contain relevant information that requires more detailed analysis."
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return f"Error synthesizing answer for '{query}': {str(e)}"
    
    def _calculate_text_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text."""
        try:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(text_words))
            total_query_words = len(query_words)
            
            if total_query_words == 0:
                return 0.0
            
            # Basic relevance score
            relevance = overlap / total_query_words
            
            # Bonus for exact phrase match
            if query.lower() in text.lower():
                relevance += 0.3
            
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Error calculating text relevance: {e}")
            return 0.0
    
    def _calculate_overall_confidence(self, evidence_sources: List[SourceEvidence],
                                    reasoning_steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence score."""
        try:
            if not evidence_sources:
                return 0.0
            
            # Average evidence confidence
            evidence_confidence = sum(e.confidence for e in evidence_sources) / len(evidence_sources)
            
            # Average reasoning confidence
            reasoning_confidence = 0.7  # Default
            if reasoning_steps:
                reasoning_confidence = sum(s.confidence for s in reasoning_steps) / len(reasoning_steps)
            
            # Combine with weights
            overall_confidence = (evidence_confidence * 0.7) + (reasoning_confidence * 0.3)
            
            # Bonus for multiple sources
            if len(evidence_sources) >= 3:
                overall_confidence += 0.1
            
            return min(1.0, overall_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.5
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level from score."""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_source_attributions(self, evidence_sources: List[SourceEvidence]) -> List[str]:
        """Generate source attribution strings."""
        try:
            attributions = []
            
            for evidence in evidence_sources:
                if evidence.confidence > 0.3:  # Only include decent confidence sources
                    attributions.append(evidence.attribution)
            
            return attributions[:10]  # Limit to top 10 sources
            
        except Exception as e:
            logger.error(f"Error generating source attributions: {e}")
            return []
    
    def _create_source_summary(self, evidence_sources: List[SourceEvidence]) -> Dict[str, Any]:
        """Create summary of sources used."""
        try:
            summary = {}
            
            # Group by source type
            for evidence in evidence_sources:
                source_type = evidence.source_type.value
                
                if source_type not in summary:
                    summary[source_type] = {
                        'count': 0,
                        'total_confidence': 0.0,
                        'avg_confidence': 0.0
                    }
                
                summary[source_type]['count'] += 1
                summary[source_type]['total_confidence'] += evidence.confidence
            
            # Calculate averages
            for source_type, data in summary.items():
                if data['count'] > 0:
                    data['avg_confidence'] = data['total_confidence'] / data['count']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating source summary: {e}")
            return {}
    
    def _create_error_response(self, query: str, error_message: str) -> MultimodalResponse:
        """Create an error response."""
        return MultimodalResponse(
            response_id=f"error_{uuid.uuid4().hex[:8]}",
            query=query,
            answer=f"I encountered an error while processing your request: {error_message}",
            source_summary={},
            reasoning_steps=[],
            confidence_level=ConfidenceLevel.VERY_LOW,
            overall_confidence=0.0,
            source_attributions=[],
            created_at=datetime.now().isoformat(),
            metadata={'error': error_message}
        )

# Global multimodal reasoning engine instance
_reasoning_engine = None

def get_reasoning_engine(ingestion_engine=None, local_search_engine=None,
                        web_search_engine=None) -> MultimodalReasoningEngine:
    """Get or create a global multimodal reasoning engine instance."""
    global _reasoning_engine
    
    if _reasoning_engine is None:
        _reasoning_engine = MultimodalReasoningEngine(
            ingestion_engine=ingestion_engine,
            local_search_engine=local_search_engine,
            web_search_engine=web_search_engine
        )
    
    return _reasoning_engine
