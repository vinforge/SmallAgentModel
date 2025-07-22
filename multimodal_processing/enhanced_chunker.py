#!/usr/bin/env python3
"""
Enhanced Memory Chunking Strategy for SAM
Implements intelligent detection of lists, bullet points, and structured content
with priority weighting for capability extraction and SBIR/proposal writing.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import dimension probing v2
try:
    from .dimension_prober_v2 import EnhancedDimensionProberV2, DimensionScoresV2
    from .dimension_explainer import DimensionExplainer
    DIMENSION_PROBING_V2_AVAILABLE = True
except ImportError:
    DIMENSION_PROBING_V2_AVAILABLE = False
    logging.warning("Dimension probing v2 not available. Enhanced chunker will work without conceptual dimensions.")

# Fallback to v1 if v2 not available
if not DIMENSION_PROBING_V2_AVAILABLE:
    try:
        from .dimension_prober import DimensionProber, DimensionScores
        DIMENSION_PROBING_V1_AVAILABLE = True
    except ImportError:
        DIMENSION_PROBING_V1_AVAILABLE = False

logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Types of content chunks with priority levels."""
    NARRATIVE = "narrative"           # Regular paragraphs (priority: 1.0)
    BULLET_LIST = "bullet_list"       # Bullet points (priority: 1.5)
    NUMBERED_LIST = "numbered_list"   # Numbered lists (priority: 1.6)
    CAPABILITY = "capability"         # Detected capabilities (priority: 2.0)
    REQUIREMENT = "requirement"       # Requirements/specs (priority: 1.8)
    HEADER = "header"                 # Section headers (priority: 1.4)
    TABLE_ROW = "table_row"          # Table content (priority: 1.3)
    CODE_BLOCK = "code_block"        # Code/technical (priority: 1.2)

@dataclass
class EnhancedChunk:
    """Enhanced chunk with structure detection and priority scoring."""
    content: str
    chunk_type: ChunkType
    priority_score: float
    metadata: Dict[str, Any]
    source_location: str
    list_level: int = 0  # For hierarchical lists
    parent_chunk: Optional[str] = None  # For list items under headers
    structured_tags: List[str] = None  # For capability tagging

    # New fields for advanced chunking strategies
    section_title: Optional[str] = None  # Title + Body fusion
    hierarchy_level: int = 0  # Document → Section → Paragraph → Sentence
    parent_section_id: Optional[str] = None  # Hierarchical linking
    overlap_content: Optional[str] = None  # Overlapping window content
    embedding_prefix: Optional[str] = None  # Task-specific embedding prefix
    page_number: Optional[int] = None  # Source page tracking
    section_name: Optional[str] = None  # Section identification

    # NEW: Conceptual dimension scores (v2 - profile-aware)
    dimension_scores: Dict[str, float] = None  # Dynamic dimensions based on profile
    dimension_profile: str = "general"  # Active profile used for scoring
    dimension_confidence: Dict[str, float] = None
    dimension_reasoning: Dict[str, str] = None
    dimension_summary: str = ""
    dimension_explanation: str = ""  # Human-readable explanation

    # Legacy v1 fields (for backward compatibility)
    danger_score: float = 0.0
    complexity_score: float = 0.0
    utility_score: float = 0.0
    sensitivity_score: float = 0.0
    moral_weight_score: float = 0.0
    classification_level: float = 0.0
    itar_sensitivity: float = 0.0

    # NEW: Table processing metadata fields (Task 23 - Phase 1)
    is_table_part: bool = False
    table_id: Optional[str] = None
    table_title: Optional[str] = None          # Table caption/title
    cell_role: Optional[str] = None            # HEADER, DATA, FORMULA, etc.
    cell_coordinates: Optional[Tuple[int, int]] = None  # (row, col)
    cell_data_type: Optional[str] = None       # number, text, date, currency
    table_structure: Optional[Dict] = None     # Overall table metadata
    confidence_score: Optional[float] = None   # Classification confidence
    table_context: Optional[str] = None        # Surrounding document context
    operational_impact: float = 0.0
    innovation_potential: float = 0.0
    technical_readiness: float = 0.0

    def __post_init__(self):
        if self.structured_tags is None:
            self.structured_tags = []
        if self.dimension_scores is None:
            self.dimension_scores = {}
        if self.dimension_confidence is None:
            self.dimension_confidence = {}
        if self.dimension_reasoning is None:
            self.dimension_reasoning = {}

    def get_dimension_score(self, dimension: str) -> float:
        """Get score for a specific dimension."""
        return self.dimension_scores.get(dimension, 0.0)

    def get_high_dimensions(self, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Get dimensions above threshold."""
        return [(dim, score) for dim, score in self.dimension_scores.items() if score > threshold]

class EnhancedChunker:
    """Enhanced chunking strategy with intelligent list detection and advanced strategies."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150,
                 enable_dimension_probing: bool = True, dimension_profile: str = "general"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dimension_profile = dimension_profile
        self.enable_dimension_probing = enable_dimension_probing

        # Initialize dimension prober v2 if available
        if self.enable_dimension_probing and DIMENSION_PROBING_V2_AVAILABLE:
            try:
                self.dimension_prober = EnhancedDimensionProberV2()
                self.dimension_prober.set_profile(dimension_profile)
                self.dimension_explainer = DimensionExplainer()
                self.probing_version = "v2"
                logging.info(f"Dimension probing v2 enabled with profile: {dimension_profile}")
            except Exception as e:
                logging.warning(f"Failed to initialize dimension prober v2: {e}")
                self.enable_dimension_probing = False
                self.dimension_prober = None
                self.dimension_explainer = None
        elif self.enable_dimension_probing and DIMENSION_PROBING_V1_AVAILABLE:
            try:
                from .dimension_prober import DimensionProber
                self.dimension_prober = DimensionProber()
                self.dimension_explainer = None
                self.probing_version = "v1"
                logging.info("Dimension probing v1 enabled (fallback)")
            except Exception as e:
                logging.warning(f"Failed to initialize dimension prober v1: {e}")
                self.enable_dimension_probing = False
                self.dimension_prober = None
                self.dimension_explainer = None
        else:
            self.dimension_prober = None
            self.dimension_explainer = None
            self.probing_version = None

        self.capability_patterns = [
            # Defense/cyber capability patterns
            r'(?i)\b(?:remote|cyber|access|control|reconnaissance|execution|escalation|persistence)\b',
            r'(?i)\b(?:payload|exploit|infiltration|wireless|visualization|sensing)\b',
            r'(?i)\b(?:capability|requirement|specification|objective|goal)\b',
            r'(?i)\b(?:develop|implement|provide|support|enable|deliver)\b',
        ]

        self.requirement_indicators = [
            r'(?i)\b(?:shall|must|will|should|required|mandatory|essential)\b',
            r'(?i)\b(?:specification|requirement|criteria|standard)\b',
            r'(?i)\breq\s*\d+|requirement\s*\d+',
        ]

        self.list_patterns = {
            'bullet': [
                r'^\s*[•·▪▫‣⁃]\s+',  # Unicode bullets
                r'^\s*[-*+]\s+',      # ASCII bullets
                r'^\s*[◦○●]\s+',      # Circle bullets
            ],
            'numbered': [
                r'^\s*\d+\.\s+',      # 1. 2. 3.
                r'^\s*\d+\)\s+',      # 1) 2) 3)
                r'^\s*\(\d+\)\s+',    # (1) (2) (3)
                r'^\s*[a-z]\.\s+',    # a. b. c.
                r'^\s*[a-z]\)\s+',    # a) b) c)
                r'^\s*[ivx]+\.\s+',   # i. ii. iii. (roman)
            ],
            'header': [
                r'^\s*#{1,6}\s+',     # Markdown headers
                r'^[A-Z][A-Z\s]{2,}:?\s*$',  # ALL CAPS headers
                r'^\d+\.\d+\s+',      # 1.1 1.2 section numbers
                r'(?i)^(?:abstract|introduction|conclusion|objective|background|methodology|results|discussion|references)[:.]?\s*$',
            ]
        }

        # Advanced chunking patterns
        self.section_patterns = [
            r'(?i)^(?:section|chapter)\s+\d+',
            r'(?i)^(?:abstract|introduction|conclusion|objective|background)',
            r'(?i)^(?:methodology|results|discussion|references|appendix)',
            r'(?i)^(?:requirements?|capabilities?|specifications?)',
        ]

        # Embedding prefixes for different content types
        self.embedding_prefixes = {
            ChunkType.CAPABILITY: "Instruction: This is a cybersecurity capability requirement chunk.\nContent: ",
            ChunkType.REQUIREMENT: "Instruction: This is a technical requirement specification chunk.\nContent: ",
            ChunkType.HEADER: "Instruction: This is a document section header chunk.\nContent: ",
            ChunkType.BULLET_LIST: "Instruction: This is a structured list of items chunk.\nContent: ",
            ChunkType.NUMBERED_LIST: "Instruction: This is a numbered requirement or procedure chunk.\nContent: ",
            ChunkType.NARRATIVE: "Instruction: This is a descriptive text content chunk.\nContent: ",
        }

    def enhanced_chunk_text(self, text: str, source_location: str) -> List[EnhancedChunk]:
        """
        Enhanced text chunking with intelligent list detection and priority scoring.
        
        Args:
            text: Raw text content to chunk
            source_location: Source location identifier
            
        Returns:
            List of enhanced chunks with structure detection
        """
        chunks = []
        
        # Split text into lines for line-by-line analysis
        lines = text.split('\n')
        current_chunk = ""
        current_type = ChunkType.NARRATIVE
        current_metadata = {}
        list_level = 0
        chunk_counter = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Skip empty lines but use them as chunk boundaries
            if not line.strip():
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        current_chunk.strip(), current_type, current_metadata,
                        source_location, chunk_counter, list_level
                    ))
                    chunk_counter += 1
                    current_chunk = ""
                    current_type = ChunkType.NARRATIVE
                    current_metadata = {}
                    list_level = 0
                i += 1
                continue
            
            # Detect chunk type and handle accordingly
            detected_type, metadata, level = self._detect_chunk_type(line)
            
            # Handle list continuation vs. new chunk
            if detected_type in [ChunkType.BULLET_LIST, ChunkType.NUMBERED_LIST]:
                # If we're already in a different type, finalize current chunk
                if current_chunk.strip() and current_type != detected_type:
                    chunks.append(self._create_chunk(
                        current_chunk.strip(), current_type, current_metadata,
                        source_location, chunk_counter, list_level
                    ))
                    chunk_counter += 1
                    current_chunk = ""
                
                # Start or continue list
                current_type = detected_type
                current_metadata.update(metadata)
                list_level = level
                
                # Collect multi-line list item
                list_item = self._collect_list_item(lines, i)
                current_chunk += list_item + "\n"
                
                # Skip lines that were part of this list item
                i += list_item.count('\n')
                
            elif detected_type == ChunkType.HEADER:
                # Headers are standalone chunks
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        current_chunk.strip(), current_type, current_metadata,
                        source_location, chunk_counter, list_level
                    ))
                    chunk_counter += 1
                
                chunks.append(self._create_chunk(
                    line, detected_type, metadata,
                    source_location, chunk_counter, 0
                ))
                chunk_counter += 1
                current_chunk = ""
                current_type = ChunkType.NARRATIVE
                current_metadata = {}
                list_level = 0
                
            else:
                # Regular narrative text
                if current_type in [ChunkType.BULLET_LIST, ChunkType.NUMBERED_LIST]:
                    # Finalize list chunk
                    chunks.append(self._create_chunk(
                        current_chunk.strip(), current_type, current_metadata,
                        source_location, chunk_counter, list_level
                    ))
                    chunk_counter += 1
                    current_chunk = ""
                    current_type = ChunkType.NARRATIVE
                    current_metadata = {}
                    list_level = 0
                
                current_chunk += line + "\n"
            
            i += 1
        
        # Handle final chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk.strip(), current_type, current_metadata,
                source_location, chunk_counter, list_level
            ))
        
        # Post-process for capability detection and tagging
        chunks = self._post_process_capabilities(chunks)

        # Apply dimension probing if enabled
        if self.enable_dimension_probing:
            chunks = self._apply_dimension_probing(chunks)

        return chunks

    def _detect_chunk_type(self, line: str) -> Tuple[ChunkType, Dict[str, Any], int]:
        """Detect the type of content chunk from a line."""
        metadata = {}
        level = 0
        
        # Check for headers first
        for pattern in self.list_patterns['header']:
            if re.match(pattern, line):
                metadata['header_level'] = len(re.match(r'^\s*#+', line).group()) if line.strip().startswith('#') else 1
                return ChunkType.HEADER, metadata, 0
        
        # Check for numbered lists
        for pattern in self.list_patterns['numbered']:
            match = re.match(pattern, line)
            if match:
                metadata['list_marker'] = match.group().strip()
                metadata['list_type'] = 'numbered'
                level = len(re.match(r'^\s*', line).group())  # Indentation level
                return ChunkType.NUMBERED_LIST, metadata, level
        
        # Check for bullet lists
        for pattern in self.list_patterns['bullet']:
            match = re.match(pattern, line)
            if match:
                metadata['list_marker'] = match.group().strip()
                metadata['list_type'] = 'bullet'
                level = len(re.match(r'^\s*', line).group())  # Indentation level
                return ChunkType.BULLET_LIST, metadata, level
        
        return ChunkType.NARRATIVE, metadata, 0

    def _collect_list_item(self, lines: List[str], start_idx: int) -> str:
        """Collect a complete list item that may span multiple lines."""
        item_lines = [lines[start_idx]]
        
        # Look ahead for continuation lines
        i = start_idx + 1
        while i < len(lines):
            line = lines[i]
            
            # Empty line ends the item
            if not line.strip():
                break
            
            # New list item or header ends current item
            if (self._detect_chunk_type(line)[0] in 
                [ChunkType.BULLET_LIST, ChunkType.NUMBERED_LIST, ChunkType.HEADER]):
                break
            
            # Indented continuation line
            if line.startswith('  ') or line.startswith('\t'):
                item_lines.append(line)
                i += 1
            else:
                break
        
        return '\n'.join(item_lines)

    def _create_chunk(self, content: str, chunk_type: ChunkType, metadata: Dict[str, Any],
                     source_location: str, chunk_id: int, list_level: int) -> EnhancedChunk:
        """Create an enhanced chunk with priority scoring."""
        
        # Calculate priority score based on type and content
        priority_score = self._calculate_priority_score(content, chunk_type)
        
        # Add chunk metadata
        metadata.update({
            'chunk_id': chunk_id,
            'word_count': len(content.split()),
            'char_count': len(content),
            'has_capabilities': self._contains_capabilities(content),
            'has_requirements': self._contains_requirements(content),
        })
        
        return EnhancedChunk(
            content=content,
            chunk_type=chunk_type,
            priority_score=priority_score,
            metadata=metadata,
            source_location=f"{source_location}_chunk_{chunk_id}",
            list_level=list_level,
            structured_tags=[]
        )

    def _calculate_priority_score(self, content: str, chunk_type: ChunkType) -> float:
        """Calculate priority score for a chunk."""
        base_score = {
            ChunkType.NARRATIVE: 1.0,
            ChunkType.BULLET_LIST: 1.5,
            ChunkType.NUMBERED_LIST: 1.6,
            ChunkType.CAPABILITY: 2.0,
            ChunkType.REQUIREMENT: 1.8,
            ChunkType.HEADER: 1.4,
            ChunkType.TABLE_ROW: 1.3,
            ChunkType.CODE_BLOCK: 1.2,
        }[chunk_type]
        
        # Boost score for capability-related content
        if self._contains_capabilities(content):
            base_score *= 1.3
        
        # Boost score for requirement-related content
        if self._contains_requirements(content):
            base_score *= 1.2
        
        # Boost score for technical terms
        technical_terms = ['system', 'protocol', 'interface', 'architecture', 'implementation']
        technical_count = sum(1 for term in technical_terms if term.lower() in content.lower())
        base_score *= (1 + technical_count * 0.1)
        
        return min(base_score, 3.0)  # Cap at 3.0

    def _contains_capabilities(self, content: str) -> bool:
        """Check if content contains capability-related terms."""
        for pattern in self.capability_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _contains_requirements(self, content: str) -> bool:
        """Check if content contains requirement-related terms."""
        for pattern in self.requirement_indicators:
            if re.search(pattern, content):
                return True
        return False

    def _post_process_capabilities(self, chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """Post-process chunks to identify and tag capabilities."""
        
        for chunk in chunks:
            if chunk.chunk_type in [ChunkType.BULLET_LIST, ChunkType.NUMBERED_LIST]:
                # Extract individual capabilities from lists
                capabilities = self._extract_capabilities_from_list(chunk.content)
                if capabilities:
                    chunk.chunk_type = ChunkType.CAPABILITY
                    chunk.priority_score = max(chunk.priority_score, 2.0)
                    chunk.structured_tags = [f"[Cap {i+1}]" for i in range(len(capabilities))]
                    chunk.metadata['extracted_capabilities'] = capabilities
        
        return chunks

    def _extract_capabilities_from_list(self, content: str) -> List[str]:
        """Extract individual capabilities from list content."""
        capabilities = []
        lines = content.split('\n')
        
        for line in lines:
            # Remove list markers
            clean_line = re.sub(r'^\s*[•·▪▫‣⁃\-*+\d+\.\)\(]+\s*', '', line).strip()
            
            if clean_line and self._contains_capabilities(clean_line):
                capabilities.append(clean_line)
        
        return capabilities

    def _apply_dimension_probing(self, chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """Apply conceptual dimension probing to chunks (v2 with profile awareness)."""
        if not self.dimension_prober:
            return chunks

        for chunk in chunks:
            try:
                # Create context for dimension probing
                context = {
                    'chunk_type': chunk.chunk_type.value,
                    'source_location': chunk.source_location,
                    'section_title': chunk.section_title,
                    'has_capabilities': chunk.metadata.get('has_capabilities', False),
                    'has_requirements': chunk.metadata.get('has_requirements', False)
                }

                if self.probing_version == "v2":
                    # Use v2 dimension probing with profiles
                    probe_result = self.dimension_prober.probe_chunk(
                        chunk.content, context, profile=self.dimension_profile
                    )

                    # Apply v2 dimension scores
                    chunk.dimension_scores = probe_result.scores.scores
                    chunk.dimension_profile = probe_result.scores.profile
                    chunk.dimension_confidence = probe_result.scores.confidence
                    chunk.dimension_reasoning = probe_result.scores.reasoning
                    chunk.dimension_summary = self.dimension_prober.get_dimension_summary(probe_result.scores)

                    # Generate human-readable explanation
                    if self.dimension_explainer:
                        profile_config = self.dimension_prober.profile_manager.get_profile(self.dimension_profile)
                        explanation_result = self.dimension_explainer.explain_scores(
                            probe_result.scores, chunk.content, profile_config
                        )
                        chunk.dimension_explanation = explanation_result.summary

                    # Update legacy fields for backward compatibility
                    self._update_legacy_dimension_fields(chunk, probe_result.scores)

                    # Calculate priority boost using v2 method
                    dimension_boost = self.dimension_prober.calculate_priority_boost(probe_result.scores)

                    # Update metadata with v2 information
                    chunk.metadata.update({
                        'dimension_probe_method': probe_result.scores.processing_method,
                        'dimension_processing_time_ms': probe_result.processing_time_ms,
                        'dimension_profile': probe_result.scores.profile,
                        'dimension_summary': chunk.dimension_summary,
                        'dimension_explanation': chunk.dimension_explanation,
                        'high_dimensions': [dim for dim, score in chunk.dimension_scores.items() if score > 0.6]
                    })

                else:
                    # Use v1 dimension probing (fallback)
                    probe_result = self.dimension_prober.probe_chunk(chunk.content, context)
                    scores = probe_result.scores

                    # Apply v1 dimension scores to legacy fields
                    chunk.danger_score = scores.danger
                    chunk.complexity_score = scores.complexity
                    chunk.utility_score = scores.utility
                    chunk.sensitivity_score = scores.sensitivity
                    chunk.moral_weight_score = scores.moral_weight
                    chunk.classification_level = scores.classification_level
                    chunk.itar_sensitivity = scores.itar_sensitivity
                    chunk.operational_impact = scores.operational_impact
                    chunk.innovation_potential = scores.innovation_potential
                    chunk.technical_readiness = scores.technical_readiness

                    # Store v1 metadata
                    chunk.dimension_confidence = scores.confidence
                    chunk.dimension_reasoning = probe_result.reasoning
                    chunk.dimension_summary = self.dimension_prober.get_dimension_summary(scores)

                    # Calculate priority boost using v1 method
                    dimension_boost = self._calculate_dimension_priority_boost_v1(scores)

                    # Update metadata with v1 information
                    chunk.metadata.update({
                        'dimension_probe_method': probe_result.probe_method,
                        'dimension_processing_time_ms': probe_result.processing_time_ms,
                        'dimension_summary': chunk.dimension_summary,
                        'high_danger': scores.danger > 0.6,
                        'high_complexity': scores.complexity > 0.6,
                        'high_sensitivity': scores.sensitivity > 0.6,
                        'high_innovation': scores.innovation_potential > 0.6
                    })

                # Apply priority boost
                chunk.priority_score = min(3.0, chunk.priority_score * dimension_boost)

            except Exception as e:
                logging.warning(f"Failed to apply dimension probing to chunk: {e}")
                # Continue without dimension scores

        return chunks

    def _update_legacy_dimension_fields(self, chunk: EnhancedChunk, scores_v2: 'DimensionScoresV2'):
        """Update legacy dimension fields for backward compatibility."""
        # Map v2 scores to v1 fields where possible
        chunk.utility_score = scores_v2.get_score('utility')
        chunk.complexity_score = scores_v2.get_score('complexity') or scores_v2.get_score('technical_depth')
        chunk.sensitivity_score = scores_v2.get_score('sensitivity') or scores_v2.get_score('compliance_risk')

        # Map profile-specific dimensions to legacy fields
        if scores_v2.profile == "researcher":
            chunk.innovation_potential = scores_v2.get_score('novelty')
            chunk.technical_readiness = scores_v2.get_score('methodology')
        elif scores_v2.profile == "business":
            chunk.operational_impact = scores_v2.get_score('market_impact')
            chunk.innovation_potential = scores_v2.get_score('roi_potential')
        elif scores_v2.profile == "legal":
            chunk.sensitivity_score = max(chunk.sensitivity_score, scores_v2.get_score('liability'))
            chunk.operational_impact = scores_v2.get_score('contractual_impact')

    def _calculate_dimension_priority_boost_v1(self, scores: 'DimensionScores') -> float:
        """Calculate priority boost based on v1 dimension scores."""
        boost = 1.0

        # High-value dimensions increase priority
        if scores.danger > 0.6:
            boost *= 1.2  # Security-critical content
        if scores.complexity > 0.6:
            boost *= 1.15  # Technical complexity
        if scores.utility > 0.6:
            boost *= 1.1  # High utility
        if scores.innovation_potential > 0.6:
            boost *= 1.25  # Innovation value for SBIR
        if scores.operational_impact > 0.6:
            boost *= 1.2  # Mission-critical content

        return min(boost, 1.5)  # Cap boost at 1.5x

    def set_dimension_profile(self, profile: str):
        """Change the dimension profile for future chunking operations."""
        if self.dimension_prober and self.probing_version == "v2":
            self.dimension_profile = profile
            self.dimension_prober.set_profile(profile)
            logging.info(f"Dimension profile changed to: {profile}")
        else:
            logging.warning("Profile switching only available with dimension probing v2")

    def get_available_profiles(self) -> List[str]:
        """Get list of available dimension profiles."""
        if self.dimension_prober and self.probing_version == "v2":
            return self.dimension_prober.profile_manager.list_profiles()
        else:
            return []

    def get_profile_description(self, profile: str) -> str:
        """Get description of a dimension profile."""
        if self.dimension_prober and self.probing_version == "v2":
            return self.dimension_prober.profile_manager.get_profile_description(profile)
        else:
            return f"Profile: {profile} (description not available)"

    def hierarchical_chunk_text(self, text: str, source_location: str, page_number: int = None) -> List[EnhancedChunk]:
        """
        Advanced hierarchical chunking with title+body fusion and overlapping windows.
        Implements strategies from steps2.md for improved document QA accuracy.
        """
        chunks = []

        # First pass: Identify document structure
        document_structure = self._analyze_document_structure(text)

        # Second pass: Create hierarchical chunks
        sections = self._extract_sections_with_titles(text, document_structure)

        for section_idx, section in enumerate(sections):
            section_chunks = self._create_section_chunks(
                section, source_location, section_idx, page_number
            )
            chunks.extend(section_chunks)

        # Third pass: Apply overlapping window strategy
        chunks = self._apply_overlapping_windows(chunks)

        # Fourth pass: Add embedding prefixes
        chunks = self._add_embedding_prefixes(chunks)

        # Fifth pass: Apply dimension probing if enabled
        if self.enable_dimension_probing:
            chunks = self._apply_dimension_probing(chunks)

        return chunks

    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure to identify sections, headers, and hierarchy."""
        lines = text.split('\n')
        structure = {
            'sections': [],
            'headers': [],
            'lists': [],
            'tables': []
        }

        current_section = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Detect section headers
            if self._is_section_header(line):
                if current_section:
                    current_section['end_line'] = i - 1
                    structure['sections'].append(current_section)

                current_section = {
                    'title': line_stripped,
                    'start_line': i,
                    'level': self._get_header_level(line),
                    'type': self._classify_section_type(line_stripped)
                }
                structure['headers'].append({
                    'text': line_stripped,
                    'line': i,
                    'level': self._get_header_level(line)
                })

            # Detect lists
            elif self._detect_chunk_type(line)[0] in [ChunkType.BULLET_LIST, ChunkType.NUMBERED_LIST]:
                structure['lists'].append({
                    'line': i,
                    'type': self._detect_chunk_type(line)[0],
                    'content': line_stripped
                })

        # Close final section
        if current_section:
            current_section['end_line'] = len(lines) - 1
            structure['sections'].append(current_section)

        return structure

    def _extract_sections_with_titles(self, text: str, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sections with title+body fusion strategy."""
        lines = text.split('\n')
        sections = []

        for section_info in structure['sections']:
            start_line = section_info['start_line']
            end_line = section_info['end_line']

            # Title + Body fusion
            title = section_info['title']
            body_lines = lines[start_line + 1:end_line + 1]
            body = '\n'.join(body_lines).strip()

            # Combine title with body for better context
            if body:
                fused_content = f"{title}\n\n{body}"
            else:
                fused_content = title

            sections.append({
                'title': title,
                'content': fused_content,
                'body': body,
                'level': section_info['level'],
                'type': section_info['type'],
                'start_line': start_line,
                'end_line': end_line
            })

        return sections

    def _create_section_chunks(self, section: Dict[str, Any], source_location: str,
                              section_idx: int, page_number: int = None) -> List[EnhancedChunk]:
        """Create chunks for a section with hierarchical metadata."""
        chunks = []
        content = section['content']

        # If section is small enough, create single chunk
        if len(content) <= self.chunk_size:
            chunk = self._create_advanced_chunk(
                content=content,
                chunk_type=ChunkType.HEADER if section['type'] == 'header' else ChunkType.NARRATIVE,
                source_location=f"{source_location}_section_{section_idx}",
                section_title=section['title'],
                hierarchy_level=section['level'],
                page_number=page_number,
                section_name=section['title'],
                chunk_id=0
            )
            chunks.append(chunk)
        else:
            # Split large sections into smaller chunks while preserving context
            section_chunks = self._split_large_section(section, source_location, section_idx, page_number)
            chunks.extend(section_chunks)

        return chunks

    def _split_large_section(self, section: Dict[str, Any], source_location: str,
                            section_idx: int, page_number: int = None) -> List[EnhancedChunk]:
        """Split large sections using semantic boundary control."""
        chunks = []
        content = section['content']
        title = section['title']

        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        current_chunk = ""
        chunk_counter = 0

        for para in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + para if current_chunk else para

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunk = self._create_advanced_chunk(
                        content=current_chunk,
                        chunk_type=self._classify_content_type(current_chunk),
                        source_location=f"{source_location}_section_{section_idx}_chunk_{chunk_counter}",
                        section_title=title,
                        hierarchy_level=section['level'] + 1,
                        page_number=page_number,
                        section_name=title,
                        chunk_id=chunk_counter
                    )
                    chunks.append(chunk)
                    chunk_counter += 1

                # Start new chunk with current paragraph
                current_chunk = para

        # Add final chunk
        if current_chunk:
            chunk = self._create_advanced_chunk(
                content=current_chunk,
                chunk_type=self._classify_content_type(current_chunk),
                source_location=f"{source_location}_section_{section_idx}_chunk_{chunk_counter}",
                section_title=title,
                hierarchy_level=section['level'] + 1,
                page_number=page_number,
                section_name=title,
                chunk_id=chunk_counter
            )
            chunks.append(chunk)

        return chunks

    def _apply_overlapping_windows(self, chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """Apply overlapping window strategy with semantic boundary control."""
        if len(chunks) <= 1:
            return chunks

        enhanced_chunks = []

        for i, chunk in enumerate(chunks):
            # Add overlap content from previous and next chunks
            overlap_content = ""

            # Add overlap from previous chunk
            if i > 0 and self.chunk_overlap > 0:
                prev_chunk = chunks[i - 1]
                prev_words = prev_chunk.content.split()
                if len(prev_words) > 0:
                    # Take last N words from previous chunk
                    overlap_words = min(self.chunk_overlap // 4, len(prev_words))
                    prev_overlap = " ".join(prev_words[-overlap_words:])
                    overlap_content = f"[Previous context: {prev_overlap}]\n\n"

            # Add overlap from next chunk
            if i < len(chunks) - 1 and self.chunk_overlap > 0:
                next_chunk = chunks[i + 1]
                next_words = next_chunk.content.split()
                if len(next_words) > 0:
                    # Take first N words from next chunk
                    overlap_words = min(self.chunk_overlap // 4, len(next_words))
                    next_overlap = " ".join(next_words[:overlap_words])
                    overlap_content += f"\n\n[Next context: {next_overlap}]"

            # Create enhanced chunk with overlap
            enhanced_chunk = EnhancedChunk(
                content=chunk.content,
                chunk_type=chunk.chunk_type,
                priority_score=chunk.priority_score,
                metadata=chunk.metadata,
                source_location=chunk.source_location,
                list_level=chunk.list_level,
                parent_chunk=chunk.parent_chunk,
                structured_tags=chunk.structured_tags,
                section_title=chunk.section_title,
                hierarchy_level=chunk.hierarchy_level,
                parent_section_id=chunk.parent_section_id,
                overlap_content=overlap_content if overlap_content.strip() else None,
                embedding_prefix=chunk.embedding_prefix,
                page_number=chunk.page_number,
                section_name=chunk.section_name
            )

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def _add_embedding_prefixes(self, chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """Add task-specific embedding prefixes for improved RAG performance."""
        for chunk in chunks:
            prefix = self.embedding_prefixes.get(chunk.chunk_type, "")
            chunk.embedding_prefix = prefix

        return chunks

    def _create_advanced_chunk(self, content: str, chunk_type: ChunkType, source_location: str,
                              section_title: str = None, hierarchy_level: int = 0,
                              page_number: int = None, section_name: str = None,
                              chunk_id: int = 0) -> EnhancedChunk:
        """Create an advanced chunk with all metadata."""

        # Calculate priority score
        priority_score = self._calculate_priority_score(content, chunk_type)

        # Enhanced metadata
        metadata = {
            'chunk_id': chunk_id,
            'word_count': len(content.split()),
            'char_count': len(content),
            'has_capabilities': self._contains_capabilities(content),
            'has_requirements': self._contains_requirements(content),
            'section_type': self._classify_section_type(section_title) if section_title else 'unknown',
            'hierarchy_level': hierarchy_level,
        }

        if page_number:
            metadata['page_number'] = page_number

        return EnhancedChunk(
            content=content,
            chunk_type=chunk_type,
            priority_score=priority_score,
            metadata=metadata,
            source_location=source_location,
            section_title=section_title,
            hierarchy_level=hierarchy_level,
            page_number=page_number,
            section_name=section_name,
            structured_tags=[]
        )

    def _is_section_header(self, line: str) -> bool:
        """Check if line is a section header."""
        for pattern in self.list_patterns['header']:
            if re.match(pattern, line):
                return True

        for pattern in self.section_patterns:
            if re.match(pattern, line):
                return True

        return False

    def _get_header_level(self, line: str) -> int:
        """Get hierarchical level of header."""
        # Markdown headers
        if line.strip().startswith('#'):
            return len(re.match(r'^#+', line.strip()).group())

        # Section numbers (1.1, 1.2.3, etc.)
        section_match = re.match(r'^\s*(\d+(?:\.\d+)*)', line)
        if section_match:
            return len(section_match.group(1).split('.'))

        # Default level
        return 1

    def _classify_section_type(self, title: str) -> str:
        """Classify section type based on title."""
        if not title:
            return 'unknown'

        title_lower = title.lower()

        if any(word in title_lower for word in ['abstract', 'summary', 'overview']):
            return 'abstract'
        elif any(word in title_lower for word in ['introduction', 'background']):
            return 'introduction'
        elif any(word in title_lower for word in ['conclusion', 'summary', 'results']):
            return 'conclusion'
        elif any(word in title_lower for word in ['objective', 'goal', 'purpose']):
            return 'objective'
        elif any(word in title_lower for word in ['requirement', 'specification', 'criteria']):
            return 'requirement'
        elif any(word in title_lower for word in ['capability', 'feature', 'function']):
            return 'capability'
        elif any(word in title_lower for word in ['methodology', 'approach', 'method']):
            return 'methodology'
        else:
            return 'content'

    def _classify_content_type(self, content: str) -> ChunkType:
        """Classify content type for chunk typing."""
        if self._contains_capabilities(content):
            return ChunkType.CAPABILITY
        elif self._contains_requirements(content):
            return ChunkType.REQUIREMENT
        elif any(re.match(pattern, line) for pattern in self.list_patterns['bullet'] for line in content.split('\n')):
            return ChunkType.BULLET_LIST
        elif any(re.match(pattern, line) for pattern in self.list_patterns['numbered'] for line in content.split('\n')):
            return ChunkType.NUMBERED_LIST
        else:
            return ChunkType.NARRATIVE
