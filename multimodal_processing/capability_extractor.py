#!/usr/bin/env python3
"""
Capability Extractor Plugin for SAM
Specialized extraction of defense-related capabilities, requirements, and SBIR-relevant content
with structured formatting for government contractors and proposal writers.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CapabilityType(Enum):
    """Types of capabilities detected."""
    CYBER_OFFENSIVE = "cyber_offensive"
    CYBER_DEFENSIVE = "cyber_defensive"
    RECONNAISSANCE = "reconnaissance"
    COMMUNICATION = "communication"
    INFRASTRUCTURE = "infrastructure"
    TECHNICAL_REQUIREMENT = "technical_requirement"
    OPERATIONAL_REQUIREMENT = "operational_requirement"
    PERFORMANCE_METRIC = "performance_metric"

@dataclass
class ExtractedCapability:
    """Structured capability with metadata."""
    title: str
    description: str
    capability_type: CapabilityType
    priority_level: int  # 1-5, 5 being highest
    structured_tag: str  # e.g., "[Req 1]", "[Cap 2]"
    keywords: List[str]
    source_location: str
    confidence_score: float  # 0.0-1.0

class CapabilityExtractor:
    """Extract and structure defense-related capabilities from documents."""
    
    def __init__(self):
        self.capability_patterns = {
            CapabilityType.CYBER_OFFENSIVE: [
                r'(?i)\b(?:remote\s+(?:code\s+)?execution|RCE)\b',
                r'(?i)\b(?:privilege\s+escalation|privesc)\b',
                r'(?i)\b(?:persistence\s+mechanism|backdoor|implant)\b',
                r'(?i)\b(?:payload|exploit|attack\s+vector)\b',
                r'(?i)\b(?:infiltration|penetration|breach)\b',
                r'(?i)\b(?:deny|disrupt|degrade|destroy)\b.*(?:system|network)',
            ],
            CapabilityType.CYBER_DEFENSIVE: [
                r'(?i)\b(?:threat\s+detection|intrusion\s+detection)\b',
                r'(?i)\b(?:security\s+monitoring|SIEM)\b',
                r'(?i)\b(?:incident\s+response|forensics)\b',
                r'(?i)\b(?:vulnerability\s+assessment|security\s+audit)\b',
                r'(?i)\b(?:access\s+control|authentication|authorization)\b',
            ],
            CapabilityType.RECONNAISSANCE: [
                r'(?i)\b(?:reconnaissance|recon|intelligence\s+gathering)\b',
                r'(?i)\b(?:surveillance|monitoring|observation)\b',
                r'(?i)\b(?:target\s+identification|asset\s+discovery)\b',
                r'(?i)\b(?:network\s+mapping|topology\s+discovery)\b',
                r'(?i)\b(?:signal\s+intelligence|SIGINT|ELINT)\b',
            ],
            CapabilityType.COMMUNICATION: [
                r'(?i)\b(?:command\s+and\s+control|C2|C&C)\b',
                r'(?i)\b(?:communication\s+protocol|messaging)\b',
                r'(?i)\b(?:data\s+exfiltration|data\s+transfer)\b',
                r'(?i)\b(?:covert\s+channel|steganography)\b',
                r'(?i)\b(?:wireless\s+communication|RF\s+communication)\b',
            ],
            CapabilityType.INFRASTRUCTURE: [
                r'(?i)\b(?:infrastructure\s+protection|critical\s+infrastructure)\b',
                r'(?i)\b(?:system\s+hardening|security\s+baseline)\b',
                r'(?i)\b(?:network\s+segmentation|air\s+gap)\b',
                r'(?i)\b(?:backup\s+system|redundancy|failover)\b',
                r'(?i)\b(?:cloud\s+security|virtualization\s+security)\b',
            ],
        }
        
        self.requirement_patterns = [
            r'(?i)\b(?:shall|must|will|should|required|mandatory|essential)\b',
            r'(?i)\b(?:specification|requirement|criteria|standard|metric)\b',
            r'(?i)\b(?:performance|throughput|latency|accuracy|reliability)\b',
            r'(?i)\b(?:compliance|certification|validation|verification)\b',
        ]
        
        self.priority_keywords = {
            5: ['critical', 'essential', 'mandatory', 'required', 'must'],
            4: ['important', 'significant', 'key', 'primary', 'should'],
            3: ['useful', 'beneficial', 'recommended', 'preferred', 'may'],
            2: ['optional', 'nice-to-have', 'could', 'might'],
            1: ['future', 'potential', 'possible', 'consideration']
        }

    def extract_capabilities(self, text: str, source_location: str) -> List[ExtractedCapability]:
        """
        Extract structured capabilities from text content.
        
        Args:
            text: Input text to analyze
            source_location: Source location identifier
            
        Returns:
            List of extracted capabilities with structured formatting
        """
        capabilities = []
        
        # Split text into sentences for analysis
        sentences = self._split_into_sentences(text)
        
        # Extract capabilities from each sentence
        for i, sentence in enumerate(sentences):
            extracted = self._analyze_sentence(sentence, i, source_location)
            capabilities.extend(extracted)
        
        # Extract from list structures
        list_capabilities = self._extract_from_lists(text, source_location)
        capabilities.extend(list_capabilities)
        
        # Remove duplicates and rank by priority
        capabilities = self._deduplicate_and_rank(capabilities)
        
        # Add structured tags
        capabilities = self._add_structured_tags(capabilities)
        
        return capabilities

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for analysis."""
        # Handle bullet points and numbered lists as separate sentences
        lines = text.split('\n')
        sentences = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if it's a list item
            if re.match(r'^\s*[•·▪▫‣⁃\-*+\d+\.\)\(]', line):
                sentences.append(line)
            else:
                # Split regular text by sentence boundaries
                sent_parts = re.split(r'[.!?]+', line)
                for part in sent_parts:
                    if part.strip():
                        sentences.append(part.strip())
        
        return sentences

    def _analyze_sentence(self, sentence: str, index: int, source_location: str) -> List[ExtractedCapability]:
        """Analyze a single sentence for capabilities."""
        capabilities = []
        
        # Check each capability type
        for cap_type, patterns in self.capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence):
                    capability = self._create_capability(
                        sentence, cap_type, index, source_location
                    )
                    if capability:
                        capabilities.append(capability)
                    break  # Only one capability type per sentence
        
        return capabilities

    def _create_capability(self, text: str, cap_type: CapabilityType, 
                          index: int, source_location: str) -> Optional[ExtractedCapability]:
        """Create a structured capability from text."""
        
        # Clean and extract title
        title = self._extract_title(text)
        if not title:
            return None
        
        # Extract description (full sentence/item)
        description = self._clean_text(text)
        
        # Calculate priority level
        priority_level = self._calculate_priority(text)
        
        # Extract keywords
        keywords = self._extract_keywords(text, cap_type)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(text, cap_type)
        
        return ExtractedCapability(
            title=title,
            description=description,
            capability_type=cap_type,
            priority_level=priority_level,
            structured_tag="",  # Will be added later
            keywords=keywords,
            source_location=f"{source_location}_cap_{index}",
            confidence_score=confidence_score
        )

    def _extract_title(self, text: str) -> str:
        """Extract a concise title from the capability text."""
        # Remove list markers
        clean_text = re.sub(r'^\s*[•·▪▫‣⁃\-*+\d+\.\)\(]+\s*', '', text).strip()
        
        # Take first meaningful phrase (up to first comma or semicolon)
        title_match = re.match(r'^([^,;]+)', clean_text)
        if title_match:
            title = title_match.group(1).strip()
            # Limit length
            if len(title) > 60:
                title = title[:57] + "..."
            return title
        
        return clean_text[:60] + "..." if len(clean_text) > 60 else clean_text

    def _clean_text(self, text: str) -> str:
        """Clean text for description."""
        # Remove list markers
        clean_text = re.sub(r'^\s*[•·▪▫‣⁃\-*+\d+\.\)\(]+\s*', '', text).strip()
        
        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text

    def _calculate_priority(self, text: str) -> int:
        """Calculate priority level based on keywords."""
        text_lower = text.lower()
        
        for priority, keywords in self.priority_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return priority
        
        # Default priority based on capability indicators
        if any(re.search(pattern, text) for pattern in self.requirement_patterns):
            return 4  # Requirements are high priority
        
        return 3  # Default medium priority

    def _extract_keywords(self, text: str, cap_type: CapabilityType) -> List[str]:
        """Extract relevant keywords from the text."""
        keywords = []
        
        # Extract technical terms
        tech_pattern = r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b'  # Acronyms
        acronyms = re.findall(tech_pattern, text)
        keywords.extend(acronyms)
        
        # Extract capability-specific terms
        for pattern in self.capability_patterns[cap_type]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend([match.lower() for match in matches if isinstance(match, str)])
        
        # Remove duplicates and limit
        keywords = list(set(keywords))[:5]
        
        return keywords

    def _calculate_confidence(self, text: str, cap_type: CapabilityType) -> float:
        """Calculate confidence score for the capability extraction."""
        score = 0.0
        
        # Base score for pattern match
        pattern_matches = sum(1 for pattern in self.capability_patterns[cap_type] 
                            if re.search(pattern, text))
        score += min(pattern_matches * 0.3, 0.6)
        
        # Bonus for requirement indicators
        req_matches = sum(1 for pattern in self.requirement_patterns 
                         if re.search(pattern, text))
        score += min(req_matches * 0.2, 0.3)
        
        # Bonus for technical terms
        tech_terms = len(re.findall(r'\b[A-Z]{2,}\b', text))
        score += min(tech_terms * 0.05, 0.1)
        
        return min(score, 1.0)

    def _extract_from_lists(self, text: str, source_location: str) -> List[ExtractedCapability]:
        """Extract capabilities specifically from list structures."""
        capabilities = []
        
        # Find list patterns
        list_pattern = r'(?:^|\n)\s*[•·▪▫‣⁃\-*+\d+\.\)\(]+\s*(.+?)(?=\n\s*[•·▪▫‣⁃\-*+\d+\.\)\(]|\n\n|\Z)'
        
        matches = re.finditer(list_pattern, text, re.MULTILINE | re.DOTALL)
        
        for i, match in enumerate(matches):
            item_text = match.group(1).strip()
            if len(item_text) > 10:  # Minimum meaningful length
                extracted = self._analyze_sentence(item_text, i, source_location)
                capabilities.extend(extracted)
        
        return capabilities

    def _deduplicate_and_rank(self, capabilities: List[ExtractedCapability]) -> List[ExtractedCapability]:
        """Remove duplicates and rank by priority and confidence."""
        # Remove near-duplicates based on title similarity
        unique_capabilities = []
        
        for cap in capabilities:
            is_duplicate = False
            for existing in unique_capabilities:
                if self._are_similar(cap.title, existing.title):
                    # Keep the one with higher confidence
                    if cap.confidence_score > existing.confidence_score:
                        unique_capabilities.remove(existing)
                        unique_capabilities.append(cap)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_capabilities.append(cap)
        
        # Sort by priority (descending) then confidence (descending)
        unique_capabilities.sort(
            key=lambda x: (x.priority_level, x.confidence_score), 
            reverse=True
        )
        
        return unique_capabilities

    def _are_similar(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """Check if two texts are similar enough to be considered duplicates."""
        # Simple similarity based on common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

    def _add_structured_tags(self, capabilities: List[ExtractedCapability]) -> List[ExtractedCapability]:
        """Add structured tags like [Req 1], [Cap 2] to capabilities."""
        
        # Group by type for different tag prefixes
        type_counters = {}
        
        for capability in capabilities:
            cap_type = capability.capability_type
            
            if cap_type not in type_counters:
                type_counters[cap_type] = 0
            
            type_counters[cap_type] += 1
            
            # Determine tag prefix based on type
            if 'requirement' in cap_type.value:
                prefix = "Req"
            elif cap_type in [CapabilityType.CYBER_OFFENSIVE, CapabilityType.CYBER_DEFENSIVE]:
                prefix = "Cyber"
            elif cap_type == CapabilityType.RECONNAISSANCE:
                prefix = "Recon"
            elif cap_type == CapabilityType.COMMUNICATION:
                prefix = "Comm"
            else:
                prefix = "Cap"
            
            capability.structured_tag = f"[{prefix} {type_counters[cap_type]}]"
        
        return capabilities

    def format_capabilities_for_output(self, capabilities: List[ExtractedCapability]) -> str:
        """Format capabilities for structured output as requested in steps1.md."""
        
        if not capabilities:
            return "No capabilities detected."
        
        output_lines = []
        output_lines.append("🔐 Extracted Capabilities\n")
        
        # Group by capability type
        grouped = {}
        for cap in capabilities:
            if cap.capability_type not in grouped:
                grouped[cap.capability_type] = []
            grouped[cap.capability_type].append(cap)
        
        for cap_type, caps in grouped.items():
            # Add type header
            type_name = cap_type.value.replace('_', ' ').title()
            output_lines.append(f"**{type_name} Capabilities**\n")
            
            for cap in caps:
                # Format as requested: bolded, tagged, structured
                output_lines.append(f"**{cap.structured_tag} {cap.title}**")
                output_lines.append(f"- {cap.description}")
                
                if cap.keywords:
                    output_lines.append(f"- *Keywords:* {', '.join(cap.keywords)}")
                
                output_lines.append(f"- *Priority:* {cap.priority_level}/5, *Confidence:* {cap.confidence_score:.2f}")
                output_lines.append("")  # Empty line
        
        return "\n".join(output_lines)
