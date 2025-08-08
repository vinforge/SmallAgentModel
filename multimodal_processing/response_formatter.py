#!/usr/bin/env python3
"""
Enhanced Response Formatter for SAM
Post-processing step that auto-formats lists with structured tags and improves
output quality for government contractors and SBIR writers.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OutputFormat(Enum):
    """Output formatting styles."""
    STRUCTURED_CAPABILITIES = "structured_capabilities"
    SBIR_PROPOSAL = "sbir_proposal"
    TECHNICAL_SUMMARY = "technical_summary"
    REQUIREMENT_LIST = "requirement_list"
    EXECUTIVE_SUMMARY = "executive_summary"

@dataclass
class FormattedSection:
    """A formatted section of the response."""
    title: str
    content: str
    section_type: str
    priority: int
    metadata: Dict[str, Any]

class ResponseFormatter:
    """Enhanced response formatter with structured output capabilities."""
    
    def __init__(self):
        self.capability_patterns = [
            r'(?i)\b(?:capability|requirement|specification|objective|goal)\b',
            r'(?i)\b(?:develop|implement|provide|support|enable|deliver)\b',
            r'(?i)\b(?:remote|cyber|access|control|reconnaissance|execution)\b',
        ]
        
        self.list_patterns = [
            r'^\s*[•·▪▫‣⁃\-*+]\s+',  # Bullet points
            r'^\s*\d+\.\s+',          # Numbered lists
            r'^\s*[a-z]\.\s+',        # Lettered lists
            r'^\s*\([a-z0-9]+\)\s+',  # Parenthetical lists
        ]
        
        self.priority_indicators = {
            'critical': 5, 'essential': 5, 'mandatory': 5, 'required': 4,
            'important': 4, 'significant': 3, 'useful': 3, 'beneficial': 2,
            'optional': 1, 'nice-to-have': 1
        }

    def format_response(self, response_text: str, output_format: OutputFormat = OutputFormat.STRUCTURED_CAPABILITIES) -> str:
        """
        Format response with enhanced structure and capability tagging.
        
        Args:
            response_text: Raw response text from SAM
            output_format: Desired output format style
            
        Returns:
            Formatted response with structured tags and improved readability
        """
        
        # Parse the response into sections
        sections = self._parse_response_sections(response_text)
        
        # Process each section based on format
        formatted_sections = []
        for section in sections:
            formatted_section = self._format_section(section, output_format)
            if formatted_section:
                formatted_sections.append(formatted_section)
        
        # Combine sections with appropriate formatting
        final_output = self._combine_sections(formatted_sections, output_format)
        
        # Apply final post-processing
        final_output = self._apply_final_formatting(final_output, output_format)
        
        return final_output

    def _parse_response_sections(self, text: str) -> List[Dict[str, Any]]:
        """Parse response text into logical sections."""
        sections = []
        
        # Split by headers or major breaks
        header_pattern = r'^(#{1,6}\s+.+|[A-Z][A-Z\s]{2,}:?\s*$|\*\*[^*]+\*\*)'
        parts = re.split(header_pattern, text, flags=re.MULTILINE)
        
        current_section = {"title": "", "content": "", "type": "content"}
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            # Check if this is a header
            if re.match(header_pattern, part, re.MULTILINE):
                # Save previous section
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": self._clean_header(part),
                    "content": "",
                    "type": self._detect_section_type(part)
                }
            else:
                current_section["content"] += part + "\n"
        
        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections

    def _clean_header(self, header: str) -> str:
        """Clean header text for display."""
        # Remove markdown formatting
        header = re.sub(r'[#*]+\s*', '', header)
        # Remove trailing colons
        header = header.rstrip(':').strip()
        return header

    def _detect_section_type(self, header: str) -> str:
        """Detect the type of section based on header content."""
        header_lower = header.lower()
        
        if any(word in header_lower for word in ['capability', 'requirement', 'specification']):
            return 'capabilities'
        elif any(word in header_lower for word in ['summary', 'overview', 'abstract']):
            return 'summary'
        elif any(word in header_lower for word in ['objective', 'goal', 'purpose']):
            return 'objectives'
        elif any(word in header_lower for word in ['technical', 'implementation', 'approach']):
            return 'technical'
        else:
            return 'content'

    def _format_section(self, section: Dict[str, Any], output_format: OutputFormat) -> Optional[FormattedSection]:
        """Format a single section based on the output format."""
        
        content = section["content"].strip()
        if not content:
            return None
        
        # Detect and format lists
        if self._contains_lists(content):
            content = self._format_lists(content, section["type"])
        
        # Apply format-specific processing
        if output_format == OutputFormat.STRUCTURED_CAPABILITIES:
            content = self._format_for_capabilities(content, section["type"])
        elif output_format == OutputFormat.SBIR_PROPOSAL:
            content = self._format_for_sbir(content, section["type"])
        elif output_format == OutputFormat.REQUIREMENT_LIST:
            content = self._format_for_requirements(content, section["type"])
        
        # Calculate priority
        priority = self._calculate_section_priority(content, section["type"])
        
        return FormattedSection(
            title=section["title"],
            content=content,
            section_type=section["type"],
            priority=priority,
            metadata={"original_type": section["type"]}
        )

    def _contains_lists(self, content: str) -> bool:
        """Check if content contains list structures."""
        lines = content.split('\n')
        list_count = 0
        
        for line in lines:
            if any(re.match(pattern, line) for pattern in self.list_patterns):
                list_count += 1
        
        return list_count >= 2  # At least 2 list items

    def _format_lists(self, content: str, section_type: str) -> str:
        """Format lists with structured tags and improved formatting."""
        lines = content.split('\n')
        formatted_lines = []
        list_counter = 0
        in_list = False
        
        for line in lines:
            stripped_line = line.strip()
            
            # Check if this is a list item
            is_list_item = any(re.match(pattern, line) for pattern in self.list_patterns)
            
            if is_list_item:
                if not in_list:
                    in_list = True
                    list_counter = 0
                
                list_counter += 1
                
                # Extract the content after the list marker
                content_match = None
                for pattern in self.list_patterns:
                    match = re.match(pattern + r'(.+)', line)
                    if match:
                        content_match = match.group(1).strip()
                        break
                
                if content_match:
                    # Determine tag type based on section and content
                    tag = self._generate_structured_tag(content_match, section_type, list_counter)
                    
                    # Format with bold tag and content
                    if self._is_capability_item(content_match):
                        formatted_line = f"**{tag} {content_match}**"
                    else:
                        formatted_line = f"- **{tag}** {content_match}"
                    
                    formatted_lines.append(formatted_line)
                else:
                    formatted_lines.append(line)
            else:
                if in_list and stripped_line == "":
                    in_list = False
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def _generate_structured_tag(self, content: str, section_type: str, counter: int) -> str:
        """Generate structured tags like [Req 1], [Cap 2] based on content."""
        
        content_lower = content.lower()
        
        # Determine tag prefix based on content and section type
        if section_type == 'capabilities' or self._is_capability_item(content):
            if any(word in content_lower for word in ['cyber', 'attack', 'exploit', 'penetration']):
                return f"[Cyber {counter}]"
            elif any(word in content_lower for word in ['recon', 'surveillance', 'intelligence']):
                return f"[Recon {counter}]"
            elif any(word in content_lower for word in ['communication', 'c2', 'command']):
                return f"[Comm {counter}]"
            else:
                return f"[Cap {counter}]"
        
        elif any(word in content_lower for word in ['requirement', 'shall', 'must', 'specification']):
            return f"[Req {counter}]"
        
        elif any(word in content_lower for word in ['objective', 'goal', 'purpose']):
            return f"[Obj {counter}]"
        
        elif any(word in content_lower for word in ['task', 'activity', 'action']):
            return f"[Task {counter}]"
        
        else:
            return f"[Item {counter}]"

    def _is_capability_item(self, content: str) -> bool:
        """Check if content represents a capability."""
        return any(re.search(pattern, content) for pattern in self.capability_patterns)

    def _format_for_capabilities(self, content: str, section_type: str) -> str:
        """Format content specifically for capability extraction."""
        
        # Add capability header if this is a capability section
        if section_type == 'capabilities' and not content.startswith('🔐'):
            content = "🔐 **Requested Capabilities**\n\n" + content
        
        # Enhance list formatting
        content = self._enhance_capability_lists(content)
        
        return content

    def _format_for_sbir(self, content: str, section_type: str) -> str:
        """Format content for SBIR proposal style."""
        
        # Add SBIR-specific headers and formatting
        if section_type == 'objectives':
            content = "**Technical Objectives:**\n\n" + content
        elif section_type == 'capabilities':
            content = "**Proposed Technical Approach:**\n\n" + content
        elif section_type == 'technical':
            content = "**Technical Implementation:**\n\n" + content
        
        return content

    def _format_for_requirements(self, content: str, section_type: str) -> str:
        """Format content as a structured requirement list."""
        
        # Convert all list items to requirements format
        lines = content.split('\n')
        formatted_lines = []
        req_counter = 0
        
        for line in lines:
            if any(re.match(pattern, line) for pattern in self.list_patterns):
                req_counter += 1
                # Extract content and format as requirement
                for pattern in self.list_patterns:
                    match = re.match(pattern + r'(.+)', line)
                    if match:
                        req_content = match.group(1).strip()
                        formatted_lines.append(f"**[REQ-{req_counter:03d}]** {req_content}")
                        break
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def _enhance_capability_lists(self, content: str) -> str:
        """Enhance capability list formatting with sub-items."""
        
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            # Check for sub-items (indented lines)
            if line.startswith('  ') and not line.startswith('    '):
                # This is a sub-item, format with bullet
                enhanced_lines.append(f"  • {line.strip()}")
            elif line.startswith('    '):
                # This is a sub-sub-item
                enhanced_lines.append(f"    ◦ {line.strip()}")
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)

    def _calculate_section_priority(self, content: str, section_type: str) -> int:
        """Calculate priority for section ordering."""
        
        base_priority = {
            'summary': 5,
            'objectives': 4,
            'capabilities': 4,
            'technical': 3,
            'content': 2
        }.get(section_type, 2)
        
        # Boost priority for capability-related content
        if any(re.search(pattern, content) for pattern in self.capability_patterns):
            base_priority += 1
        
        # Check for priority keywords
        content_lower = content.lower()
        for keyword, priority_boost in self.priority_indicators.items():
            if keyword in content_lower:
                base_priority = max(base_priority, priority_boost)
                break
        
        return min(base_priority, 5)

    def _combine_sections(self, sections: List[FormattedSection], output_format: OutputFormat) -> str:
        """Combine formatted sections into final output."""
        
        # Sort sections by priority (descending)
        sections.sort(key=lambda x: x.priority, reverse=True)
        
        output_parts = []
        
        for section in sections:
            if section.title:
                # Add section header
                if output_format == OutputFormat.STRUCTURED_CAPABILITIES:
                    output_parts.append(f"\n## {section.title}\n")
                else:
                    output_parts.append(f"\n**{section.title}**\n")
            
            output_parts.append(section.content)
            output_parts.append("")  # Empty line between sections
        
        return '\n'.join(output_parts)

    def _apply_final_formatting(self, text: str, output_format: OutputFormat) -> str:
        """Apply final formatting touches."""
        
        # Remove excessive empty lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure proper spacing around headers
        text = re.sub(r'\n(#{1,6}[^#\n]+)\n', r'\n\n\1\n\n', text)
        
        # Add format-specific footer
        if output_format == OutputFormat.STRUCTURED_CAPABILITIES:
            text += "\n\n---\n*Generated by SAM Enhanced Capability Extraction*"
        elif output_format == OutputFormat.SBIR_PROPOSAL:
            text += "\n\n---\n*Formatted for SBIR Proposal Submission*"
        
        return text.strip()
