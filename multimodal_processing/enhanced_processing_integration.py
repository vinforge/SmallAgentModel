#!/usr/bin/env python3
"""
Enhanced Processing Integration for SAM
Integrates enhanced chunking, capability extraction, and response formatting
to improve SAM's output quality for government contractors and SBIR writers.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of enhanced processing."""
    enhanced_chunks: List[Any]
    extracted_capabilities: List[Any]
    formatted_response: str
    metadata: Dict[str, Any]

class EnhancedProcessingIntegration:
    """Main integration class for enhanced processing features."""
    
    def __init__(self):
        self.chunker = None
        self.capability_extractor = None
        self.response_formatter = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize processing components."""
        try:
            from multimodal_processing.enhanced_chunker import EnhancedChunker
            from multimodal_processing.capability_extractor import CapabilityExtractor
            from multimodal_processing.response_formatter import ResponseFormatter, OutputFormat
            
            self.chunker = EnhancedChunker()
            self.capability_extractor = CapabilityExtractor()
            self.response_formatter = ResponseFormatter()
            self.OutputFormat = OutputFormat
            
            logger.info("Enhanced processing components initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Enhanced processing components not available: {e}")
            self.chunker = None
            self.capability_extractor = None
            self.response_formatter = None

    def process_document_content(self, text: str, source_location: str) -> ProcessingResult:
        """
        Process document content with enhanced chunking and capability extraction.
        
        Args:
            text: Raw document text
            source_location: Source location identifier
            
        Returns:
            ProcessingResult with enhanced chunks and extracted capabilities
        """
        
        if not self.chunker or not self.capability_extractor:
            logger.warning("Enhanced processing not available, using basic processing")
            return self._basic_processing(text, source_location)
        
        try:
            # Enhanced chunking
            enhanced_chunks = self.chunker.enhanced_chunk_text(text, source_location)
            
            # Capability extraction
            extracted_capabilities = self.capability_extractor.extract_capabilities(text, source_location)
            
            # Generate metadata
            metadata = self._generate_processing_metadata(enhanced_chunks, extracted_capabilities)
            
            # Create formatted response preview
            formatted_response = self._create_response_preview(enhanced_chunks, extracted_capabilities)
            
            return ProcessingResult(
                enhanced_chunks=enhanced_chunks,
                extracted_capabilities=extracted_capabilities,
                formatted_response=formatted_response,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Enhanced processing failed: {e}")
            return self._basic_processing(text, source_location)

    def format_sam_response(self, response_text: str, format_type: str = "structured_capabilities") -> str:
        """
        Format SAM's response with enhanced structure and capability tagging.
        
        Args:
            response_text: Raw response from SAM
            format_type: Type of formatting to apply
            
        Returns:
            Formatted response with structured tags and improved readability
        """
        
        if not self.response_formatter:
            logger.warning("Response formatter not available, returning original response")
            return response_text
        
        try:
            # Map format type to OutputFormat enum
            format_mapping = {
                "structured_capabilities": self.OutputFormat.STRUCTURED_CAPABILITIES,
                "sbir_proposal": self.OutputFormat.SBIR_PROPOSAL,
                "technical_summary": self.OutputFormat.TECHNICAL_SUMMARY,
                "requirement_list": self.OutputFormat.REQUIREMENT_LIST,
                "executive_summary": self.OutputFormat.EXECUTIVE_SUMMARY,
            }
            
            output_format = format_mapping.get(format_type, self.OutputFormat.STRUCTURED_CAPABILITIES)
            
            # Format the response
            formatted_response = self.response_formatter.format_response(response_text, output_format)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return response_text

    def extract_capabilities_from_text(self, text: str, source_location: str = "unknown") -> str:
        """
        Extract and format capabilities from text for immediate display.
        
        Args:
            text: Input text to analyze
            source_location: Source location identifier
            
        Returns:
            Formatted capability list
        """
        
        if not self.capability_extractor:
            return "Capability extraction not available."
        
        try:
            capabilities = self.capability_extractor.extract_capabilities(text, source_location)
            formatted_output = self.capability_extractor.format_capabilities_for_output(capabilities)
            return formatted_output
            
        except Exception as e:
            logger.error(f"Capability extraction failed: {e}")
            return f"Error extracting capabilities: {e}"

    def get_enhanced_chunk_summary(self, text: str, source_location: str) -> Dict[str, Any]:
        """
        Get a summary of enhanced chunking results for analysis.
        
        Args:
            text: Input text to analyze
            source_location: Source location identifier
            
        Returns:
            Summary statistics and analysis
        """
        
        if not self.chunker:
            return {"error": "Enhanced chunker not available"}
        
        try:
            chunks = self.chunker.enhanced_chunk_text(text, source_location)
            
            # Analyze chunks
            chunk_types = {}
            total_priority = 0
            high_priority_chunks = 0
            
            for chunk in chunks:
                chunk_type = chunk.chunk_type.value
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                total_priority += chunk.priority_score
                
                if chunk.priority_score >= 1.5:
                    high_priority_chunks += 1
            
            avg_priority = total_priority / len(chunks) if chunks else 0
            
            return {
                "total_chunks": len(chunks),
                "chunk_types": chunk_types,
                "average_priority": round(avg_priority, 2),
                "high_priority_chunks": high_priority_chunks,
                "priority_percentage": round((high_priority_chunks / len(chunks)) * 100, 1) if chunks else 0
            }
            
        except Exception as e:
            logger.error(f"Chunk analysis failed: {e}")
            return {"error": str(e)}

    def _basic_processing(self, text: str, source_location: str) -> ProcessingResult:
        """Fallback basic processing when enhanced components aren't available."""
        
        # Basic paragraph splitting
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Create basic chunks
        basic_chunks = []
        for i, paragraph in enumerate(paragraphs):
            basic_chunks.append({
                'content': paragraph,
                'chunk_type': 'text',
                'priority_score': 1.0,
                'source_location': f"{source_location}_basic_{i}"
            })
        
        return ProcessingResult(
            enhanced_chunks=basic_chunks,
            extracted_capabilities=[],
            formatted_response=text,
            metadata={"processing_type": "basic", "chunks": len(basic_chunks)}
        )

    def _generate_processing_metadata(self, chunks: List[Any], capabilities: List[Any]) -> Dict[str, Any]:
        """Generate metadata about the processing results."""
        
        chunk_stats = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            chunk_stats[chunk_type] = chunk_stats.get(chunk_type, 0) + 1
        
        capability_stats = {}
        for cap in capabilities:
            cap_type = cap.capability_type.value
            capability_stats[cap_type] = capability_stats.get(cap_type, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "chunk_distribution": chunk_stats,
            "total_capabilities": len(capabilities),
            "capability_distribution": capability_stats,
            "high_priority_chunks": sum(1 for chunk in chunks if chunk.priority_score >= 1.5),
            "processing_timestamp": "2025-06-08T22:00:00Z"  # Would use actual timestamp
        }

    def _create_response_preview(self, chunks: List[Any], capabilities: List[Any]) -> str:
        """Create a preview of how the response would be formatted."""
        
        preview_lines = []
        preview_lines.append("📊 **Enhanced Processing Preview**\n")
        
        # Chunk summary
        preview_lines.append(f"**Chunks Detected:** {len(chunks)}")
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        for chunk_type, count in chunk_types.items():
            preview_lines.append(f"- {chunk_type.title()}: {count}")
        
        preview_lines.append("")
        
        # Capability summary
        if capabilities:
            preview_lines.append(f"**Capabilities Extracted:** {len(capabilities)}")
            for i, cap in enumerate(capabilities[:3], 1):  # Show first 3
                preview_lines.append(f"{i}. {cap.structured_tag} {cap.title}")
            
            if len(capabilities) > 3:
                preview_lines.append(f"... and {len(capabilities) - 3} more")
        else:
            preview_lines.append("**Capabilities Extracted:** None detected")
        
        return "\n".join(preview_lines)

    def is_available(self) -> bool:
        """Check if enhanced processing is available."""
        return all([self.chunker, self.capability_extractor, self.response_formatter])

    def get_status(self) -> Dict[str, Any]:
        """Get status of enhanced processing components."""
        return {
            "enhanced_chunker": self.chunker is not None,
            "capability_extractor": self.capability_extractor is not None,
            "response_formatter": self.response_formatter is not None,
            "fully_available": self.is_available()
        }

# Global instance for easy access
enhanced_processor = EnhancedProcessingIntegration()

def process_document_with_enhancements(text: str, source_location: str) -> ProcessingResult:
    """Convenience function for document processing."""
    return enhanced_processor.process_document_content(text, source_location)

def format_response_with_enhancements(response_text: str, format_type: str = "structured_capabilities") -> str:
    """Convenience function for response formatting."""
    return enhanced_processor.format_sam_response(response_text, format_type)

def extract_capabilities(text: str, source_location: str = "unknown") -> str:
    """Convenience function for capability extraction."""
    return enhanced_processor.extract_capabilities_from_text(text, source_location)
