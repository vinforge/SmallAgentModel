#!/usr/bin/env python3
"""
Proven PDF Integration for SAM
Integrates the proven PDF processor with SAM's existing systems.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from .proven_pdf_processor import get_proven_pdf_processor, ProvenPDFProcessor

logger = logging.getLogger(__name__)

class SAMProvenPDFIntegration:
    """
    Integration layer between SAM and the proven PDF processor.
    Handles PDF uploads and queries using the proven approach.
    """
    
    def __init__(self):
        """Initialize the SAM PDF integration."""
        self.processor = get_proven_pdf_processor()
        self.session_pdfs = {}  # Track PDFs per session
        self.current_session = "default"
        logger.info("🔗 SAM Proven PDF Integration initialized")
    
    def handle_pdf_upload(self, pdf_path: str, filename: str = None, session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Handle PDF upload using the proven approach.
        
        Args:
            pdf_path: Path to the uploaded PDF
            filename: Original filename
            session_id: Session identifier
            
        Returns:
            Tuple of (success, message, metadata)
        """
        try:
            session = session_id or self.current_session
            
            if not filename:
                filename = Path(pdf_path).name
            
            # Remove .pdf extension for processing
            pdf_name = Path(filename).stem
            
            logger.info(f"📤 Handling PDF upload: {filename} for session {session}")
            
            # Process the PDF using proven approach
            success, message = self.processor.process_pdf(pdf_path, pdf_name)
            
            if success:
                # Track the PDF for this session
                if session not in self.session_pdfs:
                    self.session_pdfs[session] = []
                
                # Add to session (remove duplicates)
                if pdf_name not in self.session_pdfs[session]:
                    self.session_pdfs[session].append(pdf_name)
                
                # Set as current for this session
                self.current_session = session
                
                metadata = {
                    "pdf_name": pdf_name,
                    "original_filename": filename,
                    "session_id": session,
                    "processing_method": "proven_pdf_processor",
                    "is_current": True
                }
                
                logger.info(f"✅ PDF upload successful: {pdf_name}")
                return True, f"Successfully processed {filename}", metadata
            else:
                logger.error(f"❌ PDF upload failed: {message}")
                return False, message, {}
                
        except Exception as e:
            logger.error(f"Failed to handle PDF upload: {e}")
            return False, f"Upload failed: {str(e)}", {}
    
    def query_uploaded_pdf(self, query: str, session_id: str = None, pdf_name: str = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Query an uploaded PDF using the proven approach.
        
        Args:
            query: Question to ask
            session_id: Session identifier
            pdf_name: Specific PDF to query (optional)
            
        Returns:
            Tuple of (success, response, metadata)
        """
        try:
            session = session_id or self.current_session
            
            logger.info(f"🔍 Querying PDF for session {session}: {query[:50]}...")
            
            # Determine which PDF to query
            target_pdf = None
            
            if pdf_name:
                # Query specific PDF
                target_pdf = pdf_name
            elif session in self.session_pdfs and self.session_pdfs[session]:
                # Query most recent PDF for this session
                target_pdf = self.session_pdfs[session][-1]
            else:
                # Try current PDF
                if self.processor.current_pdf_name:
                    target_pdf = self.processor.current_pdf_name
            
            if not target_pdf:
                return False, "No PDF has been uploaded for this session", {}
            
            # Query using proven approach
            success, response, metadata = self.processor.query_pdf(query, target_pdf)
            
            if success:
                # Add session information to metadata
                metadata.update({
                    "session_id": session,
                    "query_method": "proven_pdf_processor",
                    "target_pdf": target_pdf
                })
                
                logger.info(f"✅ PDF query successful: {len(response)} characters")
                return True, response, metadata
            else:
                logger.error(f"❌ PDF query failed: {response}")
                return False, response, metadata
                
        except Exception as e:
            logger.error(f"Failed to query PDF: {e}")
            return False, f"Query failed: {str(e)}", {}
    
    def get_session_pdfs(self, session_id: str = None) -> List[str]:
        """Get list of PDFs for a session."""
        session = session_id or self.current_session
        return self.session_pdfs.get(session, [])
    
    def get_current_pdf(self, session_id: str = None) -> Optional[str]:
        """Get the current PDF for a session."""
        session = session_id or self.current_session
        pdfs = self.session_pdfs.get(session, [])
        return pdfs[-1] if pdfs else None
    
    def clear_session_pdfs(self, session_id: str = None):
        """Clear PDFs for a session."""
        session = session_id or self.current_session
        if session in self.session_pdfs:
            # Clear from processor
            for pdf_name in self.session_pdfs[session]:
                self.processor.clear_pdf(pdf_name)
            
            # Clear from session tracking
            del self.session_pdfs[session]
            
            logger.info(f"🗑️ Cleared PDFs for session: {session}")
    
    def is_pdf_query(self, query: str) -> bool:
        """
        Detect if a query is asking about an uploaded PDF.

        Args:
            query: User query

        Returns:
            True if query appears to be about a PDF
        """
        query_lower = query.lower()

        # ENHANCED: Check for Deep Analysis patterns first (high priority)
        deep_analysis_patterns = [
            '🔍 deep analysis', 'deep analysis:', 'analyze', 'analysis',
            'comprehensive analysis', 'detailed analysis', 'in-depth analysis',
            'thorough analysis', 'examine', 'review', 'breakdown'
        ]

        for pattern in deep_analysis_patterns:
            if pattern in query_lower:
                return True

        # ENHANCED: Check for arXiv and academic paper patterns
        import re
        arxiv_patterns = [
            r'\b\d{4}\.\d{5}v?\d*\.?pdf?\b',  # arXiv patterns like "2305.18290v3.pdf"
            r'\b\d{4}\.\d{5}v?\d*\b',        # arXiv without extension
            r'arxiv:\s*\d{4}\.\d{5}',        # "arxiv:2305.18290"
        ]

        for pattern in arxiv_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check for PDF-related keywords
        pdf_indicators = [
            'pdf', 'document', 'file', 'upload', 'paper',
            'summarize', 'summary', 'what is this about',
            'what does this', 'content of', 'main topic',
            'uploaded', 'discuss', 'explain', 'describe'
        ]

        # Check for specific file extensions
        file_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']

        # Legacy arXiv patterns (keeping for compatibility)
        legacy_arxiv_patterns = ['arxiv:', '2507.', '2506.', 'v1.pdf', 'v2.pdf', 'v3.pdf']

        return (
            any(indicator in query_lower for indicator in pdf_indicators) or
            any(ext in query_lower for ext in file_extensions) or
            any(pattern in query_lower for pattern in legacy_arxiv_patterns)
        )
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of the PDF integration."""
        return {
            "processor_available": self.processor is not None,
            "current_session": self.current_session,
            "total_sessions": len(self.session_pdfs),
            "current_pdf": self.processor.current_pdf_name,
            "total_processed_pdfs": len(self.processor.list_processed_pdfs()),
            "session_pdfs": dict(self.session_pdfs)
        }

# Global integration instance
_sam_pdf_integration = None

def get_sam_pdf_integration() -> SAMProvenPDFIntegration:
    """Get the global SAM PDF integration instance."""
    global _sam_pdf_integration
    if _sam_pdf_integration is None:
        _sam_pdf_integration = SAMProvenPDFIntegration()
    return _sam_pdf_integration

def handle_pdf_upload_for_sam(pdf_path: str, filename: str = None, session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Handle PDF upload for SAM using the proven approach.
    
    Args:
        pdf_path: Path to the uploaded PDF
        filename: Original filename
        session_id: Session identifier
        
    Returns:
        Tuple of (success, message, metadata)
    """
    integration = get_sam_pdf_integration()
    return integration.handle_pdf_upload(pdf_path, filename, session_id)

def query_pdf_for_sam(query: str, session_id: str = None, pdf_name: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Query PDF for SAM using the proven approach.
    
    Args:
        query: Question to ask
        session_id: Session identifier
        pdf_name: Specific PDF to query (optional)
        
    Returns:
        Tuple of (success, response, metadata)
    """
    integration = get_sam_pdf_integration()
    return integration.query_uploaded_pdf(query, session_id, pdf_name)

def is_pdf_query_for_sam(query: str) -> bool:
    """
    Detect if a query is asking about an uploaded PDF.
    
    Args:
        query: User query
        
    Returns:
        True if query appears to be about a PDF
    """
    integration = get_sam_pdf_integration()
    return integration.is_pdf_query(query)

def get_sam_pdf_status() -> Dict[str, Any]:
    """Get status of SAM's PDF integration."""
    integration = get_sam_pdf_integration()
    return integration.get_integration_status()

if __name__ == "__main__":
    # Test the SAM PDF integration
    print("🧪 Testing SAM Proven PDF Integration")
    
    integration = SAMProvenPDFIntegration()
    
    # Test PDF query detection
    test_queries = [
        "what is 2507.07957v1.pdf about?",
        "summarize the uploaded document",
        "what is the weather like?",
        "what does this file contain?"
    ]
    
    for query in test_queries:
        is_pdf = integration.is_pdf_query(query)
        print(f"Query: '{query}' -> PDF query: {is_pdf}")
    
    # Test with actual PDF if available
    test_pdf = "/Users/vinsoncornejo/Downloads/augment-projects/SAM 2/2507.07957v1.pdf"
    
    if os.path.exists(test_pdf):
        print(f"\n📄 Testing with: {test_pdf}")
        
        # Upload PDF
        success, message, metadata = integration.handle_pdf_upload(test_pdf, "2507.07957v1.pdf", "test_session")
        print(f"Upload: {success} - {message}")
        print(f"Metadata: {metadata}")
        
        if success:
            # Query PDF
            query = "What is this document about?"
            success, response, metadata = integration.query_uploaded_pdf(query, "test_session")
            print(f"Query: {success}")
            print(f"Response: {response[:200]}...")
            print(f"Metadata: {metadata}")
    else:
        print(f"❌ Test PDF not found: {test_pdf}")
    
    # Show status
    status = integration.get_integration_status()
    print(f"\nStatus: {status}")
