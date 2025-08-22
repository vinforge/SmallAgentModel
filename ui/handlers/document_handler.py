#!/usr/bin/env python3
"""
Document Handler for SAM UI
===========================

Handles document upload, processing, and analysis functionality
extracted from the monolithic secure_streamlit_app.py.

This module provides:
- Document upload handling
- File validation and processing
- Document analysis prompts
- Document-specific UI components

Author: SAM Development Team
Version: 1.0.0 - Refactored from secure_streamlit_app.py
"""

import streamlit as st
import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def generate_enhanced_summary_prompt(filename: str) -> str:
    """Generate enhanced summary prompt for documents."""
    return f"""Based on your analysis of '{filename}', provide a comprehensive summary that includes:

🎯 EXECUTIVE SUMMARY:
- Core purpose and main thesis
- Key findings or conclusions
- Primary value proposition

📊 DETAILED ANALYSIS:
- Main sections and their key points
- Important data, statistics, or evidence
- Methodologies or approaches used
- Notable insights or innovations

🔍 CRITICAL ASSESSMENT:
- Strengths and limitations
- Assumptions and potential biases
- Quality of evidence or reasoning
- Relevance and applicability

🚀 ACTIONABLE INSIGHTS:
- Key takeaways for practical application
- Recommendations or next steps
- Areas for further investigation
- Potential impact or implications

📝 FORMAT: Use clear headings, bullet points, and highlight the most important information. Aim for comprehensive yet concise coverage that captures both the breadth and depth of the document."""


def generate_enhanced_questions_prompt(filename: str) -> str:
    """Generate enhanced strategic questions prompt for documents."""
    return f"""Based on your analysis of '{filename}', generate the most strategic and insightful questions that would unlock the document's full value.

🎯 QUESTION CATEGORIES:

🔍 Clarification Questions (Understanding)
What are the key concepts that need deeper explanation?
Which assumptions or methodologies should be questioned?
What context or background would enhance understanding?

💡 Insight Questions (Analysis)
What are the broader implications of the main findings?
How do these findings connect to current trends or challenges?
What patterns or relationships emerge from the data?

🚀 Application Questions (Implementation)
How can these insights be practically applied?
What are the next logical steps or follow-up actions?
What resources or conditions are needed for implementation?

⚠️ Critical Questions (Evaluation)
What are the potential limitations or risks?
What alternative perspectives should be considered?
How reliable or generalizable are the conclusions?

🎯 QUESTION QUALITY CRITERIA:

Strategic: Focus on high-impact, decision-relevant questions
Specific: Avoid generic questions; tailor to document content
Actionable: Questions that lead to concrete insights or actions
Progressive: Build from basic understanding to advanced analysis

📝 FORMAT: Present 8-12 questions organized by category, with brief rationale for why each question is important for maximizing the document's value."""


def generate_enhanced_analysis_prompt(filename: str) -> str:
    """Generate enhanced deep analysis prompt for documents."""
    return f"""Conduct a comprehensive deep analysis of '{filename}' that goes beyond surface-level summary to provide strategic insights and actionable intelligence.

🔬 DEEP ANALYSIS FRAMEWORK:

📊 Content Architecture Analysis
- Document structure and organization
- Information hierarchy and flow
- Key themes and their interconnections
- Evidence quality and source credibility

🎯 Strategic Significance Assessment
- Primary objectives and success criteria
- Stakeholder impact and implications
- Competitive advantages or differentiators
- Market or field positioning

🔍 Critical Evaluation Matrix
- Methodology rigor and validity
- Data quality and completeness
- Logical consistency and coherence
- Potential biases or limitations

💡 Innovation and Insight Discovery
- Novel approaches or breakthrough concepts
- Unexpected findings or counterintuitive results
- Cross-disciplinary connections
- Future research or development opportunities

🚀 Implementation Roadmap
- Practical application scenarios
- Resource requirements and constraints
- Risk factors and mitigation strategies
- Success metrics and evaluation criteria

⚡ Competitive Intelligence
- How this compares to existing solutions
- Unique value propositions identified
- Market gaps or opportunities revealed
- Strategic positioning recommendations

📈 Impact Projection
- Short-term and long-term implications
- Scalability and sustainability factors
- Potential for broader adoption
- Transformative potential assessment

📝 FORMAT: Provide detailed analysis under each framework component, with specific examples and evidence from the document. Include confidence levels for key assessments and highlight areas requiring additional investigation."""


def generate_document_suggestions(filename: str, file_type: str) -> str:
    """Generate contextual suggestions for document interaction."""
    suggestions = {
        '.pdf': [
            f"📋 Summarize the key findings in {filename}",
            f"❓ What are the main research questions addressed?",
            f"🔍 Analyze the methodology used in this study",
            f"💡 What are the practical implications of these results?",
            f"📊 Extract key statistics and data points",
            f"🎯 What are the limitations and future research directions?"
        ],
        '.txt': [
            f"📝 Provide a structured summary of {filename}",
            f"🔍 Identify the main themes and topics",
            f"💭 What insights can be extracted from this text?",
            f"📊 Analyze the writing style and tone",
            f"🎯 What are the key takeaways?"
        ],
        '.docx': [
            f"📄 Summarize the document structure and content",
            f"🔍 Extract key points and recommendations",
            f"💡 What actionable insights are provided?",
            f"📊 Identify important data or evidence",
            f"🎯 What are the main conclusions?"
        ]
    }
    
    return suggestions.get(file_type, [
        f"📋 Analyze the content of {filename}",
        f"🔍 Extract key information and insights",
        f"💡 What are the main points discussed?",
        f"🎯 Provide a comprehensive summary"
    ])


def validate_uploaded_file(uploaded_file) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate uploaded file and return validation results."""
    if uploaded_file is None:
        return False, "No file uploaded", {}
    
    # Check file size (50MB limit)
    max_size = 50 * 1024 * 1024  # 50MB
    if uploaded_file.size > max_size:
        return False, f"File too large: {uploaded_file.size / (1024*1024):.1f}MB (max: 50MB)", {}
    
    # Check file type
    allowed_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md', '.csv'}
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    if file_ext not in allowed_extensions:
        return False, f"Unsupported file type: {file_ext}", {}
    
    # Generate file metadata
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metadata = {
        'filename': uploaded_file.name,
        'file_size': uploaded_file.size,
        'file_type': file_ext,
        'file_hash': file_hash,
        'upload_timestamp': timestamp,
        'session_id': st.session_state.get('session_id', 'default')
    }
    
    return True, "File validation successful", metadata


def render_document_upload_section():
    """Render the document upload section of the UI."""
    st.markdown("### 📁 Document Upload")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=['pdf', 'txt', 'docx', 'doc', 'md', 'csv'],
        help="Upload documents for AI analysis. Supported formats: PDF, TXT, DOCX, DOC, MD, CSV"
    )
    
    if uploaded_file is not None:
        # Validate file
        is_valid, message, metadata = validate_uploaded_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {message}")
            return None
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 File", metadata['filename'])
        
        with col2:
            st.metric("📊 Size", f"{metadata['file_size'] / 1024:.1f} KB")
        
        with col3:
            st.metric("🏷️ Type", metadata['file_type'].upper())
        
        # Process file button
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("📄 Processing document..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = Path(f"temp_{metadata['file_hash']}_{metadata['filename']}")
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Process the document (this would integrate with existing upload handlers)
                    success, process_message, process_metadata = process_uploaded_document(
                        str(temp_path), 
                        metadata['filename'],
                        metadata['session_id']
                    )
                    
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
                    
                    if success:
                        st.success(f"✅ {process_message}")
                        
                        # Store document info in session state
                        if 'uploaded_documents' not in st.session_state:
                            st.session_state.uploaded_documents = []
                        
                        st.session_state.uploaded_documents.append({
                            **metadata,
                            **process_metadata,
                            'processed_at': datetime.now().isoformat()
                        })
                        
                        # Show document suggestions
                        render_document_suggestions(metadata['filename'], metadata['file_type'])
                        
                    else:
                        st.error(f"❌ {process_message}")
                        
                except Exception as e:
                    logger.error(f"Document processing error: {e}")
                    st.error(f"❌ Processing failed: {str(e)}")
        
        return metadata
    
    return None


def render_document_suggestions(filename: str, file_type: str):
    """Render contextual suggestions for document interaction."""
    st.markdown("### 💡 Suggested Actions")
    
    suggestions = generate_document_suggestions(filename, file_type)
    
    # Create columns for suggestions
    cols = st.columns(2)
    
    for i, suggestion in enumerate(suggestions[:6]):  # Limit to 6 suggestions
        col = cols[i % 2]
        
        with col:
            if st.button(suggestion, key=f"suggestion_{i}"):
                # Add suggestion to chat
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": suggestion
                })
                
                st.rerun()


def process_uploaded_document(file_path: str, filename: str, session_id: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Process uploaded document using existing SAM infrastructure."""
    try:
        # This would integrate with the existing document processing pipeline
        # For now, return a placeholder implementation
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            # Use existing PDF handler
            from sam.document_processing.proven_pdf_integration import handle_pdf_upload_for_sam
            return handle_pdf_upload_for_sam(file_path, filename, session_id)
        
        elif file_ext in ['.txt', '.md']:
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic processing for text files
            metadata = {
                'document_id': f"text_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'content_length': len(content),
                'word_count': len(content.split()),
                'processing_method': 'text_extraction'
            }
            
            return True, f"Text document processed: {len(content)} characters", metadata
        
        else:
            return False, f"Unsupported file type: {file_ext}", {}
    
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        return False, f"Processing failed: {str(e)}", {}


def render_uploaded_documents_list():
    """Render list of uploaded documents in session."""
    if 'uploaded_documents' not in st.session_state or not st.session_state.uploaded_documents:
        return
    
    st.markdown("### 📚 Uploaded Documents")
    
    for i, doc in enumerate(st.session_state.uploaded_documents):
        with st.expander(f"📄 {doc['filename']} ({doc['file_type'].upper()})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Size:** {doc['file_size'] / 1024:.1f} KB")
                st.write(f"**Uploaded:** {doc['upload_timestamp']}")
            
            with col2:
                st.write(f"**Session:** {doc['session_id']}")
                st.write(f"**Hash:** {doc['file_hash']}")
            
            # Action buttons for each document
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"📋 Summarize", key=f"sum_{i}"):
                    prompt = generate_enhanced_summary_prompt(doc['filename'])
                    # Add to chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.rerun()
            
            with col2:
                if st.button(f"❓ Questions", key=f"q_{i}"):
                    prompt = generate_enhanced_questions_prompt(doc['filename'])
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.rerun()
            
            with col3:
                if st.button(f"🔍 Analyze", key=f"ana_{i}"):
                    prompt = generate_enhanced_analysis_prompt(doc['filename'])
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.rerun()
