#!/usr/bin/env python3
"""
Document Reference Resolver
Handles queries that reference specific documents by filename, title, or content.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DocumentReference:
    """A reference to a specific document."""
    reference_type: str  # "filename", "title", "content_keyword"
    reference_value: str
    confidence: float
    original_query: str

@dataclass
class ResolvedDocument:
    """A document that has been resolved from a reference."""
    document_id: str
    filename: str
    title: Optional[str]
    chunk_ids: List[str]
    total_chunks: int
    confidence: float

class DocumentReferenceResolver:
    """
    Resolves document references in user queries to actual stored document chunks.
    Handles queries like:
    - "what is a summary of Document Uploaded: filename.pdf?"
    - "What is MIRIX?" (where MIRIX is the title of a paper)
    - "Tell me about the neural network paper"
    """
    
    def __init__(self, memory_store=None):
        self.memory_store = memory_store
        
        # Patterns for detecting document references
        self.filename_patterns = [
            r'Document Uploaded:\s*([^\s?]+\.(?:pdf|docx|txt|md))',  # "Document Uploaded: file.pdf"
            r'document\s+(?:named|called)?\s*["\']?([^\s"\'?]+\.(?:pdf|docx|txt|md))["\']?',  # "document named file.pdf"
            r'file\s+["\']?([^\s"\'?]+\.(?:pdf|docx|txt|md))["\']?',  # "file filename.pdf"
            r'uploaded\s+(?:document|file)\s*[:\-]?\s*([^\s?]+\.(?:pdf|docx|txt|md))',  # "uploaded document: file.pdf"
            r'([^\s]+\.(?:pdf|docx|txt|md))',  # Any filename with extension
        ]
        
        self.title_patterns = [
            r'what\s+is\s+([A-Z][A-Z0-9\-_]{2,})\??',  # "What is MIRIX?" (all caps acronyms)
            r'tell\s+me\s+about\s+([A-Z][A-Za-z\s]{5,30})',  # "Tell me about Title Case"
            r'(?:paper|document|article)\s+(?:titled|called|about)\s+["\']?([^"\'?]+)["\']?',  # "paper titled X"
            r'Title\s+of\s+paper:\s*([^\n?]+)',  # "Title of paper: X"
        ]
        
        self.content_keyword_patterns = [
            r'(?:neural\s+network|machine\s+learning|AI|artificial\s+intelligence)\s+(?:paper|document|research)',
            r'(?:financial|quarterly|annual)\s+(?:report|statement|analysis)',
            r'(?:technical|user|installation)\s+(?:manual|guide|documentation)',
        ]
        
        # Cache for resolved documents
        self.document_cache = {}
        self.filename_to_chunks_cache = {}
        
        logger.info("DocumentReferenceResolver initialized")
    
    def resolve_document_reference(self, query: str) -> Optional[ResolvedDocument]:
        """
        Resolve a document reference in a user query.
        
        Args:
            query: User query that may contain document references
            
        Returns:
            ResolvedDocument if a document is found, None otherwise
        """
        try:
            # Extract document references from query
            references = self._extract_document_references(query)
            
            if not references:
                return None
            
            # Try to resolve each reference
            for reference in references:
                resolved_doc = self._resolve_reference(reference)
                if resolved_doc and resolved_doc.confidence > 0.7:
                    logger.info(f"📄 Resolved document reference: {reference.reference_value} -> {resolved_doc.filename}")
                    return resolved_doc
            
            # If no high-confidence match, return the best one
            best_match = None
            best_confidence = 0.0
            
            for reference in references:
                resolved_doc = self._resolve_reference(reference)
                if resolved_doc and resolved_doc.confidence > best_confidence:
                    best_match = resolved_doc
                    best_confidence = resolved_doc.confidence
            
            if best_match and best_confidence > 0.3:  # Lower threshold for fallback
                logger.info(f"📄 Best match document: {best_match.filename} (confidence: {best_confidence:.2f})")
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error resolving document reference: {e}")
            return None
    
    def _extract_document_references(self, query: str) -> List[DocumentReference]:
        """Extract potential document references from the query."""
        references = []
        
        # Check for filename patterns
        for pattern in self.filename_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                filename = match.group(1)
                confidence = 0.9 if "Document Uploaded:" in query else 0.7
                
                references.append(DocumentReference(
                    reference_type="filename",
                    reference_value=filename,
                    confidence=confidence,
                    original_query=query
                ))
        
        # Check for title patterns
        for pattern in self.title_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                title = match.group(1).strip()
                if len(title) > 2:  # Avoid single letters
                    confidence = 0.8 if len(title) > 5 else 0.6
                    
                    references.append(DocumentReference(
                        reference_type="title",
                        reference_value=title,
                        confidence=confidence,
                        original_query=query
                    ))
        
        # Check for content keyword patterns
        for pattern in self.content_keyword_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                references.append(DocumentReference(
                    reference_type="content_keyword",
                    reference_value=pattern,
                    confidence=0.5,
                    original_query=query
                ))
        
        return references
    
    def _resolve_reference(self, reference: DocumentReference) -> Optional[ResolvedDocument]:
        """Resolve a specific document reference to stored chunks."""
        if not self.memory_store:
            return None
        
        if reference.reference_type == "filename":
            return self._resolve_by_filename(reference)
        elif reference.reference_type == "title":
            return self._resolve_by_title(reference)
        elif reference.reference_type == "content_keyword":
            return self._resolve_by_content_keywords(reference)
        
        return None
    
    def _resolve_by_filename(self, reference: DocumentReference) -> Optional[ResolvedDocument]:
        """Resolve document by filename."""
        filename = reference.reference_value
        
        # Check cache first
        if filename in self.filename_to_chunks_cache:
            cached_result = self.filename_to_chunks_cache[filename]
            return ResolvedDocument(
                document_id=cached_result['document_id'],
                filename=filename,
                title=cached_result.get('title'),
                chunk_ids=cached_result['chunk_ids'],
                total_chunks=len(cached_result['chunk_ids']),
                confidence=reference.confidence
            )
        
        # Search through all memory chunks
        matching_chunks = []
        document_title = None
        
        try:
            all_chunks = list(self.memory_store.memory_chunks.values())
            
            for chunk in all_chunks:
                if chunk.memory_type.value == 'document':
                    # Check if source contains the filename
                    source_str = str(chunk.source).lower()
                    filename_lower = filename.lower()
                    
                    # Remove file extension for more flexible matching
                    filename_base = Path(filename).stem.lower()
                    
                    if (filename_lower in source_str or 
                        filename_base in source_str or
                        self._fuzzy_filename_match(filename_lower, source_str)):
                        
                        matching_chunks.append(chunk.chunk_id)
                        
                        # Try to extract document title from first chunk
                        if not document_title and len(chunk.content) > 50:
                            document_title = self._extract_title_from_content(chunk.content)
            
            if matching_chunks:
                # Cache the result
                self.filename_to_chunks_cache[filename] = {
                    'document_id': f"doc_{filename}",
                    'chunk_ids': matching_chunks,
                    'title': document_title
                }
                
                return ResolvedDocument(
                    document_id=f"doc_{filename}",
                    filename=filename,
                    title=document_title,
                    chunk_ids=matching_chunks,
                    total_chunks=len(matching_chunks),
                    confidence=reference.confidence
                )
        
        except Exception as e:
            logger.error(f"Error resolving filename {filename}: {e}")
        
        return None
    
    def _resolve_by_title(self, reference: DocumentReference) -> Optional[ResolvedDocument]:
        """Resolve document by title or content keywords."""
        title = reference.reference_value
        
        try:
            # Search for the title in document content
            if self.memory_store:
                search_results = self.memory_store.search_memories(
                    query=title,
                    max_results=20
                )
                
                # Group results by document source
                document_groups = {}
                for result in search_results:
                    if result.similarity_score > 0.6:  # High similarity threshold
                        source = result.chunk.source
                        if source not in document_groups:
                            document_groups[source] = []
                        document_groups[source].append(result.chunk.chunk_id)
                
                # Find the document with the most matching chunks
                if document_groups:
                    best_source = max(document_groups.keys(), key=lambda s: len(document_groups[s]))
                    chunk_ids = document_groups[best_source]
                    
                    # Extract filename from source
                    filename = self._extract_filename_from_source(best_source)
                    
                    return ResolvedDocument(
                        document_id=f"doc_{title}",
                        filename=filename or "unknown",
                        title=title,
                        chunk_ids=chunk_ids,
                        total_chunks=len(chunk_ids),
                        confidence=reference.confidence * 0.8  # Slightly lower confidence for title matches
                    )
        
        except Exception as e:
            logger.error(f"Error resolving title {title}: {e}")
        
        return None
    
    def _resolve_by_content_keywords(self, reference: DocumentReference) -> Optional[ResolvedDocument]:
        """Resolve document by content keywords."""
        # This is a fallback method for general content searches
        # Implementation would be similar to title resolution but with broader search terms
        return None
    
    def _fuzzy_filename_match(self, filename: str, source: str) -> bool:
        """Check for fuzzy filename matches (handles timestamps, etc.)."""
        # Remove common timestamp patterns
        source_clean = re.sub(r'\d{8}_\d{6}_', '', source)
        source_clean = re.sub(r'\d{4}\d{2}\d{2}_\d{6}_', '', source_clean)
        
        # Check if filename appears in cleaned source
        return filename in source_clean
    
    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract document title from content."""
        lines = content.split('\n')
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                # Look for title patterns
                if (line.isupper() or  # ALL CAPS
                    line.startswith('#') or  # Markdown header
                    re.match(r'^[A-Z][A-Za-z\s:]+$', line)):  # Title case
                    return line.replace('#', '').strip()
        
        return None
    
    def _extract_filename_from_source(self, source: str) -> Optional[str]:
        """Extract filename from source string."""
        # Handle different source formats
        if 'uploads/' in source:
            # Extract from path like "document:web_ui/uploads/timestamp_filename.pdf:block_1"
            parts = source.split('/')
            if parts:
                filename_part = parts[-1]
                # Remove block identifier
                filename_part = filename_part.split(':')[0]
                # Remove timestamp prefix
                filename_part = re.sub(r'^\d{8}_\d{6}_', '', filename_part)
                return filename_part
        
        elif ':' in source:
            # Handle "upload:filename.pdf" format
            parts = source.split(':')
            if len(parts) > 1:
                return parts[1]
        
        return None
    
    def get_document_chunks(self, resolved_doc: ResolvedDocument) -> List[Any]:
        """Get all chunks for a resolved document."""
        if not self.memory_store:
            return []
        
        chunks = []
        for chunk_id in resolved_doc.chunk_ids:
            chunk = self.memory_store.memory_chunks.get(chunk_id)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def clear_cache(self):
        """Clear the document resolution cache."""
        self.document_cache.clear()
        self.filename_to_chunks_cache.clear()
        logger.info("Document reference cache cleared")

# Global instance
_document_reference_resolver = None

def get_document_reference_resolver(memory_store=None) -> DocumentReferenceResolver:
    """Get or create the global document reference resolver instance."""
    global _document_reference_resolver
    if _document_reference_resolver is None:
        _document_reference_resolver = DocumentReferenceResolver(memory_store)
    elif memory_store and _document_reference_resolver.memory_store != memory_store:
        _document_reference_resolver.memory_store = memory_store
    return _document_reference_resolver
