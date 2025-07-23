#!/usr/bin/env python3
"""
Proven PDF Processor
Based on the successful LLM-powered PDF Chatbot from GitHub.
This module provides reliable PDF processing and querying capabilities.
"""

import os
import pickle
import tempfile
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    # Handle PyPDF2 version compatibility
    try:
        from PyPDF2 import PdfReader  # PyPDF2 3.0+
        PDF_READER_CLASS = PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfFileReader as PdfReader  # PyPDF2 2.x
            PDF_READER_CLASS = PdfReader
        except ImportError:
            PDF_READER_CLASS = None

    # Try to import langchain components (optional)
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        logging.warning("LangChain not available - using fallback text processing")

    # Use offline embeddings instead of OpenAI
    try:
        from sentence_transformers import SentenceTransformer
        OFFLINE_EMBEDDINGS_AVAILABLE = True
    except ImportError:
        OFFLINE_EMBEDDINGS_AVAILABLE = False
        logging.warning("SentenceTransformers not available for offline embeddings")

except ImportError as e:
    logging.warning(f"PDF processing dependencies not available: {e}")
    OFFLINE_EMBEDDINGS_AVAILABLE = False
    LANGCHAIN_AVAILABLE = False
    PDF_READER_CLASS = None

logger = logging.getLogger(__name__)


class FallbackTextSplitter:
    """Fallback text splitter when langchain is not available."""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(start, end - 200)
                sentence_end = -1

                for i in range(end, search_start, -1):
                    if text[i] in '.!?':
                        sentence_end = i + 1
                        break

                if sentence_end > 0:
                    end = sentence_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks


class FallbackDocument:
    """Fallback document class when langchain is not available."""

    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class OfflineEmbeddings:
    """Offline embeddings using SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize offline embeddings."""
        if OFFLINE_EMBEDDINGS_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Offline embeddings initialized: {model_name}")
        else:
            self.model = None
            logger.warning("❌ Offline embeddings not available")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not self.model:
            raise ValueError("Offline embeddings not available")

        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if not self.model:
            raise ValueError("Offline embeddings not available")

        embedding = self.model.encode([text])
        return embedding[0].tolist()

class ProvenPDFProcessor:
    """
    Proven PDF processor based on successful GitHub implementation.
    Provides reliable PDF processing and querying capabilities.
    """
    
    def __init__(self, storage_dir: str = None):
        """Initialize the proven PDF processor."""
        self.storage_dir = storage_dir or tempfile.gettempdir()
        self.embeddings = None
        self.vector_stores = {}  # Store multiple PDF vector stores
        self.current_pdf_name = None
        self.current_vector_store = None
        
        # Initialize offline embeddings
        try:
            self.embeddings = OfflineEmbeddings()
            logger.info("✅ Proven PDF Processor initialized with offline embeddings")
        except Exception as e:
            logger.warning(f"Failed to initialize offline embeddings: {e}")
            self.embeddings = None
    
    def process_pdf(self, pdf_path: str, pdf_name: str = None) -> Tuple[bool, str]:
        """
        Process a PDF file using the proven approach.
        
        Args:
            pdf_path: Path to the PDF file
            pdf_name: Optional name for the PDF (defaults to filename)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not os.path.exists(pdf_path):
                return False, f"PDF file not found: {pdf_path}"
            
            # Extract PDF name
            if not pdf_name:
                pdf_name = Path(pdf_path).stem
            
            logger.info(f"📄 Processing PDF: {pdf_name}")
            
            # Step 1: Extract text from PDF
            if PDF_READER_CLASS is None:
                return False, "PyPDF2 not available. Please install: pip install PyPDF2"

            # Handle both PyPDF2 2.x and 3.x
            try:
                if hasattr(PDF_READER_CLASS, '__name__') and 'PdfFileReader' in PDF_READER_CLASS.__name__:
                    # PyPDF2 2.x
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PDF_READER_CLASS(file)
                        text = ""
                        for page_num in range(pdf_reader.numPages):
                            page = pdf_reader.getPage(page_num)
                            page_text = page.extractText()
                            text += page_text + "\n"
                            logger.debug(f"   Extracted page {page_num + 1}: {len(page_text)} characters")
                else:
                    # PyPDF2 3.x
                    pdf_reader = PDF_READER_CLASS(pdf_path)
                    text = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        text += page_text + "\n"
                        logger.debug(f"   Extracted page {page_num + 1}: {len(page_text)} characters")
            except Exception as pdf_error:
                logger.error(f"PyPDF2 extraction failed: {pdf_error}")
                return False, f"Failed to extract text from PDF: {str(pdf_error)}"
            
            if not text.strip():
                return False, "No text content found in PDF"
            
            logger.info(f"   ✅ Extracted {len(text)} characters from {len(pdf_reader.pages)} pages")
            
            # Step 2: Split text into chunks
            if LANGCHAIN_AVAILABLE:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)
            else:
                # Use fallback text splitter
                text_splitter = FallbackTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)

            logger.info(f"   ✅ Split into {len(chunks)} chunks")
            
            # Step 3: Check for existing embeddings
            store_name = pdf_name
            pickle_path = os.path.join(self.storage_dir, f"{store_name}.pkl")
            
            if os.path.exists(pickle_path):
                # Load existing embeddings
                with open(pickle_path, "rb") as f:
                    vector_store = pickle.load(f)
                logger.info(f"   ✅ Loaded existing embeddings from disk")
            else:
                # Create new embeddings
                if not self.embeddings:
                    return False, "Embeddings not available"

                if LANGCHAIN_AVAILABLE:
                    vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)

                    # Save embeddings for future use
                    with open(pickle_path, "wb") as f:
                        pickle.dump(vector_store, f)
                    logger.info(f"   ✅ Created and saved new embeddings")
                else:
                    # Fallback: store chunks directly without FAISS
                    vector_store = {
                        'chunks': chunks,
                        'embeddings_available': OFFLINE_EMBEDDINGS_AVAILABLE,
                        'pdf_name': pdf_name
                    }

                    # Save chunks for future use
                    with open(pickle_path, "wb") as f:
                        pickle.dump(vector_store, f)
                    logger.info(f"   ✅ Created and saved text chunks (no vector embeddings)")
            
            # Step 4: Store the vector store
            self.vector_stores[pdf_name] = vector_store
            self.current_pdf_name = pdf_name
            self.current_vector_store = vector_store
            
            logger.info(f"🎉 Successfully processed PDF: {pdf_name}")
            return True, f"Successfully processed {pdf_name} with {len(chunks)} chunks"
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_name}: {e}")
            return False, f"Failed to process PDF: {str(e)}"
    
    def query_pdf(self, query: str, pdf_name: str = None, k: int = 3) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Query a processed PDF using the proven approach.
        
        Args:
            query: Question to ask about the PDF
            pdf_name: Name of the PDF to query (defaults to current)
            k: Number of similar chunks to retrieve
            
        Returns:
            Tuple of (success, response, metadata)
        """
        try:
            # Determine which PDF to query
            if pdf_name and pdf_name in self.vector_stores:
                vector_store = self.vector_stores[pdf_name]
                target_pdf = pdf_name
            elif self.current_vector_store:
                vector_store = self.current_vector_store
                target_pdf = self.current_pdf_name
            else:
                return False, "No PDF has been processed yet", {}
            
            logger.info(f"🔍 Querying PDF '{target_pdf}' with: {query[:50]}...")
            
            # Step 1: Similarity search
            if LANGCHAIN_AVAILABLE and hasattr(vector_store, 'similarity_search'):
                # Use FAISS similarity search
                docs = vector_store.similarity_search(query=query, k=k)

                if not docs:
                    return False, "No relevant content found in the PDF", {}

                logger.info(f"   📄 Found {len(docs)} relevant chunks")

                # Extract content from langchain documents
                relevant_chunks = [doc.page_content.strip() for doc in docs]

            else:
                # Fallback: simple text search in chunks
                if isinstance(vector_store, dict) and 'chunks' in vector_store:
                    chunks = vector_store['chunks']
                    query_lower = query.lower()

                    # Simple relevance scoring based on keyword matching
                    scored_chunks = []
                    for chunk in chunks:
                        chunk_lower = chunk.lower()
                        score = sum(1 for word in query_lower.split() if word in chunk_lower)
                        if score > 0:
                            scored_chunks.append((score, chunk))

                    # Sort by relevance and take top k
                    scored_chunks.sort(reverse=True, key=lambda x: x[0])
                    relevant_chunks = [chunk for _, chunk in scored_chunks[:k]]

                    if not relevant_chunks:
                        return False, "No relevant content found in the PDF", {}

                    logger.info(f"   📄 Found {len(relevant_chunks)} relevant chunks (keyword search)")
                else:
                    return False, "Invalid vector store format", {}

            # Step 2: Generate response using offline approach
            response_parts = []
            response_parts.append(f"Based on the PDF '{target_pdf}', here's what I found:\n")

            for i, chunk_text in enumerate(relevant_chunks, 1):
                response_parts.append(f"\n**Relevant Section {i}:**\n{chunk_text}\n")

            response = "\n".join(response_parts)

            # Prepare metadata
            metadata = {
                "pdf_name": target_pdf,
                "chunks_used": len(relevant_chunks),
                "processing_method": "offline_proven_pdf" if LANGCHAIN_AVAILABLE else "fallback_text_search",
                "chunks_content": [chunk[:200] + "..." for chunk in relevant_chunks]
            }
            
            logger.info(f"   ✅ Generated offline response: {len(response)} characters")
            logger.info(f"   📊 Used {len(docs)} chunks from PDF")

            return True, response, metadata
            
        except Exception as e:
            logger.error(f"Failed to query PDF: {e}")
            return False, f"Failed to query PDF: {str(e)}", {}
    
    def list_processed_pdfs(self) -> List[str]:
        """Get list of processed PDF names."""
        return list(self.vector_stores.keys())
    
    def get_pdf_info(self, pdf_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a processed PDF."""
        if pdf_name not in self.vector_stores:
            return None
        
        vector_store = self.vector_stores[pdf_name]
        
        return {
            "name": pdf_name,
            "vector_store_type": type(vector_store).__name__,
            "is_current": pdf_name == self.current_pdf_name,
            "pickle_path": os.path.join(self.storage_dir, f"{pdf_name}.pkl")
        }
    
    def clear_pdf(self, pdf_name: str) -> bool:
        """Remove a processed PDF from memory."""
        if pdf_name in self.vector_stores:
            del self.vector_stores[pdf_name]
            
            if self.current_pdf_name == pdf_name:
                self.current_pdf_name = None
                self.current_vector_store = None
            
            logger.info(f"🗑️ Cleared PDF: {pdf_name}")
            return True
        
        return False
    
    def clear_all_pdfs(self):
        """Clear all processed PDFs from memory."""
        self.vector_stores.clear()
        self.current_pdf_name = None
        self.current_vector_store = None
        logger.info("🗑️ Cleared all PDFs")

# Global instance
_proven_pdf_processor = None

def get_proven_pdf_processor() -> ProvenPDFProcessor:
    """Get the global proven PDF processor instance."""
    global _proven_pdf_processor
    if _proven_pdf_processor is None:
        _proven_pdf_processor = ProvenPDFProcessor()
    return _proven_pdf_processor

def process_pdf_file(pdf_path: str, pdf_name: str = None) -> Tuple[bool, str]:
    """
    Process a PDF file using the proven approach.
    
    Args:
        pdf_path: Path to the PDF file
        pdf_name: Optional name for the PDF
        
    Returns:
        Tuple of (success, message)
    """
    processor = get_proven_pdf_processor()
    return processor.process_pdf(pdf_path, pdf_name)

def query_current_pdf(query: str, k: int = 3) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Query the currently processed PDF.
    
    Args:
        query: Question to ask
        k: Number of chunks to retrieve
        
    Returns:
        Tuple of (success, response, metadata)
    """
    processor = get_proven_pdf_processor()
    return processor.query_pdf(query, k=k)

def query_specific_pdf(query: str, pdf_name: str, k: int = 3) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Query a specific processed PDF.
    
    Args:
        query: Question to ask
        pdf_name: Name of the PDF to query
        k: Number of chunks to retrieve
        
    Returns:
        Tuple of (success, response, metadata)
    """
    processor = get_proven_pdf_processor()
    return processor.query_pdf(query, pdf_name, k)

if __name__ == "__main__":
    # Test the proven PDF processor
    print("🧪 Testing Proven PDF Processor")
    
    processor = ProvenPDFProcessor()
    
    # Test with a sample PDF (if available)
    test_pdf = "/Users/vinsoncornejo/Downloads/augment-projects/SAM 2/2507.07957v1.pdf"
    
    if os.path.exists(test_pdf):
        print(f"📄 Testing with: {test_pdf}")
        
        # Process PDF
        success, message = processor.process_pdf(test_pdf, "test_pdf")
        print(f"Processing: {success} - {message}")
        
        if success:
            # Query PDF
            query = "What is this document about?"
            success, response, metadata = processor.query_pdf(query)
            print(f"Query: {success}")
            print(f"Response: {response}")
            print(f"Metadata: {metadata}")
    else:
        print(f"❌ Test PDF not found: {test_pdf}")
