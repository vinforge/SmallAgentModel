"""
RAGFlow Client
Low-level client for interacting with RAGFlow API.
"""

import requests
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

class RAGFlowClient:
    """
    Low-level client for RAGFlow API interactions.
    
    Provides direct access to RAGFlow's REST API for:
    - Knowledge base management
    - Document upload and processing
    - Query and retrieval operations
    - System configuration
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:9380/api/v1",
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize RAGFlow client.
        
        Args:
            base_url: RAGFlow API base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set default headers
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })
        
        logger.info(f"RAGFlow client initialized with base URL: {base_url}")
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     data: Optional[Dict] = None,
                     files: Optional[Dict] = None,
                     params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to RAGFlow API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without base URL)
            data: JSON data for request body
            files: Files for multipart upload
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            requests.RequestException: If request fails after retries
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                # Prepare request arguments
                kwargs = {
                    'timeout': self.timeout,
                    'params': params
                }
                
                if files:
                    # For file uploads, don't set Content-Type header
                    headers = {k: v for k, v in self.session.headers.items() 
                              if k.lower() != 'content-type'}
                    kwargs['files'] = files
                    kwargs['headers'] = headers
                    if data:
                        kwargs['data'] = data
                else:
                    # For JSON requests
                    if data:
                        kwargs['json'] = data
                
                # Make request
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                
                # Parse response
                if response.content:
                    return response.json()
                else:
                    return {'success': True}
                    
            except requests.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"All {self.max_retries} attempts failed for {method} {url}")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def create_knowledge_base(self, 
                            name: str,
                            description: str = "",
                            embedding_model: str = "BAAI/bge-large-en-v1.5",
                            chunk_template: str = "intelligent",
                            language: str = "English") -> Dict[str, Any]:
        """
        Create a new knowledge base in RAGFlow.
        
        Args:
            name: Knowledge base name
            description: Knowledge base description
            embedding_model: Embedding model to use
            chunk_template: Chunking template
            language: Primary language
            
        Returns:
            Knowledge base creation response
        """
        data = {
            'name': name,
            'description': description,
            'embedding_model': embedding_model,
            'chunk_template': chunk_template,
            'language': language
        }
        
        logger.info(f"Creating knowledge base: {name}")
        return self._make_request('POST', '/knowledge_bases', data=data)
    
    def upload_document(self, 
                       knowledge_base_id: str,
                       file_path: Union[str, Path],
                       chunk_template: Optional[str] = None,
                       metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Upload document to RAGFlow knowledge base.
        
        Args:
            knowledge_base_id: Target knowledge base ID
            file_path: Path to document file
            chunk_template: Override chunking template
            metadata: Additional metadata
            
        Returns:
            Upload response with document ID
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Prepare file upload
        files = {
            'file': (file_path.name, open(file_path, 'rb'))
        }
        
        # Prepare form data
        data = {
            'knowledge_base_id': knowledge_base_id
        }
        
        if chunk_template:
            data['chunk_template'] = chunk_template
        
        if metadata:
            data['metadata'] = json.dumps(metadata)
        
        try:
            logger.info(f"Uploading document: {file_path.name}")
            return self._make_request('POST', '/documents', data=data, files=files)
        finally:
            # Close file handle
            files['file'][1].close()
    
    def query_knowledge_base(self, 
                           knowledge_base_id: str,
                           query: str,
                           max_results: int = 5,
                           similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Query RAGFlow knowledge base for relevant documents.
        
        Args:
            knowledge_base_id: Knowledge base to query
            query: Search query
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            Query results with chunks and citations
        """
        data = {
            'knowledge_base_id': knowledge_base_id,
            'query': query,
            'max_results': max_results,
            'similarity_threshold': similarity_threshold
        }
        
        logger.debug(f"Querying knowledge base {knowledge_base_id}: {query}")
        return self._make_request('POST', '/query', data=data)
    
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """
        Get document processing status.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document status information
        """
        return self._make_request('GET', f'/documents/{document_id}/status')
    
    def get_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """
        Get document chunks for review and intervention.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document chunks with metadata
        """
        return self._make_request('GET', f'/documents/{document_id}/chunks')
    
    def update_chunk(self, 
                    chunk_id: str,
                    content: Optional[str] = None,
                    keywords: Optional[List[str]] = None,
                    metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Update document chunk content or metadata.
        
        Args:
            chunk_id: Chunk ID to update
            content: New chunk content
            keywords: Keywords to add
            metadata: Additional metadata
            
        Returns:
            Update response
        """
        data = {}
        
        if content is not None:
            data['content'] = content
        
        if keywords is not None:
            data['keywords'] = keywords
        
        if metadata is not None:
            data['metadata'] = metadata
        
        logger.info(f"Updating chunk: {chunk_id}")
        return self._make_request('PUT', f'/chunks/{chunk_id}', data=data)
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete document from knowledge base.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Deletion response
        """
        logger.info(f"Deleting document: {document_id}")
        return self._make_request('DELETE', f'/documents/{document_id}')
    
    def health_check(self) -> bool:
        """
        Check if RAGFlow service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = self._make_request('GET', '/health')
            return response.get('status') == 'healthy'
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
