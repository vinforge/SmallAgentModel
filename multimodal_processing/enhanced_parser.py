"""
Enhanced Document Parser with Improved Text Handling
Provides robust document processing with graceful fallbacks.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import mimetypes

logger = logging.getLogger(__name__)

class EnhancedDocumentParser:
    """Enhanced document parser with improved text handling and fallbacks."""
    
    def __init__(self):
        """Initialize the enhanced document parser."""
        self.supported_formats = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.py': 'text/x-python',
            '.js': 'text/javascript',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.css': 'text/css',
            '.json': 'application/json',
            '.xml': 'text/xml',
            '.csv': 'text/csv',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword'
        }
        
        # Text file extensions that should use direct text reading
        self.text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.htm', '.css', '.json', '.xml', '.csv'}
        
        # Try to initialize Docling for PDF processing
        self.docling_available = False
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
            
            # Configure pipeline options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            self.docling_available = True
            logger.info("✅ Docling initialized for PDF processing")
            
        except ImportError:
            logger.info("⚠️  Docling not available, using fallback for PDFs")
        except Exception as e:
            logger.warning(f"⚠️  Docling initialization failed: {e}")
    
    def parse_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a document using the most appropriate method.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing parsed document information
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = file_path.suffix.lower()
            
            # Handle text files directly
            if file_extension in self.text_extensions:
                logger.info(f"🔄 Processing text file directly: {file_path.name}")
                return self._parse_text_file(file_path)
            
            # Handle PDFs with Docling if available
            elif file_extension == '.pdf' and self.docling_available:
                logger.info(f"🔄 Processing PDF with Docling: {file_path.name}")
                return self._parse_pdf_with_docling(file_path)
            
            # Handle other formats or fallback
            else:
                logger.info(f"🔄 Using fallback parser for: {file_path.name}")
                return self._fallback_parse(file_path)
                
        except Exception as e:
            logger.error(f"❌ Document parsing failed for {file_path}: {e}")
            return self._create_error_result(file_path, str(e))
    
    def _parse_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse text files directly."""
        try:
            # Read text content with multiple encoding attempts
            content = ""
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if not content:
                raise ValueError("Could not decode file with any supported encoding")
            
            # Split into meaningful content blocks
            content_blocks = self._create_content_blocks(content)
            
            # Extract basic metadata
            lines = content.split('\n')
            title = self._extract_title(lines)
            
            return {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_type': self.supported_formats.get(file_path.suffix.lower(), 'text/plain'),
                'success': True,
                'parser': 'text_direct',
                'content_blocks': content_blocks,
                'metadata': {
                    'title': title,
                    'line_count': len(lines),
                    'paragraph_count': len(content_blocks),
                    'encoding': 'utf-8'
                },
                'full_text': content,
                'markdown': content,
                'tables': [],
                'images': [],
                'structure': {
                    'page_count': 1,
                    'element_count': len(content_blocks),
                    'table_count': 0,
                    'has_images': False
                }
            }
            
        except Exception as e:
            logger.error(f"Text file parsing failed: {e}")
            return self._create_error_result(file_path, str(e))
    
    def _parse_pdf_with_docling(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF files using Docling."""
        try:
            # Convert document using Docling
            result = self.converter.convert(str(file_path))
            
            # Extract content as markdown
            markdown_content = result.document.export_to_markdown()
            
            # Create content blocks from document elements
            content_blocks = []
            if hasattr(result.document, 'main_text'):
                for element in result.document.main_text:
                    block = {
                        'type': getattr(element, 'label', 'text'),
                        'content': getattr(element, 'text', str(element)),
                        'bbox': getattr(element, 'bbox', None),
                        'page': getattr(element, 'page', None)
                    }
                    content_blocks.append(block)
            
            # Extract metadata
            metadata = {}
            if hasattr(result.document, 'meta'):
                metadata = {
                    'title': getattr(result.document.meta, 'title', ''),
                    'authors': getattr(result.document.meta, 'authors', []),
                    'language': getattr(result.document.meta, 'language', ''),
                    'page_count': len(getattr(result.document, 'pages', [])),
                }
            
            return {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_type': 'application/pdf',
                'success': True,
                'parser': 'docling',
                'content_blocks': content_blocks,
                'metadata': metadata,
                'full_text': markdown_content,
                'markdown': markdown_content,
                'tables': [],
                'images': [],
                'structure': {
                    'page_count': metadata.get('page_count', 1),
                    'element_count': len(content_blocks),
                    'table_count': 0,
                    'has_images': False
                }
            }
            
        except Exception as e:
            logger.warning(f"Docling PDF parsing failed: {e}")
            return self._fallback_parse(file_path, error=str(e))
    
    def _fallback_parse(self, file_path: Path, error: str = None) -> Dict[str, Any]:
        """Fallback parsing for when other methods fail."""
        try:
            content = ""
            file_extension = file_path.suffix.lower()
            
            if file_extension in self.text_extensions:
                # Try basic text reading
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except:
                    with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                        content = f.read()
            
            elif file_extension == '.pdf':
                # Try PyPDF2 as fallback
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        content = "\n".join([page.extract_text() for page in reader.pages])
                except Exception as pdf_error:
                    content = f"PDF content extraction failed: {pdf_error}"
            
            else:
                content = f"Unsupported file format: {file_extension}"
            
            content_blocks = self._create_content_blocks(content) if content else []
            
            return {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_type': self.supported_formats.get(file_extension, 'unknown'),
                'success': True,
                'parser': 'fallback',
                'content_blocks': content_blocks,
                'metadata': {'fallback_reason': error or 'unsupported_format'},
                'full_text': content,
                'markdown': content,
                'tables': [],
                'images': [],
                'structure': {
                    'page_count': 1,
                    'element_count': len(content_blocks),
                    'table_count': 0,
                    'has_images': False
                }
            }
            
        except Exception as fallback_error:
            logger.error(f"Fallback parsing failed: {fallback_error}")
            return self._create_error_result(file_path, str(fallback_error))
    
    def _create_content_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Create content blocks from text content."""
        if not content.strip():
            return []
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        content_blocks = []
        for i, paragraph in enumerate(paragraphs):
            # Determine block type based on content
            block_type = 'paragraph'
            if paragraph.startswith('#'):
                block_type = 'heading'
            elif paragraph.startswith('-') or paragraph.startswith('•'):
                block_type = 'list'
            elif ':' in paragraph and len(paragraph.split('\n')) == 1:
                block_type = 'metadata'
            
            block = {
                'type': block_type,
                'content': paragraph,
                'bbox': None,
                'page': 1
            }
            content_blocks.append(block)
        
        return content_blocks
    
    def _extract_title(self, lines: List[str]) -> str:
        """Extract title from document lines."""
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 5:
                return line
        return "Untitled Document"
    
    def _create_error_result(self, file_path: Path, error: str) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'file_type': 'unknown',
            'success': False,
            'parser': 'error',
            'error': error,
            'content_blocks': [],
            'metadata': {},
            'full_text': '',
            'markdown': '',
            'tables': [],
            'images': [],
            'structure': {}
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.supported_formats.keys())
    
    def is_supported(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported."""
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.supported_formats

# Factory function for backward compatibility
def get_document_parser():
    """Get the enhanced document parser instance."""
    return EnhancedDocumentParser()

# Alias for compatibility
DocumentParser = EnhancedDocumentParser
