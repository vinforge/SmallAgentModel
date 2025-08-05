"""
Enhanced Document Parser using Docling
Provides superior PDF and document processing capabilities.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import mimetypes

logger = logging.getLogger(__name__)

class DoclingDocumentParser:
    """Enhanced document parser using Docling for superior document processing."""
    
    def __init__(self):
        """Initialize the Docling document parser."""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
            
            # Configure pipeline options for better PDF processing
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs
            pipeline_options.do_table_structure = True  # Extract table structure
            pipeline_options.table_structure_options.do_cell_matching = True
            
            # Initialize converter with enhanced options
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # Prioritize text files and add fallback handling
            self.supported_formats = {
                '.txt': 'text/plain',
                '.md': 'text/markdown',
                '.pdf': 'application/pdf',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.html': 'text/html',
                '.htm': 'text/html',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.tiff': 'image/tiff',
                '.wav': 'audio/wav',
                '.mp3': 'audio/mpeg'
            }
            
            # Text file extensions that should use direct text reading
            self.text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.htm', '.css', '.json', '.xml', '.csv'}
            
            logger.info("✅ Docling document parser initialized successfully")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import Docling: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Docling parser: {e}")
            raise
    
    def parse_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a document using Docling.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing parsed document information
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check if file format is supported
            file_extension = file_path.suffix.lower()
            
            # Handle text files directly without Docling
            if file_extension in self.text_extensions:
                logger.info(f"🔄 Processing text file directly: {file_path.name}")
                return self._parse_text_file(file_path)
            
            # Use Docling for other supported formats
            if file_extension not in self.supported_formats:
                logger.warning(f"Unsupported file format: {file_extension}")
                return self._fallback_parse(file_path)
            
            logger.info(f"🔄 Processing document with Docling: {file_path.name}")
            
            try:
                # Convert document using Docling
                result = self.converter.convert(str(file_path))
            except Exception as docling_error:
                logger.warning(f"Docling failed for {file_path.name}: {docling_error}")
                return self._fallback_parse(file_path, error=str(docling_error))
            
            # Extract comprehensive information
            document_info = {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_type': self.supported_formats.get(file_extension, 'unknown'),
                'success': True,
                'parser': 'docling',
                'content_blocks': [],
                'metadata': {},
                'full_text': '',
                'tables': [],
                'images': [],
                'structure': {}
            }
            
            # Extract main content as markdown
            markdown_content = result.document.export_to_markdown()
            document_info['full_text'] = markdown_content
            document_info['markdown'] = markdown_content
            
            # Extract structured content blocks
            content_blocks = []
            
            # Process document elements
            for element in result.document.main_text:
                block = {
                    'type': element.label if hasattr(element, 'label') else 'text',
                    'content': element.text if hasattr(element, 'text') else str(element),
                    'bbox': getattr(element, 'bbox', None),
                    'page': getattr(element, 'page', None)
                }
                content_blocks.append(block)
            
            document_info['content_blocks'] = content_blocks
            
            # Extract tables if present
            if hasattr(result.document, 'tables') and result.document.tables:
                tables = []
                for table in result.document.tables:
                    table_data = {
                        'content': str(table),
                        'bbox': getattr(table, 'bbox', None),
                        'page': getattr(table, 'page', None)
                    }
                    tables.append(table_data)
                document_info['tables'] = tables
            
            # Extract metadata
            if hasattr(result.document, 'meta'):
                document_info['metadata'] = {
                    'title': getattr(result.document.meta, 'title', ''),
                    'authors': getattr(result.document.meta, 'authors', []),
                    'language': getattr(result.document.meta, 'language', ''),
                    'page_count': len(getattr(result.document, 'pages', [])),
                }
            
            # Document structure information
            document_info['structure'] = {
                'page_count': len(getattr(result.document, 'pages', [])),
                'element_count': len(content_blocks),
                'table_count': len(document_info['tables']),
                'has_images': len(document_info['images']) > 0
            }
            
            logger.info(f"✅ Successfully parsed {file_path.name}: {len(content_blocks)} blocks, {len(document_info['tables'])} tables")
            
            return document_info
            
        except Exception as e:
            logger.error(f"❌ Docling parsing failed for {file_path}: {e}")
            
            # Fallback to basic text extraction
            return self._fallback_parse(file_path, error=str(e))
    

    def _parse_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse text files directly without Docling."""
        try:
            # Read text content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Split into paragraphs for content blocks
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            content_blocks = []
            for i, paragraph in enumerate(paragraphs):
                block = {
                    'type': 'paragraph',
                    'content': paragraph,
                    'bbox': None,
                    'page': 1
                }
                content_blocks.append(block)
            
            # Extract basic metadata
            lines = content.split('\n')
            title = lines[0].strip() if lines else file_path.stem
            
            return {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_type': 'text/plain',
                'success': True,
                'parser': 'text_direct',
                'content_blocks': content_blocks,
                'metadata': {
                    'title': title,
                    'line_count': len(lines),
                    'paragraph_count': len(paragraphs)
                },
                'full_text': content,
                'markdown': content,  # For text files, content is already readable
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
            return self._fallback_parse(file_path, error=str(e))
\n    def _fallback_parse(self, file_path: Path, error: str = None) -> Dict[str, Any]:
        """Fallback parsing for unsupported formats or when Docling fails."""
        try:
            logger.info(f"🔄 Using fallback parser for: {file_path.name}")
            
            # Try basic text extraction
            content = ""
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            elif file_extension == '.pdf':
                # Try PyPDF2 as fallback
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        content = "\n".join([page.extract_text() for page in reader.pages])
                except Exception as pdf_error:
                    logger.warning(f"PyPDF2 fallback failed: {pdf_error}")
                    content = f"Failed to extract content from PDF: {pdf_error}"
            else:
                content = f"Unsupported file format: {file_extension}"
            
            return {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_type': mimetypes.guess_type(str(file_path))[0] or 'unknown',
                'success': True,
                'parser': 'fallback',
                'content_blocks': [{'type': 'text', 'content': content}],
                'metadata': {'fallback_reason': error or 'unsupported_format'},
                'full_text': content,
                'tables': [],
                'images': [],
                'structure': {'page_count': 1, 'element_count': 1}
            }
            
        except Exception as fallback_error:
            logger.error(f"❌ Fallback parsing also failed: {fallback_error}")
            
            return {
                'filename': file_path.name,
                'file_path': str(file_path),
                'success': False,
                'parser': 'failed',
                'error': str(fallback_error),
                'content_blocks': [],
                'full_text': '',
                'metadata': {},
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
    """Get the Docling document parser instance."""
    return DoclingDocumentParser()
