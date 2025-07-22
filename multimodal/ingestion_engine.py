"""
Multimodal Ingestion Engine for SAM
Processes images, text, and audio inputs with OCR, captioning, and transcription.

Sprint 9 Task 1: Multimodal Ingestion (Image, Text, Audio)
"""

import logging
import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import base64

logger = logging.getLogger(__name__)

class MediaType(Enum):
    """Types of media that can be ingested."""
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"
    DOCUMENT = "document"

class ProcessingStatus(Enum):
    """Status of media processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class MediaMetadata:
    """Metadata for ingested media."""
    media_id: str
    original_filename: str
    media_type: MediaType
    file_size: int
    file_hash: str
    mime_type: str
    dimensions: Optional[Tuple[int, int]]  # For images
    duration: Optional[float]  # For audio
    created_at: str
    processed_at: Optional[str]
    processing_status: ProcessingStatus
    metadata: Dict[str, Any]

@dataclass
class ProcessingResult:
    """Result of media processing."""
    media_id: str
    media_type: MediaType
    extracted_text: Optional[str]
    caption: Optional[str]
    transcription: Optional[str]
    confidence_scores: Dict[str, float]
    processing_time_ms: int
    source_links: List[str]
    metadata: Dict[str, Any]

class MultimodalIngestionEngine:
    """
    Processes multiple types of media input for SAM.
    """
    
    def __init__(self, storage_directory: str = "multimodal_storage"):
        """
        Initialize the multimodal ingestion engine.
        
        Args:
            storage_directory: Directory for storing processed media
        """
        self.storage_dir = Path(storage_directory)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.storage_dir / "images").mkdir(exist_ok=True)
        (self.storage_dir / "audio").mkdir(exist_ok=True)
        (self.storage_dir / "documents").mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
        
        # Storage
        self.media_registry: Dict[str, MediaMetadata] = {}
        self.processing_results: Dict[str, ProcessingResult] = {}
        
        # Configuration
        self.config = {
            'max_file_size_mb': 50,
            'supported_image_formats': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
            'supported_audio_formats': ['.wav', '.mp3', '.m4a', '.ogg'],
            'supported_document_formats': ['.pdf', '.txt', '.md', '.docx'],
            'enable_ocr': True,
            'enable_captioning': True,
            'enable_transcription': True
        }
        
        logger.info(f"Multimodal ingestion engine initialized with storage in {storage_directory}")
    
    def ingest_image(self, image_data: Union[bytes, str], filename: str,
                    perform_ocr: bool = True, generate_caption: bool = True) -> str:
        """
        Ingest and process an image.
        
        Args:
            image_data: Image data (bytes or base64 string)
            filename: Original filename
            perform_ocr: Whether to perform OCR
            generate_caption: Whether to generate caption
            
        Returns:
            Media ID
        """
        try:
            media_id = f"img_{uuid.uuid4().hex[:12]}"
            
            # Convert base64 to bytes if needed
            if isinstance(image_data, str):
                image_data = base64.b64decode(image_data)
            
            # Validate file size
            file_size = len(image_data)
            if file_size > self.config['max_file_size_mb'] * 1024 * 1024:
                raise ValueError(f"File size {file_size} exceeds maximum {self.config['max_file_size_mb']}MB")
            
            # Calculate file hash
            file_hash = hashlib.sha256(image_data).hexdigest()
            
            # Determine file extension and MIME type
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.config['supported_image_formats']:
                raise ValueError(f"Unsupported image format: {file_ext}")
            
            mime_type = self._get_mime_type(file_ext)
            
            # Save image file
            image_path = self.storage_dir / "images" / f"{media_id}{file_ext}"
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Get image dimensions
            dimensions = self._get_image_dimensions(image_path)
            
            # Create metadata
            metadata = MediaMetadata(
                media_id=media_id,
                original_filename=filename,
                media_type=MediaType.IMAGE,
                file_size=file_size,
                file_hash=file_hash,
                mime_type=mime_type,
                dimensions=dimensions,
                duration=None,
                created_at=datetime.now().isoformat(),
                processed_at=None,
                processing_status=ProcessingStatus.PENDING,
                metadata={'file_path': str(image_path)}
            )
            
            self.media_registry[media_id] = metadata
            
            # Process image
            processing_result = self._process_image(media_id, image_path, perform_ocr, generate_caption)
            
            # Update metadata
            metadata.processed_at = datetime.now().isoformat()
            metadata.processing_status = ProcessingStatus.COMPLETED
            
            # Save metadata
            self._save_metadata(metadata)
            
            logger.info(f"Ingested image: {filename} ({media_id})")
            return media_id
            
        except Exception as e:
            logger.error(f"Error ingesting image {filename}: {e}")
            raise
    
    def ingest_audio(self, audio_data: bytes, filename: str,
                    perform_transcription: bool = True) -> str:
        """
        Ingest and process an audio file.
        
        Args:
            audio_data: Audio data in bytes
            filename: Original filename
            perform_transcription: Whether to perform transcription
            
        Returns:
            Media ID
        """
        try:
            media_id = f"aud_{uuid.uuid4().hex[:12]}"
            
            # Validate file size
            file_size = len(audio_data)
            if file_size > self.config['max_file_size_mb'] * 1024 * 1024:
                raise ValueError(f"File size {file_size} exceeds maximum {self.config['max_file_size_mb']}MB")
            
            # Calculate file hash
            file_hash = hashlib.sha256(audio_data).hexdigest()
            
            # Determine file extension and MIME type
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.config['supported_audio_formats']:
                raise ValueError(f"Unsupported audio format: {file_ext}")
            
            mime_type = self._get_mime_type(file_ext)
            
            # Save audio file
            audio_path = self.storage_dir / "audio" / f"{media_id}{file_ext}"
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            
            # Get audio duration
            duration = self._get_audio_duration(audio_path)
            
            # Create metadata
            metadata = MediaMetadata(
                media_id=media_id,
                original_filename=filename,
                media_type=MediaType.AUDIO,
                file_size=file_size,
                file_hash=file_hash,
                mime_type=mime_type,
                dimensions=None,
                duration=duration,
                created_at=datetime.now().isoformat(),
                processed_at=None,
                processing_status=ProcessingStatus.PENDING,
                metadata={'file_path': str(audio_path)}
            )
            
            self.media_registry[media_id] = metadata
            
            # Process audio
            processing_result = self._process_audio(media_id, audio_path, perform_transcription)
            
            # Update metadata
            metadata.processed_at = datetime.now().isoformat()
            metadata.processing_status = ProcessingStatus.COMPLETED
            
            # Save metadata
            self._save_metadata(metadata)
            
            logger.info(f"Ingested audio: {filename} ({media_id})")
            return media_id
            
        except Exception as e:
            logger.error(f"Error ingesting audio {filename}: {e}")
            raise
    
    def ingest_document(self, document_data: bytes, filename: str) -> str:
        """
        Ingest and process a document.
        
        Args:
            document_data: Document data in bytes
            filename: Original filename
            
        Returns:
            Media ID
        """
        try:
            media_id = f"doc_{uuid.uuid4().hex[:12]}"
            
            # Validate file size
            file_size = len(document_data)
            if file_size > self.config['max_file_size_mb'] * 1024 * 1024:
                raise ValueError(f"File size {file_size} exceeds maximum {self.config['max_file_size_mb']}MB")
            
            # Calculate file hash
            file_hash = hashlib.sha256(document_data).hexdigest()
            
            # Determine file extension and MIME type
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.config['supported_document_formats']:
                raise ValueError(f"Unsupported document format: {file_ext}")
            
            mime_type = self._get_mime_type(file_ext)
            
            # Save document file
            doc_path = self.storage_dir / "documents" / f"{media_id}{file_ext}"
            with open(doc_path, 'wb') as f:
                f.write(document_data)
            
            # Create metadata
            metadata = MediaMetadata(
                media_id=media_id,
                original_filename=filename,
                media_type=MediaType.DOCUMENT,
                file_size=file_size,
                file_hash=file_hash,
                mime_type=mime_type,
                dimensions=None,
                duration=None,
                created_at=datetime.now().isoformat(),
                processed_at=None,
                processing_status=ProcessingStatus.PENDING,
                metadata={'file_path': str(doc_path)}
            )
            
            self.media_registry[media_id] = metadata
            
            # Process document
            processing_result = self._process_document(media_id, doc_path)
            
            # Update metadata
            metadata.processed_at = datetime.now().isoformat()
            metadata.processing_status = ProcessingStatus.COMPLETED
            
            # Save metadata
            self._save_metadata(metadata)
            
            logger.info(f"Ingested document: {filename} ({media_id})")
            return media_id
            
        except Exception as e:
            logger.error(f"Error ingesting document {filename}: {e}")
            raise
    
    def get_processing_result(self, media_id: str) -> Optional[ProcessingResult]:
        """Get processing result for a media item."""
        return self.processing_results.get(media_id)
    
    def get_media_metadata(self, media_id: str) -> Optional[MediaMetadata]:
        """Get metadata for a media item."""
        return self.media_registry.get(media_id)
    
    def search_media(self, query: str, media_types: List[MediaType] = None) -> List[Dict[str, Any]]:
        """
        Search through processed media.
        
        Args:
            query: Search query
            media_types: Optional filter by media types
            
        Returns:
            List of matching media with relevance scores
        """
        try:
            results = []
            query_lower = query.lower()
            
            for media_id, result in self.processing_results.items():
                # Filter by media type if specified
                if media_types and result.media_type not in media_types:
                    continue
                
                relevance_score = 0.0
                
                # Search in extracted text
                if result.extracted_text and query_lower in result.extracted_text.lower():
                    relevance_score += 0.8
                
                # Search in caption
                if result.caption and query_lower in result.caption.lower():
                    relevance_score += 0.6
                
                # Search in transcription
                if result.transcription and query_lower in result.transcription.lower():
                    relevance_score += 0.7
                
                if relevance_score > 0:
                    metadata = self.media_registry.get(media_id)
                    results.append({
                        'media_id': media_id,
                        'filename': metadata.original_filename if metadata else 'Unknown',
                        'media_type': result.media_type.value,
                        'relevance_score': relevance_score,
                        'extracted_text': result.extracted_text[:200] + '...' if result.extracted_text and len(result.extracted_text) > 200 else result.extracted_text,
                        'caption': result.caption,
                        'transcription': result.transcription[:200] + '...' if result.transcription and len(result.transcription) > 200 else result.transcription
                    })
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching media: {e}")
            return []
    
    def _process_image(self, media_id: str, image_path: Path,
                      perform_ocr: bool, generate_caption: bool) -> ProcessingResult:
        """Process an image with OCR and captioning."""
        try:
            start_time = datetime.now()
            
            extracted_text = None
            caption = None
            confidence_scores = {}
            
            # Perform OCR if requested
            if perform_ocr and self.config['enable_ocr']:
                extracted_text, ocr_confidence = self._perform_ocr(image_path)
                confidence_scores['ocr'] = ocr_confidence
            
            # Generate caption if requested
            if generate_caption and self.config['enable_captioning']:
                caption, caption_confidence = self._generate_image_caption(image_path)
                confidence_scores['caption'] = caption_confidence
            
            # Calculate processing time
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create processing result
            result = ProcessingResult(
                media_id=media_id,
                media_type=MediaType.IMAGE,
                extracted_text=extracted_text,
                caption=caption,
                transcription=None,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time_ms,
                source_links=[f"file://{image_path}"],
                metadata={'image_path': str(image_path)}
            )
            
            self.processing_results[media_id] = result
            
            logger.info(f"Processed image {media_id}: OCR={extracted_text is not None}, Caption={caption is not None}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {media_id}: {e}")
            raise
    
    def _process_audio(self, media_id: str, audio_path: Path,
                      perform_transcription: bool) -> ProcessingResult:
        """Process an audio file with transcription."""
        try:
            start_time = datetime.now()
            
            transcription = None
            confidence_scores = {}
            
            # Perform transcription if requested
            if perform_transcription and self.config['enable_transcription']:
                transcription, transcription_confidence = self._perform_transcription(audio_path)
                confidence_scores['transcription'] = transcription_confidence
            
            # Calculate processing time
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create processing result
            result = ProcessingResult(
                media_id=media_id,
                media_type=MediaType.AUDIO,
                extracted_text=None,
                caption=None,
                transcription=transcription,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time_ms,
                source_links=[f"file://{audio_path}"],
                metadata={'audio_path': str(audio_path)}
            )
            
            self.processing_results[media_id] = result
            
            logger.info(f"Processed audio {media_id}: Transcription={transcription is not None}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio {media_id}: {e}")
            raise
    
    def _process_document(self, media_id: str, doc_path: Path) -> ProcessingResult:
        """Process a document with text extraction."""
        try:
            start_time = datetime.now()
            
            extracted_text, extraction_confidence = self._extract_document_text(doc_path)
            
            # Calculate processing time
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create processing result
            result = ProcessingResult(
                media_id=media_id,
                media_type=MediaType.DOCUMENT,
                extracted_text=extracted_text,
                caption=None,
                transcription=None,
                confidence_scores={'extraction': extraction_confidence},
                processing_time_ms=processing_time_ms,
                source_links=[f"file://{doc_path}"],
                metadata={'document_path': str(doc_path)}
            )
            
            self.processing_results[media_id] = result
            
            logger.info(f"Processed document {media_id}: Text extracted={extracted_text is not None}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {media_id}: {e}")
            raise
    
    def _perform_ocr(self, image_path: Path) -> Tuple[str, float]:
        """Perform OCR on an image (placeholder implementation)."""
        # This would integrate with actual OCR libraries like Tesseract, EasyOCR, etc.
        # For now, return simulated results
        
        logger.info(f"Performing OCR on {image_path}")
        
        # Simulate OCR processing
        extracted_text = f"[OCR] Simulated text extraction from {image_path.name}"
        confidence = 0.85
        
        return extracted_text, confidence
    
    def _generate_image_caption(self, image_path: Path) -> Tuple[str, float]:
        """Generate caption for an image (placeholder implementation)."""
        # This would integrate with image captioning models like BLIP, CLIP, etc.
        # For now, return simulated results
        
        logger.info(f"Generating caption for {image_path}")
        
        # Simulate caption generation
        caption = f"An image showing content from {image_path.name}"
        confidence = 0.75
        
        return caption, confidence
    
    def _perform_transcription(self, audio_path: Path) -> Tuple[str, float]:
        """Perform speech-to-text transcription (placeholder implementation)."""
        # This would integrate with speech recognition like Whisper, Vosk, etc.
        # For now, return simulated results
        
        logger.info(f"Performing transcription on {audio_path}")
        
        # Simulate transcription
        transcription = f"[TRANSCRIPTION] Simulated speech-to-text from {audio_path.name}"
        confidence = 0.80
        
        return transcription, confidence
    
    def _extract_document_text(self, doc_path: Path) -> Tuple[str, float]:
        """Extract text from a document (placeholder implementation)."""
        # This would integrate with document processing libraries like PyPDF2, python-docx, etc.
        # For now, return simulated results
        
        logger.info(f"Extracting text from {doc_path}")
        
        file_ext = doc_path.suffix.lower()
        
        if file_ext == '.txt':
            # Read plain text file
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return text, 1.0
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                return f"Error reading {doc_path.name}", 0.0
        else:
            # Simulate extraction for other formats
            extracted_text = f"[EXTRACTED] Simulated text extraction from {doc_path.name}"
            confidence = 0.90
            
            return extracted_text, confidence
    
    def _get_image_dimensions(self, image_path: Path) -> Optional[Tuple[int, int]]:
        """Get image dimensions (placeholder implementation)."""
        # This would use PIL or similar library
        # For now, return simulated dimensions
        return (800, 600)
    
    def _get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """Get audio duration (placeholder implementation)."""
        # This would use librosa, pydub, or similar library
        # For now, return simulated duration
        return 30.5  # seconds
    
    def _get_mime_type(self, file_ext: str) -> str:
        """Get MIME type for file extension."""
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.m4a': 'audio/mp4',
            '.ogg': 'audio/ogg',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        return mime_types.get(file_ext, 'application/octet-stream')
    
    def _save_metadata(self, metadata: MediaMetadata):
        """Save metadata to file."""
        try:
            metadata_file = self.storage_dir / "metadata" / f"{metadata.media_id}.json"
            
            metadata_dict = asdict(metadata)
            metadata_dict['media_type'] = metadata.media_type.value
            metadata_dict['processing_status'] = metadata.processing_status.value
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved metadata for {metadata.media_id}")
            
        except Exception as e:
            logger.error(f"Error saving metadata for {metadata.media_id}: {e}")

# Global multimodal ingestion engine instance
_ingestion_engine = None

def get_ingestion_engine(storage_directory: str = "multimodal_storage") -> MultimodalIngestionEngine:
    """Get or create a global multimodal ingestion engine instance."""
    global _ingestion_engine
    
    if _ingestion_engine is None:
        _ingestion_engine = MultimodalIngestionEngine(storage_directory=storage_directory)
    
    return _ingestion_engine
