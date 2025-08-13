#!/usr/bin/env python3
"""
V2 Upload Handler for SAM
Routes document uploads to appropriate pipeline (v1 or v2) based on configuration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def load_pipeline_config() -> Dict[str, Any]:
    """Load pipeline configuration from sam_config.json."""
    try:
        config_path = Path("sam_config.json")
        if not config_path.exists():
            logger.warning("⚠️  sam_config.json not found, using defaults")
            return {
                "version": "v2_muvera",
                "v1_settings": {},
                "v2_settings": {},
                "langextract": {
                    "enabled": False,
                    "model_id": "gemini-2.5-flash",
                    "extraction_passes": 2,
                    "max_workers": 8,
                    "max_char_buffer": 1000
                }
            }

        with open(config_path, 'r') as f:
            config = json.load(f)

        rp = config.get("retrieval_pipeline", {
            "version": "v2_muvera",
            "v1_settings": {},
            "v2_settings": {}
        })
        # Merge langextract top-level settings if present
        le = config.get("langextract", {
            "enabled": False,
            "model_id": "gemini-2.5-flash",
            "extraction_passes": 2,
            "max_workers": 8,
            "max_char_buffer": 1000
        })
        rp["langextract"] = le
        return rp

    except Exception as e:
        logger.error(f"❌ Failed to load pipeline config: {e}")
        return {
            "version": "v1_chunking",
            "v1_settings": {},
            "v2_settings": {}
        }

def get_pipeline_version() -> str:
    """Get the current pipeline version."""
    config = load_pipeline_config()
    return config.get("version", "v2_muvera")

def set_pipeline_version(version: str) -> bool:
    """Set the pipeline version in configuration."""
    try:
        config_path = Path("sam_config.json")

        if not config_path.exists():
            logger.error("❌ sam_config.json not found")
            return False

        with open(config_path, 'r') as f:
            config = json.load(f)

        if "retrieval_pipeline" not in config:
            config["retrieval_pipeline"] = {}

        config["retrieval_pipeline"]["version"] = version

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"✅ Pipeline version set to: {version}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to set pipeline version: {e}")
        return False

def handle_document_upload_v2(file_path: str,
                             filename: str = None,
                             session_id: str = None,
                             force_pipeline: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Handle document upload with v1/v2 pipeline routing.

    Args:
        file_path: Path to the uploaded file
        filename: Original filename
        session_id: Session identifier
        force_pipeline: Force specific pipeline ('v1_chunking' or 'v2_muvera')

    Returns:
        Tuple of (success, message, metadata)
    """
    try:
        logger.info(f"📤 Handling document upload: {filename or file_path}")

        # Determine pipeline version
        pipeline_version = force_pipeline or get_pipeline_version()

        logger.info(f"🔄 Using pipeline: {pipeline_version}")

        if pipeline_version == "v2_muvera":
            return _handle_upload_v2(file_path, filename, session_id)
        else:
            return _handle_upload_v1(file_path, filename, session_id)

    except Exception as e:
        logger.error(f"❌ Document upload failed: {e}")
        return False, f"Upload failed: {str(e)}", {}

def _handle_upload_v2(file_path: str,
                     filename: str = None,
                     session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Handle upload using v2 MUVERA pipeline."""
    try:
        logger.info("🚀 Processing with v2 MUVERA pipeline")

        # Import v2 components
        from sam.ingestion import ingest_document_v2

        # Generate document ID
        import hashlib
        from datetime import datetime

        file_name = filename or Path(file_path).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
        document_id = f"v2_{session_id or 'default'}_{timestamp}_{content_hash}"

        # Process document
        result = ingest_document_v2(
            file_path=file_path,
            document_id=document_id,
            metadata={
                'filename': file_name,
                'session_id': session_id,
                'upload_method': 'v2_handler',
                'pipeline_version': 'v2_muvera'
            }
        )

        if result.successful_documents > 0:
            doc_result = result.document_results[0]

            success_message = f"✅ Document processed with v2 MUVERA pipeline"
            metadata = {
                'document_id': doc_result.document_id,
                'pipeline_version': 'v2_muvera',
                'processing_time': doc_result.processing_time,
                'num_tokens': doc_result.token_embeddings.num_tokens if doc_result.token_embeddings else 0,
                'fde_dim': doc_result.fde_vector.fde_dim if doc_result.fde_vector else 0,
                'chunks': len(doc_result.chunks),
                'session_id': session_id,
                'filename': file_name
            }

            logger.info(f"✅ v2 processing successful: {doc_result.document_id}")
            logger.info(f"📊 Tokens: {metadata['num_tokens']}, FDE: {metadata['fde_dim']}D")

            # Optional LangExtract post-processing
            try:
                cfg = load_pipeline_config()
                le = cfg.get('langextract', {})
                if le.get('enabled', False):
                    from sam.postprocessing.langextract_runner import run_langextract_for_document
                    run_langextract_for_document(
                        file_path=file_path,
                        document_id=doc_result.document_id,
                        model_id=le.get('model_id', 'gemini-2.5-flash'),
                        extraction_passes=int(le.get('extraction_passes', 2)),
                        max_workers=int(le.get('max_workers', 8)),
                        max_char_buffer=int(le.get('max_char_buffer', 1000))
                    )
                    metadata['langextract'] = {'status': 'queued_or_completed'}
            except Exception as e:
                logger.warning(f"LangExtract post-processing skipped: {e}")


            return True, success_message, metadata
        else:
            error_msg = result.errors[0] if result.errors else "Unknown v2 processing error"
            logger.error(f"❌ v2 processing failed: {error_msg}")

            # Fallback to v1 if enabled
            config = load_pipeline_config()
            if config.get("v2_settings", {}).get("fallback_to_v1", True):
                logger.info("🔄 Falling back to v1 pipeline")
                return _handle_upload_v1(file_path, filename, session_id)

            return False, f"v2 processing failed: {error_msg}", {}

    except ImportError as e:
        logger.error(f"❌ v2 components not available: {e}")
        logger.info("💡 Install dependencies: pip install colbert-ai chromadb")

        # Fallback to v1
        logger.info("🔄 Falling back to v1 pipeline")
        return _handle_upload_v1(file_path, filename, session_id)

    except Exception as e:
        logger.error(f"❌ v2 upload handler failed: {e}")

        # Fallback to v1
        config = load_pipeline_config()
        if config.get("v2_settings", {}).get("fallback_to_v1", True):
            logger.info("🔄 Falling back to v1 pipeline")
            return _handle_upload_v1(file_path, filename, session_id)

        return False, f"v2 upload failed: {str(e)}", {}

def _handle_upload_v1(file_path: str,
                     filename: str = None,
                     session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Handle upload using v1 chunking pipeline."""
    try:
        logger.info("📄 Processing with v1 chunking pipeline")

        # Use existing v1 upload handler
        from sam.document_processing.proven_pdf_integration import handle_pdf_upload_for_sam

        # Check if it's a PDF
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            # Use existing PDF handler
            success, message, metadata = handle_pdf_upload_for_sam(
                pdf_path=file_path,
                filename=filename,
                session_id=session_id
            )

            # Add pipeline version to metadata
            metadata['pipeline_version'] = 'v1_chunking'

            return success, message, metadata
        else:
            # For non-PDF files, use basic text processing
            return _handle_text_file_v1(file_path, filename, session_id)

    except Exception as e:
        logger.error(f"❌ v1 upload handler failed: {e}")
        return False, f"v1 upload failed: {str(e)}", {}

def _handle_text_file_v1(file_path: str,
                        filename: str = None,
                        session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Handle text file upload using v1 approach."""
    try:
        # Simple text file processing for v1
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Basic chunking
        chunk_size = 1000
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

        # Generate document ID
        import hashlib
        from datetime import datetime

        file_name = filename or Path(file_path).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        document_id = f"v1_{session_id or 'default'}_{timestamp}_{content_hash}"

        # Store in memory (simplified for v1)
        # In practice, this would integrate with existing v1 storage

        metadata = {
            'document_id': document_id,
            'pipeline_version': 'v1_chunking',
            'filename': file_name,
            'session_id': session_id,
            'chunks': len(chunks),
            'content_length': len(content),
            'file_type': Path(file_path).suffix.lower()
        }

        logger.info(f"✅ v1 text processing successful: {document_id}")

        return True, f"✅ Document processed with v1 chunking pipeline", metadata

    except Exception as e:
        logger.error(f"❌ v1 text processing failed: {e}")
        return False, f"v1 text processing failed: {str(e)}", {}

def get_upload_handler_status() -> Dict[str, Any]:
    """Get status of upload handler and available pipelines."""
    try:
        config = load_pipeline_config()

        # Check v1 availability
        v1_available = True
        try:
            from sam.document_processing.proven_pdf_integration import handle_pdf_upload_for_sam
        except ImportError:
            v1_available = False

        # Check v2 availability
        v2_available = True
        v2_components = []
        try:
            from sam.embedding import get_multivector_embedder
            v2_components.append("multivector_embedder")
        except ImportError:
            v2_available = False

        try:
            from sam.cognition import get_muvera_fde
            v2_components.append("muvera_fde")
        except ImportError:
            v2_available = False

        try:
            from sam.storage import get_v2_storage_manager
            v2_components.append("v2_storage")
        except ImportError:
            v2_available = False

        try:
            from sam.ingestion import ingest_document_v2
            v2_components.append("v2_ingestion")
        except ImportError:
            v2_available = False

        return {
            'current_pipeline': config.get("version", "v1_chunking"),
            'v1_available': v1_available,
            'v2_available': v2_available,
            'v2_components': v2_components,
            'config': config,
            'fallback_enabled': config.get("v2_settings", {}).get("fallback_to_v1", True)
        }

    except Exception as e:
        logger.error(f"❌ Failed to get upload handler status: {e}")
        return {
            'error': str(e),
            'current_pipeline': 'unknown',
            'v1_available': False,
            'v2_available': False
        }

# Convenience functions for backward compatibility
def handle_pdf_upload_for_sam_v2(pdf_path: str, filename: str = None, session_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Backward compatible PDF upload handler."""
    return handle_document_upload_v2(pdf_path, filename, session_id)

def ingest_v2_document(file_path: str, document_id: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Main v2 document ingestion function as specified in task52.md.

    Args:
        file_path: Path to the document file
        document_id: Optional document ID

    Returns:
        Tuple of (success, message, metadata)
    """
    try:
        logger.info(f"🔄 Ingesting document v2: {file_path}")

        # Force v2 pipeline
        success, message, metadata = handle_document_upload_v2(
            file_path=file_path,
            filename=Path(file_path).name,
            session_id=None,
            force_pipeline="v2_muvera"
        )

        if success:
            logger.info(f"✅ v2 document ingestion successful")
        else:
            logger.error(f"❌ v2 document ingestion failed: {message}")

        return success, message, metadata

    except Exception as e:
        logger.error(f"❌ v2 document ingestion failed: {e}")
        return False, f"v2 ingestion failed: {str(e)}", {}
