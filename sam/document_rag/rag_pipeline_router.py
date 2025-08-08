#!/usr/bin/env python3
"""
RAG Pipeline Router for SAM
Routes queries between v1 and v2 RAG pipelines based on configuration and availability.
"""

import logging
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Global router instance
_rag_pipeline_router = None

class PipelineSelection(Enum):
    """Pipeline selection options."""
    V1_CHUNKING = "v1_chunking"
    V2_MUVERA = "v2_muvera"
    AUTO = "auto"

@dataclass
class RoutingResult:
    """Result from pipeline routing."""
    selected_pipeline: PipelineSelection
    rag_result: Any                     # V1 or V2 RAG result
    routing_reason: str                 # Why this pipeline was selected
    fallback_used: bool                 # Whether fallback was used
    processing_time: float              # Total processing time
    pipeline_available: Dict[str, bool] # Pipeline availability status

class RAGPipelineRouter:
    """
    Router that selects between v1 and v2 RAG pipelines.
    
    Handles:
    - Configuration-based routing
    - Automatic fallback when pipelines fail
    - Pipeline availability detection
    - Performance monitoring
    """
    
    def __init__(self, 
                 default_pipeline: PipelineSelection = PipelineSelection.AUTO,
                 enable_fallback: bool = True,
                 config_file: str = "sam_config.json"):
        """
        Initialize the RAG pipeline router.
        
        Args:
            default_pipeline: Default pipeline to use
            enable_fallback: Enable automatic fallback
            config_file: Configuration file path
        """
        self.default_pipeline = default_pipeline
        self.enable_fallback = enable_fallback
        self.config_file = config_file
        
        # Pipeline instances (loaded lazily)
        self.v1_pipeline = None
        self.v2_pipeline = None
        
        # Availability cache
        self._pipeline_availability = {}
        self._availability_checked = False
        
        logger.info(f"🔀 RAGPipelineRouter initialized")
        logger.info(f"📊 Default: {default_pipeline.value}, fallback: {enable_fallback}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration from file."""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                logger.warning(f"⚠️  Config file not found: {self.config_file}")
                return {}
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            return config.get("retrieval_pipeline", {})
            
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            return {}
    
    def _check_pipeline_availability(self) -> Dict[str, bool]:
        """Check availability of v1 and v2 pipelines."""
        if self._availability_checked:
            return self._pipeline_availability
        
        availability = {}
        
        # Check v1 pipeline
        try:
            from sam.document_rag import DocumentAwareRAGPipeline
            availability['v1_chunking'] = True
            logger.debug("✅ v1 RAG pipeline available")
        except ImportError as e:
            availability['v1_chunking'] = False
            logger.debug(f"❌ v1 RAG pipeline unavailable: {e}")
        
        # Check v2 pipeline
        try:
            from sam.document_rag.v2_rag_pipeline import V2RAGPipeline
            from sam.retrieval import get_v2_retrieval_engine
            from sam.storage import get_v2_storage_manager
            
            # Try to initialize key components
            storage_manager = get_v2_storage_manager()
            availability['v2_muvera'] = True
            logger.debug("✅ v2 RAG pipeline available")
        except ImportError as e:
            availability['v2_muvera'] = False
            logger.debug(f"❌ v2 RAG pipeline unavailable: {e}")
        except Exception as e:
            availability['v2_muvera'] = False
            logger.debug(f"❌ v2 RAG pipeline initialization failed: {e}")
        
        self._pipeline_availability = availability
        self._availability_checked = True
        
        return availability
    
    def _get_configured_pipeline(self) -> PipelineSelection:
        """Get pipeline selection from configuration."""
        try:
            config = self._load_config()
            version = config.get("version", "v1_chunking")
            
            if version == "v2_muvera":
                return PipelineSelection.V2_MUVERA
            elif version == "v1_chunking":
                return PipelineSelection.V1_CHUNKING
            else:
                logger.warning(f"⚠️  Unknown pipeline version in config: {version}")
                return PipelineSelection.V1_CHUNKING
                
        except Exception as e:
            logger.error(f"❌ Failed to get configured pipeline: {e}")
            return PipelineSelection.V1_CHUNKING
    
    def _select_pipeline(self, force_pipeline: Optional[PipelineSelection] = None) -> PipelineSelection:
        """Select which pipeline to use."""
        availability = self._check_pipeline_availability()
        
        # Use forced pipeline if specified
        if force_pipeline:
            if availability.get(force_pipeline.value, False):
                return force_pipeline
            else:
                logger.warning(f"⚠️  Forced pipeline {force_pipeline.value} not available")
                if not self.enable_fallback:
                    return force_pipeline  # Return anyway, will fail later
        
        # Use default pipeline logic
        if self.default_pipeline == PipelineSelection.AUTO:
            # Auto-select based on configuration and availability
            configured = self._get_configured_pipeline()
            
            if availability.get(configured.value, False):
                return configured
            elif configured == PipelineSelection.V2_MUVERA and availability.get('v1_chunking', False):
                logger.info("🔄 v2 not available, falling back to v1")
                return PipelineSelection.V1_CHUNKING
            elif configured == PipelineSelection.V1_CHUNKING and availability.get('v2_muvera', False):
                return PipelineSelection.V1_CHUNKING  # Stick with v1 if configured
            else:
                # Default fallback
                if availability.get('v1_chunking', False):
                    return PipelineSelection.V1_CHUNKING
                elif availability.get('v2_muvera', False):
                    return PipelineSelection.V2_MUVERA
                else:
                    logger.error("❌ No RAG pipelines available")
                    return PipelineSelection.V1_CHUNKING  # Will fail, but return something
        else:
            # Use explicit default
            if availability.get(self.default_pipeline.value, False):
                return self.default_pipeline
            elif self.enable_fallback:
                # Try the other pipeline
                if self.default_pipeline == PipelineSelection.V1_CHUNKING:
                    if availability.get('v2_muvera', False):
                        logger.info("🔄 v1 not available, falling back to v2")
                        return PipelineSelection.V2_MUVERA
                else:
                    if availability.get('v1_chunking', False):
                        logger.info("🔄 v2 not available, falling back to v1")
                        return PipelineSelection.V1_CHUNKING
            
            return self.default_pipeline
    
    def _get_v1_pipeline(self):
        """Get or create v1 pipeline instance."""
        if self.v1_pipeline is None:
            from sam.document_rag import DocumentAwareRAGPipeline
            self.v1_pipeline = DocumentAwareRAGPipeline()
        return self.v1_pipeline
    
    def _get_v2_pipeline(self):
        """Get or create v2 pipeline instance."""
        if self.v2_pipeline is None:
            from sam.document_rag.v2_rag_pipeline import get_v2_rag_pipeline
            self.v2_pipeline = get_v2_rag_pipeline()
        return self.v2_pipeline
    
    def route_query(self, 
                   query: str,
                   force_pipeline: Optional[PipelineSelection] = None) -> RoutingResult:
        """
        Route a query to the appropriate RAG pipeline.
        
        Args:
            query: User query
            force_pipeline: Force specific pipeline
            
        Returns:
            RoutingResult with pipeline result
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"🔀 Routing query: '{query[:50]}...'")
            
            # Select pipeline
            selected_pipeline = self._select_pipeline(force_pipeline)
            availability = self._check_pipeline_availability()
            
            logger.info(f"📊 Selected pipeline: {selected_pipeline.value}")
            
            # Process with selected pipeline
            fallback_used = False
            routing_reason = f"Selected {selected_pipeline.value}"
            
            try:
                if selected_pipeline == PipelineSelection.V1_CHUNKING:
                    pipeline = self._get_v1_pipeline()
                    rag_result = pipeline.process_query(query)
                    
                elif selected_pipeline == PipelineSelection.V2_MUVERA:
                    pipeline = self._get_v2_pipeline()
                    rag_result = pipeline.process_query(query)
                    
                else:
                    raise ValueError(f"Unknown pipeline: {selected_pipeline}")
                
                # Check if result indicates failure
                if hasattr(rag_result, 'success') and not rag_result.success:
                    raise Exception(f"Pipeline returned failure: {getattr(rag_result, 'error_message', 'Unknown error')}")
                elif isinstance(rag_result, dict) and not rag_result.get('success', True):
                    raise Exception(f"Pipeline returned failure: {rag_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"❌ {selected_pipeline.value} pipeline failed: {e}")
                
                if self.enable_fallback:
                    # Try fallback pipeline
                    fallback_pipeline = (PipelineSelection.V1_CHUNKING 
                                       if selected_pipeline == PipelineSelection.V2_MUVERA 
                                       else PipelineSelection.V2_MUVERA)
                    
                    if availability.get(fallback_pipeline.value, False):
                        logger.info(f"🔄 Falling back to {fallback_pipeline.value}")
                        
                        try:
                            if fallback_pipeline == PipelineSelection.V1_CHUNKING:
                                pipeline = self._get_v1_pipeline()
                                rag_result = pipeline.process_query(query)
                            else:
                                pipeline = self._get_v2_pipeline()
                                rag_result = pipeline.process_query(query)
                            
                            selected_pipeline = fallback_pipeline
                            fallback_used = True
                            routing_reason = f"Fallback to {fallback_pipeline.value} after {selected_pipeline.value} failed"
                            
                        except Exception as fallback_error:
                            logger.error(f"❌ Fallback pipeline also failed: {fallback_error}")
                            raise e  # Raise original error
                    else:
                        raise e
                else:
                    raise e
            
            processing_time = time.time() - start_time
            
            result = RoutingResult(
                selected_pipeline=selected_pipeline,
                rag_result=rag_result,
                routing_reason=routing_reason,
                fallback_used=fallback_used,
                processing_time=processing_time,
                pipeline_available=availability
            )
            
            logger.info(f"✅ Query routed successfully: {selected_pipeline.value}, {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Query routing failed: {e}")
            
            # Return error result
            processing_time = time.time() - start_time
            availability = self._check_pipeline_availability()
            
            return RoutingResult(
                selected_pipeline=selected_pipeline if 'selected_pipeline' in locals() else PipelineSelection.V1_CHUNKING,
                rag_result={'success': False, 'error': str(e), 'query': query},
                routing_reason=f"Routing failed: {str(e)}",
                fallback_used=False,
                processing_time=processing_time,
                pipeline_available=availability
            )
    
    def get_router_status(self) -> Dict[str, Any]:
        """Get router status and pipeline availability."""
        availability = self._check_pipeline_availability()
        config = self._load_config()
        
        return {
            'default_pipeline': self.default_pipeline.value,
            'enable_fallback': self.enable_fallback,
            'configured_pipeline': config.get('version', 'unknown'),
            'pipeline_availability': availability,
            'config_file': self.config_file,
            'router_initialized': True
        }
    
    def set_default_pipeline(self, pipeline: PipelineSelection):
        """Set the default pipeline."""
        self.default_pipeline = pipeline
        logger.info(f"🔧 Default pipeline set to: {pipeline.value}")
    
    def refresh_availability(self):
        """Refresh pipeline availability cache."""
        self._availability_checked = False
        self._pipeline_availability = {}
        availability = self._check_pipeline_availability()
        logger.info(f"🔄 Pipeline availability refreshed: {availability}")

def get_rag_pipeline_router(default_pipeline: PipelineSelection = PipelineSelection.AUTO,
                           enable_fallback: bool = True) -> RAGPipelineRouter:
    """
    Get or create a RAG pipeline router instance.
    
    Args:
        default_pipeline: Default pipeline to use
        enable_fallback: Enable automatic fallback
        
    Returns:
        RAGPipelineRouter instance
    """
    global _rag_pipeline_router
    
    if _rag_pipeline_router is None:
        _rag_pipeline_router = RAGPipelineRouter(
            default_pipeline=default_pipeline,
            enable_fallback=enable_fallback
        )
    
    return _rag_pipeline_router

def route_rag_query(query: str,
                   force_pipeline: Optional[str] = None) -> RoutingResult:
    """
    Convenience function to route a query through the RAG pipeline router.
    
    Args:
        query: User query
        force_pipeline: Force specific pipeline ('v1_chunking' or 'v2_muvera')
        
    Returns:
        RoutingResult with pipeline result
    """
    router = get_rag_pipeline_router()
    
    force_selection = None
    if force_pipeline:
        if force_pipeline == "v1_chunking":
            force_selection = PipelineSelection.V1_CHUNKING
        elif force_pipeline == "v2_muvera":
            force_selection = PipelineSelection.V2_MUVERA
    
    return router.route_query(query, force_selection)
