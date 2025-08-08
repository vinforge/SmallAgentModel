#!/usr/bin/env python3
"""
Multi-Vector Embedder for SAM v2 Retrieval Pipeline
Implements ColBERTv2-based token-level embeddings for superior document retrieval.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Global embedder instance
_multivector_embedder = None

@dataclass
class EmbeddingResult:
    """Result from multi-vector embedding generation."""
    token_embeddings: np.ndarray  # Shape: (num_tokens, embedding_dim)
    token_ids: List[int]          # Token IDs from tokenizer
    attention_mask: np.ndarray    # Attention mask for valid tokens
    text_length: int              # Original text length
    num_tokens: int               # Number of tokens processed
    embedding_dim: int            # Dimension of each token embedding
    processing_time: float        # Time taken to generate embeddings
    metadata: Dict[str, Any]      # Additional metadata

class MultiVectorEmbedder:
    """
    Multi-vector embedder using ColBERTv2 for token-level document embeddings.
    This is the core component for SAM's v2 retrieval pipeline.
    """
    
    def __init__(self, 
                 model_name: str = "colbert-ir/colbertv2.0",
                 max_length: int = 512,
                 device: str = "auto",
                 cache_dir: Optional[str] = None):
        """
        Initialize the multi-vector embedder.
        
        Args:
            model_name: ColBERT model to use
            max_length: Maximum sequence length
            device: Device to run on ('auto', 'cpu', 'cuda')
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._determine_device(device)
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "models", "colbert")
        
        # Model components (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        logger.info(f"🧠 MultiVectorEmbedder initialized with model: {model_name}")
        logger.info(f"📏 Max length: {max_length}, Device: {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_model(self) -> bool:
        """Load the ColBERT model and tokenizer."""
        if self.is_loaded:
            return True
        
        try:
            logger.info(f"🔄 Loading ColBERT model: {self.model_name}")
            
            # Try to import ColBERT
            try:
                from colbert import Indexer, Searcher
                from colbert.infra import Run, RunConfig, ColBERTConfig
                from colbert.modeling.colbert import ColBERT
                from transformers import AutoTokenizer
                
                logger.info("✅ ColBERT imports successful")
                
            except ImportError as e:
                logger.error(f"❌ ColBERT not available: {e}")
                logger.info("💡 Install with: pip install colbert-ai")
                return False
            
            # Create cache directory
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load ColBERT model
            config = ColBERTConfig(
                doc_maxlen=self.max_length,
                query_maxlen=self.max_length,
                dim=128,  # ColBERTv2 default dimension
                similarity='cosine'
            )
            
            self.model = ColBERT.from_pretrained(
                self.model_name,
                colbert_config=config
            )
            
            # Move to device
            if self.device != "cpu":
                try:
                    import torch
                    self.model = self.model.to(self.device)
                    logger.info(f"✅ Model moved to {self.device}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not move to {self.device}, using CPU: {e}")
                    self.device = "cpu"
            
            self.model.eval()  # Set to evaluation mode
            self.is_loaded = True
            
            logger.info(f"✅ ColBERT model loaded successfully")
            logger.info(f"📊 Model dimension: {config.dim}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ColBERT model: {e}")
            return False
    
    def embed_document(self, text: str, doc_id: Optional[str] = None) -> Optional[EmbeddingResult]:
        """
        Generate multi-vector embeddings for a document.
        
        Args:
            text: Document text to embed
            doc_id: Optional document identifier
            
        Returns:
            EmbeddingResult with token-level embeddings
        """
        if not self._load_model():
            logger.error("❌ Cannot embed document: model not loaded")
            return None
        
        try:
            import time
            start_time = time.time()
            
            logger.debug(f"🔄 Embedding document: {len(text)} characters")
            
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            if self.device != "cpu":
                try:
                    import torch
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                except Exception:
                    pass  # Fallback to CPU
            
            # Generate embeddings
            with torch.no_grad():
                # ColBERT forward pass for documents
                embeddings = self.model.doc(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            
            # Convert to numpy
            token_embeddings = embeddings.cpu().numpy()
            attention_mask = inputs['attention_mask'].cpu().numpy()
            token_ids = inputs['input_ids'].cpu().numpy().flatten().tolist()
            
            # Remove padding tokens
            valid_tokens = attention_mask.flatten() == 1
            token_embeddings = token_embeddings[0][valid_tokens]  # Remove batch dimension and padding
            
            processing_time = time.time() - start_time
            
            result = EmbeddingResult(
                token_embeddings=token_embeddings,
                token_ids=[tid for tid, valid in zip(token_ids, valid_tokens) if valid],
                attention_mask=attention_mask,
                text_length=len(text),
                num_tokens=len(token_embeddings),
                embedding_dim=token_embeddings.shape[1],
                processing_time=processing_time,
                metadata={
                    'doc_id': doc_id,
                    'model_name': self.model_name,
                    'device': self.device,
                    'max_length': self.max_length,
                    'truncated': len(text) > self.max_length
                }
            )
            
            logger.debug(f"✅ Document embedded: {result.num_tokens} tokens, "
                        f"{result.embedding_dim}D, {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to embed document: {e}")
            return None
    
    def embed_query(self, query: str) -> Optional[EmbeddingResult]:
        """
        Generate multi-vector embeddings for a query.
        
        Args:
            query: Query text to embed
            
        Returns:
            EmbeddingResult with token-level embeddings
        """
        if not self._load_model():
            logger.error("❌ Cannot embed query: model not loaded")
            return None
        
        try:
            import time
            start_time = time.time()
            
            logger.debug(f"🔍 Embedding query: '{query[:50]}...'")
            
            # Tokenize the query
            inputs = self.tokenizer(
                query,
                max_length=min(self.max_length, 64),  # Queries are typically shorter
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            if self.device != "cpu":
                try:
                    import torch
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                except Exception:
                    pass  # Fallback to CPU
            
            # Generate embeddings
            with torch.no_grad():
                # ColBERT forward pass for queries
                embeddings = self.model.query(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            
            # Convert to numpy
            token_embeddings = embeddings.cpu().numpy()
            attention_mask = inputs['attention_mask'].cpu().numpy()
            token_ids = inputs['input_ids'].cpu().numpy().flatten().tolist()
            
            # Remove padding tokens
            valid_tokens = attention_mask.flatten() == 1
            token_embeddings = token_embeddings[0][valid_tokens]  # Remove batch dimension and padding
            
            processing_time = time.time() - start_time
            
            result = EmbeddingResult(
                token_embeddings=token_embeddings,
                token_ids=[tid for tid, valid in zip(token_ids, valid_tokens) if valid],
                attention_mask=attention_mask,
                text_length=len(query),
                num_tokens=len(token_embeddings),
                embedding_dim=token_embeddings.shape[1],
                processing_time=processing_time,
                metadata={
                    'query': query,
                    'model_name': self.model_name,
                    'device': self.device,
                    'is_query': True
                }
            )
            
            logger.debug(f"✅ Query embedded: {result.num_tokens} tokens, "
                        f"{result.embedding_dim}D, {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to embed query: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'device': self.device,
            'cache_dir': self.cache_dir,
            'is_loaded': self.is_loaded,
            'embedding_dim': 128 if self.is_loaded else None  # ColBERTv2 default
        }

def get_multivector_embedder(model_name: str = "colbert-ir/colbertv2.0",
                            max_length: int = 512,
                            device: str = "auto",
                            cache_dir: Optional[str] = None) -> MultiVectorEmbedder:
    """
    Get or create a multi-vector embedder instance.
    
    Args:
        model_name: ColBERT model to use
        max_length: Maximum sequence length
        device: Device to run on
        cache_dir: Directory to cache models
        
    Returns:
        MultiVectorEmbedder instance
    """
    global _multivector_embedder
    
    if _multivector_embedder is None:
        _multivector_embedder = MultiVectorEmbedder(
            model_name=model_name,
            max_length=max_length,
            device=device,
            cache_dir=cache_dir
        )
    
    return _multivector_embedder
