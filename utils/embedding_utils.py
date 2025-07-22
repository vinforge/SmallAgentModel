"""
Embedding Utilities for SAM
Provides text embedding capabilities using sentence-transformers.

Sprint 2 Task 2: Embedding & Semantic Query
"""

import logging
import numpy as np
from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages text embeddings using sentence-transformers.
    Provides efficient embedding generation for semantic search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "models/embeddings"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence-transformer model
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.embedding_dim = None
        self._load_model()
        
        logger.info(f"Embedding manager initialized with model: {model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))
            
            # Get embedding dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.embedding_dim = test_embedding.shape[0]
            
            logger.info(f"Model loaded successfully, embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            normalize: Whether to normalize the embedding vector
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return np.zeros(self.embedding_dim, dtype=np.float32)
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=normalize)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def embed_batch(self, texts: List[str], normalize: bool = True, batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts to embed
            normalize: Whether to normalize the embedding vectors
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors as numpy arrays
        """
        try:
            if not texts:
                logger.warning("Empty text list provided for batch embedding")
                return []
            
            # Filter out empty texts and keep track of indices
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                logger.warning("No valid texts found in batch")
                return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    convert_to_numpy=True, 
                    normalize_embeddings=normalize,
                    batch_size=len(batch_texts)
                )
                embeddings.extend(batch_embeddings)
            
            # Create result array with zeros for invalid texts
            result = []
            valid_idx = 0
            
            for i in range(len(texts)):
                if i in valid_indices:
                    result.append(embeddings[valid_idx].astype(np.float32))
                    valid_idx += 1
                else:
                    result.append(np.zeros(self.embedding_dim, dtype=np.float32))
            
            logger.debug(f"Generated embeddings for {len(valid_texts)}/{len(texts)} texts")
            return result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
    
    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a search query.
        This is a specialized method that can apply query-specific preprocessing.
        
        Args:
            query: Search query text
            normalize: Whether to normalize the embedding vector
            
        Returns:
            Query embedding vector as numpy array
        """
        try:
            # Preprocess query (can be enhanced with query-specific logic)
            processed_query = query.strip()
            
            # Add query prefix for better retrieval (optional)
            # processed_query = f"search: {processed_query}"
            
            return self.embed(processed_query, normalize=normalize)
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        try:
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'device': str(self.model.device) if self.model else 'unknown'
        }
    
    def encode_for_storage(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """
        Encode texts for storage in vector database.
        Optimized for large batches with progress tracking.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Encoding {len(texts)} texts for storage")
            
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                batch_size=32
            )
            
            # Convert to list of individual arrays
            result = [emb.astype(np.float32) for emb in embeddings]
            
            logger.info(f"Successfully encoded {len(result)} texts")
            return result
            
        except Exception as e:
            logger.error(f"Error encoding texts for storage: {e}")
            return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]

# Global embedding manager instance
_embedding_manager = None

def get_embedding_manager(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingManager:
    """
    Get or create a global embedding manager instance.
    
    Args:
        model_name: Name of the sentence-transformer model
        
    Returns:
        EmbeddingManager instance
    """
    global _embedding_manager
    
    if _embedding_manager is None or _embedding_manager.model_name != model_name:
        _embedding_manager = EmbeddingManager(model_name)
    
    return _embedding_manager

# Convenience functions
def embed(text: str) -> np.ndarray:
    """Convenience function to embed a single text."""
    return get_embedding_manager().embed(text)

def embed_batch(texts: List[str]) -> List[np.ndarray]:
    """Convenience function to embed multiple texts."""
    return get_embedding_manager().embed_batch(texts)

def embed_query(query: str) -> np.ndarray:
    """Convenience function to embed a search query."""
    return get_embedding_manager().embed_query(query)
