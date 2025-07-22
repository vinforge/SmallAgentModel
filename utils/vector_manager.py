"""
Vector Store Manager for SAM
Handles FAISS-based vector storage with metadata for semantic retrieval.

Sprint 2 Task 1: Vector Store Initialization
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Try to import FAISS, fall back to simple vector search if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)

@dataclass
class VectorChunk:
    """Represents a chunk with its vector embedding and metadata."""
    chunk_id: str
    text: str
    vector: np.ndarray
    metadata: Dict[str, Any]

class VectorManager:
    """
    Manages vector storage and retrieval using FAISS.
    Provides semantic search capabilities for SAM's knowledge base.
    """
    
    def __init__(self, vector_store_path: str = "data/vector_store", embedding_dim: int = 384):
        """
        Initialize the vector manager.

        Args:
            vector_store_path: Path to store vector index and metadata
            embedding_dim: Dimension of embedding vectors (384 for all-MiniLM-L6-v2)
        """
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.use_faiss = FAISS_AVAILABLE

        if self.use_faiss:
            self.index_path = self.vector_store_path / "faiss_index.bin"
            self.metadata_path = self.vector_store_path / "metadata.json"

            # Initialize FAISS index
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
            self.metadata_store = {}  # chunk_id -> metadata mapping
            self.id_to_chunk_id = {}  # FAISS ID -> chunk_id mapping
            self.chunk_id_to_id = {}  # chunk_id -> FAISS ID mapping
            self.next_id = 0
        else:
            # Fallback to simple vector storage
            self.vectors_path = self.vector_store_path / "vectors.pkl"
            self.metadata_path = self.vector_store_path / "metadata.json"

            self.vectors = []  # List of (chunk_id, vector) tuples
            self.metadata_store = {}  # chunk_id -> metadata mapping
            logger.warning("FAISS not available, using simple vector storage (slower)")

        # Load existing index if available
        self.load_index()

        total_vectors = self.index.ntotal if self.use_faiss else len(self.vectors)
        backend = "FAISS" if self.use_faiss else "Simple"
        logger.info(f"Vector manager initialized with {total_vectors} vectors using {backend} backend")
    
    def load_index(self):
        """Load existing vector index and metadata from disk."""
        try:
            if self.use_faiss:
                # Load FAISS index
                if self.index_path.exists() and self.metadata_path.exists():
                    self.index = faiss.read_index(str(self.index_path))

                    # Load metadata
                    import builtins
                    with builtins.open(self.metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.metadata_store = data.get('metadata_store', {})
                        self.id_to_chunk_id = {int(k): v for k, v in data.get('id_to_chunk_id', {}).items()}
                        self.chunk_id_to_id = data.get('chunk_id_to_id', {})
                        self.next_id = data.get('next_id', 0)

                    logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
                else:
                    logger.info("No existing FAISS index found, starting fresh")
            else:
                # Load simple vector storage
                if self.vectors_path.exists() and self.metadata_path.exists():
                    import builtins
                    with builtins.open(self.vectors_path, 'rb') as f:
                        self.vectors = pickle.load(f)

                    with builtins.open(self.metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.metadata_store = data.get('metadata_store', {})

                    logger.info(f"Loaded existing simple vector store with {len(self.vectors)} vectors")
                else:
                    logger.info("No existing vector store found, starting fresh")

        except Exception as e:
            logger.error(f"Error loading vector index: {e}")
            logger.info("Starting with fresh index")
            if self.use_faiss:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.metadata_store = {}
                self.id_to_chunk_id = {}
                self.chunk_id_to_id = {}
                self.next_id = 0
            else:
                self.vectors = []
                self.metadata_store = {}
    
    def save_index(self):
        """Save vector index and metadata to disk."""
        try:
            # Ensure directories exist
            self.vector_store_path.mkdir(parents=True, exist_ok=True)

            if self.use_faiss:
                # Save FAISS index
                faiss.write_index(self.index, str(self.index_path))

                # Save metadata
                metadata_data = {
                    'metadata_store': self.metadata_store,
                    'id_to_chunk_id': {str(k): v for k, v in self.id_to_chunk_id.items()},
                    'chunk_id_to_id': self.chunk_id_to_id,
                    'next_id': self.next_id
                }

                # Use explicit builtin open function to avoid any shadowing issues
                import builtins
                with builtins.open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_data, f, indent=2)

                logger.debug(f"Saved FAISS index with {self.index.ntotal} vectors")
            else:
                # Save simple vector storage
                import builtins
                with builtins.open(self.vectors_path, 'wb') as f:
                    pickle.dump(self.vectors, f)

                metadata_data = {
                    'metadata_store': self.metadata_store
                }

                with builtins.open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_data, f, indent=2)

                logger.debug(f"Saved simple vector store with {len(self.vectors)} vectors")

        except Exception as e:
            logger.error(f"Error saving vector index: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def add_chunk(self, chunk_id: str, chunk_text: str, vector: np.ndarray, metadata: Dict[str, Any]):
        """
        Add a new chunk to the vector store.

        Args:
            chunk_id: Unique identifier for the chunk
            chunk_text: Text content of the chunk
            vector: Embedding vector for the chunk
            metadata: Metadata dictionary for the chunk
        """
        try:
            # Normalize vector for cosine similarity
            vector = vector.astype(np.float32)
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)

            if self.use_faiss:
                # Check if chunk already exists
                if chunk_id in self.chunk_id_to_id:
                    logger.warning(f"Chunk {chunk_id} already exists, skipping")
                    return

                # Add to FAISS index
                vector_2d = vector.reshape(1, -1)
                self.index.add(vector_2d)

                # Store metadata
                faiss_id = self.next_id
                self.metadata_store[chunk_id] = {
                    'text': chunk_text,
                    'metadata': metadata,
                    'faiss_id': faiss_id
                }
                self.id_to_chunk_id[faiss_id] = chunk_id
                self.chunk_id_to_id[chunk_id] = faiss_id
                self.next_id += 1
            else:
                # Check if chunk already exists
                if chunk_id in self.metadata_store:
                    logger.warning(f"Chunk {chunk_id} already exists, skipping")
                    return

                # Add to simple vector storage
                self.vectors.append((chunk_id, vector))
                self.metadata_store[chunk_id] = {
                    'text': chunk_text,
                    'metadata': metadata
                }

            logger.debug(f"Added chunk {chunk_id} to vector store")

        except Exception as e:
            logger.error(f"Error adding chunk {chunk_id}: {e}")
    
    def add_chunks_batch(self, chunks: List[Tuple[str, str, np.ndarray, Dict[str, Any]]]):
        """
        Add multiple chunks in batch for efficiency.

        Args:
            chunks: List of (chunk_id, chunk_text, vector, metadata) tuples
        """
        try:
            if self.use_faiss:
                # FAISS batch processing
                vectors_to_add = []
                chunk_ids_to_add = []

                for chunk_id, chunk_text, vector, metadata in chunks:
                    if chunk_id in self.chunk_id_to_id:
                        logger.warning(f"Chunk {chunk_id} already exists, skipping")
                        continue

                    # Normalize vector
                    vector = vector.astype(np.float32)
                    if np.linalg.norm(vector) > 0:
                        vector = vector / np.linalg.norm(vector)

                    vectors_to_add.append(vector)
                    chunk_ids_to_add.append((chunk_id, chunk_text, metadata))

                if not vectors_to_add:
                    logger.info("No new chunks to add")
                    return

                # Add vectors to FAISS index
                vectors_array = np.vstack(vectors_to_add)
                self.index.add(vectors_array)

                # Store metadata
                for i, (chunk_id, chunk_text, metadata) in enumerate(chunk_ids_to_add):
                    faiss_id = self.next_id + i
                    self.metadata_store[chunk_id] = {
                        'text': chunk_text,
                        'metadata': metadata,
                        'faiss_id': faiss_id
                    }
                    self.id_to_chunk_id[faiss_id] = chunk_id
                    self.chunk_id_to_id[chunk_id] = faiss_id

                self.next_id += len(chunk_ids_to_add)

                logger.info(f"Added {len(chunk_ids_to_add)} chunks to vector store")
            else:
                # Simple backend: add chunks one by one
                added_count = 0
                for chunk_id, chunk_text, vector, metadata in chunks:
                    if chunk_id not in self.metadata_store:
                        self.add_chunk(chunk_id, chunk_text, vector, metadata)
                        added_count += 1

                logger.info(f"Added {added_count} chunks to simple vector store")

        except Exception as e:
            logger.error(f"Error adding chunks in batch: {e}")

    def add_multimodal_chunk(self, chunk_id: str, chunk_text: str, vector: np.ndarray,
                           metadata: Dict[str, Any], content_type: str = 'text',
                           multimodal_data: Optional[Dict[str, Any]] = None):
        """
        Add a multimodal chunk to the vector store with enhanced metadata.

        Args:
            chunk_id: Unique identifier for the chunk
            chunk_text: Primary text content of the chunk
            vector: Embedding vector for the chunk
            metadata: Additional metadata for the chunk
            content_type: Type of content ('text', 'code', 'table', 'image', 'multimodal')
            multimodal_data: Additional multimodal-specific data
        """
        try:
            # Enhance metadata with multimodal information
            enhanced_metadata = metadata.copy()
            enhanced_metadata.update({
                'content_type': content_type,
                'is_multimodal': content_type != 'text' or multimodal_data is not None,
                'multimodal_timestamp': datetime.now().isoformat()
            })

            if multimodal_data:
                enhanced_metadata['multimodal_data'] = multimodal_data

            # Add using standard method with enhanced metadata
            self.add_chunk(chunk_id, chunk_text, vector, enhanced_metadata)

            logger.debug(f"Added multimodal chunk: {chunk_id} (type: {content_type})")

        except Exception as e:
            logger.error(f"Error adding multimodal chunk {chunk_id}: {e}")

    def add_consolidated_knowledge(self, consolidated_knowledge, enrichment_score):
        """
        Add consolidated multimodal knowledge to the vector store.

        Args:
            consolidated_knowledge: ConsolidatedKnowledge object
            enrichment_score: EnrichmentScore object
        """
        try:
            from utils.embedding_utils import get_embedding_manager

            # Generate embedding for the summary
            embedding_manager = get_embedding_manager()
            summary_embedding = embedding_manager.embed(consolidated_knowledge.summary)

            # Create comprehensive metadata
            metadata = {
                'source_file': consolidated_knowledge.source_document,
                'consolidation_id': consolidated_knowledge.consolidation_id,
                'key_concepts': consolidated_knowledge.key_concepts,
                'content_attribution': consolidated_knowledge.content_attribution,
                'enrichment_score': enrichment_score.overall_score,
                'priority_level': enrichment_score.priority_level,
                'component_scores': enrichment_score.component_scores,
                'enriched_metadata': consolidated_knowledge.enriched_metadata,
                'consolidation_timestamp': consolidated_knowledge.consolidation_timestamp,
                'tags': ['multimodal', 'consolidated'] + consolidated_knowledge.key_concepts[:5]
            }

            # Add multimodal data
            multimodal_data = {
                'content_types_present': list(consolidated_knowledge.content_attribution.keys()),
                'multimodal_richness': consolidated_knowledge.enriched_metadata.get('multimodal_richness', 0),
                'technical_content_ratio': consolidated_knowledge.enriched_metadata.get('technical_content_ratio', 0),
                'content_diversity_score': consolidated_knowledge.enriched_metadata.get('content_diversity_score', 0)
            }

            # Use consolidation ID as chunk ID
            chunk_id = f"consolidated_{consolidated_knowledge.consolidation_id}"

            self.add_multimodal_chunk(
                chunk_id=chunk_id,
                chunk_text=consolidated_knowledge.summary,
                vector=summary_embedding,
                metadata=metadata,
                content_type='multimodal',
                multimodal_data=multimodal_data
            )

            logger.info(f"Added consolidated knowledge to vector store: {chunk_id}")

        except Exception as e:
            logger.error(f"Error adding consolidated knowledge: {e}")

    def search(self, query_vector: np.ndarray, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of search results with metadata
        """
        try:
            # Check if vector store is empty
            total_vectors = self.index.ntotal if self.use_faiss else len(self.vectors)
            if total_vectors == 0:
                logger.warning("Vector store is empty")
                return []

            # Normalize query vector
            query_vector = query_vector.astype(np.float32)
            if np.linalg.norm(query_vector) > 0:
                query_vector = query_vector / np.linalg.norm(query_vector)

            results = []

            if self.use_faiss:
                # Search FAISS index
                query_2d = query_vector.reshape(1, -1)
                scores, indices = self.index.search(query_2d, min(top_k, self.index.ntotal))

                # Format results
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:  # FAISS returns -1 for invalid indices
                        continue

                    if score < score_threshold:
                        continue

                    chunk_id = self.id_to_chunk_id.get(idx)
                    if chunk_id and chunk_id in self.metadata_store:
                        chunk_data = self.metadata_store[chunk_id]
                        result = {
                            'chunk_id': chunk_id,
                            'text': chunk_data['text'],
                            'metadata': chunk_data['metadata'],
                            'similarity_score': float(score)
                        }
                        results.append(result)
            else:
                # Simple vector search using cosine similarity
                similarities = []
                for chunk_id, vector in self.vectors:
                    # Calculate cosine similarity
                    similarity = np.dot(query_vector, vector)
                    similarities.append((similarity, chunk_id))

                # Sort by similarity and take top_k
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_similarities = similarities[:top_k]

                # Format results
                for similarity, chunk_id in top_similarities:
                    if similarity < score_threshold:
                        continue

                    if chunk_id in self.metadata_store:
                        chunk_data = self.metadata_store[chunk_id]
                        result = {
                            'chunk_id': chunk_id,
                            'text': chunk_data['text'],
                            'metadata': chunk_data['metadata'],
                            'similarity_score': float(similarity)
                        }
                        results.append(result)

            logger.debug(f"Found {len(results)} results for query")
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID."""
        if chunk_id in self.metadata_store:
            chunk_data = self.metadata_store[chunk_id]
            return {
                'chunk_id': chunk_id,
                'text': chunk_data['text'],
                'metadata': chunk_data['metadata']
            }
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.use_faiss:
            return {
                'total_chunks': self.index.ntotal,
                'embedding_dimension': self.embedding_dim,
                'backend': 'FAISS',
                'index_size_mb': os.path.getsize(self.index_path) / (1024 * 1024) if self.index_path.exists() else 0,
                'metadata_size_mb': os.path.getsize(self.metadata_path) / (1024 * 1024) if self.metadata_path.exists() else 0
            }
        else:
            return {
                'total_chunks': len(self.vectors),
                'embedding_dimension': self.embedding_dim,
                'backend': 'Simple',
                'vectors_size_mb': os.path.getsize(self.vectors_path) / (1024 * 1024) if self.vectors_path.exists() else 0,
                'metadata_size_mb': os.path.getsize(self.metadata_path) / (1024 * 1024) if self.metadata_path.exists() else 0
            }
    
    def filter_by_tags(self, results: List[Dict[str, Any]], required_tags: List[str] = None, 
                      preferred_tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        Filter and rerank results based on tags.
        
        Args:
            results: Search results from vector search
            required_tags: Tags that must be present
            preferred_tags: Tags that boost ranking
            
        Returns:
            Filtered and reranked results
        """
        filtered_results = []
        
        for result in results:
            chunk_tags = result.get('metadata', {}).get('tags', [])
            
            # Check required tags
            if required_tags:
                if not any(tag in chunk_tags for tag in required_tags):
                    continue
            
            # Calculate tag boost for preferred tags
            tag_boost = 0.0
            if preferred_tags:
                matching_preferred = sum(1 for tag in preferred_tags if tag in chunk_tags)
                tag_boost = matching_preferred / len(preferred_tags) * 0.1  # 10% boost max
            
            # Apply tag boost to similarity score
            result['original_score'] = result['similarity_score']
            result['similarity_score'] += tag_boost
            result['tag_boost'] = tag_boost
            
            filtered_results.append(result)
        
        # Re-sort by boosted scores
        filtered_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return filtered_results
    
    def __del__(self):
        """Save index when object is destroyed."""
        try:
            self.save_index()
        except:
            pass
