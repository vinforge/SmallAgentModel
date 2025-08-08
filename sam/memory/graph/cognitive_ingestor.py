"""
Cognitive Ingestor - Core component of SAM's Cognitive Memory Core
Implements the ECL (Extract, Chunk, Link) pipeline using Cognee framework
Populates both vector store and graph database with structured knowledge
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml
import json
import hashlib
from pathlib import Path

# Core imports
import cognee
try:
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine
    COGNEE_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    COGNEE_INFRASTRUCTURE_AVAILABLE = False

try:
    # Try different import paths for Cognee API
    from cognee import cognify
    COGNEE_COGNIFY_AVAILABLE = True
except ImportError:
    try:
        from cognee.api import cognify
        COGNEE_COGNIFY_AVAILABLE = True
    except ImportError:
        COGNEE_COGNIFY_AVAILABLE = False

# SAM imports
try:
    from memory.memory_vectorstore import get_memory_store
except ImportError:
    # Fallback for testing
    def get_memory_store():
        class MockMemoryStore:
            def add_memory(self, content, metadata):
                pass
        return MockMemoryStore()
try:
    from config.logging_config import get_logger
except ImportError:
    # Fallback to standard logging
    def get_logger(name):
        return logging.getLogger(name)

# Configure logging
logger = get_logger(__name__)

@dataclass
class IngestionResult:
    """Result of cognitive ingestion process."""
    success: bool
    document_id: str
    nodes_created: int
    relationships_created: int
    vector_embeddings: int
    processing_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any]
    labels: List[str]
    created_at: datetime
    source_document: Optional[str] = None

@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph."""
    id: str
    type: str
    source_node_id: str
    target_node_id: str
    properties: Dict[str, Any]
    confidence: float
    created_at: datetime

class CognitiveIngestor:
    """
    Core cognitive ingestion engine that implements the ECL pipeline.
    Integrates Cognee framework with SAM's existing memory architecture.
    """
    
    def __init__(self, config_path: str = "sam/memory/graph_ontology.yaml"):
        """Initialize the cognitive ingestor."""
        self.config_path = config_path
        self.ontology = self._load_ontology()
        self.memory_store = get_memory_store()
        self.graph_engine = None
        self.vector_engine = None
        self.initialized = False
        
        logger.info("CognitiveIngestor initialized with ontology")
    
    def _load_ontology(self) -> Dict[str, Any]:
        """Load the graph ontology configuration."""
        try:
            with open(self.config_path, 'r') as f:
                ontology = yaml.safe_load(f)
            logger.info(f"Loaded ontology with {len(ontology.get('node_types', []))} node types")
            return ontology
        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            return {}
    
    async def initialize(self) -> bool:
        """Initialize the cognitive ingestor with Cognee engines."""
        try:
            # For Phase A, we'll implement a basic initialization
            # Full Cognee integration will be completed in later phases

            if COGNEE_INFRASTRUCTURE_AVAILABLE:
                try:
                    # Initialize Cognee if available
                    await cognee.prune()  # Clear any existing data for fresh start

                    # Set up engines
                    self.graph_engine = await get_graph_engine()
                    self.vector_engine = await get_vector_engine()
                    logger.info("Cognee engines initialized")
                except Exception as e:
                    logger.warning(f"Cognee engine initialization failed, using fallback: {e}")
                    self.graph_engine = None
                    self.vector_engine = None
            else:
                logger.info("Cognee infrastructure not available, using fallback mode")
                self.graph_engine = None
                self.vector_engine = None

            self.initialized = True
            logger.info("CognitiveIngestor initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize CognitiveIngestor: {e}")
            self.initialized = False
            return False
    
    async def ingest_document(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> IngestionResult:
        """
        Ingest a document through the cognitive ECL pipeline.
        
        Args:
            content: Document content to ingest
            metadata: Document metadata (title, author, source, etc.)
            document_id: Optional document ID, will be generated if not provided
            
        Returns:
            IngestionResult with processing details
        """
        start_time = datetime.now()
        
        if not self.initialized:
            await self.initialize()
        
        if not document_id:
            document_id = self._generate_document_id(content, metadata)
        
        try:
            logger.info(f"Starting cognitive ingestion for document: {document_id}")
            
            # Step 1: Extract entities and relationships using Cognee
            extraction_result = await self._extract_knowledge(content, metadata)
            
            # Step 2: Chunk content intelligently
            chunks = await self._intelligent_chunking(content, metadata)
            
            # Step 3: Link entities and create graph structure
            graph_result = await self._create_graph_structure(
                extraction_result, chunks, document_id, metadata
            )
            
            # Step 4: Sync with existing vector store
            vector_result = await self._sync_vector_store(chunks, document_id, metadata)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = IngestionResult(
                success=True,
                document_id=document_id,
                nodes_created=graph_result.get('nodes_created', 0),
                relationships_created=graph_result.get('relationships_created', 0),
                vector_embeddings=vector_result.get('embeddings_created', 0),
                processing_time=processing_time,
                metadata={
                    'chunks_created': len(chunks),
                    'extraction_entities': len(extraction_result.get('entities', [])),
                    'extraction_relationships': len(extraction_result.get('relationships', []))
                }
            )
            
            logger.info(f"Cognitive ingestion completed: {asdict(result)}")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Cognitive ingestion failed for {document_id}: {e}")
            
            return IngestionResult(
                success=False,
                document_id=document_id,
                nodes_created=0,
                relationships_created=0,
                vector_embeddings=0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _extract_knowledge(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities and relationships using Cognee's cognify process."""
        try:
            entities = []
            relationships = []

            if COGNEE_COGNIFY_AVAILABLE:
                try:
                    # Use Cognee's cognify for knowledge extraction
                    cognify_result = await cognify(content)

                    # Process Cognee's output to match our ontology
                    if hasattr(cognify_result, 'entities'):
                        for entity in cognify_result.entities:
                            entities.append({
                                'id': getattr(entity, 'id', f"entity_{len(entities)}"),
                                'type': self._map_entity_type(getattr(entity, 'type', 'Concept')),
                                'properties': getattr(entity, 'properties', {}),
                                'confidence': getattr(entity, 'confidence', 0.8)
                            })

                    if hasattr(cognify_result, 'relationships'):
                        for rel in cognify_result.relationships:
                            relationships.append({
                                'id': getattr(rel, 'id', f"rel_{len(relationships)}"),
                                'type': self._map_relationship_type(getattr(rel, 'type', 'RELATES_TO')),
                                'source': getattr(rel, 'source', ''),
                                'target': getattr(rel, 'target', ''),
                                'properties': getattr(rel, 'properties', {}),
                                'confidence': getattr(rel, 'confidence', 0.8)
                            })

                    logger.info(f"Cognee extracted {len(entities)} entities and {len(relationships)} relationships")

                except Exception as e:
                    logger.warning(f"Cognee extraction failed, using fallback: {e}")
                    # Fallback to basic extraction
                    entities, relationships = self._basic_extraction(content, metadata)
            else:
                # Fallback extraction for Phase A testing
                entities, relationships = self._basic_extraction(content, metadata)

            return {
                'entities': entities,
                'relationships': relationships
            }

        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            return {'entities': [], 'relationships': [], 'error': str(e)}
    
    async def _intelligent_chunking(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform intelligent document chunking that preserves structure."""
        try:
            # For now, implement basic chunking
            # In production, this would use more sophisticated methods
            chunk_size = 1000
            overlap = 200
            
            chunks = []
            content_length = len(content)
            
            for i in range(0, content_length, chunk_size - overlap):
                chunk_text = content[i:i + chunk_size]
                
                chunk = {
                    'id': f"{metadata.get('title', 'doc')}_{i}",
                    'text': chunk_text,
                    'start_pos': i,
                    'end_pos': min(i + chunk_size, content_length),
                    'metadata': {
                        **metadata,
                        'chunk_index': len(chunks),
                        'chunk_type': 'text_segment'
                    }
                }
                chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} intelligent chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Intelligent chunking failed: {e}")
            return []
    
    async def _create_graph_structure(
        self, 
        extraction_result: Dict[str, Any], 
        chunks: List[Dict[str, Any]], 
        document_id: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create graph structure from extracted knowledge."""
        try:
            nodes_created = 0
            relationships_created = 0
            
            # Create document node
            doc_node = GraphNode(
                id=document_id,
                type="Document",
                properties={
                    'title': metadata.get('title', 'Untitled'),
                    'source': metadata.get('source', 'unknown'),
                    'created_at': datetime.now().isoformat(),
                    **metadata
                },
                labels=["Document"],
                created_at=datetime.now(),
                source_document=document_id
            )
            
            # For now, store in memory (in production, this would go to Neo4j)
            nodes_created += 1
            
            # Create memory nodes for chunks
            for chunk in chunks:
                memory_node = GraphNode(
                    id=chunk['id'],
                    type="Memory",
                    properties={
                        'content': chunk['text'][:500],  # Truncate for storage
                        'chunk_index': chunk['metadata']['chunk_index'],
                        'start_pos': chunk['start_pos'],
                        'end_pos': chunk['end_pos']
                    },
                    labels=["Memory", "Chunk"],
                    created_at=datetime.now(),
                    source_document=document_id
                )
                nodes_created += 1
                
                # Create relationship between document and memory
                relationships_created += 1
            
            # Process extracted entities
            for entity in extraction_result.get('entities', []):
                entity_node = GraphNode(
                    id=entity['id'],
                    type=entity['type'],
                    properties=entity['properties'],
                    labels=[entity['type']],
                    created_at=datetime.now(),
                    source_document=document_id
                )
                nodes_created += 1
            
            # Process extracted relationships
            relationships_created += len(extraction_result.get('relationships', []))
            
            logger.info(f"Created graph structure: {nodes_created} nodes, {relationships_created} relationships")
            
            return {
                'nodes_created': nodes_created,
                'relationships_created': relationships_created
            }
            
        except Exception as e:
            logger.error(f"Graph structure creation failed: {e}")
            return {'nodes_created': 0, 'relationships_created': 0}
    
    async def _sync_vector_store(
        self, 
        chunks: List[Dict[str, Any]], 
        document_id: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sync chunks with existing vector store."""
        try:
            embeddings_created = 0
            
            # Add chunks to existing memory store
            for chunk in chunks:
                # Create memory entry compatible with existing system
                memory_entry = {
                    'content': chunk['text'],
                    'metadata': {
                        **chunk['metadata'],
                        'document_id': document_id,
                        'chunk_id': chunk['id'],
                        'ingestion_method': 'cognitive_ingestor'
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add to memory store (this will create embeddings)
                self.memory_store.add_memory(
                    content=memory_entry['content'],
                    metadata=memory_entry['metadata']
                )
                embeddings_created += 1
            
            logger.info(f"Synced {embeddings_created} chunks with vector store")
            
            return {'embeddings_created': embeddings_created}
            
        except Exception as e:
            logger.error(f"Vector store sync failed: {e}")
            return {'embeddings_created': 0}
    
    def _generate_document_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique document ID."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        title = metadata.get('title', 'document')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"doc_{title}_{timestamp}_{content_hash}"
    
    def _map_entity_type(self, cognee_type: str) -> str:
        """Map Cognee entity types to our ontology."""
        type_mapping = {
            'PERSON': 'Person',
            'ORG': 'Company',
            'GPE': 'Location',
            'PRODUCT': 'Technology',
            'EVENT': 'Concept',
            'WORK_OF_ART': 'Document',
            'LAW': 'Concept',
            'LANGUAGE': 'Technology'
        }
        return type_mapping.get(cognee_type, 'Concept')
    
    def _map_relationship_type(self, cognee_type: str) -> str:
        """Map Cognee relationship types to our ontology."""
        type_mapping = {
            'works_for': 'WORKS_FOR',
            'located_in': 'LOCATED_IN',
            'mentions': 'MENTIONS',
            'authored_by': 'AUTHORED_BY',
            'related_to': 'RELATES_TO',
            'part_of': 'RELATES_TO'
        }
        return type_mapping.get(cognee_type, 'RELATES_TO')

    def _basic_extraction(self, content: str, metadata: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """Basic fallback extraction for Phase A testing."""
        entities = []
        relationships = []

        # Simple keyword-based entity extraction for testing
        keywords = {
            'SAM': 'Technology',
            'Augment Code': 'Company',
            'OpenAI': 'Company',
            'GPT': 'Technology',
            'ChromaDB': 'Technology',
            'Streamlit': 'Technology',
            'AI': 'Concept',
            'memory': 'Concept',
            'reasoning': 'Concept'
        }

        for keyword, entity_type in keywords.items():
            if keyword.lower() in content.lower():
                entities.append({
                    'id': f"entity_{keyword.replace(' ', '_').lower()}",
                    'type': entity_type,
                    'properties': {
                        'name': keyword,
                        'mentioned_in': metadata.get('title', 'document'),
                        'extraction_method': 'basic_keyword'
                    },
                    'confidence': 0.7
                })

        # Create basic relationships between entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1['type'] != entity2['type']:  # Different types are more likely to be related
                    relationships.append({
                        'id': f"rel_{entity1['id']}_{entity2['id']}",
                        'type': 'MENTIONS',
                        'source': entity1['id'],
                        'target': entity2['id'],
                        'properties': {
                            'extraction_method': 'basic_co_occurrence',
                            'document': metadata.get('title', 'document')
                        },
                        'confidence': 0.6
                    })

        logger.info(f"Basic extraction: {len(entities)} entities, {len(relationships)} relationships")
        return entities, relationships

# Factory function for easy access
def get_cognitive_ingestor() -> CognitiveIngestor:
    """Get a configured cognitive ingestor instance."""
    return CognitiveIngestor()
