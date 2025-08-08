"""
Graph Database Interface for SAM's Cognitive Memory Core
Provides abstraction layer for graph database operations
Supports Neo4j and fallback to NetworkX for development
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import yaml
from abc import ABC, abstractmethod

# Graph database imports
try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

import networkx as nx
from pathlib import Path

# SAM imports
try:
    from config.logging_config import get_logger
except ImportError:
    # Fallback to standard logging
    def get_logger(name):
        return logging.getLogger(name)

# Configure logging
logger = get_logger(__name__)

@dataclass
class QueryResult:
    """Result of a graph database query."""
    success: bool
    data: List[Dict[str, Any]]
    execution_time: float
    query: str
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class NodeData:
    """Represents a node in the graph database."""
    id: str
    labels: List[str]
    properties: Dict[str, Any]

@dataclass
class RelationshipData:
    """Represents a relationship in the graph database."""
    id: str
    type: str
    start_node: str
    end_node: str
    properties: Dict[str, Any]

class GraphDatabaseInterface(ABC):
    """Abstract interface for graph database operations."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the graph database."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the graph database."""
        pass
    
    @abstractmethod
    async def create_node(self, node: NodeData) -> bool:
        """Create a node in the graph."""
        pass
    
    @abstractmethod
    async def create_relationship(self, relationship: RelationshipData) -> bool:
        """Create a relationship in the graph."""
        pass
    
    @abstractmethod
    async def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a query against the graph database."""
        pass
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[NodeData]:
        """Retrieve a node by ID."""
        pass
    
    @abstractmethod
    async def get_neighbors(self, node_id: str, relationship_type: Optional[str] = None) -> List[NodeData]:
        """Get neighboring nodes."""
        pass

class Neo4jDatabase(GraphDatabaseInterface):
    """Neo4j implementation of the graph database interface."""
    
    def __init__(self, connection_string: str, username: str = "neo4j", password: str = "password"):
        self.connection_string = connection_string
        self.username = username
        self.password = password
        self.driver: Optional[Driver] = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            if not NEO4J_AVAILABLE:
                logger.error("Neo4j driver not available")
                return False
            
            self.driver = GraphDatabase.driver(
                self.connection_string,
                auth=(self.username, self.password)
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self.connected = True
                    logger.info("Connected to Neo4j database")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Neo4j database."""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Disconnected from Neo4j database")
    
    async def create_node(self, node: NodeData) -> bool:
        """Create a node in Neo4j."""
        try:
            if not self.connected:
                return False
            
            labels_str = ":".join(node.labels)
            query = f"CREATE (n:{labels_str} $properties) RETURN n"
            
            with self.driver.session() as session:
                session.run(query, properties=node.properties)
            
            logger.debug(f"Created node: {node.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create node {node.id}: {e}")
            return False
    
    async def create_relationship(self, relationship: RelationshipData) -> bool:
        """Create a relationship in Neo4j."""
        try:
            if not self.connected:
                return False
            
            query = """
            MATCH (a), (b)
            WHERE a.id = $start_node AND b.id = $end_node
            CREATE (a)-[r:%s $properties]->(b)
            RETURN r
            """ % relationship.type
            
            with self.driver.session() as session:
                session.run(
                    query,
                    start_node=relationship.start_node,
                    end_node=relationship.end_node,
                    properties=relationship.properties
                )
            
            logger.debug(f"Created relationship: {relationship.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create relationship {relationship.id}: {e}")
            return False
    
    async def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a Cypher query."""
        start_time = datetime.now()
        
        try:
            if not self.connected:
                return QueryResult(
                    success=False,
                    data=[],
                    execution_time=0.0,
                    query=query,
                    error_message="Not connected to database"
                )
            
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                data = [record.data() for record in result]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                success=True,
                data=data,
                execution_time=execution_time,
                query=query
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Query failed: {e}")
            
            return QueryResult(
                success=False,
                data=[],
                execution_time=execution_time,
                query=query,
                error_message=str(e)
            )
    
    async def get_node(self, node_id: str) -> Optional[NodeData]:
        """Retrieve a node by ID."""
        try:
            query = "MATCH (n {id: $node_id}) RETURN n, labels(n) as labels"
            result = await self.query(query, {"node_id": node_id})
            
            if result.success and result.data:
                node_data = result.data[0]
                return NodeData(
                    id=node_id,
                    labels=node_data["labels"],
                    properties=dict(node_data["n"])
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    async def get_neighbors(self, node_id: str, relationship_type: Optional[str] = None) -> List[NodeData]:
        """Get neighboring nodes."""
        try:
            if relationship_type:
                query = f"""
                MATCH (n {{id: $node_id}})-[:{relationship_type}]-(neighbor)
                RETURN neighbor, labels(neighbor) as labels
                """
            else:
                query = """
                MATCH (n {id: $node_id})--(neighbor)
                RETURN neighbor, labels(neighbor) as labels
                """
            
            result = await self.query(query, {"node_id": node_id})
            
            neighbors = []
            if result.success:
                for record in result.data:
                    neighbor_props = dict(record["neighbor"])
                    neighbors.append(NodeData(
                        id=neighbor_props.get("id", ""),
                        labels=record["labels"],
                        properties=neighbor_props
                    ))
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Failed to get neighbors for {node_id}: {e}")
            return []

class NetworkXDatabase(GraphDatabaseInterface):
    """NetworkX implementation for development and fallback."""
    
    def __init__(self, persist_path: Optional[str] = None):
        self.graph = nx.MultiDiGraph()
        self.persist_path = persist_path
        self.connected = False
        self.node_counter = 0
        self.relationship_counter = 0
    
    async def connect(self) -> bool:
        """Initialize NetworkX graph."""
        try:
            if self.persist_path and Path(self.persist_path).exists():
                # Load existing graph
                self.graph = nx.read_gml(self.persist_path)
                logger.info(f"Loaded existing graph from {self.persist_path}")
            else:
                # Create new graph
                self.graph = nx.MultiDiGraph()
                logger.info("Created new NetworkX graph")
            
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize NetworkX graph: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Save and disconnect from NetworkX graph."""
        try:
            if self.persist_path:
                nx.write_gml(self.graph, self.persist_path)
                logger.info(f"Saved graph to {self.persist_path}")
            
            self.connected = False
            
        except Exception as e:
            logger.error(f"Failed to save NetworkX graph: {e}")
    
    async def create_node(self, node: NodeData) -> bool:
        """Create a node in NetworkX graph."""
        try:
            if not self.connected:
                return False
            
            # Add node with properties
            self.graph.add_node(
                node.id,
                labels=node.labels,
                **node.properties
            )
            
            self.node_counter += 1
            logger.debug(f"Created NetworkX node: {node.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create NetworkX node {node.id}: {e}")
            return False
    
    async def create_relationship(self, relationship: RelationshipData) -> bool:
        """Create a relationship in NetworkX graph."""
        try:
            if not self.connected:
                return False
            
            # Add edge with properties
            self.graph.add_edge(
                relationship.start_node,
                relationship.end_node,
                key=relationship.id,
                type=relationship.type,
                **relationship.properties
            )
            
            self.relationship_counter += 1
            logger.debug(f"Created NetworkX relationship: {relationship.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create NetworkX relationship {relationship.id}: {e}")
            return False
    
    async def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a simplified query on NetworkX graph."""
        start_time = datetime.now()
        
        try:
            # For NetworkX, implement basic query patterns
            # This is a simplified implementation
            data = []
            
            if "MATCH" in query.upper() and "RETURN" in query.upper():
                # Basic node retrieval
                if "RETURN n" in query:
                    for node_id in self.graph.nodes():
                        node_data = self.graph.nodes[node_id]
                        data.append({"n": {"id": node_id, **node_data}})
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                success=True,
                data=data,
                execution_time=execution_time,
                query=query,
                metadata={"backend": "networkx"}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"NetworkX query failed: {e}")
            
            return QueryResult(
                success=False,
                data=[],
                execution_time=execution_time,
                query=query,
                error_message=str(e)
            )
    
    async def get_node(self, node_id: str) -> Optional[NodeData]:
        """Retrieve a node by ID from NetworkX graph."""
        try:
            if node_id in self.graph.nodes:
                node_data = self.graph.nodes[node_id]
                return NodeData(
                    id=node_id,
                    labels=node_data.get("labels", []),
                    properties={k: v for k, v in node_data.items() if k != "labels"}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get NetworkX node {node_id}: {e}")
            return None
    
    async def get_neighbors(self, node_id: str, relationship_type: Optional[str] = None) -> List[NodeData]:
        """Get neighboring nodes from NetworkX graph."""
        try:
            neighbors = []
            
            if node_id in self.graph.nodes:
                for neighbor_id in self.graph.neighbors(node_id):
                    # Check relationship type if specified
                    if relationship_type:
                        edge_data = self.graph.get_edge_data(node_id, neighbor_id)
                        if not any(data.get("type") == relationship_type for data in edge_data.values()):
                            continue
                    
                    neighbor_data = self.graph.nodes[neighbor_id]
                    neighbors.append(NodeData(
                        id=neighbor_id,
                        labels=neighbor_data.get("labels", []),
                        properties={k: v for k, v in neighbor_data.items() if k != "labels"}
                    ))
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Failed to get NetworkX neighbors for {node_id}: {e}")
            return []

class GraphDatabaseManager:
    """Manager for graph database operations with fallback support."""
    
    def __init__(self, config_path: str = "sam/memory/graph_ontology.yaml"):
        self.config = self._load_config(config_path)
        self.database: Optional[GraphDatabaseInterface] = None
        self.backend_type = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load graph database configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load graph config: {e}")
            return {}
    
    async def initialize(self) -> bool:
        """Initialize the graph database with fallback."""
        settings = self.config.get("settings", {})
        backend = settings.get("graph_backend", "neo4j")
        
        if backend == "neo4j" and NEO4J_AVAILABLE:
            # Try Neo4j first
            self.database = Neo4jDatabase(
                connection_string=settings.get("connection_string", "bolt://localhost:7687")
            )
            
            if await self.database.connect():
                self.backend_type = "neo4j"
                logger.info("Using Neo4j graph database")
                return True
            else:
                logger.warning("Neo4j connection failed, falling back to NetworkX")
        
        # Fallback to NetworkX
        self.database = NetworkXDatabase(
            persist_path="sam/memory/graph/networkx_graph.gml"
        )
        
        if await self.database.connect():
            self.backend_type = "networkx"
            logger.info("Using NetworkX graph database (fallback)")
            return True
        
        logger.error("Failed to initialize any graph database")
        return False
    
    async def get_database(self) -> Optional[GraphDatabaseInterface]:
        """Get the active graph database interface."""
        if not self.database:
            await self.initialize()
        return self.database

# Factory function
def get_graph_database() -> GraphDatabaseManager:
    """Get a configured graph database manager."""
    return GraphDatabaseManager()
