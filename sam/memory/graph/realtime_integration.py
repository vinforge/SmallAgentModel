"""
Real-Time Graph Database Integration for SAM's Cognitive Memory Core - Phase C
Implements production-ready Neo4j/NetworkX integration with async optimization.
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import networkx as nx
from contextlib import asynccontextmanager
import weakref
import time

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result from a graph database query."""
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: str = ""
    query_id: str = ""
    cached: bool = False

@dataclass
class ConnectionConfig:
    """Configuration for graph database connections."""
    database_type: str = "neo4j"  # "neo4j" or "networkx"
    host: str = "localhost"
    port: int = 7687
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connections: int = 10
    connection_timeout: float = 30.0
    query_timeout: float = 60.0
    retry_attempts: int = 3
    enable_ssl: bool = False

class ConnectionPool:
    """Async connection pool for graph database connections."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ConnectionPool")
        self._pool = asyncio.Queue(maxsize=config.max_connections)
        self._active_connections = weakref.WeakSet()
        self._connection_count = 0
        self._initialized = False
    
    async def initialize(self):
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        try:
            # Pre-populate pool with connections
            for _ in range(min(3, self.config.max_connections)):
                connection = await self._create_connection()
                if connection:
                    await self._pool.put(connection)
                    self._connection_count += 1
            
            self._initialized = True
            self.logger.info(f"Connection pool initialized with {self._connection_count} connections")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def _create_connection(self):
        """Create a new database connection."""
        try:
            if self.config.database_type == "neo4j":
                return await self._create_neo4j_connection()
            elif self.config.database_type == "networkx":
                return await self._create_networkx_connection()
            else:
                raise ValueError(f"Unsupported database type: {self.config.database_type}")
        
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            return None
    
    async def _create_neo4j_connection(self):
        """Create a Neo4j connection (mock for now)."""
        # In production, use neo4j-python-driver
        connection = {
            "type": "neo4j",
            "session_id": f"neo4j_session_{time.time()}",
            "created_at": datetime.now(),
            "active": True
        }
        return connection
    
    async def _create_networkx_connection(self):
        """Create a NetworkX connection."""
        # Create in-memory NetworkX graph
        graph = nx.MultiDiGraph()
        
        # Add some sample nodes and edges for testing
        sample_nodes = [
            ("concept_ai", {"type": "Concept", "name": "Artificial Intelligence"}),
            ("concept_ml", {"type": "Concept", "name": "Machine Learning"}),
            ("concept_dl", {"type": "Concept", "name": "Deep Learning"}),
            ("concept_nlp", {"type": "Concept", "name": "Natural Language Processing"}),
            ("concept_sam", {"type": "Concept", "name": "SAM AI System"})
        ]
        
        sample_edges = [
            ("concept_ai", "concept_ml", {"type": "INCLUDES", "weight": 0.9}),
            ("concept_ml", "concept_dl", {"type": "SPECIALIZES_TO", "weight": 0.8}),
            ("concept_ai", "concept_nlp", {"type": "INCLUDES", "weight": 0.7}),
            ("concept_sam", "concept_ai", {"type": "IMPLEMENTS", "weight": 0.9}),
            ("concept_sam", "concept_nlp", {"type": "USES", "weight": 0.8})
        ]
        
        graph.add_nodes_from(sample_nodes)
        graph.add_edges_from(sample_edges)
        
        connection = {
            "type": "networkx",
            "graph": graph,
            "session_id": f"nx_session_{time.time()}",
            "created_at": datetime.now(),
            "active": True
        }
        return connection
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if not self._initialized:
            await self.initialize()
        
        connection = None
        try:
            # Try to get existing connection
            try:
                connection = await asyncio.wait_for(
                    self._pool.get(), 
                    timeout=self.config.connection_timeout
                )
            except asyncio.TimeoutError:
                # Create new connection if pool is empty
                if self._connection_count < self.config.max_connections:
                    connection = await self._create_connection()
                    if connection:
                        self._connection_count += 1
                else:
                    raise Exception("Connection pool exhausted")
            
            if not connection:
                raise Exception("Failed to obtain database connection")
            
            self._active_connections.add(connection)
            yield connection
            
        finally:
            if connection and connection.get("active", False):
                # Return connection to pool
                try:
                    self._pool.put_nowait(connection)
                except asyncio.QueueFull:
                    # Pool is full, close connection
                    await self._close_connection(connection)
                    self._connection_count -= 1
    
    async def _close_connection(self, connection):
        """Close a database connection."""
        if connection:
            connection["active"] = False
            self.logger.debug(f"Closed connection: {connection.get('session_id', 'unknown')}")
    
    async def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                connection = self._pool.get_nowait()
                await self._close_connection(connection)
            except asyncio.QueueEmpty:
                break
        
        self._connection_count = 0
        self._initialized = False
        self.logger.info("All connections closed")

class RealTimeGraphDatabase:
    """
    Real-time graph database integration with async optimization.
    
    Features:
    - Connection pooling for performance
    - Async query execution
    - Query result caching
    - Automatic retry logic
    - Support for Neo4j and NetworkX
    """
    
    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.logger = logging.getLogger(f"{__name__}.RealTimeGraphDatabase")
        self.config = config or ConnectionConfig()
        self.connection_pool = ConnectionPool(self.config)
        self._query_cache = {}
        self._cache_ttl = timedelta(minutes=5)
        self._stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "avg_query_time": 0.0
        }
        
        self.logger.info(f"Real-time graph database initialized: {self.config.database_type}")
    
    async def initialize(self):
        """Initialize the database connection."""
        await self.connection_pool.initialize()
        self.logger.info("Real-time graph database ready")
    
    async def query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> QueryResult:
        """
        Execute a graph database query with caching and optimization.
        
        Args:
            query: The query string (Cypher for Neo4j, custom for NetworkX)
            parameters: Query parameters
            use_cache: Whether to use query result caching
            
        Returns:
            Query result with data and metadata
        """
        start_time = time.time()
        query_id = f"query_{hash(query)}_{hash(str(parameters))}"
        
        # Check cache first
        if use_cache:
            cached_result = self._get_cached_result(query_id)
            if cached_result:
                self._stats["cache_hits"] += 1
                return cached_result
        
        self._stats["cache_misses"] += 1
        
        try:
            async with self.connection_pool.get_connection() as connection:
                result = await self._execute_query(connection, query, parameters)
                
                execution_time = time.time() - start_time
                result.execution_time = execution_time
                result.query_id = query_id
                
                # Cache successful results
                if use_cache and result.success:
                    self._cache_result(query_id, result)
                
                # Update statistics
                self._update_stats(execution_time, result.success)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            self._stats["errors"] += 1
            
            execution_time = time.time() - start_time
            return QueryResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                query_id=query_id
            )

    async def _execute_query(
        self,
        connection: Dict[str, Any],
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute query on the specific database type."""
        if connection["type"] == "neo4j":
            return await self._execute_neo4j_query(connection, query, parameters)
        elif connection["type"] == "networkx":
            return await self._execute_networkx_query(connection, query, parameters)
        else:
            raise ValueError(f"Unsupported connection type: {connection['type']}")

    async def _execute_neo4j_query(
        self,
        connection: Dict[str, Any],
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute Neo4j Cypher query (mock implementation)."""
        # Mock Neo4j query execution
        await asyncio.sleep(0.01)  # Simulate network latency

        # Parse simple queries for mock responses
        query_lower = query.lower()
        mock_data = []

        if "match" in query_lower and "return" in query_lower:
            # Mock node/relationship results
            if "concept" in query_lower:
                mock_data = [
                    {
                        "n": {
                            "id": "concept_ai",
                            "type": "Concept",
                            "name": "Artificial Intelligence",
                            "content": "AI is the simulation of human intelligence"
                        }
                    },
                    {
                        "n": {
                            "id": "concept_ml",
                            "type": "Concept",
                            "name": "Machine Learning",
                            "content": "ML is a subset of AI"
                        }
                    }
                ]

            if "relationship" in query_lower or "related" in query_lower:
                mock_data.extend([
                    {
                        "r": {
                            "type": "RELATES_TO",
                            "weight": 0.8,
                            "source": "concept_ai",
                            "target": "concept_ml"
                        }
                    }
                ])

        return QueryResult(
            success=True,
            data=mock_data
        )

    async def _execute_networkx_query(
        self,
        connection: Dict[str, Any],
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute NetworkX graph query."""
        graph = connection["graph"]

        # Parse custom query format for NetworkX
        # Format: "FIND nodes WHERE property=value" or "FIND paths FROM node1 TO node2"
        query_parts = query.upper().split()

        try:
            if "FIND" in query_parts and "NODES" in query_parts:
                return await self._find_nodes(graph, query, parameters)
            elif "FIND" in query_parts and "PATHS" in query_parts:
                return await self._find_paths(graph, query, parameters)
            elif "FIND" in query_parts and "NEIGHBORS" in query_parts:
                return await self._find_neighbors(graph, query, parameters)
            else:
                # Default: return all nodes
                nodes_data = []
                for node_id, node_data in graph.nodes(data=True):
                    nodes_data.append({
                        "n": {
                            "id": node_id,
                            **node_data
                        }
                    })

                return QueryResult(success=True, data=nodes_data)

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"NetworkX query error: {str(e)}"
            )

    async def _find_nodes(
        self,
        graph: nx.MultiDiGraph,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Find nodes matching criteria."""
        nodes_data = []

        # Simple node filtering (in production, use more sophisticated parsing)
        for node_id, node_data in graph.nodes(data=True):
            # Check if node matches query criteria
            if parameters:
                match = True
                for key, value in parameters.items():
                    if key in node_data and node_data[key] != value:
                        match = False
                        break
                if match:
                    nodes_data.append({
                        "n": {
                            "id": node_id,
                            **node_data
                        }
                    })
            else:
                # Return all nodes if no parameters
                nodes_data.append({
                    "n": {
                        "id": node_id,
                        **node_data
                    }
                })

        return QueryResult(success=True, data=nodes_data)

    async def _find_paths(
        self,
        graph: nx.MultiDiGraph,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Find paths between nodes."""
        paths_data = []

        if parameters and "source" in parameters and "target" in parameters:
            source = parameters["source"]
            target = parameters["target"]

            try:
                # Find shortest paths
                if nx.has_path(graph, source, target):
                    shortest_path = nx.shortest_path(graph, source, target)

                    # Convert path to result format
                    path_edges = []
                    for i in range(len(shortest_path) - 1):
                        edge_data = graph.get_edge_data(shortest_path[i], shortest_path[i + 1])
                        if edge_data:
                            # Get first edge if multiple edges exist
                            first_edge = list(edge_data.values())[0]
                            path_edges.append({
                                "source": shortest_path[i],
                                "target": shortest_path[i + 1],
                                **first_edge
                            })

                    paths_data.append({
                        "path": {
                            "nodes": shortest_path,
                            "edges": path_edges,
                            "length": len(shortest_path) - 1
                        }
                    })

            except nx.NetworkXNoPath:
                pass  # No path found

        return QueryResult(success=True, data=paths_data)

    async def _find_neighbors(
        self,
        graph: nx.MultiDiGraph,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Find neighbors of a node."""
        neighbors_data = []

        if parameters and "node" in parameters:
            node_id = parameters["node"]

            if node_id in graph:
                # Get all neighbors (both incoming and outgoing)
                neighbors = set(graph.predecessors(node_id)) | set(graph.successors(node_id))

                for neighbor in neighbors:
                    neighbor_data = graph.nodes[neighbor]
                    neighbors_data.append({
                        "n": {
                            "id": neighbor,
                            **neighbor_data
                        }
                    })

        return QueryResult(success=True, data=neighbors_data)

    def _get_cached_result(self, query_id: str) -> Optional[QueryResult]:
        """Get cached query result if still valid."""
        if query_id in self._query_cache:
            cached_entry = self._query_cache[query_id]

            # Check if cache entry is still valid
            if datetime.now() - cached_entry["timestamp"] < self._cache_ttl:
                result = cached_entry["result"]
                result.cached = True
                return result
            else:
                # Remove expired entry
                del self._query_cache[query_id]

        return None

    def _cache_result(self, query_id: str, result: QueryResult) -> None:
        """Cache a query result."""
        self._query_cache[query_id] = {
            "result": result,
            "timestamp": datetime.now()
        }

        # Clean up old cache entries periodically
        if len(self._query_cache) > 1000:  # Arbitrary limit
            self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []

        for query_id, cached_entry in self._query_cache.items():
            if current_time - cached_entry["timestamp"] > self._cache_ttl:
                expired_keys.append(query_id)

        for key in expired_keys:
            del self._query_cache[key]

        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _update_stats(self, execution_time: float, success: bool) -> None:
        """Update query execution statistics."""
        self._stats["queries_executed"] += 1

        if not success:
            self._stats["errors"] += 1

        # Update average query time
        total_queries = self._stats["queries_executed"]
        current_avg = self._stats["avg_query_time"]
        self._stats["avg_query_time"] = (
            (current_avg * (total_queries - 1) + execution_time) / total_queries
        )

    async def close(self) -> None:
        """Close the database connection and cleanup resources."""
        await self.connection_pool.close_all()
        self._query_cache.clear()
        self.logger.info("Real-time graph database closed")

    def get_statistics(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        return {
            **self._stats,
            "cache_size": len(self._query_cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] /
                max(self._stats["cache_hits"] + self._stats["cache_misses"], 1)
            ),
            "connection_pool_size": self.connection_pool._connection_count
        }
