"""
Enhanced Citation and Source Tracking Engine for SAM's Cognitive Memory Core - Phase C
Implements comprehensive source attribution, citation graphs, and credibility scoring.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import json
import networkx as nx
from enum import Enum

logger = logging.getLogger(__name__)

class SourceType(Enum):
    """Types of information sources."""
    DOCUMENT = "document"
    WEB_PAGE = "web_page"
    CONVERSATION = "conversation"
    MEMORY = "memory"
    KNOWLEDGE_BASE = "knowledge_base"
    USER_INPUT = "user_input"
    SYSTEM_GENERATED = "system_generated"

class CredibilityLevel(Enum):
    """Credibility levels for sources."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    UNKNOWN = "unknown"

@dataclass
class SourceMetadata:
    """Metadata for a source."""
    source_id: str
    source_type: SourceType
    title: str = ""
    author: str = ""
    publication_date: Optional[datetime] = None
    url: str = ""
    domain: str = ""
    content_hash: str = ""
    file_path: str = ""
    credibility_score: float = 0.5
    credibility_level: CredibilityLevel = CredibilityLevel.UNKNOWN
    verification_status: str = "unverified"
    last_updated: Optional[datetime] = None

@dataclass
class Citation:
    """A citation linking content to its source."""
    citation_id: str
    source_id: str
    content_snippet: str
    context: str = ""
    confidence: float = 1.0
    page_number: Optional[int] = None
    section: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    verification_method: str = "automatic"

@dataclass
class CitationGraph:
    """A graph representing citation relationships."""
    graph_id: str
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class CredibilityAssessment:
    """Assessment of source credibility."""
    source_id: str
    overall_score: float
    factors: Dict[str, float] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    assessment_date: datetime = field(default_factory=datetime.now)
    assessor: str = "system"

class CitationEngine:
    """
    Enhanced citation and source tracking engine.
    
    Features:
    - Comprehensive source attribution
    - Citation graph generation
    - Credibility scoring and assessment
    - Provenance tracking
    - Source verification
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CitationEngine")
        self._sources: Dict[str, SourceMetadata] = {}
        self._citations: Dict[str, Citation] = {}
        self._citation_graph = nx.DiGraph()
        self._credibility_cache: Dict[str, CredibilityAssessment] = {}
        
        # Credibility scoring weights
        self.credibility_weights = {
            "domain_authority": 0.25,
            "author_reputation": 0.20,
            "publication_quality": 0.20,
            "content_accuracy": 0.15,
            "recency": 0.10,
            "peer_validation": 0.10
        }
        
        self.logger.info("Citation Engine initialized")
    
    async def register_source(
        self,
        source_data: Dict[str, Any],
        content: str = ""
    ) -> SourceMetadata:
        """
        Register a new source in the citation system.
        
        Args:
            source_data: Source information dictionary
            content: Source content for hash generation
            
        Returns:
            SourceMetadata object for the registered source
        """
        # Generate source ID
        source_id = self._generate_source_id(source_data, content)
        
        # Create source metadata
        source_metadata = SourceMetadata(
            source_id=source_id,
            source_type=SourceType(source_data.get("type", "document")),
            title=source_data.get("title", ""),
            author=source_data.get("author", ""),
            publication_date=source_data.get("publication_date"),
            url=source_data.get("url", ""),
            domain=self._extract_domain(source_data.get("url", "")),
            content_hash=self._generate_content_hash(content),
            file_path=source_data.get("file_path", ""),
            last_updated=datetime.now()
        )
        
        # Assess initial credibility
        credibility_assessment = await self._assess_credibility(source_metadata, content)
        source_metadata.credibility_score = credibility_assessment.overall_score
        source_metadata.credibility_level = self._score_to_level(credibility_assessment.overall_score)
        
        # Store source
        self._sources[source_id] = source_metadata
        self._credibility_cache[source_id] = credibility_assessment
        
        # Add to citation graph
        self._citation_graph.add_node(source_id, **source_metadata.__dict__)
        
        self.logger.info(f"Registered source: {source_id} ({source_metadata.source_type.value})")
        return source_metadata
    
    async def create_citation(
        self,
        source_id: str,
        content_snippet: str,
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Citation:
        """
        Create a citation linking content to its source.
        
        Args:
            source_id: ID of the source being cited
            content_snippet: The specific content being cited
            context: Context around the citation
            metadata: Additional citation metadata
            
        Returns:
            Citation object
        """
        if source_id not in self._sources:
            raise ValueError(f"Source {source_id} not found")
        
        # Generate citation ID
        citation_id = self._generate_citation_id(source_id, content_snippet)
        
        # Create citation
        citation = Citation(
            citation_id=citation_id,
            source_id=source_id,
            content_snippet=content_snippet,
            context=context,
            confidence=metadata.get("confidence", 1.0) if metadata else 1.0,
            page_number=metadata.get("page_number") if metadata else None,
            section=metadata.get("section", "") if metadata else "",
            verification_method=metadata.get("verification_method", "automatic") if metadata else "automatic"
        )
        
        # Store citation
        self._citations[citation_id] = citation
        
        # Add citation edge to graph
        self._citation_graph.add_edge(
            "content_" + citation_id,
            source_id,
            citation_id=citation_id,
            snippet=content_snippet[:100],  # Truncate for graph storage
            confidence=citation.confidence
        )
        
        self.logger.debug(f"Created citation: {citation_id}")
        return citation
    
    async def generate_citation_graph(
        self,
        query: str,
        max_depth: int = 3
    ) -> CitationGraph:
        """
        Generate a citation graph for a specific query or topic.
        
        Args:
            query: Query or topic to generate graph for
            max_depth: Maximum depth for graph traversal
            
        Returns:
            CitationGraph object
        """
        graph_id = f"citation_graph_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        
        # Find relevant sources and citations
        relevant_sources = await self._find_relevant_sources(query)
        relevant_citations = await self._find_relevant_citations(query)
        
        # Build subgraph
        subgraph = self._citation_graph.subgraph(
            [s.source_id for s in relevant_sources] + 
            [f"content_{c.citation_id}" for c in relevant_citations]
        )
        
        # Convert to serializable format
        nodes = []
        for node_id, node_data in subgraph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "type": "source" if not node_id.startswith("content_") else "content",
                **{k: str(v) if isinstance(v, datetime) else v for k, v in node_data.items()}
            })
        
        edges = []
        for source, target, edge_data in subgraph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                **edge_data
            })
        
        citation_graph = CitationGraph(
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
            metadata={
                "query": query,
                "source_count": len(relevant_sources),
                "citation_count": len(relevant_citations),
                "max_depth": max_depth
            }
        )
        
        self.logger.info(f"Generated citation graph: {graph_id} with {len(nodes)} nodes, {len(edges)} edges")
        return citation_graph
    
    async def _assess_credibility(
        self,
        source_metadata: SourceMetadata,
        content: str = ""
    ) -> CredibilityAssessment:
        """Assess the credibility of a source."""
        factors = {}
        reasoning = []
        
        # Domain authority assessment
        domain_score = await self._assess_domain_authority(source_metadata.domain)
        factors["domain_authority"] = domain_score
        if domain_score > 0.7:
            reasoning.append(f"High domain authority for {source_metadata.domain}")
        elif domain_score < 0.3:
            reasoning.append(f"Low domain authority for {source_metadata.domain}")
        
        # Author reputation assessment
        author_score = await self._assess_author_reputation(source_metadata.author)
        factors["author_reputation"] = author_score
        if source_metadata.author and author_score > 0.6:
            reasoning.append(f"Reputable author: {source_metadata.author}")
        
        # Publication quality assessment
        pub_score = await self._assess_publication_quality(source_metadata)
        factors["publication_quality"] = pub_score
        
        # Content accuracy assessment
        content_score = await self._assess_content_accuracy(content)
        factors["content_accuracy"] = content_score
        
        # Recency assessment
        recency_score = self._assess_recency(source_metadata.publication_date)
        factors["recency"] = recency_score
        if recency_score < 0.3:
            reasoning.append("Content may be outdated")
        
        # Peer validation assessment
        peer_score = await self._assess_peer_validation(source_metadata.source_id)
        factors["peer_validation"] = peer_score
        
        # Calculate overall score
        overall_score = sum(
            factors[factor] * self.credibility_weights[factor]
            for factor in factors
        )
        
        return CredibilityAssessment(
            source_id=source_metadata.source_id,
            overall_score=overall_score,
            factors=factors,
            reasoning=reasoning
        )

    async def _assess_domain_authority(self, domain: str) -> float:
        """Assess the authority of a domain."""
        if not domain:
            return 0.5

        # High-authority domains
        high_authority = {
            "wikipedia.org": 0.9,
            "nature.com": 0.95,
            "science.org": 0.95,
            "arxiv.org": 0.85,
            "pubmed.ncbi.nlm.nih.gov": 0.9,
            "scholar.google.com": 0.8,
            "ieee.org": 0.9,
            "acm.org": 0.85
        }

        # Medium-authority domains
        medium_authority = {
            "github.com": 0.7,
            "stackoverflow.com": 0.7,
            "medium.com": 0.6,
            "researchgate.net": 0.7
        }

        # Check exact matches
        if domain in high_authority:
            return high_authority[domain]
        if domain in medium_authority:
            return medium_authority[domain]

        # Check domain patterns
        if any(pattern in domain for pattern in [".edu", ".gov", ".org"]):
            return 0.75
        elif any(pattern in domain for pattern in [".com", ".net"]):
            return 0.6
        else:
            return 0.5

    async def _assess_author_reputation(self, author: str) -> float:
        """Assess the reputation of an author."""
        if not author:
            return 0.5

        # Mock author reputation assessment
        # In production, this would check academic databases, citation counts, etc.
        known_authors = {
            "Geoffrey Hinton": 0.95,
            "Yann LeCun": 0.95,
            "Yoshua Bengio": 0.95,
            "Andrew Ng": 0.9,
            "Fei-Fei Li": 0.9
        }

        return known_authors.get(author, 0.6)

    async def _assess_publication_quality(self, source_metadata: SourceMetadata) -> float:
        """Assess the quality of the publication venue."""
        # Mock publication quality assessment
        if source_metadata.source_type == SourceType.DOCUMENT:
            return 0.8
        elif source_metadata.source_type == SourceType.WEB_PAGE:
            return 0.6
        elif source_metadata.source_type == SourceType.KNOWLEDGE_BASE:
            return 0.9
        else:
            return 0.5

    async def _assess_content_accuracy(self, content: str) -> float:
        """Assess the accuracy of content."""
        if not content:
            return 0.5

        # Mock content accuracy assessment
        # In production, this would use fact-checking APIs, cross-referencing, etc.
        accuracy_indicators = [
            "peer-reviewed", "published", "research", "study", "analysis",
            "data", "evidence", "methodology", "conclusion", "abstract"
        ]

        content_lower = content.lower()
        indicator_count = sum(1 for indicator in accuracy_indicators if indicator in content_lower)

        # Score based on presence of accuracy indicators
        base_score = 0.5
        indicator_boost = min(0.4, indicator_count * 0.05)

        return base_score + indicator_boost

    def _assess_recency(self, publication_date: Optional[datetime]) -> float:
        """Assess the recency of the source."""
        if not publication_date:
            return 0.5

        days_old = (datetime.now() - publication_date).days

        if days_old <= 30:
            return 1.0
        elif days_old <= 365:
            return 0.8
        elif days_old <= 1825:  # 5 years
            return 0.6
        elif days_old <= 3650:  # 10 years
            return 0.4
        else:
            return 0.2

    async def _assess_peer_validation(self, source_id: str) -> float:
        """Assess peer validation through citations and references."""
        # Mock peer validation assessment
        # In production, this would check citation counts, cross-references, etc.

        # Count how many other sources reference this one
        reference_count = sum(
            1 for citation in self._citations.values()
            if citation.source_id == source_id
        )

        # Score based on reference count
        if reference_count >= 10:
            return 0.9
        elif reference_count >= 5:
            return 0.7
        elif reference_count >= 2:
            return 0.6
        elif reference_count >= 1:
            return 0.5
        else:
            return 0.3

    def _score_to_level(self, score: float) -> CredibilityLevel:
        """Convert credibility score to level."""
        if score >= 0.9:
            return CredibilityLevel.VERY_HIGH
        elif score >= 0.7:
            return CredibilityLevel.HIGH
        elif score >= 0.5:
            return CredibilityLevel.MEDIUM
        elif score >= 0.3:
            return CredibilityLevel.LOW
        else:
            return CredibilityLevel.VERY_LOW

    async def _find_relevant_sources(self, query: str) -> List[SourceMetadata]:
        """Find sources relevant to a query."""
        relevant_sources = []
        query_lower = query.lower()

        for source in self._sources.values():
            # Simple relevance check (in production, use semantic similarity)
            if (query_lower in source.title.lower() or
                query_lower in source.author.lower() or
                any(word in source.title.lower() for word in query_lower.split())):
                relevant_sources.append(source)

        # Sort by credibility score
        relevant_sources.sort(key=lambda s: s.credibility_score, reverse=True)
        return relevant_sources[:20]  # Limit results

    async def _find_relevant_citations(self, query: str) -> List[Citation]:
        """Find citations relevant to a query."""
        relevant_citations = []
        query_lower = query.lower()

        for citation in self._citations.values():
            # Simple relevance check
            if (query_lower in citation.content_snippet.lower() or
                query_lower in citation.context.lower()):
                relevant_citations.append(citation)

        # Sort by confidence
        relevant_citations.sort(key=lambda c: c.confidence, reverse=True)
        return relevant_citations[:50]  # Limit results

    def _generate_source_id(self, source_data: Dict[str, Any], content: str) -> str:
        """Generate a unique source ID."""
        # Create hash from key source attributes
        hash_input = f"{source_data.get('url', '')}{source_data.get('title', '')}{source_data.get('file_path', '')}"
        return f"source_{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"

    def _generate_citation_id(self, source_id: str, content_snippet: str) -> str:
        """Generate a unique citation ID."""
        hash_input = f"{source_id}{content_snippet[:100]}"
        return f"cite_{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"

    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ""

        # Simple domain extraction
        if "://" in url:
            domain = url.split("://")[1].split("/")[0]
        else:
            domain = url.split("/")[0]

        # Remove www prefix
        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered sources."""
        if not self._sources:
            return {"total_sources": 0}

        credibility_distribution = defaultdict(int)
        source_type_distribution = defaultdict(int)

        for source in self._sources.values():
            credibility_distribution[source.credibility_level.value] += 1
            source_type_distribution[source.source_type.value] += 1

        avg_credibility = sum(s.credibility_score for s in self._sources.values()) / len(self._sources)

        return {
            "total_sources": len(self._sources),
            "total_citations": len(self._citations),
            "average_credibility": round(avg_credibility, 3),
            "credibility_distribution": dict(credibility_distribution),
            "source_type_distribution": dict(source_type_distribution),
            "graph_nodes": self._citation_graph.number_of_nodes(),
            "graph_edges": self._citation_graph.number_of_edges()
        }

    async def get_source_by_id(self, source_id: str) -> Optional[SourceMetadata]:
        """Get source metadata by ID."""
        return self._sources.get(source_id)

    async def get_citations_for_source(self, source_id: str) -> List[Citation]:
        """Get all citations for a specific source."""
        return [citation for citation in self._citations.values()
                if citation.source_id == source_id]

    async def update_credibility_assessment(
        self,
        source_id: str,
        new_factors: Dict[str, float]
    ) -> Optional[CredibilityAssessment]:
        """Update credibility assessment for a source."""
        if source_id not in self._sources:
            return None

        # Get existing assessment or create new one
        assessment = self._credibility_cache.get(source_id)
        if not assessment:
            source = self._sources[source_id]
            assessment = await self._assess_credibility(source)

        # Update factors
        assessment.factors.update(new_factors)

        # Recalculate overall score
        assessment.overall_score = sum(
            assessment.factors[factor] * self.credibility_weights[factor]
            for factor in assessment.factors
        )

        # Update source metadata
        source = self._sources[source_id]
        source.credibility_score = assessment.overall_score
        source.credibility_level = self._score_to_level(assessment.overall_score)

        # Update cache
        self._credibility_cache[source_id] = assessment

        return assessment
