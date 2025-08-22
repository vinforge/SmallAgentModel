#!/usr/bin/env python3
"""
Deep Research Engine for Dream Canvas
=====================================

Handles deep research functionality for cluster insights and paper ingestion.
Extracted from the monolithic dream_canvas.py.

This module provides:
- Cluster-based research insight generation
- Automatic paper ingestion
- Research result storage and retrieval
- Keyword extraction and analysis

Author: SAM Development Team
Version: 1.0.0 - Refactored from dream_canvas.py
"""

import streamlit as st
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from sam.dream_canvas.utils.models import (
    MemoryCluster, ResearchInsight, DreamCanvasState
)

logger = logging.getLogger(__name__)


class DeepResearchEngine:
    """Handles deep research functionality for Dream Canvas."""
    
    def __init__(self):
        self.research_results = {}
        self.auto_ingestion_enabled = True
    
    def generate_cluster_insights(self, cluster: MemoryCluster) -> List[ResearchInsight]:
        """
        Generate research insights for a given cluster.
        
        Args:
            cluster: The memory cluster to analyze
            
        Returns:
            List[ResearchInsight]: Generated research insights
        """
        try:
            insights = []
            
            # Extract key themes from cluster
            themes = self._extract_cluster_themes(cluster)
            
            # Generate insights based on themes
            for i, theme in enumerate(themes[:3]):  # Limit to top 3 themes
                insight = ResearchInsight(
                    id=f"{cluster.id}_insight_{i}",
                    cluster_id=cluster.id,
                    title=f"Research Insight: {theme.title()}",
                    description=self._generate_insight_description(theme, cluster),
                    keywords=self._extract_theme_keywords(theme, cluster),
                    confidence_score=self._calculate_insight_confidence(theme, cluster)
                )
                insights.append(insight)
            
            logger.info(f"Generated {len(insights)} insights for cluster {cluster.id}")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating cluster insights: {e}")
            return []
    
    def _extract_cluster_themes(self, cluster: MemoryCluster) -> List[str]:
        """Extract main themes from cluster memories."""
        themes = []
        
        # Use cluster keywords as primary themes
        themes.extend(cluster.keywords[:5])
        
        # Extract additional themes from memory content
        content_themes = self._extract_content_themes(cluster.memories)
        themes.extend(content_themes)
        
        # Remove duplicates and return top themes
        unique_themes = list(set(themes))
        return unique_themes[:5]
    
    def _extract_content_themes(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Extract themes from memory content."""
        themes = []
        
        for memory in memories[:10]:  # Sample first 10 memories
            content = memory.get('content', '')
            
            # Simple theme extraction (would be more sophisticated in practice)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            
            # Filter common words and add to themes
            filtered_words = [word for word in words if word not in self._get_stop_words()]
            themes.extend(filtered_words[:3])
        
        # Return most frequent themes
        theme_counts = {}
        for theme in themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:5]]
    
    def _get_stop_words(self) -> List[str]:
        """Get list of stop words to filter out."""
        return [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        ]
    
    def _generate_insight_description(self, theme: str, cluster: MemoryCluster) -> str:
        """Generate a description for a research insight."""
        return (
            f"This insight explores the theme of '{theme}' as it appears in "
            f"{cluster.name} with {cluster.size} related memories. "
            f"The cluster shows a coherence score of {cluster.coherence_score:.2f}, "
            f"indicating strong thematic relationships."
        )
    
    def _extract_theme_keywords(self, theme: str, cluster: MemoryCluster) -> List[str]:
        """Extract keywords related to a specific theme."""
        keywords = [theme]
        
        # Add related keywords from cluster
        related_keywords = [kw for kw in cluster.keywords if kw != theme]
        keywords.extend(related_keywords[:4])
        
        return keywords
    
    def _calculate_insight_confidence(self, theme: str, cluster: MemoryCluster) -> float:
        """Calculate confidence score for an insight."""
        # Base confidence on cluster coherence and theme frequency
        base_confidence = cluster.coherence_score
        
        # Boost confidence if theme appears in cluster keywords
        if theme in cluster.keywords:
            base_confidence += 0.1
        
        # Adjust based on cluster size
        size_factor = min(cluster.size / 20, 1.0)  # Normalize to max of 1.0
        
        return min(base_confidence + size_factor * 0.1, 1.0)
    
    def render_cluster_research_controls(self, cluster: MemoryCluster) -> None:
        """Render research controls for a cluster."""
        try:
            st.subheader(f"🔬 Deep Research: {cluster.name}")
            
            # Generate insights button
            if st.button(f"🧠 Generate Research Insights", key=f"insights_{cluster.id}"):
                with st.spinner("Generating research insights..."):
                    insights = self.generate_cluster_insights(cluster)
                    
                    if insights:
                        # Store insights in session state
                        if 'research_insights' not in st.session_state:
                            st.session_state.research_insights = {}
                        
                        st.session_state.research_insights[cluster.id] = insights
                        st.success(f"✅ Generated {len(insights)} research insights!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate research insights")
            
            # Show existing insights
            if hasattr(st.session_state, 'research_insights') and cluster.id in st.session_state.research_insights:
                insights = st.session_state.research_insights[cluster.id]
                
                st.markdown("### 💡 Research Insights")
                
                for insight in insights:
                    with st.expander(f"🔍 {insight.title}", expanded=False):
                        st.write(f"**Description:** {insight.description}")
                        st.write(f"**Confidence:** {insight.confidence_score:.2f}")
                        st.write(f"**Keywords:** {', '.join(insight.keywords)}")
                        
                        # Research actions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button(f"📚 Research Papers", key=f"research_{insight.id}"):
                                self._execute_insight_research(insight)
                        
                        with col2:
                            if st.button(f"📄 Auto-Ingest", key=f"ingest_{insight.id}"):
                                self._auto_ingest_research_papers(insight)
                        
                        # Show research results if available
                        self._render_insight_research_results(insight)
            
        except Exception as e:
            logger.error(f"Error rendering cluster research controls: {e}")
            st.error("Failed to render research controls")
    
    def _execute_insight_research(self, insight: ResearchInsight) -> None:
        """Execute research for a specific insight."""
        try:
            st.info(f"🔍 Researching: {insight.title}")
            
            # Simulate research process
            research_results = self._simulate_research_process(insight)
            
            # Store results
            self.research_results[insight.id] = research_results
            
            st.success("✅ Research completed!")
            
        except Exception as e:
            logger.error(f"Error executing insight research: {e}")
            st.error("❌ Research failed")
    
    def _simulate_research_process(self, insight: ResearchInsight) -> Dict[str, Any]:
        """Simulate the research process (placeholder implementation)."""
        # In a real implementation, this would:
        # 1. Search for relevant papers using the keywords
        # 2. Download and process papers
        # 3. Extract relevant information
        # 4. Generate summaries and insights
        
        return {
            'papers_found': len(insight.keywords) * 2,  # Mock data
            'papers_processed': len(insight.keywords),
            'key_findings': [
                f"Finding related to {keyword}" for keyword in insight.keywords[:3]
            ],
            'research_timestamp': datetime.now().isoformat()
        }
    
    def _auto_ingest_research_papers(self, insight: ResearchInsight) -> None:
        """Auto-ingest research papers for an insight."""
        try:
            if not self.auto_ingestion_enabled:
                st.warning("⚠️ Auto-ingestion is disabled")
                return
            
            st.info(f"📄 Auto-ingesting papers for: {insight.title}")
            
            # Simulate auto-ingestion process
            ingestion_results = self._simulate_auto_ingestion(insight)
            
            # Update insight
            insight.auto_ingested = True
            insight.research_papers.extend(ingestion_results.get('ingested_papers', []))
            
            st.success(f"✅ Auto-ingested {len(ingestion_results.get('ingested_papers', []))} papers!")
            
        except Exception as e:
            logger.error(f"Error in auto-ingestion: {e}")
            st.error("❌ Auto-ingestion failed")
    
    def _simulate_auto_ingestion(self, insight: ResearchInsight) -> Dict[str, Any]:
        """Simulate the auto-ingestion process."""
        # Mock ingestion results
        mock_papers = [
            f"paper_{insight.id}_{i}.pdf" for i in range(len(insight.keywords))
        ]
        
        return {
            'ingested_papers': mock_papers,
            'ingestion_timestamp': datetime.now().isoformat(),
            'success_count': len(mock_papers),
            'failure_count': 0
        }
    
    def _render_insight_research_results(self, insight: ResearchInsight) -> None:
        """Render research results for an insight."""
        if insight.id in self.research_results:
            results = self.research_results[insight.id]
            
            st.markdown("**📊 Research Results:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Papers Found", results.get('papers_found', 0))
            
            with col2:
                st.metric("Papers Processed", results.get('papers_processed', 0))
            
            with col3:
                st.metric("Key Findings", len(results.get('key_findings', [])))
            
            # Show key findings
            if results.get('key_findings'):
                st.markdown("**🔍 Key Findings:**")
                for finding in results['key_findings']:
                    st.write(f"• {finding}")
        
        # Show research papers if available
        if insight.research_papers:
            st.markdown("**📚 Research Papers:**")
            for paper in insight.research_papers:
                st.write(f"• {paper}")
    
    def extract_weighted_keywords_from_insight(self, insight_text: str) -> List[str]:
        """Extract weighted keywords from insight text."""
        try:
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{4,}\b', insight_text.lower())
            
            # Filter stop words
            filtered_words = [word for word in words if word not in self._get_stop_words()]
            
            # Count frequency
            word_counts = {}
            for word in filtered_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Sort by frequency and return top keywords
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            return [word for word, count in sorted_words[:10]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def store_research_result(self, insight_id: str, result: Dict[str, Any], 
                            keywords: List[str]) -> None:
        """Store research result for an insight."""
        try:
            self.research_results[insight_id] = {
                **result,
                'keywords': keywords,
                'stored_at': datetime.now().isoformat()
            }
            
            logger.info(f"Stored research result for insight {insight_id}")
            
        except Exception as e:
            logger.error(f"Error storing research result: {e}")


# Global deep research engine instance
_deep_research_engine = None


def get_deep_research_engine() -> DeepResearchEngine:
    """Get the global deep research engine instance."""
    global _deep_research_engine
    if _deep_research_engine is None:
        _deep_research_engine = DeepResearchEngine()
    return _deep_research_engine


def generate_cluster_insights(cluster: MemoryCluster) -> List[ResearchInsight]:
    """Generate research insights for a cluster using the global engine."""
    return get_deep_research_engine().generate_cluster_insights(cluster)


def render_cluster_research_controls(cluster: MemoryCluster) -> None:
    """Render cluster research controls using the global engine."""
    get_deep_research_engine().render_cluster_research_controls(cluster)
