"""
Main Synthesis Engine for SAM's Cognitive Synthesis System ("Dream Catcher")

This module orchestrates the complete cognitive synthesis process, from clustering
analysis through insight generation to output logging.
"""

import os
# Prevent torch conflicts with Streamlit
os.environ['PYTORCH_DISABLE_PER_OP_PROFILING'] = '1'

import logging
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .clustering_service import ClusteringService, ConceptCluster
from .prompt_generator import SynthesisPromptGenerator, SynthesisPrompt
from .insight_generator import InsightGenerator, SynthesizedInsight
from .chunk_formatter import SyntheticChunkFormatter, format_synthesis_output
from ..memory_vectorstore import MemoryVectorStore, MemoryType

logger = logging.getLogger(__name__)

@dataclass
class SynthesisConfig:
    """Configuration for the synthesis engine."""
    # Clustering parameters
    clustering_eps: float = 0.3
    clustering_min_samples: int = 3
    min_cluster_size: int = 5
    max_clusters: int = 20
    quality_threshold: float = 0.6

    # Prompt generation parameters
    max_chunks_per_prompt: int = 8
    max_content_length: int = 2000

    # Insight generation parameters
    llm_temperature: float = 0.7
    max_tokens: int = 200

    # Output parameters
    output_directory: str = "synthesis_output"
    min_insight_quality: float = 0.3  # Lowered from 0.6 to 0.3 for better insight acceptance
    fallback_insight_quality: float = 0.2  # Even lower threshold for fallback mode

    # Re-ingestion parameters (Phase 8B)
    enable_reingestion: bool = True
    enable_deduplication: bool = True

@dataclass
class SynthesisResult:
    """Result of a complete synthesis run."""
    run_id: str
    timestamp: str
    clusters_found: int
    insights_generated: int
    insights: List[SynthesizedInsight]
    synthesis_log: Dict[str, Any]
    output_file: str
    visualization_data: Optional[List[Dict[str, Any]]] = None  # Phase 8C: Dream Canvas data

class SynthesisEngine:
    """
    Main orchestrator for SAM's cognitive synthesis process.
    
    The SynthesisEngine coordinates the complete "dream state" analysis:
    1. Cluster analysis of memory vectors
    2. Prompt generation for each cluster
    3. LLM-based insight synthesis
    4. Quality filtering and output logging
    """
    
    def __init__(self, config: Optional[SynthesisConfig] = None, llm_client=None):
        """
        Initialize the synthesis engine.
        
        Args:
            config: Synthesis configuration (uses defaults if None)
            llm_client: LLM client for insight generation
        """
        self.config = config or SynthesisConfig()
        
        # Initialize component services
        self.clustering_service = ClusteringService(
            eps=self.config.clustering_eps,
            min_samples=self.config.clustering_min_samples,
            min_cluster_size=self.config.min_cluster_size,
            max_clusters=self.config.max_clusters,
            quality_threshold=self.config.quality_threshold
        )
        
        self.prompt_generator = SynthesisPromptGenerator(
            max_chunks_per_prompt=self.config.max_chunks_per_prompt,
            max_content_length=self.config.max_content_length
        )
        
        self.insight_generator = InsightGenerator(
            llm_client=llm_client,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Ensure output directory exists
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🧠 SynthesisEngine initialized - Dream Catcher ready")
    
    def run_synthesis(self, memory_store: MemoryVectorStore, visualize: bool = False) -> SynthesisResult:
        """
        Run the complete cognitive synthesis process.
        
        Args:
            memory_store: The memory vector store to analyze
            
        Returns:
            Complete synthesis results with generated insights
        """
        run_id = f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🌙 Starting cognitive synthesis run: {run_id}")
        
        try:
            # Phase 1: Cluster Analysis
            logger.info("Phase 1: Analyzing memory clusters...")
            clusters = self.clustering_service.find_concept_clusters(memory_store)
            
            if not clusters:
                logger.warning("No suitable clusters found for synthesis")
                return self._create_empty_result(run_id, "No clusters found")
            
            logger.info(f"Found {len(clusters)} concept clusters for synthesis")
            
            # Phase 2: Prompt Generation
            logger.info("Phase 2: Generating synthesis prompts...")
            synthesis_prompts = []
            
            for cluster in clusters:
                try:
                    logger.info(f"Attempting to generate prompt for cluster {cluster.cluster_id} with {cluster.size} chunks")
                    prompt = self.prompt_generator.generate_synthesis_prompt(cluster)
                    if prompt:
                        synthesis_prompts.append(prompt)
                        logger.info(f"✅ Successfully generated prompt for {cluster.cluster_id}")
                    else:
                        logger.warning(f"❌ Prompt generator returned None for {cluster.cluster_id}")
                except Exception as e:
                    logger.error(f"Error generating prompt for {cluster.cluster_id}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            if not synthesis_prompts:
                logger.warning("No synthesis prompts generated")
                return self._create_empty_result(run_id, "No prompts generated")
            
            logger.info(f"Generated {len(synthesis_prompts)} synthesis prompts")
            
            # Phase 3: Insight Generation
            logger.info("Phase 3: Generating synthesized insights...")
            insights = []

            # Track insights for fallback mode
            low_quality_insights = []

            for prompt in synthesis_prompts:
                try:
                    logger.info(f"Attempting to generate insight for cluster {prompt.cluster_id}")
                    insight = self.insight_generator.generate_insight(prompt)

                    if insight:
                        logger.info(f"Generated insight with confidence {insight.confidence_score:.2f} (min required: {self.config.min_insight_quality})")
                        if insight.confidence_score >= self.config.min_insight_quality:
                            insights.append(insight)
                            logger.info(f"✨ Accepted insight: {insight.insight_id}")
                        elif insight.confidence_score >= self.config.fallback_insight_quality:
                            # Store for potential fallback use
                            low_quality_insights.append(insight)
                            logger.info(f"📝 Stored low-quality insight for fallback: {insight.insight_id} (confidence: {insight.confidence_score:.2f})")
                        else:
                            logger.warning(f"⚠️ Insight quality too low: {insight.confidence_score:.2f} < {self.config.fallback_insight_quality}")
                    else:
                        logger.warning(f"❌ No insight generated for cluster {prompt.cluster_id}")

                except Exception as e:
                    logger.error(f"Error generating insight for {prompt.cluster_id}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

            # Fallback mode: If no high-quality insights, use low-quality ones
            if not insights and low_quality_insights:
                logger.warning(f"No high-quality insights found. Using {len(low_quality_insights)} low-quality insights as fallback.")
                insights = low_quality_insights[:5]  # Limit to top 5 low-quality insights
                for insight in insights:
                    logger.info(f"🔄 Using fallback insight: {insight.insight_id} (confidence: {insight.confidence_score:.2f})")

            logger.info(f"Generated {len(insights)} insights ({len(low_quality_insights)} additional low-quality insights available)")
            
            # Phase 4: Output Generation
            logger.info("Phase 4: Creating synthesis output...")
            synthesis_log = self._create_synthesis_log(run_id, clusters, synthesis_prompts, insights)
            output_file = self._save_synthesis_output(run_id, insights, synthesis_log)

            # Phase 5: Re-ingestion & Persistence (Phase 8B)
            reingested_count = 0
            if self.config.enable_reingestion and insights:
                logger.info("Phase 5: Re-ingesting synthetic insights into memory store...")
                reingested_count = self._reingest_synthetic_insights(output_file, memory_store)

            # Phase 6: Register clusters in cluster registry for UI access
            try:
                from .cluster_registry import get_cluster_registry
                registry = get_cluster_registry()
                registered_count = registry.register_clusters_from_synthesis(Path(output_file))
                logger.info(f"Phase 6: Registered {registered_count} clusters in registry")
            except Exception as e:
                logger.warning(f"Could not register clusters in registry: {e}")

            # Create final result
            result = SynthesisResult(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                clusters_found=len(clusters),
                insights_generated=len(insights),
                insights=insights,
                synthesis_log=synthesis_log,
                output_file=output_file
            )

            # Add re-ingestion info to synthesis log
            synthesis_log['reingestion'] = {
                'enabled': self.config.enable_reingestion,
                'chunks_reingested': reingested_count,
                'deduplication_enabled': self.config.enable_deduplication
            }

            # Phase 8C: Dream Canvas Visualization
            visualization_data = None
            if visualize:
                logger.info("Phase 6: Generating Dream Canvas visualization data...")
                visualization_data = self._generate_visualization_data(memory_store, clusters)
                synthesis_log['visualization'] = {
                    'enabled': True,
                    'total_points': len(visualization_data) if visualization_data else 0,
                    'clusters_visualized': len(clusters)
                }

            # Add visualization data to result if generated
            if visualization_data:
                result.visualization_data = visualization_data

            logger.info(f"🎉 Synthesis complete: {len(insights)} insights generated")
            logger.info(f"📄 Output saved to: {output_file}")

            return result
            
        except Exception as e:
            logger.error(f"Error in synthesis run {run_id}: {e}")
            return self._create_empty_result(run_id, f"Error: {e}")
    
    def _create_empty_result(self, run_id: str, reason: str) -> SynthesisResult:
        """Create an empty synthesis result for failed runs."""
        return SynthesisResult(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            clusters_found=0,
            insights_generated=0,
            insights=[],
            synthesis_log={"status": "failed", "reason": reason},
            output_file=""
        )
    
    def _create_synthesis_log(self, run_id: str, clusters: List[ConceptCluster], 
                             prompts: List[SynthesisPrompt], insights: List[SynthesizedInsight]) -> Dict[str, Any]:
        """Create comprehensive synthesis log."""
        return {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "configuration": asdict(self.config),
            "statistics": {
                "clusters_analyzed": len(clusters),
                "prompts_generated": len(prompts),
                "insights_created": len(insights),
                "avg_cluster_size": sum(c.size for c in clusters) / len(clusters) if clusters else 0,
                "avg_coherence_score": sum(c.coherence_score for c in clusters) / len(clusters) if clusters else 0,
                "avg_insight_confidence": sum(i.confidence_score for i in insights) / len(insights) if insights else 0
            },
            "cluster_summary": [
                {
                    "cluster_id": cluster.cluster_id,
                    "size": cluster.size,
                    "coherence_score": cluster.coherence_score,
                    "dominant_themes": cluster.dominant_themes,
                    "insight_generated": any(i.cluster_id == cluster.cluster_id for i in insights)
                }
                for cluster in clusters
            ]
        }
    
    def _save_synthesis_output(self, run_id: str, insights: List[SynthesizedInsight],
                              synthesis_log: Dict[str, Any]) -> str:
        """Save synthesis output to JSON file with enhanced cluster metadata."""
        # Create output data structure
        output_data = {
            "synthesis_run_log": synthesis_log,
            "insights": [],
            "cluster_metadata": {}  # NEW: Store cluster metadata for UI retrieval
        }

        # Extract cluster metadata from synthesis log
        if "cluster_summary" in synthesis_log:
            for cluster_info in synthesis_log["cluster_summary"]:
                cluster_id = cluster_info["cluster_id"]
                output_data["cluster_metadata"][cluster_id] = {
                    "cluster_id": cluster_id,
                    "size": cluster_info["size"],
                    "coherence_score": cluster_info["coherence_score"],
                    "dominant_themes": cluster_info["dominant_themes"],
                    "memory_count": cluster_info["size"],
                    "avg_importance": 0.0,  # Will be calculated from source chunks
                    "sources": [],  # Will be populated from source chunks
                    "insight_generated": cluster_info["insight_generated"]
                }

        # Add insights in the format expected by Phase 8B
        for insight in insights:
            insight_data = {
                "cluster_id": insight.cluster_id,
                "insight_id": insight.insight_id,
                "synthesized_text": insight.synthesized_text,
                "source_chunk_ids": insight.source_chunk_ids,
                "confidence_score": insight.confidence_score,
                "novelty_score": insight.novelty_score,
                "utility_score": insight.utility_score,
                "generated_at": insight.generated_at,
                "synthesis_metadata": insight.synthesis_metadata
            }
            output_data["insights"].append(insight_data)

            # Enhance cluster metadata with insight-specific data
            cluster_id = insight.cluster_id
            if cluster_id in output_data["cluster_metadata"]:
                cluster_meta = output_data["cluster_metadata"][cluster_id]

                # Calculate average importance from source chunks
                if insight.source_chunks:
                    importance_scores = [chunk.importance_score for chunk in insight.source_chunks]
                    cluster_meta["avg_importance"] = sum(importance_scores) / len(importance_scores)

                    # Extract unique sources
                    sources = list(set(chunk.source for chunk in insight.source_chunks))
                    cluster_meta["sources"] = sources
                    cluster_meta["source_count"] = len(sources)
        
        # Save to file
        output_file = self.output_dir / f"synthesis_run_log_{run_id}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Synthesis output saved to: {output_file}")
        return str(output_file)
    
    def get_synthesis_history(self) -> List[Dict[str, Any]]:
        """Get history of previous synthesis runs."""
        history = []
        
        for log_file in self.output_dir.glob("synthesis_run_log_*.json"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                synthesis_log = data.get("synthesis_run_log", {})
                history.append({
                    "run_id": synthesis_log.get("run_id"),
                    "timestamp": synthesis_log.get("timestamp"),
                    "insights_generated": len(data.get("insights", [])),
                    "clusters_analyzed": synthesis_log.get("statistics", {}).get("clusters_analyzed", 0),
                    "file_path": str(log_file)
                })
                
            except Exception as e:
                logger.error(f"Error reading synthesis log {log_file}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return history

    def _reingest_synthetic_insights(self, output_file: str, memory_store: MemoryVectorStore) -> int:
        """
        Re-ingest synthetic insights into the memory store (Phase 8B).

        Args:
            output_file: Path to synthesis output file
            memory_store: Memory store to ingest into

        Returns:
            Number of chunks successfully re-ingested
        """
        try:
            logger.info("🔄 Starting re-ingestion of synthetic insights...")

            # Create chunk formatter with deduplication support
            chunk_formatter = SyntheticChunkFormatter(
                memory_store=memory_store if self.config.enable_deduplication else None
            )

            # Format synthesis output into memory chunks
            synthetic_chunks = chunk_formatter.format_synthesis_output(output_file)

            if not synthetic_chunks:
                logger.warning("No synthetic chunks to re-ingest")
                return 0

            logger.info(f"Formatted {len(synthetic_chunks)} synthetic chunks for re-ingestion")

            # Re-ingest chunks using existing memory store pipeline
            reingested_count = 0

            for chunk in synthetic_chunks:
                try:
                    # Use the memory store's add_memory method for proper integration
                    chunk_id = memory_store.add_memory(
                        content=chunk.content,
                        memory_type=chunk.memory_type,
                        source=chunk.source,
                        tags=chunk.tags,
                        importance_score=chunk.importance_score,
                        metadata=chunk.metadata
                    )

                    if chunk_id:
                        reingested_count += 1
                        logger.debug(f"✅ Re-ingested synthetic chunk: {chunk_id}")

                except Exception as e:
                    logger.error(f"Error re-ingesting chunk {chunk.chunk_id}: {e}")
                    continue

            logger.info(f"🎉 Successfully re-ingested {reingested_count}/{len(synthetic_chunks)} synthetic insights")
            return reingested_count

        except Exception as e:
            logger.error(f"Error during re-ingestion: {e}")
            return 0

    def _generate_visualization_data(self, memory_store: MemoryVectorStore,
                                   clusters: List[ConceptCluster]) -> Optional[List[Dict[str, Any]]]:
        """
        Generate visualization data for the Dream Canvas (Phase 8C).

        Args:
            memory_store: Memory store containing all vectors
            clusters: Concept clusters from analysis

        Returns:
            List of visualization data points or None if generation fails
        """
        try:
            logger.info("🎨 Generating Dream Canvas visualization data...")

            # Import visualization dependencies
            try:
                import umap
            except ImportError:
                logger.error("UMAP not available - install with: pip install umap-learn")
                return None

            # Get all memories and their embeddings
            all_memories = memory_store.get_all_memories()

            if len(all_memories) < 10:
                logger.warning("Too few memories for meaningful visualization")
                return None

            logger.info(f"Processing {len(all_memories)} memories for visualization")

            # Extract embeddings and create mapping
            embeddings = []
            memory_mapping = {}

            for i, memory in enumerate(all_memories):
                if memory.embedding and len(memory.embedding) > 0:
                    embeddings.append(memory.embedding)
                    memory_mapping[i] = memory
                else:
                    # Generate embedding if missing
                    embedding = memory_store._generate_embedding(memory.content)
                    embeddings.append(embedding)
                    memory_mapping[i] = memory

            if len(embeddings) == 0:
                logger.error("No valid embeddings found for visualization")
                return None

            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            logger.info(f"Prepared {embeddings_array.shape[0]} embeddings for UMAP projection")

            # Run UMAP dimensionality reduction
            logger.info("Running UMAP dimensionality reduction...")
            umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )

            coordinates_2d = umap_model.fit_transform(embeddings_array)
            logger.info("✅ UMAP projection completed")

            # Create cluster mapping
            cluster_mapping = {}
            for cluster in clusters:
                for chunk in cluster.chunks:
                    cluster_mapping[chunk.chunk_id] = cluster.cluster_id

            # Generate visualization data points
            visualization_data = []

            for i, (memory, coords) in enumerate(zip(memory_mapping.values(), coordinates_2d)):
                # Get cluster ID (default to -1 for noise)
                cluster_id = cluster_mapping.get(memory.chunk_id, -1)

                # Create content snippet
                content_snippet = memory.content[:100] + "..." if len(memory.content) > 100 else memory.content

                # Create data point
                data_point = {
                    "chunk_id": memory.chunk_id,
                    "coordinates": {
                        "x": float(coords[0]),
                        "y": float(coords[1])
                    },
                    "cluster_id": cluster_id,
                    "content_snippet": content_snippet,
                    "memory_type": memory.memory_type.value,
                    "source": memory.source,
                    "importance_score": memory.importance_score,
                    "tags": memory.tags[:3],  # Limit tags for performance
                    "is_synthetic": memory.metadata.get('is_synthetic', False)
                }

                visualization_data.append(data_point)

            logger.info(f"✅ Generated {len(visualization_data)} visualization data points")
            return visualization_data

        except Exception as e:
            logger.error(f"Error generating visualization data: {e}")
            return None
