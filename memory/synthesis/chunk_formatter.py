"""
Synthetic Chunk Formatter for SAM's Cognitive Synthesis Engine

This module converts synthesis engine output into properly formatted memory chunks
for re-ingestion into SAM's main memory store with full transparency metadata.
"""

import logging
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..memory_vectorstore import MemoryChunk, MemoryType, MemoryVectorStore
from .insight_generator import SynthesizedInsight

logger = logging.getLogger(__name__)

class SyntheticChunkFormatter:
    """
    Formats synthesis engine output into memory chunks for re-ingestion.
    
    This class handles the critical "Waking Up" phase where synthesized insights
    from the Dream Catcher are converted into properly formatted memory chunks
    with full transparency metadata for storage in SAM's main memory store.
    """
    
    def __init__(self, memory_store: Optional[MemoryVectorStore] = None):
        """
        Initialize the synthetic chunk formatter.
        
        Args:
            memory_store: Memory store for de-duplication checks (optional)
        """
        self.memory_store = memory_store
        logger.info("SyntheticChunkFormatter initialized")
    
    def format_synthesis_output(self, synthesis_output_file: str) -> List[MemoryChunk]:
        """
        Convert synthesis output JSON to formatted memory chunks.
        
        Args:
            synthesis_output_file: Path to synthesis_run_log.json file
            
        Returns:
            List of formatted MemoryChunk objects ready for ingestion
        """
        try:
            logger.info(f"🔄 Formatting synthesis output from: {synthesis_output_file}")
            
            # Load synthesis output
            synthesis_data = self._load_synthesis_output(synthesis_output_file)
            
            if not synthesis_data or 'insights' not in synthesis_data:
                logger.warning("No insights found in synthesis output")
                return []
            
            insights = synthesis_data['insights']
            logger.info(f"Processing {len(insights)} synthesized insights")
            
            # Format each insight into a memory chunk
            formatted_chunks = []
            
            for insight in insights:
                try:
                    # Check for duplicates and handle accordingly
                    existing_chunk = self._check_for_duplicate(insight)
                    
                    if existing_chunk:
                        # Update existing chunk instead of creating new one
                        updated_chunk = self._update_existing_chunk(existing_chunk, insight)
                        if updated_chunk:
                            formatted_chunks.append(updated_chunk)
                            logger.info(f"✅ Updated existing synthetic chunk for cluster {insight['cluster_id']}")
                    else:
                        # Create new synthetic chunk
                        new_chunk = self._create_synthetic_chunk(insight)
                        if new_chunk:
                            formatted_chunks.append(new_chunk)
                            logger.info(f"✨ Created new synthetic chunk: {new_chunk.chunk_id}")
                
                except Exception as e:
                    logger.error(f"Error formatting insight {insight.get('insight_id', 'unknown')}: {e}")
                    continue
            
            logger.info(f"🎉 Successfully formatted {len(formatted_chunks)} synthetic chunks")
            return formatted_chunks
            
        except Exception as e:
            logger.error(f"Error formatting synthesis output: {e}")
            return []
    
    def _load_synthesis_output(self, output_file: str) -> Dict[str, Any]:
        """Load and validate synthesis output JSON."""
        try:
            output_path = Path(output_file)
            if not output_path.exists():
                raise FileNotFoundError(f"Synthesis output file not found: {output_file}")
            
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if 'synthesis_run_log' not in data:
                raise ValueError("Invalid synthesis output: missing synthesis_run_log")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading synthesis output: {e}")
            raise
    
    def _check_for_duplicate(self, insight: Dict[str, Any]) -> Optional[MemoryChunk]:
        """
        Check if a synthetic chunk for this cluster already exists.
        
        Args:
            insight: Insight data from synthesis output
            
        Returns:
            Existing MemoryChunk if found, None otherwise
        """
        if not self.memory_store:
            return None
        
        try:
            cluster_id = insight.get('cluster_id')
            if not cluster_id:
                return None
            
            # Search for existing synthetic chunks with this cluster ID
            # Using metadata filter to find synthetic chunks
            existing_memories = self.memory_store.get_all_memories()
            
            for memory in existing_memories:
                if (memory.memory_type == MemoryType.SYNTHESIS and 
                    memory.metadata.get('synthesis_cluster_id') == cluster_id):
                    logger.debug(f"Found existing synthetic chunk for cluster {cluster_id}")
                    return memory
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking for duplicates: {e}")
            return None
    
    def _update_existing_chunk(self, existing_chunk: MemoryChunk, insight: Dict[str, Any]) -> Optional[MemoryChunk]:
        """
        Update an existing synthetic chunk with new insight data.
        
        Args:
            existing_chunk: Existing memory chunk to update
            insight: New insight data
            
        Returns:
            Updated MemoryChunk or None if no update needed
        """
        try:
            # Check if the new insight is better than the existing one
            new_confidence = insight.get('confidence_score', 0.0)
            existing_confidence = existing_chunk.metadata.get('synthesis_confidence_score', 0.0)
            
            # Only update if new insight has higher confidence or is significantly newer
            if new_confidence <= existing_confidence:
                logger.debug(f"Skipping update - existing chunk has higher confidence")
                return None
            
            # Create updated chunk with new content but preserve chunk_id
            updated_chunk = MemoryChunk(
                chunk_id=existing_chunk.chunk_id,  # Keep same ID
                content=insight['synthesized_text'],
                content_hash=self._generate_content_hash(insight['synthesized_text']),
                embedding=None,  # Will be regenerated during ingestion
                memory_type=MemoryType.SYNTHESIS,
                source="SAM Cognitive Synthesis",
                timestamp=datetime.now().isoformat(),
                tags=self._generate_synthesis_tags(insight),
                importance_score=self._calculate_priority_score(insight),
                access_count=existing_chunk.access_count,  # Preserve access count
                last_accessed=existing_chunk.last_accessed,
                metadata=self._create_synthesis_metadata(insight, is_update=True)
            )
            
            logger.info(f"Updated synthetic chunk {existing_chunk.chunk_id} with higher confidence insight")
            return updated_chunk
            
        except Exception as e:
            logger.error(f"Error updating existing chunk: {e}")
            return None
    
    def _create_synthetic_chunk(self, insight: Dict[str, Any]) -> Optional[MemoryChunk]:
        """
        Create a new synthetic memory chunk from insight data.
        
        Args:
            insight: Insight data from synthesis output
            
        Returns:
            New MemoryChunk object or None if creation fails
        """
        try:
            # Generate unique chunk ID for synthetic content
            chunk_id = f"synth_{insight['cluster_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create the synthetic memory chunk
            synthetic_chunk = MemoryChunk(
                chunk_id=chunk_id,
                content=insight['synthesized_text'],
                content_hash=self._generate_content_hash(insight['synthesized_text']),
                embedding=None,  # Will be generated during ingestion
                memory_type=MemoryType.SYNTHESIS,
                source="SAM Cognitive Synthesis",
                timestamp=datetime.now().isoformat(),
                tags=self._generate_synthesis_tags(insight),
                importance_score=self._calculate_priority_score(insight),
                access_count=0,
                last_accessed=datetime.now().isoformat(),
                metadata=self._create_synthesis_metadata(insight)
            )
            
            return synthetic_chunk
            
        except Exception as e:
            logger.error(f"Error creating synthetic chunk: {e}")
            return None
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate content hash for deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _generate_synthesis_tags(self, insight: Dict[str, Any]) -> List[str]:
        """Generate appropriate tags for synthetic content."""
        tags = ['synthetic', 'cognitive_synthesis', 'dream_catcher']
        
        # Add cluster-based tag
        if 'cluster_id' in insight:
            tags.append(f"cluster_{insight['cluster_id']}")
        
        # Add quality-based tags
        confidence = insight.get('confidence_score', 0.0)
        if confidence >= 0.8:
            tags.append('high_confidence')
        elif confidence >= 0.6:
            tags.append('medium_confidence')
        else:
            tags.append('low_confidence')
        
        # Add novelty tags
        novelty = insight.get('novelty_score', 0.0)
        if novelty >= 0.7:
            tags.append('novel_insight')
        
        # Add utility tags
        utility = insight.get('utility_score', 0.0)
        if utility >= 0.7:
            tags.append('high_utility')
        
        return tags
    
    def _calculate_priority_score(self, insight: Dict[str, Any]) -> float:
        """Calculate priority score for synthetic content."""
        # Combine quality scores for overall priority
        confidence = insight.get('confidence_score', 0.5)
        novelty = insight.get('novelty_score', 0.5)
        utility = insight.get('utility_score', 0.5)
        
        # Weighted combination (utility weighted higher for actionable insights)
        priority = (confidence * 0.4 + utility * 0.4 + novelty * 0.2)
        
        # Ensure priority is in valid range
        return max(0.1, min(1.0, priority))
    
    def _create_synthesis_metadata(self, insight: Dict[str, Any], is_update: bool = False) -> Dict[str, Any]:
        """Create comprehensive metadata for synthetic chunks."""
        metadata = {
            # --- CRITICAL METADATA FOR TRANSPARENCY ---
            'source_name': "SAM Cognitive Synthesis",
            'author': "SAM",
            'is_synthetic': True,
            'synthesized_from_chunks': insight.get('source_chunk_ids', []),
            'synthesis_cluster_id': insight.get('cluster_id'),
            'synthesis_confidence_score': insight.get('confidence_score', 0.0),
            'synthesis_novelty_score': insight.get('novelty_score', 0.0),
            'synthesis_utility_score': insight.get('utility_score', 0.0),
            
            # --- SYNTHESIS PROCESS METADATA ---
            'synthesis_timestamp': insight.get('generated_at', datetime.now().isoformat()),
            'synthesis_method': 'cognitive_consolidation',
            'synthesis_engine_version': 'phase_8b',
            
            # --- QUALITY AND TRACKING ---
            'insight_id': insight.get('insight_id'),
            'source_count': len(insight.get('source_chunk_ids', [])),
            'is_update': is_update,
            'last_synthesis_update': datetime.now().isoformat() if is_update else None,
            
            # --- ADDITIONAL SYNTHESIS METADATA ---
            'synthesis_metadata': insight.get('synthesis_metadata', {})
        }
        
        return metadata


def format_synthesis_output(synthesis_output_file: str, memory_store: Optional[MemoryVectorStore] = None) -> List[MemoryChunk]:
    """
    Convenience function to format synthesis output into memory chunks.
    
    Args:
        synthesis_output_file: Path to synthesis_run_log.json file
        memory_store: Optional memory store for de-duplication
        
    Returns:
        List of formatted MemoryChunk objects
    """
    formatter = SyntheticChunkFormatter(memory_store)
    return formatter.format_synthesis_output(synthesis_output_file)
