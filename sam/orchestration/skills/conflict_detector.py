"""
ConflictDetectorSkill - Information Conflict Detection and Resolution
====================================================================

Wraps SAM's Phase 5 conflict detection system into the SOF skill framework.
Detects and resolves conflicts between different information sources including
memory, external content, and tool outputs.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from ..uif import SAM_UIF
from .base import BaseSkillModule, SkillExecutionError

logger = logging.getLogger(__name__)


class ConflictDetectorSkill(BaseSkillModule):
    """
    Skill that detects and resolves conflicts between information sources.
    
    Analyzes conflicts between:
    - Memory retrieval results
    - External web content
    - Tool outputs
    - Document sources
    """
    
    skill_name = "ConflictDetectorSkill"
    skill_version = "1.0.0"
    skill_description = "Detects and resolves conflicts between information sources"
    skill_category = "analysis"
    
    # Dependency declarations
    required_inputs = ["input_query"]
    optional_inputs = [
        "memory_results", "external_content", "tool_outputs", 
        "retrieved_documents", "vetted_content"
    ]
    output_keys = ["conflict_analysis", "resolved_information", "conflict_confidence"]
    
    # Skill characteristics
    requires_external_access = False
    requires_vetting = False
    can_run_parallel = True
    estimated_execution_time = 1.0  # seconds
    
    def __init__(self):
        super().__init__()
        self._conflict_detector = None
        self._initialize_conflict_detection()
    
    def _initialize_conflict_detection(self) -> None:
        """Initialize conflict detection system."""
        try:
            # Try to import SAM's existing conflict detection
            from reasoning.conflict_detection import ConflictDetector
            self._conflict_detector = ConflictDetector()
            self.logger.info("Conflict detection system initialized")
            
        except ImportError:
            self.logger.warning("SAM conflict detection not available, using built-in logic")
            self._conflict_detector = None
    
    def execute(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Analyze and resolve conflicts between information sources.
        
        Args:
            uif: Universal Interface Format with information sources
            
        Returns:
            Updated UIF with conflict analysis and resolution
        """
        try:
            query = uif.input_query
            
            self.logger.info(f"Analyzing conflicts for query: {query[:100]}...")
            
            # Gather all information sources
            sources = self._gather_information_sources(uif)
            
            if len(sources) < 2:
                # No conflicts possible with less than 2 sources
                self._handle_no_conflicts(uif, sources)
                return uif
            
            # Perform conflict detection
            conflict_analysis = self._detect_conflicts(sources, query)
            
            # Resolve conflicts if any found
            resolved_info = self._resolve_conflicts(conflict_analysis, sources)
            
            # Calculate confidence in conflict resolution
            confidence = self._calculate_conflict_confidence(conflict_analysis, resolved_info)
            
            # Store results in UIF
            uif.intermediate_data["conflict_analysis"] = conflict_analysis
            uif.intermediate_data["resolved_information"] = resolved_info
            uif.intermediate_data["conflict_confidence"] = confidence
            
            # Set skill outputs
            uif.set_skill_output(self.skill_name, {
                "conflicts_found": len(conflict_analysis.get("conflicts", [])),
                "sources_analyzed": len(sources),
                "resolution_confidence": confidence,
                "resolution_method": conflict_analysis.get("resolution_method", "none")
            })
            
            # Add warnings if significant conflicts found
            conflicts = conflict_analysis.get("conflicts", [])
            if conflicts:
                uif.add_warning(f"Found {len(conflicts)} information conflicts that were resolved")
            
            self.logger.info(f"Conflict analysis completed: {len(conflicts)} conflicts found")
            
            return uif
            
        except Exception as e:
            self.logger.exception("Error during conflict detection")
            raise SkillExecutionError(f"Conflict detection failed: {str(e)}")
    
    def _gather_information_sources(self, uif: SAM_UIF) -> List[Dict[str, Any]]:
        """
        Gather all available information sources for conflict analysis.
        
        Returns:
            List of information sources with metadata
        """
        sources = []
        
        # Memory results
        memory_results = uif.intermediate_data.get("memory_results")
        if memory_results:
            sources.extend(self._extract_memory_sources(memory_results))
        
        # External content
        external_content = uif.intermediate_data.get("external_content")
        if external_content:
            sources.append({
                "type": "external",
                "content": external_content,
                "confidence": 0.7,  # Default confidence for external content
                "source_id": "external_web"
            })
        
        # Vetted content
        vetted_content = uif.intermediate_data.get("vetted_content")
        if vetted_content:
            sources.extend(self._extract_vetted_sources(vetted_content))
        
        # Tool outputs
        tool_outputs = uif.intermediate_data.get("tool_outputs", {})
        for tool_name, output in tool_outputs.items():
            sources.append({
                "type": "tool",
                "content": output,
                "confidence": 0.9,  # High confidence for tool outputs
                "source_id": f"tool_{tool_name}"
            })
        
        return sources
    
    def _extract_memory_sources(self, memory_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract individual sources from memory results."""
        sources = []
        
        all_results = memory_results.get("all_results", [])
        for result in all_results:
            sources.append({
                "type": "memory",
                "content": result.get("content", ""),
                "confidence": result.get("confidence", 0.0),
                "source_id": f"memory_{result.get('source', 'unknown')}"
            })
        
        return sources
    
    def _extract_vetted_sources(self, vetted_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sources from vetted content."""
        sources = []
        
        if isinstance(vetted_content, dict):
            for source_id, content in vetted_content.items():
                sources.append({
                    "type": "vetted",
                    "content": content,
                    "confidence": 0.8,  # High confidence for vetted content
                    "source_id": f"vetted_{source_id}"
                })
        
        return sources
    
    def _detect_conflicts(self, sources: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Detect conflicts between information sources.
        
        Returns:
            Dictionary with conflict analysis results
        """
        if self._conflict_detector:
            return self._use_sam_conflict_detector(sources, query)
        else:
            return self._use_builtin_conflict_detection(sources, query)
    
    def _use_sam_conflict_detector(self, sources: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Use SAM's built-in conflict detection system."""
        try:
            # Convert sources to format expected by SAM conflict detector
            formatted_sources = []
            for source in sources:
                formatted_sources.append({
                    "content": source["content"],
                    "source": source["source_id"],
                    "confidence": source["confidence"]
                })
            
            # Run conflict detection
            conflicts = self._conflict_detector.detect_conflicts(formatted_sources, query)
            
            return {
                "conflicts": conflicts,
                "method": "sam_conflict_detector",
                "resolution_method": "weighted_confidence"
            }
            
        except Exception as e:
            self.logger.error(f"SAM conflict detector failed: {e}")
            return self._use_builtin_conflict_detection(sources, query)
    
    def _use_builtin_conflict_detection(self, sources: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Use built-in conflict detection logic."""
        conflicts = []
        
        # Simple conflict detection based on contradictory statements
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources[i+1:], i+1):
                conflict = self._check_source_conflict(source1, source2, query)
                if conflict:
                    conflicts.append(conflict)
        
        return {
            "conflicts": conflicts,
            "method": "builtin_detection",
            "resolution_method": "confidence_weighted"
        }
    
    def _check_source_conflict(self, source1: Dict[str, Any], source2: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """
        Check if two sources conflict with each other.
        
        Returns:
            Conflict description if conflict found, None otherwise
        """
        content1 = str(source1["content"]).lower()
        content2 = str(source2["content"]).lower()
        
        # Simple keyword-based conflict detection
        conflict_indicators = [
            ("yes", "no"), ("true", "false"), ("correct", "incorrect"),
            ("valid", "invalid"), ("confirmed", "denied"), ("supports", "contradicts")
        ]
        
        for positive, negative in conflict_indicators:
            if positive in content1 and negative in content2:
                return {
                    "type": "contradiction",
                    "source1": source1["source_id"],
                    "source2": source2["source_id"],
                    "confidence1": source1["confidence"],
                    "confidence2": source2["confidence"],
                    "description": f"Contradiction between {positive} and {negative}"
                }
            elif negative in content1 and positive in content2:
                return {
                    "type": "contradiction",
                    "source1": source1["source_id"],
                    "source2": source2["source_id"],
                    "confidence1": source1["confidence"],
                    "confidence2": source2["confidence"],
                    "description": f"Contradiction between {negative} and {positive}"
                }
        
        return None
    
    def _resolve_conflicts(self, conflict_analysis: Dict[str, Any], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve detected conflicts using confidence-weighted resolution.
        
        Returns:
            Dictionary with resolved information
        """
        conflicts = conflict_analysis.get("conflicts", [])
        
        if not conflicts:
            return {
                "resolution_needed": False,
                "resolved_sources": sources,
                "resolution_notes": "No conflicts detected"
            }
        
        # Group sources by conflict involvement
        conflicted_sources = set()
        for conflict in conflicts:
            conflicted_sources.add(conflict["source1"])
            conflicted_sources.add(conflict["source2"])
        
        # Resolve conflicts by confidence weighting
        resolved_sources = []
        resolution_notes = []
        
        for source in sources:
            if source["source_id"] not in conflicted_sources:
                # No conflict, include as-is
                resolved_sources.append(source)
            else:
                # Check if this source wins its conflicts
                if self._source_wins_conflicts(source, conflicts):
                    resolved_sources.append(source)
                    resolution_notes.append(f"Kept {source['source_id']} due to higher confidence")
                else:
                    resolution_notes.append(f"Excluded {source['source_id']} due to lower confidence")
        
        return {
            "resolution_needed": True,
            "resolved_sources": resolved_sources,
            "resolution_notes": resolution_notes,
            "conflicts_resolved": len(conflicts)
        }
    
    def _source_wins_conflicts(self, source: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> bool:
        """
        Determine if a source wins its conflicts based on confidence.
        
        Returns:
            True if source should be kept, False if excluded
        """
        source_id = source["source_id"]
        source_confidence = source["confidence"]
        
        for conflict in conflicts:
            if conflict["source1"] == source_id:
                if source_confidence <= conflict["confidence2"]:
                    return False
            elif conflict["source2"] == source_id:
                if source_confidence <= conflict["confidence1"]:
                    return False
        
        return True
    
    def _calculate_conflict_confidence(self, conflict_analysis: Dict[str, Any], resolved_info: Dict[str, Any]) -> float:
        """
        Calculate confidence in the conflict resolution.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        conflicts = conflict_analysis.get("conflicts", [])
        
        if not conflicts:
            return 1.0  # Perfect confidence when no conflicts
        
        # Base confidence decreases with number of conflicts
        base_confidence = max(0.3, 1.0 - (len(conflicts) * 0.2))
        
        # Boost confidence if resolution was clear (high confidence differences)
        if resolved_info.get("resolution_needed"):
            resolved_sources = resolved_info.get("resolved_sources", [])
            if resolved_sources:
                avg_confidence = sum(s["confidence"] for s in resolved_sources) / len(resolved_sources)
                base_confidence += avg_confidence * 0.3
        
        return min(1.0, base_confidence)
    
    def _handle_no_conflicts(self, uif: SAM_UIF, sources: List[Dict[str, Any]]) -> None:
        """Handle case where no conflicts are possible."""
        uif.intermediate_data["conflict_analysis"] = {
            "conflicts": [],
            "method": "no_analysis_needed",
            "resolution_method": "none"
        }
        
        uif.intermediate_data["resolved_information"] = {
            "resolution_needed": False,
            "resolved_sources": sources,
            "resolution_notes": f"Only {len(sources)} source(s) available, no conflicts possible"
        }
        
        uif.intermediate_data["conflict_confidence"] = 1.0
        
        uif.set_skill_output(self.skill_name, {
            "conflicts_found": 0,
            "sources_analyzed": len(sources),
            "resolution_confidence": 1.0,
            "resolution_method": "none"
        })
