#!/usr/bin/env python3
"""
STEF Program Registry
Central registry for all available STEF TaskPrograms.
"""

import logging
from typing import Dict, List, Optional
from .task_definitions import TaskProgram
from .programs.fact_check_calculator import (
    fact_check_and_calculate_program,
    simple_calculation_program,
    document_calculation_program
)

logger = logging.getLogger(__name__)

class ProgramRegistry:
    """
    Central registry for managing STEF TaskPrograms.
    
    Provides methods for registering, discovering, and selecting programs
    based on query analysis.
    """
    
    def __init__(self):
        self._programs: Dict[str, TaskProgram] = {}
        self._initialize_default_programs()
        
        logger.info(f"ProgramRegistry initialized with {len(self._programs)} programs")
    
    def _initialize_default_programs(self):
        """Initialize the registry with default programs."""
        default_programs = [
            fact_check_and_calculate_program,
            simple_calculation_program,
            document_calculation_program
        ]
        
        for program in default_programs:
            self.register_program(program)
    
    def register_program(self, program: TaskProgram) -> bool:
        """
        Register a new TaskProgram in the registry.
        
        Args:
            program: The TaskProgram to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Validate program before registration
            validation_errors = program.validate_program()
            if validation_errors:
                logger.error(f"Cannot register program '{program.program_name}': {validation_errors}")
                return False
            
            # Check for name conflicts
            if program.program_name in self._programs:
                logger.warning(f"Program '{program.program_name}' already exists, overwriting")
            
            self._programs[program.program_name] = program
            logger.info(f"✅ Registered program: {program.program_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register program '{program.program_name}': {e}")
            return False
    
    def unregister_program(self, program_name: str) -> bool:
        """
        Remove a program from the registry.
        
        Args:
            program_name: Name of the program to remove
            
        Returns:
            True if removal successful, False if program not found
        """
        if program_name in self._programs:
            del self._programs[program_name]
            logger.info(f"🗑️ Unregistered program: {program_name}")
            return True
        else:
            logger.warning(f"Cannot unregister program '{program_name}': not found")
            return False
    
    def get_program(self, program_name: str) -> Optional[TaskProgram]:
        """
        Get a specific program by name.
        
        Args:
            program_name: Name of the program to retrieve
            
        Returns:
            TaskProgram if found, None otherwise
        """
        return self._programs.get(program_name)
    
    def list_programs(self) -> List[str]:
        """
        Get a list of all registered program names.
        
        Returns:
            List of program names
        """
        return list(self._programs.keys())
    
    def get_all_programs(self) -> Dict[str, TaskProgram]:
        """
        Get all registered programs.
        
        Returns:
            Dictionary mapping program names to TaskProgram objects
        """
        return self._programs.copy()
    
    def find_matching_program(self, query: str, 
                            intent: Optional[str] = None,
                            confidence: Optional[float] = None) -> Optional[TaskProgram]:
        """
        Find the best matching program for a given query.
        
        Phase 1: Simple keyword matching
        Phase 2: Will be enhanced with intent and confidence scoring
        
        Args:
            query: The user query to match
            intent: Optional intent classification from SmartQueryRouter
            confidence: Optional confidence score from SmartQueryRouter
            
        Returns:
            Best matching TaskProgram or None if no match found
        """
        query_lower = query.lower()
        
        # Phase 1: Simple keyword matching with priority ordering
        matching_programs = []
        
        for program in self._programs.values():
            if program.matches_query(query):
                # Calculate match strength based on keyword matches
                match_strength = self._calculate_match_strength(program, query_lower)
                matching_programs.append((program, match_strength))
        
        if not matching_programs:
            logger.info(f"No STEF programs matched query: '{query}'")
            return None
        
        # Sort by match strength (highest first)
        matching_programs.sort(key=lambda x: x[1], reverse=True)
        best_program = matching_programs[0][0]
        
        logger.info(f"🎯 Selected STEF program: {best_program.program_name}")
        return best_program
    
    def _calculate_match_strength(self, program: TaskProgram, query_lower: str) -> float:
        """
        Calculate how well a program matches a query.
        
        Args:
            program: The TaskProgram to evaluate
            query_lower: Lowercase version of the query
            
        Returns:
            Match strength score (higher = better match)
        """
        strength = 0.0
        
        # Count keyword matches
        for keyword in program.trigger_keywords:
            if keyword.lower() in query_lower:
                # Longer keywords get higher weight
                strength += len(keyword) * 0.1
                
                # Exact word matches get bonus
                if f" {keyword.lower()} " in f" {query_lower} ":
                    strength += 0.5
        
        # Program specificity bonus (fewer keywords = more specific)
        if len(program.trigger_keywords) < 5:
            strength += 0.2
        
        return strength
    
    def get_programs_by_intent(self, intent: str) -> List[TaskProgram]:
        """
        Get programs that are compatible with a specific intent.
        
        Args:
            intent: The intent type to match
            
        Returns:
            List of compatible programs
        """
        compatible_programs = []
        
        for program in self._programs.values():
            if intent in program.compatible_intents:
                compatible_programs.append(program)
        
        return compatible_programs
    
    def get_registry_stats(self) -> Dict[str, any]:
        """
        Get statistics about the program registry.
        
        Returns:
            Dictionary with registry statistics
        """
        total_programs = len(self._programs)
        total_keywords = sum(len(p.trigger_keywords) for p in self._programs.values())
        total_steps = sum(len(p.steps) for p in self._programs.values())
        
        program_types = {}
        for program in self._programs.values():
            for intent in program.compatible_intents:
                program_types[intent] = program_types.get(intent, 0) + 1
        
        return {
            'total_programs': total_programs,
            'total_trigger_keywords': total_keywords,
            'total_steps': total_steps,
            'average_steps_per_program': total_steps / total_programs if total_programs > 0 else 0,
            'programs_by_intent': program_types,
            'program_names': self.list_programs()
        }

# Global registry instance
_program_registry = None

def get_program_registry() -> ProgramRegistry:
    """Get or create the global program registry instance."""
    global _program_registry
    if _program_registry is None:
        _program_registry = ProgramRegistry()
    return _program_registry

# Convenience function for backward compatibility
def get_program(program_name: str) -> Optional[TaskProgram]:
    """Get a program by name from the global registry."""
    return get_program_registry().get_program(program_name)

def find_matching_program(query: str, intent: Optional[str] = None, 
                         confidence: Optional[float] = None) -> Optional[TaskProgram]:
    """Find a matching program from the global registry."""
    return get_program_registry().find_matching_program(query, intent, confidence)

# Dictionary mapping for backward compatibility with original design
PROGRAM_REGISTRY = {
    'fact_check_and_calculate': fact_check_and_calculate_program,
    'simple_calculation': simple_calculation_program,
    'document_calculation': document_calculation_program
}
