"""
Cognitive Distillation Engine
============================

Main orchestrator for discovering cognitive principles from successful interactions.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

from .collector import InteractionCollector, SuccessfulInteraction
from .registry import PrincipleRegistry, CognitivePrinciple
from .validator import PrincipleValidator
from .schema import distillation_schema
from .llm_integration import LLMIntegration, LLMResponse

logger = logging.getLogger(__name__)

class DistillationEngine:
    """Main engine for cognitive principle discovery."""
    
    def __init__(self):
        """Initialize the distillation engine."""
        self.collector = InteractionCollector()
        self.registry = PrincipleRegistry()
        self.validator = PrincipleValidator()
        self.llm_integration = LLMIntegration()
        self.schema = distillation_schema

        logger.info("Distillation engine initialized with LLM integration")
    
    def discover_principle(self, strategy_id: str, 
                          interaction_limit: int = 20) -> Optional[CognitivePrinciple]:
        """
        Discover a cognitive principle from successful interactions.
        
        Args:
            strategy_id: The strategy to analyze
            interaction_limit: Maximum interactions to analyze
            
        Returns:
            Discovered cognitive principle or None if discovery failed
        """
        run_id = str(uuid.uuid4())
        
        try:
            # Start distillation run tracking
            self._start_distillation_run(run_id, strategy_id, interaction_limit)
            
            # Step 1: Collect successful interactions
            logger.info(f"Collecting interactions for strategy {strategy_id}")
            interactions = self.collector.collect_successful_interactions(
                strategy_id, interaction_limit
            )
            
            if len(interactions) < 3:
                logger.warning(f"Insufficient interactions ({len(interactions)}) for principle discovery")
                self._complete_distillation_run(run_id, 'failed', 
                                              error_message="Insufficient interaction data")
                return None
            
            # Step 2: Analyze interactions and discover principle
            logger.info(f"Analyzing {len(interactions)} interactions")
            principle_text = self._analyze_interactions(interactions)
            
            if not principle_text:
                logger.error("Failed to discover principle from interactions")
                self._complete_distillation_run(run_id, 'failed', 
                                              error_message="Principle discovery failed")
                return None
            
            # Step 3: Validate discovered principle
            validation_result = self.validator.validate_principle(principle_text)
            
            if not validation_result['is_valid']:
                logger.warning(f"Discovered principle failed validation: {validation_result['issues']}")
                self._complete_distillation_run(run_id, 'failed', 
                                              error_message=f"Validation failed: {validation_result['issues']}")
                return None
            
            # Step 4: Create and store principle
            principle = self.registry.create_principle(
                principle_text=principle_text,
                source_strategy_id=strategy_id,
                domain_tags=self._extract_domain_tags(interactions),
                confidence_score=validation_result['confidence'],
                metadata={
                    'discovery_run_id': run_id,
                    'interactions_analyzed': len(interactions),
                    'validation_result': validation_result,
                    'discovery_timestamp': datetime.now().isoformat()
                }
            )
            
            # Complete distillation run
            self._complete_distillation_run(run_id, 'completed', 
                                          interactions_analyzed=len(interactions),
                                          principles_discovered=1,
                                          results={'principle_id': principle.principle_id})
            
            logger.info(f"Successfully discovered principle: {principle.principle_id}")
            return principle
            
        except Exception as e:
            logger.error(f"Distillation failed for strategy {strategy_id}: {e}")
            self._complete_distillation_run(run_id, 'failed', error_message=str(e))
            return None
    
    def _analyze_interactions(self, interactions: List[SuccessfulInteraction]) -> Optional[str]:
        """Analyze interactions using the Symbolic Analyst prompt and LLM."""
        try:
            # Prepare interaction data for analysis
            interaction_data = []
            for interaction in interactions:
                interaction_data.append({
                    'query': interaction.query_text,
                    'context': interaction.context_provided,
                    'response': interaction.response_text,
                    'success_metrics': interaction.success_metrics,
                    'timestamp': interaction.timestamp.isoformat(),
                    'source': interaction.source_system,
                    'metadata': interaction.metadata
                })

            # Use LLM integration for principle discovery
            logger.info(f"Analyzing {len(interaction_data)} interactions with LLM")
            llm_response = self.llm_integration.discover_principle(interaction_data)

            if llm_response and llm_response.discovered_principle:
                logger.info(f"LLM discovered principle with confidence {llm_response.confidence:.2f}")

                # Store LLM analysis metadata for future reference
                self._store_llm_analysis(interaction_data, llm_response)

                return llm_response.discovered_principle
            else:
                logger.warning("LLM failed to discover principle, falling back to heuristic method")
                # Fallback to heuristic method
                return self._heuristic_principle_discovery(interaction_data)

        except Exception as e:
            logger.error(f"Failed to analyze interactions: {e}")
            # Fallback to heuristic method on error
            return self._heuristic_principle_discovery(interaction_data)
    
    def _create_symbolic_analyst_prompt(self, interaction_data: List[Dict]) -> str:
        """Create the specialized prompt for principle discovery."""
        prompt = """You are an expert cognitive scientist and AI strategist. Your task is to discover the underlying, simple, human-readable 'Principle of Reasoning' that explains the success of the following interactions.

Analyze the relationship between the query, the provided context, and the final response. Avoid simply describing the process; find the abstract rule that governs it.

--- SUCCESSFUL INTERACTION DATA ---
{}

Instructions:
1. Look for patterns across all interactions
2. Identify what makes these responses successful
3. Extract the underlying reasoning principle
4. Express it as a simple, actionable rule

Provide your discovered principle as a single, concise statement in a JSON object under the key 'discovered_principle'.

Example format:
{{"discovered_principle": "For financial topics, prioritize source recency and cite specific data points."}}
""".format(json.dumps(interaction_data, indent=2))
        
        return prompt
    
    def _store_llm_analysis(self, interaction_data: List[Dict], llm_response: LLMResponse):
        """Store LLM analysis metadata for future reference."""
        try:
            analysis_metadata = {
                'llm_response': {
                    'discovered_principle': llm_response.discovered_principle,
                    'confidence': llm_response.confidence,
                    'reasoning': llm_response.reasoning,
                    'metadata': llm_response.metadata
                },
                'interaction_count': len(interaction_data),
                'analysis_timestamp': datetime.now().isoformat(),
                'model_used': llm_response.metadata.get('model_used', 'unknown')
            }

            # Store in database for future analysis
            with self.schema.get_connection() as conn:
                conn.execute("""
                    INSERT INTO llm_analysis_log (
                        analysis_id, interaction_count, llm_response,
                        timestamp, model_used
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    len(interaction_data),
                    json.dumps(analysis_metadata),
                    datetime.now().isoformat(),
                    llm_response.metadata.get('model_used', 'unknown')
                ))

        except Exception as e:
            logger.warning(f"Failed to store LLM analysis metadata: {e}")

    def _heuristic_principle_discovery(self, interaction_data: List[Dict]) -> str:
        """Heuristic principle discovery as fallback when LLM fails."""
        # Analyze interaction patterns to generate a basic principle
        domains = set()
        patterns = []
        
        for interaction in interaction_data:
            query = interaction['query'].lower()
            response = interaction['response'].lower()
            
            # Extract domain indicators
            if any(word in query for word in ['financial', 'money', 'investment', 'stock']):
                domains.add('financial')
            elif any(word in query for word in ['technical', 'code', 'programming', 'software']):
                domains.add('technical')
            elif any(word in query for word in ['research', 'study', 'analysis', 'data']):
                domains.add('research')
            
            # Extract response patterns
            if 'source' in response or 'according to' in response:
                patterns.append('source_citation')
            if 'recent' in response or 'latest' in response:
                patterns.append('recency_focus')
            if 'specific' in response or 'detailed' in response:
                patterns.append('specificity')
        
        # Generate principle based on patterns
        if 'financial' in domains and 'source_citation' in patterns:
            return "For financial queries, prioritize recent sources and cite specific data points"
        elif 'technical' in domains and 'specificity' in patterns:
            return "For technical queries, provide specific examples and detailed explanations"
        elif 'research' in domains and 'source_citation' in patterns:
            return "For research queries, emphasize authoritative sources and comprehensive analysis"
        else:
            return "Provide contextually relevant responses with appropriate supporting evidence"
    
    def _extract_domain_tags(self, interactions: List[SuccessfulInteraction]) -> List[str]:
        """Extract domain tags from interactions."""
        domains = set()
        
        for interaction in interactions:
            query = interaction.query_text.lower()
            
            # Simple domain detection
            if any(word in query for word in ['financial', 'money', 'investment', 'stock', 'market']):
                domains.add('financial')
            if any(word in query for word in ['technical', 'code', 'programming', 'software', 'development']):
                domains.add('technical')
            if any(word in query for word in ['research', 'study', 'analysis', 'academic', 'scientific']):
                domains.add('research')
            if any(word in query for word in ['health', 'medical', 'medicine', 'treatment']):
                domains.add('health')
            if any(word in query for word in ['legal', 'law', 'regulation', 'compliance']):
                domains.add('legal')
        
        return list(domains) if domains else ['general']
    
    def _start_distillation_run(self, run_id: str, strategy_id: str, interaction_limit: int):
        """Start tracking a distillation run."""
        try:
            with self.schema.get_connection() as conn:
                conn.execute("""
                    INSERT INTO distillation_runs (
                        run_id, strategy_id, start_timestamp, status, configuration
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    run_id,
                    strategy_id,
                    datetime.now().isoformat(),
                    'running',
                    json.dumps({'interaction_limit': interaction_limit})
                ))
                
        except Exception as e:
            logger.error(f"Failed to start distillation run tracking: {e}")
    
    def _complete_distillation_run(self, run_id: str, status: str, 
                                  interactions_analyzed: int = 0,
                                  principles_discovered: int = 0,
                                  error_message: str = None,
                                  results: Dict = None):
        """Complete distillation run tracking."""
        try:
            with self.schema.get_connection() as conn:
                conn.execute("""
                    UPDATE distillation_runs 
                    SET end_timestamp = ?, status = ?, interactions_analyzed = ?,
                        principles_discovered = ?, error_message = ?, results = ?
                    WHERE run_id = ?
                """, (
                    datetime.now().isoformat(),
                    status,
                    interactions_analyzed,
                    principles_discovered,
                    error_message,
                    json.dumps(results) if results else None,
                    run_id
                ))
                
        except Exception as e:
            logger.error(f"Failed to complete distillation run tracking: {e}")
    
    def get_distillation_stats(self) -> Dict[str, Any]:
        """Get distillation engine statistics."""
        try:
            with self.schema.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_runs,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_runs,
                        SUM(principles_discovered) as total_principles_discovered,
                        AVG(interactions_analyzed) as avg_interactions_per_run
                    FROM distillation_runs
                """)
                
                row = cursor.fetchone()
                return {
                    'total_runs': row[0] or 0,
                    'successful_runs': row[1] or 0,
                    'total_principles_discovered': row[2] or 0,
                    'avg_interactions_per_run': round(row[3] or 0.0, 1),
                    'success_rate': round((row[1] or 0) / max(row[0] or 1, 1) * 100, 1)
                }
                
        except Exception as e:
            logger.error(f"Failed to get distillation stats: {e}")
            return {}

# Global engine instance
distillation_engine = DistillationEngine()
