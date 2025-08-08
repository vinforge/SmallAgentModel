"""
SAM Integration Interface
========================

Main interface for integrating cognitive distillation with SAM's core systems.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

from .prompt_augmentation import PromptAugmentation, AugmentedPrompt
from .thought_transparency import ThoughtTransparency, ThoughtTransparencyData
from .automation import AutomatedDistillation, AutomationConfig
from .registry import PrincipleRegistry

logger = logging.getLogger(__name__)

class SAMCognitiveDistillation:
    """Main interface for SAM's cognitive distillation capabilities."""
    
    def __init__(self, enable_automation: bool = True):
        """Initialize SAM cognitive distillation integration."""
        self.prompt_augmentation = PromptAugmentation()
        self.thought_transparency = ThoughtTransparency()
        self.registry = PrincipleRegistry()
        
        # Initialize automation if enabled
        self.automation = None
        if enable_automation:
            self.automation = AutomatedDistillation()
            self.automation.setup_default_triggers()
            self.automation.start_automation()
        
        logger.info("SAM cognitive distillation integration initialized")
    
    def enhance_reasoning(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance reasoning with cognitive principles.
        
        Args:
            prompt: Original reasoning prompt
            context: Additional context for principle selection
            
        Returns:
            Tuple of (enhanced_prompt, transparency_data)
        """
        try:
            # Augment prompt with principles
            augmented_prompt = self.prompt_augmentation.augment_prompt(prompt, context)
            
            # Create transparency data
            transparency_data = self.thought_transparency.create_transparency_data(augmented_prompt)
            
            # Convert transparency data to dict for JSON serialization
            transparency_dict = {
                'active_principles': transparency_data.active_principles,
                'meta_cognition': transparency_data.meta_cognition,
                'principle_impact': transparency_data.principle_impact,
                'reasoning_trace': None  # Will be updated after response
            }
            
            logger.info(f"Enhanced reasoning with {len(augmented_prompt.applied_principles)} principles")
            
            return augmented_prompt.augmented_prompt, transparency_dict
            
        except Exception as e:
            logger.error(f"Failed to enhance reasoning: {e}")
            return prompt, {'error': str(e)}
    
    def complete_reasoning_trace(self, transparency_data: Dict[str, Any], 
                               response: str, reasoning_steps: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete the reasoning trace after response generation.
        
        Args:
            transparency_data: Transparency data from enhance_reasoning
            response: The generated response
            reasoning_steps: Detailed reasoning steps (optional)
            
        Returns:
            Updated transparency data with complete reasoning trace
        """
        try:
            # Reconstruct augmented prompt from transparency data
            # This is a simplified reconstruction - in practice, you'd store the full object
            applied_principles = []
            for principle_data in transparency_data.get('active_principles', []):
                # Create a minimal principle object for the trace
                principle = type('CognitivePrinciple', (), {
                    'principle_id': principle_data['id'],
                    'principle_text': principle_data['text'],
                    'confidence_score': principle_data['confidence'],
                    'domain_tags': principle_data['domains'],
                    'usage_count': principle_data['usage_count'],
                    'success_rate': principle_data['success_rate'],
                    'source_strategy_id': principle_data['source_strategy'],
                    'date_discovered': principle_data['discovered_date']
                })()
                applied_principles.append(principle)
            
            # Create a minimal augmented prompt for the trace
            augmented_prompt = type('AugmentedPrompt', (), {
                'applied_principles': applied_principles,
                'confidence_boost': transparency_data.get('principle_impact', {}).get('total_impact', 0.0),
                'augmentation_metadata': transparency_data.get('meta_cognition', {}),
                'original_prompt': 'Original prompt not stored'  # In practice, store this
            })()
            
            # Update transparency data with complete trace
            updated_transparency = self.thought_transparency.create_transparency_data(
                augmented_prompt, response, reasoning_steps
            )
            
            # Convert to dict
            result = {
                'active_principles': updated_transparency.active_principles,
                'meta_cognition': updated_transparency.meta_cognition,
                'principle_impact': updated_transparency.principle_impact,
                'reasoning_trace': asdict(updated_transparency.reasoning_trace) if updated_transparency.reasoning_trace else None
            }
            
            # Convert datetime to string for JSON serialization
            if result['reasoning_trace']:
                result['reasoning_trace']['timestamp'] = updated_transparency.reasoning_trace.timestamp.isoformat()
                result['reasoning_trace']['applied_principles'] = [
                    {
                        'id': p.principle_id,
                        'text': p.principle_text,
                        'confidence': p.confidence_score
                    }
                    for p in updated_transparency.reasoning_trace.applied_principles
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to complete reasoning trace: {e}")
            transparency_data['trace_error'] = str(e)
            return transparency_data
    
    def get_active_principles(self, domain: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get active cognitive principles for display."""
        try:
            if domain:
                principles = self.registry.search_principles_by_domain(domain)
            else:
                principles = self.registry.get_active_principles(limit=limit)
            
            # Format for display
            formatted_principles = []
            for principle in principles:
                formatted_principle = {
                    'id': principle.principle_id,
                    'text': principle.principle_text,
                    'confidence': round(principle.confidence_score, 2),
                    'domains': principle.domain_tags,
                    'usage_count': principle.usage_count,
                    'success_rate': round(principle.success_rate, 2),
                    'source_strategy': principle.source_strategy_id,
                    'discovered_date': principle.date_discovered.strftime('%Y-%m-%d'),
                    'validation_status': principle.validation_status
                }
                formatted_principles.append(formatted_principle)
            
            return formatted_principles
            
        except Exception as e:
            logger.error(f"Failed to get active principles: {e}")
            return []
    
    def update_principle_feedback(self, principle_id: str, outcome: str, 
                                feedback: str = None) -> bool:
        """Update principle performance based on user feedback."""
        try:
            # Map outcome to confidence delta
            confidence_deltas = {
                'success': 0.05,
                'partial_success': 0.02,
                'neutral': 0.0,
                'failure': -0.03,
                'harmful': -0.10
            }
            
            confidence_delta = confidence_deltas.get(outcome, 0.0)
            
            success = self.registry.update_principle_performance(
                principle_id, outcome, confidence_delta
            )
            
            if success:
                logger.info(f"Updated principle {principle_id} with outcome: {outcome}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update principle feedback: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and statistics."""
        try:
            # Get registry stats
            registry_stats = self.registry.get_principle_stats()
            
            # Get augmentation stats
            augmentation_stats = self.prompt_augmentation.get_augmentation_stats()
            
            # Get transparency stats
            transparency_stats = self.thought_transparency.get_transparency_stats()
            
            # Get automation stats if available
            automation_stats = {}
            if self.automation:
                automation_stats = self.automation.get_automation_stats()
            
            return {
                'registry': registry_stats,
                'augmentation': augmentation_stats,
                'transparency': transparency_stats,
                'automation': automation_stats,
                'system_health': self._assess_system_health(registry_stats, automation_stats)
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def _assess_system_health(self, registry_stats: Dict[str, Any], 
                            automation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system health."""
        health = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check principle count
        active_principles = registry_stats.get('active_principles', 0)
        if active_principles < 5:
            health['issues'].append('Low number of active principles')
            health['recommendations'].append('Run principle discovery on more strategies')
        
        # Check principle quality
        avg_confidence = registry_stats.get('avg_confidence', 0)
        if avg_confidence < 0.5:
            health['issues'].append('Low average principle confidence')
            health['recommendations'].append('Review and improve principle validation criteria')
        
        # Check automation
        if automation_stats.get('is_running', False):
            success_rate = 0
            total_discoveries = (automation_stats.get('successful_discoveries', 0) + 
                               automation_stats.get('failed_discoveries', 0))
            if total_discoveries > 0:
                success_rate = automation_stats.get('successful_discoveries', 0) / total_discoveries
            
            if success_rate < 0.7:
                health['issues'].append('Low automation success rate')
                health['recommendations'].append('Review automation triggers and data quality')
        else:
            health['issues'].append('Automation not running')
            health['recommendations'].append('Enable automation for continuous improvement')
        
        # Set overall status
        if len(health['issues']) > 2:
            health['status'] = 'degraded'
        elif len(health['issues']) > 0:
            health['status'] = 'warning'
        
        return health
    
    def manual_principle_discovery(self, strategy_id: str, interaction_limit: int = 20) -> Optional[Dict[str, Any]]:
        """Manually trigger principle discovery for a strategy."""
        try:
            if self.automation:
                principle_id = self.automation.manual_trigger(strategy_id, interaction_limit)
                
                if principle_id:
                    principle = self.registry.get_principle(principle_id)
                    if principle:
                        return {
                            'success': True,
                            'principle': {
                                'id': principle.principle_id,
                                'text': principle.principle_text,
                                'confidence': principle.confidence_score,
                                'domains': principle.domain_tags
                            }
                        }
            
            return {'success': False, 'error': 'Discovery failed'}
            
        except Exception as e:
            logger.error(f"Manual principle discovery failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_recent_reasoning_traces(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent reasoning traces for analysis."""
        try:
            return self.thought_transparency.get_recent_traces(limit)
        except Exception as e:
            logger.error(f"Failed to get recent traces: {e}")
            return []
    
    def shutdown(self):
        """Shutdown the cognitive distillation system."""
        try:
            if self.automation:
                self.automation.stop_automation()
            
            logger.info("SAM cognitive distillation system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Global SAM integration instance
sam_cognitive_distillation = SAMCognitiveDistillation()
