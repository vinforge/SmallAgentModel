"""
Automated Distillation Integration
=================================

Integrates cognitive distillation with SAM's Self-Improvement Engine for automated principle discovery.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Thread, Event
import time

from .engine import DistillationEngine
from .registry import PrincipleRegistry
from .collector import InteractionCollector

logger = logging.getLogger(__name__)

@dataclass
class AutomationConfig:
    """Configuration for automated distillation."""
    enabled: bool = True
    check_interval_minutes: int = 60
    min_interactions_for_discovery: int = 10
    min_success_rate_threshold: float = 0.8
    max_principles_per_strategy: int = 5
    auto_activate_principles: bool = True
    notification_callback: Optional[Callable] = None

@dataclass
class DistillationTrigger:
    """Represents a trigger for automated distillation."""
    trigger_id: str
    strategy_id: str
    trigger_type: str  # 'interaction_threshold', 'time_based', 'performance_drop'
    trigger_condition: Dict[str, Any]
    last_triggered: Optional[datetime] = None
    is_active: bool = True

class AutomatedDistillation:
    """Manages automated cognitive distillation integration."""
    
    def __init__(self, config: AutomationConfig = None):
        """Initialize automated distillation system."""
        self.config = config or AutomationConfig()
        self.engine = DistillationEngine()
        self.registry = PrincipleRegistry()
        self.collector = InteractionCollector()
        
        # Automation state
        self.is_running = False
        self.automation_thread = None
        self.stop_event = Event()
        self.triggers = []
        
        # Performance tracking
        self.automation_stats = {
            'total_runs': 0,
            'successful_discoveries': 0,
            'failed_discoveries': 0,
            'last_run_timestamp': None,
            'avg_discovery_time': 0.0
        }
        
        logger.info("Automated distillation system initialized")
    
    def start_automation(self):
        """Start the automated distillation process."""
        if self.is_running:
            logger.warning("Automation already running")
            return
        
        if not self.config.enabled:
            logger.info("Automation disabled in configuration")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        self.automation_thread = Thread(target=self._automation_loop, daemon=True)
        self.automation_thread.start()
        
        logger.info("Automated distillation started")
    
    def stop_automation(self):
        """Stop the automated distillation process."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.automation_thread:
            self.automation_thread.join(timeout=5.0)
        
        logger.info("Automated distillation stopped")
    
    def _automation_loop(self):
        """Main automation loop."""
        logger.info("Automation loop started")
        
        while not self.stop_event.is_set():
            try:
                # Check for distillation opportunities
                self._check_distillation_triggers()
                
                # Wait for next check
                self.stop_event.wait(timeout=self.config.check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in automation loop: {e}")
                # Continue running despite errors
                time.sleep(60)  # Wait a minute before retrying
    
    def _check_distillation_triggers(self):
        """Check all triggers and run distillation if needed."""
        logger.debug("Checking distillation triggers")
        
        # Update automation stats
        self.automation_stats['total_runs'] += 1
        self.automation_stats['last_run_timestamp'] = datetime.now()
        
        # Check each trigger
        for trigger in self.triggers:
            if not trigger.is_active:
                continue
            
            try:
                if self._should_trigger_distillation(trigger):
                    logger.info(f"Trigger activated: {trigger.trigger_id}")
                    self._run_automated_distillation(trigger)
                    trigger.last_triggered = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error processing trigger {trigger.trigger_id}: {e}")
    
    def _should_trigger_distillation(self, trigger: DistillationTrigger) -> bool:
        """Check if a trigger should activate distillation."""
        try:
            if trigger.trigger_type == 'interaction_threshold':
                return self._check_interaction_threshold_trigger(trigger)
            elif trigger.trigger_type == 'time_based':
                return self._check_time_based_trigger(trigger)
            elif trigger.trigger_type == 'performance_drop':
                return self._check_performance_drop_trigger(trigger)
            else:
                logger.warning(f"Unknown trigger type: {trigger.trigger_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking trigger {trigger.trigger_id}: {e}")
            return False
    
    def _check_interaction_threshold_trigger(self, trigger: DistillationTrigger) -> bool:
        """Check if interaction threshold trigger should activate."""
        strategy_id = trigger.strategy_id
        condition = trigger.trigger_condition
        
        min_interactions = condition.get('min_interactions', self.config.min_interactions_for_discovery)
        min_success_rate = condition.get('min_success_rate', self.config.min_success_rate_threshold)
        
        # Get interaction summary
        summary = self.collector.get_interaction_summary(strategy_id)
        
        if summary.get('total_interactions', 0) < min_interactions:
            return False
        
        if summary.get('avg_quality_score', 0) < min_success_rate:
            return False
        
        # Check if we already have enough principles for this strategy
        existing_principles = self.registry.search_principles_by_domain(strategy_id)
        if len(existing_principles) >= self.config.max_principles_per_strategy:
            return False
        
        # Check cooldown period
        cooldown_hours = condition.get('cooldown_hours', 24)
        if trigger.last_triggered:
            time_since_last = datetime.now() - trigger.last_triggered
            if time_since_last < timedelta(hours=cooldown_hours):
                return False
        
        return True
    
    def _check_time_based_trigger(self, trigger: DistillationTrigger) -> bool:
        """Check if time-based trigger should activate."""
        condition = trigger.trigger_condition
        interval_hours = condition.get('interval_hours', 168)  # Default: weekly
        
        if not trigger.last_triggered:
            return True  # First run
        
        time_since_last = datetime.now() - trigger.last_triggered
        return time_since_last >= timedelta(hours=interval_hours)
    
    def _check_performance_drop_trigger(self, trigger: DistillationTrigger) -> bool:
        """Check if performance drop trigger should activate."""
        # This would integrate with SAM's performance monitoring
        # For now, return False as we don't have performance data
        logger.debug("Performance drop trigger not implemented yet")
        return False
    
    def _run_automated_distillation(self, trigger: DistillationTrigger):
        """Run automated distillation for a triggered strategy."""
        start_time = datetime.now()
        strategy_id = trigger.strategy_id
        
        try:
            logger.info(f"Running automated distillation for strategy: {strategy_id}")
            
            # Discover principle
            principle = self.engine.discover_principle(
                strategy_id, 
                interaction_limit=trigger.trigger_condition.get('interaction_limit', 20)
            )
            
            if principle:
                logger.info(f"Successfully discovered principle: {principle.principle_id}")
                
                # Auto-activate if configured
                if self.config.auto_activate_principles:
                    self.registry.update_principle_performance(
                        principle.principle_id, "auto_activated", 0.0
                    )
                
                # Send notification if callback provided
                if self.config.notification_callback:
                    try:
                        self.config.notification_callback({
                            'type': 'principle_discovered',
                            'strategy_id': strategy_id,
                            'principle': {
                                'id': principle.principle_id,
                                'text': principle.principle_text,
                                'confidence': principle.confidence_score
                            },
                            'trigger_id': trigger.trigger_id
                        })
                    except Exception as e:
                        logger.warning(f"Notification callback failed: {e}")
                
                self.automation_stats['successful_discoveries'] += 1
                
            else:
                logger.warning(f"Failed to discover principle for strategy: {strategy_id}")
                self.automation_stats['failed_discoveries'] += 1
            
            # Update timing stats
            duration = (datetime.now() - start_time).total_seconds()
            current_avg = self.automation_stats['avg_discovery_time']
            total_discoveries = (self.automation_stats['successful_discoveries'] + 
                               self.automation_stats['failed_discoveries'])
            
            self.automation_stats['avg_discovery_time'] = (
                (current_avg * (total_discoveries - 1) + duration) / total_discoveries
            )
            
        except Exception as e:
            logger.error(f"Automated distillation failed for strategy {strategy_id}: {e}")
            self.automation_stats['failed_discoveries'] += 1
    
    def add_trigger(self, strategy_id: str, trigger_type: str, 
                   trigger_condition: Dict[str, Any]) -> str:
        """Add a new distillation trigger."""
        import uuid
        
        trigger = DistillationTrigger(
            trigger_id=str(uuid.uuid4()),
            strategy_id=strategy_id,
            trigger_type=trigger_type,
            trigger_condition=trigger_condition
        )
        
        self.triggers.append(trigger)
        logger.info(f"Added trigger {trigger.trigger_id} for strategy {strategy_id}")
        
        return trigger.trigger_id
    
    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a distillation trigger."""
        for i, trigger in enumerate(self.triggers):
            if trigger.trigger_id == trigger_id:
                del self.triggers[i]
                logger.info(f"Removed trigger {trigger_id}")
                return True
        
        logger.warning(f"Trigger {trigger_id} not found")
        return False
    
    def get_trigger_status(self) -> List[Dict[str, Any]]:
        """Get status of all triggers."""
        status_list = []
        
        for trigger in self.triggers:
            status = {
                'trigger_id': trigger.trigger_id,
                'strategy_id': trigger.strategy_id,
                'trigger_type': trigger.trigger_type,
                'is_active': trigger.is_active,
                'last_triggered': trigger.last_triggered.isoformat() if trigger.last_triggered else None,
                'condition': trigger.trigger_condition
            }
            status_list.append(status)
        
        return status_list
    
    def setup_default_triggers(self):
        """Setup default triggers for common strategies."""
        default_triggers = [
            {
                'strategy_id': 'financial_analysis',
                'trigger_type': 'interaction_threshold',
                'condition': {
                    'min_interactions': 15,
                    'min_success_rate': 0.8,
                    'cooldown_hours': 48
                }
            },
            {
                'strategy_id': 'technical_support',
                'trigger_type': 'interaction_threshold',
                'condition': {
                    'min_interactions': 20,
                    'min_success_rate': 0.75,
                    'cooldown_hours': 72
                }
            },
            {
                'strategy_id': 'research_queries',
                'trigger_type': 'time_based',
                'condition': {
                    'interval_hours': 168  # Weekly
                }
            }
        ]
        
        for trigger_config in default_triggers:
            self.add_trigger(
                trigger_config['strategy_id'],
                trigger_config['trigger_type'],
                trigger_config['condition']
            )
        
        logger.info(f"Setup {len(default_triggers)} default triggers")
    
    def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation statistics."""
        stats = self.automation_stats.copy()
        
        # Add current status
        stats.update({
            'is_running': self.is_running,
            'active_triggers': len([t for t in self.triggers if t.is_active]),
            'total_triggers': len(self.triggers),
            'config': {
                'enabled': self.config.enabled,
                'check_interval_minutes': self.config.check_interval_minutes,
                'min_interactions_for_discovery': self.config.min_interactions_for_discovery,
                'auto_activate_principles': self.config.auto_activate_principles
            }
        })
        
        # Format timestamp
        if stats['last_run_timestamp']:
            stats['last_run_timestamp'] = stats['last_run_timestamp'].isoformat()
        
        return stats
    
    def manual_trigger(self, strategy_id: str, interaction_limit: int = 20) -> Optional[str]:
        """Manually trigger distillation for a strategy."""
        try:
            logger.info(f"Manual distillation trigger for strategy: {strategy_id}")
            
            principle = self.engine.discover_principle(strategy_id, interaction_limit)
            
            if principle:
                logger.info(f"Manual distillation successful: {principle.principle_id}")
                return principle.principle_id
            else:
                logger.warning(f"Manual distillation failed for strategy: {strategy_id}")
                return None
                
        except Exception as e:
            logger.error(f"Manual distillation error: {e}")
            return None

# Global automated distillation instance
automated_distillation = AutomatedDistillation()
