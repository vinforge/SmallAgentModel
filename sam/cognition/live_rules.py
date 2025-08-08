#!/usr/bin/env python3
"""
SAM Introspection Dashboard - Live Rule System
==============================================

Phase 3: Dynamic rule configuration system that allows real-time modification
of SAM's behavior without code deployment or system restart.

Author: SAM Development Team
Version: 3.0.0 (Phase 3)
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import importlib

logger = logging.getLogger(__name__)

class RuleType(Enum):
    """Types of live rules."""
    DECISION_OVERRIDE = "decision_override"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    BEHAVIOR_MODIFICATION = "behavior_modification"
    ROUTING_RULE = "routing_rule"
    VALIDATION_RULE = "validation_rule"

class RuleStatus(Enum):
    """Rule status states."""
    ACTIVE = "active"
    DISABLED = "disabled"
    TESTING = "testing"
    EXPIRED = "expired"

@dataclass
class LiveRule:
    """Dynamic rule definition."""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    target_module: str
    target_function: str
    condition: str  # Python expression
    action: str     # Python code to execute
    priority: int = 100
    status: RuleStatus = RuleStatus.ACTIVE
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    application_count: int = 0
    max_applications: Optional[int] = None
    expires_at: Optional[datetime] = None
    enabled: bool = True
    test_mode: bool = False

@dataclass
class RuleApplication:
    """Record of a rule being applied."""
    application_id: str
    rule_id: str
    applied_at: datetime
    target_module: str
    target_function: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class LiveRuleManager:
    """
    Dynamic rule management system for real-time behavior modification.
    
    Provides comprehensive rule functionality including:
    - Dynamic rule creation and modification
    - Real-time rule application
    - Safe rule testing and validation
    - Rule performance monitoring
    - Automatic rule expiration and cleanup
    - Integration with SAM's core modules
    """
    
    def __init__(self, config_path: str = "config/live_rules.json"):
        """
        Initialize the live rule manager.
        
        Args:
            config_path: Path to live rules configuration file
        """
        self.config_path = config_path
        self.rules: Dict[str, LiveRule] = {}
        self.rule_applications: Dict[str, RuleApplication] = {}
        self._lock = threading.RLock()
        
        # Configuration
        self.config = {
            'enable_live_rules': True,
            'max_rules': 100,
            'max_applications_history': 10000,
            'rule_timeout_seconds': 10,
            'enable_rule_sandbox': True,
            'auto_cleanup_expired': True,
            'enable_performance_monitoring': True,
            'safe_mode': True  # Prevents dangerous operations
        }
        
        # Performance monitoring
        self.performance_stats = {
            'total_applications': 0,
            'successful_applications': 0,
            'failed_applications': 0,
            'average_execution_time': 0.0,
            'rules_by_type': {},
            'last_cleanup': datetime.now()
        }
        
        # Load existing rules
        self._load_rules()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("LiveRuleManager initialized")
    
    def create_rule(self, name: str, description: str, rule_type: RuleType,
                   target_module: str, target_function: str, condition: str,
                   action: str, created_by: str = "system", priority: int = 100,
                   max_applications: Optional[int] = None,
                   expires_in_hours: Optional[int] = None,
                   test_mode: bool = False) -> str:
        """
        Create a new live rule.
        
        Args:
            name: Rule name
            description: Rule description
            rule_type: Type of rule
            target_module: Target module name
            target_function: Target function name
            condition: Condition expression
            action: Action code
            created_by: User who created the rule
            priority: Rule priority (higher = more important)
            max_applications: Maximum applications before disabling
            expires_in_hours: Hours until rule expires
            test_mode: Whether rule is in test mode
            
        Returns:
            Rule ID
        """
        with self._lock:
            if len(self.rules) >= self.config['max_rules']:
                raise ValueError(f"Maximum rules ({self.config['max_rules']}) reached")
            
            # Validate rule syntax
            if not self._validate_rule_syntax(condition, action):
                raise ValueError("Invalid rule syntax")
            
            # Check for safety violations
            if self.config['safe_mode'] and self._has_unsafe_operations(action):
                raise ValueError("Rule contains unsafe operations")
            
            rule_id = str(uuid.uuid4())
            
            expires_at = None
            if expires_in_hours:
                expires_at = datetime.now() + timedelta(hours=expires_in_hours)
            
            rule = LiveRule(
                rule_id=rule_id,
                name=name,
                description=description,
                rule_type=rule_type,
                target_module=target_module,
                target_function=target_function,
                condition=condition,
                action=action,
                priority=priority,
                created_by=created_by,
                max_applications=max_applications,
                expires_at=expires_at,
                test_mode=test_mode
            )
            
            self.rules[rule_id] = rule
            self._save_rules()
            
            logger.info(f"Created live rule {name} ({rule_id}) by {created_by}")
            return rule_id
    
    def apply_rules(self, module_name: str, function_name: str, 
                   input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply matching live rules to function input.
        
        Args:
            module_name: Module name
            function_name: Function name
            input_data: Function input data
            
        Returns:
            Modified input data with rule applications
        """
        if not self.config['enable_live_rules']:
            return input_data
        
        modified_data = input_data.copy()
        applied_rules = []
        
        with self._lock:
            # Get matching rules sorted by priority
            matching_rules = self._get_matching_rules(module_name, function_name)
            
            for rule in matching_rules:
                try:
                    start_time = time.time()
                    
                    # Check if rule should be applied
                    if self._should_apply_rule(rule, modified_data):
                        # Apply rule
                        result = self._apply_rule(rule, modified_data)
                        
                        if result['success']:
                            modified_data = result['output_data']
                            applied_rules.append(rule.rule_id)
                            
                            # Update rule statistics
                            rule.last_applied = datetime.now()
                            rule.application_count += 1
                            
                            # Record application
                            self._record_rule_application(rule, input_data, modified_data, True)
                            
                            # Check if rule should be disabled
                            if rule.max_applications and rule.application_count >= rule.max_applications:
                                rule.status = RuleStatus.DISABLED
                                logger.info(f"Rule {rule.name} disabled after {rule.application_count} applications")
                        else:
                            self._record_rule_application(rule, input_data, modified_data, False, result.get('error'))
                    
                    # Update performance stats
                    execution_time = time.time() - start_time
                    self._update_performance_stats(rule, execution_time, True)
                    
                except Exception as e:
                    logger.error(f"Error applying rule {rule.name}: {e}")
                    self._record_rule_application(rule, input_data, modified_data, False, str(e))
                    self._update_performance_stats(rule, 0, False)
        
        # Add metadata about applied rules
        if applied_rules:
            modified_data['_applied_rules'] = applied_rules
            logger.debug(f"Applied {len(applied_rules)} rules to {module_name}.{function_name}")
        
        return modified_data
    
    def _get_matching_rules(self, module_name: str, function_name: str) -> List[LiveRule]:
        """Get rules that match the target module and function."""
        matching_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled or rule.status != RuleStatus.ACTIVE:
                continue
            
            # Check if rule has expired
            if rule.expires_at and datetime.now() > rule.expires_at:
                rule.status = RuleStatus.EXPIRED
                continue
            
            # Check module and function match
            if self._matches_target(module_name, rule.target_module) and \
               self._matches_target(function_name, rule.target_function):
                matching_rules.append(rule)
        
        # Sort by priority (higher priority first)
        matching_rules.sort(key=lambda r: r.priority, reverse=True)
        return matching_rules
    
    def _matches_target(self, actual: str, pattern: str) -> bool:
        """Check if actual target matches pattern (supports wildcards)."""
        if pattern == "*":
            return True
        
        # Simple wildcard matching
        if "*" in pattern:
            parts = pattern.split("*")
            if len(parts) == 2:
                prefix, suffix = parts
                return actual.startswith(prefix) and actual.endswith(suffix)
        
        return actual == pattern
    
    def _should_apply_rule(self, rule: LiveRule, data: Dict[str, Any]) -> bool:
        """Check if rule condition is met."""
        try:
            # Create safe evaluation environment
            safe_dict = {
                '__builtins__': {},
                'data': data,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr
            }
            
            # Evaluate condition
            result = eval(rule.condition, safe_dict)
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Error evaluating rule condition '{rule.condition}': {e}")
            return False
    
    def _apply_rule(self, rule: LiveRule, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rule action to data."""
        try:
            # Create safe execution environment
            safe_dict = {
                '__builtins__': {},
                'data': data.copy(),
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round
            }
            
            # Execute action
            exec(rule.action, safe_dict)
            
            return {
                'success': True,
                'output_data': safe_dict['data'],
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'output_data': data,
                'error': str(e)
            }
    
    def _validate_rule_syntax(self, condition: str, action: str) -> bool:
        """Validate rule syntax."""
        try:
            # Check condition syntax
            compile(condition, '<condition>', 'eval')
            
            # Check action syntax
            compile(action, '<action>', 'exec')
            
            return True
        except:
            return False
    
    def _has_unsafe_operations(self, code: str) -> bool:
        """Check for unsafe operations in code."""
        unsafe_keywords = [
            'import', 'exec', 'eval', 'open', 'file', '__import__',
            'globals', 'locals', 'vars', 'dir', 'delattr', 'setattr',
            'subprocess', 'os.', 'sys.', 'shutil.', 'pickle.'
        ]
        
        for keyword in unsafe_keywords:
            if keyword in code:
                return True
        
        return False
    
    def _record_rule_application(self, rule: LiveRule, input_data: Dict[str, Any],
                                output_data: Dict[str, Any], success: bool,
                                error_message: Optional[str] = None):
        """Record a rule application."""
        application_id = str(uuid.uuid4())
        
        application = RuleApplication(
            application_id=application_id,
            rule_id=rule.rule_id,
            applied_at=datetime.now(),
            target_module=rule.target_module,
            target_function=rule.target_function,
            input_data=input_data,
            output_data=output_data,
            success=success,
            error_message=error_message
        )
        
        self.rule_applications[application_id] = application
        
        # Limit application history
        if len(self.rule_applications) > self.config['max_applications_history']:
            # Remove oldest applications
            oldest_apps = sorted(self.rule_applications.items(),
                               key=lambda x: x[1].applied_at)
            for app_id, _ in oldest_apps[:100]:  # Remove 100 oldest
                del self.rule_applications[app_id]
    
    def _update_performance_stats(self, rule: LiveRule, execution_time: float, success: bool):
        """Update performance statistics."""
        self.performance_stats['total_applications'] += 1
        
        if success:
            self.performance_stats['successful_applications'] += 1
        else:
            self.performance_stats['failed_applications'] += 1
        
        # Update average execution time
        total_apps = self.performance_stats['total_applications']
        current_avg = self.performance_stats['average_execution_time']
        self.performance_stats['average_execution_time'] = \
            (current_avg * (total_apps - 1) + execution_time) / total_apps
        
        # Update rule type stats
        rule_type = rule.rule_type.value
        if rule_type not in self.performance_stats['rules_by_type']:
            self.performance_stats['rules_by_type'][rule_type] = 0
        self.performance_stats['rules_by_type'][rule_type] += 1
    
    def _load_rules(self):
        """Load rules from configuration file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    
                    for rule_data in data.get('rules', []):
                        rule = LiveRule(
                            rule_id=rule_data['rule_id'],
                            name=rule_data['name'],
                            description=rule_data['description'],
                            rule_type=RuleType(rule_data['rule_type']),
                            target_module=rule_data['target_module'],
                            target_function=rule_data['target_function'],
                            condition=rule_data['condition'],
                            action=rule_data['action'],
                            priority=rule_data.get('priority', 100),
                            status=RuleStatus(rule_data.get('status', 'active')),
                            created_by=rule_data.get('created_by', 'system'),
                            created_at=datetime.fromisoformat(rule_data.get('created_at', datetime.now().isoformat())),
                            last_applied=datetime.fromisoformat(rule_data['last_applied']) if rule_data.get('last_applied') else None,
                            application_count=rule_data.get('application_count', 0),
                            max_applications=rule_data.get('max_applications'),
                            expires_at=datetime.fromisoformat(rule_data['expires_at']) if rule_data.get('expires_at') else None,
                            enabled=rule_data.get('enabled', True),
                            test_mode=rule_data.get('test_mode', False)
                        )
                        self.rules[rule.rule_id] = rule
                        
        except Exception as e:
            logger.error(f"Error loading live rules: {e}")
    
    def _save_rules(self):
        """Save rules to configuration file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            data = {
                'rules': [
                    {
                        'rule_id': rule.rule_id,
                        'name': rule.name,
                        'description': rule.description,
                        'rule_type': rule.rule_type.value,
                        'target_module': rule.target_module,
                        'target_function': rule.target_function,
                        'condition': rule.condition,
                        'action': rule.action,
                        'priority': rule.priority,
                        'status': rule.status.value,
                        'created_by': rule.created_by,
                        'created_at': rule.created_at.isoformat(),
                        'last_applied': rule.last_applied.isoformat() if rule.last_applied else None,
                        'application_count': rule.application_count,
                        'max_applications': rule.max_applications,
                        'expires_at': rule.expires_at.isoformat() if rule.expires_at else None,
                        'enabled': rule.enabled,
                        'test_mode': rule.test_mode
                    }
                    for rule in self.rules.values()
                ]
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving live rules: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks for rule management."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    self._cleanup_expired_rules()
                except Exception as e:
                    logger.error(f"Error in rule cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_rules(self):
        """Clean up expired rules."""
        if not self.config['auto_cleanup_expired']:
            return
        
        with self._lock:
            expired_ids = []
            for rule_id, rule in self.rules.items():
                if rule.expires_at and datetime.now() > rule.expires_at:
                    expired_ids.append(rule_id)
                elif rule.max_applications and rule.application_count >= rule.max_applications:
                    expired_ids.append(rule_id)
            
            for rule_id in expired_ids:
                del self.rules[rule_id]
                logger.info(f"Cleaned up expired rule {rule_id}")
            
            if expired_ids:
                self._save_rules()
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all rules."""
        with self._lock:
            return [
                {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'description': rule.description,
                    'rule_type': rule.rule_type.value,
                    'target_module': rule.target_module,
                    'target_function': rule.target_function,
                    'condition': rule.condition,
                    'action': rule.action,
                    'priority': rule.priority,
                    'status': rule.status.value,
                    'created_by': rule.created_by,
                    'created_at': rule.created_at.isoformat(),
                    'last_applied': rule.last_applied.isoformat() if rule.last_applied else None,
                    'application_count': rule.application_count,
                    'max_applications': rule.max_applications,
                    'expires_at': rule.expires_at.isoformat() if rule.expires_at else None,
                    'enabled': rule.enabled,
                    'test_mode': rule.test_mode
                }
                for rule in self.rules.values()
            ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get rule performance statistics."""
        return self.performance_stats.copy()

# Global live rule manager instance
_live_rule_manager = None

def get_live_rule_manager() -> LiveRuleManager:
    """Get the global live rule manager instance."""
    global _live_rule_manager
    if _live_rule_manager is None:
        _live_rule_manager = LiveRuleManager()
    return _live_rule_manager
