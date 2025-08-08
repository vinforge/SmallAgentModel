"""
Goal Safety Validator for SAM Autonomy
======================================

This module implements comprehensive safety validation for autonomous goals
to prevent harmful actions, infinite loops, and security violations.

Author: SAM Development Team
Version: 2.0.0
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque

from ..goals import Goal

logger = logging.getLogger(__name__)

class GoalSafetyValidator:
    """
    Comprehensive safety validator for autonomous goals.
    
    This validator implements multiple layers of protection:
    1. Deny-list checking for harmful actions
    2. Loop detection and prevention
    3. Rate limiting for goal creation
    4. Pattern analysis for suspicious behavior
    5. Resource protection for critical system files
    
    All goals must pass validation before being added to the GoalStack.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the safety validator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.GoalSafetyValidator")
        
        # Default configuration
        self.config = {
            'max_goals_per_minute': 10,
            'max_goals_per_hour': 100,
            'max_similar_goals': 3,
            'similarity_threshold': 0.8,
            'loop_detection_window': 24,  # hours
            'enable_pattern_analysis': True,
            'enable_resource_protection': True,
            'enable_rate_limiting': True
        }
        
        if config:
            self.config.update(config)
        
        # Rate limiting tracking
        self.goal_creation_times: deque = deque()
        self.hourly_goal_count = 0
        self.last_hour_reset = datetime.now()
        
        # Loop detection tracking
        self.goal_history: List[Dict] = []
        self.similar_goal_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Initialize deny-lists and protection patterns
        self._init_security_patterns()
        
        self.logger.info("GoalSafetyValidator initialized with enhanced protection")
    
    def _init_security_patterns(self) -> None:
        """Initialize security patterns and deny-lists."""
        
        # Harmful action patterns (case-insensitive)
        self.harmful_patterns = [
            r'delete.*config',
            r'remove.*security',
            r'modify.*auth',
            r'disable.*safety',
            r'bypass.*validation',
            r'override.*security',
            r'escalate.*privilege',
            r'access.*private.*key',
            r'modify.*encryption',
            r'disable.*logging',
            r'clear.*audit',
            r'tamper.*log',
            r'inject.*code',
            r'execute.*shell',
            r'run.*system.*command',
            r'access.*root',
            r'modify.*system.*file',
            r'change.*permission',
            r'alter.*database.*schema'
        ]
        
        # Protected file patterns
        self.protected_files = [
            r'.*\.key$',
            r'.*\.pem$',
            r'.*\.crt$',
            r'.*config.*\.json$',
            r'.*\.env$',
            r'.*password.*',
            r'.*secret.*',
            r'.*auth.*\.py$',
            r'.*security.*\.py$',
            r'.*encryption.*\.py$',
            r'sam/autonomy/safety/.*',
            r'config/.*',
            r'logs/autonomy_audit\.log$'
        ]
        
        # Suspicious skill combinations
        self.suspicious_combinations = [
            {'MEMOIR_EditSkill', 'security'},
            {'file_access', 'config'},
            {'system_command', 'privilege'}
        ]
        
        # Compile patterns for efficiency
        self.compiled_harmful_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.harmful_patterns
        ]
        self.compiled_protected_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.protected_files
        ]
    
    def validate_goal(self, goal: Goal) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive validation of a goal for safety.
        
        Args:
            goal: Goal to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # 1. Rate limiting check
            if self.config['enable_rate_limiting']:
                rate_valid, rate_error = self._check_rate_limits()
                if not rate_valid:
                    return False, rate_error
            
            # 2. Harmful action detection
            harmful_valid, harmful_error = self._check_harmful_actions(goal)
            if not harmful_valid:
                return False, harmful_error
            
            # 3. Resource protection check
            if self.config['enable_resource_protection']:
                resource_valid, resource_error = self._check_protected_resources(goal)
                if not resource_valid:
                    return False, resource_error
            
            # 4. Loop detection
            loop_valid, loop_error = self._check_loop_patterns(goal)
            if not loop_valid:
                return False, loop_error
            
            # 5. Pattern analysis
            if self.config['enable_pattern_analysis']:
                pattern_valid, pattern_error = self._analyze_suspicious_patterns(goal)
                if not pattern_valid:
                    return False, pattern_error
            
            # 6. Similarity check
            similarity_valid, similarity_error = self._check_goal_similarity(goal)
            if not similarity_valid:
                return False, similarity_error
            
            # All checks passed
            self._record_validated_goal(goal)
            self.logger.debug(f"Goal validation passed: {goal.goal_id}")
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error during goal validation: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _check_rate_limits(self) -> Tuple[bool, Optional[str]]:
        """Check if goal creation rate limits are exceeded."""
        now = datetime.now()
        
        # Clean old entries (older than 1 minute)
        minute_ago = now - timedelta(minutes=1)
        while self.goal_creation_times and self.goal_creation_times[0] < minute_ago:
            self.goal_creation_times.popleft()
        
        # Check per-minute limit
        if len(self.goal_creation_times) >= self.config['max_goals_per_minute']:
            return False, f"Rate limit exceeded: {self.config['max_goals_per_minute']} goals per minute"
        
        # Check hourly limit
        if now - self.last_hour_reset > timedelta(hours=1):
            self.hourly_goal_count = 0
            self.last_hour_reset = now
        
        if self.hourly_goal_count >= self.config['max_goals_per_hour']:
            return False, f"Rate limit exceeded: {self.config['max_goals_per_hour']} goals per hour"
        
        return True, None
    
    def _check_harmful_actions(self, goal: Goal) -> Tuple[bool, Optional[str]]:
        """Check for harmful action patterns in goal description and context."""
        
        # Check description against harmful patterns
        for pattern in self.compiled_harmful_patterns:
            if pattern.search(goal.description):
                return False, f"Harmful action detected: {pattern.pattern}"
        
        # Check source context for harmful content
        context_str = str(goal.source_context).lower()
        for pattern in self.compiled_harmful_patterns:
            if pattern.search(context_str):
                return False, f"Harmful content in context: {pattern.pattern}"
        
        return True, None
    
    def _check_protected_resources(self, goal: Goal) -> Tuple[bool, Optional[str]]:
        """Check if goal attempts to access protected resources."""
        
        # Check for protected file access
        combined_text = f"{goal.description} {goal.source_context}".lower()
        
        for pattern in self.compiled_protected_patterns:
            if pattern.search(combined_text):
                return False, f"Attempted access to protected resource: {pattern.pattern}"
        
        return True, None
    
    def _check_loop_patterns(self, goal: Goal) -> Tuple[bool, Optional[str]]:
        """Detect potential infinite loops in goal generation."""
        
        # Create a signature for this goal
        goal_signature = self._create_goal_signature(goal)
        
        # Check recent history for similar goals
        cutoff_time = datetime.now() - timedelta(hours=self.config['loop_detection_window'])
        recent_similar = [
            g for g in self.goal_history 
            if g['timestamp'] > cutoff_time and g['signature'] == goal_signature
        ]
        
        if len(recent_similar) >= self.config['max_similar_goals']:
            return False, f"Loop detected: {len(recent_similar)} similar goals in {self.config['loop_detection_window']} hours"
        
        return True, None
    
    def _analyze_suspicious_patterns(self, goal: Goal) -> Tuple[bool, Optional[str]]:
        """Analyze goal for suspicious patterns and combinations."""
        
        # Check for suspicious skill combinations
        goal_text = goal.description.lower()
        source_skill = goal.source_skill.lower()
        
        for suspicious_combo in self.suspicious_combinations:
            if all(term in f"{goal_text} {source_skill}" for term in suspicious_combo):
                return False, f"Suspicious pattern detected: {suspicious_combo}"
        
        # Check for rapid escalation patterns
        if 'urgent' in goal_text and 'immediate' in goal_text and 'critical' in goal_text:
            return False, "Suspicious urgency escalation pattern detected"
        
        return True, None
    
    def _check_goal_similarity(self, goal: Goal) -> Tuple[bool, Optional[str]]:
        """Check if too many similar goals have been created recently."""
        
        goal_signature = self._create_goal_signature(goal)
        now = datetime.now()
        
        # Clean old entries
        cutoff_time = now - timedelta(hours=1)
        self.similar_goal_patterns[goal_signature] = [
            timestamp for timestamp in self.similar_goal_patterns[goal_signature]
            if timestamp > cutoff_time
        ]
        
        # Check similarity count
        similar_count = len(self.similar_goal_patterns[goal_signature])
        if similar_count >= self.config['max_similar_goals']:
            return False, f"Too many similar goals: {similar_count} in the last hour"
        
        return True, None
    
    def _create_goal_signature(self, goal: Goal) -> str:
        """Create a signature for goal similarity detection."""
        # Normalize description for comparison
        normalized_desc = re.sub(r'\W+', ' ', goal.description.lower()).strip()
        words = normalized_desc.split()
        
        # Use first 5 significant words as signature
        significant_words = [w for w in words if len(w) > 3][:5]
        signature = f"{goal.source_skill}:{' '.join(significant_words)}"
        
        return signature
    
    def _record_validated_goal(self, goal: Goal) -> None:
        """Record a successfully validated goal for tracking."""
        now = datetime.now()
        
        # Update rate limiting counters
        self.goal_creation_times.append(now)
        self.hourly_goal_count += 1
        
        # Update goal history
        goal_record = {
            'goal_id': goal.goal_id,
            'signature': self._create_goal_signature(goal),
            'timestamp': now,
            'source_skill': goal.source_skill
        }
        
        self.goal_history.append(goal_record)
        
        # Update similarity tracking
        goal_signature = self._create_goal_signature(goal)
        self.similar_goal_patterns[goal_signature].append(now)
        
        # Cleanup old history (keep last 1000 entries)
        if len(self.goal_history) > 1000:
            self.goal_history = self.goal_history[-1000:]
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics for monitoring."""
        now = datetime.now()
        
        return {
            'goals_last_minute': len(self.goal_creation_times),
            'goals_last_hour': self.hourly_goal_count,
            'total_goals_validated': len(self.goal_history),
            'unique_signatures': len(self.similar_goal_patterns),
            'rate_limit_per_minute': self.config['max_goals_per_minute'],
            'rate_limit_per_hour': self.config['max_goals_per_hour'],
            'last_validation': now.isoformat()
        }
    
    def reset_counters(self) -> None:
        """Reset all counters (for testing or maintenance)."""
        self.goal_creation_times.clear()
        self.hourly_goal_count = 0
        self.last_hour_reset = datetime.now()
        self.goal_history.clear()
        self.similar_goal_patterns.clear()
        
        self.logger.info("Safety validator counters reset")
    
    def update_config(self, new_config: Dict) -> None:
        """Update validator configuration."""
        self.config.update(new_config)
        self.logger.info(f"Safety validator configuration updated: {new_config}")
