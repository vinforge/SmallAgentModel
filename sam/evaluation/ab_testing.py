#!/usr/bin/env python3
"""
SAM A/B Testing Framework - Task 30 Phase 3
===========================================

Implements A/B testing framework for comparing response generation pipelines.
Enables controlled experiments to validate conversational coherence improvements.

Part of Task 30: Advanced Conversational Coherence Engine
Author: SAM Development Team
Version: 1.0.0
"""

import logging
import json
import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    test_id: str
    test_name: str
    description: str
    control_pipeline: str  # 'single_stage' or 'two_stage'
    treatment_pipeline: str
    traffic_split: float  # 0.0 to 1.0 (percentage to treatment)
    start_date: str
    end_date: str
    success_metrics: List[str]
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTestConfig':
        return cls(**data)

@dataclass
class ABTestResult:
    """Result of a single A/B test interaction."""
    test_id: str
    user_id: str
    session_id: str
    pipeline_used: str  # 'control' or 'treatment'
    user_question: str
    response_generated: str
    response_time_ms: float
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTestResult':
        return cls(**data)

class ABTestingFramework:
    """
    A/B testing framework for response pipeline comparison.
    
    Features:
    - Controlled traffic splitting
    - Multiple concurrent tests
    - Result collection and analysis
    - Statistical significance testing
    - Performance metrics tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the A/B testing framework.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.ABTestingFramework")
        
        # Default configuration
        self.config = {
            'storage_directory': 'ab_tests',
            'max_results_per_test': 1000,
            'enable_ab_testing': True,
            'default_traffic_split': 0.5,
            'min_sample_size': 30,
            'confidence_level': 0.95
        }
        
        if config:
            self.config.update(config)
        
        # Storage setup
        self.storage_dir = Path(self.config['storage_directory'])
        self.storage_dir.mkdir(exist_ok=True)
        
        # Test management
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, List[ABTestResult]] = {}
        self._lock = threading.RLock()
        
        # Load existing tests and results
        self._load_tests()
        self._load_results()
        
        self.logger.info(f"ABTestingFramework initialized with {len(self.active_tests)} active tests")
    
    def create_test(self, test_config: ABTestConfig) -> bool:
        """
        Create a new A/B test.
        
        Args:
            test_config: Test configuration
            
        Returns:
            True if test created successfully
        """
        try:
            with self._lock:
                # Validate test configuration
                if not self._validate_test_config(test_config):
                    return False
                
                # Add to active tests
                self.active_tests[test_config.test_id] = test_config
                self.test_results[test_config.test_id] = []
                
                # Save test configuration
                self._save_test_config(test_config)
                
                self.logger.info(f"Created A/B test: {test_config.test_id} - {test_config.test_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating A/B test: {e}")
            return False
    
    def should_use_treatment(self, user_id: str, test_id: str) -> bool:
        """
        Determine if user should receive treatment pipeline.
        
        Args:
            user_id: User identifier for consistent assignment
            test_id: Test identifier
            
        Returns:
            True if user should receive treatment
        """
        try:
            test_config = self.active_tests.get(test_id)
            if not test_config or not test_config.enabled:
                return False
            
            # Check if test is currently active
            if not self._is_test_active(test_config):
                return False
            
            # Consistent user assignment using hash
            user_hash = hashlib.md5(f"{user_id}_{test_id}".encode()).hexdigest()
            hash_value = int(user_hash[:8], 16) / (16**8)
            
            return hash_value < test_config.traffic_split
            
        except Exception as e:
            self.logger.error(f"Error determining treatment assignment: {e}")
            return False
    
    def record_result(self, test_id: str, user_id: str, session_id: str,
                     pipeline_used: str, user_question: str, response_generated: str,
                     response_time_ms: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record an A/B test result.
        
        Args:
            test_id: Test identifier
            user_id: User identifier
            session_id: Session identifier
            pipeline_used: 'control' or 'treatment'
            user_question: User's question
            response_generated: Generated response
            response_time_ms: Response generation time
            metadata: Additional metadata
            
        Returns:
            True if result recorded successfully
        """
        try:
            with self._lock:
                if test_id not in self.active_tests:
                    return False
                
                # Create result record
                result = ABTestResult(
                    test_id=test_id,
                    user_id=user_id,
                    session_id=session_id,
                    pipeline_used=pipeline_used,
                    user_question=user_question,
                    response_generated=response_generated,
                    response_time_ms=response_time_ms,
                    timestamp=datetime.now().isoformat(),
                    metadata=metadata or {}
                )
                
                # Add to results
                if test_id not in self.test_results:
                    self.test_results[test_id] = []
                
                self.test_results[test_id].append(result)
                
                # Limit results per test
                max_results = self.config['max_results_per_test']
                if len(self.test_results[test_id]) > max_results:
                    self.test_results[test_id] = self.test_results[test_id][-max_results:]
                
                # Save result
                self._save_test_result(result)
                
                self.logger.debug(f"Recorded A/B test result for {test_id}: {pipeline_used}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error recording A/B test result: {e}")
            return False
    
    def get_test_summary(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics for a test.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test summary dictionary
        """
        try:
            if test_id not in self.active_tests:
                return None
            
            test_config = self.active_tests[test_id]
            results = self.test_results.get(test_id, [])
            
            # Basic statistics
            total_results = len(results)
            control_results = [r for r in results if r.pipeline_used == 'control']
            treatment_results = [r for r in results if r.pipeline_used == 'treatment']
            
            # Performance metrics
            control_avg_time = sum(r.response_time_ms for r in control_results) / len(control_results) if control_results else 0
            treatment_avg_time = sum(r.response_time_ms for r in treatment_results) / len(treatment_results) if treatment_results else 0
            
            # Response length metrics
            control_avg_length = sum(len(r.response_generated.split()) for r in control_results) / len(control_results) if control_results else 0
            treatment_avg_length = sum(len(r.response_generated.split()) for r in treatment_results) / len(treatment_results) if treatment_results else 0
            
            summary = {
                'test_config': test_config.to_dict(),
                'total_results': total_results,
                'control_count': len(control_results),
                'treatment_count': len(treatment_results),
                'control_avg_response_time_ms': control_avg_time,
                'treatment_avg_response_time_ms': treatment_avg_time,
                'control_avg_response_length': control_avg_length,
                'treatment_avg_response_length': treatment_avg_length,
                'performance_improvement': ((control_avg_time - treatment_avg_time) / control_avg_time * 100) if control_avg_time > 0 else 0,
                'is_statistically_significant': total_results >= self.config['min_sample_size'],
                'test_status': 'active' if self._is_test_active(test_config) else 'inactive'
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating test summary: {e}")
            return None
    
    def get_all_test_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all tests."""
        summaries = {}
        for test_id in self.active_tests:
            summary = self.get_test_summary(test_id)
            if summary:
                summaries[test_id] = summary
        return summaries
    
    def _validate_test_config(self, test_config: ABTestConfig) -> bool:
        """Validate test configuration."""
        if not test_config.test_id or not test_config.test_name:
            self.logger.error("Test ID and name are required")
            return False
        
        if test_config.traffic_split < 0 or test_config.traffic_split > 1:
            self.logger.error("Traffic split must be between 0 and 1")
            return False
        
        if test_config.control_pipeline == test_config.treatment_pipeline:
            self.logger.error("Control and treatment pipelines must be different")
            return False
        
        return True
    
    def _is_test_active(self, test_config: ABTestConfig) -> bool:
        """Check if test is currently active."""
        try:
            now = datetime.now()
            start_date = datetime.fromisoformat(test_config.start_date)
            end_date = datetime.fromisoformat(test_config.end_date)
            
            return start_date <= now <= end_date and test_config.enabled
        except:
            return False
    
    def _save_test_config(self, test_config: ABTestConfig) -> None:
        """Save test configuration to disk."""
        try:
            config_file = self.storage_dir / f"{test_config.test_id}_config.json"
            with open(config_file, 'w') as f:
                json.dump(test_config.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving test config: {e}")
    
    def _save_test_result(self, result: ABTestResult) -> None:
        """Save test result to disk."""
        try:
            results_file = self.storage_dir / f"{result.test_id}_results.jsonl"
            with open(results_file, 'a') as f:
                f.write(json.dumps(result.to_dict()) + '\n')
        except Exception as e:
            self.logger.error(f"Error saving test result: {e}")
    
    def _load_tests(self) -> None:
        """Load test configurations from disk."""
        try:
            for config_file in self.storage_dir.glob("*_config.json"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    test_config = ABTestConfig.from_dict(config_data)
                    self.active_tests[test_config.test_id] = test_config
                    
                except Exception as e:
                    self.logger.warning(f"Error loading test config {config_file}: {e}")
            
            self.logger.info(f"Loaded {len(self.active_tests)} test configurations")
            
        except Exception as e:
            self.logger.error(f"Error loading test configurations: {e}")
    
    def _load_results(self) -> None:
        """Load test results from disk."""
        try:
            for results_file in self.storage_dir.glob("*_results.jsonl"):
                test_id = results_file.stem.replace('_results', '')
                
                if test_id not in self.test_results:
                    self.test_results[test_id] = []
                
                try:
                    with open(results_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                result_data = json.loads(line)
                                result = ABTestResult.from_dict(result_data)
                                self.test_results[test_id].append(result)
                
                except Exception as e:
                    self.logger.warning(f"Error loading results from {results_file}: {e}")
            
            total_results = sum(len(results) for results in self.test_results.values())
            self.logger.info(f"Loaded {total_results} test results across {len(self.test_results)} tests")
            
        except Exception as e:
            self.logger.error(f"Error loading test results: {e}")

# Global A/B testing framework instance
_ab_testing_framework: Optional[ABTestingFramework] = None
_ab_testing_lock = threading.Lock()

def get_ab_testing_framework(config: Optional[Dict[str, Any]] = None) -> ABTestingFramework:
    """Get the global A/B testing framework instance."""
    global _ab_testing_framework

    with _ab_testing_lock:
        if _ab_testing_framework is None:
            _ab_testing_framework = ABTestingFramework(config)
        return _ab_testing_framework
