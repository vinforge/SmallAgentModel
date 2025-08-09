"""
User Journey Validator

Validates the complete feedback-to-personalization loop for individual users
in the SAM Personalized Tuner system.

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationStep:
    """Represents a step in the user journey validation."""
    step_name: str
    description: str
    required: bool = True
    completed: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UserJourneyResult:
    """Result of a complete user journey validation."""
    user_id: str
    journey_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    steps: List[ValidationStep] = None
    overall_success: bool = False
    success_rate: float = 0.0
    total_time_seconds: float = 0.0
    personalization_improvement: Optional[float] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []


class UserJourneyValidator:
    """
    Validates the complete user journey from feedback to personalization.
    
    This class orchestrates a comprehensive validation of the entire
    personalization pipeline for a specific user.
    """
    
    def __init__(self):
        """Initialize the user journey validator."""
        self.logger = logging.getLogger(f"{__name__}.UserJourneyValidator")
        
        # Component references
        self.dpo_manager = None
        self.feedback_handler = None
        self.training_manager = None
        self.model_manager = None
        self.sam_client = None
        self.performance_monitor = None
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("User Journey Validator initialized")
    
    def _initialize_components(self):
        """Initialize all required components."""
        try:
            # DPO Data Manager
            from sam.learning.dpo_data_manager import get_dpo_data_manager
            from memory.episodic_store import create_episodic_store
            
            store = create_episodic_store()
            self.dpo_manager = get_dpo_data_manager(store)
            
            # Feedback Handler
            from sam.learning.feedback_handler import get_feedback_handler
            self.feedback_handler = get_feedback_handler()
            
            # DPO Components
            try:
                from sam.cognition.dpo import (
                    get_dpo_training_manager,
                    get_dpo_model_manager,
                    get_personalized_sam_client
                )
                
                self.training_manager = get_dpo_training_manager()
                self.model_manager = get_dpo_model_manager()
                self.sam_client = get_personalized_sam_client()
                
            except ImportError as e:
                self.logger.warning(f"Some DPO components not available: {e}")
            
            # Performance Monitor
            try:
                from .performance_monitor import get_dpo_performance_monitor
                self.performance_monitor = get_dpo_performance_monitor()
            except ImportError:
                self.logger.warning("Performance monitor not available")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
    
    def validate_user_journey(self, user_id: str, test_prompts: Optional[List[str]] = None) -> UserJourneyResult:
        """
        Validate the complete user journey for a specific user.
        
        Args:
            user_id: User identifier
            test_prompts: Optional list of test prompts
            
        Returns:
            UserJourneyResult with validation details
        """
        journey_id = f"journey_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        result = UserJourneyResult(
            user_id=user_id,
            journey_id=journey_id,
            start_time=start_time
        )
        
        self.logger.info(f"Starting user journey validation for {user_id}")
        
        # Define validation steps
        validation_steps = [
            ("data_collection", "Validate preference data collection", True),
            ("data_quality", "Assess data quality and readiness", True),
            ("training_eligibility", "Check training eligibility", True),
            ("model_availability", "Verify personalized model availability", False),
            ("model_activation", "Test model activation", False),
            ("personalized_inference", "Test personalized response generation", False),
            ("improvement_measurement", "Measure personalization improvement", False)
        ]
        
        # Execute validation steps
        for step_name, description, required in validation_steps:
            step = ValidationStep(
                step_name=step_name,
                description=description,
                required=required
            )
            
            try:
                self.logger.debug(f"Executing step: {step_name}")
                success = self._execute_validation_step(step, user_id, test_prompts)
                step.completed = success
                
                if not success and required:
                    self.logger.warning(f"Required step failed: {step_name}")
                
            except Exception as e:
                step.error = str(e)
                step.completed = False
                self.logger.error(f"Step {step_name} failed with error: {e}")
            
            result.steps.append(step)
        
        # Calculate results
        result.end_time = datetime.now()
        result.total_time_seconds = (result.end_time - result.start_time).total_seconds()
        
        completed_steps = sum(1 for step in result.steps if step.completed)
        required_steps = sum(1 for step in result.steps if step.required)
        completed_required = sum(1 for step in result.steps if step.required and step.completed)
        
        result.success_rate = completed_steps / len(result.steps)
        result.overall_success = completed_required == required_steps
        
        # Extract personalization improvement if available
        improvement_step = next((s for s in result.steps if s.step_name == "improvement_measurement"), None)
        if improvement_step and improvement_step.completed:
            result.personalization_improvement = improvement_step.metadata.get('improvement_score')
        
        self.logger.info(f"User journey validation completed: {completed_steps}/{len(result.steps)} steps successful")
        
        return result
    
    def _execute_validation_step(self, step: ValidationStep, user_id: str, 
                               test_prompts: Optional[List[str]]) -> bool:
        """Execute a specific validation step."""
        
        if step.step_name == "data_collection":
            return self._validate_data_collection(step, user_id)
        
        elif step.step_name == "data_quality":
            return self._validate_data_quality(step, user_id)
        
        elif step.step_name == "training_eligibility":
            return self._validate_training_eligibility(step, user_id)
        
        elif step.step_name == "model_availability":
            return self._validate_model_availability(step, user_id)
        
        elif step.step_name == "model_activation":
            return self._validate_model_activation(step, user_id)
        
        elif step.step_name == "personalized_inference":
            return self._validate_personalized_inference(step, user_id, test_prompts)
        
        elif step.step_name == "improvement_measurement":
            return self._validate_improvement_measurement(step, user_id, test_prompts)
        
        else:
            step.error = f"Unknown validation step: {step.step_name}"
            return False
    
    def _validate_data_collection(self, step: ValidationStep, user_id: str) -> bool:
        """Validate that preference data has been collected."""
        try:
            if not self.dpo_manager:
                step.error = "DPO manager not available"
                return False
            
            stats = self.dpo_manager.get_user_stats(user_id)
            total_pairs = stats.get('total_pairs', 0)
            
            step.metadata = {
                'total_pairs': total_pairs,
                'stats': stats
            }
            
            if total_pairs > 0:
                self.logger.debug(f"Found {total_pairs} preference pairs for user {user_id}")
                return True
            else:
                step.error = "No preference data found"
                return False
                
        except Exception as e:
            step.error = f"Error validating data collection: {e}"
            return False
    
    def _validate_data_quality(self, step: ValidationStep, user_id: str) -> bool:
        """Validate the quality of collected preference data."""
        try:
            if not self.dpo_manager:
                step.error = "DPO manager not available"
                return False
            
            stats = self.dpo_manager.get_user_stats(user_id)
            training_ready = stats.get('training_ready_pairs', 0)
            avg_confidence = stats.get('avg_confidence', 0.0)
            avg_quality = stats.get('avg_quality_score', 0.0)
            
            step.metadata = {
                'training_ready_pairs': training_ready,
                'average_confidence': avg_confidence,
                'average_quality': avg_quality
            }
            
            # Quality thresholds
            min_pairs = 5
            min_confidence = 0.7
            min_quality = 0.6
            
            if training_ready >= min_pairs and avg_confidence >= min_confidence and avg_quality >= min_quality:
                return True
            else:
                step.error = f"Data quality insufficient: {training_ready} pairs, {avg_confidence:.2f} confidence, {avg_quality:.2f} quality"
                return False
                
        except Exception as e:
            step.error = f"Error validating data quality: {e}"
            return False
    
    def _validate_training_eligibility(self, step: ValidationStep, user_id: str) -> bool:
        """Validate that the user is eligible for training."""
        try:
            if not self.training_manager:
                step.error = "Training manager not available"
                return False
            
            # Check if user can create a training job
            try:
                # This is a dry run check - we don't actually create the job
                stats = self.dpo_manager.get_user_stats(user_id) if self.dpo_manager else {}
                training_ready = stats.get('training_ready_pairs', 0)
                
                step.metadata = {
                    'training_ready_pairs': training_ready,
                    'eligible': training_ready >= 5
                }
                
                return training_ready >= 5
                
            except Exception as e:
                step.error = f"Training eligibility check failed: {e}"
                return False
                
        except Exception as e:
            step.error = f"Error validating training eligibility: {e}"
            return False
    
    def _validate_model_availability(self, step: ValidationStep, user_id: str) -> bool:
        """Validate that a personalized model is available."""
        try:
            if not self.model_manager:
                step.error = "Model manager not available"
                return False
            
            user_models = self.model_manager.get_user_models(user_id)
            validated_models = [m for m in user_models if m.is_validated]
            
            step.metadata = {
                'total_models': len(user_models),
                'validated_models': len(validated_models),
                'model_ids': [m.model_id for m in user_models]
            }
            
            if validated_models:
                return True
            else:
                step.error = f"No validated models available ({len(user_models)} total models)"
                return False
                
        except Exception as e:
            step.error = f"Error validating model availability: {e}"
            return False
    
    def _validate_model_activation(self, step: ValidationStep, user_id: str) -> bool:
        """Validate that a model can be activated."""
        try:
            if not self.sam_client or not self.model_manager:
                step.error = "SAM client or model manager not available"
                return False
            
            # Get available models
            user_models = self.model_manager.get_user_models(user_id)
            validated_models = [m for m in user_models if m.is_validated]
            
            if not validated_models:
                step.error = "No validated models to activate"
                return False
            
            # Try to activate the first validated model
            model = validated_models[0]
            success = self.sam_client.activate_personalized_model(user_id, model.model_id)
            
            if success:
                # Verify activation
                status = self.sam_client.get_personalization_status(user_id)
                activated = status.get('has_active_model', False)
                
                step.metadata = {
                    'activated_model': model.model_id,
                    'activation_verified': activated
                }
                
                return activated
            else:
                step.error = "Model activation failed"
                return False
                
        except Exception as e:
            step.error = f"Error validating model activation: {e}"
            return False
    
    def _validate_personalized_inference(self, step: ValidationStep, user_id: str, 
                                       test_prompts: Optional[List[str]]) -> bool:
        """Validate that personalized inference works."""
        try:
            if not self.sam_client:
                step.error = "SAM client not available"
                return False
            
            # Use default test prompts if none provided
            if not test_prompts:
                test_prompts = [
                    "What is the capital of France?",
                    "How do you make coffee?",
                    "Explain machine learning."
                ]
            
            results = []
            personalized_count = 0
            
            for prompt in test_prompts[:3]:  # Limit to 3 tests
                try:
                    from sam.cognition.dpo import generate_personalized_response
                    
                    response = generate_personalized_response(
                        prompt=prompt,
                        user_id=user_id
                    )
                    
                    if response.is_personalized:
                        personalized_count += 1
                    
                    results.append({
                        'prompt': prompt,
                        'response': response.content[:100] + "..." if len(response.content) > 100 else response.content,
                        'is_personalized': response.is_personalized,
                        'model_id': response.model_id,
                        'inference_time': response.inference_time
                    })
                    
                except Exception as e:
                    results.append({
                        'prompt': prompt,
                        'error': str(e)
                    })
            
            step.metadata = {
                'test_results': results,
                'personalized_responses': personalized_count,
                'total_tests': len(test_prompts[:3])
            }
            
            # Success if at least one response was personalized
            return personalized_count > 0
            
        except Exception as e:
            step.error = f"Error validating personalized inference: {e}"
            return False
    
    def _validate_improvement_measurement(self, step: ValidationStep, user_id: str, 
                                        test_prompts: Optional[List[str]]) -> bool:
        """Validate that personalization shows improvement."""
        try:
            if not self.sam_client:
                step.error = "SAM client not available"
                return False
            
            # This is a simplified improvement measurement
            # In a real implementation, you might use more sophisticated metrics
            
            # Get recent user feedback if available
            if self.performance_monitor:
                user_metrics = self.performance_monitor.get_user_metrics(user_id, days=7)
                feedback_summary = user_metrics.get('feedback_summary', {})
                
                avg_rating = feedback_summary.get('average_rating', 0.0)
                improvements = feedback_summary.get('improvements_detected', 0)
                total_responses = feedback_summary.get('total_responses', 0)
                
                improvement_rate = improvements / max(1, total_responses)
                
                step.metadata = {
                    'average_rating': avg_rating,
                    'improvement_rate': improvement_rate,
                    'improvements_detected': improvements,
                    'total_responses': total_responses,
                    'improvement_score': avg_rating * 0.5 + improvement_rate * 0.5
                }
                
                # Success if average rating > 3.0 or improvement rate > 0.2
                return avg_rating > 3.0 or improvement_rate > 0.2
            else:
                step.error = "Performance monitor not available for improvement measurement"
                return False
                
        except Exception as e:
            step.error = f"Error validating improvement measurement: {e}"
            return False
    
    def generate_journey_report(self, result: UserJourneyResult) -> str:
        """Generate a human-readable report of the user journey validation."""
        report = []
        report.append(f"User Journey Validation Report")
        report.append(f"=" * 40)
        report.append(f"User ID: {result.user_id}")
        report.append(f"Journey ID: {result.journey_id}")
        report.append(f"Start Time: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Duration: {result.total_time_seconds:.1f} seconds")
        report.append(f"Overall Success: {'✅ PASS' if result.overall_success else '❌ FAIL'}")
        report.append(f"Success Rate: {result.success_rate*100:.1f}%")
        
        if result.personalization_improvement is not None:
            report.append(f"Personalization Improvement: {result.personalization_improvement:.2f}")
        
        report.append("")
        report.append("Step Details:")
        report.append("-" * 20)
        
        for step in result.steps:
            status = "✅" if step.completed else "❌"
            required = "(Required)" if step.required else "(Optional)"
            report.append(f"{status} {step.step_name} {required}")
            report.append(f"   {step.description}")
            
            if step.error:
                report.append(f"   Error: {step.error}")
            
            if step.metadata:
                for key, value in step.metadata.items():
                    if key not in ['test_results']:  # Skip verbose data
                        report.append(f"   {key}: {value}")
            
            report.append("")
        
        return "\n".join(report)


# Global validator instance
_user_journey_validator = None

def get_user_journey_validator() -> UserJourneyValidator:
    """Get or create a global user journey validator instance."""
    global _user_journey_validator
    
    if _user_journey_validator is None:
        _user_journey_validator = UserJourneyValidator()
    
    return _user_journey_validator


def validate_user_personalization_journey(user_id: str, test_prompts: Optional[List[str]] = None) -> UserJourneyResult:
    """
    Convenience function to validate a user's personalization journey.
    
    Args:
        user_id: User identifier
        test_prompts: Optional list of test prompts
        
    Returns:
        UserJourneyResult with validation details
    """
    validator = get_user_journey_validator()
    return validator.validate_user_journey(user_id, test_prompts)
