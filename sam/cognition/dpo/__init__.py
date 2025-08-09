"""
SAM DPO (Direct Preference Optimization) Module

This module provides Direct Preference Optimization capabilities for 
personalizing SAM models based on user feedback and preferences.

Components:
- Configuration management for DPO training
- Training pipeline and orchestration
- Model management and LoRA adapters
- Integration with SAM's feedback system

Author: SAM Development Team
Version: 1.0.0
"""

from .dpo_config import (
    DPOConfig,
    DPOConfigManager,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    DataConfig,
    OutputConfig,
    get_dpo_config_manager
)

from .training_manager import (
    DPOTrainingManager,
    TrainingJob,
    TrainingStatus,
    get_dpo_training_manager
)

from .model_manager import (
    DPOModelManager,
    PersonalizedModel,
    get_dpo_model_manager
)

from .inference_engine import (
    PersonalizedInferenceEngine,
    get_personalized_inference_engine
)

from .sam_integration import (
    PersonalizedSAMClient,
    PersonalizedResponse,
    get_personalized_sam_client,
    generate_personalized_response,
    activate_user_model,
    deactivate_user_model,
    get_user_personalization_status
)

from .performance_monitor import (
    DPOPerformanceMonitor,
    get_dpo_performance_monitor
)

from .user_journey_validator import (
    UserJourneyValidator,
    UserJourneyResult,
    get_user_journey_validator,
    validate_user_personalization_journey
)

__all__ = [
    # Configuration classes
    'DPOConfig',
    'DPOConfigManager',
    'ModelConfig',
    'LoRAConfig',
    'TrainingConfig',
    'DataConfig',
    'OutputConfig',

    # Training management
    'DPOTrainingManager',
    'TrainingJob',
    'TrainingStatus',

    # Model management
    'DPOModelManager',
    'PersonalizedModel',

    # Inference engine
    'PersonalizedInferenceEngine',

    # SAM integration
    'PersonalizedSAMClient',
    'PersonalizedResponse',

    # Performance monitoring
    'DPOPerformanceMonitor',

    # User journey validation
    'UserJourneyValidator',
    'UserJourneyResult',

    # Factory functions
    'get_dpo_config_manager',
    'get_dpo_training_manager',
    'get_dpo_model_manager',
    'get_personalized_inference_engine',
    'get_personalized_sam_client',
    'get_dpo_performance_monitor',
    'get_user_journey_validator',

    # Convenience functions
    'generate_personalized_response',
    'activate_user_model',
    'deactivate_user_model',
    'get_user_personalization_status',
    'validate_user_personalization_journey',
]

__version__ = "1.0.0"
