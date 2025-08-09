"""
DPO Model Manager

Manages personalized LoRA adapters, model loading, and runtime switching
for the SAM Personalized Tuner system.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import json
import logging
import shutil
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PersonalizedModel:
    """Represents a personalized model (LoRA adapter)."""
    model_id: str
    user_id: str
    base_model: str
    model_path: str
    created_at: datetime
    training_job_id: str
    
    # Model metadata
    lora_config: Dict[str, Any] = field(default_factory=dict)
    training_stats: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    is_active: bool = False
    is_validated: bool = False
    
    # Usage tracking
    usage_count: int = 0
    last_used: Optional[datetime] = None


class DPOModelManager:
    """
    Manager for personalized LoRA adapters and model switching.
    """
    
    def __init__(self, models_dir: str = "./models/personalized"):
        """
        Initialize the DPO model manager.
        
        Args:
            models_dir: Directory containing personalized models
        """
        self.logger = logging.getLogger(f"{__name__}.DPOModelManager")
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.models: Dict[str, PersonalizedModel] = {}
        self.active_models: Dict[str, str] = {}  # user_id -> model_id
        
        # Model registry file
        self.registry_file = self.models_dir / "model_registry.json"
        
        # Load existing models
        self.load_model_registry()
        self.scan_for_models()
        
        self.logger.info("DPO Model Manager initialized")
    
    def register_model(self, user_id: str, model_path: str, training_job_id: str,
                      base_model: str, lora_config: Dict[str, Any],
                      training_stats: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new personalized model.
        
        Args:
            user_id: User identifier
            model_path: Path to the trained model
            training_job_id: ID of the training job that created this model
            base_model: Base model name
            lora_config: LoRA configuration used for training
            training_stats: Training statistics
            
        Returns:
            Model ID
        """
        try:
            # Generate model ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{user_id}_{timestamp}"
            
            # Validate model path
            model_path = Path(model_path)
            if not model_path.exists():
                raise ValueError(f"Model path does not exist: {model_path}")
            
            # Create model entry
            model = PersonalizedModel(
                model_id=model_id,
                user_id=user_id,
                base_model=base_model,
                model_path=str(model_path),
                created_at=datetime.now(),
                training_job_id=training_job_id,
                lora_config=lora_config,
                training_stats=training_stats or {},
                is_active=False,
                is_validated=False
            )
            
            # Validate model
            if self._validate_model(model):
                model.is_validated = True
                self.logger.info(f"Model validation successful: {model_id}")
            else:
                self.logger.warning(f"Model validation failed: {model_id}")
            
            # Register model
            self.models[model_id] = model
            self.save_model_registry()
            
            self.logger.info(f"Registered personalized model: {model_id} for user: {user_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
            raise
    
    def activate_model(self, user_id: str, model_id: str) -> bool:
        """
        Activate a personalized model for a user.
        
        Args:
            user_id: User identifier
            model_id: Model identifier
            
        Returns:
            True if activated successfully
        """
        try:
            if model_id not in self.models:
                self.logger.error(f"Model not found: {model_id}")
                return False
            
            model = self.models[model_id]
            
            if model.user_id != user_id:
                self.logger.error(f"Model {model_id} does not belong to user {user_id}")
                return False
            
            if not model.is_validated:
                self.logger.warning(f"Activating unvalidated model: {model_id}")
            
            # Deactivate current model if any
            if user_id in self.active_models:
                old_model_id = self.active_models[user_id]
                if old_model_id in self.models:
                    self.models[old_model_id].is_active = False
            
            # Activate new model
            model.is_active = True
            model.last_used = datetime.now()
            model.usage_count += 1
            self.active_models[user_id] = model_id
            
            self.save_model_registry()
            self.logger.info(f"Activated model {model_id} for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error activating model: {e}")
            return False
    
    def deactivate_model(self, user_id: str) -> bool:
        """
        Deactivate the current personalized model for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if deactivated successfully
        """
        try:
            if user_id not in self.active_models:
                return True  # Already deactivated
            
            model_id = self.active_models[user_id]
            if model_id in self.models:
                self.models[model_id].is_active = False
            
            del self.active_models[user_id]
            self.save_model_registry()
            
            self.logger.info(f"Deactivated personalized model for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deactivating model: {e}")
            return False
    
    def get_active_model(self, user_id: str) -> Optional[PersonalizedModel]:
        """
        Get the active personalized model for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Active model or None
        """
        if user_id not in self.active_models:
            return None
        
        model_id = self.active_models[user_id]
        return self.models.get(model_id)
    
    def get_user_models(self, user_id: str) -> List[PersonalizedModel]:
        """
        Get all models for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user's models
        """
        return [model for model in self.models.values() if model.user_id == user_id]
    
    def delete_model(self, model_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a personalized model.
        
        Args:
            model_id: Model identifier
            user_id: User identifier (for authorization)
            
        Returns:
            True if deleted successfully
        """
        try:
            if model_id not in self.models:
                return False
            
            model = self.models[model_id]
            
            # Check authorization
            if user_id and model.user_id != user_id:
                self.logger.error(f"User {user_id} not authorized to delete model {model_id}")
                return False
            
            # Deactivate if active
            if model.is_active:
                self.deactivate_model(model.user_id)
            
            # Delete model files
            model_path = Path(model.model_path)
            if model_path.exists():
                if model_path.is_dir():
                    shutil.rmtree(model_path)
                else:
                    model_path.unlink()
            
            # Remove from registry
            del self.models[model_id]
            self.save_model_registry()
            
            self.logger.info(f"Deleted model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information dictionary
        """
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        
        # Calculate model size
        model_size = self._calculate_model_size(model.model_path)
        
        return {
            'model_id': model.model_id,
            'user_id': model.user_id,
            'base_model': model.base_model,
            'model_path': model.model_path,
            'created_at': model.created_at.isoformat(),
            'training_job_id': model.training_job_id,
            'lora_config': model.lora_config,
            'training_stats': model.training_stats,
            'performance_metrics': model.performance_metrics,
            'is_active': model.is_active,
            'is_validated': model.is_validated,
            'usage_count': model.usage_count,
            'last_used': model.last_used.isoformat() if model.last_used else None,
            'model_size_mb': model_size
        }
    
    def load_model_for_inference(self, model_id: str):
        """
        Load a personalized model for inference.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded model and tokenizer
        """
        try:
            # Import here to avoid dependency issues
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")
            
            model = self.models[model_id]
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model.base_model,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model.base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load LoRA adapter
            personalized_model = PeftModel.from_pretrained(
                base_model,
                model.model_path
            )
            
            # Update usage tracking
            model.last_used = datetime.now()
            model.usage_count += 1
            self.save_model_registry()
            
            self.logger.info(f"Loaded personalized model: {model_id}")
            return personalized_model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model for inference: {e}")
            raise
    
    def _validate_model(self, model: PersonalizedModel) -> bool:
        """Validate a personalized model."""
        try:
            model_path = Path(model.model_path)
            
            # Check if path exists
            if not model_path.exists():
                return False
            
            # Check for required files
            required_files = ['adapter_config.json', 'adapter_model.safetensors']
            for file_name in required_files:
                if not (model_path / file_name).exists():
                    self.logger.warning(f"Missing required file: {file_name}")
                    return False
            
            # Validate adapter config
            config_file = model_path / 'adapter_config.json'
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Basic config validation
            if config.get('peft_type') != 'LORA':
                self.logger.warning("Model is not a LoRA adapter")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            return False
    
    def _calculate_model_size(self, model_path: str) -> float:
        """Calculate model size in MB."""
        try:
            path = Path(model_path)
            if not path.exists():
                return 0.0
            
            total_size = 0
            if path.is_file():
                total_size = path.stat().st_size
            else:
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 0.0
    
    def scan_for_models(self):
        """Scan the models directory for unregistered models."""
        try:
            for user_dir in self.models_dir.iterdir():
                if not user_dir.is_dir():
                    continue
                
                user_id = user_dir.name
                
                for model_dir in user_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    
                    # Check if model is already registered
                    model_id = model_dir.name
                    if any(m.model_path == str(model_dir) for m in self.models.values()):
                        continue
                    
                    # Check for training results
                    results_file = model_dir / 'training_results.json'
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                            
                            if results.get('success'):
                                # Auto-register the model
                                self.register_model(
                                    user_id=user_id,
                                    model_path=str(model_dir),
                                    training_job_id=model_id,
                                    base_model=results.get('model_name', 'unknown'),
                                    lora_config=results.get('lora_config', {}),
                                    training_stats=results
                                )
                                self.logger.info(f"Auto-registered model: {model_id}")
                        except Exception as e:
                            self.logger.warning(f"Error auto-registering model {model_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error scanning for models: {e}")
    
    def save_model_registry(self):
        """Save the model registry to disk."""
        try:
            registry_data = {
                'models': {},
                'active_models': self.active_models,
                'last_updated': datetime.now().isoformat()
            }
            
            for model_id, model in self.models.items():
                registry_data['models'][model_id] = {
                    'model_id': model.model_id,
                    'user_id': model.user_id,
                    'base_model': model.base_model,
                    'model_path': model.model_path,
                    'created_at': model.created_at.isoformat(),
                    'training_job_id': model.training_job_id,
                    'lora_config': model.lora_config,
                    'training_stats': model.training_stats,
                    'performance_metrics': model.performance_metrics,
                    'is_active': model.is_active,
                    'is_validated': model.is_validated,
                    'usage_count': model.usage_count,
                    'last_used': model.last_used.isoformat() if model.last_used else None
                }
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving model registry: {e}")
    
    def load_model_registry(self):
        """Load the model registry from disk."""
        try:
            if not self.registry_file.exists():
                return
            
            with open(self.registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Load models
            for model_id, model_data in registry_data.get('models', {}).items():
                model = PersonalizedModel(
                    model_id=model_data['model_id'],
                    user_id=model_data['user_id'],
                    base_model=model_data['base_model'],
                    model_path=model_data['model_path'],
                    created_at=datetime.fromisoformat(model_data['created_at']),
                    training_job_id=model_data['training_job_id'],
                    lora_config=model_data.get('lora_config', {}),
                    training_stats=model_data.get('training_stats', {}),
                    performance_metrics=model_data.get('performance_metrics', {}),
                    is_active=model_data.get('is_active', False),
                    is_validated=model_data.get('is_validated', False),
                    usage_count=model_data.get('usage_count', 0),
                    last_used=datetime.fromisoformat(model_data['last_used']) if model_data.get('last_used') else None
                )
                self.models[model_id] = model
            
            # Load active models
            self.active_models = registry_data.get('active_models', {})
            
            self.logger.info(f"Loaded {len(self.models)} models from registry")
            
        except Exception as e:
            self.logger.error(f"Error loading model registry: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model manager statistics."""
        total_models = len(self.models)
        active_models = len(self.active_models)
        validated_models = sum(1 for m in self.models.values() if m.is_validated)
        
        # Calculate total size
        total_size = sum(self._calculate_model_size(m.model_path) for m in self.models.values())
        
        # User statistics
        users_with_models = len(set(m.user_id for m in self.models.values()))
        
        return {
            'total_models': total_models,
            'active_models': active_models,
            'validated_models': validated_models,
            'total_size_mb': round(total_size, 2),
            'users_with_models': users_with_models,
            'models_by_user': {
                user_id: len([m for m in self.models.values() if m.user_id == user_id])
                for user_id in set(m.user_id for m in self.models.values())
            }
        }


# Global model manager instance
_model_manager = None

def get_dpo_model_manager() -> DPOModelManager:
    """Get or create a global DPO model manager instance."""
    global _model_manager
    
    if _model_manager is None:
        _model_manager = DPOModelManager()
    
    return _model_manager
