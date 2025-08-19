#!/usr/bin/env python3
"""
Multi-LoRA Adapter Manager
==========================

Enhanced model manager that supports loading and stacking multiple LoRA adapters
simultaneously. Enables combining DPO (style) and SSRL (reasoning) adapters
for comprehensive personalization.

Features:
- Sequential adapter loading with priority ordering
- Robust error handling and graceful degradation
- Integration with existing DPO infrastructure
- Support for adapter activation/deactivation
- Performance monitoring and logging

Author: SAM Development Team
Version: 1.0.0
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_ADAPTERS_DIR = "./models/adapters"
DEFAULT_BASE_MODEL = "microsoft/DialoGPT-medium"
MAX_ADAPTERS_DEFAULT = 5
REGISTRY_FILENAME = "adapter_registry.json"


class AdapterType(Enum):
    """Types of LoRA adapters."""
    DPO_STYLE = "dpo_style"
    SSRL_REASONING = "ssrl_reasoning"
    CUSTOM = "custom"


class AdapterPriority(Enum):
    """Priority levels for adapter loading order."""
    HIGH = 1      # Load first (e.g., SSRL reasoning)
    MEDIUM = 2    # Load second (e.g., DPO style)
    LOW = 3       # Load last (e.g., custom adapters)


@dataclass
class AdapterInfo:
    """Information about a LoRA adapter."""
    adapter_id: str
    adapter_type: AdapterType
    model_path: str
    user_id: str
    priority: AdapterPriority
    is_active: bool = False
    is_validated: bool = False
    created_at: Optional[str] = None
    description: str = ""
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAdapterConfig:
    """Configuration for multi-adapter loading."""
    user_id: str
    active_adapters: List[str] = field(default_factory=list)
    adapter_order: List[str] = field(default_factory=list)
    base_model_name: str = "microsoft/DialoGPT-medium"
    max_adapters: int = 5
    enable_fallback: bool = True
    merge_adapters: bool = False  # Whether to merge adapters into base model


class MultiAdapterManager:
    """
    Manager for multiple LoRA adapters with stacking support.
    
    Handles loading, stacking, and managing multiple LoRA adapters
    on a single base model for comprehensive personalization.
    """
    
    def __init__(self, adapters_dir: str = DEFAULT_ADAPTERS_DIR) -> None:
        """
        Initialize multi-adapter manager.

        Args:
            adapters_dir: Directory containing adapter models

        Raises:
            OSError: If adapters directory cannot be created
            PermissionError: If insufficient permissions for directory operations
        """
        self.logger = logging.getLogger(f"{__name__}.MultiAdapterManager")

        # Validate and setup adapters directory
        try:
            self.adapters_dir = Path(adapters_dir).resolve()
            self.adapters_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            self.logger.error(f"Failed to create adapters directory {adapters_dir}: {e}")
            raise

        # Adapter registry
        self.adapters: Dict[str, AdapterInfo] = {}
        self.user_configs: Dict[str, MultiAdapterConfig] = {}

        # Model cache with thread safety consideration
        self.loaded_models: Dict[str, Any] = {}  # user_id -> loaded model
        self.base_models: Dict[str, Any] = {}    # model_name -> base model

        # Registry file
        self.registry_file = self.adapters_dir / REGISTRY_FILENAME

        # Performance tracking
        self._load_start_time = time.time()

        try:
            # Load existing adapters
            self.load_adapter_registry()
            self.scan_for_adapters()

            load_time = time.time() - self._load_start_time
            self.logger.info(
                f"Multi-Adapter Manager initialized successfully in {load_time:.2f}s: "
                f"{len(self.adapters)} adapters, {len(self.user_configs)} user configs"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize adapter manager: {e}")
            # Don't raise here - allow partial initialization
            self.logger.warning("Continuing with partial initialization")

    @contextmanager
    def _safe_model_operation(self, operation_name: str):
        """
        Context manager for safe model operations with proper cleanup.

        Args:
            operation_name: Name of the operation for logging
        """
        start_time = time.time()
        self.logger.debug(f"Starting {operation_name}")

        try:
            yield
        except Exception as e:
            self.logger.error(f"{operation_name} failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.logger.debug(f"{operation_name} completed in {duration:.2f}s")

    def cleanup(self) -> None:
        """
        Cleanup resources and save state.

        Should be called when shutting down the adapter manager.
        """
        try:
            self.logger.info("Cleaning up Multi-Adapter Manager")

            # Save current state
            self.save_adapter_registry()

            # Clear model caches to free memory
            self.loaded_models.clear()
            self.base_models.clear()

            self.logger.info("Multi-Adapter Manager cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def register_adapter(self, 
                        adapter_id: str,
                        adapter_type: AdapterType,
                        model_path: str,
                        user_id: str,
                        priority: AdapterPriority = AdapterPriority.MEDIUM,
                        description: str = "") -> bool:
        """
        Register a new LoRA adapter.
        
        Args:
            adapter_id: Unique adapter identifier
            adapter_type: Type of adapter (DPO, SSRL, etc.)
            model_path: Path to adapter model files
            user_id: User who owns this adapter
            priority: Loading priority
            description: Human-readable description
            
        Returns:
            True if registration successful
        """
        try:
            # Validate adapter path
            adapter_path = Path(model_path)
            if not adapter_path.exists():
                raise ValueError(f"Adapter path does not exist: {adapter_path}")
            
            # Create adapter info
            adapter_info = AdapterInfo(
                adapter_id=adapter_id,
                adapter_type=adapter_type,
                model_path=str(adapter_path),
                user_id=user_id,
                priority=priority,
                description=description,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Validate adapter
            if self._validate_adapter(adapter_info):
                adapter_info.is_validated = True
                self.logger.info(f"Adapter validation successful: {adapter_id}")
            else:
                self.logger.warning(f"Adapter validation failed: {adapter_id}")
            
            # Register adapter
            self.adapters[adapter_id] = adapter_info
            self.save_adapter_registry()
            
            self.logger.info(f"Registered adapter: {adapter_id} ({adapter_type.value}) for user: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering adapter {adapter_id}: {e}")
            return False
    
    def configure_user_adapters(self, 
                               user_id: str,
                               active_adapters: List[str],
                               base_model_name: str = None) -> bool:
        """
        Configure active adapters for a user.
        
        Args:
            user_id: User identifier
            active_adapters: List of adapter IDs to activate
            base_model_name: Base model to use (optional)
            
        Returns:
            True if configuration successful
        """
        try:
            # Validate adapters exist and belong to user
            valid_adapters = []
            for adapter_id in active_adapters:
                if adapter_id not in self.adapters:
                    self.logger.warning(f"Adapter not found: {adapter_id}")
                    continue
                
                adapter = self.adapters[adapter_id]
                if adapter.user_id != user_id:
                    self.logger.warning(f"Adapter {adapter_id} does not belong to user {user_id}")
                    continue
                
                if not adapter.is_validated:
                    self.logger.warning(f"Adapter {adapter_id} is not validated")
                    continue
                
                valid_adapters.append(adapter_id)
            
            # Sort adapters by priority
            sorted_adapters = self._sort_adapters_by_priority(valid_adapters)
            
            # Create user configuration
            config = MultiAdapterConfig(
                user_id=user_id,
                active_adapters=valid_adapters,
                adapter_order=sorted_adapters,
                base_model_name=base_model_name or "microsoft/DialoGPT-medium"
            )
            
            self.user_configs[user_id] = config
            
            # Mark adapters as active
            for adapter_id in self.adapters:
                self.adapters[adapter_id].is_active = adapter_id in valid_adapters
            
            self.save_adapter_registry()
            
            self.logger.info(f"Configured {len(valid_adapters)} adapters for user {user_id}")
            self.logger.info(f"Adapter order: {sorted_adapters}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring adapters for user {user_id}: {e}")
            return False
    
    def load_user_model(self, user_id: str, force_reload: bool = False) -> Optional[Any]:
        """
        Load a model with all active adapters for a user.
        
        Args:
            user_id: User identifier
            force_reload: Force reload even if cached
            
        Returns:
            Loaded model with stacked adapters or None
        """
        try:
            # Check if model is already loaded
            if not force_reload and user_id in self.loaded_models:
                self.logger.info(f"Using cached model for user {user_id}")
                return self.loaded_models[user_id]
            
            # Get user configuration
            if user_id not in self.user_configs:
                self.logger.warning(f"No adapter configuration found for user {user_id}")
                return None
            
            config = self.user_configs[user_id]
            
            if not config.active_adapters:
                self.logger.info(f"No active adapters for user {user_id}")
                return None
            
            # Load base model
            base_model = self._load_base_model(config.base_model_name)
            if base_model is None:
                return None
            
            # Load and stack adapters
            model_with_adapters = self._stack_adapters(base_model, config)
            
            if model_with_adapters is not None:
                # Cache the loaded model
                self.loaded_models[user_id] = model_with_adapters
                self.logger.info(f"Successfully loaded model with {len(config.active_adapters)} adapters for user {user_id}")
            
            return model_with_adapters
            
        except Exception as e:
            self.logger.error(f"Error loading model for user {user_id}: {e}")
            return None
    
    def _load_base_model(self, model_name: str) -> Optional[Any]:
        """Load base model with caching."""
        try:
            # Check cache first
            if model_name in self.base_models:
                return self.base_models[model_name]
            
            # Import here to avoid dependency issues
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.logger.info(f"Loading base model: {model_name}")
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                load_in_4bit=True  # Memory optimization
            )
            
            # Cache the model
            self.base_models[model_name] = model
            
            self.logger.info(f"Base model loaded successfully: {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load base model {model_name}: {e}")
            return None
    
    def _stack_adapters(self, base_model: Any, config: MultiAdapterConfig) -> Optional[Any]:
        """
        Stack multiple LoRA adapters onto the base model.
        
        Args:
            base_model: Base model to apply adapters to
            config: User configuration with adapter order
            
        Returns:
            Model with stacked adapters or None
        """
        try:
            from peft import PeftModel
            
            current_model = base_model
            loaded_adapters = []
            
            # Load adapters in priority order
            for adapter_id in config.adapter_order:
                try:
                    adapter_info = self.adapters[adapter_id]
                    
                    self.logger.info(f"Loading adapter: {adapter_id} ({adapter_info.adapter_type.value})")
                    
                    # Load adapter onto current model
                    current_model = PeftModel.from_pretrained(
                        current_model,
                        adapter_info.model_path,
                        adapter_name=adapter_id
                    )
                    
                    loaded_adapters.append(adapter_id)
                    self.logger.info(f"Successfully loaded adapter: {adapter_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load adapter {adapter_id}: {e}")
                    
                    # Continue with other adapters if one fails
                    if not config.enable_fallback:
                        raise
                    else:
                        self.logger.warning(f"Skipping failed adapter {adapter_id}, continuing with others")
                        continue
            
            if loaded_adapters:
                self.logger.info(f"Successfully stacked {len(loaded_adapters)} adapters: {loaded_adapters}")
                return current_model
            else:
                self.logger.warning("No adapters were successfully loaded")
                return None
                
        except Exception as e:
            self.logger.error(f"Error stacking adapters: {e}")
            return None

    def _sort_adapters_by_priority(self, adapter_ids: List[str]) -> List[str]:
        """Sort adapters by priority (high to low)."""
        def get_priority(adapter_id: str) -> int:
            """Get priority value for an adapter ID."""
            if adapter_id in self.adapters:
                return self.adapters[adapter_id].priority.value
            return AdapterPriority.LOW.value

        return sorted(adapter_ids, key=get_priority)

    def _validate_adapter(self, adapter_info: AdapterInfo) -> bool:
        """Validate a LoRA adapter."""
        try:
            adapter_path = Path(adapter_info.model_path)

            # Check if path exists
            if not adapter_path.exists():
                return False

            # Check for required files
            required_files = ['adapter_config.json']
            for file_name in required_files:
                if not (adapter_path / file_name).exists():
                    self.logger.warning(f"Missing required file: {file_name}")
                    return False

            # Validate adapter config
            config_file = adapter_path / 'adapter_config.json'
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Basic config validation
            if config.get('peft_type') != 'LORA':
                self.logger.warning("Adapter is not a LoRA adapter")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating adapter {adapter_info.adapter_id}: {e}")
            return False

    def scan_for_adapters(self):
        """Scan for new adapters in the adapters directory."""
        try:
            # Scan for DPO adapters
            dpo_dir = self.adapters_dir / "dpo"
            if dpo_dir.exists():
                self._scan_dpo_adapters(dpo_dir)

            # Scan for SSRL adapters
            ssrl_dir = self.adapters_dir / "ssrl"
            if ssrl_dir.exists():
                self._scan_ssrl_adapters(ssrl_dir)

            # Scan for custom adapters
            custom_dir = self.adapters_dir / "custom"
            if custom_dir.exists():
                self._scan_custom_adapters(custom_dir)

        except Exception as e:
            self.logger.error(f"Error scanning for adapters: {e}")

    def _scan_dpo_adapters(self, dpo_dir: Path):
        """Scan for DPO adapters."""
        try:
            # Import DPO manager to get existing models
            from sam.cognition.dpo.model_manager import get_dpo_model_manager

            dpo_manager = get_dpo_model_manager()

            # Register DPO models as adapters
            for model_id, model in dpo_manager.models.items():
                adapter_id = f"dpo_{model_id}"

                if adapter_id not in self.adapters:
                    self.register_adapter(
                        adapter_id=adapter_id,
                        adapter_type=AdapterType.DPO_STYLE,
                        model_path=model.model_path,
                        user_id=model.user_id,
                        priority=AdapterPriority.MEDIUM,
                        description=f"DPO Style Adapter (Training Job: {model.training_job_id})"
                    )

        except Exception as e:
            self.logger.warning(f"Error scanning DPO adapters: {e}")

    def _scan_ssrl_adapters(self, ssrl_dir: Path):
        """Scan for SSRL adapters."""
        try:
            for user_dir in ssrl_dir.iterdir():
                if user_dir.is_dir():
                    user_id = user_dir.name

                    for model_dir in user_dir.iterdir():
                        if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
                            adapter_id = f"ssrl_{user_id}_{model_dir.name}"

                            if adapter_id not in self.adapters:
                                self.register_adapter(
                                    adapter_id=adapter_id,
                                    adapter_type=AdapterType.SSRL_REASONING,
                                    model_path=str(model_dir),
                                    user_id=user_id,
                                    priority=AdapterPriority.HIGH,
                                    description="SSRL Reasoning Adapter"
                                )

        except Exception as e:
            self.logger.warning(f"Error scanning SSRL adapters: {e}")

    def _scan_custom_adapters(self, custom_dir: Path):
        """Scan for custom adapters."""
        try:
            for user_dir in custom_dir.iterdir():
                if user_dir.is_dir():
                    user_id = user_dir.name

                    for model_dir in user_dir.iterdir():
                        if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
                            adapter_id = f"custom_{user_id}_{model_dir.name}"

                            if adapter_id not in self.adapters:
                                self.register_adapter(
                                    adapter_id=adapter_id,
                                    adapter_type=AdapterType.CUSTOM,
                                    model_path=str(model_dir),
                                    user_id=user_id,
                                    priority=AdapterPriority.LOW,
                                    description="Custom Adapter"
                                )

        except Exception as e:
            self.logger.warning(f"Error scanning custom adapters: {e}")

    def get_user_adapters(self, user_id: str) -> List[AdapterInfo]:
        """Get all adapters for a specific user."""
        return [adapter for adapter in self.adapters.values() if adapter.user_id == user_id]

    def get_active_adapters(self, user_id: str) -> List[AdapterInfo]:
        """Get active adapters for a specific user."""
        user_adapters = self.get_user_adapters(user_id)
        return [adapter for adapter in user_adapters if adapter.is_active]

    def deactivate_adapter(self, user_id: str, adapter_id: str) -> bool:
        """Deactivate a specific adapter for a user."""
        try:
            if adapter_id not in self.adapters:
                return False

            adapter = self.adapters[adapter_id]
            if adapter.user_id != user_id:
                return False

            adapter.is_active = False

            # Update user configuration
            if user_id in self.user_configs:
                config = self.user_configs[user_id]
                if adapter_id in config.active_adapters:
                    config.active_adapters.remove(adapter_id)
                if adapter_id in config.adapter_order:
                    config.adapter_order.remove(adapter_id)

            # Clear cached model
            if user_id in self.loaded_models:
                del self.loaded_models[user_id]

            self.save_adapter_registry()

            self.logger.info(f"Deactivated adapter {adapter_id} for user {user_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error deactivating adapter {adapter_id}: {e}")
            return False

    def save_adapter_registry(self):
        """Save adapter registry to file."""
        try:
            registry_data = {
                'adapters': {},
                'user_configs': {}
            }

            # Save adapters
            for adapter_id, adapter in self.adapters.items():
                registry_data['adapters'][adapter_id] = {
                    'adapter_id': adapter.adapter_id,
                    'adapter_type': adapter.adapter_type.value,
                    'model_path': adapter.model_path,
                    'user_id': adapter.user_id,
                    'priority': adapter.priority.value,
                    'is_active': adapter.is_active,
                    'is_validated': adapter.is_validated,
                    'created_at': adapter.created_at,
                    'description': adapter.description,
                    'performance_metrics': adapter.performance_metrics
                }

            # Save user configurations
            for user_id, config in self.user_configs.items():
                registry_data['user_configs'][user_id] = {
                    'user_id': config.user_id,
                    'active_adapters': config.active_adapters,
                    'adapter_order': config.adapter_order,
                    'base_model_name': config.base_model_name,
                    'max_adapters': config.max_adapters,
                    'enable_fallback': config.enable_fallback,
                    'merge_adapters': config.merge_adapters
                }

            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving adapter registry: {e}")

    def load_adapter_registry(self):
        """Load adapter registry from file."""
        try:
            if not self.registry_file.exists():
                return

            with open(self.registry_file, 'r') as f:
                registry_data = json.load(f)

            # Load adapters
            for adapter_id, adapter_data in registry_data.get('adapters', {}).items():
                adapter = AdapterInfo(
                    adapter_id=adapter_data['adapter_id'],
                    adapter_type=AdapterType(adapter_data['adapter_type']),
                    model_path=adapter_data['model_path'],
                    user_id=adapter_data['user_id'],
                    priority=AdapterPriority(adapter_data['priority']),
                    is_active=adapter_data.get('is_active', False),
                    is_validated=adapter_data.get('is_validated', False),
                    created_at=adapter_data.get('created_at'),
                    description=adapter_data.get('description', ''),
                    performance_metrics=adapter_data.get('performance_metrics', {})
                )
                self.adapters[adapter_id] = adapter

            # Load user configurations
            for user_id, config_data in registry_data.get('user_configs', {}).items():
                config = MultiAdapterConfig(
                    user_id=config_data['user_id'],
                    active_adapters=config_data.get('active_adapters', []),
                    adapter_order=config_data.get('adapter_order', []),
                    base_model_name=config_data.get('base_model_name', 'microsoft/DialoGPT-medium'),
                    max_adapters=config_data.get('max_adapters', 5),
                    enable_fallback=config_data.get('enable_fallback', True),
                    merge_adapters=config_data.get('merge_adapters', False)
                )
                self.user_configs[user_id] = config

        except Exception as e:
            self.logger.error(f"Error loading adapter registry: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter manager statistics."""
        total_adapters = len(self.adapters)
        active_adapters = sum(1 for a in self.adapters.values() if a.is_active)
        validated_adapters = sum(1 for a in self.adapters.values() if a.is_validated)

        adapters_by_type = {}
        for adapter_type in AdapterType:
            count = sum(1 for a in self.adapters.values() if a.adapter_type == adapter_type)
            adapters_by_type[adapter_type.value] = count

        users_with_adapters = len(set(a.user_id for a in self.adapters.values()))

        return {
            'total_adapters': total_adapters,
            'active_adapters': active_adapters,
            'validated_adapters': validated_adapters,
            'adapters_by_type': adapters_by_type,
            'users_with_adapters': users_with_adapters,
            'loaded_models': len(self.loaded_models),
            'cached_base_models': len(self.base_models)
        }


# Global instance
_multi_adapter_manager = None

def get_multi_adapter_manager() -> MultiAdapterManager:
    """Get or create a global multi-adapter manager instance."""
    global _multi_adapter_manager

    if _multi_adapter_manager is None:
        _multi_adapter_manager = MultiAdapterManager()

    return _multi_adapter_manager
