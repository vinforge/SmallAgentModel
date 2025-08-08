"""
Configuration for Table Processing Module
=========================================

Centralized configuration management for all table processing components
including model paths, thresholds, and processing parameters.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    role_classifier_path: str
    table_detector_path: str
    tokenizer_path: Optional[str] = None
    model_type: str = "distilbert"
    max_sequence_length: int = 512
    batch_size: int = 32
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class DetectionConfig:
    """Configuration for table detection."""
    confidence_threshold: float = 0.7
    min_table_size: tuple = (2, 2)  # (min_rows, min_cols)
    max_table_size: tuple = (100, 50)  # (max_rows, max_cols)
    enable_html_detection: bool = True
    enable_markdown_detection: bool = True
    enable_pdf_detection: bool = True
    enable_image_detection: bool = True
    enable_csv_detection: bool = True


@dataclass
class ClassificationConfig:
    """Configuration for role classification."""
    confidence_threshold: float = 0.8
    enable_context_awareness: bool = True
    context_window_size: int = 3  # sentences before/after table
    enable_batch_processing: bool = True
    enable_confidence_calibration: bool = True
    fallback_to_heuristics: bool = True


@dataclass
class ValidationConfig:
    """Configuration for table validation."""
    quality_threshold: float = 0.6
    enable_structure_validation: bool = True
    enable_content_validation: bool = True
    enable_consistency_checks: bool = True
    max_empty_cell_ratio: float = 0.5
    min_data_cell_ratio: float = 0.2


@dataclass
class EnhancementConfig:
    """Configuration for table enhancement."""
    enable_post_processing: bool = True
    enable_cell_type_detection: bool = True
    enable_relationship_analysis: bool = True
    enable_semantic_enrichment: bool = True
    enable_quality_scoring: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_size: int = 1000
    enable_parallel_processing: bool = True
    max_workers: int = 4
    timeout_seconds: int = 30
    memory_limit_mb: int = 512


@dataclass
class TableProcessingConfig:
    """Complete configuration for table processing system."""
    model: ModelConfig
    detection: DetectionConfig
    classification: ClassificationConfig
    validation: ValidationConfig
    enhancement: EnhancementConfig
    performance: PerformanceConfig
    
    # Global settings
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_output_path: str = "logs/table_processing_metrics.json"


# Default configuration
DEFAULT_CONFIG = TableProcessingConfig(
    model=ModelConfig(
        role_classifier_path="sam/assets/table_role_classifier.bin",
        table_detector_path="sam/assets/table_detector.bin",
        tokenizer_path="sam/assets/table_tokenizer",
        model_type="distilbert",
        max_sequence_length=512,
        batch_size=32,
        device="auto"
    ),
    detection=DetectionConfig(
        confidence_threshold=0.7,
        min_table_size=(2, 2),
        max_table_size=(100, 50),
        enable_html_detection=True,
        enable_markdown_detection=True,
        enable_pdf_detection=True,
        enable_image_detection=True,
        enable_csv_detection=True
    ),
    classification=ClassificationConfig(
        confidence_threshold=0.8,
        enable_context_awareness=True,
        context_window_size=3,
        enable_batch_processing=True,
        enable_confidence_calibration=True,
        fallback_to_heuristics=True
    ),
    validation=ValidationConfig(
        quality_threshold=0.6,
        enable_structure_validation=True,
        enable_content_validation=True,
        enable_consistency_checks=True,
        max_empty_cell_ratio=0.5,
        min_data_cell_ratio=0.2
    ),
    enhancement=EnhancementConfig(
        enable_post_processing=True,
        enable_cell_type_detection=True,
        enable_relationship_analysis=True,
        enable_semantic_enrichment=True,
        enable_quality_scoring=True
    ),
    performance=PerformanceConfig(
        enable_caching=True,
        cache_size=1000,
        enable_parallel_processing=True,
        max_workers=4,
        timeout_seconds=30,
        memory_limit_mb=512
    ),
    debug_mode=False,
    log_level="INFO",
    enable_metrics=True,
    metrics_output_path="logs/table_processing_metrics.json"
)

# Global configuration instance
_config: Optional[TableProcessingConfig] = None


def get_table_config() -> TableProcessingConfig:
    """Get the current table processing configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def load_config(config_path: Optional[str] = None) -> TableProcessingConfig:
    """Load configuration from file or return default."""
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return _dict_to_config(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    
    return DEFAULT_CONFIG


def save_config(config: TableProcessingConfig, config_path: str):
    """Save configuration to file."""
    try:
        config_dict = asdict(config)
        os.makedirs(Path(config_path).parent, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")


def update_config(updates: Dict[str, Any]):
    """Update current configuration with new values."""
    global _config
    config = get_table_config()
    
    # Apply updates recursively
    _update_config_recursive(config, updates)
    _config = config


def _update_config_recursive(obj, updates: Dict[str, Any]):
    """Recursively update configuration object."""
    for key, value in updates.items():
        if hasattr(obj, key):
            current_value = getattr(obj, key)
            if isinstance(current_value, (ModelConfig, DetectionConfig, 
                                        ClassificationConfig, ValidationConfig,
                                        EnhancementConfig, PerformanceConfig)):
                if isinstance(value, dict):
                    _update_config_recursive(current_value, value)
                else:
                    setattr(obj, key, value)
            else:
                setattr(obj, key, value)


def _dict_to_config(config_dict: Dict[str, Any]) -> TableProcessingConfig:
    """Convert dictionary to TableProcessingConfig object."""
    try:
        return TableProcessingConfig(
            model=ModelConfig(**config_dict.get('model', {})),
            detection=DetectionConfig(**config_dict.get('detection', {})),
            classification=ClassificationConfig(**config_dict.get('classification', {})),
            validation=ValidationConfig(**config_dict.get('validation', {})),
            enhancement=EnhancementConfig(**config_dict.get('enhancement', {})),
            performance=PerformanceConfig(**config_dict.get('performance', {})),
            debug_mode=config_dict.get('debug_mode', False),
            log_level=config_dict.get('log_level', 'INFO'),
            enable_metrics=config_dict.get('enable_metrics', True),
            metrics_output_path=config_dict.get('metrics_output_path', 'logs/table_processing_metrics.json')
        )
    except Exception as e:
        logger.error(f"Failed to parse config dictionary: {e}")
        return DEFAULT_CONFIG


def get_model_path(model_name: str) -> str:
    """Get path for a specific model."""
    config = get_table_config()
    if model_name == "role_classifier":
        return config.model.role_classifier_path
    elif model_name == "table_detector":
        return config.model.table_detector_path
    elif model_name == "tokenizer":
        return config.model.tokenizer_path or "sam/assets/table_tokenizer"
    else:
        raise ValueError(f"Unknown model: {model_name}")


def validate_config(config: TableProcessingConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check model paths
    if not Path(config.model.role_classifier_path).parent.exists():
        issues.append(f"Role classifier directory does not exist: {config.model.role_classifier_path}")
    
    # Check thresholds
    if not 0 <= config.detection.confidence_threshold <= 1:
        issues.append("Detection confidence threshold must be between 0 and 1")
    
    if not 0 <= config.classification.confidence_threshold <= 1:
        issues.append("Classification confidence threshold must be between 0 and 1")
    
    if not 0 <= config.validation.quality_threshold <= 1:
        issues.append("Validation quality threshold must be between 0 and 1")
    
    # Check performance settings
    if config.performance.max_workers < 1:
        issues.append("Max workers must be at least 1")
    
    if config.performance.timeout_seconds < 1:
        issues.append("Timeout must be at least 1 second")
    
    return issues
