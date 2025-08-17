# SAM Engine Upgrade Framework - API Reference

## Overview

This document provides comprehensive API reference for the SAM Engine Upgrade Framework, including all classes, methods, and configuration options.

## 🔧 Core Classes

### BaseModelEngine

**Location**: `sam.core.model_interface.BaseModelEngine`

Abstract base class for all model engines in the upgrade framework.

#### Constructor

```python
def __init__(self, engine_id: str, model_name: str, model_path: str)
```

**Parameters**:
- `engine_id` (str): Unique identifier for the engine
- `model_name` (str): Display name of the model
- `model_path` (str): Path or URL to the model

#### Abstract Methods

```python
@abc.abstractmethod
def load_model(self) -> bool:
    """Load the model into memory. Returns True if successful."""

@abc.abstractmethod
def generate(self, prompt: str, **kwargs) -> str:
    """Generate text using the loaded model."""

@abc.abstractmethod
def embed(self, text: str) -> List[float]:
    """Generate embeddings for the given text."""

@abc.abstractmethod
def unload_model(self) -> bool:
    """Unload the model from memory. Returns True if successful."""
```

#### Properties

```python
engine_id: str          # Engine identifier
model_name: str         # Model display name
model_path: str         # Model path/URL
status: ModelStatus     # Current engine status
is_loaded: bool         # Whether model is loaded
```

#### Methods

```python
def get_engine_info(self) -> Dict[str, Any]:
    """Get comprehensive engine information."""
```

---

### ModelLibraryManager

**Location**: `sam.core.model_library_manager.ModelLibraryManager`

Manages the library of downloadable core models.

#### Constructor

```python
def __init__(self, models_dir: str = "./sam/assets/core_models")
```

**Parameters**:
- `models_dir` (str): Directory to store downloaded models

#### Methods

##### Model Catalog

```python
def get_available_models(self) -> List[ModelInfo]:
    """Get list of available models for download."""

def get_downloaded_models(self) -> List[DownloadedModel]:
    """Get list of downloaded models."""

def get_model_status(self, model_id: str) -> str:
    """
    Get the status of a model.
    
    Returns: "not_downloaded", "downloading", "downloaded", "corrupted", "missing"
    """
```

##### Download Management

```python
def download_model(self, model_id: str) -> bool:
    """
    Download a model from the catalog.
    
    Returns: True if download started successfully
    """

def get_download_progress(self, model_id: str) -> Optional[Dict[str, Any]]:
    """Get download progress for a model."""

def cancel_download(self, model_id: str) -> bool:
    """Cancel an active download."""

def delete_model(self, model_id: str) -> bool:
    """Delete a downloaded model."""
```

#### Global Access

```python
from sam.core.model_library_manager import get_model_library_manager

manager = get_model_library_manager()
```

---

### MigrationController

**Location**: `sam.core.migration_controller.MigrationController`

Orchestrates the engine upgrade process.

#### Constructor

```python
def __init__(self, backup_dir: str = "./sam/assets/migration_backups")
```

**Parameters**:
- `backup_dir` (str): Directory for migration backups

#### Methods

##### Migration Planning

```python
def create_migration_plan(
    self, 
    from_engine: str, 
    to_engine: str, 
    user_id: str,
    backup_lora: bool = True, 
    re_embed: bool = True,
    update_prompts: bool = True
) -> str:
    """
    Create a migration plan for engine upgrade.
    
    Returns: Migration ID
    """
```

##### Migration Execution

```python
def execute_migration(self, migration_id: str) -> bool:
    """
    Execute a migration plan.
    
    Returns: True if migration started successfully
    """

def get_migration_status(self, migration_id: str) -> Optional[MigrationPlan]:
    """Get status of a migration."""

def cancel_migration(self, migration_id: str) -> bool:
    """Cancel an active migration."""
```

#### Global Access

```python
from sam.core.migration_controller import get_migration_controller

controller = get_migration_controller()
```

---

## 📊 Data Classes

### ModelInfo

**Location**: `sam.core.model_library_manager.ModelInfo`

Information about a downloadable model.

```python
@dataclass
class ModelInfo:
    model_id: str               # Unique model identifier
    display_name: str           # Human-readable name
    description: str            # Model description
    model_family: str           # Model family (e.g., "llama", "deepseek")
    huggingface_repo: str       # Hugging Face repository
    filename: str               # Model filename
    file_size: int              # File size in bytes
    quantization: str           # Quantization type (e.g., "Q4_K_M")
    context_length: int         # Maximum context length
    recommended: bool           # Whether model is recommended
    tags: List[str]             # Model tags
```

### DownloadedModel

**Location**: `sam.core.model_library_manager.DownloadedModel`

Information about a downloaded model.

```python
@dataclass
class DownloadedModel:
    model_id: str               # Model identifier
    local_path: str             # Local file path
    downloaded_at: datetime     # Download timestamp
    file_size: int              # File size in bytes
    checksum: str               # SHA256 checksum
    model_info: ModelInfo       # Original model information
    status: str                 # Current status
```

### MigrationPlan

**Location**: `sam.core.migration_controller.MigrationPlan`

Plan for engine migration.

```python
@dataclass
class MigrationPlan:
    migration_id: str                   # Unique migration identifier
    from_engine: str                    # Source engine
    to_engine: str                      # Target engine
    user_id: str                        # User identifier
    created_at: datetime                # Creation timestamp
    
    # Migration options
    backup_lora_adapters: bool          # Whether to backup LoRA adapters
    re_embed_knowledge: bool            # Whether to re-embed knowledge base
    update_prompt_templates: bool       # Whether to update prompts
    
    # Affected components
    affected_lora_adapters: List[str]   # List of affected LoRA adapters
    knowledge_base_size: int            # Size of knowledge base
    estimated_duration_minutes: int     # Estimated migration time
    
    # Status tracking
    status: MigrationStatus             # Current status
    current_step: str                   # Current operation
    progress_percentage: float          # Progress (0-100)
    error_message: Optional[str]        # Error message if failed
```

---

## 🔄 Enums

### MigrationStatus

**Location**: `sam.core.migration_controller.MigrationStatus`

```python
class MigrationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### ModelStatus

**Location**: `sam.core.model_interface.ModelStatus`

```python
class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNAVAILABLE = "unavailable"
```

---

## 🔧 Configuration

### Engine Configuration

Engine upgrade settings are stored in SAM's main configuration:

```python
# Access configuration
from sam.config import get_config_manager

config_manager = get_config_manager()
config = config_manager.get_config()

# Engine upgrade configuration
if hasattr(config, 'engine_upgrade'):
    engine_config = config.engine_upgrade
    active_engine = engine_config.get('active_engine_id')
    engine_name = engine_config.get('engine_model_name')
    engine_family = engine_config.get('engine_family')
    last_upgrade = engine_config.get('last_upgrade')
    model_path = engine_config.get('local_model_path')
```

### Model Catalog Configuration

Model catalog structure in `sam/assets/core_models/model_catalog.json`:

```json
{
  "version": "1.0.0",
  "updated_at": "2025-01-15T10:30:00",
  "models": {
    "model-id": {
      "model_id": "model-id",
      "display_name": "Model Display Name",
      "description": "Model description",
      "model_family": "family",
      "huggingface_repo": "organization/repository",
      "filename": "model-file.gguf",
      "file_size": 4800000000,
      "quantization": "Q4_K_M",
      "context_length": 16000,
      "recommended": true,
      "tags": ["tag1", "tag2"]
    }
  }
}
```

---

## 🎯 Usage Examples

### Basic Engine Switch

```python
from sam.core.model_library_manager import get_model_library_manager
from sam.core.migration_controller import get_migration_controller

# Get managers
library_manager = get_model_library_manager()
migration_controller = get_migration_controller()

# Check available models
models = library_manager.get_available_models()
target_model = models[0]  # Choose first available model

# Download model if needed
if library_manager.get_model_status(target_model.model_id) == "not_downloaded":
    library_manager.download_model(target_model.model_id)

# Create and execute migration
migration_id = migration_controller.create_migration_plan(
    from_engine="current",
    to_engine=target_model.model_id,
    user_id="user123"
)

success = migration_controller.execute_migration(migration_id)
```

### Monitor Migration Progress

```python
# Check migration status
plan = migration_controller.get_migration_status(migration_id)

if plan:
    print(f"Status: {plan.status.value}")
    print(f"Progress: {plan.progress_percentage}%")
    print(f"Current Step: {plan.current_step}")
    
    if plan.status == MigrationStatus.FAILED:
        print(f"Error: {plan.error_message}")
```

### Custom Engine Implementation

```python
from sam.core.model_interface import BaseModelEngine

class CustomEngine(BaseModelEngine):
    def __init__(self, engine_id: str, model_name: str, model_path: str):
        super().__init__(engine_id, model_name, model_path)
        self.custom_model = None
    
    def load_model(self) -> bool:
        try:
            # Load your custom model
            self.custom_model = load_custom_model(self.model_path)
            self.status = ModelStatus.READY
            self.is_loaded = True
            return True
        except Exception as e:
            self.status = ModelStatus.ERROR
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Generate with your custom model
        return self.custom_model.generate(prompt, **kwargs)
    
    def embed(self, text: str) -> List[float]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Generate embeddings
        return self.custom_model.embed(text)
    
    def unload_model(self) -> bool:
        self.custom_model = None
        self.status = ModelStatus.UNAVAILABLE
        self.is_loaded = False
        return True
```

---

## 🔍 Error Handling

### Common Exceptions

```python
# Model loading errors
try:
    engine.load_model()
except RuntimeError as e:
    print(f"Model loading failed: {e}")

# Migration errors
try:
    migration_controller.execute_migration(migration_id)
except Exception as e:
    print(f"Migration failed: {e}")

# Download errors
try:
    library_manager.download_model(model_id)
except Exception as e:
    print(f"Download failed: {e}")
```

### Status Checking

```python
# Check model status before operations
status = library_manager.get_model_status(model_id)
if status == "downloaded":
    # Safe to activate
    pass
elif status == "downloading":
    # Wait for download to complete
    pass
elif status == "corrupted":
    # Re-download needed
    library_manager.delete_model(model_id)
    library_manager.download_model(model_id)
```

---

## 📚 Integration Points

### SAM Configuration Integration

```python
# Update SAM configuration after engine switch
from sam.config import get_config_manager

config_manager = get_config_manager()
config = config_manager.get_config()

# Engine upgrade framework updates this automatically
config.engine_upgrade = {
    'active_engine_id': 'new-engine-id',
    'engine_model_name': 'New Engine Name',
    'engine_family': 'new_family',
    'last_upgrade': datetime.now().isoformat(),
    'local_model_path': '/path/to/model'
}

config_manager.save_config(config)
```

### LoRA Adapter Integration

```python
# Check for LoRA compatibility after engine switch
from sam.cognition.dpo.model_manager import get_dpo_model_manager

dpo_manager = get_dpo_model_manager()
user_models = dpo_manager.get_user_models(user_id)

for model in user_models:
    if hasattr(model, 'training_stats') and 'migration_notes' in model.training_stats:
        # Check if model was invalidated by migration
        migration_notes = model.training_stats['migration_notes']
        if any(note.get('action') == 'invalidated_by_migration' for note in migration_notes):
            print(f"Model {model.model_id} needs retraining")
```

---

**For More Information**: See the User Guide and Developer Guide documentation.
