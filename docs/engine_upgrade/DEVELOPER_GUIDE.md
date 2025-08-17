# SAM Engine Upgrade Framework - Developer Guide

## Architecture Overview

The SAM Engine Upgrade Framework is designed as a modular system that enables seamless switching between different AI model engines while maintaining data integrity and user experience.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
├─────────────────────────────────────────────────────────────┤
│  Core Engines UI  │  Migration Wizard  │  Engine Indicators │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Management Layer                          │
├─────────────────────────────────────────────────────────────┤
│ MigrationController │ ModelLibraryManager │ BackgroundTasks │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Engine Abstraction                       │
├─────────────────────────────────────────────────────────────┤
│   BaseModelEngine   │   DeepSeekEngine   │   LlamaEngine    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                           │
├─────────────────────────────────────────────────────────────┤
│  Model Storage  │  Configuration  │  Backups  │  Metadata  │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Core Components

### 1. BaseModelEngine (Abstract Interface)

**Location**: `sam/core/model_interface.py`

```python
class BaseModelEngine(abc.ABC):
    """Abstract base class for all model engines."""
    
    @abc.abstractmethod
    def load_model(self) -> bool:
        """Load the model into memory."""
        pass
    
    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the loaded model."""
        pass
    
    @abc.abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        pass
    
    @abc.abstractmethod
    def unload_model(self) -> bool:
        """Unload the model from memory."""
        pass
```

**Key Features**:
- Standardized interface for all engines
- Consistent API across different model families
- Built-in status tracking and error handling

### 2. ModelLibraryManager

**Location**: `sam/core/model_library_manager.py`

**Responsibilities**:
- Manage catalog of available models
- Handle model downloads with resume capability
- Track downloaded models and metadata
- Provide status information for UI

**Key Methods**:
```python
def download_model(self, model_id: str) -> bool:
    """Download a model from the catalog."""
    
def get_model_status(self, model_id: str) -> str:
    """Get status: not_downloaded, downloading, downloaded, etc."""
    
def get_available_models(self) -> List[ModelInfo]:
    """Get list of models available for download."""
```

### 3. MigrationController

**Location**: `sam/core/migration_controller.py`

**Responsibilities**:
- Orchestrate engine upgrade process
- Handle LoRA adapter invalidation
- Manage backups and rollbacks
- Update SAM configuration
- Coordinate background tasks

**Migration Workflow**:
```python
def execute_migration(self, migration_id: str) -> bool:
    """
    1. Create backup
    2. Invalidate LoRA adapters
    3. Update configuration
    4. Update prompt templates
    5. Start re-embedding task
    """
```

## 🔌 Adding New Engine Support

### Step 1: Implement Engine Class

Create a new engine class inheriting from `BaseModelEngine`:

```python
class NewModelEngine(BaseModelEngine):
    def __init__(self, engine_id: str, model_name: str, model_path: str):
        super().__init__(engine_id, model_name, model_path)
        # Engine-specific initialization
    
    def load_model(self) -> bool:
        # Implement model loading logic
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Implement text generation
        pass
    
    def embed(self, text: str) -> List[float]:
        # Implement embedding generation
        pass
    
    def unload_model(self) -> bool:
        # Implement model unloading
        pass
```

### Step 2: Add to Model Catalog

Update the model catalog in `ModelLibraryManager._create_default_catalog()`:

```python
ModelInfo(
    model_id="new-model-8b-q4",
    display_name="New Model 8B (Q4_K_M)",
    description="Description of the new model",
    model_family="new_family",
    huggingface_repo="organization/model-repo",
    filename="model-file.gguf",
    file_size=4800000000,
    quantization="Q4_K_M",
    context_length=32000,
    recommended=True,
    tags=["new", "efficient"]
)
```

### Step 3: Update Prompt Templates

Add family-specific prompts in `MigrationController._update_prompt_templates()`:

```python
prompt_templates = {
    'new_family': {
        'system_prompt': "You are a helpful AI assistant...",
        'reasoning_prompt': "Let me analyze this...",
        'chat_format': "<|user|>{user_input}<|assistant|>"
    }
}
```

## 🧪 Testing Framework

### Unit Tests

**Location**: `tests/test_engine_upgrade_e2e.py`

Run comprehensive tests:
```bash
python tests/test_engine_upgrade_e2e.py
```

### Benchmarking

**Location**: `scripts/benchmark_engine_upgrade.py`

Run performance benchmarks:
```bash
python scripts/benchmark_engine_upgrade.py --test all
```

### Test Scenarios

1. **Happy Path**: Full migration with all options
2. **Cautious User**: Migration with minimal options
3. **Rollback Path**: Failure recovery testing
4. **Performance**: Resource usage and timing

## 🔧 Configuration Management

### Engine Configuration

Engine settings are stored in SAM's main configuration:

```python
config.engine_upgrade = {
    'active_engine_id': 'llama-3.1-8b-q4',
    'engine_model_name': 'Llama 3.1 8B (Q4_K_M)',
    'engine_family': 'llama',
    'last_upgrade': '2025-01-15T10:30:00',
    'local_model_path': '/path/to/model'
}
```

### Model Catalog

Models are defined in `sam/assets/core_models/model_catalog.json`:

```json
{
  "version": "1.0.0",
  "models": {
    "model-id": {
      "model_id": "model-id",
      "display_name": "Model Display Name",
      "model_family": "family",
      "huggingface_repo": "org/repo",
      "filename": "model.gguf",
      "file_size": 4800000000,
      "context_length": 16000
    }
  }
}
```

## 🔄 Migration State Management

### Migration Plans

Each migration creates a `MigrationPlan` object:

```python
@dataclass
class MigrationPlan:
    migration_id: str
    from_engine: str
    to_engine: str
    user_id: str
    created_at: datetime
    
    # Options
    backup_lora_adapters: bool
    re_embed_knowledge: bool
    update_prompt_templates: bool
    
    # Status
    status: MigrationStatus
    progress_percentage: float
    current_step: str
```

### State Persistence

Migration state is persisted in:
- `sam/assets/migration_backups/migration_state.json`
- Individual backup directories for each migration
- Progress files for background tasks

## 🎨 UI Integration

### Adding Engine Awareness

To make a UI component engine-aware:

```python
def render_my_component():
    # Get current engine info
    from sam.core.model_interface import get_current_model_info
    from sam.config import get_config_manager
    
    current_info = get_current_model_info()
    config = get_config_manager().get_config()
    
    if hasattr(config, 'engine_upgrade') and config.engine_upgrade:
        engine_info = config.engine_upgrade
        st.info(f"Current Engine: {engine_info['engine_model_name']}")
    else:
        st.info(f"Current Model: {current_info['primary_model']}")
```

### Status Indicators

Use consistent status indicators across the UI:
- 🔧 for engine-related features
- ✅ for successful states
- ⚠️ for warnings
- ❌ for errors
- 🔄 for in-progress operations

## 📊 Performance Considerations

### Memory Management

- Models are loaded/unloaded as needed
- LoRA adapters are cached intelligently
- Background tasks use separate processes

### Disk Usage

- Models: 2-8GB each
- Backups: 10-100MB per migration
- Progress files: <1MB each
- Logs: Variable based on activity

### Network Optimization

- Resume capability for interrupted downloads
- Checksum validation for integrity
- Progress tracking for user feedback

## 🔒 Security Considerations

### Model Validation

- SHA256 checksums for downloaded models
- Signature verification (future enhancement)
- Sandboxed model loading

### Backup Security

- Encrypted backups (future enhancement)
- Access control for backup directories
- Audit logging for migrations

### Configuration Protection

- Validation of configuration changes
- Rollback capabilities
- Backup of previous configurations

## 🚀 Deployment Guidelines

### Production Deployment

1. **Test thoroughly** in staging environment
2. **Backup existing data** before deployment
3. **Monitor resource usage** during rollout
4. **Have rollback plan** ready
5. **Update documentation** for users

### Scaling Considerations

- **Multi-user support**: Separate migration state per user
- **Concurrent migrations**: Queue management
- **Resource limits**: CPU/memory/disk quotas
- **Load balancing**: Distribute model loading

## 🔮 Future Enhancements

### Planned Features

1. **Multi-user Support**: Per-user engine preferences
2. **Model Versioning**: Track and manage model versions
3. **Automatic Updates**: Background model updates
4. **Cloud Integration**: Remote model storage
5. **Advanced Scheduling**: Scheduled migrations

### Extension Points

- **Custom Engines**: Plugin architecture for third-party engines
- **Migration Hooks**: Custom actions during migration
- **Monitoring Integration**: Metrics and alerting
- **API Endpoints**: REST API for programmatic control

## 📚 API Reference

### Core Classes

- `BaseModelEngine`: Abstract engine interface
- `ModelLibraryManager`: Model catalog and download management
- `MigrationController`: Migration orchestration
- `MigrationPlan`: Migration state and configuration

### Key Functions

- `get_model_library_manager()`: Get global library manager
- `get_migration_controller()`: Get global migration controller
- `get_current_model_info()`: Get active model information

### Configuration

- `get_config_manager()`: Access SAM configuration
- Engine settings in `config.engine_upgrade`
- Model catalog in `sam/assets/core_models/`

---

**For More Information**: See the User Guide and API Reference documentation.
