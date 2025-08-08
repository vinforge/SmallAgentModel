"""
Configuration Management System for SAM
Config templates, export/import tools, and memory snapshots.

Sprint 13 Task 2: Config Template & Export Tools
"""

import json
import logging
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class SAMConfig:
    """SAM configuration structure."""
    # System settings
    version: str = "1.0.0"
    created_at: str = ""
    last_updated: str = ""
    
    # Agent mode settings
    agent_mode: str = "solo"  # solo or collaborative
    collaboration_key_path: str = "config/collab_key.json"
    
    # Memory settings
    memory_backend: str = "simple"  # simple, faiss, chroma
    memory_storage_dir: str = "memory_store"
    memory_max_size_mb: int = 1000
    memory_auto_cleanup: bool = True
    memory_similarity_threshold: float = 0.3

    # Chroma-specific settings
    chroma_persist_path: str = "web_ui/chroma_db"
    chroma_collection_name: str = "sam_memory_store"
    chroma_distance_function: str = "cosine"
    chroma_batch_size: int = 100
    chroma_enable_hnsw: bool = True
    chroma_hnsw_space: str = "cosine"
    chroma_hnsw_construction_ef: int = 200
    chroma_hnsw_search_ef: int = 50
    
    # Server settings
    chat_port: int = 5001
    memory_ui_port: int = 8501
    streamlit_chat_port: int = 8502
    host: str = "localhost"
    debug_mode: bool = False
    
    # Model settings
    model_provider: str = "ollama"  # ollama, openai, local
    model_name: str = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
    model_api_url: str = "http://localhost:11434"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # UI settings
    theme: str = "default"
    auto_open_browser: bool = True
    show_onboarding: bool = True
    enable_tooltips: bool = True
    
    # Security settings
    enable_auth: bool = False
    auth_secret_key: str = ""
    allowed_origins: List[str] = None
    
    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    memory_cache_size: int = 1000
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/sam.log"
    log_max_size_mb: int = 100
    log_backup_count: int = 5

    # Web Retrieval settings
    web_retrieval_provider: str = "cocoindex"  # cocoindex, legacy
    cocoindex_num_pages: int = 5
    cocoindex_search_provider: str = "duckduckgo"  # serper, duckduckgo (default: free)
    serper_api_key: str = ""
    newsapi_api_key: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:5001", "http://localhost:8501"]

class ConfigManager:
    """
    Manages SAM configuration with templates, export/import, and validation.
    """
    
    def __init__(self, config_file: str = "config/sam_config.json"):
        """Initialize the configuration manager."""
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(exist_ok=True)
        
        self.config = self._load_config()
        
        logger.info(f"Configuration manager initialized: {config_file}")
    
    def _load_config(self) -> SAMConfig:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)

                    # Check if this is the old format and migrate
                    if "mode" in data and "agent_mode" not in data:
                        logger.info("Migrating configuration from old format...")
                        migrated_data = self._migrate_old_config(data)
                        return SAMConfig(**migrated_data)
                    else:
                        return SAMConfig(**data)
            else:
                # Create default config
                config = SAMConfig()
                self._save_config(config)
                return config

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
            return SAMConfig()

    def _migrate_old_config(self, old_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old configuration format to new format."""
        try:
            # Map old fields to new fields
            migrated = {
                "version": "1.0.0",
                "agent_mode": old_data.get("mode", "solo"),
                "collaboration_key_path": old_data.get("collab_key", "config/collab_key.json"),
                "memory_backend": old_data.get("vector_store", "simple"),
                "memory_storage_dir": old_data.get("memory_config", {}).get("storage_directory", "memory_store"),
                "memory_max_size_mb": 1000,
                "memory_auto_cleanup": old_data.get("memory_config", {}).get("auto_cleanup_enabled", True),
                "memory_similarity_threshold": old_data.get("memory_config", {}).get("similarity_threshold", 0.3),
                "chat_port": 5001,
                "memory_ui_port": 8501,
                "host": "localhost",
                "debug_mode": False,
                "model_provider": "ollama",
                "model_name": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                "model_api_url": "http://localhost:11434",
                "embedding_model": "all-MiniLM-L6-v2",
                "theme": "default",
                "auto_open_browser": True,
                "show_onboarding": True,
                "enable_tooltips": True,
                "enable_auth": False,
                "auth_secret_key": "",
                "allowed_origins": ["http://localhost:5001", "http://localhost:8501"],
                "max_concurrent_requests": 10,
                "request_timeout_seconds": 30,
                "memory_cache_size": old_data.get("performance", {}).get("memory_cache_size", 1000),
                "log_level": old_data.get("logging", {}).get("level", "INFO"),
                "log_file": old_data.get("logging", {}).get("log_file", "logs/sam.log"),
                "log_max_size_mb": old_data.get("logging", {}).get("max_log_size_mb", 100),
                "log_backup_count": old_data.get("logging", {}).get("backup_count", 5)
            }

            # Save the migrated config
            migrated_config = SAMConfig(**migrated)
            self._save_config(migrated_config)
            logger.info("Configuration migration completed")

            return migrated

        except Exception as e:
            logger.error(f"Error migrating config: {e}")
            # Return default config if migration fails
            return asdict(SAMConfig())
    
    def _save_config(self, config: SAMConfig):
        """Save configuration to file."""
        try:
            config.last_updated = datetime.now().isoformat()
            
            with open(self.config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2)
                
            logger.info(f"Configuration saved: {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_config(self) -> SAMConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        try:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")
            
            self._save_config(self.config)
            logger.info(f"Configuration updated: {list(updates.keys())}")
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
    
    def export_config(self, export_path: str) -> bool:
        """Export current configuration to file."""
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(exist_ok=True)
            
            export_data = {
                "config": asdict(self.config),
                "export_timestamp": datetime.now().isoformat(),
                "export_version": "1.0",
                "source_file": str(self.config_file)
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Configuration exported: {export_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting config: {e}")
            return False
    
    def import_config(self, import_path: str, merge: bool = False) -> bool:
        """Import configuration from file."""
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                logger.error(f"Import file not found: {import_file}")
                return False
            
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            if "config" in import_data:
                config_data = import_data["config"]
            else:
                config_data = import_data
            
            if merge:
                # Merge with existing config
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            else:
                # Replace entire config
                self.config = SAMConfig(**config_data)
            
            self._save_config(self.config)
            logger.info(f"Configuration imported: {import_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing config: {e}")
            return False
    
    def create_template(self, template_path: str, template_type: str = "default") -> bool:
        """Create configuration template."""
        try:
            template_file = Path(template_path)
            template_file.parent.mkdir(exist_ok=True)
            
            if template_type == "minimal":
                template_config = SAMConfig(
                    agent_mode="solo",
                    memory_backend="simple",
                    chat_port=5001,
                    memory_ui_port=8501,
                    debug_mode=False
                )
            elif template_type == "production":
                template_config = SAMConfig(
                    agent_mode="collaborative",
                    memory_backend="faiss",
                    chat_port=5001,
                    memory_ui_port=8501,
                    debug_mode=False,
                    enable_auth=True,
                    log_level="WARNING",
                    max_concurrent_requests=50
                )
            elif template_type == "development":
                template_config = SAMConfig(
                    agent_mode="solo",
                    memory_backend="simple",
                    chat_port=5001,
                    memory_ui_port=8501,
                    debug_mode=True,
                    log_level="DEBUG",
                    auto_open_browser=True
                )
            else:  # default
                template_config = SAMConfig()
            
            template_data = {
                "template_type": template_type,
                "template_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "description": f"SAM configuration template for {template_type} deployment",
                "config": asdict(template_config)
            }
            
            with open(template_file, 'w') as f:
                json.dump(template_data, f, indent=2)
            
            logger.info(f"Configuration template created: {template_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return False
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            # Check ports
            if self.config.chat_port == self.config.memory_ui_port:
                validation_result["errors"].append("Chat port and Memory UI port cannot be the same")
                validation_result["valid"] = False
            
            if not (1024 <= self.config.chat_port <= 65535):
                validation_result["errors"].append("Chat port must be between 1024 and 65535")
                validation_result["valid"] = False
            
            if not (1024 <= self.config.memory_ui_port <= 65535):
                validation_result["errors"].append("Memory UI port must be between 1024 and 65535")
                validation_result["valid"] = False
            
            # Check memory settings
            if self.config.memory_max_size_mb < 10:
                validation_result["warnings"].append("Memory max size is very low (< 10MB)")
            
            if self.config.memory_similarity_threshold < 0 or self.config.memory_similarity_threshold > 1:
                validation_result["errors"].append("Memory similarity threshold must be between 0 and 1")
                validation_result["valid"] = False
            
            # Check paths
            memory_dir = Path(self.config.memory_storage_dir)
            if not memory_dir.exists():
                validation_result["suggestions"].append(f"Memory storage directory will be created: {memory_dir}")
            
            # Check model settings
            if self.config.model_provider not in ["ollama", "openai", "local"]:
                validation_result["warnings"].append("Unknown model provider")
            
            # Check security settings
            if self.config.enable_auth and not self.config.auth_secret_key:
                validation_result["errors"].append("Authentication enabled but no secret key provided")
                validation_result["valid"] = False
            
            # Performance checks
            if self.config.max_concurrent_requests > 100:
                validation_result["warnings"].append("High concurrent request limit may impact performance")
            
            if self.config.request_timeout_seconds < 5:
                validation_result["warnings"].append("Very low request timeout may cause failures")
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
            validation_result["valid"] = False
        
        return validation_result
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "version": self.config.version,
            "agent_mode": self.config.agent_mode,
            "memory_backend": self.config.memory_backend,
            "ports": {
                "chat": self.config.chat_port,
                "memory_ui": self.config.memory_ui_port
            },
            "model": {
                "provider": self.config.model_provider,
                "name": self.config.model_name
            },
            "security": {
                "auth_enabled": self.config.enable_auth
            },
            "last_updated": self.config.last_updated,
            "config_file": str(self.config_file)
        }

class MemorySnapshotManager:
    """
    Manages memory database snapshots and backups.
    """
    
    def __init__(self, memory_dir: str = "memory_store"):
        """Initialize the snapshot manager."""
        self.memory_dir = Path(memory_dir)
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info("Memory snapshot manager initialized")
    
    def create_snapshot(self, snapshot_name: str = None) -> str:
        """Create a memory snapshot."""
        try:
            if not snapshot_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_name = f"memory_snapshot_{timestamp}"
            
            snapshot_file = self.backup_dir / f"{snapshot_name}.zip"
            
            with zipfile.ZipFile(snapshot_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add memory files
                if self.memory_dir.exists():
                    for file_path in self.memory_dir.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(self.memory_dir.parent)
                            zipf.write(file_path, arcname)
                
                # Add metadata
                metadata = {
                    "snapshot_name": snapshot_name,
                    "created_at": datetime.now().isoformat(),
                    "memory_dir": str(self.memory_dir),
                    "file_count": len(list(self.memory_dir.rglob("*"))) if self.memory_dir.exists() else 0
                }
                
                zipf.writestr("snapshot_metadata.json", json.dumps(metadata, indent=2))
            
            logger.info(f"Memory snapshot created: {snapshot_file}")
            return str(snapshot_file)
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return ""
    
    def restore_snapshot(self, snapshot_path: str, backup_existing: bool = True) -> bool:
        """Restore memory from snapshot."""
        try:
            snapshot_file = Path(snapshot_path)
            
            if not snapshot_file.exists():
                logger.error(f"Snapshot file not found: {snapshot_file}")
                return False
            
            # Backup existing memory if requested
            if backup_existing and self.memory_dir.exists():
                backup_name = f"pre_restore_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.create_snapshot(backup_name)
            
            # Clear existing memory directory
            if self.memory_dir.exists():
                shutil.rmtree(self.memory_dir)
            
            # Extract snapshot
            with zipfile.ZipFile(snapshot_file, 'r') as zipf:
                zipf.extractall(self.memory_dir.parent)
            
            logger.info(f"Memory snapshot restored: {snapshot_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring snapshot: {e}")
            return False
    
    def export_memory_json(self, export_path: str) -> bool:
        """Export memory to JSON format."""
        try:
            from memory.memory_vectorstore import get_memory_store
            
            memory_store = get_memory_store()
            export_file = Path(export_path)
            export_file.parent.mkdir(exist_ok=True)
            
            # Get all memories
            memories = []
            for chunk in memory_store.memory_chunks.values():
                memory_data = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "memory_type": chunk.memory_type.value,
                    "source": chunk.source,
                    "tags": chunk.tags,
                    "importance_score": chunk.importance_score,
                    "timestamp": chunk.timestamp,
                    "last_accessed": chunk.last_accessed,
                    "access_count": chunk.access_count,
                    "metadata": chunk.metadata
                }
                memories.append(memory_data)
            
            # Create export data
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "export_version": "1.0",
                "memory_count": len(memories),
                "memories": memories
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Memory exported to JSON: {export_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting memory to JSON: {e}")
            return False
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        snapshots = []
        
        try:
            for snapshot_file in self.backup_dir.glob("*.zip"):
                try:
                    with zipfile.ZipFile(snapshot_file, 'r') as zipf:
                        if "snapshot_metadata.json" in zipf.namelist():
                            metadata_content = zipf.read("snapshot_metadata.json")
                            metadata = json.loads(metadata_content)
                        else:
                            metadata = {
                                "snapshot_name": snapshot_file.stem,
                                "created_at": datetime.fromtimestamp(snapshot_file.stat().st_mtime).isoformat()
                            }
                    
                    snapshot_info = {
                        "file_path": str(snapshot_file),
                        "file_size_mb": snapshot_file.stat().st_size / (1024 * 1024),
                        **metadata
                    }
                    
                    snapshots.append(snapshot_info)
                    
                except Exception as e:
                    logger.warning(f"Error reading snapshot {snapshot_file}: {e}")
            
            # Sort by creation date
            snapshots.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing snapshots: {e}")
        
        return snapshots

# Global instances
_config_manager = None
_snapshot_manager = None

def get_config_manager() -> ConfigManager:
    """Get or create a global config manager instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager

def get_snapshot_manager() -> MemorySnapshotManager:
    """Get or create a global snapshot manager instance."""
    global _snapshot_manager
    
    if _snapshot_manager is None:
        _snapshot_manager = MemorySnapshotManager()
    
    return _snapshot_manager
