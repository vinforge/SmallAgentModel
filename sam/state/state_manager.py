"""
SAM State Manager - Centralized State Management
===============================================

Thread-safe state management system for SAM's persistent data with:
- Atomic read/write operations
- File locking for concurrent access
- Automatic backup and recovery
- Type-safe state operations
- Event notifications for state changes

Part of SAM's Task 27: Automated "Dream & Discover" Engine
Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, List
from dataclasses import dataclass
from enum import Enum
import fcntl
import tempfile
import shutil

logger = logging.getLogger(__name__)

# Type aliases for better type hints
StateKey = str
StateValue = Union[str, int, float, bool, dict, list, None]

class StateChangeEvent(Enum):
    """Types of state change events."""
    SET = "set"
    DELETE = "delete"
    CLEAR = "clear"

@dataclass
class StateChange:
    """Represents a state change event."""
    event_type: StateChangeEvent
    key: StateKey
    old_value: StateValue
    new_value: StateValue
    timestamp: str

class StateManager:
    """
    Thread-safe state manager for SAM's persistent data.
    
    Features:
    - Atomic file operations with locking
    - Automatic backup and recovery
    - Event notifications for state changes
    - Type validation and serialization
    - Concurrent access protection
    """
    
    def __init__(self, state_file: Optional[str] = None, backup_count: int = 5):
        """Initialize the state manager."""
        self.logger = logging.getLogger(__name__)
        
        # State file configuration
        if state_file:
            self.state_file = Path(state_file)
        else:
            # Use SAM's standard state directory
            state_dir = Path.home() / ".sam"
            state_dir.mkdir(exist_ok=True)
            self.state_file = state_dir / "sam_state.json"
        
        # Backup configuration
        self.backup_count = backup_count
        self.backup_dir = self.state_file.parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        self._file_lock = None
        
        # Event system
        self._change_listeners: List[Callable[[StateChange], None]] = []
        
        # In-memory cache
        self._cache: Dict[StateKey, StateValue] = {}
        self._cache_dirty = False
        
        # Initialize state file if it doesn't exist
        self._initialize_state_file()
        
        # Load initial state
        self._load_state()
        
        self.logger.info(f"StateManager initialized with file: {self.state_file}")
    
    def get(self, key: StateKey, default: StateValue = None) -> StateValue:
        """Get a value from state with optional default."""
        with self._lock:
            return self._cache.get(key, default)
    
    def set(self, key: StateKey, value: StateValue) -> None:
        """Set a value in state with atomic persistence."""
        with self._lock:
            old_value = self._cache.get(key)
            self._cache[key] = value
            self._cache_dirty = True
            
            # Persist to disk
            self._save_state()
            
            # Notify listeners
            self._notify_change(StateChangeEvent.SET, key, old_value, value)
            
            self.logger.debug(f"State set: {key} = {value}")
    
    def delete(self, key: StateKey) -> bool:
        """Delete a key from state."""
        with self._lock:
            if key in self._cache:
                old_value = self._cache[key]
                del self._cache[key]
                self._cache_dirty = True
                
                # Persist to disk
                self._save_state()
                
                # Notify listeners
                self._notify_change(StateChangeEvent.DELETE, key, old_value, None)
                
                self.logger.debug(f"State deleted: {key}")
                return True
            return False
    
    def update(self, updates: Dict[StateKey, StateValue]) -> None:
        """Update multiple state values atomically."""
        with self._lock:
            changes = []
            
            # Prepare changes
            for key, value in updates.items():
                old_value = self._cache.get(key)
                self._cache[key] = value
                changes.append((key, old_value, value))
            
            self._cache_dirty = True
            
            # Persist to disk
            self._save_state()
            
            # Notify listeners for all changes
            for key, old_value, new_value in changes:
                self._notify_change(StateChangeEvent.SET, key, old_value, new_value)
            
            self.logger.debug(f"State updated: {len(updates)} keys")
    
    def clear(self) -> None:
        """Clear all state data."""
        with self._lock:
            old_cache = self._cache.copy()
            self._cache.clear()
            self._cache_dirty = True
            
            # Persist to disk
            self._save_state()
            
            # Notify listeners
            for key, old_value in old_cache.items():
                self._notify_change(StateChangeEvent.DELETE, key, old_value, None)
            
            self.logger.info("State cleared")
    
    def get_all(self) -> Dict[StateKey, StateValue]:
        """Get all state data as a dictionary."""
        with self._lock:
            return self._cache.copy()
    
    def has_key(self, key: StateKey) -> bool:
        """Check if a key exists in state."""
        with self._lock:
            return key in self._cache
    
    def keys(self) -> List[StateKey]:
        """Get all state keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def add_change_listener(self, listener: Callable[[StateChange], None]) -> None:
        """Add a listener for state changes."""
        with self._lock:
            self._change_listeners.append(listener)
            self.logger.debug(f"Added state change listener: {listener}")
    
    def remove_change_listener(self, listener: Callable[[StateChange], None]) -> None:
        """Remove a state change listener."""
        with self._lock:
            if listener in self._change_listeners:
                self._change_listeners.remove(listener)
                self.logger.debug(f"Removed state change listener: {listener}")
    
    def reload(self) -> None:
        """Reload state from disk."""
        with self._lock:
            self._load_state()
            self.logger.info("State reloaded from disk")
    
    def backup(self) -> str:
        """Create a backup of the current state file."""
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"sam_state_{timestamp}.json"
            
            if self.state_file.exists():
                shutil.copy2(self.state_file, backup_file)
                
                # Clean up old backups
                self._cleanup_old_backups()
                
                self.logger.info(f"State backed up to: {backup_file}")
                return str(backup_file)
            else:
                self.logger.warning("No state file to backup")
                return ""
    
    def restore_backup(self, backup_file: str) -> bool:
        """Restore state from a backup file."""
        with self._lock:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                self.logger.error(f"Backup file not found: {backup_file}")
                return False
            
            try:
                # Validate backup file
                with open(backup_path, 'r') as f:
                    backup_data = json.load(f)
                
                # Create current backup before restore
                self.backup()
                
                # Restore from backup
                shutil.copy2(backup_path, self.state_file)
                self._load_state()
                
                self.logger.info(f"State restored from backup: {backup_file}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to restore backup: {e}")
                return False
    
    def _initialize_state_file(self) -> None:
        """Initialize state file if it doesn't exist."""
        if not self.state_file.exists():
            try:
                with open(self.state_file, 'w') as f:
                    json.dump({}, f, indent=2)
                self.logger.info(f"Initialized new state file: {self.state_file}")
            except Exception as e:
                self.logger.error(f"Failed to initialize state file: {e}")
                raise
    
    def _load_state(self) -> None:
        """Load state from disk with file locking."""
        try:
            if not self.state_file.exists():
                self._cache = {}
                return
            
            with open(self.state_file, 'r') as f:
                # Acquire file lock
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                    self._cache = data if isinstance(data, dict) else {}
                    self._cache_dirty = False
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            self.logger.debug(f"Loaded state: {len(self._cache)} keys")
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            self._cache = {}
    
    def _save_state(self) -> None:
        """Save state to disk with atomic write and file locking."""
        if not self._cache_dirty:
            return
        
        try:
            # Create temporary file for atomic write
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                dir=self.state_file.parent,
                prefix=f".{self.state_file.name}_",
                suffix=".tmp",
                delete=False
            )
            
            try:
                # Acquire file lock
                fcntl.flock(temp_file.fileno(), fcntl.LOCK_EX)
                
                # Write data
                json.dump(self._cache, temp_file, indent=2)
                temp_file.flush()
                
                # Atomic move
                shutil.move(temp_file.name, self.state_file)
                self._cache_dirty = False
                
                self.logger.debug(f"State saved: {len(self._cache)} keys")
                
            finally:
                # Clean up temp file if it still exists
                temp_path = Path(temp_file.name)
                if temp_path.exists():
                    temp_path.unlink()
        
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            raise
    
    def _notify_change(self, event_type: StateChangeEvent, key: StateKey, 
                      old_value: StateValue, new_value: StateValue) -> None:
        """Notify listeners of state changes."""
        if not self._change_listeners:
            return
        
        change = StateChange(
            event_type=event_type,
            key=key,
            old_value=old_value,
            new_value=new_value,
            timestamp=datetime.now().isoformat()
        )
        
        for listener in self._change_listeners:
            try:
                listener(change)
            except Exception as e:
                self.logger.error(f"State change listener error: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """Clean up old backup files."""
        try:
            backup_files = sorted(
                self.backup_dir.glob("sam_state_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Remove old backups beyond the limit
            for backup_file in backup_files[self.backup_count:]:
                backup_file.unlink()
                self.logger.debug(f"Removed old backup: {backup_file}")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")

# Global instance for easy access
_state_manager = None

def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
