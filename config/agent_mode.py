"""
Agent Mode Controller for SAM
Handles mode-switching logic between Solo Analyst and Collaborative Swarm modes.

Sprint 11 Task 1 & 2: Default Agent Mode & Swarm Unlock via Collaboration Key
"""

import logging
import json
import hmac
import hashlib
import base64
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class AgentMode(Enum):
    """Agent operation modes."""
    SOLO = "solo"
    COLLABORATIVE = "collab"

class KeyValidationResult(Enum):
    """Results of key validation."""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    MISSING = "missing"
    MALFORMED = "malformed"

@dataclass
class CollaborationKey:
    """Collaboration key structure."""
    key_id: str
    issued_at: str
    expires_at: str
    permissions: List[str]
    signature: str
    metadata: Dict[str, Any]

@dataclass
class AgentModeStatus:
    """Current agent mode status."""
    current_mode: AgentMode
    key_status: KeyValidationResult
    key_expires_at: Optional[str]
    enabled_capabilities: List[str]
    disabled_capabilities: List[str]
    last_mode_change: str
    uptime_seconds: int

class AgentModeController:
    """
    Controls SAM's operation mode between Solo Analyst and Collaborative Swarm.
    """
    
    def __init__(self, config_file: str = "config/sam_config.json",
                 key_file: str = "config/collab_key.json"):
        """
        Initialize the agent mode controller.
        
        Args:
            config_file: Path to SAM configuration file
            key_file: Path to collaboration key file
        """
        self.config_file = Path(config_file)
        self.key_file = Path(key_file)
        
        # Ensure config directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self.current_mode = AgentMode.SOLO
        self.collaboration_key: Optional[CollaborationKey] = None
        self.mode_start_time = datetime.now()
        
        # Configuration
        self.config = self._load_config()
        
        # Security settings
        self.security_config = {
            'hmac_secret': os.getenv('SAM_HMAC_SECRET', 'default_secret_change_in_production'),
            'key_expiry_hours': 24,
            'require_signature': True,
            'allowed_permissions': [
                'swarm_coordination',
                'task_delegation',
                'agent_communication',
                'distributed_execution',
                'voting_participation',
                'shared_context_access'
            ]
        }
        
        # Initialize mode
        self._initialize_mode()
        
        logger.info(f"Agent mode controller initialized: {self.current_mode.value}")
    
    def get_current_mode(self) -> AgentMode:
        """Get the current agent mode."""
        return self.current_mode
    
    def get_mode_status(self) -> AgentModeStatus:
        """Get detailed mode status."""
        try:
            key_status = self._validate_collaboration_key()
            
            # Determine enabled/disabled capabilities
            if self.current_mode == AgentMode.SOLO:
                enabled_capabilities = [
                    'multimodal_reasoning',
                    'semantic_search',
                    'chat_ui',
                    'document_processing',
                    'memory_persistence'
                ]
                disabled_capabilities = [
                    'task_delegation',
                    'voting_validation',
                    'critic_roles',
                    'remote_swarm',
                    'distributed_coordination'
                ]
            else:
                enabled_capabilities = [
                    'multimodal_reasoning',
                    'semantic_search',
                    'chat_ui',
                    'document_processing',
                    'memory_persistence',
                    'task_delegation',
                    'voting_validation',
                    'critic_roles',
                    'remote_swarm',
                    'distributed_coordination'
                ]
                disabled_capabilities = []
            
            uptime = int((datetime.now() - self.mode_start_time).total_seconds())
            
            return AgentModeStatus(
                current_mode=self.current_mode,
                key_status=key_status,
                key_expires_at=self.collaboration_key.expires_at if self.collaboration_key else None,
                enabled_capabilities=enabled_capabilities,
                disabled_capabilities=disabled_capabilities,
                last_mode_change=self.mode_start_time.isoformat(),
                uptime_seconds=uptime
            )
            
        except Exception as e:
            logger.error(f"Error getting mode status: {e}")
            return AgentModeStatus(
                current_mode=self.current_mode,
                key_status=KeyValidationResult.INVALID,
                key_expires_at=None,
                enabled_capabilities=[],
                disabled_capabilities=[],
                last_mode_change=self.mode_start_time.isoformat(),
                uptime_seconds=0
            )
    
    def check_capability_enabled(self, capability: str) -> bool:
        """
        Check if a specific capability is enabled in current mode.
        
        Args:
            capability: Capability to check
            
        Returns:
            True if capability is enabled
        """
        status = self.get_mode_status()
        return capability in status.enabled_capabilities
    
    def request_mode_change(self, target_mode: AgentMode, force: bool = False) -> bool:
        """
        Request a change to a different mode.
        
        Args:
            target_mode: Target mode to switch to
            force: Force mode change even if key is invalid
            
        Returns:
            True if mode change was successful
        """
        try:
            if target_mode == self.current_mode:
                logger.info(f"Already in {target_mode.value} mode")
                return True
            
            if target_mode == AgentMode.COLLABORATIVE:
                # Validate collaboration key
                key_status = self._validate_collaboration_key()
                
                if key_status != KeyValidationResult.VALID and not force:
                    logger.warning(f"Cannot switch to collaborative mode: {key_status.value}")
                    return False
                
                # Switch to collaborative mode
                self.current_mode = AgentMode.COLLABORATIVE
                self.mode_start_time = datetime.now()
                
                logger.info("Switched to collaborative swarm mode")
                
            elif target_mode == AgentMode.SOLO:
                # Switch to solo mode (always allowed)
                self.current_mode = AgentMode.SOLO
                self.mode_start_time = datetime.now()
                
                logger.info("Switched to solo analyst mode")
            
            # Update config
            self.config['mode'] = target_mode.value
            self._save_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Error changing mode to {target_mode.value}: {e}")
            return False
    
    def refresh_collaboration_key(self) -> KeyValidationResult:
        """
        Refresh and validate the collaboration key.
        
        Returns:
            Key validation result
        """
        try:
            self.collaboration_key = None
            return self._validate_collaboration_key()
            
        except Exception as e:
            logger.error(f"Error refreshing collaboration key: {e}")
            return KeyValidationResult.INVALID
    
    def generate_collaboration_key(self, permissions: List[str] = None,
                                 expires_in_hours: int = 24) -> str:
        """
        Generate a new collaboration key (for testing/admin use).
        
        Args:
            permissions: List of permissions to grant
            expires_in_hours: Key expiry time in hours
            
        Returns:
            Generated key ID
        """
        try:
            import uuid
            
            key_id = f"collab_{uuid.uuid4().hex[:12]}"
            issued_at = datetime.now()
            expires_at = issued_at + timedelta(hours=expires_in_hours)
            
            permissions = permissions or self.security_config['allowed_permissions']
            
            # Create key data
            key_data = {
                'key_id': key_id,
                'issued_at': issued_at.isoformat(),
                'expires_at': expires_at.isoformat(),
                'permissions': permissions,
                'metadata': {
                    'generated_by': 'sam_admin',
                    'purpose': 'swarm_collaboration'
                }
            }
            
            # Generate signature
            signature = self._generate_key_signature(key_data)
            key_data['signature'] = signature
            
            # Save key
            with open(self.key_file, 'w', encoding='utf-8') as f:
                json.dump(key_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated collaboration key: {key_id}")
            return key_id
            
        except Exception as e:
            logger.error(f"Error generating collaboration key: {e}")
            raise
    
    def revoke_collaboration_key(self) -> bool:
        """
        Revoke the current collaboration key.
        
        Returns:
            True if successful
        """
        try:
            # Remove key file
            if self.key_file.exists():
                self.key_file.unlink()
            
            # Clear in-memory key
            self.collaboration_key = None
            
            # Switch to solo mode
            self.request_mode_change(AgentMode.SOLO, force=True)
            
            logger.info("Collaboration key revoked")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking collaboration key: {e}")
            return False
    
    def _initialize_mode(self):
        """Initialize the agent mode based on configuration."""
        try:
            # Check configured mode
            configured_mode = self.config.get('mode', 'solo')
            
            if configured_mode == 'collab':
                # Try to switch to collaborative mode
                if not self.request_mode_change(AgentMode.COLLABORATIVE):
                    logger.warning("Failed to initialize collaborative mode, falling back to solo")
                    self.current_mode = AgentMode.SOLO
            else:
                self.current_mode = AgentMode.SOLO
            
        except Exception as e:
            logger.error(f"Error initializing mode: {e}")
            self.current_mode = AgentMode.SOLO
    
    def _validate_collaboration_key(self) -> KeyValidationResult:
        """Validate the collaboration key."""
        try:
            # Check if key file exists
            if not self.key_file.exists():
                return KeyValidationResult.MISSING
            
            # Load key
            try:
                with open(self.key_file, 'r', encoding='utf-8') as f:
                    key_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                return KeyValidationResult.MALFORMED
            
            # Validate key structure
            required_fields = ['key_id', 'issued_at', 'expires_at', 'permissions', 'signature']
            if not all(field in key_data for field in required_fields):
                return KeyValidationResult.MALFORMED
            
            # Check expiration
            try:
                expires_at = datetime.fromisoformat(key_data['expires_at'])
                if datetime.now() > expires_at:
                    return KeyValidationResult.EXPIRED
            except ValueError:
                return KeyValidationResult.MALFORMED
            
            # Validate signature
            if self.security_config['require_signature']:
                expected_signature = self._generate_key_signature(key_data)
                if not hmac.compare_digest(key_data['signature'], expected_signature):
                    return KeyValidationResult.INVALID
            
            # Create collaboration key object
            self.collaboration_key = CollaborationKey(
                key_id=key_data['key_id'],
                issued_at=key_data['issued_at'],
                expires_at=key_data['expires_at'],
                permissions=key_data['permissions'],
                signature=key_data['signature'],
                metadata=key_data.get('metadata', {})
            )
            
            return KeyValidationResult.VALID
            
        except Exception as e:
            logger.error(f"Error validating collaboration key: {e}")
            return KeyValidationResult.INVALID
    
    def _generate_key_signature(self, key_data: Dict[str, Any]) -> str:
        """Generate HMAC signature for key data."""
        try:
            # Create signature payload (exclude signature field)
            payload_data = {k: v for k, v in key_data.items() if k != 'signature'}
            payload = json.dumps(payload_data, sort_keys=True)
            
            # Generate HMAC
            secret = self.security_config['hmac_secret'].encode()
            signature = hmac.new(secret, payload.encode(), hashlib.sha256).digest()
            
            return base64.b64encode(signature).decode()
            
        except Exception as e:
            logger.error(f"Error generating key signature: {e}")
            return ""
    
    def _load_config(self) -> Dict[str, Any]:
        """Load SAM configuration."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create default configuration
                default_config = {
                    'mode': 'solo',
                    'collab_key': str(self.key_file),
                    'vector_store': 'simple',
                    'memory_enabled': True,
                    'auto_memory_update': True,
                    'security': {
                        'require_key_signature': True,
                        'key_expiry_hours': 24
                    }
                }
                
                self._save_config(default_config)
                return default_config
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _save_config(self, config: Dict[str, Any] = None):
        """Save SAM configuration."""
        try:
            config_to_save = config or self.config
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving config: {e}")

# Global agent mode controller instance
_mode_controller = None

def get_mode_controller(config_file: str = "config/sam_config.json",
                       key_file: str = "config/collab_key.json") -> AgentModeController:
    """Get or create a global agent mode controller instance."""
    global _mode_controller
    
    if _mode_controller is None:
        _mode_controller = AgentModeController(
            config_file=config_file,
            key_file=key_file
        )
    
    return _mode_controller
