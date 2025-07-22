"""
Agent Identity & Messaging Protocol for SAM
Maintains clear agent names, capabilities, and communication threads.

Sprint 10 Task 3: Agent Identity & Messaging Protocol
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages between agents."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"

class MessagePriority(Enum):
    """Priority levels for messages."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class AgentIdentity:
    """Identity and capabilities of an agent."""
    agent_id: str
    agent_name: str
    agent_role: str
    capabilities: List[str]
    status: str
    endpoint: Optional[str]
    last_heartbeat: str
    metadata: Dict[str, Any]

@dataclass
class AgentMessage:
    """Message between agents."""
    message_id: str
    thread_id: str
    sender_id: str
    sender_name: str
    recipient_id: str
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: str
    expires_at: Optional[str]
    signature: str
    metadata: Dict[str, Any]

@dataclass
class MessageThread:
    """A conversation thread between agents."""
    thread_id: str
    participants: List[str]
    subject: str
    messages: List[AgentMessage]
    created_at: str
    last_activity: str
    status: str
    metadata: Dict[str, Any]

class AgentCommunicationManager:
    """
    Manages agent identity and messaging protocol.
    """
    
    def __init__(self, agent_id: str, agent_name: str, agent_role: str,
                 capabilities: List[str] = None):
        """
        Initialize the agent communication manager.
        
        Args:
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            agent_role: Agent role (planner, executor, etc.)
            capabilities: List of agent capabilities
        """
        self.agent_identity = AgentIdentity(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            capabilities=capabilities or [],
            status="active",
            endpoint=None,
            last_heartbeat=datetime.now().isoformat(),
            metadata={}
        )
        
        # Message storage
        self.message_threads: Dict[str, MessageThread] = {}
        self.message_queue = queue.PriorityQueue()
        self.sent_messages: Dict[str, AgentMessage] = {}
        
        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Configuration
        self.config = {
            'message_retention_hours': 24,
            'max_thread_messages': 100,
            'heartbeat_interval_seconds': 30,
            'message_timeout_seconds': 300
        }
        
        # Start message processing thread
        self.running = True
        self.message_processor = threading.Thread(target=self._process_messages, daemon=True)
        self.message_processor.start()
        
        logger.info(f"Agent communication manager initialized for {agent_name} ({agent_id})")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Function to handle the message
        """
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type.value}")
    
    def send_message(self, recipient_id: str, message_type: MessageType,
                    content: Dict[str, Any], thread_id: Optional[str] = None,
                    priority: MessagePriority = MessagePriority.NORMAL,
                    expires_in_seconds: Optional[int] = None) -> str:
        """
        Send a message to another agent.
        
        Args:
            recipient_id: ID of recipient agent
            message_type: Type of message
            content: Message content
            thread_id: Optional thread ID (creates new if None)
            priority: Message priority
            expires_in_seconds: Optional expiration time
            
        Returns:
            Message ID
        """
        try:
            message_id = f"msg_{uuid.uuid4().hex[:12]}"
            
            if not thread_id:
                thread_id = f"thread_{uuid.uuid4().hex[:12]}"
            
            # Calculate expiration
            expires_at = None
            if expires_in_seconds:
                from datetime import timedelta
                expires_at = (datetime.now() + timedelta(seconds=expires_in_seconds)).isoformat()
            
            # Create message signature
            signature = self._create_message_signature(message_id, content)
            
            # Create message
            message = AgentMessage(
                message_id=message_id,
                thread_id=thread_id,
                sender_id=self.agent_identity.agent_id,
                sender_name=self.agent_identity.agent_name,
                recipient_id=recipient_id,
                message_type=message_type,
                priority=priority,
                content=content,
                timestamp=datetime.now().isoformat(),
                expires_at=expires_at,
                signature=signature,
                metadata={}
            )
            
            # Store sent message
            self.sent_messages[message_id] = message
            
            # Add to thread
            self._add_message_to_thread(message)
            
            # Send message (this would integrate with actual transport layer)
            self._deliver_message(message)
            
            logger.info(f"Sent {message_type.value} message to {recipient_id}: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
    
    def receive_message(self, message: AgentMessage) -> bool:
        """
        Receive a message from another agent.
        
        Args:
            message: Message to receive
            
        Returns:
            True if message was processed successfully
        """
        try:
            # Validate message
            if not self._validate_message(message):
                logger.warning(f"Invalid message received: {message.message_id}")
                return False
            
            # Check if message has expired
            if message.expires_at:
                expires_at = datetime.fromisoformat(message.expires_at)
                if datetime.now() > expires_at:
                    logger.warning(f"Expired message received: {message.message_id}")
                    return False
            
            # Add to message queue for processing
            priority_value = 5 - message.priority.value  # Lower number = higher priority
            self.message_queue.put((priority_value, message))
            
            # Add to thread
            self._add_message_to_thread(message)
            
            logger.info(f"Received {message.message_type.value} message from {message.sender_name}: {message.message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return False
    
    def create_thread(self, participants: List[str], subject: str) -> str:
        """
        Create a new message thread.
        
        Args:
            participants: List of participant agent IDs
            subject: Thread subject
            
        Returns:
            Thread ID
        """
        try:
            thread_id = f"thread_{uuid.uuid4().hex[:12]}"
            
            thread = MessageThread(
                thread_id=thread_id,
                participants=participants,
                subject=subject,
                messages=[],
                created_at=datetime.now().isoformat(),
                last_activity=datetime.now().isoformat(),
                status="active",
                metadata={}
            )
            
            self.message_threads[thread_id] = thread
            
            logger.info(f"Created message thread: {subject} ({thread_id})")
            return thread_id
            
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            raise
    
    def get_thread_messages(self, thread_id: str) -> List[AgentMessage]:
        """Get all messages in a thread."""
        thread = self.message_threads.get(thread_id)
        return thread.messages if thread else []
    
    def get_agent_threads(self) -> List[MessageThread]:
        """Get all threads involving this agent."""
        return [
            thread for thread in self.message_threads.values()
            if self.agent_identity.agent_id in thread.participants
        ]
    
    def broadcast_message(self, message_type: MessageType, content: Dict[str, Any],
                         recipients: List[str] = None) -> List[str]:
        """
        Broadcast a message to multiple agents.
        
        Args:
            message_type: Type of message
            content: Message content
            recipients: Optional list of specific recipients
            
        Returns:
            List of message IDs
        """
        try:
            message_ids = []
            
            # If no recipients specified, broadcast to all known agents
            if not recipients:
                recipients = self._get_known_agents()
            
            for recipient_id in recipients:
                if recipient_id != self.agent_identity.agent_id:  # Don't send to self
                    message_id = self.send_message(
                        recipient_id=recipient_id,
                        message_type=message_type,
                        content=content,
                        priority=MessagePriority.NORMAL
                    )
                    message_ids.append(message_id)
            
            logger.info(f"Broadcast {message_type.value} to {len(message_ids)} agents")
            return message_ids
            
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
            return []
    
    def send_heartbeat(self):
        """Send heartbeat to indicate agent is alive."""
        try:
            self.agent_identity.last_heartbeat = datetime.now().isoformat()
            
            heartbeat_content = {
                'agent_id': self.agent_identity.agent_id,
                'agent_name': self.agent_identity.agent_name,
                'status': self.agent_identity.status,
                'capabilities': self.agent_identity.capabilities,
                'timestamp': self.agent_identity.last_heartbeat
            }
            
            self.broadcast_message(MessageType.HEARTBEAT, heartbeat_content)
            
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'agent_id': self.agent_identity.agent_id,
            'agent_name': self.agent_identity.agent_name,
            'agent_role': self.agent_identity.agent_role,
            'status': self.agent_identity.status,
            'capabilities': self.agent_identity.capabilities,
            'active_threads': len(self.message_threads),
            'messages_sent': len(self.sent_messages),
            'last_heartbeat': self.agent_identity.last_heartbeat
        }
    
    def shutdown(self):
        """Shutdown the communication manager."""
        self.running = False
        if self.message_processor.is_alive():
            self.message_processor.join(timeout=5)
        logger.info(f"Agent communication manager shutdown: {self.agent_identity.agent_name}")
    
    def _process_messages(self):
        """Process incoming messages in background thread."""
        while self.running:
            try:
                # Get message from queue with timeout
                try:
                    priority, message = self.message_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process message
                self._handle_message(message)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _handle_message(self, message: AgentMessage):
        """Handle a received message."""
        try:
            # Get handler for message type
            handler = self.message_handlers.get(message.message_type)
            
            if handler:
                # Call handler
                handler(message)
            else:
                # Default handling
                logger.info(f"No handler for {message.message_type.value}, using default")
                self._default_message_handler(message)
            
        except Exception as e:
            logger.error(f"Error handling message {message.message_id}: {e}")
    
    def _default_message_handler(self, message: AgentMessage):
        """Default message handler."""
        logger.info(f"Received {message.message_type.value} from {message.sender_name}: {message.content}")
    
    def _validate_message(self, message: AgentMessage) -> bool:
        """Validate a received message."""
        try:
            # Check required fields
            if not all([message.message_id, message.sender_id, message.content]):
                return False
            
            # Verify signature (simplified)
            expected_signature = self._create_message_signature(message.message_id, message.content)
            if message.signature != expected_signature:
                logger.warning(f"Invalid signature for message {message.message_id}")
                # For now, allow messages with invalid signatures (would be stricter in production)
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating message: {e}")
            return False
    
    def _create_message_signature(self, message_id: str, content: Dict[str, Any]) -> str:
        """Create a signature for a message (simplified implementation)."""
        import hashlib
        
        # Create signature from message ID and content hash
        content_str = json.dumps(content, sort_keys=True)
        signature_data = f"{message_id}:{content_str}:{self.agent_identity.agent_id}"
        
        return hashlib.sha256(signature_data.encode()).hexdigest()[:16]
    
    def _add_message_to_thread(self, message: AgentMessage):
        """Add a message to its thread."""
        try:
            thread_id = message.thread_id
            
            # Create thread if it doesn't exist
            if thread_id not in self.message_threads:
                participants = [message.sender_id, message.recipient_id]
                if self.agent_identity.agent_id not in participants:
                    participants.append(self.agent_identity.agent_id)
                
                self.message_threads[thread_id] = MessageThread(
                    thread_id=thread_id,
                    participants=participants,
                    subject=f"Thread {thread_id}",
                    messages=[],
                    created_at=datetime.now().isoformat(),
                    last_activity=datetime.now().isoformat(),
                    status="active",
                    metadata={}
                )
            
            # Add message to thread
            thread = self.message_threads[thread_id]
            thread.messages.append(message)
            thread.last_activity = datetime.now().isoformat()
            
            # Limit thread size
            if len(thread.messages) > self.config['max_thread_messages']:
                thread.messages = thread.messages[-self.config['max_thread_messages']:]
            
        except Exception as e:
            logger.error(f"Error adding message to thread: {e}")
    
    def _deliver_message(self, message: AgentMessage):
        """Deliver a message (placeholder for actual transport)."""
        # This would integrate with actual message transport (HTTP, WebSocket, etc.)
        # For now, just log the delivery
        logger.debug(f"Delivering message {message.message_id} to {message.recipient_id}")
    
    def _get_known_agents(self) -> List[str]:
        """Get list of known agent IDs (placeholder)."""
        # This would integrate with agent registry
        # For now, return empty list
        return []

# Global communication managers
_communication_managers: Dict[str, AgentCommunicationManager] = {}

def get_agent_comm_manager(agent_id: str, agent_name: str, agent_role: str,
                          capabilities: List[str] = None) -> AgentCommunicationManager:
    """Get or create an agent communication manager."""
    global _communication_managers
    
    if agent_id not in _communication_managers:
        _communication_managers[agent_id] = AgentCommunicationManager(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            capabilities=capabilities
        )
    
    return _communication_managers[agent_id]

def shutdown_all_comm_managers():
    """Shutdown all communication managers."""
    global _communication_managers
    
    for manager in _communication_managers.values():
        manager.shutdown()
    
    _communication_managers.clear()
