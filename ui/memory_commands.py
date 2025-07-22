"""
Enhanced Memory Recall Commands for SAM
Chat and CLI commands for accessing specific memories during conversations.

Sprint 12 Task 4: Enhanced Memory Recall Commands (Chat + CLI)
"""

import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_vectorstore import MemoryVectorStore, MemoryType, get_memory_store
from memory.memory_reasoning import MemoryDrivenReasoningEngine, get_memory_reasoning_engine

logger = logging.getLogger(__name__)

@dataclass
class CommandResult:
    """Result of a memory command execution."""
    success: bool
    message: str
    data: Any = None
    command: str = ""
    execution_time_ms: int = 0

class MemoryCommandProcessor:
    """
    Processes memory recall commands in chat and CLI contexts.
    """
    
    def __init__(self):
        """Initialize the memory command processor."""
        self.memory_store = get_memory_store()
        self.memory_reasoning = get_memory_reasoning_engine()
        
        # Command patterns
        self.command_patterns = {
            'recall_topic': r'!recall\s+topic\s+(.+)',
            'recall_last': r'!recall\s+last\s+(\d+)',
            'search_mem': r'!searchmem\s+(.+)',
            'search_tag': r'!searchmem\s+tag:(\w+)',
            'search_source': r'!searchmem\s+source:(.+)',
            'search_date': r'!searchmem\s+date:(\d{4}-\d{2}-\d{2})',
            'search_user': r'!searchmem\s+user:(\w+)',
            'search_type': r'!searchmem\s+type:(\w+)',
            'memory_stats': r'!memstats',
            'memory_help': r'!memhelp'
        }
        
        # Command handlers
        self.command_handlers = {
            'recall_topic': self._handle_recall_topic,
            'recall_last': self._handle_recall_last,
            'search_mem': self._handle_search_memory,
            'search_tag': self._handle_search_tag,
            'search_source': self._handle_search_source,
            'search_date': self._handle_search_date,
            'search_user': self._handle_search_user,
            'search_type': self._handle_search_type,
            'memory_stats': self._handle_memory_stats,
            'memory_help': self._handle_memory_help
        }
    
    def process_command(self, command_text: str, user_id: str = None, 
                       output_format: str = "text") -> CommandResult:
        """
        Process a memory command.
        
        Args:
            command_text: The command text to process
            user_id: Optional user ID for filtering
            output_format: Output format ('text' or 'json')
            
        Returns:
            CommandResult with execution results
        """
        start_time = datetime.now()
        
        try:
            # Clean and normalize command
            command_text = command_text.strip()
            
            # Find matching command pattern
            for command_name, pattern in self.command_patterns.items():
                match = re.match(pattern, command_text, re.IGNORECASE)
                if match:
                    # Execute command handler
                    handler = self.command_handlers[command_name]
                    result = handler(match, user_id, output_format)
                    
                    # Calculate execution time
                    end_time = datetime.now()
                    execution_time = int((end_time - start_time).total_seconds() * 1000)
                    result.execution_time_ms = execution_time
                    result.command = command_name
                    
                    return result
            
            # No matching command found
            return CommandResult(
                success=False,
                message="Unknown command. Type !memhelp for available commands.",
                command="unknown"
            )
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return CommandResult(
                success=False,
                message=f"Error executing command: {str(e)}",
                command="error"
            )
    
    def get_available_commands(self) -> List[Dict[str, str]]:
        """Get list of available memory commands."""
        return [
            {
                'command': '!recall topic [keyword]',
                'description': 'Recall memories related to a specific topic',
                'example': '!recall topic artificial intelligence'
            },
            {
                'command': '!recall last N',
                'description': 'Recall the last N memories',
                'example': '!recall last 5'
            },
            {
                'command': '!searchmem [query]',
                'description': 'Search memories with a general query',
                'example': '!searchmem machine learning algorithms'
            },
            {
                'command': '!searchmem tag:[tag]',
                'description': 'Search memories by tag',
                'example': '!searchmem tag:important'
            },
            {
                'command': '!searchmem source:[source]',
                'description': 'Search memories by source',
                'example': '!searchmem source:research_paper'
            },
            {
                'command': '!searchmem date:[YYYY-MM-DD]',
                'description': 'Search memories by date',
                'example': '!searchmem date:2024-01-15'
            },
            {
                'command': '!searchmem user:[user_id]',
                'description': 'Search memories by user',
                'example': '!searchmem user:alice'
            },
            {
                'command': '!searchmem type:[type]',
                'description': 'Search memories by type',
                'example': '!searchmem type:conversation'
            },
            {
                'command': '!memstats',
                'description': 'Show memory statistics',
                'example': '!memstats'
            },
            {
                'command': '!memhelp',
                'description': 'Show this help message',
                'example': '!memhelp'
            }
        ]
    
    def _handle_recall_topic(self, match, user_id: str, output_format: str) -> CommandResult:
        """Handle !recall topic command."""
        try:
            topic = match.group(1).strip()
            
            # Search for memories related to the topic
            results = self.memory_reasoning.search_memories(
                query=topic,
                user_id=user_id,
                max_results=5
            )
            
            if not results:
                return CommandResult(
                    success=True,
                    message=f"No memories found for topic: {topic}",
                    data=[]
                )
            
            if output_format == "json":
                data = [
                    {
                        'chunk_id': r.chunk.chunk_id,
                        'content': r.chunk.content,
                        'similarity_score': r.similarity_score,
                        'memory_type': r.chunk.memory_type.value,
                        'source': r.chunk.source,
                        'timestamp': r.chunk.timestamp,
                        'tags': r.chunk.tags
                    }
                    for r in results
                ]
                
                return CommandResult(
                    success=True,
                    message=f"Found {len(results)} memories for topic: {topic}",
                    data=data
                )
            else:
                # Format as text
                message_parts = [f"📚 Found {len(results)} memories for topic: {topic}\n"]
                
                for i, result in enumerate(results, 1):
                    memory = result.chunk
                    message_parts.append(
                        f"{i}. **{memory.memory_type.value.title()}** ({result.similarity_score:.2f} similarity)\n"
                        f"   Source: {memory.source}\n"
                        f"   Date: {memory.timestamp[:10]}\n"
                        f"   Content: {memory.content[:150]}{'...' if len(memory.content) > 150 else ''}\n"
                    )
                
                return CommandResult(
                    success=True,
                    message="\n".join(message_parts),
                    data=results
                )
                
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error recalling topic: {str(e)}"
            )
    
    def _handle_recall_last(self, match, user_id: str, output_format: str) -> CommandResult:
        """Handle !recall last N command."""
        try:
            count = int(match.group(1))
            
            if count <= 0 or count > 50:
                return CommandResult(
                    success=False,
                    message="Count must be between 1 and 50"
                )
            
            # Get recent memories
            all_memories = list(self.memory_store.memory_chunks.values())
            
            # Filter by user if specified
            if user_id:
                all_memories = [
                    m for m in all_memories 
                    if m.metadata.get('user_id') == user_id
                ]
            
            # Sort by timestamp (newest first)
            all_memories.sort(key=lambda m: m.timestamp, reverse=True)
            recent_memories = all_memories[:count]
            
            if not recent_memories:
                return CommandResult(
                    success=True,
                    message="No recent memories found",
                    data=[]
                )
            
            if output_format == "json":
                data = [
                    {
                        'chunk_id': m.chunk_id,
                        'content': m.content,
                        'memory_type': m.memory_type.value,
                        'source': m.source,
                        'timestamp': m.timestamp,
                        'tags': m.tags,
                        'importance_score': m.importance_score
                    }
                    for m in recent_memories
                ]
                
                return CommandResult(
                    success=True,
                    message=f"Retrieved last {len(recent_memories)} memories",
                    data=data
                )
            else:
                # Format as text
                message_parts = [f"🕒 Last {len(recent_memories)} memories:\n"]
                
                for i, memory in enumerate(recent_memories, 1):
                    message_parts.append(
                        f"{i}. **{memory.memory_type.value.title()}** - {memory.source}\n"
                        f"   Date: {memory.timestamp[:10]}\n"
                        f"   Content: {memory.content[:100]}{'...' if len(memory.content) > 100 else ''}\n"
                    )
                
                return CommandResult(
                    success=True,
                    message="\n".join(message_parts),
                    data=recent_memories
                )
                
        except ValueError:
            return CommandResult(
                success=False,
                message="Invalid number format"
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error recalling last memories: {str(e)}"
            )
    
    def _handle_search_memory(self, match, user_id: str, output_format: str) -> CommandResult:
        """Handle !searchmem general query command."""
        try:
            query = match.group(1).strip()
            
            results = self.memory_reasoning.search_memories(
                query=query,
                user_id=user_id,
                max_results=10
            )
            
            return self._format_search_results(results, f"search query: {query}", output_format)
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error searching memories: {str(e)}"
            )
    
    def _handle_search_tag(self, match, user_id: str, output_format: str) -> CommandResult:
        """Handle !searchmem tag: command."""
        try:
            tag = match.group(1).strip()
            
            # Find memories with the specified tag
            matching_memories = []
            for memory in self.memory_store.memory_chunks.values():
                if (tag in memory.tags and 
                    (not user_id or memory.metadata.get('user_id') == user_id)):
                    matching_memories.append(memory)
            
            # Sort by importance and recency
            matching_memories.sort(key=lambda m: (m.importance_score, m.timestamp), reverse=True)
            
            return self._format_memory_list(matching_memories[:10], f"tag: {tag}", output_format)
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error searching by tag: {str(e)}"
            )
    
    def _handle_search_source(self, match, user_id: str, output_format: str) -> CommandResult:
        """Handle !searchmem source: command."""
        try:
            source = match.group(1).strip()
            
            # Find memories from the specified source
            matching_memories = []
            for memory in self.memory_store.memory_chunks.values():
                if (source.lower() in memory.source.lower() and 
                    (not user_id or memory.metadata.get('user_id') == user_id)):
                    matching_memories.append(memory)
            
            # Sort by timestamp (newest first)
            matching_memories.sort(key=lambda m: m.timestamp, reverse=True)
            
            return self._format_memory_list(matching_memories[:10], f"source: {source}", output_format)
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error searching by source: {str(e)}"
            )
    
    def _handle_search_date(self, match, user_id: str, output_format: str) -> CommandResult:
        """Handle !searchmem date: command."""
        try:
            date_str = match.group(1).strip()
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Find memories from the specified date
            matching_memories = []
            for memory in self.memory_store.memory_chunks.values():
                memory_date = datetime.fromisoformat(memory.timestamp).date()
                if (memory_date == target_date and 
                    (not user_id or memory.metadata.get('user_id') == user_id)):
                    matching_memories.append(memory)
            
            # Sort by timestamp
            matching_memories.sort(key=lambda m: m.timestamp)
            
            return self._format_memory_list(matching_memories, f"date: {date_str}", output_format)
            
        except ValueError:
            return CommandResult(
                success=False,
                message="Invalid date format. Use YYYY-MM-DD"
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error searching by date: {str(e)}"
            )
    
    def _handle_search_user(self, match, user_id: str, output_format: str) -> CommandResult:
        """Handle !searchmem user: command."""
        try:
            target_user = match.group(1).strip()
            
            # Find memories from the specified user
            matching_memories = []
            for memory in self.memory_store.memory_chunks.values():
                if memory.metadata.get('user_id') == target_user:
                    matching_memories.append(memory)
            
            # Sort by timestamp (newest first)
            matching_memories.sort(key=lambda m: m.timestamp, reverse=True)
            
            return self._format_memory_list(matching_memories[:20], f"user: {target_user}", output_format)
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error searching by user: {str(e)}"
            )
    
    def _handle_search_type(self, match, user_id: str, output_format: str) -> CommandResult:
        """Handle !searchmem type: command."""
        try:
            memory_type_str = match.group(1).strip().lower()
            
            # Validate memory type
            try:
                memory_type = MemoryType(memory_type_str)
            except ValueError:
                valid_types = [mt.value for mt in MemoryType]
                return CommandResult(
                    success=False,
                    message=f"Invalid memory type. Valid types: {', '.join(valid_types)}"
                )
            
            # Find memories of the specified type
            matching_memories = []
            for memory in self.memory_store.memory_chunks.values():
                if (memory.memory_type == memory_type and 
                    (not user_id or memory.metadata.get('user_id') == user_id)):
                    matching_memories.append(memory)
            
            # Sort by importance and timestamp
            matching_memories.sort(key=lambda m: (m.importance_score, m.timestamp), reverse=True)
            
            return self._format_memory_list(matching_memories[:15], f"type: {memory_type_str}", output_format)
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error searching by type: {str(e)}"
            )
    
    def _handle_memory_stats(self, match, user_id: str, output_format: str) -> CommandResult:
        """Handle !memstats command."""
        try:
            stats = self.memory_store.get_memory_stats()
            
            if output_format == "json":
                return CommandResult(
                    success=True,
                    message="Memory statistics retrieved",
                    data=stats
                )
            else:
                # Format as text
                message_parts = [
                    "📊 **Memory Statistics**\n",
                    f"Total Memories: {stats['total_memories']}",
                    f"Storage Size: {stats['total_size_mb']:.2f} MB",
                    f"Store Type: {stats['store_type']}"
                ]
                
                if stats['memory_types']:
                    message_parts.append("\n**Memory Types:**")
                    for mem_type, count in stats['memory_types'].items():
                        message_parts.append(f"  {mem_type}: {count}")
                
                if stats.get('oldest_memory'):
                    message_parts.append(f"\nOldest Memory: {stats['oldest_memory'][:10]}")
                if stats.get('newest_memory'):
                    message_parts.append(f"Newest Memory: {stats['newest_memory'][:10]}")
                
                return CommandResult(
                    success=True,
                    message="\n".join(message_parts),
                    data=stats
                )
                
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error getting memory statistics: {str(e)}"
            )
    
    def _handle_memory_help(self, match, user_id: str, output_format: str) -> CommandResult:
        """Handle !memhelp command."""
        try:
            commands = self.get_available_commands()
            
            if output_format == "json":
                return CommandResult(
                    success=True,
                    message="Memory commands help",
                    data=commands
                )
            else:
                # Format as text
                message_parts = ["🔍 **Memory Recall Commands**\n"]
                
                for cmd in commands:
                    message_parts.append(f"**{cmd['command']}**")
                    message_parts.append(f"  {cmd['description']}")
                    message_parts.append(f"  Example: `{cmd['example']}`\n")
                
                return CommandResult(
                    success=True,
                    message="\n".join(message_parts),
                    data=commands
                )
                
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error getting help: {str(e)}"
            )
    
    def _format_search_results(self, results, query_desc: str, output_format: str) -> CommandResult:
        """Format search results for display."""
        if not results:
            return CommandResult(
                success=True,
                message=f"No memories found for {query_desc}",
                data=[]
            )
        
        if output_format == "json":
            data = [
                {
                    'chunk_id': r.chunk.chunk_id,
                    'content': r.chunk.content,
                    'similarity_score': r.similarity_score,
                    'memory_type': r.chunk.memory_type.value,
                    'source': r.chunk.source,
                    'timestamp': r.chunk.timestamp,
                    'tags': r.chunk.tags
                }
                for r in results
            ]
            
            return CommandResult(
                success=True,
                message=f"Found {len(results)} memories for {query_desc}",
                data=data
            )
        else:
            # Format as text
            message_parts = [f"🔍 Found {len(results)} memories for {query_desc}:\n"]
            
            for i, result in enumerate(results, 1):
                memory = result.chunk
                message_parts.append(
                    f"{i}. **{memory.memory_type.value.title()}** ({result.similarity_score:.2f})\n"
                    f"   Source: {memory.source} | Date: {memory.timestamp[:10]}\n"
                    f"   Content: {memory.content[:120]}{'...' if len(memory.content) > 120 else ''}\n"
                )
            
            return CommandResult(
                success=True,
                message="\n".join(message_parts),
                data=results
            )
    
    def _format_memory_list(self, memories, filter_desc: str, output_format: str) -> CommandResult:
        """Format a list of memories for display."""
        if not memories:
            return CommandResult(
                success=True,
                message=f"No memories found for {filter_desc}",
                data=[]
            )
        
        if output_format == "json":
            data = [
                {
                    'chunk_id': m.chunk_id,
                    'content': m.content,
                    'memory_type': m.memory_type.value,
                    'source': m.source,
                    'timestamp': m.timestamp,
                    'tags': m.tags,
                    'importance_score': m.importance_score
                }
                for m in memories
            ]
            
            return CommandResult(
                success=True,
                message=f"Found {len(memories)} memories for {filter_desc}",
                data=data
            )
        else:
            # Format as text
            message_parts = [f"📋 Found {len(memories)} memories for {filter_desc}:\n"]
            
            for i, memory in enumerate(memories, 1):
                message_parts.append(
                    f"{i}. **{memory.memory_type.value.title()}** - {memory.source}\n"
                    f"   Date: {memory.timestamp[:10]} | Importance: {memory.importance_score:.2f}\n"
                    f"   Content: {memory.content[:100]}{'...' if len(memory.content) > 100 else ''}\n"
                )
            
            return CommandResult(
                success=True,
                message="\n".join(message_parts),
                data=memories
            )

# Global command processor instance
_command_processor = None

def get_command_processor() -> MemoryCommandProcessor:
    """Get or create a global memory command processor instance."""
    global _command_processor
    
    if _command_processor is None:
        _command_processor = MemoryCommandProcessor()
    
    return _command_processor
