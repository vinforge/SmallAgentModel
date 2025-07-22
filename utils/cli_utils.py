"""
CLI Utilities for SAM Administration
Provides command-line interface for mode switching, memory management, and system status.

Sprint 11 Task 6: Admin Toggle (CLI)
"""

import logging
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.agent_mode import AgentModeController, AgentMode, KeyValidationResult, get_mode_controller
from memory.memory_vectorstore import MemoryVectorStore, VectorStoreType, MemoryType, get_memory_store
from memory.memory_reasoning import MemoryDrivenReasoningEngine, get_memory_reasoning_engine

logger = logging.getLogger(__name__)

class SAMAdminCLI:
    """
    Command-line interface for SAM administration.
    """
    
    def __init__(self):
        """Initialize the SAM admin CLI."""
        self.mode_controller = get_mode_controller()
        self.memory_store = get_memory_store()
        self.memory_reasoning = get_memory_reasoning_engine()
        
        # CLI configuration
        self.config = {
            'verbose': False,
            'json_output': False
        }
    
    def run(self, args: List[str] = None):
        """Run the CLI with provided arguments."""
        try:
            parser = self._create_parser()
            parsed_args = parser.parse_args(args)
            
            # Set global options
            self.config['verbose'] = parsed_args.verbose
            self.config['json_output'] = parsed_args.json
            
            # Execute command
            if hasattr(parsed_args, 'func'):
                parsed_args.func(parsed_args)
            else:
                parser.print_help()
                
        except Exception as e:
            self._error(f"CLI error: {e}")
            sys.exit(1)
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="SAM Administration CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  sam-admin status                    # Show system status
  sam-admin mode --set solo          # Switch to solo mode
  sam-admin mode --set collab        # Switch to collaborative mode
  sam-admin key --generate           # Generate collaboration key
  sam-admin memory --stats           # Show memory statistics
  sam-admin memory --search "AI"     # Search memories
  sam-admin memory --clear --older-than 30  # Clear old memories
            """
        )
        
        # Global options
        parser.add_argument('-v', '--verbose', action='store_true',
                          help='Enable verbose output')
        parser.add_argument('-j', '--json', action='store_true',
                          help='Output in JSON format')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show system status')
        status_parser.set_defaults(func=self._cmd_status)
        
        # Mode command
        mode_parser = subparsers.add_parser('mode', help='Manage agent mode')
        mode_parser.add_argument('--set', choices=['solo', 'collab'],
                               help='Set agent mode')
        mode_parser.add_argument('--force', action='store_true',
                               help='Force mode change even if key is invalid')
        mode_parser.set_defaults(func=self._cmd_mode)
        
        # Key command
        key_parser = subparsers.add_parser('key', help='Manage collaboration key')
        key_parser.add_argument('--generate', action='store_true',
                              help='Generate new collaboration key')
        key_parser.add_argument('--revoke', action='store_true',
                              help='Revoke current collaboration key')
        key_parser.add_argument('--validate', action='store_true',
                              help='Validate current collaboration key')
        key_parser.add_argument('--expires-in', type=int, default=24,
                              help='Key expiry time in hours (default: 24)')
        key_parser.set_defaults(func=self._cmd_key)
        
        # Memory command
        memory_parser = subparsers.add_parser('memory', help='Manage memory store')
        memory_parser.add_argument('--stats', action='store_true',
                                 help='Show memory statistics')
        memory_parser.add_argument('--search', type=str,
                                 help='Search memories')
        memory_parser.add_argument('--clear', action='store_true',
                                 help='Clear memories')
        memory_parser.add_argument('--export', type=str,
                                 help='Export memories to file')
        memory_parser.add_argument('--import', type=str, dest='import_file',
                                 help='Import memories from file')
        memory_parser.add_argument('--older-than', type=int,
                                 help='Filter by age in days')
        memory_parser.add_argument('--type', choices=['document', 'conversation', 'reasoning', 'insight', 'fact', 'procedure'],
                                 help='Filter by memory type')
        memory_parser.add_argument('--user', type=str,
                                 help='Filter by user ID')
        memory_parser.set_defaults(func=self._cmd_memory)
        
        return parser
    
    def _cmd_status(self, args):
        """Show system status."""
        try:
            # Get mode status
            mode_status = self.mode_controller.get_mode_status()
            
            # Get memory stats
            memory_stats = self.memory_store.get_memory_stats()
            
            # Get memory reasoning stats
            memory_summary = self.memory_reasoning.get_memory_summary()
            
            if self.config['json_output']:
                import json
                status_data = {
                    'mode': {
                        'current_mode': mode_status.current_mode.value,
                        'key_status': mode_status.key_status.value,
                        'key_expires_at': mode_status.key_expires_at,
                        'uptime_seconds': mode_status.uptime_seconds,
                        'enabled_capabilities': mode_status.enabled_capabilities,
                        'disabled_capabilities': mode_status.disabled_capabilities
                    },
                    'memory': memory_stats,
                    'timestamp': datetime.now().isoformat()
                }
                print(json.dumps(status_data, indent=2))
            else:
                self._print_status_table(mode_status, memory_stats, memory_summary)
                
        except Exception as e:
            self._error(f"Error getting status: {e}")
    
    def _cmd_mode(self, args):
        """Manage agent mode."""
        try:
            if args.set:
                target_mode = AgentMode.SOLO if args.set == 'solo' else AgentMode.COLLABORATIVE
                
                success = self.mode_controller.request_mode_change(target_mode, force=args.force)
                
                if success:
                    self._success(f"Switched to {target_mode.value} mode")
                else:
                    self._error(f"Failed to switch to {target_mode.value} mode")
            else:
                # Show current mode
                status = self.mode_controller.get_mode_status()
                
                if self.config['json_output']:
                    import json
                    print(json.dumps({
                        'current_mode': status.current_mode.value,
                        'key_status': status.key_status.value,
                        'uptime_seconds': status.uptime_seconds
                    }, indent=2))
                else:
                    self._info(f"Current mode: {status.current_mode.value}")
                    self._info(f"Key status: {status.key_status.value}")
                    self._info(f"Uptime: {status.uptime_seconds} seconds")
                    
        except Exception as e:
            self._error(f"Error managing mode: {e}")
    
    def _cmd_key(self, args):
        """Manage collaboration key."""
        try:
            if args.generate:
                key_id = self.mode_controller.generate_collaboration_key(
                    expires_in_hours=args.expires_in
                )
                self._success(f"Generated collaboration key: {key_id}")
                
            elif args.revoke:
                success = self.mode_controller.revoke_collaboration_key()
                if success:
                    self._success("Collaboration key revoked")
                else:
                    self._error("Failed to revoke collaboration key")
                    
            elif args.validate:
                key_status = self.mode_controller.refresh_collaboration_key()
                
                if self.config['json_output']:
                    import json
                    print(json.dumps({'key_status': key_status.value}, indent=2))
                else:
                    self._info(f"Key validation result: {key_status.value}")
                    
            else:
                # Show key status
                status = self.mode_controller.get_mode_status()
                
                if self.config['json_output']:
                    import json
                    print(json.dumps({
                        'key_status': status.key_status.value,
                        'expires_at': status.key_expires_at
                    }, indent=2))
                else:
                    self._info(f"Key status: {status.key_status.value}")
                    if status.key_expires_at:
                        self._info(f"Expires at: {status.key_expires_at}")
                        
        except Exception as e:
            self._error(f"Error managing key: {e}")
    
    def _cmd_memory(self, args):
        """Manage memory store."""
        try:
            if args.stats:
                stats = self.memory_store.get_memory_stats()
                
                if self.config['json_output']:
                    import json
                    print(json.dumps(stats, indent=2))
                else:
                    self._print_memory_stats(stats)
                    
            elif args.search:
                # Parse memory type filter
                memory_types = None
                if args.type:
                    memory_types = [MemoryType(args.type)]
                
                results = self.memory_reasoning.search_memories(
                    query=args.search,
                    user_id=args.user,
                    memory_types=memory_types,
                    max_results=10
                )
                
                if self.config['json_output']:
                    import json
                    results_data = [
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
                    print(json.dumps(results_data, indent=2))
                else:
                    self._print_search_results(results)
                    
            elif args.clear:
                # Parse filters
                memory_types = None
                if args.type:
                    memory_types = [MemoryType(args.type)]
                
                cleared_count = self.memory_store.clear_memories(
                    memory_types=memory_types,
                    older_than_days=args.older_than
                )
                
                self._success(f"Cleared {cleared_count} memories")
                
            elif args.export:
                success = self.memory_store.export_memories(args.export)
                if success:
                    self._success(f"Exported memories to {args.export}")
                else:
                    self._error(f"Failed to export memories to {args.export}")
                    
            elif args.import_file:
                imported_count = self.memory_store.import_memories(args.import_file)
                self._success(f"Imported {imported_count} memories from {args.import_file}")
                
            else:
                # Show memory summary
                summary = self.memory_reasoning.get_memory_summary(args.user)
                
                if self.config['json_output']:
                    import json
                    print(json.dumps(summary, indent=2))
                else:
                    self._print_memory_summary(summary)
                    
        except Exception as e:
            self._error(f"Error managing memory: {e}")
    
    def _print_status_table(self, mode_status, memory_stats, memory_summary):
        """Print formatted status table."""
        print("=" * 60)
        print("SAM System Status")
        print("=" * 60)
        
        # Mode status
        print(f"Agent Mode: {mode_status.current_mode.value.upper()}")
        print(f"Key Status: {mode_status.key_status.value}")
        if mode_status.key_expires_at:
            print(f"Key Expires: {mode_status.key_expires_at}")
        print(f"Uptime: {mode_status.uptime_seconds} seconds")
        
        print("\nEnabled Capabilities:")
        for capability in mode_status.enabled_capabilities:
            print(f"  ✅ {capability}")
        
        if mode_status.disabled_capabilities:
            print("\nDisabled Capabilities:")
            for capability in mode_status.disabled_capabilities:
                print(f"  ❌ {capability}")
        
        # Memory status
        print(f"\nMemory Store: {memory_stats['store_type']}")
        print(f"Total Memories: {memory_stats['total_memories']}")
        print(f"Storage Size: {memory_stats['total_size_mb']:.2f} MB")
        
        if memory_stats['memory_types']:
            print("\nMemory Types:")
            for mem_type, count in memory_stats['memory_types'].items():
                print(f"  {mem_type}: {count}")
        
        print("=" * 60)
    
    def _print_memory_stats(self, stats):
        """Print formatted memory statistics."""
        print("=" * 50)
        print("Memory Store Statistics")
        print("=" * 50)
        
        print(f"Store Type: {stats['store_type']}")
        print(f"Total Memories: {stats['total_memories']}")
        print(f"Storage Size: {stats['total_size_mb']:.2f} MB")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        
        if stats['oldest_memory']:
            print(f"Oldest Memory: {stats['oldest_memory'][:10]}")
        if stats['newest_memory']:
            print(f"Newest Memory: {stats['newest_memory'][:10]}")
        
        if stats['memory_types']:
            print("\nMemory Types:")
            for mem_type, count in stats['memory_types'].items():
                print(f"  {mem_type}: {count}")
        
        if stats['most_accessed']:
            print(f"\nMost Accessed: {stats['most_accessed']['access_count']} times")
            print(f"  Content: {stats['most_accessed']['content_preview']}...")
        
        print("=" * 50)
    
    def _print_search_results(self, results):
        """Print formatted search results."""
        if not results:
            print("No memories found.")
            return
        
        print(f"Found {len(results)} memories:")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            memory = result.chunk
            print(f"{i}. [{memory.memory_type.value}] {memory.source}")
            print(f"   Similarity: {result.similarity_score:.1%}")
            print(f"   Date: {memory.timestamp[:10]}")
            print(f"   Content: {memory.content[:100]}...")
            if memory.tags:
                print(f"   Tags: {', '.join(memory.tags[:3])}")
            print()
    
    def _print_memory_summary(self, summary):
        """Print formatted memory summary."""
        print("=" * 40)
        print("Memory Summary")
        print("=" * 40)
        
        print(f"Total Memories: {summary['total_memories']}")
        if summary.get('user_memories') is not None:
            print(f"User Memories: {summary['user_memories']}")
        print(f"Storage Size: {summary['storage_size_mb']:.2f} MB")
        print(f"Store Type: {summary['store_type']}")
        
        if summary['memory_types']:
            print("\nMemory Types:")
            for mem_type, count in summary['memory_types'].items():
                print(f"  {mem_type}: {count}")
        
        print("=" * 40)
    
    def _info(self, message: str):
        """Print info message."""
        if self.config['verbose'] or not self.config['json_output']:
            print(f"ℹ️  {message}")
    
    def _success(self, message: str):
        """Print success message."""
        if not self.config['json_output']:
            print(f"✅ {message}")
    
    def _error(self, message: str):
        """Print error message."""
        if not self.config['json_output']:
            print(f"❌ {message}", file=sys.stderr)
        else:
            import json
            print(json.dumps({'error': message}), file=sys.stderr)

def main():
    """Main CLI entry point."""
    cli = SAMAdminCLI()
    cli.run()

if __name__ == "__main__":
    main()
