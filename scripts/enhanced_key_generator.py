#!/usr/bin/env python3
"""
Enhanced SAM Pro Key Generator with Distribution Tracking

Generates SAM Pro activation keys with proper key-hash tracking for distribution.
Maintains a secure key database for the distribution system.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import json
import uuid
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedKeyGenerator:
    """
    Enhanced key generator with distribution tracking.
    
    Features:
    - Generates UUID-format activation keys
    - Creates SHA-256 hashes for validation
    - Maintains secure key-hash database
    - Tracks key generation metadata
    - Integrates with distribution system
    """
    
    def __init__(self):
        """Initialize the enhanced key generator."""
        self.entitlements_path = Path("sam/config/entitlements.json")
        self.key_database_path = Path("data/key_database.json")
        
        # Create data directory
        self.key_database_path.parent.mkdir(exist_ok=True)
        
        # Load existing key database
        self.key_database = self._load_key_database()
        
        logger.info("Enhanced key generator initialized")
    
    def _load_key_database(self) -> Dict:
        """Load the key database."""
        if self.key_database_path.exists():
            try:
                with open(self.key_database_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading key database: {e}")
                return self._create_empty_database()
        else:
            return self._create_empty_database()
    
    def _create_empty_database(self) -> Dict:
        """Create empty key database structure."""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_keys_generated": 0,
            "keys": {}
        }
    
    def _save_key_database(self):
        """Save the key database."""
        try:
            self.key_database["last_updated"] = datetime.now().isoformat()
            with open(self.key_database_path, 'w') as f:
                json.dump(self.key_database, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving key database: {e}")
    
    def generate_activation_key(self) -> str:
        """Generate a cryptographically secure UUID activation key."""
        return str(uuid.uuid4())
    
    def hash_activation_key(self, key: str) -> str:
        """Generate SHA-256 hash of activation key."""
        return hashlib.sha256(key.encode('utf-8')).hexdigest()
    
    def generate_keys_batch(self, count: int, batch_name: str = "") -> List[Tuple[str, str]]:
        """
        Generate a batch of activation keys.
        
        Args:
            count: Number of keys to generate
            batch_name: Optional batch identifier
            
        Returns:
            List of (key, hash) tuples
        """
        if count < 1 or count > 1000:
            raise ValueError("Count must be between 1 and 1000")
        
        logger.info(f"Generating batch of {count} keys...")
        
        keys = []
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for i in range(count):
            key = self.generate_activation_key()
            key_hash = self.hash_activation_key(key)
            
            # Store in database
            key_id = f"key_{len(self.key_database['keys']) + 1:06d}"
            self.key_database["keys"][key_hash] = {
                "key_id": key_id,
                "key_uuid": key,
                "key_hash": key_hash,
                "generated_at": datetime.now().isoformat(),
                "batch_id": batch_id,
                "batch_name": batch_name,
                "status": "available",
                "distributed_to": None,
                "distributed_at": None,
                "activated": False,
                "activation_date": None
            }
            
            keys.append((key, key_hash))
            
            if count <= 20:  # Show progress for small batches
                logger.info(f"Generated key {i+1}/{count}: {key_id}")
        
        # Update database metadata
        self.key_database["total_keys_generated"] += count
        self.key_database["last_batch"] = {
            "batch_id": batch_id,
            "batch_name": batch_name,
            "count": count,
            "generated_at": datetime.now().isoformat()
        }
        
        # Save database
        self._save_key_database()
        
        logger.info(f"Generated {count} keys in batch: {batch_id}")
        return keys
    
    def update_entitlements_config(self, key_hashes: List[str]) -> bool:
        """
        Update the entitlements configuration with new key hashes.
        
        Args:
            key_hashes: List of key hashes to add
            
        Returns:
            True if successful
        """
        try:
            # Load existing entitlements
            if self.entitlements_path.exists():
                with open(self.entitlements_path, 'r') as f:
                    config = json.load(f)
            else:
                logger.error("Entitlements config not found")
                return False
            
            # Add new hashes
            existing_hashes = set(config.get("valid_key_hashes", []))
            new_hashes = [h for h in key_hashes if h not in existing_hashes]
            
            if new_hashes:
                config["valid_key_hashes"] = list(existing_hashes) + new_hashes
                
                # Update metadata
                config["metadata"]["total_keys"] = len(config["valid_key_hashes"])
                config["metadata"]["last_updated"] = datetime.now().isoformat()
                
                # Save config
                with open(self.entitlements_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info(f"Added {len(new_hashes)} new key hashes to entitlements")
                return True
            else:
                logger.info("No new hashes to add")
                return True
                
        except Exception as e:
            logger.error(f"Error updating entitlements: {e}")
            return False
    
    def get_available_keys(self) -> List[Dict]:
        """Get list of available (undistributed) keys."""
        available = []
        for key_hash, key_data in self.key_database["keys"].items():
            if key_data["status"] == "available":
                available.append({
                    "key_id": key_data["key_id"],
                    "key_hash": key_hash,
                    "generated_at": key_data["generated_at"],
                    "batch_id": key_data["batch_id"]
                })
        return available
    
    def mark_key_distributed(self, key_hash: str, distributed_to: str) -> bool:
        """
        Mark a key as distributed.
        
        Args:
            key_hash: Hash of the distributed key
            distributed_to: Email address of recipient
            
        Returns:
            True if successful
        """
        try:
            if key_hash in self.key_database["keys"]:
                self.key_database["keys"][key_hash].update({
                    "status": "distributed",
                    "distributed_to": distributed_to,
                    "distributed_at": datetime.now().isoformat()
                })
                self._save_key_database()
                logger.info(f"Marked key as distributed: {key_hash[:16]}... to {distributed_to}")
                return True
            else:
                logger.error(f"Key hash not found: {key_hash}")
                return False
        except Exception as e:
            logger.error(f"Error marking key as distributed: {e}")
            return False
    
    def mark_key_activated(self, key_hash: str) -> bool:
        """
        Mark a key as activated.
        
        Args:
            key_hash: Hash of the activated key
            
        Returns:
            True if successful
        """
        try:
            if key_hash in self.key_database["keys"]:
                self.key_database["keys"][key_hash].update({
                    "activated": True,
                    "activation_date": datetime.now().isoformat()
                })
                self._save_key_database()
                logger.info(f"Marked key as activated: {key_hash[:16]}...")
                return True
            else:
                logger.error(f"Key hash not found: {key_hash}")
                return False
        except Exception as e:
            logger.error(f"Error marking key as activated: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get key generation and distribution statistics."""
        try:
            total_keys = len(self.key_database["keys"])
            available_keys = sum(1 for k in self.key_database["keys"].values() if k["status"] == "available")
            distributed_keys = sum(1 for k in self.key_database["keys"].values() if k["status"] == "distributed")
            activated_keys = sum(1 for k in self.key_database["keys"].values() if k["activated"])
            
            return {
                "total_generated": total_keys,
                "available": available_keys,
                "distributed": distributed_keys,
                "activated": activated_keys,
                "activation_rate": (activated_keys / distributed_keys * 100) if distributed_keys > 0 else 0,
                "database_version": self.key_database["version"],
                "last_updated": self.key_database["last_updated"]
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced SAM Pro Key Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 keys for initial distribution
  python scripts/enhanced_key_generator.py generate --count 10 --batch-name "initial_release"
  
  # Generate keys and show them
  python scripts/enhanced_key_generator.py generate --count 5 --show-keys
  
  # View statistics
  python scripts/enhanced_key_generator.py stats
  
  # List available keys
  python scripts/enhanced_key_generator.py list-available
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate new activation keys')
    gen_parser.add_argument('--count', '-c', type=int, default=10, help='Number of keys to generate')
    gen_parser.add_argument('--batch-name', '-b', help='Batch name for organization')
    gen_parser.add_argument('--show-keys', '-s', action='store_true', help='Display generated keys (security sensitive)')
    
    # Statistics command
    subparsers.add_parser('stats', help='Show key statistics')
    
    # List available command
    list_parser = subparsers.add_parser('list-available', help='List available keys')
    list_parser.add_argument('--limit', type=int, default=20, help='Number of keys to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize generator
    generator = EnhancedKeyGenerator()
    
    if args.command == 'generate':
        try:
            keys = generator.generate_keys_batch(args.count, args.batch_name or "")
            
            # Update entitlements
            key_hashes = [key_hash for _, key_hash in keys]
            if generator.update_entitlements_config(key_hashes):
                print(f"✅ Generated {len(keys)} keys and updated entitlements")
            else:
                print(f"⚠️ Generated {len(keys)} keys but failed to update entitlements")
            
            if args.show_keys:
                print(f"\n🔑 Generated Keys:")
                print("=" * 80)
                for i, (key, key_hash) in enumerate(keys, 1):
                    print(f"Key {i:2d}: {key}")
                    print(f"Hash  : {key_hash}")
                    print("-" * 80)
                
                print("\n⚠️ SECURITY WARNING:")
                print("• Store these keys securely")
                print("• Distribute only to authorized users")
                print("• Keys cannot be recovered if lost")
            else:
                print("\n💡 Use --show-keys to display the generated keys")
        
        except Exception as e:
            print(f"❌ Error generating keys: {e}")
    
    elif args.command == 'stats':
        stats = generator.get_statistics()
        print("\n📊 Key Generation Statistics")
        print("=" * 40)
        print(f"Total Generated: {stats['total_generated']}")
        print(f"Available: {stats['available']}")
        print(f"Distributed: {stats['distributed']}")
        print(f"Activated: {stats['activated']}")
        print(f"Activation Rate: {stats['activation_rate']:.1f}%")
        print(f"Last Updated: {stats['last_updated']}")
    
    elif args.command == 'list-available':
        available = generator.get_available_keys()
        print(f"\n📋 Available Keys (showing {min(len(available), args.limit)})")
        print("=" * 60)
        for key in available[:args.limit]:
            print(f"{key['key_id']} | {key['key_hash'][:16]}... | {key['generated_at']}")

if __name__ == "__main__":
    main()
