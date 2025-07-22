#!/usr/bin/env python3
"""
SAM Pro Activation Key Generator

Generates new SAM Pro activation keys and updates the entitlements configuration.
This script creates UUID-format activation keys and their corresponding SHA-256 hashes.

Usage:
    python scripts/generate_pro_activation_key.py [--count N] [--output-keys]

Security Note:
- Generated keys are cryptographically secure UUIDs
- Hashes are stored in entitlements.json for validation
- Keys should be distributed securely to authorized users

Author: SAM Development Team
Version: 1.0.0
"""

import argparse
import hashlib
import json
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

def generate_activation_key() -> str:
    """
    Generate a cryptographically secure activation key in UUID format.
    
    Returns:
        UUID-format activation key
    """
    return str(uuid.uuid4())

def hash_activation_key(key: str) -> str:
    """
    Generate SHA-256 hash of activation key for storage.
    
    Args:
        key: The activation key to hash
        
    Returns:
        SHA-256 hash as hexadecimal string
    """
    return hashlib.sha256(key.encode('utf-8')).hexdigest()

def load_entitlements_config(config_path: Path) -> dict:
    """
    Load the current entitlements configuration.
    
    Args:
        config_path: Path to entitlements.json
        
    Returns:
        Entitlements configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Entitlements config not found at: {config_path}")
        print("Please ensure you're running this from the SAM root directory.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in entitlements config: {e}")
        sys.exit(1)

def save_entitlements_config(config: dict, config_path: Path) -> bool:
    """
    Save the updated entitlements configuration.
    
    Args:
        config: Updated configuration dictionary
        config_path: Path to entitlements.json
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create backup
        backup_path = config_path.with_suffix('.json.backup')
        if config_path.exists():
            import shutil
            shutil.copy2(config_path, backup_path)
            print(f"📋 Backup created: {backup_path}")
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save entitlements config: {e}")
        return False

def generate_keys(count: int) -> List[Tuple[str, str]]:
    """
    Generate multiple activation keys and their hashes.
    
    Args:
        count: Number of keys to generate
        
    Returns:
        List of (key, hash) tuples
    """
    keys = []
    
    print(f"🔑 Generating {count} SAM Pro activation key(s)...")
    
    for i in range(count):
        key = generate_activation_key()
        key_hash = hash_activation_key(key)
        keys.append((key, key_hash))
        
        if count <= 10:  # Show progress for small batches
            print(f"  Generated key {i+1}/{count}")
    
    return keys

def update_entitlements(keys: List[Tuple[str, str]], config_path: Path) -> bool:
    """
    Update entitlements configuration with new key hashes.
    
    Args:
        keys: List of (key, hash) tuples
        config_path: Path to entitlements.json
        
    Returns:
        True if successful, False otherwise
    """
    # Load current config
    config = load_entitlements_config(config_path)
    
    # Add new hashes
    if 'valid_key_hashes' not in config:
        config['valid_key_hashes'] = []
    
    new_hashes = [key_hash for _, key_hash in keys]
    config['valid_key_hashes'].extend(new_hashes)
    
    # Update metadata
    if 'metadata' not in config:
        config['metadata'] = {}
    
    config['metadata'].update({
        'last_updated': datetime.now().isoformat(),
        'total_keys': len(config['valid_key_hashes']),
        'hash_algorithm': 'SHA-256'
    })
    
    # Save updated config
    return save_entitlements_config(config, config_path)

def display_keys(keys: List[Tuple[str, str]], show_keys: bool = False):
    """
    Display generated keys and statistics.
    
    Args:
        keys: List of (key, hash) tuples
        show_keys: Whether to display the actual keys
    """
    print(f"\n✅ Successfully generated {len(keys)} activation key(s)")
    print(f"📊 Statistics:")
    print(f"  • Keys generated: {len(keys)}")
    print(f"  • Hash algorithm: SHA-256")
    print(f"  • Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if show_keys:
        print(f"\n🔑 Generated Activation Keys:")
        print("=" * 60)
        for i, (key, key_hash) in enumerate(keys, 1):
            print(f"Key {i:2d}: {key}")
            print(f"Hash  : {key_hash}")
            print("-" * 60)
        
        print("\n⚠️  SECURITY WARNING:")
        print("• Store these keys securely")
        print("• Distribute only to authorized users")
        print("• Keys cannot be recovered if lost")
        print("• Each key can only be used once")
    else:
        print(f"\n💡 Use --output-keys to display the generated keys")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate SAM Pro activation keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_pro_activation_key.py
  python scripts/generate_pro_activation_key.py --count 5
  python scripts/generate_pro_activation_key.py --count 10 --output-keys

Security Notes:
  - Keys are cryptographically secure UUIDs
  - Hashes are stored in sam/config/entitlements.json
  - Original keys are not stored anywhere
  - Distribute keys securely to authorized users
        """
    )
    
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=1,
        help='Number of keys to generate (default: 1)'
    )
    
    parser.add_argument(
        '--output-keys', '-o',
        action='store_true',
        help='Display the generated keys (security sensitive)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.count < 1:
        print("❌ Count must be at least 1")
        sys.exit(1)
    
    if args.count > 100:
        print("❌ Maximum 100 keys per batch for security reasons")
        sys.exit(1)
    
    # Check if we're in the SAM directory
    config_path = Path("sam/config/entitlements.json")
    if not config_path.exists():
        print("❌ SAM entitlements config not found")
        print("Please run this script from the SAM root directory")
        sys.exit(1)
    
    print("🚀 SAM Pro Activation Key Generator")
    print("=" * 50)
    
    # Generate keys
    keys = generate_keys(args.count)
    
    # Update entitlements configuration
    print(f"\n📝 Updating entitlements configuration...")
    if update_entitlements(keys, config_path):
        print(f"✅ Entitlements configuration updated successfully")
    else:
        print(f"❌ Failed to update entitlements configuration")
        sys.exit(1)
    
    # Display results
    display_keys(keys, args.output_keys)
    
    print(f"\n🎉 Key generation complete!")
    print(f"📁 Configuration updated: {config_path}")
    
    if not args.output_keys:
        print(f"\n💡 To see the generated keys, run:")
        print(f"   python scripts/generate_pro_activation_key.py --count {args.count} --output-keys")

if __name__ == "__main__":
    main()
