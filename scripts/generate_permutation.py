#!/usr/bin/env python3
"""
Generate Permutation Matrix for MEMOIR TopHashFingerprinter

This script generates and saves a fixed random permutation matrix
used by the TopHashFingerprinter for deterministic mask generation.

Usage:
    python scripts/generate_permutation.py --hidden_size 4096 --output data/permutation_4096.pkl

Author: SAM Development Team
Version: 1.0.0
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def generate_permutation_matrix(hidden_size: int, seed: int = 42) -> np.ndarray:
    """
    Generate a random permutation matrix.
    
    Args:
        hidden_size: Size of the permutation matrix
        seed: Random seed for reproducibility
        
    Returns:
        Permutation matrix as numpy array
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating permutation matrix of size {hidden_size} with seed {seed}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate permutation
    permutation = np.random.permutation(hidden_size)
    
    # Validate permutation
    assert len(permutation) == hidden_size
    assert len(set(permutation)) == hidden_size  # All unique values
    assert min(permutation) == 0 and max(permutation) == hidden_size - 1
    
    logger.info("Permutation matrix generated and validated")
    return permutation

def save_permutation_matrix(
    permutation: np.ndarray, 
    output_path: str, 
    hidden_size: int, 
    seed: int
):
    """
    Save permutation matrix to file.
    
    Args:
        permutation: Permutation matrix
        output_path: Output file path
        hidden_size: Hidden size parameter
        seed: Random seed used
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data to save
    permutation_data = {
        'hidden_size': hidden_size,
        'seed': seed,
        'permutation': permutation,
        'created_at': np.datetime64('now'),
        'version': '1.0.0',
        'description': 'Fixed permutation matrix for MEMOIR TopHashFingerprinter'
    }
    
    # Save to file
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(permutation_data, f)
        
        logger.info(f"Permutation matrix saved to {output_file}")
        logger.info(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Failed to save permutation matrix: {e}")
        raise

def load_and_validate_permutation(file_path: str) -> dict:
    """
    Load and validate a saved permutation matrix.
    
    Args:
        file_path: Path to the permutation file
        
    Returns:
        Loaded permutation data
    """
    logger = logging.getLogger(__name__)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Validate data structure
        required_keys = ['hidden_size', 'seed', 'permutation', 'created_at']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate permutation
        permutation = data['permutation']
        hidden_size = data['hidden_size']
        
        assert len(permutation) == hidden_size
        assert len(set(permutation)) == hidden_size
        assert min(permutation) == 0 and max(permutation) == hidden_size - 1
        
        logger.info(f"Permutation matrix loaded and validated from {file_path}")
        logger.info(f"Hidden size: {hidden_size}, Seed: {data['seed']}")
        logger.info(f"Created at: {data['created_at']}")
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to load permutation matrix: {e}")
        raise

def main():
    """Main function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description="Generate permutation matrix for MEMOIR TopHashFingerprinter"
    )
    
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=4096,
        help='Hidden size for the permutation matrix (default: 4096)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/memoir_permutation.pkl',
        help='Output file path (default: data/memoir_permutation.pkl)'
    )
    
    parser.add_argument(
        '--validate',
        type=str,
        help='Validate an existing permutation file'
    )
    
    args = parser.parse_args()
    
    try:
        if args.validate:
            # Validate existing file
            logger.info(f"Validating permutation file: {args.validate}")
            data = load_and_validate_permutation(args.validate)
            logger.info("Validation successful!")
            
            # Print summary
            print(f"\nPermutation Matrix Summary:")
            print(f"  Hidden Size: {data['hidden_size']}")
            print(f"  Seed: {data['seed']}")
            print(f"  Created: {data['created_at']}")
            print(f"  Version: {data.get('version', 'Unknown')}")
            
        else:
            # Generate new permutation matrix
            logger.info("Generating new permutation matrix")
            
            permutation = generate_permutation_matrix(args.hidden_size, args.seed)
            save_permutation_matrix(permutation, args.output, args.hidden_size, args.seed)
            
            # Validate the saved file
            logger.info("Validating saved file")
            load_and_validate_permutation(args.output)
            
            print(f"\nPermutation matrix generated successfully!")
            print(f"  Hidden Size: {args.hidden_size}")
            print(f"  Seed: {args.seed}")
            print(f"  Output: {args.output}")
            print(f"  Size: {Path(args.output).stat().st_size / 1024:.2f} KB")
            
            # Show sample of permutation
            print(f"\nSample permutation values:")
            print(f"  First 10: {permutation[:10].tolist()}")
            print(f"  Last 10: {permutation[-10:].tolist()}")
            
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
