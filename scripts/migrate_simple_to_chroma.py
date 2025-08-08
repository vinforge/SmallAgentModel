#!/usr/bin/env python3
"""
SAM Phase 2: Migration Script (Steps2.md Compliant)
Migrates existing SIMPLE memory store to ChromaDB with full validation.

Requirements:
- Configuration-driven (uses ConfigManager)
- Idempotent with --clean flag
- Batch ingestion for efficiency
- Comprehensive validation and verification
"""

import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from memory.memory_vectorstore import MemoryVectorStore, VectorStoreType, MemoryType, MemoryChunk
from utils.chroma_client import get_chroma_client, get_chroma_collection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChromaMigrator:
    """Configuration-driven migration from SIMPLE to ChromaDB."""
    
    def __init__(self):
        """Initialize migrator with configuration."""
        self.stats = {
            "source_files": 0,
            "migrated": 0,
            "skipped": 0,
            "errors": 0,
            "batches": 0
        }

        # Load configuration directly from JSON (handle current format)
        self.config_data = self._load_config_json()

        # Configuration-driven paths
        memory_config = self.config_data.get("memory", {})
        self.simple_store_path = Path("web_ui") / memory_config.get("storage_dir", "memory_store")

        chroma_config = memory_config.get("chroma_config", {})
        self.chroma_persist_path = chroma_config.get("persist_path", "web_ui/chroma_db")
        self.collection_name = chroma_config.get("collection_name", "sam_memory_store")
        self.batch_size = chroma_config.get("batch_size", 100)
        
        logger.info(f"Configuration loaded:")
        logger.info(f"  Simple store: {self.simple_store_path}")
        logger.info(f"  ChromaDB path: {self.chroma_persist_path}")
        logger.info(f"  Collection: {self.collection_name}")
        logger.info(f"  Batch size: {self.batch_size}")

    def _load_config_json(self) -> Dict[str, Any]:
        """Load configuration directly from JSON file."""
        try:
            config_file = Path("config/sam_config.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Config file not found, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def discover_memory_files(self) -> List[Path]:
        """Discover all mem_*.json files in the SIMPLE store."""
        if not self.simple_store_path.exists():
            logger.error(f"Simple store directory not found: {self.simple_store_path}")
            return []
        
        memory_files = list(self.simple_store_path.glob("mem_*.json"))
        self.stats["source_files"] = len(memory_files)
        
        logger.info(f"Found {len(memory_files)} memories in legacy store to migrate.")
        return memory_files
    
    def _prepare_chroma_metadata(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform JSON data into ChromaDB-compatible metadata."""
        try:
            # Extract core fields
            chunk_id = json_data.get("chunk_id", "")
            content = json_data.get("content", "")
            source = json_data.get("source", "")
            metadata = json_data.get("metadata", {})
            
            # Parse source information
            source_parts = source.split(':')
            source_type = source_parts[0] if len(source_parts) > 0 else "unknown"
            source_path = source_parts[1] if len(source_parts) > 1 else ""
            block_info = source_parts[2] if len(source_parts) > 2 else ""
            
            # Extract source name
            source_name = "unknown"
            if source_path:
                source_name = Path(source_path).name if source_path.startswith("uploads/") else source_path
            
            # Extract page/chunk information
            chunk_index = 0
            page_number = 1
            if block_info and "block_" in block_info:
                try:
                    chunk_index = int(block_info.split("_")[1])
                except (IndexError, ValueError):
                    chunk_index = 0
            
            # Calculate document position
            document_position = 0.0
            if "block_index" in metadata:
                block_idx = metadata.get("block_index", 0)
                total_blocks = metadata.get("total_blocks", 10)
                document_position = min(block_idx / max(total_blocks, 1), 1.0)
            
            # Determine confidence
            importance_score = json_data.get("importance_score", 0.0)
            confidence_score = float(importance_score)
            confidence_indicator = "✓" if confidence_score > 0.7 else "~" if confidence_score > 0.4 else "?"
            
            # Build ChromaDB-compatible metadata
            chroma_metadata = {
                # Citation schema fields
                "source_name": source_name,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "paragraph_number": 1,
                "section_title": metadata.get("section_title", "Content"),
                "document_position": document_position,
                "confidence_indicator": confidence_indicator,
                "confidence_score": confidence_score,
                "text_content": content[:500] + "..." if len(content) > 500 else content,
                "created_at": json_data.get("timestamp", ""),
                
                # Extended metadata
                "memory_type": json_data.get("memory_type", "document"),
                "source_type": source_type,
                "source_path": source_path,
                "content_hash": json_data.get("content_hash", ""),
                "importance_score": confidence_score,
                "access_count": json_data.get("access_count", 0),
                "last_accessed": json_data.get("last_accessed", ""),
                "tags": ",".join(json_data.get("tags", [])),  # Convert list to string
                
                # Document-specific metadata (ChromaDB compatible)
                "content_type": metadata.get("content_type", "text"),
                "file_name": metadata.get("file_name", source_name),
                "document_id": metadata.get("document_id", ""),
                "block_length": len(content),
                "processing_timestamp": metadata.get("processing_timestamp", ""),
                "upload_timestamp": metadata.get("upload_timestamp", "")
            }
            
            # Add extra metadata (convert lists to strings)
            for key, value in metadata.items():
                if key not in chroma_metadata:
                    if isinstance(value, list):
                        chroma_metadata[f"extra_{key}"] = ",".join(str(v) for v in value)
                    elif isinstance(value, (str, int, float, bool)):
                        chroma_metadata[f"extra_{key}"] = value
                    elif value is not None:
                        chroma_metadata[f"extra_{key}"] = str(value)
            
            return chroma_metadata
            
        except Exception as e:
            logger.error(f"Error preparing metadata: {e}")
            return {
                "source_name": "unknown",
                "chunk_index": 0,
                "confidence_score": 0.0,
                "created_at": "",
                "memory_type": "document",
                "text_content": json_data.get("content", "")[:100]
            }
    
    def load_memory_from_json(self, file_path: Path) -> Optional[Tuple[str, str, List[float], Dict[str, Any]]]:
        """Load memory data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            # Extract required fields
            chunk_id = json_data.get("chunk_id")
            content = json_data.get("content", "")
            embedding = json_data.get("embedding", [])
            
            if not chunk_id or not embedding:
                logger.warning(f"Skipping incomplete file: {file_path.name}")
                return None
            
            # Prepare metadata
            metadata = self._prepare_chroma_metadata(json_data)
            
            return chunk_id, content, embedding, metadata
            
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping corrupt file {file_path.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            return None

    def migrate_with_batching(self, memory_files: List[Path], clean: bool = False) -> bool:
        """Migrate memories to ChromaDB using efficient batch ingestion."""
        try:
            # Initialize ChromaDB collection
            logger.info(f"Using ChromaDB collection '{self.collection_name}' at '{self.chroma_persist_path}'.")

            client = get_chroma_client(self.chroma_persist_path)

            if clean:
                logger.info(f"Re-creating collection '{self.collection_name}'.")
                try:
                    client.delete_collection(self.collection_name)
                except Exception:
                    pass  # Collection might not exist

            collection = get_chroma_collection(self.collection_name, self.chroma_persist_path)

            # Check for existing IDs if not cleaning
            existing_ids = set()
            if not clean:
                try:
                    existing_data = collection.get(include=[])
                    existing_ids = set(existing_data["ids"])
                    logger.info(f"Found {len(existing_ids)} existing memories in collection.")
                except Exception as e:
                    logger.warning(f"Could not check existing IDs: {e}")

            # Prepare batches
            batch_ids = []
            batch_documents = []
            batch_embeddings = []
            batch_metadatas = []

            total_batches = (len(memory_files) + self.batch_size - 1) // self.batch_size
            current_batch = 1

            for file_path in memory_files:
                # Load memory data
                memory_data = self.load_memory_from_json(file_path)
                if not memory_data:
                    self.stats["skipped"] += 1
                    continue

                chunk_id, content, embedding, metadata = memory_data

                # Skip if already exists (idempotency)
                if chunk_id in existing_ids:
                    self.stats["skipped"] += 1
                    continue

                # Add to batch
                batch_ids.append(chunk_id)
                batch_documents.append(content)
                batch_embeddings.append(embedding)
                batch_metadatas.append(metadata)

                # Process batch when full
                if len(batch_ids) >= self.batch_size:
                    logger.info(f"Migrating batch {current_batch}/{total_batches} ({len(batch_ids)} memories)...")

                    collection.add(
                        ids=batch_ids,
                        documents=batch_documents,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas
                    )

                    self.stats["migrated"] += len(batch_ids)
                    self.stats["batches"] += 1
                    current_batch += 1

                    # Clear batch
                    batch_ids = []
                    batch_documents = []
                    batch_embeddings = []
                    batch_metadatas = []

            # Process final batch
            if batch_ids:
                logger.info(f"Migrating final batch ({len(batch_ids)} memories)...")

                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )

                self.stats["migrated"] += len(batch_ids)
                self.stats["batches"] += 1

            logger.info("Migration complete.")
            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    def validate_migration(self) -> bool:
        """Comprehensive validation and verification."""
        logger.info("Starting validation and verification...")

        try:
            # Count verification
            collection = get_chroma_collection(self.collection_name, self.chroma_persist_path)
            chroma_count = collection.count()

            logger.info(f"Source file count: {self.stats['source_files']}")
            logger.info(f"ChromaDB count: {chroma_count}")
            logger.info(f"Successfully migrated: {self.stats['migrated']}")
            logger.info(f"Skipped: {self.stats['skipped']}")

            # Verify counts are reasonable (total should match source files)
            total_processed = self.stats["migrated"] + self.stats["skipped"]
            if total_processed != self.stats["source_files"]:
                logger.error(f"Processing mismatch! Source files: {self.stats['source_files']}, Processed: {total_processed}")
                return False

            # If we migrated new items, verify they're in ChromaDB
            if self.stats["migrated"] > 0:
                if chroma_count < self.stats["migrated"]:
                    logger.error(f"ChromaDB missing items! Expected at least {self.stats['migrated']}, got {chroma_count}")
                    return False

            logger.info(f"SUCCESS: Verification complete. Total source files ({self.stats['source_files']}) = Processed ({total_processed}). ChromaDB has {chroma_count} items.")

            # Data spot-check
            if chroma_count > 0:
                logger.info("Performing data spot-check...")

                # Get random ID from ChromaDB
                all_data = collection.get(limit=min(10, chroma_count), include=["metadatas"])
                if all_data["ids"]:
                    random_id = random.choice(all_data["ids"])

                    # Fetch from ChromaDB
                    chroma_data = collection.get(ids=[random_id], include=["metadatas", "documents"])
                    chroma_metadata = chroma_data["metadatas"][0]
                    chroma_content = chroma_data["documents"][0]

                    # Load original JSON
                    json_file = self.simple_store_path / f"{random_id}.json"
                    if json_file.exists():
                        with open(json_file, 'r') as f:
                            original_data = json.load(f)

                        # Compare key fields
                        original_content = original_data.get("content", "")
                        original_source = original_data.get("source", "")

                        # Verify content matches
                        if chroma_content != original_content:
                            logger.error("Content mismatch in spot-check!")
                            return False

                        # Verify source information
                        expected_source_name = Path(original_source.split(':')[1]).name if ':' in original_source else "unknown"
                        actual_source_name = chroma_metadata.get("source_name", "")

                        if expected_source_name != actual_source_name:
                            logger.warning(f"Source name difference: expected '{expected_source_name}', got '{actual_source_name}'")

                        logger.info(f"SUCCESS: Spot-check passed for ID {random_id}")
                    else:
                        logger.warning(f"Original file not found for spot-check: {json_file}")

            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def print_summary(self):
        """Print migration summary."""
        logger.info("Migration Summary:")
        logger.info(f"  Source files found: {self.stats['source_files']}")
        logger.info(f"  Successfully migrated: {self.stats['migrated']}")
        logger.info(f"  Skipped: {self.stats['skipped']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"  Batches processed: {self.stats['batches']}")

        if self.stats["source_files"] > 0:
            success_rate = (self.stats["migrated"] / self.stats["source_files"]) * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")

def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate SAM SIMPLE store to ChromaDB")
    parser.add_argument("--clean", action="store_true",
                       help="Delete and re-create ChromaDB collection (prevents duplicates)")

    args = parser.parse_args()

    print("🔄 SAM Phase 2: SIMPLE to ChromaDB Migration")
    print("=" * 50)

    # Initialize migrator
    migrator = ChromaMigrator()

    # Discover memory files
    memory_files = migrator.discover_memory_files()
    if not memory_files:
        logger.error("No memory files found. Aborting migration.")
        return False

    # Perform migration
    success = migrator.migrate_with_batching(memory_files, clean=args.clean)
    if not success:
        logger.error("Migration failed.")
        return False

    # Validate migration
    validation_success = migrator.validate_migration()

    # Print summary
    migrator.print_summary()

    if success and validation_success:
        print("\n🎉 Migration completed successfully!")
        print("SAM is ready to use ChromaDB backend.")
        return True
    else:
        print("\n❌ Migration failed validation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
