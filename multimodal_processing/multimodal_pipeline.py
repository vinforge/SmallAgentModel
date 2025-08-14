"""
Multimodal Processing Pipeline for SAM
Orchestrates the complete multimodal document processing workflow.

Sprint 4 Task 5: Chat + Dashboard Integration
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import asdict

from .document_parser import get_document_parser, ParsedDocument
from .knowledge_consolidator import get_knowledge_consolidator, ConsolidatedKnowledge
from .enrichment_scorer import get_enrichment_scorer, EnrichmentScore
from utils.vector_manager import VectorManager
from utils.embedding_utils import get_embedding_manager
from memory.memory_vectorstore import get_memory_store, MemoryType
from sam.cognition.table_processing.sam_integration import get_table_aware_chunker, TableProcessingResult

logger = logging.getLogger(__name__)

class MultimodalProcessingPipeline:
    """
    Complete pipeline for processing multimodal documents and integrating with SAM's knowledge base.
    """
    
    def __init__(self, 
                 vector_manager: Optional[VectorManager] = None,
                 output_dir: str = "multimodal_output"):
        """
        Initialize the multimodal processing pipeline.
        
        Args:
            vector_manager: Vector store manager for knowledge integration
            output_dir: Directory to store processing outputs
        """
        self.document_parser = get_document_parser()
        self.knowledge_consolidator = get_knowledge_consolidator()
        self.enrichment_scorer = get_enrichment_scorer()
        self.vector_manager = vector_manager or VectorManager()
        self.embedding_manager = get_embedding_manager()

        # Memory store for Q&A retrieval
        self.memory_store = get_memory_store()

        # Table processing system
        self.table_aware_chunker = get_table_aware_chunker()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Processing statistics
        self.processing_stats = {
            'documents_processed': 0,
            'total_content_blocks': 0,
            'consolidated_knowledge_items': 0,
            'vector_store_additions': 0,
            'memory_store_additions': 0,
            'processing_errors': 0
        }
        
        logger.info(f"Multimodal processing pipeline initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def process_document(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Process a single multimodal document through the complete pipeline.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            Processing results dictionary or None if processing failed
        """
        file_path = Path(file_path)
        
        try:
            logger.info(f"Starting multimodal processing: {file_path}")
            
            # Step 1: Parse document
            parsed_doc = self.document_parser.parse_document(file_path)
            if not parsed_doc:
                logger.error(f"Failed to parse document: {file_path}")
                self.processing_stats['processing_errors'] += 1
                return None
            
            logger.info(f"Parsed document: {len(parsed_doc.content_blocks)} content blocks")
            self.processing_stats['total_content_blocks'] += len(parsed_doc.content_blocks)

            # Step 1.5: Process tables with semantic role classification
            table_processing_result = self._process_tables_in_document(parsed_doc, file_path)

            # Step 2: Consolidate knowledge
            consolidated = self.knowledge_consolidator.consolidate_document(parsed_doc)
            if not consolidated:
                logger.error(f"Failed to consolidate knowledge: {file_path}")
                self.processing_stats['processing_errors'] += 1
                return None
            
            logger.info(f"Consolidated knowledge: {len(consolidated.summary)} chars summary")
            self.processing_stats['consolidated_knowledge_items'] += 1
            
            # Step 3: Score enrichment value
            enrichment_score = self.enrichment_scorer.score_consolidated_knowledge(consolidated)
            
            logger.info(f"Enrichment score: {enrichment_score.overall_score:.2f} ({enrichment_score.priority_level})")
            
            # Step 4: Add to vector store
            self.vector_manager.add_consolidated_knowledge(consolidated, enrichment_score)
            self.processing_stats['vector_store_additions'] += 1

            # Step 4.5: Store in memory system for Q&A retrieval
            memory_storage_result = self._store_document_in_memory(parsed_doc, consolidated, enrichment_score)

            # Step 5: Save processing outputs
            processing_result = self._save_processing_outputs(parsed_doc, consolidated, enrichment_score)

            # Add memory storage information to processing result
            if memory_storage_result:
                processing_result['memory_storage'] = memory_storage_result

            # Add table processing information to processing result
            if table_processing_result:
                processing_result['table_processing'] = {
                    'tables_found': len(table_processing_result.tables),
                    'enhanced_chunks': len(table_processing_result.enhanced_chunks),
                    'processing_metrics': table_processing_result.processing_metrics
                }

            # Update statistics
            self.processing_stats['documents_processed'] += 1
            
            logger.info(f"Successfully processed multimodal document: {file_path}")
            return processing_result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            self.processing_stats['processing_errors'] += 1
            return None
    
    def process_documents_batch(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of document paths to process
            
        Returns:
            List of processing results
        """
        results = []
        
        logger.info(f"Starting batch processing: {len(file_paths)} documents")
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing document {i}/{len(file_paths)}: {file_path}")
            
            result = self.process_document(file_path)
            if result:
                results.append(result)
        
        # Save vector store after batch processing
        try:
            self.vector_manager.save_index()
            logger.info("Vector store saved after batch processing")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
        
        logger.info(f"Batch processing completed: {len(results)}/{len(file_paths)} successful")
        return results
    
    def _save_processing_outputs(self, parsed_doc: ParsedDocument, 
                               consolidated: ConsolidatedKnowledge,
                               enrichment_score: EnrichmentScore) -> Dict[str, Any]:
        """Save all processing outputs to disk."""
        try:
            # Create output subdirectory for this document
            doc_output_dir = self.output_dir / f"doc_{parsed_doc.document_id}"
            doc_output_dir.mkdir(exist_ok=True)
            
            # Save parsed document
            parsed_path = doc_output_dir / "parsed_document.json"
            with open(parsed_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(parsed_doc), f, indent=2, ensure_ascii=False, default=str)
            
            # Save consolidated knowledge
            consolidated_path = doc_output_dir / "consolidated_knowledge.json"
            with open(consolidated_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(consolidated), f, indent=2, ensure_ascii=False, default=str)
            
            # Save enrichment score
            score_path = doc_output_dir / "enrichment_score.json"
            with open(score_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(enrichment_score), f, indent=2, ensure_ascii=False, default=str)
            
            # Create human-readable summary
            summary_path = doc_output_dir / "summary.md"
            summary_content = self._create_summary_markdown(parsed_doc, consolidated, enrichment_score)
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            processing_result = {
                'document_id': parsed_doc.document_id,
                'source_file': parsed_doc.source_file,
                'output_directory': str(doc_output_dir),
                'content_blocks': len(parsed_doc.content_blocks),
                'summary_length': len(consolidated.summary),
                'key_concepts': len(consolidated.key_concepts),
                'enrichment_score': enrichment_score.overall_score,
                'priority_level': enrichment_score.priority_level,
                'content_types': list(consolidated.content_attribution.keys()),
                'processing_timestamp': consolidated.consolidation_timestamp
            }
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Error saving processing outputs: {e}")
            return {}
    
    def _create_summary_markdown(self, parsed_doc: ParsedDocument, 
                               consolidated: ConsolidatedKnowledge,
                               enrichment_score: EnrichmentScore) -> str:
        """Create a human-readable markdown summary."""
        
        content = f"""# Multimodal Document Processing Summary

## Document Information
- **Source File:** {parsed_doc.source_file}
- **Document ID:** {parsed_doc.document_id}
- **Processing Date:** {consolidated.consolidation_timestamp}
- **File Size:** {parsed_doc.document_metadata.get('file_size', 0):,} bytes

## Content Analysis
- **Total Content Blocks:** {len(parsed_doc.content_blocks)}
- **Content Types:** {', '.join(consolidated.content_attribution.keys())}
- **Multimodal Richness:** {consolidated.enriched_metadata.get('multimodal_richness', 0):.2f}
- **Technical Content Ratio:** {consolidated.enriched_metadata.get('technical_content_ratio', 0):.2f}

## Enrichment Scoring
- **Overall Score:** {enrichment_score.overall_score:.2f}
- **Priority Level:** {enrichment_score.priority_level.upper()}
- **Score Explanation:** {enrichment_score.score_explanation}

### Component Scores
"""
        
        for component, score in enrichment_score.component_scores.items():
            content += f"- **{component.replace('_', ' ').title()}:** {score:.2f}\n"
        
        content += f"""
## Knowledge Summary
{consolidated.summary}

## Key Concepts
"""
        
        for i, concept in enumerate(consolidated.key_concepts, 1):
            content += f"{i}. {concept}\n"
        
        content += f"""
## Content Attribution
"""
        
        for content_type, locations in consolidated.content_attribution.items():
            content += f"- **{content_type.title()}:** {len(locations)} blocks\n"
            for location in locations[:3]:  # Show first 3 locations
                content += f"  - {location}\n"
            if len(locations) > 3:
                content += f"  - ... and {len(locations) - 3} more\n"
        
        # Add programming languages if present
        if 'programming_languages' in consolidated.enriched_metadata:
            languages = consolidated.enriched_metadata['programming_languages']
            content += f"""
## Programming Languages Detected
"""
            for lang, count in languages.items():
                content += f"- **{lang}:** {count} blocks\n"
        
        # Add table statistics if present
        if 'table_statistics' in consolidated.enriched_metadata:
            table_stats = consolidated.enriched_metadata['table_statistics']
            content += f"""
## Table Statistics
- **Total Tables:** {table_stats.get('total_tables', 0)}
- **Total Rows:** {table_stats.get('total_rows', 0)}
- **Average Columns:** {table_stats.get('avg_columns', 0):.1f}
"""
        
        return content
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def search_multimodal_content(self, query: str, top_k: int = 5, 
                                content_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for multimodal content using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            content_type_filter: Filter by content type ('text', 'code', 'table', 'image', 'multimodal')
            
        Returns:
            List of search results with multimodal metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_query(query)
            
            # Search vector store
            results = self.vector_manager.search(query_embedding, top_k=top_k * 2)  # Get more for filtering
            
            # Filter by content type if specified
            if content_type_filter:
                filtered_results = []
                for result in results:
                    metadata = result.get('metadata', {})
                    if metadata.get('content_type') == content_type_filter:
                        filtered_results.append(result)
                results = filtered_results[:top_k]
            else:
                results = results[:top_k]
            
            # Enhance results with multimodal information
            enhanced_results = []
            for result in results:
                metadata = result.get('metadata', {})
                
                enhanced_result = result.copy()
                enhanced_result['is_multimodal'] = metadata.get('is_multimodal', False)
                enhanced_result['content_type'] = metadata.get('content_type', 'text')
                enhanced_result['enrichment_score'] = metadata.get('enrichment_score', 0.0)
                enhanced_result['priority_level'] = metadata.get('priority_level', 'unknown')
                enhanced_result['key_concepts'] = metadata.get('key_concepts', [])
                
                # Add multimodal data if available
                if 'multimodal_data' in metadata:
                    enhanced_result['multimodal_data'] = metadata['multimodal_data']
                
                enhanced_results.append(enhanced_result)
            
            logger.debug(f"Multimodal search completed: {len(enhanced_results)} results")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in multimodal search: {e}")
            return []
    
    def get_content_type_stats(self) -> Dict[str, Any]:
        """Get statistics about content types in the vector store."""
        try:
            # This would require iterating through all chunks in the vector store
            # For now, return basic stats from processing
            return {
                'total_processed': self.processing_stats['documents_processed'],
                'total_content_blocks': self.processing_stats['total_content_blocks'],
                'consolidated_items': self.processing_stats['consolidated_knowledge_items'],
                'vector_store_size': self.vector_manager.get_stats()['total_chunks']
            }
        except Exception as e:
            logger.error(f"Error getting content type stats: {e}")
            return {}

    def _store_document_in_memory(self, parsed_doc: ParsedDocument,
                                 consolidated: ConsolidatedKnowledge,
                                 enrichment_score: EnrichmentScore):
        """
        Store document content in memory system for Q&A retrieval.

        This is the critical fix for Sprint 14: ensuring uploaded documents
        are immediately accessible via semantic question-answering.
        """
        try:
            logger.info(f"Storing document in memory system: {parsed_doc.source_file}")

            # Store the consolidated summary as a document memory
            summary_content = f"""Document: {parsed_doc.source_file}

Summary:
{consolidated.summary}

Key Concepts: {', '.join(consolidated.key_concepts)}

Content Types: {', '.join(consolidated.content_attribution.keys())}
Enrichment Score: {enrichment_score.overall_score:.2f} ({enrichment_score.priority_level})
"""

            # Create memory chunk for the document summary
            summary_chunk_id = self.memory_store.add_memory(
                content=summary_content,
                memory_type=MemoryType.DOCUMENT,
                source=f"document:{parsed_doc.source_file}",
                tags=[
                    "document",
                    "uploaded",
                    "summary",
                    enrichment_score.priority_level,
                    *[concept.lower().replace(" ", "_") for concept in consolidated.key_concepts[:5]]  # Normalized tags
                ],
                importance_score=min(enrichment_score.overall_score / 10.0, 1.0),  # Normalize to 0-1
                metadata={
                    "document_id": parsed_doc.document_id,
                    "source_file": parsed_doc.source_file,
                    "file_name": Path(parsed_doc.source_file).name,
                    "file_extension": Path(parsed_doc.source_file).suffix,
                    "content_types": list(consolidated.content_attribution.keys()),
                    "enrichment_score": enrichment_score.overall_score,
                    "priority_level": enrichment_score.priority_level,
                    "key_concepts": consolidated.key_concepts,
                    "processing_timestamp": consolidated.consolidation_timestamp,
                    "content_blocks_count": len(parsed_doc.content_blocks),
                    "multimodal_richness": consolidated.enriched_metadata.get('multimodal_richness', 0),
                    "technical_content_ratio": consolidated.enriched_metadata.get('technical_content_ratio', 0),
                    "document_type": "summary",
                    "searchable_keywords": " ".join(consolidated.key_concepts).lower(),
                    "file_size": parsed_doc.document_metadata.get('file_size', 0),
                    "upload_timestamp": datetime.now().isoformat()
                }
            )

            # Store individual content blocks for detailed Q&A
            content_chunk_ids = []
            for i, content_block in enumerate(parsed_doc.content_blocks):
                # Handle different content types properly
                content_str = ""
                if isinstance(content_block.content, str):
                    content_str = content_block.content.strip()
                elif isinstance(content_block.content, list):
                    # Handle table content (list of lists)
                    if content_block.content and all(isinstance(row, list) for row in content_block.content):
                        # Convert table to string representation
                        content_str = "\n".join(["\t".join(row) for row in content_block.content])
                    else:
                        content_str = str(content_block.content)
                elif isinstance(content_block.content, dict):
                    # Handle image/metadata content
                    content_str = str(content_block.content)
                else:
                    content_str = str(content_block.content)

                if content_str:  # Only store non-empty content

                    # Create detailed content for this block
                    block_content = f"""Document: {parsed_doc.source_file} (Block {i+1})
Content Type: {content_block.content_type}

{content_str}
"""

                    # Add metadata if available
                    if content_block.metadata:
                        metadata_str = "\n".join([f"{k}: {v}" for k, v in content_block.metadata.items()])
                        block_content += f"\n\nMetadata:\n{metadata_str}"

                    # Determine importance based on content type and enrichment score
                    block_importance = enrichment_score.overall_score / 10.0
                    if content_block.content_type in ['code', 'table']:
                        block_importance *= 1.2  # Boost technical content
                    elif content_block.content_type == 'image':
                        block_importance *= 0.8  # Lower importance for images without text

                    block_importance = min(block_importance, 1.0)  # Cap at 1.0

                    chunk_id = self.memory_store.add_memory(
                        content=block_content,
                        memory_type=MemoryType.DOCUMENT,
                        source=f"document:{parsed_doc.source_file}:block_{i+1}",
                        tags=[
                            "document",
                            "content_block",
                            content_block.content_type,
                            enrichment_score.priority_level,
                            f"block_{i+1}",
                            *[concept.lower().replace(" ", "_") for concept in consolidated.key_concepts[:3]]  # Normalized concepts
                        ],
                        importance_score=block_importance,
                        metadata={
                            "document_id": parsed_doc.document_id,
                            "source_file": parsed_doc.source_file,
                            "file_name": Path(parsed_doc.source_file).name,
                            "block_index": i,
                            "content_type": content_block.content_type,
                            "block_metadata": content_block.metadata,
                            "parent_summary_chunk": summary_chunk_id,
                            "document_type": "content_block",
                            "block_length": len(content_str),
                            "processing_timestamp": consolidated.consolidation_timestamp,
                            "upload_timestamp": datetime.now().isoformat()
                        }
                    )

                    content_chunk_ids.append(chunk_id)

            # Update processing statistics
            self.processing_stats['memory_store_additions'] += 1 + len(content_chunk_ids)

            logger.info(f"Document stored in memory: {len(content_chunk_ids)} content blocks + 1 summary")
            logger.info(f"Summary chunk ID: {summary_chunk_id}")

            return {
                "summary_chunk_id": summary_chunk_id,
                "content_chunk_ids": content_chunk_ids,
                "total_chunks_stored": 1 + len(content_chunk_ids)
            }

        except Exception as e:
            logger.error(f"Error storing document in memory: {e}")
            return None

    def _process_tables_in_document(self, parsed_doc: ParsedDocument, file_path: Path) -> Optional[TableProcessingResult]:
        """
        Process tables in the document using the table processing system.

        Args:
            parsed_doc: Parsed document with content blocks
            file_path: Path to the original document file

        Returns:
            TableProcessingResult or None if no tables found or processing failed
        """
        try:
            # Extract document content for table processing
            doc_content = ""
            doc_type = file_path.suffix.lower().lstrip('.')

            # Combine all text content for table detection
            for content_block in parsed_doc.content_blocks:
                if content_block.content_type == 'text':
                    doc_content += content_block.content + "\n"
                elif content_block.content_type == 'table':
                    # Convert table data back to text format for processing
                    if isinstance(content_block.content, list):
                        table_text = "\n".join(["\t".join(row) for row in content_block.content])
                        doc_content += table_text + "\n"

            if not doc_content.strip():
                logger.info("No text content found for table processing")
                return None

            # Process document with table intelligence
            document_context = f"Document: {file_path.name}"
            table_result = self.table_aware_chunker.process_document_with_tables(
                doc_content, doc_type, document_context
            )

            if table_result and table_result.tables:
                logger.info(f"Table processing completed: {len(table_result.tables)} tables, "
                          f"{len(table_result.enhanced_chunks)} enhanced chunks")

                # Store table chunks in memory with enhanced metadata
                self._store_table_chunks_in_memory(table_result, file_path)

                return table_result
            else:
                logger.info("No tables detected in document")
                return None

        except Exception as e:
            logger.error(f"Error processing tables in document: {e}")
            return None

    def _store_table_chunks_in_memory(self, table_result: TableProcessingResult, file_path: Path):
        """
        Store table chunks in memory with enhanced metadata.

        Args:
            table_result: Result from table processing
            file_path: Path to the original document file
        """
        try:
            for chunk_metadata in table_result.enhanced_chunks:
                # Ensure standard metadata for citations and traceability
                meta = dict(chunk_metadata or {})
                # Inject document_id if missing (derive from current parsed document if available)
                if 'document_id' not in meta or not meta.get('document_id'):
                    # Try to use a nearby value in the result; fall back to filename stem
                    meta['document_id'] = getattr(table_result, 'document_id', None) or Path(file_path).stem
                # Add consistent content/source typing
                meta.setdefault('content_type', 'table_chunk')
                meta.setdefault('source_type', 'bulk_ingestion')

                # Create memory chunk with table metadata
                chunk_id = self.memory_store.add_memory(
                    content=meta.get('content', ''),
                    memory_type=MemoryType.DOCUMENT,
                    source=str(file_path),
                    tags=['table', 'structured_data'] + meta.get('tags', []),
                    importance_score=meta.get('confidence_score', 0.5),
                    metadata=meta
                )

                logger.debug(f"Stored table chunk: {chunk_id}")

        except Exception as e:
            logger.error(f"Error storing table chunks in memory: {e}")

# Global pipeline instance
_multimodal_pipeline = None

def get_multimodal_pipeline() -> MultimodalProcessingPipeline:
    """Get or create a global multimodal processing pipeline instance."""
    global _multimodal_pipeline
    
    if _multimodal_pipeline is None:
        _multimodal_pipeline = MultimodalProcessingPipeline()
    
    return _multimodal_pipeline
