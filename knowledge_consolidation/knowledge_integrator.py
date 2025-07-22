#!/usr/bin/env python3
"""
Knowledge Integrator - Integrates processed content into SAM's knowledge base
"""

import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class KnowledgeIntegrator:
    """Integrates processed content into SAM's vector store and memory system."""
    
    def __init__(self):
        self.vector_store = None
        self.memory_system = None
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize SAM's vector store and memory systems."""
        try:
            # Try to initialize secure memory store first (for SAM compatibility)
            try:
                from memory.secure_memory_vectorstore import get_secure_memory_store, VectorStoreType

                # Try to get the security manager from Streamlit session state if available
                security_manager = None
                try:
                    import streamlit as st
                    if hasattr(st, 'session_state') and 'security_manager' in st.session_state:
                        security_manager = st.session_state.security_manager
                        logger.info("Using security manager from Streamlit session")
                except:
                    logger.info("Streamlit session not available, creating new security manager")

                # Initialize with same settings as SAM's interface
                self.vector_store = get_secure_memory_store(
                    store_type=VectorStoreType.CHROMA,
                    storage_directory="memory_store",
                    embedding_dimension=384,
                    enable_encryption=True,  # Enable encryption to match SAM's interface
                    security_manager=security_manager  # Use same security manager if available
                )
                logger.info("Secure vector store initialized for knowledge integration (encryption enabled)")
            except ImportError:
                # Fallback to regular memory store
                from memory.memory_vectorstore import MemoryVectorStore
                self.vector_store = MemoryVectorStore()
                logger.info("Regular vector store initialized for knowledge integration")

            # Initialize memory system if available
            try:
                from memory.memory_manager import MemoryManager
                self.memory_system = MemoryManager()
                logger.info("Memory system initialized for knowledge integration")
            except ImportError:
                logger.warning("Memory system not available, using vector store only")

        except Exception as e:
            logger.error(f"Failed to initialize knowledge systems: {e}")
            raise
    
    def integrate_knowledge_items(self, knowledge_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate knowledge items into SAM's knowledge base."""
        try:
            logger.info(f"Integrating {len(knowledge_items)} knowledge items")
            
            if not knowledge_items:
                return {'success': True, 'message': 'No items to integrate', 'integrated_count': 0}
            
            integrated_count = 0
            failed_count = 0
            
            for item in knowledge_items:
                try:
                    success = self._integrate_single_item(item)
                    if success:
                        integrated_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Failed to integrate item {item.get('id', 'unknown')}: {e}")
                    failed_count += 1
            
            # Update memory statistics
            self._update_memory_stats(integrated_count)
            
            return {
                'success': True,
                'integrated_count': integrated_count,
                'failed_count': failed_count,
                'total_items': len(knowledge_items),
                'integration_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Knowledge integration failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _integrate_single_item(self, item: Dict[str, Any]) -> bool:
        """Integrate a single knowledge item."""
        try:
            # Prepare content for vector store
            content_text = self._prepare_content_for_storage(item)
            
            # Create enhanced metadata for vector store with query and temporal context
            item_metadata = item.get('metadata', {})
            current_time = datetime.now()

            metadata = {
                'id': item.get('id', ''),
                'title': item.get('title', ''),
                'source': item.get('source', ''),
                'url': item.get('url', ''),
                'content_type': item_metadata.get('content_type', 'web_content'),
                'processed_at': item_metadata.get('processed_at', current_time.isoformat()),
                'word_count': item_metadata.get('word_count', 0),
                'integration_method': 'knowledge_consolidation',

                # Enhanced query and temporal context
                'original_query': item_metadata.get('query', ''),
                'retrieval_date': current_time.strftime("%Y-%m-%d"),
                'retrieval_timestamp': current_time.isoformat(),
                'published_date': item_metadata.get('published_at', ''),

                # Content categorization for better retrieval
                'is_news_content': 'news' in item_metadata.get('query', '').lower() or
                                 item_metadata.get('content_type') == 'news_article',
                'is_current_content': any(term in item_metadata.get('query', '').lower()
                                        for term in ['latest', 'today', 'current', 'recent']),

                # Source categorization
                'source_domain': self._extract_domain_from_url(item.get('url', '')),
                'tool_source': item_metadata.get('tool_source', 'unknown')
            }
            
            # Add to vector store
            if self.vector_store:
                from memory.memory_vectorstore import MemoryType
                self.vector_store.add_memory(
                    content=content_text,
                    memory_type=MemoryType.DOCUMENT,
                    source=item.get('source', 'web_content'),
                    metadata=metadata
                )
                logger.debug(f"Added item {item.get('id')} to vector store")
            
            # Add to memory system if available
            if self.memory_system:
                self.memory_system.store_memory(
                    content=content_text,
                    metadata=metadata,
                    memory_type='consolidated_web_content'
                )
                logger.debug(f"Added item {item.get('id')} to memory system")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate single item: {e}")
            return False
    
    def _prepare_content_for_storage(self, item: Dict[str, Any]) -> str:
        """Prepare content text for storage in vector store with query association and temporal context."""
        title = item.get('title', '')
        content = item.get('content', '')
        source = item.get('source', '')
        url = item.get('url', '')

        # Create comprehensive content text with query association
        content_parts = []

        # Add temporal context first for better retrieval
        metadata = item.get('metadata', {})
        processed_date = datetime.now().strftime("%B %d, %Y")

        if metadata.get('query'):
            original_query = metadata['query']
            content_parts.append(f"Question: {original_query}")
            content_parts.append(f"Retrieved on: {processed_date}")

            # Add query variations for better matching
            query_variations = self._generate_query_variations(original_query)
            if query_variations:
                content_parts.append(f"Related searches: {', '.join(query_variations)}")

        # Add publication date if available
        if metadata.get('published_at'):
            content_parts.append(f"Published: {metadata['published_at']}")

        # Add title and content
        if title:
            content_parts.append(f"Title: {title}")

        if content:
            content_parts.append(f"Content: {content}")

        # Add source information
        if source:
            content_parts.append(f"Source: {source}")

        if url:
            content_parts.append(f"URL: {url}")

        # Add content type for better categorization
        content_type = metadata.get('content_type', 'web_content')
        if content_type == 'news_article':
            content_parts.append("Category: News Article")
        elif 'news' in original_query.lower() if metadata.get('query') else False:
            content_parts.append("Category: News Content")

        return '\n'.join(content_parts)

    def _generate_query_variations(self, original_query: str) -> List[str]:
        """Generate query variations for better retrieval matching."""
        variations = []
        query_lower = original_query.lower()

        # Common news query patterns
        if 'latest' in query_lower or 'today' in query_lower or 'current' in query_lower:
            variations.extend([
                'current news',
                'latest updates',
                'today\'s news',
                'recent news'
            ])

        # CNN specific patterns
        if 'cnn' in query_lower:
            variations.extend([
                'CNN news',
                'CNN updates',
                'CNN headlines'
            ])

        # Health news patterns
        if 'health' in query_lower:
            variations.extend([
                'health news',
                'medical news',
                'healthcare updates'
            ])

        # Political news patterns
        if 'politic' in query_lower:
            variations.extend([
                'political news',
                'politics updates',
                'government news'
            ])

        return variations[:5]  # Limit to 5 variations

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL for source categorization."""
        try:
            if not url or url == '':
                return 'unknown'

            # Simple domain extraction
            if '://' in url:
                domain_part = url.split('://')[1]
            else:
                domain_part = url

            domain = domain_part.split('/')[0].lower()

            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain

        except Exception:
            return 'unknown'
    
    def _update_memory_stats(self, integrated_count: int):
        """Update memory statistics after integration."""
        try:
            if self.vector_store:
                # Get current memory count
                stats = self.vector_store.get_memory_stats()
                current_count = stats.get('total_memories', 0)
                logger.info(f"Vector store now contains {current_count} memories after integrating {integrated_count} new items")
                
        except Exception as e:
            logger.warning(f"Failed to update memory stats: {e}")
    
    def verify_integration(self, sample_queries: List[str]) -> Dict[str, Any]:
        """Verify that integrated content is searchable."""
        try:
            if not self.vector_store:
                return {'success': False, 'error': 'Vector store not available'}
            
            verification_results = []
            
            for query in sample_queries:
                try:
                    # Search for the query
                    results = self.vector_store.search_memories(query, max_results=5)
                    
                    verification_results.append({
                        'query': query,
                        'results_found': len(results),
                        'success': len(results) > 0
                    })
                    
                except Exception as e:
                    verification_results.append({
                        'query': query,
                        'results_found': 0,
                        'success': False,
                        'error': str(e)
                    })
            
            successful_queries = sum(1 for r in verification_results if r['success'])
            
            return {
                'success': True,
                'verification_results': verification_results,
                'successful_queries': successful_queries,
                'total_queries': len(sample_queries),
                'success_rate': successful_queries / len(sample_queries) if sample_queries else 0
            }
            
        except Exception as e:
            logger.error(f"Integration verification failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current status of the knowledge integration system."""
        try:
            status = {
                'vector_store_available': self.vector_store is not None,
                'memory_system_available': self.memory_system is not None,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.vector_store:
                try:
                    stats = self.vector_store.get_memory_stats()
                    status['total_memories'] = stats.get('total_memories', 0)
                except:
                    status['total_memories'] = 'unknown'
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get integration status: {e}")
            return {'error': str(e)}
    
    def cleanup_old_web_content(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up old web content from the knowledge base."""
        try:
            if not self.vector_store:
                return {'success': False, 'error': 'Vector store not available'}
            
            # This would require implementing a cleanup method in the vector store
            # For now, return a placeholder
            return {
                'success': True,
                'message': f'Cleanup functionality not yet implemented for content older than {days_old} days',
                'cleaned_count': 0
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {'success': False, 'error': str(e)}
