#!/usr/bin/env python3
"""
Content Processor for Knowledge Consolidation
Processes approved vetted content and prepares it for integration into SAM's knowledge base
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class ContentProcessor:
    """Processes approved content for knowledge base integration."""
    
    def __init__(self):
        self.processed_content = []
        
    def process_approved_content(self, approved_file_path: str) -> Dict[str, Any]:
        """Process a single approved content file."""
        try:
            logger.info(f"Processing approved content: {approved_file_path}")
            
            # Load approved content
            with open(approved_file_path, 'r', encoding='utf-8') as f:
                approved_data = json.load(f)
            
            # Extract content based on file type
            if 'result' in approved_data and 'data' in approved_data['result']:
                # Intelligent web system format
                return self._process_intelligent_web_content(approved_data)
            elif 'articles' in approved_data:
                # Direct articles format
                return self._process_articles_content(approved_data)
            elif 'content' in approved_data:
                # Direct content format
                return self._process_direct_content(approved_data)
            else:
                logger.warning(f"Unknown content format in {approved_file_path}")
                return {'success': False, 'error': 'Unknown content format'}
                
        except Exception as e:
            logger.error(f"Error processing approved content {approved_file_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_intelligent_web_content(self, approved_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process content from intelligent web system."""
        try:
            query = approved_data.get('query', '')
            result_data = approved_data['result']['data']
            
            knowledge_items = []
            
            # Process articles if available
            if 'articles' in result_data:
                for article in result_data['articles']:
                    # Enhanced content preparation for news articles
                    article_content = article.get('description', '')
                    if not article_content and article.get('content'):
                        article_content = article.get('content', '')

                    # Create enriched content that includes query context
                    enriched_content = self._create_enriched_content(
                        title=article.get('title', ''),
                        content=article_content,
                        query=query,
                        source=article.get('source', ''),
                        published_at=article.get('published_at', '')
                    )

                    knowledge_item = self._create_knowledge_item(
                        title=article.get('title', ''),
                        content=enriched_content,
                        source=article.get('source', ''),
                        url=article.get('url', ''),
                        metadata={
                            'query': query,
                            'published_at': article.get('published_at', ''),
                            'tool_source': article.get('tool_source', ''),
                            'content_type': 'news_article',
                            'original_query': query,
                            'retrieval_context': f"Retrieved in response to: {query}"
                        }
                    )
                    knowledge_items.append(knowledge_item)
            
            # Process search results if available
            elif 'search_results' in result_data:
                for result in result_data['search_results']:
                    knowledge_item = self._create_knowledge_item(
                        title=result.get('title', ''),
                        content=result.get('snippet', ''),
                        source='search_result',
                        url=result.get('url', ''),
                        metadata={
                            'query': query,
                            'content_type': 'search_result'
                        }
                    )
                    knowledge_items.append(knowledge_item)
            
            # Process direct content if available
            elif 'content' in result_data:
                knowledge_item = self._create_knowledge_item(
                    title=f"Web Content for: {query}",
                    content=result_data['content'],
                    source='web_extraction',
                    url=result_data.get('url', ''),
                    metadata={
                        'query': query,
                        'content_type': 'extracted_content'
                    }
                )
                knowledge_items.append(knowledge_item)
            
            return {
                'success': True,
                'knowledge_items': knowledge_items,
                'total_items': len(knowledge_items),
                'source_query': query
            }
            
        except Exception as e:
            logger.error(f"Error processing intelligent web content: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_articles_content(self, approved_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process direct articles content."""
        try:
            articles = approved_data['articles']
            knowledge_items = []
            
            for article in articles:
                knowledge_item = self._create_knowledge_item(
                    title=article.get('title', ''),
                    content=article.get('description', '') or article.get('content', ''),
                    source=article.get('source', ''),
                    url=article.get('url', '') or article.get('link', ''),
                    metadata={
                        'published_at': article.get('published_at', '') or article.get('pub_date', ''),
                        'author': article.get('author', ''),
                        'content_type': 'article'
                    }
                )
                knowledge_items.append(knowledge_item)
            
            return {
                'success': True,
                'knowledge_items': knowledge_items,
                'total_items': len(knowledge_items)
            }
            
        except Exception as e:
            logger.error(f"Error processing articles content: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_direct_content(self, approved_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process direct content format."""
        try:
            knowledge_item = self._create_knowledge_item(
                title=approved_data.get('title', 'Web Content'),
                content=approved_data['content'],
                source=approved_data.get('source', ''),
                url=approved_data.get('url', ''),
                metadata={
                    'timestamp': approved_data.get('timestamp', ''),
                    'content_type': 'direct_content'
                }
            )
            
            return {
                'success': True,
                'knowledge_items': [knowledge_item],
                'total_items': 1
            }
            
        except Exception as e:
            logger.error(f"Error processing direct content: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_knowledge_item(self, title: str, content: str, source: str, 
                              url: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized knowledge item."""
        
        # Generate unique ID
        content_hash = hashlib.md5(f"{title}{content}{url}".encode()).hexdigest()[:12]
        
        # Clean and validate content
        title = title.strip() if title else 'Untitled'
        content = content.strip() if content else ''
        
        if len(content) < 10:  # Skip very short content
            return None
        
        return {
            'id': f"web_{content_hash}",
            'title': title,
            'content': content,
            'source': source,
            'url': url,
            'metadata': {
                **metadata,
                'processed_at': datetime.now().isoformat(),
                'content_length': len(content),
                'word_count': len(content.split())
            }
        }
    
    def batch_process_approved_content(self, approved_dir: str) -> Dict[str, Any]:
        """Process all approved content files in a directory."""
        try:
            approved_path = Path(approved_dir)
            if not approved_path.exists():
                return {'success': False, 'error': f'Approved directory not found: {approved_dir}'}
            
            approved_files = list(approved_path.glob('*.json'))
            if not approved_files:
                return {'success': True, 'message': 'No approved files to process', 'total_items': 0}
            
            all_knowledge_items = []
            processed_files = 0
            failed_files = 0
            
            for file_path in approved_files:
                try:
                    result = self.process_approved_content(str(file_path))
                    if result['success']:
                        all_knowledge_items.extend(result['knowledge_items'])
                        processed_files += 1
                    else:
                        failed_files += 1
                        logger.error(f"Failed to process {file_path}: {result.get('error')}")
                except Exception as e:
                    failed_files += 1
                    logger.error(f"Error processing {file_path}: {e}")
            
            return {
                'success': True,
                'knowledge_items': all_knowledge_items,
                'total_items': len(all_knowledge_items),
                'processed_files': processed_files,
                'failed_files': failed_files,
                'files_found': len(approved_files)
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_content_statistics(self, knowledge_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about processed content."""
        if not knowledge_items:
            return {'total_items': 0}
        
        content_types = {}
        sources = {}
        total_words = 0
        
        for item in knowledge_items:
            # Count content types
            content_type = item.get('metadata', {}).get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Count sources
            source = item.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            
            # Count words
            total_words += item.get('metadata', {}).get('word_count', 0)
        
        return {
            'total_items': len(knowledge_items),
            'content_types': content_types,
            'sources': sources,
            'total_words': total_words,
            'average_words_per_item': total_words / len(knowledge_items) if knowledge_items else 0
        }

    def filter_knowledge_items(self, knowledge_items: List[Dict[str, Any]],
                              min_content_length: int = 50) -> List[Dict[str, Any]]:
        """Filter knowledge items based on quality criteria."""
        filtered_items = []

        for item in knowledge_items:
            if item is None:
                continue

            content = item.get('content', '')
            title = item.get('title', '')

            # Skip items with insufficient content
            if len(content) < min_content_length:
                continue

            # Skip items with generic titles
            if title.lower() in ['untitled', 'no title', '']:
                continue

            filtered_items.append(item)

        return filtered_items

    def _create_enriched_content(self, title: str, content: str, query: str,
                                source: str, published_at: str) -> str:
        """Create enriched content that includes query context and temporal information."""
        enriched_parts = []

        # Add query context at the beginning for better retrieval
        if query:
            enriched_parts.append(f"This content was retrieved for the question: '{query}'")

        # Add temporal context
        current_date = datetime.now().strftime("%B %d, %Y")
        enriched_parts.append(f"Retrieved on: {current_date}")

        if published_at:
            enriched_parts.append(f"Originally published: {published_at}")

        # Add the actual content
        if title:
            enriched_parts.append(f"Headline: {title}")

        if content:
            enriched_parts.append(f"Article content: {content}")

        # Add source context
        if source:
            enriched_parts.append(f"News source: {source}")

        # Add relevance indicators for news content
        if any(term in query.lower() for term in ['latest', 'today', 'current', 'recent']):
            enriched_parts.append("Content type: Current news and updates")

        if 'cnn' in query.lower():
            enriched_parts.append("Source category: CNN news content")

        return '\n\n'.join(enriched_parts)
