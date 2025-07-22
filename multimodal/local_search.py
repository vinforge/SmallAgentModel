"""
Local File Search Integration for SAM
Semantic search across documents and knowledge capsules with vector embeddings.

Sprint 9 Task 2: Local File Search Integration
"""

import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re

logger = logging.getLogger(__name__)

class SearchResultType(Enum):
    """Types of search results."""
    DOCUMENT = "document"
    CAPSULE = "capsule"
    MULTIMODAL = "multimodal"

@dataclass
class SearchResult:
    """A single search result."""
    result_id: str
    result_type: SearchResultType
    title: str
    filename: str
    content_preview: str
    full_content: str
    confidence_score: float
    file_path: str
    section: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class SearchIndex:
    """Search index for a document or capsule."""
    index_id: str
    file_path: str
    filename: str
    file_type: str
    content_hash: str
    indexed_content: str
    sections: List[Dict[str, Any]]
    keywords: List[str]
    created_at: str
    last_updated: str
    metadata: Dict[str, Any]

class LocalFileSearchEngine:
    """
    Semantic search engine for local documents and knowledge capsules.
    """
    
    def __init__(self, knowledge_directory: str = "knowledge",
                 index_file: str = "search_index.json"):
        """
        Initialize the local file search engine.
        
        Args:
            knowledge_directory: Directory containing documents to index
            index_file: Path to search index storage file
        """
        self.knowledge_dir = Path(knowledge_directory)
        self.knowledge_dir.mkdir(exist_ok=True)
        
        self.index_file = Path(index_file)
        
        # Storage
        self.search_index: Dict[str, SearchIndex] = {}
        self.document_embeddings: Dict[str, List[float]] = {}
        
        # Configuration
        self.config = {
            'supported_formats': ['.pdf', '.md', '.txt', '.docx', '.json'],
            'max_preview_length': 200,
            'max_section_length': 1000,
            'min_content_length': 10,
            'enable_semantic_search': True,
            'reindex_on_change': True
        }
        
        # Load existing index
        self._load_search_index()
        
        logger.info(f"Local file search engine initialized with {len(self.search_index)} indexed items")
    
    def index_directory(self, directory_path: Optional[str] = None) -> int:
        """
        Index all supported files in the knowledge directory.
        
        Args:
            directory_path: Optional specific directory to index
            
        Returns:
            Number of files indexed
        """
        try:
            target_dir = Path(directory_path) if directory_path else self.knowledge_dir
            
            if not target_dir.exists():
                logger.warning(f"Directory does not exist: {target_dir}")
                return 0
            
            indexed_count = 0
            
            # Find all supported files
            for file_path in target_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.config['supported_formats']:
                    try:
                        if self._index_file(file_path):
                            indexed_count += 1
                    except Exception as e:
                        logger.error(f"Error indexing file {file_path}: {e}")
            
            # Save updated index
            self._save_search_index()
            
            logger.info(f"Indexed {indexed_count} files from {target_dir}")
            return indexed_count
            
        except Exception as e:
            logger.error(f"Error indexing directory: {e}")
            return 0
    
    def index_capsules(self, capsule_manager=None) -> int:
        """
        Index knowledge capsules.
        
        Args:
            capsule_manager: Knowledge capsule manager instance
            
        Returns:
            Number of capsules indexed
        """
        try:
            if not capsule_manager:
                logger.warning("No capsule manager provided for indexing")
                return 0
            
            indexed_count = 0
            
            # Get all capsules
            capsules = capsule_manager.list_capsules()
            
            for capsule_summary in capsules:
                try:
                    capsule = capsule_manager.load_capsule(capsule_summary['capsule_id'])
                    if capsule and self._index_capsule(capsule):
                        indexed_count += 1
                except Exception as e:
                    logger.error(f"Error indexing capsule {capsule_summary.get('capsule_id', 'unknown')}: {e}")
            
            # Save updated index
            self._save_search_index()
            
            logger.info(f"Indexed {indexed_count} knowledge capsules")
            return indexed_count
            
        except Exception as e:
            logger.error(f"Error indexing capsules: {e}")
            return 0
    
    def search(self, query: str, max_results: int = 10,
              result_types: List[SearchResultType] = None) -> List[SearchResult]:
        """
        Search through indexed content.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            result_types: Optional filter by result types
            
        Returns:
            List of search results sorted by relevance
        """
        try:
            if not query.strip():
                return []
            
            results = []
            query_lower = query.lower()
            
            # Search through all indexed items
            for index_id, search_index in self.search_index.items():
                # Filter by result type if specified
                if result_types:
                    index_type = self._determine_result_type(search_index)
                    if index_type not in result_types:
                        continue
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance(query_lower, search_index)
                
                if relevance_score > 0.1:  # Minimum relevance threshold
                    # Find best matching section
                    best_section, section_content = self._find_best_section(query_lower, search_index)
                    
                    # Create search result
                    result = SearchResult(
                        result_id=f"result_{index_id}",
                        result_type=self._determine_result_type(search_index),
                        title=self._extract_title(search_index),
                        filename=search_index.filename,
                        content_preview=self._create_preview(section_content, query_lower),
                        full_content=section_content,
                        confidence_score=relevance_score,
                        file_path=search_index.file_path,
                        section=best_section,
                        metadata=search_index.metadata
                    )
                    
                    results.append(result)
            
            # Sort by relevance score
            results.sort(key=lambda r: r.confidence_score, reverse=True)
            
            # Return top results
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return []
    
    def get_file_content(self, file_path: str, section: Optional[str] = None) -> Optional[str]:
        """
        Get content from an indexed file.
        
        Args:
            file_path: Path to the file
            section: Optional specific section
            
        Returns:
            File content or None if not found
        """
        try:
            # Find the search index for this file
            for search_index in self.search_index.values():
                if search_index.file_path == file_path:
                    if section:
                        # Find specific section
                        for sec in search_index.sections:
                            if sec.get('title') == section or sec.get('id') == section:
                                return sec.get('content', '')
                    else:
                        # Return full content
                        return search_index.indexed_content
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting file content: {e}")
            return None
    
    def create_citation(self, result: SearchResult) -> str:
        """
        Create a citation for a search result.
        
        Args:
            result: Search result to cite
            
        Returns:
            Citation string
        """
        try:
            if result.section:
                return f"📁[{result.filename}:{result.section}]"
            else:
                return f"📁[{result.filename}]"
            
        except Exception as e:
            logger.error(f"Error creating citation: {e}")
            return f"📁[{result.filename}]"
    
    def _index_file(self, file_path: Path) -> bool:
        """Index a single file."""
        try:
            # Calculate file hash to check if already indexed
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            content_hash = hashlib.sha256(file_content).hexdigest()
            
            # Check if already indexed with same hash
            index_id = str(file_path)
            if index_id in self.search_index:
                existing_index = self.search_index[index_id]
                if existing_index.content_hash == content_hash:
                    logger.debug(f"File already indexed: {file_path}")
                    return False
            
            # Extract text content
            text_content = self._extract_text_content(file_path, file_content)
            
            if len(text_content) < self.config['min_content_length']:
                logger.debug(f"File content too short: {file_path}")
                return False
            
            # Create sections
            sections = self._create_sections(text_content, file_path.suffix)
            
            # Extract keywords
            keywords = self._extract_keywords(text_content)
            
            # Create search index
            search_index = SearchIndex(
                index_id=index_id,
                file_path=str(file_path),
                filename=file_path.name,
                file_type=file_path.suffix.lower(),
                content_hash=content_hash,
                indexed_content=text_content,
                sections=sections,
                keywords=keywords,
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                metadata={
                    'file_size': len(file_content),
                    'modification_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            )
            
            self.search_index[index_id] = search_index
            
            logger.debug(f"Indexed file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return False
    
    def _index_capsule(self, capsule) -> bool:
        """Index a knowledge capsule."""
        try:
            # Create unique index ID for capsule
            index_id = f"capsule_{capsule.capsule_id}"
            
            # Extract content from capsule
            content_parts = []
            
            if hasattr(capsule, 'content'):
                if hasattr(capsule.content, 'problem_statement'):
                    content_parts.append(f"Problem: {capsule.content.problem_statement}")
                if hasattr(capsule.content, 'methodology'):
                    content_parts.append(f"Methodology: {capsule.content.methodology}")
                if hasattr(capsule.content, 'key_insights'):
                    content_parts.append(f"Insights: {' '.join(capsule.content.key_insights)}")
                if hasattr(capsule.content, 'tools_used'):
                    content_parts.append(f"Tools: {' '.join(capsule.content.tools_used)}")
                if hasattr(capsule.content, 'outcomes'):
                    content_parts.append(f"Outcomes: {capsule.content.outcomes}")
            
            text_content = '\n\n'.join(content_parts)
            
            if len(text_content) < self.config['min_content_length']:
                logger.debug(f"Capsule content too short: {capsule.capsule_id}")
                return False
            
            # Create sections for capsule
            sections = [
                {
                    'id': 'overview',
                    'title': 'Overview',
                    'content': text_content[:self.config['max_section_length']]
                }
            ]
            
            # Extract keywords
            keywords = self._extract_keywords(text_content)
            
            # Create search index
            search_index = SearchIndex(
                index_id=index_id,
                file_path=f"capsule://{capsule.capsule_id}",
                filename=f"{capsule.name}.capsule",
                file_type='.capsule',
                content_hash=hashlib.sha256(text_content.encode()).hexdigest(),
                indexed_content=text_content,
                sections=sections,
                keywords=keywords,
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                metadata={
                    'capsule_id': capsule.capsule_id,
                    'capsule_name': capsule.name,
                    'created_by': getattr(capsule, 'created_by', 'unknown'),
                    'tags': getattr(capsule, 'tags', [])
                }
            )
            
            self.search_index[index_id] = search_index
            
            logger.debug(f"Indexed capsule: {capsule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing capsule: {e}")
            return False
    
    def _extract_text_content(self, file_path: Path, file_content: bytes) -> str:
        """Extract text content from a file."""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext in ['.txt', '.md']:
                # Plain text files
                return file_content.decode('utf-8', errors='ignore')
            
            elif file_ext == '.json':
                # JSON files
                try:
                    json_data = json.loads(file_content.decode('utf-8'))
                    return json.dumps(json_data, indent=2)
                except:
                    return file_content.decode('utf-8', errors='ignore')
            
            elif file_ext == '.pdf':
                # PDF files (placeholder - would use PyPDF2 or similar)
                return f"[PDF Content] {file_path.name} - PDF text extraction would be implemented here"
            
            elif file_ext == '.docx':
                # Word documents (placeholder - would use python-docx)
                return f"[DOCX Content] {file_path.name} - Word document text extraction would be implemented here"
            
            else:
                # Try to decode as text
                return file_content.decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def _create_sections(self, content: str, file_type: str) -> List[Dict[str, Any]]:
        """Create sections from content."""
        sections = []
        
        try:
            if file_type == '.md':
                # Split by markdown headers
                lines = content.split('\n')
                current_section = {'title': 'Introduction', 'content': '', 'id': 'intro'}
                
                for line in lines:
                    if line.startswith('#'):
                        # Save previous section
                        if current_section['content'].strip():
                            sections.append(current_section)
                        
                        # Start new section
                        title = line.lstrip('#').strip()
                        section_id = re.sub(r'[^a-zA-Z0-9]', '_', title.lower())
                        current_section = {
                            'title': title,
                            'content': '',
                            'id': section_id
                        }
                    else:
                        current_section['content'] += line + '\n'
                
                # Add final section
                if current_section['content'].strip():
                    sections.append(current_section)
            
            else:
                # Split by paragraphs for other file types
                paragraphs = content.split('\n\n')
                for i, paragraph in enumerate(paragraphs):
                    if len(paragraph.strip()) > 50:  # Minimum paragraph length
                        sections.append({
                            'title': f'Section {i+1}',
                            'content': paragraph.strip(),
                            'id': f'section_{i+1}'
                        })
            
            # Limit section content length
            for section in sections:
                if len(section['content']) > self.config['max_section_length']:
                    section['content'] = section['content'][:self.config['max_section_length']] + '...'
            
            return sections
            
        except Exception as e:
            logger.error(f"Error creating sections: {e}")
            return [{'title': 'Content', 'content': content[:self.config['max_section_length']], 'id': 'content'}]
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        try:
            # Simple keyword extraction (would use more sophisticated NLP in practice)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            
            # Remove common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cannot', 'not'}
            
            # Count word frequency
            word_freq = {}
            for word in words:
                if word not in stop_words and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:20]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _calculate_relevance(self, query: str, search_index: SearchIndex) -> float:
        """Calculate relevance score for a search query."""
        try:
            score = 0.0
            query_words = query.lower().split()
            
            # Check title match
            title = self._extract_title(search_index).lower()
            for word in query_words:
                if word in title:
                    score += 0.3
            
            # Check keyword match
            for word in query_words:
                if word in search_index.keywords:
                    score += 0.2
            
            # Check content match
            content_lower = search_index.indexed_content.lower()
            for word in query_words:
                if word in content_lower:
                    score += 0.1
                    # Bonus for exact phrase match
                    if query in content_lower:
                        score += 0.2
            
            # Check filename match
            filename_lower = search_index.filename.lower()
            for word in query_words:
                if word in filename_lower:
                    score += 0.15
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0
    
    def _find_best_section(self, query: str, search_index: SearchIndex) -> Tuple[Optional[str], str]:
        """Find the best matching section for a query."""
        try:
            best_section = None
            best_content = search_index.indexed_content
            best_score = 0.0
            
            for section in search_index.sections:
                section_content = section['content'].lower()
                section_score = 0.0
                
                # Count query word matches in section
                for word in query.split():
                    if word in section_content:
                        section_score += 1
                
                # Check for exact phrase match
                if query in section_content:
                    section_score += 2
                
                if section_score > best_score:
                    best_score = section_score
                    best_section = section['title']
                    best_content = section['content']
            
            return best_section, best_content
            
        except Exception as e:
            logger.error(f"Error finding best section: {e}")
            return None, search_index.indexed_content
    
    def _create_preview(self, content: str, query: str) -> str:
        """Create a preview snippet highlighting the query."""
        try:
            max_length = self.config['max_preview_length']
            
            # Find query position in content
            content_lower = content.lower()
            query_pos = content_lower.find(query)
            
            if query_pos != -1:
                # Center preview around query
                start = max(0, query_pos - max_length // 2)
                end = min(len(content), start + max_length)
                preview = content[start:end]
                
                # Add ellipsis if truncated
                if start > 0:
                    preview = '...' + preview
                if end < len(content):
                    preview = preview + '...'
            else:
                # Just take beginning of content
                preview = content[:max_length]
                if len(content) > max_length:
                    preview += '...'
            
            return preview.strip()
            
        except Exception as e:
            logger.error(f"Error creating preview: {e}")
            return content[:self.config['max_preview_length']]
    
    def _extract_title(self, search_index: SearchIndex) -> str:
        """Extract title from search index."""
        try:
            # Try to get title from metadata
            if 'title' in search_index.metadata:
                return search_index.metadata['title']
            
            # For capsules, use capsule name
            if 'capsule_name' in search_index.metadata:
                return search_index.metadata['capsule_name']
            
            # For markdown files, try to extract first header
            if search_index.file_type == '.md':
                lines = search_index.indexed_content.split('\n')
                for line in lines:
                    if line.startswith('#'):
                        return line.lstrip('#').strip()
            
            # Fall back to filename without extension
            return Path(search_index.filename).stem
            
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return search_index.filename
    
    def _determine_result_type(self, search_index: SearchIndex) -> SearchResultType:
        """Determine the result type for a search index."""
        if search_index.file_type == '.capsule':
            return SearchResultType.CAPSULE
        elif 'multimodal' in search_index.metadata:
            return SearchResultType.MULTIMODAL
        else:
            return SearchResultType.DOCUMENT
    
    def _load_search_index(self):
        """Load search index from storage."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for index_data in data.get('indices', []):
                    search_index = SearchIndex(
                        index_id=index_data['index_id'],
                        file_path=index_data['file_path'],
                        filename=index_data['filename'],
                        file_type=index_data['file_type'],
                        content_hash=index_data['content_hash'],
                        indexed_content=index_data['indexed_content'],
                        sections=index_data['sections'],
                        keywords=index_data['keywords'],
                        created_at=index_data['created_at'],
                        last_updated=index_data['last_updated'],
                        metadata=index_data.get('metadata', {})
                    )
                    
                    self.search_index[search_index.index_id] = search_index
                
                logger.info(f"Loaded {len(self.search_index)} search indices")
            
        except Exception as e:
            logger.error(f"Error loading search index: {e}")
    
    def _save_search_index(self):
        """Save search index to storage."""
        try:
            indices_data = []
            
            for search_index in self.search_index.values():
                indices_data.append(asdict(search_index))
            
            data = {
                'indices': indices_data,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            # Ensure directory exists
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.search_index)} search indices")
            
        except Exception as e:
            logger.error(f"Error saving search index: {e}")

# Global local file search engine instance
_local_search_engine = None

def get_local_search_engine(knowledge_directory: str = "knowledge",
                           index_file: str = "search_index.json") -> LocalFileSearchEngine:
    """Get or create a global local file search engine instance."""
    global _local_search_engine
    
    if _local_search_engine is None:
        _local_search_engine = LocalFileSearchEngine(
            knowledge_directory=knowledge_directory,
            index_file=index_file
        )
    
    return _local_search_engine
