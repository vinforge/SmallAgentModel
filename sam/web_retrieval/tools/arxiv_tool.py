"""
ArXiv Search Tool - Research Paper Discovery and Download
========================================================

Intelligent arXiv search tool for SAM's automated research discovery pipeline.
Provides sophisticated paper selection, download, and quarantine management
with keyword-based scoring and relevance assessment.

Features:
- Intelligent arXiv search with relevance scoring
- Keyword-based paper selection rubric
- Automatic PDF download to quarantine directory
- Metadata extraction and enrichment
- Integration with SAM's vetting pipeline
- Error handling and retry mechanisms

Part of SAM's Task 27: Automated "Dream & Discover" Engine
Author: SAM Development Team
Version: 1.0.0
"""

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)

class ArxivSearchTool:
    """
    Tool for searching and downloading research papers from arXiv.

    Provides intelligent paper selection based on keyword scoring,
    automatic download to quarantine, and metadata extraction for
    SAM's research discovery pipeline.
    """

    def __init__(self, quarantine_dir: Optional[str] = None):
        """Initialize the arXiv search tool."""
        self.logger = logging.getLogger(__name__)

        # Quarantine directory configuration
        if quarantine_dir:
            self.quarantine_dir = Path(quarantine_dir)
        else:
            self.quarantine_dir = Path("memory/quarantine")

        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Search configuration
        self.max_results = 3
        self.timeout = 30

        # Keyword scoring weights for paper selection
        self.keyword_weights = {
            'high_priority': 3.0,    # Core research terms
            'medium_priority': 2.0,  # Related terms
            'low_priority': 1.0,     # General terms
            'domain_specific': 2.5,  # Domain-specific terminology
            'methodology': 2.0,      # Research methodology terms
            'novelty': 3.0          # Innovation and novelty indicators
        }

        # Ensure arxiv library is available
        self._ensure_arxiv_library()

        self.logger.info("ArxivSearchTool initialized")

    def search_and_download(self, query: str, insight_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Search arXiv and download the best matching paper.

        Args:
            query: Search query for arXiv
            insight_text: Original insight text for relevance scoring

        Returns:
            Dictionary with download results and metadata
        """
        try:
            self.logger.info(f"🔍 Starting arXiv search: '{query}'")

            # Search arXiv
            search_results = self._search_arxiv(query)

            if not search_results:
                return {
                    'success': False,
                    'error': 'No papers found for query',
                    'query': query,
                    'results_count': 0
                }

            # Select best paper using scoring rubric
            best_paper = self._select_best_paper(search_results, query, insight_text)

            if not best_paper:
                return {
                    'success': False,
                    'error': 'No suitable papers found after scoring',
                    'query': query,
                    'results_count': len(search_results)
                }

            # Download the selected paper
            download_result = self._download_paper(best_paper)

            if download_result['success']:
                self.logger.info(f"✅ Successfully downloaded paper: {best_paper['title']}")

                return {
                    'success': True,
                    'paper_metadata': {
                        'title': best_paper['title'],
                        'authors': best_paper['authors'],
                        'summary': best_paper['summary'],
                        'arxiv_id': best_paper['arxiv_id'],
                        'published': best_paper['published'],
                        'categories': best_paper['categories'],
                        'pdf_url': best_paper['pdf_url'],
                        'selection_score': best_paper['selection_score'],
                        'selection_reasons': best_paper['selection_reasons']
                    },
                    'local_path': download_result['local_path'],
                    'query': query,
                    'insight_text': insight_text,
                    'download_timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f"Download failed: {download_result['error']}",
                    'paper_metadata': best_paper,
                    'query': query
                }

        except Exception as e:
            self.logger.error(f"ArXiv search and download failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }

    def _search_arxiv(self, query: str) -> List[Dict[str, Any]]:
        """Search arXiv and return formatted results."""
        try:
            import arxiv

            # Create search with relevance sorting
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            results = []
            for result in search.results():
                # Extract paper information
                paper_data = {
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary,
                    'arxiv_id': result.entry_id.split('/')[-1],  # Extract ID from URL
                    'published': result.published.isoformat(),
                    'categories': result.categories,
                    'pdf_url': result.pdf_url,
                    'entry_id': result.entry_id,
                    'primary_category': result.primary_category
                }

                results.append(paper_data)
                self.logger.debug(f"Found paper: {paper_data['title']}")

            self.logger.info(f"📄 Found {len(results)} papers for query: '{query}'")
            return results

        except Exception as e:
            self.logger.error(f"ArXiv search failed: {e}")
            return []


    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Public search API expected by DeepResearchStrategy.
        Returns a list of paper dicts limited to max_results.
        """
        try:
            results = self._search_arxiv(query)
            if not results:
                return []
            return results[:max_results]
        except Exception as e:
            self.logger.error(f"search_papers failed: {e}")
            return []

    def _select_best_paper(self, papers: List[Dict[str, Any]], query: str,
                          insight_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Select the best paper using keyword-based scoring rubric."""
        try:
            scored_papers = []

            for paper in papers:
                score, reasons = self._calculate_paper_score(paper, query, insight_text)
                paper['selection_score'] = score
                paper['selection_reasons'] = reasons
                scored_papers.append(paper)

                self.logger.debug(f"Paper scored: {paper['title']} - Score: {score:.2f}")

            # Sort by score (highest first)
            scored_papers.sort(key=lambda p: p['selection_score'], reverse=True)

            best_paper = scored_papers[0]
            self.logger.info(f"🏆 Selected best paper: {best_paper['title']} (Score: {best_paper['selection_score']:.2f})")

            return best_paper

        except Exception as e:
            self.logger.error(f"Paper selection failed: {e}")
            return None

    def _calculate_paper_score(self, paper: Dict[str, Any], query: str,
                              insight_text: Optional[str] = None) -> Tuple[float, List[str]]:
        """Calculate relevance score for a paper using keyword-based rubric."""
        score = 0.0
        reasons = []

        # Combine text for analysis
        paper_text = f"{paper['title']} {paper['summary']}".lower()
        query_lower = query.lower()
        insight_lower = insight_text.lower() if insight_text else ""

        # 1. Query term matching (base score)
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        matched_terms = sum(1 for term in query_terms if term in paper_text)
        query_score = (matched_terms / len(query_terms)) * 10.0 if query_terms else 0.0
        score += query_score

        if matched_terms > 0:
            reasons.append(f"Query terms matched: {matched_terms}/{len(query_terms)}")

        # 2. High-priority keywords (innovation, novel, breakthrough, etc.)
        high_priority_keywords = [
            'novel', 'new', 'innovative', 'breakthrough', 'advanced', 'improved',
            'state-of-the-art', 'cutting-edge', 'pioneering', 'revolutionary'
        ]
        high_priority_matches = sum(1 for kw in high_priority_keywords if kw in paper_text)
        if high_priority_matches > 0:
            score += high_priority_matches * self.keyword_weights['high_priority']
            reasons.append(f"Innovation keywords: {high_priority_matches}")

        # 3. Methodology keywords
        methodology_keywords = [
            'method', 'approach', 'algorithm', 'framework', 'model', 'technique',
            'system', 'architecture', 'design', 'implementation'
        ]
        methodology_matches = sum(1 for kw in methodology_keywords if kw in paper_text)
        if methodology_matches > 0:
            score += methodology_matches * self.keyword_weights['methodology']
            reasons.append(f"Methodology terms: {methodology_matches}")

        # 4. Recency bonus (prefer recent papers)
        try:
            published_date = datetime.fromisoformat(paper['published'].replace('Z', '+00:00'))
            days_old = (datetime.now().replace(tzinfo=published_date.tzinfo) - published_date).days

            if days_old <= 30:
                recency_bonus = 5.0
                reasons.append("Very recent (≤30 days)")
            elif days_old <= 90:
                recency_bonus = 3.0
                reasons.append("Recent (≤90 days)")
            elif days_old <= 365:
                recency_bonus = 1.0
                reasons.append("Recent (≤1 year)")
            else:
                recency_bonus = 0.0

            score += recency_bonus

        except Exception:
            pass  # Skip recency scoring if date parsing fails

        # 5. Insight relevance (if provided)
        if insight_text:
            insight_terms = set(re.findall(r'\b\w+\b', insight_lower))
            insight_matches = sum(1 for term in insight_terms if term in paper_text and len(term) > 3)
            if insight_matches > 0:
                insight_score = (insight_matches / len(insight_terms)) * 5.0
                score += insight_score
                reasons.append(f"Insight relevance: {insight_matches}/{len(insight_terms)} terms")

        # 6. Category relevance (prefer CS, AI, ML categories)
        preferred_categories = ['cs.', 'stat.ml', 'math.', 'physics.']
        category_bonus = 0.0
        for category in paper.get('categories', []):
            if any(cat in category.lower() for cat in preferred_categories):
                category_bonus += 2.0

        if category_bonus > 0:
            score += category_bonus
            reasons.append(f"Preferred categories: {paper.get('categories', [])}")

        return score, reasons

    def _download_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Download paper PDF to quarantine directory."""
        try:
            import arxiv

            # Create filename from arXiv ID
            arxiv_id = paper['arxiv_id']
            filename = f"{arxiv_id}.pdf"
            local_path = self.quarantine_dir / filename

            # Check if file already exists
            if local_path.exists():
                self.logger.info(f"📄 Paper already exists: {local_path}")
                return {
                    'success': True,
                    'local_path': str(local_path),
                    'already_existed': True
                }

            # Create arxiv.Result object for download
            result = next(arxiv.Search(id_list=[arxiv_id]).results())

            # Download PDF
            self.logger.info(f"⬇️ Downloading paper: {paper['title']}")
            result.download_pdf(dirpath=str(self.quarantine_dir), filename=filename)

            # Verify download
            if local_path.exists() and local_path.stat().st_size > 0:
                self.logger.info(f"✅ Download successful: {local_path} ({local_path.stat().st_size} bytes)")
                return {
                    'success': True,
                    'local_path': str(local_path),
                    'file_size': local_path.stat().st_size,
                    'already_existed': False
                }
            else:
                return {
                    'success': False,
                    'error': 'Downloaded file is empty or missing'
                }

        except Exception as e:
            self.logger.error(f"Paper download failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _ensure_arxiv_library(self) -> None:
        """Ensure the arxiv library is installed."""
        try:
            import arxiv
            self.logger.debug("ArXiv library is available")
        except ImportError:
            self.logger.info("Installing arxiv library...")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 'arxiv'
                ])
                self.logger.info("✅ ArXiv library installed successfully")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to install arxiv library: {e}")
                raise RuntimeError("ArXiv library installation failed")

    def get_quarantine_files(self) -> List[Dict[str, Any]]:
        """Get list of files in quarantine directory."""
        try:
            files = []
            for file_path in self.quarantine_dir.glob("*.pdf"):
                files.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })

            return files

        except Exception as e:
            self.logger.error(f"Failed to list quarantine files: {e}")
            return []

    def cleanup_quarantine(self, days_old: int = 30) -> int:
        """Clean up old files from quarantine directory."""
        try:
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            removed_count = 0

            for file_path in self.quarantine_dir.glob("*.pdf"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
                    self.logger.debug(f"Removed old quarantine file: {file_path}")

            if removed_count > 0:
                self.logger.info(f"🧹 Cleaned up {removed_count} old quarantine files")

            return removed_count

        except Exception as e:
            self.logger.error(f"Quarantine cleanup failed: {e}")
            return 0

# Global instance for easy access
_arxiv_tool = None

def get_arxiv_tool() -> ArxivSearchTool:
    """Get the global arXiv search tool instance."""
    global _arxiv_tool
    if _arxiv_tool is None:
        _arxiv_tool = ArxivSearchTool()
    return _arxiv_tool
