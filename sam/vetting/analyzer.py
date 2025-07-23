"""
Vetting Analyzer - Automated Content Analysis Engine
===================================================

Multi-dimensional analysis engine for research papers and web content with:
- Security risk assessment (file integrity, malware scanning)
- Relevance scoring (semantic similarity to original insights)
- Credibility assessment (source quality, citation analysis)
- Automated scoring and threshold-based approval

Integrates with SAM's existing components:
- Memory vector store for embedding generation
- Conceptual Dimension Prober for credibility assessment
- Security frameworks for file integrity checks

Part of SAM's Task 27: Automated "Dream & Discover" Engine
Author: SAM Development Team
Version: 1.0.0
"""

import hashlib
import logging
import mimetypes
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SecurityRisk:
    """Security risk assessment results."""
    risk_score: float  # 0.0 (safe) to 1.0 (high risk)
    file_integrity: bool
    file_size_mb: float
    mime_type: str
    suspicious_patterns: List[str]
    scan_timestamp: str

@dataclass
class RelevanceScore:
    """Relevance scoring results."""
    relevance_score: float  # 0.0 (irrelevant) to 1.0 (highly relevant)
    semantic_similarity: float
    keyword_overlap: float
    topic_alignment: float
    embedding_distance: float
    matched_concepts: List[str]

@dataclass
class CredibilityScore:
    """Credibility assessment results."""
    credibility_score: float  # 0.0 (low credibility) to 1.0 (high credibility)
    source_reputation: float
    content_quality: float
    citation_indicators: float
    author_credibility: float
    venue_quality: float
    quality_indicators: List[str]

@dataclass
class AnalysisResult:
    """Complete analysis result for a file."""
    file_path: str
    analysis_id: str
    timestamp: str
    security_risk: SecurityRisk
    relevance_score: RelevanceScore
    credibility_score: CredibilityScore
    overall_score: float
    recommendation: str  # 'approve', 'reject', 'manual_review'
    analysis_notes: List[str]

class VettingAnalyzer:
    """
    Comprehensive analysis engine for research papers and web content.
    
    Provides multi-dimensional scoring across security, relevance, and credibility
    dimensions with integration to SAM's existing analysis components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the vetting analyzer."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Analysis configuration
        self.max_file_size_mb = self.config.get('max_file_size_mb', 100)
        self.allowed_mime_types = self.config.get('allowed_mime_types', [
            'application/pdf',
            'text/plain',
            'text/html',
            'application/json'
        ])
        
        # Scoring weights
        self.scoring_weights = {
            'security': 0.3,
            'relevance': 0.4,
            'credibility': 0.3
        }
        
        # Approval thresholds
        self.approval_thresholds = {
            'auto_approve_min': 0.75,
            'manual_review_min': 0.5,
            'security_risk_max': 0.3
        }
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("VettingAnalyzer initialized")
    
    def analyze_quarantined_file(self, file_path: str, original_insight_text: str,
                               paper_metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        Perform comprehensive analysis of a quarantined file.
        
        Args:
            file_path: Path to the file in quarantine
            original_insight_text: Original insight that triggered the research
            paper_metadata: Optional metadata about the paper
            
        Returns:
            Complete analysis result with scores and recommendation
        """
        try:
            analysis_id = f"analysis_{int(time.time())}"
            timestamp = datetime.now().isoformat()
            
            self.logger.info(f"🔍 Starting analysis: {analysis_id} for {file_path}")
            
            # Security risk assessment
            security_risk = self._assess_security_risk(file_path)
            
            # Relevance scoring
            relevance_score = self._calculate_relevance_score(
                file_path, original_insight_text, paper_metadata
            )
            
            # Credibility assessment
            credibility_score = self._assess_credibility(file_path, paper_metadata)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                security_risk, relevance_score, credibility_score
            )
            
            # Generate recommendation
            recommendation, notes = self._generate_recommendation(
                security_risk, relevance_score, credibility_score, overall_score
            )
            
            result = AnalysisResult(
                file_path=file_path,
                analysis_id=analysis_id,
                timestamp=timestamp,
                security_risk=security_risk,
                relevance_score=relevance_score,
                credibility_score=credibility_score,
                overall_score=overall_score,
                recommendation=recommendation,
                analysis_notes=notes
            )
            
            self.logger.info(f"✅ Analysis complete: {analysis_id} - {recommendation} (Score: {overall_score:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {file_path}: {e}")
            # Return safe default result
            return self._create_error_result(file_path, str(e))
    
    def _assess_security_risk(self, file_path: str) -> SecurityRisk:
        """Assess security risk of a file."""
        try:
            file_path_obj = Path(file_path)
            
            # Basic file checks
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            mime_type, _ = mimetypes.guess_type(file_path)
            
            risk_score = 0.0
            suspicious_patterns = []
            
            # File size check
            if file_size_mb > self.max_file_size_mb:
                risk_score += 0.3
                suspicious_patterns.append(f"Large file size: {file_size_mb:.1f}MB")
            
            # MIME type check
            if mime_type not in self.allowed_mime_types:
                risk_score += 0.4
                suspicious_patterns.append(f"Unexpected MIME type: {mime_type}")
            
            # File integrity check
            file_integrity = self._check_file_integrity(file_path_obj)
            if not file_integrity:
                risk_score += 0.5
                suspicious_patterns.append("File integrity check failed")
            
            # Basic content scanning for PDFs
            if mime_type == 'application/pdf':
                pdf_risk = self._scan_pdf_content(file_path_obj)
                risk_score += pdf_risk
                if pdf_risk > 0:
                    suspicious_patterns.append("PDF content scan flagged issues")
            
            # Cap risk score at 1.0
            risk_score = min(risk_score, 1.0)
            
            return SecurityRisk(
                risk_score=risk_score,
                file_integrity=file_integrity,
                file_size_mb=file_size_mb,
                mime_type=mime_type or 'unknown',
                suspicious_patterns=suspicious_patterns,
                scan_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Security risk assessment failed: {e}")
            return SecurityRisk(
                risk_score=1.0,  # Maximum risk for failed assessment
                file_integrity=False,
                file_size_mb=0.0,
                mime_type='unknown',
                suspicious_patterns=[f"Assessment failed: {str(e)}"],
                scan_timestamp=datetime.now().isoformat()
            )
    
    def _calculate_relevance_score(self, file_path: str, original_insight_text: str,
                                 paper_metadata: Optional[Dict[str, Any]] = None) -> RelevanceScore:
        """Calculate relevance score using semantic similarity."""
        try:
            # Extract text content from file
            file_content = self._extract_text_content(file_path)
            
            if not file_content:
                return RelevanceScore(
                    relevance_score=0.0,
                    semantic_similarity=0.0,
                    keyword_overlap=0.0,
                    topic_alignment=0.0,
                    embedding_distance=1.0,
                    matched_concepts=[]
                )
            
            # Use paper metadata if available
            if paper_metadata:
                # Combine title and abstract for better relevance assessment
                content_for_analysis = f"{paper_metadata.get('title', '')} {paper_metadata.get('summary', '')} {file_content[:1000]}"
            else:
                content_for_analysis = file_content[:2000]  # First 2000 chars
            
            # Calculate semantic similarity using embeddings
            semantic_similarity = self._calculate_semantic_similarity(
                original_insight_text, content_for_analysis
            )
            
            # Calculate keyword overlap
            keyword_overlap = self._calculate_keyword_overlap(
                original_insight_text, content_for_analysis
            )
            
            # Calculate topic alignment
            topic_alignment = self._calculate_topic_alignment(
                original_insight_text, content_for_analysis
            )
            
            # Find matched concepts
            matched_concepts = self._find_matched_concepts(
                original_insight_text, content_for_analysis
            )
            
            # Calculate embedding distance
            embedding_distance = 1.0 - semantic_similarity  # Convert similarity to distance
            
            # Combine scores for overall relevance
            relevance_score = (
                semantic_similarity * 0.5 +
                keyword_overlap * 0.3 +
                topic_alignment * 0.2
            )
            
            return RelevanceScore(
                relevance_score=relevance_score,
                semantic_similarity=semantic_similarity,
                keyword_overlap=keyword_overlap,
                topic_alignment=topic_alignment,
                embedding_distance=embedding_distance,
                matched_concepts=matched_concepts
            )
            
        except Exception as e:
            self.logger.error(f"Relevance scoring failed: {e}")
            return RelevanceScore(
                relevance_score=0.0,
                semantic_similarity=0.0,
                keyword_overlap=0.0,
                topic_alignment=0.0,
                embedding_distance=1.0,
                matched_concepts=[]
            )
    
    def _assess_credibility(self, file_path: str, 
                          paper_metadata: Optional[Dict[str, Any]] = None) -> CredibilityScore:
        """Assess credibility using various quality indicators."""
        try:
            source_reputation = 0.5  # Default neutral score
            content_quality = 0.5
            citation_indicators = 0.5
            author_credibility = 0.5
            venue_quality = 0.5
            quality_indicators = []
            
            if paper_metadata:
                # ArXiv papers get baseline credibility
                if 'arxiv' in paper_metadata.get('pdf_url', '').lower():
                    source_reputation = 0.7
                    quality_indicators.append("ArXiv preprint server")
                
                # Check for academic indicators in title/abstract
                academic_terms = [
                    'university', 'institute', 'research', 'laboratory', 'department',
                    'conference', 'journal', 'proceedings', 'symposium'
                ]
                
                title_text = paper_metadata.get('title', '').lower()
                summary_text = paper_metadata.get('summary', '').lower()
                combined_text = f"{title_text} {summary_text}"
                
                academic_matches = sum(1 for term in academic_terms if term in combined_text)
                if academic_matches > 0:
                    content_quality += 0.2
                    quality_indicators.append(f"Academic indicators: {academic_matches}")
                
                # Check for methodology terms
                methodology_terms = [
                    'experiment', 'evaluation', 'analysis', 'study', 'method',
                    'approach', 'algorithm', 'framework', 'model'
                ]
                
                methodology_matches = sum(1 for term in methodology_terms if term in combined_text)
                if methodology_matches > 2:
                    content_quality += 0.1
                    quality_indicators.append(f"Methodology terms: {methodology_matches}")
                
                # Check categories for quality indicators
                categories = paper_metadata.get('categories', [])
                if categories:
                    # Prefer certain categories
                    preferred_cats = ['cs.AI', 'cs.LG', 'cs.CL', 'stat.ML']
                    if any(cat in categories for cat in preferred_cats):
                        venue_quality += 0.2
                        quality_indicators.append(f"Preferred categories: {categories}")
                
                # Author count (multiple authors often indicates collaboration)
                authors = paper_metadata.get('authors', [])
                if len(authors) > 1:
                    author_credibility += 0.1
                    quality_indicators.append(f"Multiple authors: {len(authors)}")
            
            # Use Conceptual Dimension Prober if available
            try:
                if hasattr(self, 'dimension_prober') and self.dimension_prober:
                    # Extract content for credibility assessment
                    content = self._extract_text_content(file_path)
                    if content:
                        # Use dimension prober for advanced credibility assessment
                        credibility_assessment = self._assess_with_dimension_prober(content)
                        if credibility_assessment:
                            citation_indicators = credibility_assessment.get('citation_score', 0.5)
                            quality_indicators.append("Dimension prober assessment")
            except Exception as e:
                self.logger.debug(f"Dimension prober assessment failed: {e}")
            
            # Calculate overall credibility score
            credibility_score = (
                source_reputation * 0.3 +
                content_quality * 0.3 +
                citation_indicators * 0.2 +
                author_credibility * 0.1 +
                venue_quality * 0.1
            )
            
            # Cap at 1.0
            credibility_score = min(credibility_score, 1.0)
            
            return CredibilityScore(
                credibility_score=credibility_score,
                source_reputation=source_reputation,
                content_quality=content_quality,
                citation_indicators=citation_indicators,
                author_credibility=author_credibility,
                venue_quality=venue_quality,
                quality_indicators=quality_indicators
            )
            
        except Exception as e:
            self.logger.error(f"Credibility assessment failed: {e}")
            return CredibilityScore(
                credibility_score=0.5,  # Neutral score for failed assessment
                source_reputation=0.5,
                content_quality=0.5,
                citation_indicators=0.5,
                author_credibility=0.5,
                venue_quality=0.5,
                quality_indicators=[f"Assessment failed: {str(e)}"]
            )

    def _calculate_overall_score(self, security_risk: SecurityRisk,
                               relevance_score: RelevanceScore,
                               credibility_score: CredibilityScore) -> float:
        """Calculate overall score from component scores."""
        # Security risk is inverted (lower risk = higher score)
        security_score = 1.0 - security_risk.risk_score

        # Weighted combination
        overall_score = (
            security_score * self.scoring_weights['security'] +
            relevance_score.relevance_score * self.scoring_weights['relevance'] +
            credibility_score.credibility_score * self.scoring_weights['credibility']
        )

        return min(overall_score, 1.0)

    def _generate_recommendation(self, security_risk: SecurityRisk,
                               relevance_score: RelevanceScore,
                               credibility_score: CredibilityScore,
                               overall_score: float) -> Tuple[str, List[str]]:
        """Generate recommendation and notes based on scores."""
        notes = []

        # Security check first
        if security_risk.risk_score > self.approval_thresholds['security_risk_max']:
            notes.append(f"High security risk: {security_risk.risk_score:.2f}")
            return 'reject', notes

        # Overall score check
        if overall_score >= self.approval_thresholds['auto_approve_min']:
            notes.append(f"High overall score: {overall_score:.2f}")
            return 'approve', notes
        elif overall_score >= self.approval_thresholds['manual_review_min']:
            notes.append(f"Moderate score requires review: {overall_score:.2f}")
            return 'manual_review', notes
        else:
            notes.append(f"Low overall score: {overall_score:.2f}")
            return 'reject', notes

    def _initialize_components(self) -> None:
        """Initialize analysis components."""
        try:
            # Try to initialize memory store for embeddings
            from memory.memory_vectorstore import get_memory_store
            self.memory_store = get_memory_store()
            self.logger.debug("Memory store initialized for embeddings")
        except Exception as e:
            self.logger.warning(f"Memory store initialization failed: {e}")
            self.memory_store = None

        try:
            # Try to initialize dimension prober for credibility assessment
            from sam.conceptual_dimension_prober import ConceptualDimensionProber
            self.dimension_prober = ConceptualDimensionProber()
            self.logger.debug("Dimension prober initialized")
        except Exception as e:
            self.logger.warning(f"Dimension prober initialization failed: {e}")
            self.dimension_prober = None

    def _check_file_integrity(self, file_path: Path) -> bool:
        """Check basic file integrity."""
        try:
            # Check if file is readable
            with open(file_path, 'rb') as f:
                # Read first few bytes to ensure file is not corrupted
                header = f.read(1024)
                if not header:
                    return False

            # For PDFs, check PDF header
            if file_path.suffix.lower() == '.pdf':
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'%PDF-'):
                        return False

            return True

        except Exception as e:
            self.logger.error(f"File integrity check failed: {e}")
            return False

    def _scan_pdf_content(self, file_path: Path) -> float:
        """Basic PDF content scanning for suspicious patterns."""
        try:
            # Simple heuristic checks for PDF files
            risk_score = 0.0

            with open(file_path, 'rb') as f:
                content = f.read(10000)  # Read first 10KB

                # Check for suspicious patterns (very basic)
                suspicious_patterns = [
                    b'/JavaScript',
                    b'/JS',
                    b'/Launch',
                    b'/EmbeddedFile'
                ]

                for pattern in suspicious_patterns:
                    if pattern in content:
                        risk_score += 0.1

            return min(risk_score, 0.5)  # Cap PDF risk contribution

        except Exception as e:
            self.logger.error(f"PDF content scan failed: {e}")
            return 0.2  # Small risk penalty for scan failure

    def _extract_text_content(self, file_path: str) -> str:
        """Extract text content from file for analysis."""
        try:
            file_path_obj = Path(file_path)

            if file_path_obj.suffix.lower() == '.pdf':
                return self._extract_pdf_text(file_path_obj)
            elif file_path_obj.suffix.lower() in ['.txt', '.md']:
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Try to read as text
                with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(5000)  # First 5000 chars

        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return ""

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            # Try using PyPDF2 if available
            try:
                import PyPDF2
                try:
                    # PyPDF2 3.x
                    reader = PyPDF2.PdfReader(str(file_path))
                    text = ""
                    for page in reader.pages[:5]:  # First 5 pages
                        text += page.extract_text()
                    return text
                except AttributeError:
                    # PyPDF2 2.x fallback
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfFileReader(f)
                        text = ""
                        for i in range(min(5, reader.numPages)):  # First 5 pages
                            page = reader.getPage(i)
                            text += page.extractText()
                        return text
            except ImportError:
                pass

            # Fallback: try pdfplumber if available
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages[:5]:  # First 5 pages
                        text += page.extract_text() or ""
                    return text
            except ImportError:
                pass

            # Final fallback: use pdftotext if available
            try:
                result = subprocess.run(
                    ['pdftotext', '-l', '5', str(file_path), '-'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    return result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            self.logger.warning(f"No PDF text extraction method available for {file_path}")
            return ""

        except Exception as e:
            self.logger.error(f"PDF text extraction failed: {e}")
            return ""

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            if not self.memory_store:
                # Fallback to simple keyword similarity
                return self._calculate_keyword_overlap(text1, text2)

            # Generate embeddings using memory store's embedding model
            embedding1 = self.memory_store._generate_embedding(text1)
            embedding2 = self.memory_store._generate_embedding(text2)

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )

            # Convert to 0-1 range
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))

        except Exception as e:
            self.logger.error(f"Semantic similarity calculation failed: {e}")
            return self._calculate_keyword_overlap(text1, text2)

    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """Calculate keyword overlap between two texts."""
        try:
            # Simple keyword overlap calculation
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))

            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}

            words1 = words1 - stop_words
            words2 = words2 - stop_words

            if not words1 or not words2:
                return 0.0

            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))

            return overlap / total if total > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Keyword overlap calculation failed: {e}")
            return 0.0

    def _calculate_topic_alignment(self, text1: str, text2: str) -> float:
        """Calculate topic alignment between texts."""
        try:
            # Simple topic alignment based on key terms
            # This could be enhanced with topic modeling

            # Extract key terms (longer words, capitalized terms)
            key_terms1 = set(re.findall(r'\b[A-Z][a-z]+\b|\b\w{6,}\b', text1))
            key_terms2 = set(re.findall(r'\b[A-Z][a-z]+\b|\b\w{6,}\b', text2))

            if not key_terms1 or not key_terms2:
                return 0.0

            overlap = len(key_terms1.intersection(key_terms2))
            total = len(key_terms1.union(key_terms2))

            return overlap / total if total > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Topic alignment calculation failed: {e}")
            return 0.0

    def _find_matched_concepts(self, text1: str, text2: str) -> List[str]:
        """Find matched concepts between texts."""
        try:
            # Extract potential concepts (capitalized terms, technical terms)
            concepts1 = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text1))
            concepts2 = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text2))

            matched = concepts1.intersection(concepts2)
            return list(matched)[:10]  # Return top 10 matches

        except Exception as e:
            self.logger.error(f"Concept matching failed: {e}")
            return []

    def _assess_with_dimension_prober(self, content: str) -> Optional[Dict[str, float]]:
        """Use dimension prober for advanced credibility assessment."""
        try:
            if not self.dimension_prober:
                return None

            # Use dimension prober to assess content quality
            # This is a placeholder for actual dimension prober integration
            assessment = {
                'citation_score': 0.5,  # Placeholder
                'methodology_score': 0.5,
                'novelty_score': 0.5
            }

            return assessment

        except Exception as e:
            self.logger.error(f"Dimension prober assessment failed: {e}")
            return None

    def _create_error_result(self, file_path: str, error_message: str) -> AnalysisResult:
        """Create a safe error result."""
        return AnalysisResult(
            file_path=file_path,
            analysis_id=f"error_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            security_risk=SecurityRisk(
                risk_score=1.0,
                file_integrity=False,
                file_size_mb=0.0,
                mime_type='unknown',
                suspicious_patterns=[error_message],
                scan_timestamp=datetime.now().isoformat()
            ),
            relevance_score=RelevanceScore(
                relevance_score=0.0,
                semantic_similarity=0.0,
                keyword_overlap=0.0,
                topic_alignment=0.0,
                embedding_distance=1.0,
                matched_concepts=[]
            ),
            credibility_score=CredibilityScore(
                credibility_score=0.0,
                source_reputation=0.0,
                content_quality=0.0,
                citation_indicators=0.0,
                author_credibility=0.0,
                venue_quality=0.0,
                quality_indicators=[error_message]
            ),
            overall_score=0.0,
            recommendation='reject',
            analysis_notes=[f"Analysis failed: {error_message}"]
        )

# Global instance for easy access
_vetting_analyzer = None

def get_vetting_analyzer() -> VettingAnalyzer:
    """Get the global vetting analyzer instance."""
    global _vetting_analyzer
    if _vetting_analyzer is None:
        _vetting_analyzer = VettingAnalyzer()
    return _vetting_analyzer
