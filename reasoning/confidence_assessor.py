"""
Phase 8.1: ConfidenceAssessor Module
Intelligent assessment of retrieval quality to determine when web search escalation is needed.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence levels for retrieval quality assessment."""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH" 
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"

class RecommendationAction(Enum):
    """Recommended actions based on confidence assessment."""
    ANSWER_LOCALLY = "ANSWER_LOCALLY"
    ANSWER_WITH_CAVEAT = "ANSWER_WITH_CAVEAT"
    OFFER_WEB_SEARCH = "OFFER_WEB_SEARCH"
    RECOMMEND_WEB_SEARCH = "RECOMMEND_WEB_SEARCH"
    REQUIRE_WEB_SEARCH = "REQUIRE_WEB_SEARCH"

@dataclass
class ConfidenceAssessment:
    """Detailed confidence assessment result."""
    status: str  # CONFIDENT | NOT_CONFIDENT for backward compatibility
    confidence_level: ConfidenceLevel
    confidence_score: float  # 0.0 - 1.0
    reasons: List[str]
    recommendation: RecommendationAction
    suggested_search_query: Optional[str] = None
    explanation: Optional[str] = None

class ConfidenceAssessor:
    """
    Intelligent assessment of retrieval quality to determine when web search is needed.
    
    Analyzes search results across multiple dimensions:
    - Sufficiency: Number of relevant results
    - Relevance: Quality of semantic matches
    - Timeliness: Freshness for time-sensitive queries
    - Coverage: Completeness of information
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize confidence assessor with configurable thresholds."""
        self.config = config or {}
        
        # Sufficiency thresholds (made more strict to encourage web search)
        self.min_results_confident = self.config.get('min_results_confident', 5)  # Increased from 3
        self.min_results_acceptable = self.config.get('min_results_acceptable', 2)  # Increased from 1

        # Relevance thresholds (similarity scores) - made more strict
        self.high_relevance_threshold = self.config.get('high_relevance_threshold', 0.85)  # Increased from 0.8
        self.medium_relevance_threshold = self.config.get('medium_relevance_threshold', 0.7)  # Increased from 0.6
        self.low_relevance_threshold = self.config.get('low_relevance_threshold', 0.5)  # Increased from 0.4
        
        # Timeliness thresholds (days)
        self.recent_threshold_days = self.config.get('recent_threshold_days', 30)
        self.outdated_threshold_days = self.config.get('outdated_threshold_days', 365)
        
        # Temporal keywords for time-sensitive queries
        self.temporal_keywords = {
            'immediate': ['now', 'current', 'currently', 'today', 'this week'],
            'recent': ['latest', 'recent', 'new', 'updated', 'this year', 'this month'],
            'specific_time': ['since', 'after', 'before', 'in 2024', 'in 2023'],
            'trending': ['trending', 'popular', 'hot', 'viral', 'breaking']
        }
        
        # Query type patterns (order matters - more specific patterns first)
        self.query_patterns = {
            'news': [r'\bnews\b', r'\bbreaking\b', r'\bupdate\b', r'\bannouncement\b', r'\bheadline\b', r'\bcnn\b', r'\bbbc\b'],
            'temporal': [r'\blatest\b', r'\brecent\b', r'\bcurrent\b', r'\btoday\b', r'\bnow\b'],
            'comparative': [r'\bvs\b', r'\bversus\b', r'\bcompare\b', r'\bdifference\b'],
            'procedural': [r'\bhow to\b', r'\bsteps\b', r'\btutorial\b', r'\bguide\b'],
            'factual': [r'\bwhat is\b', r'\bdefine\b', r'\bexplain\b', r'\bmean\b']
        }
        
        logger.info("ConfidenceAssessor initialized with configurable thresholds")
    
    def assess_retrieval_quality(self, search_results: List[Dict[str, Any]], 
                               query: str = "") -> ConfidenceAssessment:
        """
        Comprehensive assessment of retrieval quality.
        
        Args:
            search_results: List of search result dictionaries with 'similarity_score' and metadata
            query: Original user query for context analysis
            
        Returns:
            ConfidenceAssessment with detailed analysis and recommendations
        """
        try:
            logger.info(f"Assessing confidence for {len(search_results)} results, query: '{query[:50]}...'")
            
            # Initialize assessment components
            reasons = []
            confidence_factors = []
            
            # 1. Sufficiency Check
            sufficiency_score, sufficiency_reasons = self._assess_sufficiency(search_results)
            confidence_factors.append(sufficiency_score)
            reasons.extend(sufficiency_reasons)
            
            # 2. Relevance Check  
            relevance_score, relevance_reasons = self._assess_relevance(search_results)
            confidence_factors.append(relevance_score)
            reasons.extend(relevance_reasons)
            
            # 3. Timeliness Check
            timeliness_score, timeliness_reasons = self._assess_timeliness(search_results, query)
            confidence_factors.append(timeliness_score)
            reasons.extend(timeliness_reasons)
            
            # 4. Query Type Analysis
            query_type = self._classify_query_type(query)
            query_score, query_reasons = self._assess_query_coverage(search_results, query, query_type)
            confidence_factors.append(query_score)
            reasons.extend(query_reasons)
            
            # Calculate overall confidence score
            overall_confidence = sum(confidence_factors) / len(confidence_factors)

            # Debug logging for confidence assessment
            logger.info(f"Confidence factors: {confidence_factors}")
            logger.info(f"Query type: {query_type}, Overall confidence: {overall_confidence:.3f}")
            logger.info(f"Reasons: {reasons}")

            # Determine confidence level and recommendation
            confidence_level = self._determine_confidence_level(overall_confidence)
            recommendation = self._determine_recommendation(confidence_level, query_type, reasons)
            
            # Generate explanation and search query suggestion
            explanation = self._generate_explanation(confidence_level, reasons, query_type)
            suggested_search_query = self._suggest_search_query(query, query_type, reasons)
            
            # Create assessment result with special handling for news queries
            query_type = self._classify_query_type(query)

            # Check if this is a document-related query (more lenient confidence)
            document_keywords = [
                'summarize', 'summary', 'analyze', 'analysis', 'document', 'pdf', 'file',
                'upload', 'content', 'text', 'report', 'paper', 'article', 'synthesis',
                'comprehensive', 'overview', 'review', 'extract', 'key points', 'main points'
            ]

            is_document_query = any(keyword in query.lower() for keyword in document_keywords)

            # For document queries, be more lenient with confidence thresholds
            if is_document_query:
                status = "CONFIDENT" if confidence_level in [ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH] else "NOT_CONFIDENT"
                logger.info(f"📄 Document query detected - using lenient confidence threshold")
            # For news queries, be more conservative about confidence
            elif query_type == 'news':
                status = "CONFIDENT" if confidence_level == ConfidenceLevel.VERY_HIGH else "NOT_CONFIDENT"
            else:
                status = "CONFIDENT" if confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH] else "NOT_CONFIDENT"

            assessment = ConfidenceAssessment(
                status=status,
                confidence_level=confidence_level,
                confidence_score=overall_confidence,
                reasons=reasons,
                recommendation=recommendation,
                suggested_search_query=suggested_search_query,
                explanation=explanation
            )
            
            logger.info(f"Confidence assessment: {confidence_level.value} ({overall_confidence:.2f}) - {recommendation.value}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error in confidence assessment: {e}")
            # Return safe fallback
            return ConfidenceAssessment(
                status="NOT_CONFIDENT",
                confidence_level=ConfidenceLevel.LOW,
                confidence_score=0.3,
                reasons=["assessment_error"],
                recommendation=RecommendationAction.OFFER_WEB_SEARCH,
                explanation="Unable to assess confidence - offering web search as fallback"
            )
    
    def _assess_sufficiency(self, search_results: List[Dict[str, Any]]) -> tuple[float, List[str]]:
        """Assess if we have sufficient number of results."""
        num_results = len(search_results)
        reasons = []
        
        if num_results >= self.min_results_confident:
            score = 1.0
        elif num_results >= self.min_results_acceptable:
            score = 0.6
            reasons.append("limited_results")
        else:
            score = 0.2
            reasons.append("insufficient_results")
            
        return score, reasons
    
    def _assess_relevance(self, search_results: List[Dict[str, Any]]) -> tuple[float, List[str]]:
        """Assess relevance quality of search results."""
        if not search_results:
            return 0.0, ["no_results"]

        # Extract similarity scores and content for quality assessment
        similarity_scores = []
        content_quality_issues = []

        for result in search_results:
            # Get similarity score
            if 'similarity_score' in result:
                similarity_scores.append(result['similarity_score'])
            elif 'score' in result:
                similarity_scores.append(result['score'])

            # Check content quality
            content = result.get('content', '')
            if isinstance(content, str):
                # Check for binary content indicators
                if 'Binary content' in content or 'bytes)' in content:
                    content_quality_issues.append("binary_content")
                # Check for very short or empty content
                elif len(content.strip()) < 50:
                    content_quality_issues.append("insufficient_content")
                # Check for metadata-only content
                elif content.startswith('Document:') and content.count('\n') < 3:
                    content_quality_issues.append("metadata_only")

        if not similarity_scores:
            return 0.5, ["no_similarity_scores"]

        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        max_similarity = max(similarity_scores)

        reasons = []

        # Penalize heavily if most results have content quality issues
        quality_issue_ratio = len(content_quality_issues) / len(search_results)
        if quality_issue_ratio >= 0.8:  # 80% of results have quality issues
            score = 0.1
            reasons.append("poor_content_quality")
        elif quality_issue_ratio >= 0.5:  # 50% of results have quality issues
            score = 0.3
            reasons.append("mixed_content_quality")
        elif max_similarity >= self.high_relevance_threshold:
            score = 1.0
        elif avg_similarity >= self.medium_relevance_threshold:
            score = 0.7
        elif max_similarity >= self.low_relevance_threshold:
            score = 0.4
            reasons.append("low_relevance")
        else:
            score = 0.2
            reasons.append("very_low_relevance")

        return score, reasons
    
    def _assess_timeliness(self, search_results: List[Dict[str, Any]], query: str) -> tuple[float, List[str]]:
        """Assess timeliness of results for time-sensitive queries."""
        is_temporal_query = self._is_temporal_query(query)
        query_type = self._classify_query_type(query)

        # News queries are always time-sensitive, even if not explicitly temporal
        if not is_temporal_query and query_type != 'news':
            return 1.0, []  # Timeliness not relevant

        # Check timestamps in results
        timestamps = []
        for result in search_results:
            metadata = result.get('metadata', {})
            if 'timestamp' in metadata:
                try:
                    timestamp = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except:
                    continue
            elif 'created_at' in metadata:
                try:
                    timestamp = datetime.fromisoformat(metadata['created_at'].replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except:
                    continue

        if not timestamps:
            # For news and temporal queries, lack of timestamps is a major issue
            if query_type == 'news':
                return 0.1, ["no_timestamps_for_news"]
            elif is_temporal_query:
                return 0.2, ["no_timestamps_for_temporal"]
            return 0.5, ["no_timestamps"]

        now = datetime.now()

        # Different thresholds for different query types
        if query_type == 'news':
            # News needs very recent content (within 7 days)
            recent_threshold = now - timedelta(days=7)
            outdated_threshold = now - timedelta(days=30)
        else:
            # Regular temporal queries use standard thresholds
            recent_threshold = now - timedelta(days=self.recent_threshold_days)
            outdated_threshold = now - timedelta(days=self.outdated_threshold_days)

        recent_count = sum(1 for ts in timestamps if ts >= recent_threshold)
        outdated_count = sum(1 for ts in timestamps if ts <= outdated_threshold)

        reasons = []

        if query_type == 'news':
            # Stricter requirements for news queries
            if recent_count >= len(timestamps) * 0.8:  # 80% very recent
                score = 1.0
            elif recent_count >= len(timestamps) * 0.5:  # 50% recent
                score = 0.4
                reasons.append("mixed_timeliness_for_news")
            else:
                score = 0.1
                reasons.append("outdated_information_for_news")
        else:
            # Standard temporal query assessment
            if recent_count >= len(timestamps) * 0.7:  # 70% recent
                score = 1.0
            elif recent_count >= len(timestamps) * 0.3:  # 30% recent
                score = 0.6
                reasons.append("mixed_timeliness")
            elif outdated_count >= len(timestamps) * 0.7:  # 70% outdated
                score = 0.2
                reasons.append("outdated_information")
            else:
                score = 0.4
                reasons.append("questionable_timeliness")

        return score, reasons
    
    def _is_temporal_query(self, query: str) -> bool:
        """Check if query contains temporal keywords indicating time sensitivity."""
        query_lower = query.lower()
        
        for category, keywords in self.temporal_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return True
                    
        # Check for year patterns
        if re.search(r'\b(20\d{2}|19\d{2})\b', query):
            return True
            
        return False
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query to understand information needs."""
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
                    
        return 'general'
    
    def _assess_query_coverage(self, search_results: List[Dict[str, Any]],
                             query: str, query_type: str) -> tuple[float, List[str]]:
        """Assess how well results cover the query requirements."""
        reasons = []

        # Check for content relevance to the actual query
        if query_type == 'news':
            # For news queries, check if content actually contains news-related information
            news_relevant_count = 0
            for result in search_results:
                content = result.get('content', '').lower()
                # Check if content contains news-related terms or is actually news content
                if any(term in content for term in ['news', 'breaking', 'report', 'update', 'announcement', 'headline']):
                    news_relevant_count += 1
                # Penalize binary/metadata content heavily for news queries
                elif 'binary content' in content or 'document:' in content:
                    continue

            if news_relevant_count == 0:
                return 0.05, ["no_news_content_found"]
            elif news_relevant_count < len(search_results) * 0.3:
                return 0.15, ["insufficient_news_content"]

        # Enhanced temporal query handling
        if query_type == 'temporal' and not self._has_recent_content(search_results):
            return 0.2, ["lacks_recent_content"]
        elif query_type == 'news' and not self._has_recent_content(search_results):
            return 0.1, ["lacks_recent_content"]  # News queries need very recent content
        elif query_type == 'comparative' and len(search_results) < 2:
            return 0.4, ["insufficient_for_comparison"]
        elif query_type == 'procedural' and not self._has_procedural_content(search_results):
            return 0.4, ["lacks_procedural_content"]

        # Additional penalty for news queries with old content
        if query_type == 'news':
            return 0.3, ["news_requires_current_sources"]
        elif query_type == 'temporal':
            return 0.5, ["temporal_query_needs_recent_info"]

        return 0.8, []
    
    def _has_recent_content(self, search_results: List[Dict[str, Any]]) -> bool:
        """Check if results contain recent content."""
        if not search_results:
            return False

        # Check for recent timestamps in results
        now = datetime.now()
        recent_threshold = now - timedelta(days=self.recent_threshold_days)

        recent_count = 0
        total_with_timestamps = 0

        for result in search_results:
            metadata = result.get('metadata', {})
            timestamp = None

            # Try to extract timestamp
            if 'timestamp' in metadata:
                try:
                    timestamp = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                    total_with_timestamps += 1
                except:
                    continue
            elif 'created_at' in metadata:
                try:
                    timestamp = datetime.fromisoformat(metadata['created_at'].replace('Z', '+00:00'))
                    total_with_timestamps += 1
                except:
                    continue

            if timestamp and timestamp >= recent_threshold:
                recent_count += 1

        # If no timestamps available, assume content is not recent for temporal queries
        if total_with_timestamps == 0:
            return False

        # Require at least 30% of timestamped content to be recent
        return recent_count >= max(1, total_with_timestamps * 0.3)
    
    def _has_procedural_content(self, search_results: List[Dict[str, Any]]) -> bool:
        """Check if results contain procedural/how-to content."""
        # Simplified check - could be enhanced
        return len(search_results) > 0
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Map confidence score to confidence level (adjusted to encourage web search)."""
        if confidence_score >= 0.95:  # Increased from 0.9
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.8:   # Increased from 0.7
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:   # Increased from 0.5
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.4:   # Increased from 0.3
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _determine_recommendation(self, confidence_level: ConfidenceLevel,
                                query_type: str, reasons: List[str]) -> RecommendationAction:
        """Determine recommended action based on confidence and context (more aggressive web search)."""
        if confidence_level == ConfidenceLevel.VERY_HIGH:
            # Even for very high confidence, suggest web search for news/temporal queries
            if query_type in ['news', 'temporal'] or 'outdated_information' in reasons:
                return RecommendationAction.OFFER_WEB_SEARCH
            return RecommendationAction.ANSWER_LOCALLY
        elif confidence_level == ConfidenceLevel.HIGH:
            # More aggressive web search suggestions
            if query_type in ['news', 'temporal'] or 'outdated_information' in reasons:
                return RecommendationAction.OFFER_WEB_SEARCH
            return RecommendationAction.ANSWER_WITH_CAVEAT
        elif confidence_level == ConfidenceLevel.MEDIUM:
            return RecommendationAction.OFFER_WEB_SEARCH
        elif confidence_level == ConfidenceLevel.LOW:
            return RecommendationAction.RECOMMEND_WEB_SEARCH
        else:
            return RecommendationAction.REQUIRE_WEB_SEARCH
    
    def _generate_explanation(self, confidence_level: ConfidenceLevel, 
                            reasons: List[str], query_type: str) -> str:
        """Generate human-readable explanation of confidence assessment."""
        explanations = {
            'insufficient_results': "I found very few relevant results in my knowledge base",
            'limited_results': "I have some relevant information, but it may not be comprehensive",
            'low_relevance': "The information I found doesn't closely match your query",
            'very_low_relevance': "I couldn't find closely relevant information",
            'outdated_information': "The information I have might be outdated",
            'mixed_timeliness': "I have a mix of recent and older information",
            'lacks_recent_content': "I don't have recent information on this topic",
            'insufficient_for_comparison': "I need more sources to provide a good comparison",
            'lacks_procedural_content': "I don't have detailed step-by-step information"
        }
        
        if confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
            return "I have good information to answer your question from my knowledge base."
        
        reason_explanations = [explanations.get(reason, reason) for reason in reasons[:2]]
        return ". ".join(reason_explanations) + "."
    
    def _suggest_search_query(self, original_query: str, query_type: str, 
                            reasons: List[str]) -> Optional[str]:
        """Suggest an optimized search query for web search."""
        if query_type == 'temporal':
            return f"{original_query} 2024 latest"
        elif 'outdated_information' in reasons:
            return f"{original_query} recent updates"
        elif query_type == 'news':
            return f"{original_query} news"
        
        return original_query

def get_confidence_assessor(config: Optional[Dict[str, Any]] = None) -> ConfidenceAssessor:
    """Factory function to create ConfidenceAssessor instance."""
    return ConfidenceAssessor(config)
