#!/usr/bin/env python3
"""
Natural Language Query Parser for SAM Phase 3
Parses natural language filters and intent from user queries to enable
intuitive dimension-aware search capabilities.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Query intent classification."""
    SEARCH = "search"
    FILTER = "filter"
    COMPARE = "compare"
    ANALYZE = "analyze"
    SUMMARIZE = "summarize"

class DimensionFilter(Enum):
    """Dimension filter types."""
    HIGH = "high"
    LOW = "low"
    MEDIUM = "medium"
    NONE = "none"

@dataclass
class ParsedQuery:
    """Parsed query with extracted intent and filters."""
    original_query: str
    clean_query: str  # Query with filter terms removed
    intent: QueryIntent
    dimension_filters: Dict[str, DimensionFilter]
    profile_hint: Optional[str]
    confidence: float
    filter_phrases: List[str]

class NaturalLanguageQueryParser:
    """
    Advanced natural language query parser that extracts dimension filters,
    intent, and profile hints from user queries.
    """
    
    def __init__(self):
        """Initialize the query parser with pattern definitions."""
        self._init_dimension_patterns()
        self._init_intent_patterns()
        self._init_profile_patterns()
    
    def _init_dimension_patterns(self):
        """Initialize dimension filter patterns."""
        self.dimension_patterns = {
            # Utility patterns
            'utility': {
                'high': [
                    r'\b(?:high[- ]utility|very useful|highly valuable|extremely useful)\b',
                    r'\b(?:valuable|useful|beneficial|practical|actionable)\b',
                    r'\b(?:important|critical|essential|vital|key)\b'
                ],
                'low': [
                    r'\b(?:low[- ]utility|not useful|little value|useless)\b',
                    r'\b(?:irrelevant|unimportant|trivial|minor)\b'
                ]
            },
            
            # Risk/Danger patterns
            'danger': {
                'high': [
                    r'\b(?:high[- ]risk|dangerous|risky|hazardous|unsafe)\b',
                    r'\b(?:threat|security risk|vulnerability|exploit)\b',
                    r'\b(?:classified|sensitive|restricted|confidential)\b'
                ],
                'low': [
                    r'\b(?:low[- ]risk|safe|secure|harmless|benign)\b',
                    r'\b(?:public|unclassified|open|non[- ]sensitive)\b'
                ]
            },
            
            # Complexity patterns
            'complexity': {
                'high': [
                    r'\b(?:complex|complicated|sophisticated|advanced|difficult)\b',
                    r'\b(?:technical|detailed|in[- ]depth|comprehensive)\b',
                    r'\b(?:expert[- ]level|specialized|intricate)\b'
                ],
                'low': [
                    r'\b(?:simple|easy|basic|straightforward|elementary)\b',
                    r'\b(?:beginner|introductory|overview|summary)\b'
                ]
            },
            
            # Innovation/Novelty patterns
            'novelty': {
                'high': [
                    r'\b(?:innovative|novel|cutting[- ]edge|breakthrough|revolutionary)\b',
                    r'\b(?:new|latest|emerging|state[- ]of[- ]the[- ]art|pioneering)\b',
                    r'\b(?:experimental|prototype|research|development)\b'
                ],
                'low': [
                    r'\b(?:traditional|conventional|standard|established|mature)\b',
                    r'\b(?:legacy|old|existing|current|proven)\b'
                ]
            },
            
            # Business patterns
            'market_impact': {
                'high': [
                    r'\b(?:high[- ]impact|game[- ]changing|disruptive|transformative)\b',
                    r'\b(?:strategic|competitive advantage|market[- ]leading)\b'
                ],
                'low': [
                    r'\b(?:low[- ]impact|incremental|minor improvement)\b'
                ]
            },
            
            'roi_potential': {
                'high': [
                    r'\b(?:profitable|high[- ]ROI|lucrative|valuable|cost[- ]effective)\b',
                    r'\b(?:revenue|profit|financial benefit|return)\b'
                ],
                'low': [
                    r'\b(?:expensive|costly|low[- ]ROI|unprofitable)\b'
                ]
            },
            
            'feasibility': {
                'high': [
                    r'\b(?:feasible|practical|implementable|viable|achievable)\b',
                    r'\b(?:realistic|doable|possible|workable)\b'
                ],
                'low': [
                    r'\b(?:infeasible|impractical|unrealistic|impossible)\b'
                ]
            },
            
            # Legal patterns
            'compliance_risk': {
                'high': [
                    r'\b(?:non[- ]compliant|violation|breach|illegal|prohibited)\b',
                    r'\b(?:regulatory risk|compliance issue|legal problem)\b'
                ],
                'low': [
                    r'\b(?:compliant|legal|approved|authorized|permitted)\b',
                    r'\b(?:regulatory[- ]approved|certified|validated)\b'
                ]
            },
            
            # Quality patterns
            'credibility': {
                'high': [
                    r'\b(?:reliable|trustworthy|credible|authoritative|verified)\b',
                    r'\b(?:peer[- ]reviewed|validated|proven|established)\b',
                    r'\b(?:high[- ]quality|excellent|superior|top[- ]tier)\b'
                ],
                'low': [
                    r'\b(?:unreliable|questionable|unverified|dubious)\b',
                    r'\b(?:low[- ]quality|poor|inferior)\b'
                ]
            }
        }
    
    def _init_intent_patterns(self):
        """Initialize intent classification patterns."""
        self.intent_patterns = {
            QueryIntent.SEARCH: [
                r'\b(?:find|search|look for|locate|discover)\b',
                r'\b(?:show me|give me|provide|list)\b',
                r'\b(?:what|where|which|who)\b'
            ],
            QueryIntent.FILTER: [
                r'\b(?:filter|only|exclude|include|limit to)\b',
                r'\b(?:high|low|medium)[- ](?:utility|risk|complexity|quality)\b',
                r'\b(?:safe|dangerous|simple|complex|useful)\b'
            ],
            QueryIntent.COMPARE: [
                r'\b(?:compare|versus|vs|difference|contrast)\b',
                r'\b(?:better|worse|best|worst|superior|inferior)\b'
            ],
            QueryIntent.ANALYZE: [
                r'\b(?:analyze|analysis|examine|evaluate|assess)\b',
                r'\b(?:why|how|explain|understand|reason)\b'
            ],
            QueryIntent.SUMMARIZE: [
                r'\b(?:summarize|summary|overview|brief|outline)\b',
                r'\b(?:key points|main ideas|highlights)\b'
            ]
        }
    
    def _init_profile_patterns(self):
        """Initialize profile hint patterns."""
        self.profile_patterns = {
            'researcher': [
                r'\b(?:research|study|analysis|investigation|experiment)\b',
                r'\b(?:methodology|algorithm|technical|scientific)\b',
                r'\b(?:peer[- ]reviewed|publication|journal|academic)\b',
                r'\b(?:hypothesis|theory|empirical|data)\b'
            ],
            'business': [
                r'\b(?:market|business|commercial|revenue|profit)\b',
                r'\b(?:ROI|investment|financial|cost|budget)\b',
                r'\b(?:strategy|competitive|opportunity|growth)\b',
                r'\b(?:customer|client|stakeholder|shareholder)\b'
            ],
            'legal': [
                r'\b(?:legal|law|regulation|compliance|contract)\b',
                r'\b(?:liability|risk|violation|breach|penalty)\b',
                r'\b(?:court|judge|ruling|precedent|case)\b',
                r'\b(?:ITAR|export|controlled|classified)\b'
            ]
        }
    
    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query to extract intent, filters, and profile hints.
        
        Args:
            query: Natural language query string
            
        Returns:
            ParsedQuery object with extracted information
        """
        try:
            # Extract dimension filters
            dimension_filters, filter_phrases = self._extract_dimension_filters(query)
            
            # Classify intent
            intent = self._classify_intent(query)
            
            # Detect profile hint
            profile_hint = self._detect_profile_hint(query)
            
            # Clean query by removing filter phrases
            clean_query = self._clean_query(query, filter_phrases)
            
            # Calculate confidence
            confidence = self._calculate_confidence(query, dimension_filters, intent, profile_hint)
            
            return ParsedQuery(
                original_query=query,
                clean_query=clean_query,
                intent=intent,
                dimension_filters=dimension_filters,
                profile_hint=profile_hint,
                confidence=confidence,
                filter_phrases=filter_phrases
            )
            
        except Exception as e:
            logger.error(f"Error parsing query '{query}': {e}")
            # Return basic parsed query
            return ParsedQuery(
                original_query=query,
                clean_query=query,
                intent=QueryIntent.SEARCH,
                dimension_filters={},
                profile_hint=None,
                confidence=0.5,
                filter_phrases=[]
            )
    
    def _extract_dimension_filters(self, query: str) -> Tuple[Dict[str, DimensionFilter], List[str]]:
        """Extract dimension filters from query text."""
        filters = {}
        filter_phrases = []
        
        query_lower = query.lower()
        
        for dimension, patterns in self.dimension_patterns.items():
            for filter_level, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.finditer(pattern, query_lower)
                    for match in matches:
                        filters[dimension] = DimensionFilter(filter_level)
                        filter_phrases.append(match.group())
                        logger.debug(f"Found {dimension}={filter_level} filter: '{match.group()}'")
                        break  # Take first match for this dimension
                if dimension in filters:
                    break  # Move to next dimension
        
        return filters, filter_phrases
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of the query."""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            intent_scores[intent] = score
        
        # Return intent with highest score, default to SEARCH
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                return best_intent[0]
        
        return QueryIntent.SEARCH
    
    def _detect_profile_hint(self, query: str) -> Optional[str]:
        """Detect profile hints in the query."""
        query_lower = query.lower()
        profile_scores = {}
        
        for profile, patterns in self.profile_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            profile_scores[profile] = score
        
        # Return profile with highest score if above threshold
        if profile_scores:
            best_profile = max(profile_scores.items(), key=lambda x: x[1])
            if best_profile[1] >= 2:  # Require at least 2 matches
                return best_profile[0]
        
        return None
    
    def _clean_query(self, query: str, filter_phrases: List[str]) -> str:
        """Remove filter phrases from query to get clean search terms."""
        clean_query = query

        # Remove filter phrases
        for phrase in filter_phrases:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(phrase) + r'\b'
            clean_query = re.sub(pattern, '', clean_query, flags=re.IGNORECASE)

        # Remove common filter words that might not be in phrases
        filter_words = ['high-utility', 'low-risk', 'high-risk', 'low-utility', 'simple', 'complex']
        for word in filter_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            clean_query = re.sub(pattern, '', clean_query, flags=re.IGNORECASE)

        # Clean up punctuation and extra whitespace
        clean_query = re.sub(r'[,;]+', ' ', clean_query)  # Replace commas/semicolons with spaces
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()  # Normalize whitespace

        # If query becomes too short, return original
        if len(clean_query) < 3:
            return query

        return clean_query
    
    def _calculate_confidence(self, query: str, filters: Dict[str, DimensionFilter], 
                            intent: QueryIntent, profile_hint: Optional[str]) -> float:
        """Calculate confidence in the parsing results."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for each filter found
        confidence += len(filters) * 0.1
        
        # Boost confidence for clear intent
        if intent != QueryIntent.SEARCH:
            confidence += 0.1
        
        # Boost confidence for profile hint
        if profile_hint:
            confidence += 0.15
        
        # Boost confidence for longer queries (more context)
        word_count = len(query.split())
        if word_count > 5:
            confidence += 0.1
        
        # Cap at 1.0
        return min(1.0, confidence)


# Convenience function for easy access
def parse_natural_language_query(query: str) -> ParsedQuery:
    """Parse a natural language query using the default parser."""
    parser = NaturalLanguageQueryParser()
    return parser.parse_query(query)
