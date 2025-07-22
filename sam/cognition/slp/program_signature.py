"""
Program Signature Generation
============================

Multi-dimensional signature generation for task fingerprinting.
Creates unique, comparable "signatures" for queries and their contexts
to enable pattern matching and program retrieval.
"""

import re
import hashlib
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProgramSignature:
    """
    Multi-dimensional signature for task identification.
    """
    primary_intent: str
    secondary_intents: List[str]
    complexity_level: str
    document_types: List[str]
    content_domains: List[str]
    conceptual_dimensions: Dict[str, float]
    user_profile: Optional[str]
    session_context: Dict[str, Any]
    time_sensitivity: str
    scope_breadth: str
    signature_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'primary_intent': self.primary_intent,
            'secondary_intents': self.secondary_intents,
            'complexity_level': self.complexity_level,
            'document_types': self.document_types,
            'content_domains': self.content_domains,
            'conceptual_dimensions': self.conceptual_dimensions,
            'user_profile': self.user_profile,
            'session_context': self.session_context,
            'time_sensitivity': self.time_sensitivity,
            'scope_breadth': self.scope_breadth,
            'signature_hash': self.signature_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgramSignature':
        """Create from dictionary."""
        return cls(
            primary_intent=data.get('primary_intent', ''),
            secondary_intents=data.get('secondary_intents', []),
            complexity_level=data.get('complexity_level', 'medium'),
            document_types=data.get('document_types', []),
            content_domains=data.get('content_domains', []),
            conceptual_dimensions=data.get('conceptual_dimensions', {}),
            user_profile=data.get('user_profile'),
            session_context=data.get('session_context', {}),
            time_sensitivity=data.get('time_sensitivity', 'normal'),
            scope_breadth=data.get('scope_breadth', 'medium'),
            signature_hash=data.get('signature_hash', '')
        )
    
    def calculate_similarity(self, other: 'ProgramSignature') -> float:
        """Calculate similarity score with another signature (0.0 to 1.0)."""
        if not isinstance(other, ProgramSignature):
            return 0.0
        
        scores = []
        
        # Intent similarity (40% weight)
        intent_score = 1.0 if self.primary_intent == other.primary_intent else 0.0
        secondary_overlap = len(set(self.secondary_intents) & set(other.secondary_intents))
        secondary_total = len(set(self.secondary_intents) | set(other.secondary_intents))
        secondary_score = secondary_overlap / max(secondary_total, 1)
        intent_similarity = 0.7 * intent_score + 0.3 * secondary_score
        scores.append(('intent', intent_similarity, 0.4))
        
        # Document type similarity (20% weight)
        doc_overlap = len(set(self.document_types) & set(other.document_types))
        doc_total = len(set(self.document_types) | set(other.document_types))
        doc_similarity = doc_overlap / max(doc_total, 1)
        scores.append(('document_types', doc_similarity, 0.2))
        
        # Content domain similarity (20% weight)
        domain_overlap = len(set(self.content_domains) & set(other.content_domains))
        domain_total = len(set(self.content_domains) | set(other.content_domains))
        domain_similarity = domain_overlap / max(domain_total, 1)
        scores.append(('content_domains', domain_similarity, 0.2))
        
        # Complexity and scope similarity (10% weight)
        complexity_score = 1.0 if self.complexity_level == other.complexity_level else 0.5
        scope_score = 1.0 if self.scope_breadth == other.scope_breadth else 0.5
        context_similarity = (complexity_score + scope_score) / 2
        scores.append(('context', context_similarity, 0.1))
        
        # User profile similarity (10% weight)
        profile_score = 1.0 if self.user_profile == other.user_profile else 0.0
        scores.append(('user_profile', profile_score, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        
        return min(1.0, max(0.0, total_score))


def generate_signature(query: str, context: Dict[str, Any], 
                      user_profile: Optional[str] = None) -> ProgramSignature:
    """
    Generate a multi-dimensional signature for a query and context.
    
    Args:
        query: The user's query text
        context: Context information including documents, session data, etc.
        user_profile: Optional user profile identifier
    
    Returns:
        ProgramSignature object with all dimensions analyzed
    """
    try:
        # Extract intent analysis
        primary_intent = extract_intent_verb(query)
        secondary_intents = extract_secondary_intents(query)
        complexity_level = assess_query_complexity(query)
        
        # Extract context fingerprint
        document_types = get_document_types(context)
        content_domains = identify_content_domains(context)
        conceptual_dimensions = get_dimension_profile(context)
        
        # Extract temporal and scope information
        time_sensitivity = detect_time_requirements(query)
        scope_breadth = measure_scope_breadth(query, context)
        
        # Get session context
        session_context = get_session_patterns(context)
        
        # Create signature object
        signature = ProgramSignature(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            complexity_level=complexity_level,
            document_types=document_types,
            content_domains=content_domains,
            conceptual_dimensions=conceptual_dimensions,
            user_profile=user_profile,
            session_context=session_context,
            time_sensitivity=time_sensitivity,
            scope_breadth=scope_breadth,
            signature_hash=""  # Will be calculated below
        )
        
        # Calculate hash for quick comparison
        signature.signature_hash = _calculate_signature_hash(signature)
        
        return signature
        
    except Exception as e:
        logger.error(f"Error generating signature: {e}")
        # Return a basic signature as fallback
        return ProgramSignature(
            primary_intent="unknown",
            secondary_intents=[],
            complexity_level="medium",
            document_types=[],
            content_domains=[],
            conceptual_dimensions={},
            user_profile=user_profile,
            session_context={},
            time_sensitivity="normal",
            scope_breadth="medium",
            signature_hash=""
        )


def extract_intent_verb(query: str) -> str:
    """Extract the primary intent verb from the query."""
    query_lower = query.lower().strip()
    
    # Define intent patterns
    intent_patterns = {
        'summarize': ['summarize', 'summary', 'sum up', 'brief', 'overview'],
        'analyze': ['analyze', 'analysis', 'examine', 'study', 'investigate'],
        'compare': ['compare', 'contrast', 'difference', 'versus', 'vs'],
        'explain': ['explain', 'clarify', 'describe', 'what is', 'how does'],
        'find': ['find', 'search', 'locate', 'look for', 'identify'],
        'create': ['create', 'generate', 'make', 'build', 'write'],
        'translate': ['translate', 'convert', 'transform'],
        'extract': ['extract', 'pull out', 'get', 'retrieve'],
        'classify': ['classify', 'categorize', 'group', 'sort'],
        'evaluate': ['evaluate', 'assess', 'judge', 'rate', 'review']
    }
    
    for intent, patterns in intent_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            return intent
    
    return 'general'


def extract_secondary_intents(query: str) -> List[str]:
    """Extract secondary intents from the query."""
    query_lower = query.lower()
    secondary_intents = []
    
    # Look for secondary intent indicators
    if any(word in query_lower for word in ['also', 'additionally', 'furthermore']):
        secondary_intents.append('additional_analysis')
    
    if any(word in query_lower for word in ['format', 'structure', 'organize']):
        secondary_intents.append('formatting')
    
    if any(word in query_lower for word in ['quick', 'fast', 'brief']):
        secondary_intents.append('speed_priority')
    
    if any(word in query_lower for word in ['detailed', 'comprehensive', 'thorough']):
        secondary_intents.append('depth_priority')
    
    return secondary_intents


def assess_query_complexity(query: str) -> str:
    """Assess the complexity level of the query."""
    word_count = len(query.split())
    question_marks = query.count('?')
    complex_words = len([w for w in query.split() if len(w) > 8])
    
    complexity_score = word_count * 0.1 + question_marks * 2 + complex_words * 0.5
    
    if complexity_score < 5:
        return 'simple'
    elif complexity_score < 15:
        return 'medium'
    else:
        return 'complex'


def get_document_types(context: Dict[str, Any]) -> List[str]:
    """Extract document types from context."""
    doc_types = set()
    
    # Check for document information in context
    if 'documents' in context:
        for doc in context['documents']:
            if isinstance(doc, dict) and 'type' in doc:
                doc_types.add(doc['type'])
            elif isinstance(doc, str):
                # Infer type from file extension
                ext = Path(doc).suffix.lower().lstrip('.')
                if ext:
                    doc_types.add(ext)
    
    # Check for file paths in context
    if 'files' in context:
        for file_path in context['files']:
            ext = Path(file_path).suffix.lower().lstrip('.')
            if ext:
                doc_types.add(ext)
    
    return list(doc_types)


def identify_content_domains(context: Dict[str, Any]) -> List[str]:
    """Identify content domains from context."""
    domains = set()
    
    # Domain keywords mapping
    domain_keywords = {
        'technical': ['code', 'programming', 'software', 'algorithm', 'technical'],
        'business': ['business', 'finance', 'market', 'strategy', 'revenue'],
        'academic': ['research', 'study', 'academic', 'paper', 'journal'],
        'legal': ['legal', 'law', 'contract', 'regulation', 'compliance'],
        'medical': ['medical', 'health', 'patient', 'diagnosis', 'treatment'],
        'scientific': ['science', 'experiment', 'hypothesis', 'data', 'analysis']
    }
    
    # Check context content for domain indicators
    context_text = str(context).lower()
    for domain, keywords in domain_keywords.items():
        if any(keyword in context_text for keyword in keywords):
            domains.add(domain)
    
    return list(domains) if domains else ['general']


def get_dimension_profile(context: Dict[str, Any]) -> Dict[str, float]:
    """Get conceptual dimension profile from context."""
    # Try to integrate with existing dimension analysis if available
    try:
        # Check if we have access to SAM's dimension analysis
        if 'dimension_analysis' in context:
            return context['dimension_analysis']

        # Basic dimension analysis based on content
        dimensions = {
            'technical_depth': 0.5,
            'complexity': 0.5,
            'urgency': 0.5,
            'scope': 0.5
        }

        # Analyze context text for dimension indicators
        context_text = str(context).lower()

        # Technical depth indicators
        tech_indicators = ['code', 'algorithm', 'technical', 'implementation', 'system']
        tech_score = sum(1 for indicator in tech_indicators if indicator in context_text)
        dimensions['technical_depth'] = min(1.0, tech_score / 3.0)

        # Complexity indicators
        complex_indicators = ['complex', 'detailed', 'comprehensive', 'analysis', 'multiple']
        complex_score = sum(1 for indicator in complex_indicators if indicator in context_text)
        dimensions['complexity'] = min(1.0, complex_score / 3.0)

        # Urgency indicators
        urgent_indicators = ['urgent', 'immediate', 'asap', 'critical', 'priority']
        urgent_score = sum(1 for indicator in urgent_indicators if indicator in context_text)
        dimensions['urgency'] = min(1.0, urgent_score / 2.0)

        # Scope indicators
        scope_indicators = ['all', 'entire', 'complete', 'comprehensive', 'full']
        scope_score = sum(1 for indicator in scope_indicators if indicator in context_text)
        dimensions['scope'] = min(1.0, scope_score / 3.0)

        return dimensions

    except Exception as e:
        logger.warning(f"Error in dimension analysis: {e}")
        return {
            'technical_depth': 0.5,
            'complexity': 0.5,
            'urgency': 0.5,
            'scope': 0.5
        }


def detect_time_requirements(query: str) -> str:
    """Detect time sensitivity from query."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['urgent', 'asap', 'immediately', 'now']):
        return 'urgent'
    elif any(word in query_lower for word in ['quick', 'fast', 'briefly']):
        return 'high'
    elif any(word in query_lower for word in ['when you can', 'no rush', 'eventually']):
        return 'low'
    else:
        return 'normal'


def measure_scope_breadth(query: str, context: Dict[str, Any]) -> str:
    """Measure the scope breadth of the query."""
    word_count = len(query.split())
    doc_count = len(context.get('documents', []))
    
    scope_score = word_count * 0.1 + doc_count * 2
    
    if scope_score < 5:
        return 'narrow'
    elif scope_score < 20:
        return 'medium'
    else:
        return 'broad'


def get_session_patterns(context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract session patterns from context."""
    return {
        'session_length': context.get('session_length', 0),
        'previous_queries': context.get('previous_queries', 0),
        'user_preferences': context.get('user_preferences', {})
    }


def _calculate_signature_hash(signature: ProgramSignature) -> str:
    """Calculate a hash for the signature for quick comparison."""
    # Create a string representation of key signature components
    hash_components = [
        signature.primary_intent,
        '|'.join(sorted(signature.secondary_intents)),
        signature.complexity_level,
        '|'.join(sorted(signature.document_types)),
        '|'.join(sorted(signature.content_domains)),
        signature.time_sensitivity,
        signature.scope_breadth,
        str(signature.user_profile or '')
    ]
    
    hash_string = '||'.join(hash_components)
    return hashlib.md5(hash_string.encode()).hexdigest()[:16]
