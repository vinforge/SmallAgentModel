#!/usr/bin/env python3
"""
Intelligent Cache Service
Advanced caching system with predictive prefetching, adaptive TTL, and performance optimization.
"""

import logging
import hashlib
import pickle
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Enhanced cache entry with intelligence metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: timedelta = field(default_factory=lambda: timedelta(hours=1))
    priority_score: float = 1.0
    query_pattern: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    
    @property
    def age(self) -> timedelta:
        """Get age of cache entry."""
        return datetime.now() - self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return self.age > self.ttl
    
    @property
    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per hour)."""
        age_hours = max(self.age.total_seconds() / 3600, 0.1)
        return self.access_count / age_hours

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0
    total_size_bytes: int = 0
    avg_access_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def prefetch_effectiveness(self) -> float:
        """Calculate prefetch effectiveness."""
        return (self.prefetch_hits / self.hits * 100) if self.hits > 0 else 0.0

class QueryPatternAnalyzer:
    """Analyzes query patterns for predictive caching."""
    
    def __init__(self):
        self.query_sequences = deque(maxlen=1000)
        self.pattern_frequencies = defaultdict(int)
        self.user_patterns = defaultdict(lambda: deque(maxlen=100))
    
    def record_query(self, query: str, user_id: Optional[str] = None):
        """Record a query for pattern analysis."""
        normalized_query = self._normalize_query(query)
        timestamp = datetime.now()
        
        # Record global sequence
        self.query_sequences.append((normalized_query, timestamp))
        
        # Record user-specific sequence
        if user_id:
            self.user_patterns[user_id].append((normalized_query, timestamp))
        
        # Update pattern frequencies
        self._update_patterns()
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching."""
        # Convert to lowercase and extract key terms
        normalized = query.lower().strip()
        
        # Extract query type patterns
        if any(term in normalized for term in ['document', 'file', 'upload', 'sam story']):
            return 'document_query'
        elif any(term in normalized for term in ['conversation', 'discuss', 'talk']):
            return 'conversation_query'
        elif any(term in normalized for term in ['correct', 'wrong', 'fix']):
            return 'correction_query'
        elif any(term in normalized for term in ['explain', 'what is', 'how']):
            return 'knowledge_query'
        else:
            return 'general_query'
    
    def _update_patterns(self):
        """Update pattern frequency analysis."""
        # Analyze recent sequences for patterns
        recent_queries = list(self.query_sequences)[-10:]
        
        for i in range(len(recent_queries) - 1):
            current_pattern = recent_queries[i][0]
            next_pattern = recent_queries[i + 1][0]
            pattern_key = f"{current_pattern}->{next_pattern}"
            self.pattern_frequencies[pattern_key] += 1
    
    def predict_next_queries(self, current_query: str, user_id: Optional[str] = None) -> List[str]:
        """Predict likely next queries based on patterns."""
        current_pattern = self._normalize_query(current_query)
        predictions = []
        
        # Find patterns that follow the current pattern
        for pattern_key, frequency in self.pattern_frequencies.items():
            if pattern_key.startswith(f"{current_pattern}->"):
                next_pattern = pattern_key.split("->")[1]
                predictions.append((next_pattern, frequency))
        
        # Sort by frequency and return top predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [pattern for pattern, _ in predictions[:3]]

class IntelligentCacheService:
    """
    Advanced caching service with intelligent features:
    - Adaptive TTL based on access patterns
    - Predictive prefetching
    - Priority-based eviction
    - Performance optimization
    - Usage analytics
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0):
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        
        # Core cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # Intelligence components
        self.pattern_analyzer = QueryPatternAnalyzer()
        self.stats = CacheStats()
        
        # Prefetch queue
        self.prefetch_queue = deque(maxlen=100)
        self.prefetch_thread = None
        self.prefetch_enabled = True
        
        # Performance tracking
        self.access_times = deque(maxlen=1000)
        
        logger.info(f"IntelligentCacheService initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def get(self, key: str, user_id: Optional[str] = None) -> Optional[Any]:
        """Get value from cache with intelligence tracking."""
        start_time = time.time()
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None or entry.is_expired:
                self.stats.misses += 1
                if entry and entry.is_expired:
                    self._remove_entry(key)
                return None
            
            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Update priority score based on access pattern
            entry.priority_score = self._calculate_priority_score(entry)
            
            self.stats.hits += 1
            
            # Record access time
            access_time = (time.time() - start_time) * 1000
            self.access_times.append(access_time)
            self._update_avg_access_time()
            
            return entry.value
    
    def put(self, key: str, value: Any, query: Optional[str] = None, 
            user_id: Optional[str] = None, custom_ttl: Optional[timedelta] = None) -> None:
        """Put value in cache with intelligent metadata."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Default estimate
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Determine adaptive TTL
            ttl = custom_ttl or self._calculate_adaptive_ttl(query, user_id)
            
            # Create entry with intelligence metadata
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl=ttl,
                query_pattern=self.pattern_analyzer._normalize_query(query) if query else None,
                user_context={'user_id': user_id} if user_id else None
            )
            
            # Add to cache
            self._cache[key] = entry
            self.stats.total_size_bytes += size_bytes
            
            # Record query pattern
            if query:
                self.pattern_analyzer.record_query(query, user_id)
                
                # Trigger predictive prefetching
                if self.prefetch_enabled:
                    self._schedule_prefetch(query, user_id)
            
            # Evict if necessary
            self._evict_if_needed()
    
    def _calculate_adaptive_ttl(self, query: Optional[str], user_id: Optional[str]) -> timedelta:
        """Calculate adaptive TTL based on query type and user patterns."""
        base_ttl = timedelta(hours=1)
        
        if not query:
            return base_ttl
        
        query_pattern = self.pattern_analyzer._normalize_query(query)
        
        # Adjust TTL based on query type
        ttl_multipliers = {
            'document_query': 2.0,      # Document content changes less frequently
            'knowledge_query': 1.5,     # Knowledge queries can be cached longer
            'conversation_query': 0.5,  # Conversation context changes quickly
            'correction_query': 0.3,    # Corrections are time-sensitive
            'general_query': 1.0        # Default
        }
        
        multiplier = ttl_multipliers.get(query_pattern, 1.0)
        return timedelta(seconds=base_ttl.total_seconds() * multiplier)
    
    def _calculate_priority_score(self, entry: CacheEntry) -> float:
        """Calculate priority score for eviction decisions."""
        # Factors: access frequency, recency, size efficiency
        frequency_score = min(entry.access_frequency, 10.0) / 10.0
        recency_score = max(0, 1.0 - (entry.age.total_seconds() / 3600))  # Decay over 1 hour
        size_efficiency = 1.0 / (1.0 + entry.size_bytes / 1024)  # Prefer smaller entries
        
        return (frequency_score * 0.5 + recency_score * 0.3 + size_efficiency * 0.2)
    
    def _schedule_prefetch(self, query: str, user_id: Optional[str]):
        """Schedule predictive prefetching based on query patterns."""
        predictions = self.pattern_analyzer.predict_next_queries(query, user_id)
        
        for predicted_pattern in predictions:
            # Add to prefetch queue if not already present
            prefetch_item = (predicted_pattern, user_id, datetime.now())
            if prefetch_item not in self.prefetch_queue:
                self.prefetch_queue.append(prefetch_item)
    
    def _evict_if_needed(self):
        """Evict entries if cache limits are exceeded."""
        # Check size limit
        while len(self._cache) > self.max_size:
            self._evict_lowest_priority()
        
        # Check memory limit
        while self.stats.total_size_bytes > self.max_memory_bytes:
            self._evict_lowest_priority()
    
    def _evict_lowest_priority(self):
        """Evict the entry with the lowest priority score."""
        if not self._cache:
            return
        
        # Find entry with lowest priority
        lowest_entry = min(self._cache.values(), key=lambda e: e.priority_score)
        self._remove_entry(lowest_entry.key)
        self.stats.evictions += 1
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and update stats."""
        if key in self._cache:
            entry = self._cache[key]
            self.stats.total_size_bytes -= entry.size_bytes
            del self._cache[key]
    
    def _update_avg_access_time(self):
        """Update average access time statistic."""
        if self.access_times:
            self.stats.avg_access_time_ms = sum(self.access_times) / len(self.access_times)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return {
                'performance': {
                    'hit_rate': self.stats.hit_rate,
                    'hits': self.stats.hits,
                    'misses': self.stats.misses,
                    'avg_access_time_ms': self.stats.avg_access_time_ms,
                    'prefetch_effectiveness': self.stats.prefetch_effectiveness
                },
                'capacity': {
                    'current_entries': len(self._cache),
                    'max_entries': self.max_size,
                    'current_memory_mb': self.stats.total_size_bytes / (1024 * 1024),
                    'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                    'evictions': self.stats.evictions
                },
                'intelligence': {
                    'pattern_count': len(self.pattern_analyzer.pattern_frequencies),
                    'prefetch_queue_size': len(self.prefetch_queue),
                    'top_patterns': dict(list(self.pattern_analyzer.pattern_frequencies.items())[:5])
                }
            }
    
    def clear_cache(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats = CacheStats()
            logger.info("Cache cleared")

# Global instance for easy access
_intelligent_cache_service = None

def get_intelligent_cache_service() -> IntelligentCacheService:
    """Get or create the global intelligent cache service instance."""
    global _intelligent_cache_service
    if _intelligent_cache_service is None:
        _intelligent_cache_service = IntelligentCacheService()
    return _intelligent_cache_service
