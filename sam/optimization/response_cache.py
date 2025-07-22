#!/usr/bin/env python3
"""
SAM Response Caching and Optimization System - Task 30 Phase 3
==============================================================

Implements intelligent response caching and performance optimization
for the conversational coherence engine.

Part of Task 30: Advanced Conversational Coherence Engine
Author: SAM Development Team
Version: 1.0.0
"""

import logging
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cached response entry."""
    cache_key: str
    user_question: str
    response: str
    pipeline_used: str
    generation_time_ms: float
    created_at: str
    last_accessed: str
    access_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        return cls(**data)

class ResponseCache:
    """
    Intelligent response caching system with LRU eviction.
    
    Features:
    - Content-based cache keys
    - LRU eviction policy
    - Performance metrics tracking
    - Configurable TTL and size limits
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the response cache.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.ResponseCache")
        
        # Default configuration
        self.config = {
            'enable_caching': True,
            'max_cache_size': 1000,
            'cache_ttl_hours': 24,
            'min_generation_time_ms': 100,  # Only cache responses that took time to generate
            'cache_hit_threshold': 0.85,  # Similarity threshold for cache hits
            'storage_directory': 'response_cache',
            'persist_cache': True
        }
        
        if config:
            self.config.update(config)
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'total_time_saved_ms': 0.0,
            'evictions': 0
        }
        
        # Storage setup
        if self.config['persist_cache']:
            self.storage_dir = Path(self.config['storage_directory'])
            self.storage_dir.mkdir(exist_ok=True)
            self._load_cache()
        
        self.logger.info(f"ResponseCache initialized with max size: {self.config['max_cache_size']}")
    
    def get_cached_response(self, user_question: str, conversation_context: Optional[str] = None,
                           persona_context: Optional[str] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get cached response if available.
        
        Args:
            user_question: User's question
            conversation_context: Recent conversation context
            persona_context: User persona context
            
        Returns:
            Tuple of (cached_response, metadata) or None if not cached
        """
        try:
            if not self.config['enable_caching']:
                return None
            
            with self._lock:
                self.metrics['total_requests'] += 1
                
                # Generate cache key
                cache_key = self._generate_cache_key(user_question, conversation_context, persona_context)
                
                # Check for exact match
                if cache_key in self.cache:
                    entry = self.cache[cache_key]
                    
                    # Check TTL
                    if self._is_entry_valid(entry):
                        # Move to end (LRU)
                        self.cache.move_to_end(cache_key)
                        
                        # Update access metrics
                        entry.last_accessed = datetime.now().isoformat()
                        entry.access_count += 1
                        
                        self.metrics['cache_hits'] += 1
                        self.metrics['total_time_saved_ms'] += entry.generation_time_ms
                        
                        self.logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                        
                        return entry.response, {
                            'cache_hit': True,
                            'cache_key': cache_key,
                            'original_generation_time_ms': entry.generation_time_ms,
                            'access_count': entry.access_count,
                            'pipeline_used': entry.pipeline_used
                        }
                    else:
                        # Entry expired, remove it
                        del self.cache[cache_key]
                        self.logger.debug(f"Removed expired cache entry: {cache_key[:16]}...")
                
                self.metrics['cache_misses'] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting cached response: {e}")
            return None
    
    def cache_response(self, user_question: str, response: str, pipeline_used: str,
                      generation_time_ms: float, conversation_context: Optional[str] = None,
                      persona_context: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Cache a response.
        
        Args:
            user_question: User's question
            response: Generated response
            pipeline_used: Pipeline that generated the response
            generation_time_ms: Time taken to generate response
            conversation_context: Recent conversation context
            persona_context: User persona context
            metadata: Additional metadata
            
        Returns:
            True if response was cached
        """
        try:
            if not self.config['enable_caching']:
                return False
            
            # Only cache responses that took significant time to generate
            if generation_time_ms < self.config['min_generation_time_ms']:
                return False
            
            with self._lock:
                # Generate cache key
                cache_key = self._generate_cache_key(user_question, conversation_context, persona_context)
                
                # Create cache entry
                entry = CacheEntry(
                    cache_key=cache_key,
                    user_question=user_question,
                    response=response,
                    pipeline_used=pipeline_used,
                    generation_time_ms=generation_time_ms,
                    created_at=datetime.now().isoformat(),
                    last_accessed=datetime.now().isoformat(),
                    access_count=0,
                    metadata=metadata or {}
                )
                
                # Add to cache
                self.cache[cache_key] = entry
                
                # Enforce size limit (LRU eviction)
                while len(self.cache) > self.config['max_cache_size']:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    self.metrics['evictions'] += 1
                    self.logger.debug(f"Evicted cache entry: {oldest_key[:16]}...")
                
                # Persist if enabled
                if self.config['persist_cache']:
                    self._save_cache_entry(entry)
                
                self.logger.debug(f"Cached response with key: {cache_key[:16]}...")
                return True
                
        except Exception as e:
            self.logger.error(f"Error caching response: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self.metrics['total_requests']
            cache_hits = self.metrics['cache_hits']
            
            hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
            avg_time_saved = (self.metrics['total_time_saved_ms'] / cache_hits) if cache_hits > 0 else 0
            
            return {
                'cache_size': len(self.cache),
                'max_cache_size': self.config['max_cache_size'],
                'total_requests': total_requests,
                'cache_hits': cache_hits,
                'cache_misses': self.metrics['cache_misses'],
                'hit_rate_percent': hit_rate,
                'total_time_saved_ms': self.metrics['total_time_saved_ms'],
                'average_time_saved_ms': avg_time_saved,
                'evictions': self.metrics['evictions'],
                'cache_enabled': self.config['enable_caching']
            }
    
    def clear_cache(self) -> int:
        """Clear all cached responses."""
        with self._lock:
            cache_size = len(self.cache)
            self.cache.clear()
            
            # Clear persisted cache
            if self.config['persist_cache']:
                try:
                    for cache_file in self.storage_dir.glob("*.json"):
                        cache_file.unlink()
                except Exception as e:
                    self.logger.error(f"Error clearing persisted cache: {e}")
            
            self.logger.info(f"Cleared {cache_size} cached responses")
            return cache_size
    
    def _generate_cache_key(self, user_question: str, conversation_context: Optional[str],
                           persona_context: Optional[str]) -> str:
        """Generate a cache key based on input parameters."""
        # Normalize inputs
        question_normalized = user_question.lower().strip()
        context_normalized = (conversation_context or "").lower().strip()
        persona_normalized = (persona_context or "").lower().strip()
        
        # Create combined string
        combined = f"{question_normalized}|{context_normalized}|{persona_normalized}"
        
        # Generate hash
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _is_entry_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid (not expired)."""
        try:
            created_at = datetime.fromisoformat(entry.created_at)
            ttl = timedelta(hours=self.config['cache_ttl_hours'])
            
            return datetime.now() - created_at < ttl
        except:
            return False
    
    def _save_cache_entry(self, entry: CacheEntry) -> None:
        """Save cache entry to disk."""
        try:
            cache_file = self.storage_dir / f"{entry.cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache entry: {e}")
    
    def _load_cache(self) -> None:
        """Load cache entries from disk."""
        try:
            loaded_count = 0
            
            for cache_file in self.storage_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        entry_data = json.load(f)
                    
                    entry = CacheEntry.from_dict(entry_data)
                    
                    # Only load valid entries
                    if self._is_entry_valid(entry):
                        self.cache[entry.cache_key] = entry
                        loaded_count += 1
                    else:
                        # Remove expired cache file
                        cache_file.unlink()
                
                except Exception as e:
                    self.logger.warning(f"Error loading cache entry {cache_file}: {e}")
            
            self.logger.info(f"Loaded {loaded_count} valid cache entries from disk")
            
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")

class PerformanceOptimizer:
    """
    Performance optimization coordinator for the conversational coherence engine.
    
    Features:
    - Response caching management
    - Pipeline performance monitoring
    - Adaptive optimization strategies
    - Resource usage tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance optimizer."""
        self.logger = logging.getLogger(f"{__name__}.PerformanceOptimizer")
        
        self.config = {
            'enable_optimization': True,
            'performance_monitoring': True,
            'adaptive_caching': True,
            'optimization_interval_minutes': 60
        }
        
        if config:
            self.config.update(config)
        
        # Initialize response cache
        self.response_cache = ResponseCache(config)
        
        # Performance tracking
        self.performance_metrics = {
            'single_stage_avg_time_ms': 0.0,
            'two_stage_avg_time_ms': 0.0,
            'cache_effectiveness': 0.0,
            'total_responses_generated': 0,
            'optimization_recommendations': []
        }
        
        self.logger.info("PerformanceOptimizer initialized")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get current optimization recommendations."""
        recommendations = []
        
        cache_stats = self.response_cache.get_cache_stats()
        
        # Cache hit rate recommendations
        if cache_stats['hit_rate_percent'] < 20:
            recommendations.append("Consider increasing cache TTL or size - low hit rate detected")
        
        # Performance recommendations
        if self.performance_metrics['two_stage_avg_time_ms'] > self.performance_metrics['single_stage_avg_time_ms'] * 2:
            recommendations.append("Two-stage pipeline significantly slower - consider optimization")
        
        # Resource usage recommendations
        if cache_stats['cache_size'] >= cache_stats['max_cache_size'] * 0.9:
            recommendations.append("Cache near capacity - consider increasing max size")
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cache_stats = self.response_cache.get_cache_stats()
        
        return {
            'cache_performance': cache_stats,
            'pipeline_performance': self.performance_metrics,
            'optimization_recommendations': self.get_optimization_recommendations(),
            'optimizer_enabled': self.config['enable_optimization']
        }

# Global instances
_response_cache: Optional[ResponseCache] = None
_performance_optimizer: Optional[PerformanceOptimizer] = None

def get_response_cache(config: Optional[Dict[str, Any]] = None) -> ResponseCache:
    """Get the global response cache instance."""
    global _response_cache
    
    if _response_cache is None:
        _response_cache = ResponseCache(config)
    return _response_cache

def get_performance_optimizer(config: Optional[Dict[str, Any]] = None) -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(config)
    return _performance_optimizer
