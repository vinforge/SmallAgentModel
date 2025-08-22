#!/usr/bin/env python3
"""
SAM Performance Caching Framework
=================================

Intelligent caching system for SAM applications to improve performance
and reduce redundant computations across all modules.

This module provides:
- Multi-level caching strategies
- Memory-aware cache management
- Automatic cache invalidation
- Performance metrics and monitoring
- Thread-safe cache operations

Author: SAM Development Team
Version: 1.0.0 - Performance Optimization Framework
"""

import streamlit as st
import logging
import time
import hashlib
import pickle
import threading
from typing import Dict, Any, Optional, Callable, Union, List
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import weakref

logger = logging.getLogger(__name__)


class CacheLevel:
    """Cache level constants."""
    MEMORY = "memory"
    SESSION = "session"
    DISK = "disk"
    DISTRIBUTED = "distributed"


class CacheEntry:
    """Represents a single cache entry with metadata."""
    
    def __init__(self, value: Any, ttl: Optional[int] = None, 
                 access_count: int = 0, size_bytes: int = 0):
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.ttl = ttl  # Time to live in seconds
        self.access_count = access_count
        self.size_bytes = size_bytes or self._estimate_size(value)
        self.is_expired = False
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            return 1024  # Default estimate
    
    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        if self.is_expired:
            return False
        
        if self.ttl is None:
            return True
        
        return (time.time() - self.created_at) < self.ttl
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at


class MemoryCache:
    """In-memory cache with LRU eviction and size limits."""
    
    def __init__(self, max_size: int = 100, max_memory_mb: int = 50):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_memory = 0
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            if not entry.is_valid():
                self._remove_key(key)
                self.stats['misses'] += 1
                return None
            
            entry.touch()
            self._update_access_order(key)
            self.stats['hits'] += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        with self.lock:
            entry = CacheEntry(value, ttl)
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_key(key)
            
            # Check if we need to evict entries
            self._ensure_capacity(entry.size_bytes)
            
            # Add new entry
            self.cache[key] = entry
            self.access_order.append(key)
            self.current_memory += entry.size_bytes
    
    def _remove_key(self, key: str):
        """Remove key from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory -= entry.size_bytes
            del self.cache[key]
            
            if key in self.access_order:
                self.access_order.remove(key)
    
    def _update_access_order(self, key: str):
        """Update access order for LRU."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry."""
        # Evict by size first
        while (len(self.cache) >= self.max_size or 
               self.current_memory + new_entry_size > self.max_memory_bytes):
            
            if not self.access_order:
                break
            
            # Remove least recently used
            lru_key = self.access_order[0]
            self._remove_key(lru_key)
            self.stats['evictions'] += 1
            
            if self.current_memory + new_entry_size > self.max_memory_bytes:
                self.stats['size_evictions'] += 1
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'size': len(self.cache),
                'memory_usage_mb': self.current_memory / (1024 * 1024),
                'memory_usage_percent': (self.current_memory / self.max_memory_bytes) * 100
            }


class SessionCache:
    """Session-based cache using Streamlit session state."""
    
    def __init__(self, namespace: str = "sam_cache"):
        self.namespace = namespace
        self.cache_key = f"{namespace}_cache"
        self.stats_key = f"{namespace}_cache_stats"
        
        # Initialize cache in session state
        if self.cache_key not in st.session_state:
            st.session_state[self.cache_key] = {}
        
        if self.stats_key not in st.session_state:
            st.session_state[self.stats_key] = {'hits': 0, 'misses': 0}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from session cache."""
        cache = st.session_state[self.cache_key]
        stats = st.session_state[self.stats_key]
        
        if key not in cache:
            stats['misses'] += 1
            return None
        
        entry = cache[key]
        
        if not entry.is_valid():
            del cache[key]
            stats['misses'] += 1
            return None
        
        entry.touch()
        stats['hits'] += 1
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in session cache."""
        cache = st.session_state[self.cache_key]
        cache[key] = CacheEntry(value, ttl)
    
    def clear(self):
        """Clear session cache."""
        st.session_state[self.cache_key] = {}
        st.session_state[self.stats_key] = {'hits': 0, 'misses': 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session cache statistics."""
        stats = st.session_state[self.stats_key]
        cache = st.session_state[self.cache_key]
        
        total_requests = stats['hits'] + stats['misses']
        hit_rate = stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **stats,
            'hit_rate': hit_rate,
            'size': len(cache)
        }


class MultiLevelCache:
    """Multi-level cache combining memory and session caches."""
    
    def __init__(self, namespace: str = "sam_multilevel"):
        self.memory_cache = MemoryCache(max_size=50, max_memory_mb=25)
        self.session_cache = SessionCache(f"{namespace}_session")
        self.namespace = namespace
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        # Try memory cache first (fastest)
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try session cache
        value = self.session_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            levels: List[str] = None):
        """Set value in multi-level cache."""
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.SESSION]
        
        if CacheLevel.MEMORY in levels:
            self.memory_cache.set(key, value, ttl)
        
        if CacheLevel.SESSION in levels:
            self.session_cache.set(key, value, ttl)
    
    def clear(self, levels: List[str] = None):
        """Clear multi-level cache."""
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.SESSION]
        
        if CacheLevel.MEMORY in levels:
            self.memory_cache.clear()
        
        if CacheLevel.SESSION in levels:
            self.session_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'memory': self.memory_cache.get_stats(),
            'session': self.session_cache.get_stats()
        }


# Global cache instances
_global_caches: Dict[str, MultiLevelCache] = {}
_cache_lock = threading.RLock()


def get_cache(namespace: str = "default") -> MultiLevelCache:
    """Get or create a cache instance for a namespace."""
    with _cache_lock:
        if namespace not in _global_caches:
            _global_caches[namespace] = MultiLevelCache(namespace)
        return _global_caches[namespace]


def cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(namespace: str = "default", ttl: Optional[int] = None, 
          levels: List[str] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache(namespace)
            key = f"{func.__name__}_{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl, levels)
            
            return result
        
        # Add cache management methods to function
        wrapper.cache_clear = lambda: get_cache(namespace).clear()
        wrapper.cache_stats = lambda: get_cache(namespace).get_stats()
        
        return wrapper
    
    return decorator


def render_cache_dashboard():
    """Render a cache performance dashboard."""
    st.subheader("🚀 Cache Performance Dashboard")
    
    # Get all cache statistics
    cache_stats = {}
    for namespace, cache in _global_caches.items():
        cache_stats[namespace] = cache.get_stats()
    
    if not cache_stats:
        st.info("No cache data available yet.")
        return
    
    # Overall metrics
    total_hits = sum(stats['memory']['hits'] + stats['session']['hits'] 
                    for stats in cache_stats.values())
    total_misses = sum(stats['memory']['misses'] + stats['session']['misses'] 
                      for stats in cache_stats.values())
    overall_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Hit Rate", f"{overall_hit_rate:.1%}")
    
    with col2:
        st.metric("Total Hits", total_hits)
    
    with col3:
        st.metric("Total Misses", total_misses)
    
    # Per-namespace statistics
    for namespace, stats in cache_stats.items():
        with st.expander(f"📦 {namespace.title()} Cache", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Memory Cache:**")
                mem_stats = stats['memory']
                st.write(f"Hit Rate: {mem_stats['hit_rate']:.1%}")
                st.write(f"Size: {mem_stats['size']} entries")
                st.write(f"Memory: {mem_stats['memory_usage_mb']:.1f} MB")
            
            with col2:
                st.markdown("**Session Cache:**")
                sess_stats = stats['session']
                st.write(f"Hit Rate: {sess_stats['hit_rate']:.1%}")
                st.write(f"Size: {sess_stats['size']} entries")
    
    # Cache management
    st.markdown("### 🔧 Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Caches"):
            for cache in _global_caches.values():
                cache.clear()
            st.success("All caches cleared!")
            st.rerun()
    
    with col2:
        if st.button("📊 Refresh Stats"):
            st.rerun()
