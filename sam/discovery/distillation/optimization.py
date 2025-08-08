"""
Performance Optimization Module
==============================

Optimizes cognitive distillation performance through caching, semantic search, and efficiency improvements.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

from .registry import CognitivePrinciple

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached computation result."""
    key: str
    value: Any
    timestamp: datetime
    access_count: int
    ttl_seconds: int

class SemanticCache:
    """Semantic caching for principle relevance calculations."""
    
    def __init__(self, cache_dir: str = "cache/distillation"):
        """Initialize semantic cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_db_path = self.cache_dir / "semantic_cache.db"
        self.max_cache_size = 1000
        self.default_ttl = 3600  # 1 hour
        
        self._init_cache_db()
        logger.info("Semantic cache initialized")
    
    def _init_cache_db(self):
        """Initialize cache database."""
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    cache_key TEXT PRIMARY KEY,
                    value_data BLOB,
                    timestamp TEXT,
                    access_count INTEGER DEFAULT 0,
                    ttl_seconds INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON semantic_cache(timestamp)
            """)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT value_data, timestamp, access_count, ttl_seconds
                    FROM semantic_cache WHERE cache_key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                value_data, timestamp_str, access_count, ttl_seconds = row
                timestamp = datetime.fromisoformat(timestamp_str)
                
                # Check if expired
                if datetime.now() - timestamp > timedelta(seconds=ttl_seconds):
                    self.delete(key)
                    return None
                
                # Update access count
                conn.execute("""
                    UPDATE semantic_cache 
                    SET access_count = access_count + 1 
                    WHERE cache_key = ?
                """, (key,))
                
                # Deserialize value
                value = pickle.loads(value_data)
                return value
                
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl_seconds or self.default_ttl
            value_data = pickle.dumps(value)
            timestamp = datetime.now()
            
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO semantic_cache 
                    (cache_key, value_data, timestamp, access_count, ttl_seconds)
                    VALUES (?, ?, ?, 0, ?)
                """, (key, value_data, timestamp.isoformat(), ttl))
            
            # Clean up old entries if cache is too large
            self._cleanup_cache()
            return True
            
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("DELETE FROM semantic_cache WHERE cache_key = ?", (key,))
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    def _cleanup_cache(self):
        """Clean up old cache entries."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Count current entries
                cursor = conn.execute("SELECT COUNT(*) FROM semantic_cache")
                count = cursor.fetchone()[0]
                
                if count > self.max_cache_size:
                    # Delete oldest entries
                    entries_to_delete = count - self.max_cache_size + 100  # Delete extra for buffer
                    conn.execute("""
                        DELETE FROM semantic_cache 
                        WHERE cache_key IN (
                            SELECT cache_key FROM semantic_cache 
                            ORDER BY timestamp ASC 
                            LIMIT ?
                        )
                    """, (entries_to_delete,))
                    
                    logger.info(f"Cleaned up {entries_to_delete} old cache entries")
                    
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        AVG(access_count) as avg_access_count,
                        SUM(CASE WHEN datetime(timestamp) > datetime('now', '-1 hour') THEN 1 ELSE 0 END) as recent_entries
                    FROM semantic_cache
                """)
                
                row = cursor.fetchone()
                return {
                    'total_entries': row[0] or 0,
                    'avg_access_count': round(row[1] or 0, 1),
                    'recent_entries': row[2] or 0,
                    'max_cache_size': self.max_cache_size,
                    'default_ttl': self.default_ttl
                }
                
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

class PrincipleOptimizer:
    """Optimizes principle selection and application."""
    
    def __init__(self):
        """Initialize principle optimizer."""
        self.cache = SemanticCache()
        self.similarity_cache = {}
        self.performance_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'optimization_time_saved': 0.0
        }
        
        logger.info("Principle optimizer initialized")
    
    def get_optimized_principles(self, prompt: str, domain_info: Dict[str, Any],
                               available_principles: List[CognitivePrinciple],
                               max_principles: int = 3) -> List[CognitivePrinciple]:
        """Get optimized principle selection using caching and fast algorithms."""
        start_time = datetime.now()
        
        # Create cache key
        cache_key = self._create_cache_key(prompt, domain_info, 
                                         [p.principle_id for p in available_principles])
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.performance_metrics['cache_hits'] += 1
            logger.debug("Using cached principle selection")
            
            # Filter cached principle IDs to current available principles
            cached_principle_ids = set(cached_result)
            result = [p for p in available_principles if p.principle_id in cached_principle_ids]
            return result[:max_principles]
        
        self.performance_metrics['cache_misses'] += 1
        
        # Compute optimized selection
        optimized_principles = self._compute_optimized_selection(
            prompt, domain_info, available_principles, max_principles
        )
        
        # Cache the result
        principle_ids = [p.principle_id for p in optimized_principles]
        self.cache.set(cache_key, principle_ids, ttl_seconds=1800)  # 30 minutes
        
        # Update performance metrics
        computation_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics['optimization_time_saved'] += max(0, computation_time - 0.1)
        
        return optimized_principles
    
    def _create_cache_key(self, prompt: str, domain_info: Dict[str, Any], 
                         principle_ids: List[str]) -> str:
        """Create a cache key for principle selection."""
        # Create a hash of the input parameters
        key_data = {
            'prompt_hash': hashlib.md5(prompt.encode()).hexdigest()[:16],
            'domains': sorted(domain_info.get('domains', [])),
            'query_type': domain_info.get('query_type', 'general'),
            'principle_ids_hash': hashlib.md5(''.join(sorted(principle_ids)).encode()).hexdigest()[:16]
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _compute_optimized_selection(self, prompt: str, domain_info: Dict[str, Any],
                                   available_principles: List[CognitivePrinciple],
                                   max_principles: int) -> List[CognitivePrinciple]:
        """Compute optimized principle selection using fast algorithms."""
        if not available_principles:
            return []
        
        # Fast domain filtering
        prompt_domains = set(domain_info.get('domains', []))
        domain_filtered = []
        
        for principle in available_principles:
            principle_domains = set(principle.domain_tags)
            
            # Exact domain match gets highest priority
            if prompt_domains & principle_domains:
                domain_filtered.append((principle, 2.0))
            # General principles get lower priority
            elif 'general' in principle_domains or not principle_domains:
                domain_filtered.append((principle, 1.0))
            # No domain match gets lowest priority
            else:
                domain_filtered.append((principle, 0.5))
        
        # Fast text similarity using cached word sets
        prompt_words = set(prompt.lower().split())
        scored_principles = []
        
        for principle, domain_score in domain_filtered:
            # Get or compute word similarity
            similarity_key = f"{principle.principle_id}:{hash(prompt)}"
            
            if similarity_key in self.similarity_cache:
                text_similarity = self.similarity_cache[similarity_key]
            else:
                principle_words = set(principle.principle_text.lower().split())
                common_words = prompt_words & principle_words
                text_similarity = len(common_words) / max(len(principle_words), len(prompt_words))
                
                # Cache similarity (keep cache small)
                if len(self.similarity_cache) < 1000:
                    self.similarity_cache[similarity_key] = text_similarity
            
            # Combined score
            combined_score = (
                domain_score * 0.4 +
                text_similarity * 0.3 +
                principle.confidence_score * 0.2 +
                min(1.0, principle.success_rate) * 0.1
            )
            
            scored_principles.append((principle, combined_score))
        
        # Sort by score and return top principles
        scored_principles.sort(key=lambda x: x[1], reverse=True)
        return [principle for principle, score in scored_principles[:max_principles]]
    
    def precompute_similarities(self, principles: List[CognitivePrinciple], 
                              common_prompts: List[str]):
        """Precompute similarities for common prompts to improve performance."""
        logger.info(f"Precomputing similarities for {len(principles)} principles and {len(common_prompts)} prompts")
        
        for prompt in common_prompts:
            prompt_words = set(prompt.lower().split())
            
            for principle in principles:
                similarity_key = f"{principle.principle_id}:{hash(prompt)}"
                
                if similarity_key not in self.similarity_cache:
                    principle_words = set(principle.principle_text.lower().split())
                    common_words = prompt_words & principle_words
                    text_similarity = len(common_words) / max(len(principle_words), len(prompt_words))
                    
                    self.similarity_cache[similarity_key] = text_similarity
        
        logger.info(f"Precomputed {len(self.similarity_cache)} similarity scores")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        total_requests = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
        cache_hit_rate = (self.performance_metrics['cache_hits'] / max(total_requests, 1)) * 100
        
        stats = {
            'cache_hit_rate': round(cache_hit_rate, 1),
            'total_requests': total_requests,
            'cache_hits': self.performance_metrics['cache_hits'],
            'cache_misses': self.performance_metrics['cache_misses'],
            'time_saved_seconds': round(self.performance_metrics['optimization_time_saved'], 2),
            'similarity_cache_size': len(self.similarity_cache),
            'semantic_cache_stats': self.cache.get_cache_stats()
        }
        
        return stats
    
    def clear_caches(self):
        """Clear all caches."""
        self.similarity_cache.clear()
        
        # Clear semantic cache
        try:
            with sqlite3.connect(self.cache.cache_db_path) as conn:
                conn.execute("DELETE FROM semantic_cache")
            logger.info("Cleared all caches")
        except Exception as e:
            logger.error(f"Failed to clear semantic cache: {e}")

class PerformanceMonitor:
    """Monitors and reports on distillation system performance."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {
            'principle_applications': 0,
            'successful_augmentations': 0,
            'failed_augmentations': 0,
            'avg_augmentation_time': 0.0,
            'principle_effectiveness': {},
            'domain_performance': {}
        }
        
        logger.info("Performance monitor initialized")
    
    def record_augmentation(self, success: bool, duration_seconds: float, 
                          principles_applied: int, domain: str = None):
        """Record a prompt augmentation event."""
        self.metrics['principle_applications'] += principles_applied
        
        if success:
            self.metrics['successful_augmentations'] += 1
        else:
            self.metrics['failed_augmentations'] += 1
        
        # Update average duration
        total_augmentations = (self.metrics['successful_augmentations'] + 
                             self.metrics['failed_augmentations'])
        
        current_avg = self.metrics['avg_augmentation_time']
        self.metrics['avg_augmentation_time'] = (
            (current_avg * (total_augmentations - 1) + duration_seconds) / total_augmentations
        )
        
        # Track domain performance
        if domain:
            if domain not in self.metrics['domain_performance']:
                self.metrics['domain_performance'][domain] = {
                    'total': 0, 'successful': 0, 'avg_duration': 0.0
                }
            
            domain_stats = self.metrics['domain_performance'][domain]
            domain_stats['total'] += 1
            if success:
                domain_stats['successful'] += 1
            
            # Update domain average duration
            domain_stats['avg_duration'] = (
                (domain_stats['avg_duration'] * (domain_stats['total'] - 1) + duration_seconds) / 
                domain_stats['total']
            )
    
    def record_principle_effectiveness(self, principle_id: str, effectiveness_score: float):
        """Record principle effectiveness."""
        if principle_id not in self.metrics['principle_effectiveness']:
            self.metrics['principle_effectiveness'][principle_id] = {
                'scores': [], 'avg_effectiveness': 0.0
            }
        
        effectiveness_data = self.metrics['principle_effectiveness'][principle_id]
        effectiveness_data['scores'].append(effectiveness_score)
        
        # Keep only recent scores (last 100)
        if len(effectiveness_data['scores']) > 100:
            effectiveness_data['scores'] = effectiveness_data['scores'][-100:]
        
        # Update average
        effectiveness_data['avg_effectiveness'] = sum(effectiveness_data['scores']) / len(effectiveness_data['scores'])
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        total_augmentations = (self.metrics['successful_augmentations'] + 
                             self.metrics['failed_augmentations'])
        
        success_rate = 0.0
        if total_augmentations > 0:
            success_rate = (self.metrics['successful_augmentations'] / total_augmentations) * 100
        
        # Top performing principles
        top_principles = []
        for principle_id, data in self.metrics['principle_effectiveness'].items():
            if len(data['scores']) >= 5:  # Minimum sample size
                top_principles.append({
                    'principle_id': principle_id,
                    'avg_effectiveness': round(data['avg_effectiveness'], 3),
                    'sample_size': len(data['scores'])
                })
        
        top_principles.sort(key=lambda x: x['avg_effectiveness'], reverse=True)
        
        return {
            'overall_performance': {
                'total_augmentations': total_augmentations,
                'success_rate': round(success_rate, 1),
                'avg_augmentation_time': round(self.metrics['avg_augmentation_time'], 3),
                'total_principle_applications': self.metrics['principle_applications']
            },
            'domain_performance': self.metrics['domain_performance'],
            'top_performing_principles': top_principles[:10],
            'system_health': self._assess_performance_health()
        }
    
    def _assess_performance_health(self) -> Dict[str, Any]:
        """Assess system performance health."""
        total_augmentations = (self.metrics['successful_augmentations'] + 
                             self.metrics['failed_augmentations'])
        
        health = {'status': 'healthy', 'issues': []}
        
        if total_augmentations > 0:
            success_rate = (self.metrics['successful_augmentations'] / total_augmentations) * 100
            
            if success_rate < 80:
                health['status'] = 'degraded'
                health['issues'].append(f'Low success rate: {success_rate:.1f}%')
            
            if self.metrics['avg_augmentation_time'] > 1.0:
                health['status'] = 'warning' if health['status'] == 'healthy' else health['status']
                health['issues'].append(f'High avg augmentation time: {self.metrics["avg_augmentation_time"]:.2f}s')
        
        return health

# Global optimization instances
principle_optimizer = PrincipleOptimizer()
performance_monitor = PerformanceMonitor()
