"""
Performance Optimization Suite for SAM's Cognitive Memory Core - Phase C
Implements query caching, result pagination, memory optimization, and concurrent processing.
"""

import logging
import asyncio
import time
import weakref
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import wraps, lru_cache
import pickle
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Entry in the performance cache."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[timedelta] = None

@dataclass
class PaginationResult:
    """Result with pagination metadata."""
    data: List[Any]
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    avg_query_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    concurrent_operations: int = 0
    total_operations: int = 0
    error_rate: float = 0.0

class LRUCache:
    """
    High-performance LRU cache with TTL and size limits.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: timedelta = timedelta(hours=1)):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if entry.ttl and datetime.now() - entry.created_at > entry.ttl:
                self._remove_entry(key)
                self._stats["misses"] += 1
                return None
            
            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._stats["hits"] += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Default estimate
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            # Add to cache
            self._cache[key] = entry
            self._stats["size_bytes"] += size_bytes
            
            # Evict if necessary
            while len(self._cache) > self.max_size:
                self._evict_lru()
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self._stats["size_bytes"] -= entry.size_bytes
            del self._cache[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            key = next(iter(self._cache))
            self._remove_entry(key)
            self._stats["evictions"] += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0, "size_bytes": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / max(total_requests, 1)
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "miss_rate": 1 - hit_rate,
                "size_mb": self._stats["size_bytes"] / (1024 * 1024),
                **self._stats
            }

class QueryPaginator:
    """
    Efficient query result pagination.
    """
    
    def __init__(self, default_page_size: int = 20, max_page_size: int = 100):
        self.default_page_size = default_page_size
        self.max_page_size = max_page_size
    
    def paginate(
        self,
        data: List[Any],
        page: int = 1,
        page_size: Optional[int] = None
    ) -> PaginationResult:
        """Paginate a list of data."""
        if page_size is None:
            page_size = self.default_page_size
        
        page_size = min(page_size, self.max_page_size)
        page = max(1, page)  # Ensure page is at least 1
        
        total_items = len(data)
        total_pages = (total_items + page_size - 1) // page_size  # Ceiling division
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        page_data = data[start_idx:end_idx]
        
        return PaginationResult(
            data=page_data,
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )
    
    async def paginate_async_generator(
        self,
        async_gen: AsyncGenerator[Any, None],
        page: int = 1,
        page_size: Optional[int] = None
    ) -> PaginationResult:
        """Paginate an async generator."""
        if page_size is None:
            page_size = self.default_page_size
        
        page_size = min(page_size, self.max_page_size)
        page = max(1, page)
        
        # Collect all data (in production, consider streaming pagination)
        all_data = []
        async for item in async_gen:
            all_data.append(item)
        
        return self.paginate(all_data, page, page_size)

class MemoryOptimizer:
    """
    Memory usage optimization and monitoring.
    """
    
    def __init__(self, memory_threshold_mb: float = 1024.0):
        self.memory_threshold_mb = memory_threshold_mb
        self.logger = logging.getLogger(f"{__name__}.MemoryOptimizer")
        self._weak_refs: weakref.WeakSet = weakref.WeakSet()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        memory_usage = self.get_memory_usage()
        return memory_usage["rss_mb"] > self.memory_threshold_mb
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        initial_memory = self.get_memory_usage()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear weak references
        self._weak_refs.clear()
        
        final_memory = self.get_memory_usage()
        
        freed_mb = initial_memory["rss_mb"] - final_memory["rss_mb"]
        
        self.logger.info(f"Memory optimization: freed {freed_mb:.2f} MB, collected {collected} objects")
        
        return {
            "initial_memory_mb": initial_memory["rss_mb"],
            "final_memory_mb": final_memory["rss_mb"],
            "freed_mb": freed_mb,
            "objects_collected": collected
        }
    
    def register_for_cleanup(self, obj: Any) -> None:
        """Register object for automatic cleanup."""
        self._weak_refs.add(obj)

class ConcurrentProcessor:
    """
    Concurrent processing manager for graph operations.
    """
    
    def __init__(self, max_workers: int = 4, use_processes: bool = False):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self._executor = None
        self._active_tasks: Set[asyncio.Task] = set()
        self.logger = logging.getLogger(f"{__name__}.ConcurrentProcessor")
    
    def __enter__(self):
        if self.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable,
        batch_size: int = 10
    ) -> List[Any]:
        """Process items in concurrent batches."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Create tasks for batch
            tasks = []
            for item in batch:
                if asyncio.iscoroutinefunction(processor_func):
                    task = asyncio.create_task(processor_func(item))
                else:
                    loop = asyncio.get_event_loop()
                    task = loop.run_in_executor(self._executor, processor_func, item)
                
                tasks.append(task)
                self._active_tasks.add(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Clean up completed tasks
            for task in tasks:
                self._active_tasks.discard(task)
            
            # Filter out exceptions and add to results
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)
                else:
                    self.logger.warning(f"Batch processing error: {result}")
        
        return results
    
    def get_active_task_count(self) -> int:
        """Get number of active concurrent tasks."""
        return len(self._active_tasks)

class PerformanceOptimizer:
    """
    Main performance optimization coordinator.

    Features:
    - Multi-level caching (query, result, computation)
    - Intelligent pagination
    - Memory optimization and monitoring
    - Concurrent processing management
    - Performance metrics collection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(f"{__name__}.PerformanceOptimizer")

        # Configuration
        config = config or {}
        self.cache_size = config.get("cache_size", 1000)
        self.cache_ttl = timedelta(minutes=config.get("cache_ttl_minutes", 30))
        self.memory_threshold_mb = config.get("memory_threshold_mb", 1024.0)
        self.max_concurrent_workers = config.get("max_workers", 4)
        self.enable_process_pool = config.get("enable_process_pool", False)

        # Components
        self.query_cache = LRUCache(self.cache_size, self.cache_ttl)
        self.result_cache = LRUCache(self.cache_size // 2, self.cache_ttl)
        self.paginator = QueryPaginator()
        self.memory_optimizer = MemoryOptimizer(self.memory_threshold_mb)

        # Metrics
        self._metrics = PerformanceMetrics()
        self._operation_times: List[float] = []
        self._error_count = 0

        # Background tasks
        self._cleanup_task = None
        self._monitoring_task = None

        self.logger.info("Performance Optimizer initialized")

    async def start_background_tasks(self):
        """Start background optimization tasks."""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._monitoring_task = asyncio.create_task(self._performance_monitoring())
        self.logger.info("Background optimization tasks started")

    async def stop_background_tasks(self):
        """Stop background optimization tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(
            self._cleanup_task, self._monitoring_task,
            return_exceptions=True
        )

        self.logger.info("Background optimization tasks stopped")

    def cached_query(self, cache_key: Optional[str] = None, ttl: Optional[timedelta] = None):
        """Decorator for caching query results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key:
                    key = cache_key
                else:
                    key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                    key = hashlib.md5(key_data.encode()).hexdigest()

                # Try cache first
                cached_result = self.query_cache.get(key)
                if cached_result is not None:
                    return cached_result

                # Execute function
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)

                    # Cache result
                    self.query_cache.put(key, result, ttl)

                    # Update metrics
                    execution_time = time.time() - start_time
                    self._update_operation_metrics(execution_time, success=True)

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    self._update_operation_metrics(execution_time, success=False)
                    raise

            return wrapper
        return decorator

    async def paginate_results(
        self,
        data_source: Union[List[Any], AsyncGenerator[Any, None]],
        page: int = 1,
        page_size: Optional[int] = None
    ) -> PaginationResult:
        """Paginate results with caching."""
        if isinstance(data_source, list):
            return self.paginator.paginate(data_source, page, page_size)
        else:
            return await self.paginator.paginate_async_generator(data_source, page, page_size)

    async def process_concurrently(
        self,
        items: List[Any],
        processor_func: Callable,
        batch_size: int = 10
    ) -> List[Any]:
        """Process items concurrently with optimization."""
        with ConcurrentProcessor(
            max_workers=self.max_concurrent_workers,
            use_processes=self.enable_process_pool
        ) as processor:

            start_time = time.time()
            try:
                results = await processor.process_batch(items, processor_func, batch_size)

                execution_time = time.time() - start_time
                self._update_operation_metrics(execution_time, success=True)

                return results

            except Exception as e:
                execution_time = time.time() - start_time
                self._update_operation_metrics(execution_time, success=False)
                raise

    def optimize_memory_if_needed(self) -> Optional[Dict[str, Any]]:
        """Optimize memory if under pressure."""
        if self.memory_optimizer.check_memory_pressure():
            self.logger.warning("Memory pressure detected, optimizing...")
            return self.memory_optimizer.optimize_memory()
        return None

    async def _periodic_cleanup(self):
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Memory optimization
                self.optimize_memory_if_needed()

                # Cache cleanup (handled by TTL, but we can force cleanup here)
                # Clear old entries beyond TTL

                self.logger.debug("Periodic cleanup completed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")

    async def _performance_monitoring(self):
        """Performance monitoring task."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute

                # Update metrics
                self._update_performance_metrics()

                # Log performance summary
                metrics = self.get_performance_metrics()
                self.logger.info(
                    f"Performance: Cache hit rate: {metrics.cache_hit_rate:.2f}, "
                    f"Avg query time: {metrics.avg_query_time:.3f}s, "
                    f"Memory: {metrics.memory_usage_mb:.1f}MB"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")

    def _update_operation_metrics(self, execution_time: float, success: bool):
        """Update operation metrics."""
        self._operation_times.append(execution_time)

        # Keep only recent times (last 1000 operations)
        if len(self._operation_times) > 1000:
            self._operation_times = self._operation_times[-1000:]

        if not success:
            self._error_count += 1

    def _update_performance_metrics(self):
        """Update performance metrics."""
        # Cache metrics
        query_stats = self.query_cache.get_stats()
        self._metrics.cache_hit_rate = query_stats["hit_rate"]
        self._metrics.cache_miss_rate = query_stats["miss_rate"]

        # Query time metrics
        if self._operation_times:
            self._metrics.avg_query_time = sum(self._operation_times) / len(self._operation_times)

        # Memory metrics
        memory_usage = self.memory_optimizer.get_memory_usage()
        self._metrics.memory_usage_mb = memory_usage["rss_mb"]

        # CPU metrics
        self._metrics.cpu_usage_percent = psutil.cpu_percent()

        # Error rate
        total_ops = len(self._operation_times)
        self._metrics.error_rate = self._error_count / max(total_ops, 1)
        self._metrics.total_operations = total_ops

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        self._update_performance_metrics()
        return self._metrics

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "query_cache": self.query_cache.get_stats(),
            "result_cache": self.result_cache.get_stats(),
            "total_cache_size_mb": (
                self.query_cache.get_stats()["size_mb"] +
                self.result_cache.get_stats()["size_mb"]
            )
        }

    def clear_all_caches(self):
        """Clear all caches."""
        self.query_cache.clear()
        self.result_cache.clear()
        self.logger.info("All caches cleared")

    async def shutdown(self):
        """Shutdown the performance optimizer."""
        await self.stop_background_tasks()
        self.clear_all_caches()
        self.logger.info("Performance Optimizer shutdown complete")
