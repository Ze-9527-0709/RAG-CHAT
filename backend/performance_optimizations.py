# Performance Optimization Module for RAG Chat Agent
# 🚀 Speed improvements for faster responses

import asyncio
import time
from functools import lru_cache, wraps
from typing import Dict, List, Optional, Tuple
import sqlite3
import threading
from contextlib import contextmanager

# ===== 1. RESPONSE CACHING SYSTEM =====

class ResponseCache:
    """In-memory cache for frequently asked questions"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl_seconds
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response"""
        with self.lock:
            if key in self.cache and not self._is_expired(key):
                return self.cache[key]
            elif key in self.cache:
                # Remove expired entry
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: str):
        """Cache response"""
        with self.lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

# Global cache instance
response_cache = ResponseCache()

# ===== 2. DATABASE CONNECTION POOLING =====

class DatabasePool:
    """Connection pool for SQLite databases"""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.pool = []
        self.max_connections = max_connections
        self.lock = threading.RLock()
        
        # Pre-create connections
        for _ in range(min(3, max_connections)):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute('PRAGMA journal_mode = WAL')  # Better concurrency
            conn.execute('PRAGMA synchronous = NORMAL')  # Faster writes
            self.pool.append(conn)
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        conn = None
        try:
            with self.lock:
                if self.pool:
                    conn = self.pool.pop()
                else:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.execute('PRAGMA journal_mode = WAL')
                    conn.execute('PRAGMA synchronous = NORMAL')
            
            yield conn
            
        finally:
            if conn:
                with self.lock:
                    if len(self.pool) < self.max_connections:
                        self.pool.append(conn)
                    else:
                        conn.close()

# ===== 3. CONTEXT RETRIEVAL OPTIMIZATION =====

class OptimizedRetriever:
    """Optimized context retrieval with caching and parallel processing"""
    
    def __init__(self, retriever, cache_size: int = 500):
        self.retriever = retriever
        self.context_cache = {}
        self.cache_timestamps = {}
        self.cache_size = cache_size
        self.lock = threading.RLock()
    
    def _cache_key(self, query: str, k: int) -> str:
        """Generate cache key for query"""
        return f"{query.lower().strip()}:{k}"
    
    async def retrieve_context_optimized(self, query: str, k: int = 4) -> Tuple[str, List]:
        """Optimized context retrieval with caching"""
        cache_key = self._cache_key(query, k)
        
        # Check cache first
        with self.lock:
            if cache_key in self.context_cache:
                cache_time = self.cache_timestamps.get(cache_key, 0)
                # Cache valid for 10 minutes
                if time.time() - cache_time < 600:
                    return self.context_cache[cache_key]
        
        # Retrieve from vector store
        try:
            if self.retriever is None:
                return "", []
            
            # Use asyncio for non-blocking retrieval
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, self.retriever.invoke, query)
            
            # Process results
            cites, parts = [], []
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page")
                preview = d.page_content[:240].replace("\n", " ")
                cites.append({"source": src, "preview": preview, "page": str(page) if page is not None else ""})
                parts.append(f"[{i}] {d.page_content}\n(source: {src}" + (f", page: {page})" if page is not None else ")"))
            
            context = "\n\n".join(parts)
            result = (context, cites)
            
            # Cache result
            with self.lock:
                # Remove oldest cache entries if needed
                if len(self.context_cache) >= self.cache_size:
                    oldest_key = min(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
                    del self.context_cache[oldest_key]
                    del self.cache_timestamps[oldest_key]
                
                self.context_cache[cache_key] = result
                self.cache_timestamps[cache_key] = time.time()
            
            return result
            
        except Exception as e:
            print(f"⚠️ Optimized RAG retrieval failed: {e}")
            return "", []

# ===== 4. MEMORY SYSTEM OPTIMIZATION =====

class OptimizedMemorySystem:
    """Optimized memory system with batch operations and indexing"""
    
    def __init__(self, db_path: str):
        self.db_pool = DatabasePool(db_path)
        self.memory_cache = {}
        self.cache_lock = threading.RLock()
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for better query performance"""
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Indexes for conversations table
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversations_session_timestamp 
                ON conversations(session_id, timestamp DESC)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversations_user_message 
                ON conversations(user_message)
            ''')
            
            # Indexes for learned_knowledge table
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_learned_knowledge_concept 
                ON learned_knowledge(concept)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_learned_knowledge_usage 
                ON learned_knowledge(usage_count DESC)
            ''')
            
            conn.commit()
    
    @lru_cache(maxsize=100)
    def get_user_style_cached(self, session_id: str) -> Dict[str, float]:
        """Cached user style analysis"""
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_message FROM conversations 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''', (session_id,))
            
            messages = [row[0] for row in cursor.fetchall()]
            
            if not messages:
                return {'formality': 0.5, 'detail_level': 0.5, 'friendliness': 0.5}
            
            # Fast style analysis
            total_chars = sum(len(msg) for msg in messages)
            avg_length = total_chars / len(messages)
            
            formal_words = ['please', 'thank', 'kindly']
            informal_words = ['hi', 'hey', 'thanks']
            
            formal_count = sum(msg.lower().count(word) for msg in messages for word in formal_words)
            informal_count = sum(msg.lower().count(word) for msg in messages for word in informal_words)
            
            return {
                'formality': min(formal_count / max(informal_count + formal_count, 1), 1.0),
                'detail_level': min(avg_length / 100, 1.0),
                'friendliness': 0.7 if informal_count > formal_count else 0.5
            }

# ===== 5. BATCH PROCESSING OPTIMIZATION =====

class BatchProcessor:
    """Batch processor for memory operations"""
    
    def __init__(self, batch_size: int = 10, flush_interval: float = 5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending_operations = []
        self.last_flush = time.time()
        self.lock = threading.RLock()
    
    def add_operation(self, operation: Dict):
        """Add operation to batch"""
        with self.lock:
            self.pending_operations.append(operation)
            
            # Auto-flush if batch is full or time interval passed
            if (len(self.pending_operations) >= self.batch_size or 
                time.time() - self.last_flush > self.flush_interval):
                self._flush_operations()
    
    def _flush_operations(self):
        """Execute batched operations"""
        if not self.pending_operations:
            return
        
        # Group operations by type
        conversations = []
        feedbacks = []
        
        for op in self.pending_operations:
            if op['type'] == 'conversation':
                conversations.append(op)
            elif op['type'] == 'feedback':
                feedbacks.append(op)
        
        # Execute batched inserts (implement in memory system)
        # This would be integrated with the actual memory system
        
        self.pending_operations.clear()
        self.last_flush = time.time()

# ===== 6. ASYNC UTILITIES =====

def async_cache(ttl_seconds: int = 300):
    """Decorator for caching async function results"""
    def decorator(func):
        cache = {}
        cache_timestamps = {}
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if key in cache and time.time() - cache_timestamps[key] < ttl_seconds:
                return cache[key]
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[key] = result
            cache_timestamps[key] = time.time()
            
            return result
        
        return wrapper
    return decorator

# ===== 7. PERFORMANCE MONITORING =====

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.RLock()
    
    def record_timing(self, operation: str, duration: float):
        """Record operation timing"""
        with self.lock:
            if operation not in self.metrics:
                self.metrics[operation] = []
            
            self.metrics[operation].append(duration)
            
            # Keep only last 100 measurements
            if len(self.metrics[operation]) > 100:
                self.metrics[operation] = self.metrics[operation][-100:]
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get performance statistics"""
        with self.lock:
            if operation not in self.metrics or not self.metrics[operation]:
                return {}
            
            timings = self.metrics[operation]
            return {
                'avg': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings),
                'count': len(timings)
            }

# Global performance monitor
perf_monitor = PerformanceMonitor()

def timing_decorator(operation_name: str):
    """Decorator to measure function execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                perf_monitor.record_timing(operation_name, duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                perf_monitor.record_timing(operation_name, duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator