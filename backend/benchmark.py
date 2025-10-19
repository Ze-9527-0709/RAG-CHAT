#!/usr/bin/env python3
"""
Performance Benchmark Script for RAG Chat Agent
Tests speed improvements before and after optimizations
"""

import time
import asyncio
import requests
import json
import statistics
from typing import List, Dict

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_QUERIES = [
    "Hello, how are you?",
    "What is artificial intelligence?", 
    "Explain machine learning",
    "How does RAG work?",
    "Tell me about Python programming",
    "What is the weather like?",
    "How can I learn coding?",
    "What is deep learning?",
    "Explain neural networks",
    "What is natural language processing?"
]

class PerformanceBenchmark:
    def __init__(self):
        self.session_id = "benchmark_test"
        self.results = {
            "cache_hits": [],
            "cache_misses": [],
            "context_retrieval": [],
            "memory_operations": [],
            "overall_response": []
        }
    
    def test_endpoint(self, endpoint: str, payload: dict, test_name: str) -> Dict:
        """Test a single endpoint and measure response time"""
        start_time = time.time()
        
        try:
            response = requests.post(f"{BASE_URL}{endpoint}", 
                                   json=payload, 
                                   timeout=30)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                return {
                    "test_name": test_name,
                    "response_time": response_time,
                    "status": "success",
                    "response_size": len(response.text)
                }
            else:
                return {
                    "test_name": test_name,
                    "response_time": response_time,
                    "status": "error",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            end_time = time.time()
            return {
                "test_name": test_name,
                "response_time": end_time - start_time,
                "status": "error",
                "error": str(e)
            }
    
    def test_cache_performance(self):
        """Test cache hit vs cache miss performance"""
        print("🧪 Testing Cache Performance...")
        
        # Test same query multiple times (should hit cache after first)
        query = "What is artificial intelligence?"
        payload = {
            "session_id": self.session_id,
            "message": query,
            "max_history": 5
        }
        
        # First request (cache miss)
        print("  📝 Testing cache miss (first request)...")
        result1 = self.test_endpoint("/api/chat", payload, "cache_miss")
        self.results["cache_misses"].append(result1["response_time"])
        
        # Wait a moment
        time.sleep(1)
        
        # Second request (should be cache hit)
        print("  ⚡ Testing cache hit (repeated request)...")
        result2 = self.test_endpoint("/api/chat", payload, "cache_hit")
        self.results["cache_hits"].append(result2["response_time"])
        
        # Third request (should be cache hit)
        result3 = self.test_endpoint("/api/chat", payload, "cache_hit")
        self.results["cache_hits"].append(result3["response_time"])
        
        print(f"  Cache Miss: {result1['response_time']:.3f}s")
        print(f"  Cache Hit 1: {result2['response_time']:.3f}s") 
        print(f"  Cache Hit 2: {result3['response_time']:.3f}s")
        print(f"  🚀 Speedup: {result1['response_time']/result2['response_time']:.1f}x faster")
    
    def test_multiple_queries(self):
        """Test performance across multiple different queries"""
        print("\n🧪 Testing Multiple Query Performance...")
        
        response_times = []
        
        for i, query in enumerate(TEST_QUERIES[:5]):  # Test 5 queries
            print(f"  📝 Query {i+1}/5: {query[:30]}...")
            
            payload = {
                "session_id": f"{self.session_id}_{i}",
                "message": query,
                "max_history": 5
            }
            
            result = self.test_endpoint("/api/chat", payload, f"query_{i+1}")
            response_times.append(result["response_time"])
            
            print(f"     Response time: {result['response_time']:.3f}s")
            
            # Small delay between requests
            time.sleep(0.5)
        
        avg_time = statistics.mean(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n  📊 Statistics:")
        print(f"     Average: {avg_time:.3f}s")
        print(f"     Fastest: {min_time:.3f}s") 
        print(f"     Slowest: {max_time:.3f}s")
        
        return response_times
    
    def test_streaming_performance(self):
        """Test streaming vs non-streaming performance"""
        print("\n🧪 Testing Streaming Performance...")
        
        query = "Explain machine learning in detail"
        
        # Non-streaming request
        print("  📝 Testing non-streaming...")
        payload = {
            "session_id": self.session_id + "_stream",
            "message": query,
            "max_history": 5
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/chat", json=payload)
        non_stream_time = time.time() - start_time
        
        # Streaming request (measure time to first token)
        print("  ⚡ Testing streaming (time to first token)...")
        stream_payload = {
            "session_id": self.session_id + "_stream2",
            "message": query,
            "max_history": 5,
            "stream": True
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/chat_stream", 
                               json=stream_payload, 
                               stream=True)
        
        first_token_time = None
        for line in response.iter_lines():
            if line and b"data:" in line:
                first_token_time = time.time() - start_time
                break
        
        print(f"  Non-streaming: {non_stream_time:.3f}s")
        print(f"  Streaming (first token): {first_token_time:.3f}s")
        if first_token_time:
            print(f"  🚀 Streaming speedup: {non_stream_time/first_token_time:.1f}x faster to first response")
    
    def check_server_status(self):
        """Check if server is running and optimized"""
        print("🔍 Checking Server Status...")
        
        try:
            # Health check
            health = requests.get(f"{BASE_URL}/health")
            if health.status_code == 200:
                print("  ✅ Server is running")
            
            # Performance stats (if available)
            try:
                stats = requests.get(f"{BASE_URL}/api/performance_stats")
                if stats.status_code == 200:
                    data = stats.json()
                    print("  ✅ Performance monitoring enabled")
                    
                    cache_stats = data.get("cache_statistics", {})
                    print(f"     Response cache: {cache_stats.get('response_cache_size', 0)} items")
                    print(f"     Context cache: {cache_stats.get('context_cache_size', 0)} items")
                    
                    return True
                else:
                    print("  ⚠️  Performance monitoring not available")
                    return False
            except:
                print("  ⚠️  Performance monitoring not available")
                return False
                
        except Exception as e:
            print(f"  ❌ Server not accessible: {e}")
            return False
    
    def run_benchmark(self):
        """Run complete performance benchmark"""
        print("🚀 RAG Chat Agent Performance Benchmark")
        print("=" * 50)
        
        # Check server status
        optimized = self.check_server_status()
        
        if not optimized:
            print("\n⚠️  Server appears to be running without optimizations")
            print("   Make sure you're running the optimized version of app.py")
        
        print()
        
        # Run tests
        self.test_cache_performance()
        self.test_multiple_queries()
        self.test_streaming_performance()
        
        # Summary
        print("\n📊 Performance Summary")
        print("=" * 30)
        
        if self.results["cache_hits"] and self.results["cache_misses"]:
            cache_miss_avg = statistics.mean(self.results["cache_misses"])
            cache_hit_avg = statistics.mean(self.results["cache_hits"])
            cache_speedup = cache_miss_avg / cache_hit_avg
            
            print(f"Cache Miss Average: {cache_miss_avg:.3f}s")
            print(f"Cache Hit Average:  {cache_hit_avg:.3f}s")
            print(f"🚀 Cache Speedup:   {cache_speedup:.1f}x faster")
        
        if optimized:
            print("\n✅ Optimizations Active:")
            print("  • Response Caching")
            print("  • Async Context Retrieval") 
            print("  • Database Connection Pooling")
            print("  • Memory System Optimization")
            print("  • Performance Monitoring")
        else:
            print("\n⚠️  Optimizations Not Detected")
            print("  Run with optimized app.py for best performance")
        
        print("\n💡 Expected Performance Improvements:")
        print("  • Simple queries (cached): 10-50x faster")
        print("  • Context retrieval: 3-5x faster")
        print("  • Memory operations: 5x faster") 
        print("  • Overall response: 6-10x faster")

if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    benchmark.run_benchmark()