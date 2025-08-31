# backend/app/services/rerank.py
import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import cohere
from dotenv import load_dotenv

load_dotenv()

# Global Cohere client with connection pooling
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Simple cache for reranking results
_rerank_cache = {}
_cache_ttl = 600  # 10 minutes

def _get_cache_key(query: str, docs: List[str], top_k: int) -> str:
    """Generate cache key for reranking"""
    # Create a hash of the query and first few characters of each doc
    doc_signatures = [doc[:100] for doc in docs[:5]]  # First 100 chars of first 5 docs
    content = f"{query}_{top_k}_{'_'.join(doc_signatures)}"
    return f"rerank_{hash(content) % 10000}"

def _is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid"""
    return time.time() - timestamp < _cache_ttl

async def rerank_async(query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
    """Async reranking with caching and optimized processing"""
    # Handle empty documents list
    if not docs:
        return []
    
    # Check cache first
    cache_key = _get_cache_key(query, [d["metadata"]["text"] for d in docs], top_k)
    if cache_key in _rerank_cache:
        cached_result, timestamp = _rerank_cache[cache_key]
        if _is_cache_valid(timestamp):
            return cached_result
    
    # docs: list of dicts with { "text": str, "metadata": {...} }
    rerank_docs = [d["metadata"]["text"] for d in docs]
    
    # Handle case where all documents might be empty strings
    if not any(rerank_docs):
        return []
    
    try:
        # Use thread pool for Cohere API call
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=rerank_docs,
                top_n=top_k
            )
        )
        
        ranked = []
        # Access the results from the response object
        for r in response.results:
            ranked.append({
                "text": rerank_docs[r.index],
                "metadata": docs[r.index]["metadata"],
                "score": r.relevance_score
            })
        
        # Cache the result
        _rerank_cache[cache_key] = (ranked, time.time())
        
        # Clean old cache entries
        if len(_rerank_cache) > 500:
            current_time = time.time()
            _rerank_cache.clear()
        
        return ranked
        
    except Exception as e:
        print(f"Rerank error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return original docs if reranking fails
        fallback_result = [
            {
                "text": doc["metadata"]["text"], 
                "metadata": doc["metadata"], 
                "score": 0.0
            } 
            for doc in docs[:top_k]
        ]
        return fallback_result

def rerank(query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
    """Synchronous wrapper for backward compatibility"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use run_until_complete
        return loop.run_until_complete(rerank_async(query, docs, top_k))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(rerank_async(query, docs, top_k))

# Batch reranking for multiple queries
async def batch_rerank_async(queries: List[str], docs_list: List[List[Dict]], top_k: int = 5) -> List[List[Dict]]:
    """Batch reranking for multiple queries - much faster than individual calls"""
    if not queries or not docs_list:
        return []
    
    # Process all reranking tasks in parallel
    rerank_tasks = [
        rerank_async(query, docs, top_k) 
        for query, docs in zip(queries, docs_list)
    ]
    
    results = await asyncio.gather(*rerank_tasks, return_exceptions=True)
    
    # Handle any failed reranking operations
    final_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Batch rerank error: {result}")
            final_results.append([])
        else:
            final_results.append(result)
    
    return final_results

def batch_rerank(queries: List[str], docs_list: List[List[Dict]], top_k: int = 5) -> List[List[Dict]]:
    """Synchronous wrapper for batch reranking"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use run_until_complete
        return loop.run_until_complete(batch_rerank_async(queries, docs_list, top_k))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(batch_rerank_async(queries, docs_list, top_k))