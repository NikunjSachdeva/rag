# backend/app/services/retrieve.py
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import functools

load_dotenv()

# Global connection pool for Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Optimized embeddings with connection pooling
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    request_timeout=30,
    max_retries=2
)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Simple in-memory cache for query embeddings
_embedding_cache = {}
_cache_ttl = 300  # 5 minutes

def _get_cache_key(query: str) -> str:
    """Generate cache key for query"""
    return f"embed_{hash(query) % 10000}"

def _is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid"""
    return time.time() - timestamp < _cache_ttl

async def safe_embed_query_async(query: str, retries: int = 3, delay: float = 1.0) -> List[float]:
    """Async embedding with caching and retries"""
    # Check cache first
    cache_key = _get_cache_key(query)
    if cache_key in _embedding_cache:
        cached_emb, timestamp = _embedding_cache[cache_key]
        if _is_cache_valid(timestamp):
            return cached_emb
    
    # Not in cache or expired, fetch new embedding
    for attempt in range(retries):
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(executor, embeddings.embed_query, query)
            
            # Cache the result
            _embedding_cache[cache_key] = (embedding, time.time())
            
            # Clean old cache entries
            if len(_embedding_cache) > 1000:
                current_time = time.time()
                _embedding_cache.clear()
            
            return embedding
            
        except Exception as e:
            print(f"Embedding attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                raise e

def safe_embed_query(query: str, retries: int = 3, delay: float = 1.0) -> List[float]:
    """Synchronous wrapper for backward compatibility"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use run_in_executor
        return loop.run_until_complete(safe_embed_query_async(query, retries, delay))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(safe_embed_query_async(query, retries, delay))

async def retrieve_async(query: str, top_k: int = 10) -> List:
    """Async retrieval with optimized query processing"""
    if len(query) > 2000:  # safeguard for Gemini
        query = query[:2000]
    
    # Get query embedding
    query_emb = await safe_embed_query_async(query)
    
    # Query Pinecone with optimized parameters
    try:
        results = index.query(
            vector=query_emb, 
            top_k=top_k, 
            include_metadata=True,
            include_values=False  # Don't return vectors to save bandwidth
        )
        return results.matches
    except Exception as e:
        print(f"Pinecone query error: {e}")
        return []

def retrieve(query: str, top_k: int = 10) -> List:
    """Synchronous wrapper for backward compatibility"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use run_until_complete
        return loop.run_until_complete(retrieve_async(query, top_k))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(retrieve_async(query, top_k))

# Batch retrieval for multiple queries
async def batch_retrieve_async(queries: List[str], top_k: int = 10) -> List[List]:
    """Batch retrieval for multiple queries - much faster than individual calls"""
    if not queries:
        return []
    
    # Get embeddings for all queries in parallel
    embedding_tasks = [safe_embed_query_async(query) for query in queries]
    embeddings_list = await asyncio.gather(*embedding_tasks, return_exceptions=True)
    
    # Filter out failed embeddings
    valid_embeddings = []
    valid_queries = []
    for i, emb in enumerate(embeddings_list):
        if not isinstance(emb, Exception):
            valid_embeddings.append(emb)
            valid_queries.append(queries[i])
    
    if not valid_embeddings:
        return [[] for _ in queries]
    
    # Batch query Pinecone (if supported) or parallel individual queries
    try:
        # Try batch query if available
        results = index.query(
            vectors=valid_embeddings,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        return results.matches
    except:
        # Fallback to parallel individual queries
        query_tasks = [retrieve_async(query, top_k) for query in valid_queries]
        results = await asyncio.gather(*query_tasks, return_exceptions=True)
        
        # Pad results for failed queries
        final_results = []
        query_idx = 0
        for i in range(len(queries)):
            if query_idx < len(valid_queries) and queries[i] == valid_queries[query_idx]:
                if isinstance(results[query_idx], Exception):
                    final_results.append([])
                else:
                    final_results.append(results[query_idx])
                query_idx += 1
            else:
                final_results.append([])
        
        return final_results
