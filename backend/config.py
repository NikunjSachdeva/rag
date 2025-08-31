# backend/config.py
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Performance-optimized configuration for RAG system"""
    
    # Pinecone settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")
    
    # Google AI settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Cohere settings
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    
    # Performance optimization settings
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))
    THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "4"))
    
    # Caching settings
    EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "300"))  # 5 minutes
    RERANK_CACHE_TTL = int(os.getenv("RERANK_CACHE_TTL", "600"))       # 10 minutes
    MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    
    # Text processing settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    
    # API timeout settings
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
    
    # Query optimization
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
    
    @classmethod
    def get_optimization_settings(cls) -> Dict[str, Any]:
        """Get all optimization settings as a dictionary"""
        return {
            "embedding_batch_size": cls.EMBEDDING_BATCH_SIZE,
            "upsert_batch_size": cls.UPSERT_BATCH_SIZE,
            "thread_pool_workers": cls.THREAD_POOL_WORKERS,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "request_timeout": cls.REQUEST_TIMEOUT,
            "max_retries": cls.MAX_RETRIES,
        }
    
    @classmethod
    def get_cache_settings(cls) -> Dict[str, Any]:
        """Get cache-related settings"""
        return {
            "embedding_cache_ttl": cls.EMBEDDING_CACHE_TTL,
            "rerank_cache_ttl": cls.RERANK_CACHE_TTL,
            "max_cache_size": cls.MAX_CACHE_SIZE,
        }

# Performance monitoring
class PerformanceMetrics:
    """Track performance metrics for optimization"""
    
    def __init__(self):
        self.ingestion_times = []
        self.query_times = []
        self.embedding_times = []
        self.pinecone_times = []
    
    def add_ingestion_time(self, time_taken: float):
        self.ingestion_times.append(time_taken)
        if len(self.ingestion_times) > 100:
            self.ingestion_times.pop(0)
    
    def add_query_time(self, time_taken: float):
        self.query_times.append(time_taken)
        if len(self.query_times) > 100:
            self.query_times.pop(0)
    
    def get_average_ingestion_time(self) -> float:
        return sum(self.ingestion_times) / len(self.ingestion_times) if self.ingestion_times else 0
    
    def get_average_query_time(self) -> float:
        return sum(self.query_times) / len(self.query_times) if self.query_times else 0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            "avg_ingestion_time": self.get_average_ingestion_time(),
            "avg_query_time": self.get_average_query_time(),
            "total_ingestions": len(self.ingestion_times),
            "total_queries": len(self.query_times),
        }

# Global performance tracker
performance_metrics = PerformanceMetrics()
