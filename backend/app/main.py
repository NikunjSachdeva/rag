# File: rag/backend/app/main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.ingest import ingest_text_async, ingest_text
from services.retrieve import retrieve_async, retrieve
from services.rerank import rerank_async, rerank
from services.answer import generate_answer_with_citations, generate_answer_with_citations_sync
import asyncio
import time
from typing import Optional, List

app = FastAPI(
    title="Mini-RAG Backend with Pinecone",
    description="A high-performance RAG system with citations and scores",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestRequest(BaseModel):
    text: str
    source: str | None = "user_input"
    title: str | None = "Untitled"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    include_scores: bool = True

class BatchQueryRequest(BaseModel):
    queries: List[str]
    top_k: int = 10
    include_scores: bool = True

@app.post("/ingest")
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks
):
    """Ingest text with enhanced error handling and timeout prevention"""
    try:
        print(f"🚀 Starting ingestion for text of {len(request.text)} characters")
        
        # Use configuration-optimized settings
        result = await ingest_text_async(
            text=request.text,
            source=request.source,
            title=request.title,
            embed_batch_size=32,  # Reduced for reliability
            upsert_batch_size=50  # Reduced for reliability
        )
        
        if result['status'] == 'success':
            print(f"✅ Ingestion successful: {result['chunks_ingested']} chunks")
            return {
                "status": "success",
                "message": f"Successfully ingested {result['chunks_ingested']} chunks",
                "details": result
            }
        else:
            print(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
            return {
                "status": "error",
                "message": f"Ingestion failed: {result.get('error', 'Unknown error')}",
                "details": result
            }
            
    except Exception as e:
        error_msg = f"Ingestion failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Check if it's a timeout error
        if "504" in str(e) or "Deadline Exceeded" in str(e):
            return {
                "status": "error",
                "message": "Request timed out. Please try with a smaller text or check your connection.",
                "error_type": "timeout",
                "suggestion": "Consider breaking your text into smaller sections."
            }
        
        return {
            "status": "error",
            "message": error_msg,
            "error_type": "general"
        }

@app.post("/ingest/sync")
async def ingest_sync(req: IngestRequest):
    """Synchronous ingestion for backward compatibility"""
    try:
        result = ingest_text(req.text, req.source, req.title)
        return {
            "status": "success",
            "data": result,
            "message": f"Successfully ingested {result.get('chunks_ingested', 0)} chunks"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query")
async def query_docs(req: QueryRequest):
    """Complete RAG pipeline: retrieve → rerank → answer with citations"""
    start_time = time.time()
    
    try:
        # Step 1: Retrieve relevant documents
        print(f"🔍 Retrieving documents for query: {req.query}")
        matches = await retrieve_async(req.query, top_k=req.top_k)
        
        if not matches:
            return {
                "status": "no_results",
                "answer": "No relevant documents found for your query.",
                "citations": [],
                "sources": [],
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "documents_retrieved": 0,
                    "status": "no_documents"
                },
                "query": req.query
            }

        # Step 2: Prepare documents for reranking
        docs = []
        for match in matches:
            doc = {
                "metadata": match.metadata,
                "text": match.metadata.get("text", ""),
                "score": getattr(match, 'score', 0.0)  # Pinecone similarity score
            }
            docs.append(doc)

        print(f"📚 Retrieved {len(docs)} documents, proceeding to reranking...")

        # Step 3: Rerank documents for better relevance
        reranked = await rerank_async(req.query, docs, top_k=min(req.top_k, 5))
        
        if not reranked:
            print("⚠️ Reranking failed, using original documents")
            reranked = docs[:5]  # Fallback to top 5 original docs

        print(f"🎯 Reranked to {len(reranked)} documents, generating answer...")

        # Step 4: Generate answer with citations and scores
        answer_result = await generate_answer_with_citations(req.query, reranked)
        
        # Add overall processing time
        total_time = time.time() - start_time
        answer_result["metadata"]["total_processing_time"] = total_time
        answer_result["metadata"]["retrieval_time"] = total_time - answer_result["metadata"].get("processing_time", 0)
        
        print(f"✅ Answer generated in {total_time:.2f}s")
        
        return {
            "status": "success",
            **answer_result,
            "request_time": total_time
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Query processing failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Query processing failed: {str(e)}"
        )

@app.post("/query/sync")
async def query_docs_sync(req: QueryRequest):
    """Synchronous query endpoint for backward compatibility"""
    start_time = time.time()
    
    try:
        # Step 1: Retrieve using sync function
        matches = retrieve(req.query, top_k=req.top_k)
        if not matches:
            return {
                "status": "no_results",
                "answer": "No relevant documents found for your query.",
                "citations": [],
                "sources": [],
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "documents_retrieved": 0,
                    "status": "no_documents"
                },
                "query": req.query
            }

        # Step 2: Prepare documents
        docs = []
        for match in matches:
            doc = {
                "metadata": match.metadata,
                "text": match.metadata.get("text", ""),
                "score": getattr(match, 'score', 0.0)
            }
            docs.append(doc)

        # Step 3: Rerank using sync function
        reranked = rerank(req.query, docs, top_k=min(req.top_k, 5))
        if not reranked:
            reranked = docs[:5]

        # Step 4: Generate answer with citations
        answer_result = generate_answer_with_citations_sync(req.query, reranked)
        
        total_time = time.time() - start_time
        answer_result["metadata"]["total_processing_time"] = total_time
        
        return {
            "status": "success",
            **answer_result,
            "request_time": total_time
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        raise HTTPException(
            status_code=500, 
            detail=f"Query processing failed: {str(e)}"
        )

@app.post("/batch-query")
async def batch_query_docs(req: BatchQueryRequest):
    """Batch query processing for multiple questions"""
    start_time = time.time()
    
    try:
        results = []
        for query in req.queries:
            # Process each query individually
            query_result = await query_docs(QueryRequest(
                query=query,
                top_k=req.top_k,
                include_scores=req.include_scores
            ))
            results.append(query_result)
        
        total_time = time.time() - start_time
        
        return {
            "status": "success",
            "results": results,
            "total_queries": len(req.queries),
            "total_processing_time": total_time,
            "average_time_per_query": total_time / len(req.queries) if req.queries else 0
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Batch query processing failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Mini-RAG Backend",
        "version": "1.0.0",
        "features": [
            "async_ingestion",
            "vector_retrieval", 
            "document_reranking",
            "llm_answering",
            "citation_system",
            "score_tracking"
        ]
    }

@app.get("/performance")
async def performance_metrics():
    """Get performance metrics"""
    from config import performance_metrics
    return {
        "status": "success",
        "metrics": performance_metrics.get_performance_summary(),
        "system_info": {
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "embedding_dimensions": 768,
            "vector_db": "Pinecone",
            "llm_provider": "Google Gemini Pro",
            "reranker": "Cohere"
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        from services.retrieve import index
        
        # Get index stats from Pinecone
        stats = index.describe_index_stats()
        
        return {
            "status": "success",
            "vector_db_stats": {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_name": stats.index_name,
                "metric": stats.metric
            },
            "system_config": {
                "chunk_size": 1000,
                "chunk_overlap": 150,
                "embedding_model": "models/embedding-001",
                "llm_model": "gemini-1.5-flash",
                "reranker_model": "rerank-english-v3.0"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get stats: {str(e)}"
        )
