# File: rag/backend/app/main.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from services.ingest import ingest_text_async, ingest_text
from services.retrieve import retrieve_async, retrieve
from services.rerank import rerank_async, rerank
from services.answer import generate_answer
import asyncio
import time

app = FastAPI(title="Mini-RAG Backend with Pinecone")

class IngestRequest(BaseModel):
    text: str
    source: str | None = "user_input"
    title: str | None = "Untitled"

@app.post("/ingest")
async def ingest(req: IngestRequest):
    """Optimized async ingestion endpoint"""
    start_time = time.time()
    n_chunks = await ingest_text_async(req.text, req.source, req.title)
    elapsed_time = time.time() - start_time
    
    return {
        "status": "ok", 
        "chunks_ingested": n_chunks,
        "processing_time": f"{elapsed_time:.2f}s"
    }

@app.post("/ingest/sync")
async def ingest_sync(req: IngestRequest):
    """Synchronous ingestion for backward compatibility"""
    n_chunks = ingest_text(req.text, req.source, req.title)
    return {"status": "ok", "chunks_ingested": n_chunks}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/query")
async def query_docs(req: QueryRequest):
    """Optimized query endpoint with parallel processing"""
    start_time = time.time()
    
    # Step 1: retrieve using async function
    matches = await retrieve_async(req.query, top_k=req.top_k)
    if not matches:
        return {"results": [], "answer": "No relevant documents found."}

    docs = [{"metadata": m.metadata, "text": m.metadata["text"]} for m in matches]

    # Step 2: rerank using async function
    reranked = await rerank_async(req.query, docs, top_k=min(req.top_k, 5))

    # Step 3: generate final answer
    answer = generate_answer(req.query, reranked)
    
    elapsed_time = time.time() - start_time
    
    return {
        "results": reranked, 
        "answer": answer,
        "processing_time": f"{elapsed_time:.2f}s",
        "chunks_retrieved": len(matches)
    }

@app.post("/query/sync")
async def query_docs_sync(req: QueryRequest):
    """Synchronous query endpoint for backward compatibility"""
    start_time = time.time()
    
    # Step 1: retrieve using sync function
    matches = retrieve(req.query, top_k=req.top_k)
    if not matches:
        return {"results": [], "answer": "No relevant documents found."}

    docs = [{"metadata": m.metadata, "text": m.metadata["text"]} for m in matches]

    # Step 2: rerank using sync function
    reranked = rerank(req.query, docs, top_k=min(req.top_k, 5))

    # Step 3: generate final answer
    answer = generate_answer(req.query, reranked)
    
    elapsed_time = time.time() - start_time
    
    return {
        "results": reranked, 
        "answer": answer,
        "processing_time": f"{elapsed_time:.2f}s",
        "chunks_retrieved": len(matches)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Mini-RAG Backend"}

@app.get("/performance")
async def performance_metrics():
    """Get performance metrics"""
    from config import performance_metrics
    return performance_metrics.get_performance_summary()
