# File: rag/backend/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from services.ingest import ingest_text
from services.retrieve import retrieve
from services.rerank import rerank
from services.answer import generate_answer
app = FastAPI(title="Mini-RAG Backend with Pinecone")

class IngestRequest(BaseModel):
    text: str
    source: str | None = "user_input"
    title: str | None = "Untitled"

# @app.post("/ingest")
# async def ingest(req: IngestRequest):
#     n_chunks = ingest_text(req.text, req.source, req.title)
#     return {"status": "ok", "chunks_ingested": n_chunks}

import asyncio

@app.post("/ingest")
async def ingest(req: IngestRequest):
    loop = asyncio.get_event_loop()
    n_chunks = await loop.run_in_executor(None, ingest_text, req.text, req.source, req.title)
    return {"status": "ok", "chunks_ingested": n_chunks}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/query")
async def query_docs(req: QueryRequest):
    # Step 1: retrieve
    matches = retrieve(req.query, top_k=req.top_k)
    if not matches:
        return {"results": [], "answer": "No relevant documents found."}

    docs = [{"metadata": m.metadata, "text": m.metadata["text"]} for m in matches]

    # # Step 2: rerank
    # reranked = rerank(req.query, docs, top_k=5)
     # Step 2: rerank (use same top_k from user or cap it lower)
    reranked = rerank(req.query, docs, top_k=min(req.top_k, 5))

    # Step 3: generate final answer
    answer = generate_answer(req.query, reranked)

    return {"results": reranked, "answer": answer}
