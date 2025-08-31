import os
import time
import asyncio
import multiprocessing
import re
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import HumanMessage, SystemMessage
import cohere
from dotenv import load_dotenv

load_dotenv()

# ========== GLOBAL CONFIGURATION ==========
MAX_WORKERS = min(64, multiprocessing.cpu_count() * 8)  # 8x CPU cores, max 64
EMBED_BATCH_SIZE = 60  # Larger batches for throughput
UPSERT_BATCH_SIZE = 60  # Larger batches for throughput
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

print(f"🚀 Using {MAX_WORKERS} threads for maximum performance")

# ========== GLOBAL SERVICES INIT ==========
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
co = cohere.Client(os.getenv("COHERE_API_KEY"))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Optimized embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    request_timeout=60,
    max_retries=2
)

# Optimized text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Optimized LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,
    max_output_tokens=512,
    request_timeout=20,
    max_retries=1
)

# Pre-compiled regex for fast citation extraction
citation_pattern = re.compile(r'\[(\d+)\]')

# ========== OPTIMIZED INGESTION ==========
def _batch_list(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

async def _parallel_embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(executor, embeddings.embed_documents, texts)
    except Exception:
        return [[0.0] * 768 for _ in range(len(texts))]

async def _parallel_pinecone_upsert(vectors: List[Dict[str, Any]]):
    if not vectors:
        return
    
    batches = list(_batch_list(vectors, UPSERT_BATCH_SIZE))
    
    async def upsert_batch(batch):
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(executor, lambda: index.upsert(vectors=batch))
        except Exception:
            pass
    
    await asyncio.gather(*[upsert_batch(batch) for batch in batches])

def _create_metadata(doc: Document, source: str, title: str, chunk_id: int, total_chunks: int) -> Dict[str, Any]:
    return {
        "source": source,
        "title": title,
        "text": doc.page_content,
        "chunk_id": chunk_id,
        "total_chunks": total_chunks,
        "hash": sha256(doc.page_content.encode("utf-8")).hexdigest()[:16],
    }

async def ingest_text_async(text: str, source: str = "user_input", title: str = "Untitled") -> int:
    start_time = time.time()
    
    try:
        # Parallel text splitting
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(executor, text_splitter.split_text, text)
        if not docs:
            return 0
        
        total_chunks = len(docs)
        
        # Create documents with metadata
        documents = []
        for i, doc in enumerate(docs):
            metadata = _create_metadata(Document(page_content=doc), source, title, i, total_chunks)
            documents.append(Document(page_content=doc, metadata=metadata))
        
        # Parallel batch embedding
        texts = [d.page_content for d in documents]
        embeddings_list = await _parallel_embed_batch(texts)
        
        # Prepare vectors
        vectors = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings_list)):
            vectors.append({
                "id": f"{source}_{title}_{i}_{doc.metadata['hash']}",
                "values": emb,
                "metadata": doc.metadata
            })
        
        # Parallel Pinecone upsert
        await _parallel_pinecone_upsert(vectors)
        
        print(f"✅ Ingested {total_chunks} chunks in {time.time() - start_time:.2f}s")
        return total_chunks
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return 0

# ========== OPTIMIZED RETRIEVAL ==========
_embedding_cache = {}

async def safe_embed_query_async(query: str) -> List[float]:
    cache_key = hash(query)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    
    try:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(executor, embeddings.embed_query, query)
        _embedding_cache[cache_key] = embedding
        return embedding
    except Exception:
        return [0.0] * 768

async def retrieve_async(query: str, top_k: int = TOP_K) -> List:
    if len(query) > 2000:
        query = query[:2000]
    
    query_emb = await safe_embed_query_async(query)
    
    try:
        results = index.query(
            vector=query_emb, 
            top_k=top_k, 
            include_metadata=True,
            include_values=False
        )
        return results.matches
    except Exception:
        return []

# ========== OPTIMIZED RERANKING ==========
_rerank_cache = {}

async def rerank_async(query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
    cache_key = hash(f"{query}_{top_k}")
    if cache_key in _rerank_cache:
        return _rerank_cache[cache_key]
    
    if not docs:
        return []
    
    rerank_docs = [d["metadata"]["text"] for d in docs]
    if not any(rerank_docs):
        return []
    
    try:
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
        for r in response.results:
            ranked.append({
                "text": rerank_docs[r.index],
                "metadata": docs[r.index]["metadata"],
                "score": r.relevance_score
            })
        
        _rerank_cache[cache_key] = ranked
        return ranked
        
    except Exception:
        return [{"text": doc["metadata"]["text"], "metadata": doc["metadata"], "score": 0.0} for doc in docs[:top_k]]

# ========== OPTIMIZED ANSWER GENERATION ==========
system_prompt = SystemMessage(content="You are an expert AI assistant that provides accurate, well-cited answers based on retrieved documents. Use inline citations [1], [2], [3] etc. when referencing information from documents.")

async def generate_answer_with_citations(query: str, reranked_docs: List[Dict]) -> Dict[str, Any]:
    start_time = time.time()
    
    if not reranked_docs:
        return {
            "answer": "I cannot answer this question as no relevant documents were found.",
            "citations": [],
            "sources": [],
            "metadata": {"processing_time": time.time() - start_time}
        }
    
    try:
        # Format documents
        formatted_docs = []
        for i, doc in enumerate(reranked_docs[:8]):
            text = doc.get("text", "")
            score = doc.get("score", 0.0)
            formatted_docs.append(f"[{i+1}] (Score: {score:.3f})\nText: {text[:250]}{'...' if len(text) > 250 else ''}")
        
        # Prepare sources
        sources = []
        for doc in reranked_docs[:8]:
            metadata = doc.get("metadata", {})
            sources.append({
                "source": metadata.get("source", "Unknown"),
                "title": metadata.get("title", "Untitled"),
                "relevance_score": round(doc.get("score", 0.0), 3),
            })
        
        # Generate answer - fixed f-string syntax
        doc_separator = "---\n"
        human_message_content = f"Question: {query}\n\nRetrieved Documents:\n{doc_separator.join(formatted_docs)}\n\nAnswer with citations:"
        human_message = HumanMessage(content=human_message_content)
        
        try:
            response = await asyncio.wait_for(llm.ainvoke([system_prompt, human_message]), timeout=15.0)
            answer = response.content
        except asyncio.TimeoutError:
            answer = "I apologize, but the response took too long to generate."
        
        # Extract citations
        matches = citation_pattern.findall(answer)
        unique_citations = sorted(set(matches), key=int)[:8]
        citations = []
        
        for cid in unique_citations:
            if 1 <= int(cid) <= len(reranked_docs):
                doc = reranked_docs[int(cid) - 1]
                metadata = doc.get("metadata", {})
                citations.append({
                    "citation_id": int(cid),
                    "marker": f"[{cid}]",
                    "source": metadata.get("source", "Unknown"),
                    "title": metadata.get("title", "Untitled"),
                    "relevance_score": round(doc.get("score", 0.0), 3),
                })
        
        return {
            "answer": answer,
            "citations": citations,
            "sources": sources,
            "metadata": {"processing_time": time.time() - start_time}
        }
        
    except Exception as e:
        return {
            "answer": f"I encountered an error: {str(e)[:100]}",
            "citations": [],
            "sources": [],
            "metadata": {"processing_time": time.time() - start_time}
        }

# ========== FASTAPI APP ==========
app = FastAPI(title="Ultra-Fast RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestRequest(BaseModel):
    text: str
    source: str = "user_input"
    title: str = "Untitled"

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K

@app.post("/ingest")
async def ingest(req: IngestRequest):
    n_chunks = await ingest_text_async(req.text, req.source, req.title)
    return {"status": "ok", "chunks_ingested": n_chunks}

@app.post("/query")
async def query_docs(req: QueryRequest):
    start_time = time.time()
    
    # Retrieve documents
    matches = await retrieve_async(req.query, top_k=req.top_k)
    if not matches:
        return {
            "status": "no_results",
            "answer": "No relevant documents found.",
            "citations": [],
            "sources": [],
            "metadata": {"processing_time": time.time() - start_time}
        }

    # Prepare for reranking
    docs = [{"metadata": match.metadata, "text": match.metadata.get("text", ""), "score": getattr(match, 'score', 0.0)} for match in matches]
    
    # Rerank
    reranked = await rerank_async(req.query, docs, top_k=min(req.top_k, 8))
    if not reranked:
        reranked = docs[:8]
    
    # Generate answer
    answer_result = await generate_answer_with_citations(req.query, reranked)
    
    return {
        "status": "success",
        **answer_result,
        "request_time": time.time() - start_time
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=multiprocessing.cpu_count())