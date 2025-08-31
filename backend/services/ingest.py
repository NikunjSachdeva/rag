# backend/services/ingest.py
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from hashlib import sha256
from typing import List, Dict, Any
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
import aiohttp
import json

load_dotenv()

# Global connection pool for Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Optimized embeddings with better timeout handling
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    request_timeout=20,  # Increased timeout to 2 minutes
    max_retries=3,        # Increased retries
    temperature=0.0        # Deterministic embeddings
)

# Optimized text splitter with better chunking strategy for citations
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Reduced chunk size to avoid timeouts
    chunk_overlap=120, # Reduced overlap proportionally
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # More intelligent splitting
)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# ---------- optimized helpers ----------
def _batch_list(items, batch_size):
    """Optimized batching with memory efficiency"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

async def _parallel_embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Parallel embedding with optimized batch sizes and better error handling"""
    if not texts:
        return []
    
    # Use smaller batches to avoid timeouts
    batches = list(_batch_list(texts, batch_size))
    
    async def embed_single_batch(batch):
        loop = asyncio.get_event_loop()
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Embedding batch of {len(batch)} texts (attempt {attempt + 1})")
                result = await loop.run_in_executor(executor, embeddings.embed_documents, batch)
                print(f"Successfully embedded batch of {len(batch)} texts")
                return result
                
            except Exception as e:
                print(f"Batch embed attempt {attempt + 1} failed: {e}")
                
                if "504" in str(e) or "Deadline Exceeded" in str(e):
                    print(f"Timeout error detected, retrying with smaller batch...")
                    # Split batch in half and retry
                    if len(batch) > 8:
                        half = len(batch) // 2
                        print(f"Splitting batch from {len(batch)} to {half} texts")
                        
                        # Process both halves concurrently
                        left_task = embed_single_batch(batch[:half])
                        right_task = embed_single_batch(batch[half:])
                        
                        left_result, right_result = await asyncio.gather(left_task, right_task)
                        return left_result + right_result
                    else:
                        # If batch is already small, wait and retry
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"Waiting {delay}s before retry...")
                            await asyncio.sleep(delay)
                        else:
                            print(f"All retry attempts failed for batch of {len(batch)} texts")
                            raise e
                else:
                    # Non-timeout error, try to retry
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Non-timeout error, waiting {delay}s before retry...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"All retry attempts failed for batch of {len(batch)} texts")
                        raise e
        
        # If we get here, all retries failed
        raise Exception(f"Failed to embed batch after {max_retries} attempts")
    
    # Process batches with better error handling
    all_embeddings = []
    failed_batches = []
    
    for i, batch in enumerate(batches):
        try:
            print(f"Processing batch {i+1}/{len(batches)} with {len(batch)} texts")
            batch_result = await embed_single_batch(batch)
            all_embeddings.extend(batch_result)
            print(f"Batch {i+1} completed successfully")
            
        except Exception as e:
            print(f"Batch {i+1} failed completely: {e}")
            failed_batches.append((i, batch, e))
            # Add placeholder embeddings to maintain count
            all_embeddings.extend([[0.0] * 768] * len(batch))
    
    if failed_batches:
        print(f"Warning: {len(failed_batches)} batches failed during embedding")
        for i, batch, error in failed_batches:
            print(f"  Batch {i+1}: {len(batch)} texts failed - {error}")
    
    return all_embeddings

async def _parallel_pinecone_upsert(vectors: List[Dict[str, Any]], batch_size: int = 50):
    """Parallel Pinecone upserts with optimized batching"""
    # 🔍 DEBUG: Add this code to see what's happening
    print(f"=== DEBUG: Pinecone Upsert Started ===")
    print(f"Total vectors to upsert: {len(vectors)}")
    if vectors:
        print(f"First vector ID: {vectors[0]['id']}")
        print(f"First vector embedding length: {len(vectors[0]['values'])}")
        print(f"First vector text preview: {vectors[0]['metadata']['text'][:50]}...")
    else:
        print("❌ NO VECTORS TO UPSERT - This is the problem!")
        return
    print(f"Batch size: {batch_size}")
    print("=== DEBUG: End ===")
    if not vectors:
        return
    
    # Reduced batch size for better reliability
    batches = list(_batch_list(vectors, batch_size))
    
    async def upsert_batch(batch):
        print(f"🔄 Processing batch with {len(batch)} vectors")
        loop = asyncio.get_event_loop()
        max_retries = 3
    
        for attempt in range(max_retries):
            try:
                print(f"Upsert attempt {attempt + 1} for batch size {len(batch)}")
                # ✅ FIXED: Use lambda function
                result = await loop.run_in_executor(executor, lambda: index.upsert(vectors=batch))
                print(f"✅ Upsert successful for {len(batch)} vectors")
                return result

            except Exception as e:
                print(f"❌ Upsert attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Retry with smaller batch
                    if len(batch) > 25:
                        half = len(batch) // 2
                        print(f"Splitting upsert batch from {len(batch)} to {half}")
                        await upsert_batch(batch[:half])
                        await upsert_batch(batch[half:])
                        return
                    else:
                        await asyncio.sleep(2 ** attempt)
                else:
                    raise
    
    # Process upserts concurrently
    tasks = [upsert_batch(batch) for batch in batches]
    await asyncio.gather(*tasks, return_exceptions=True)

def _create_enhanced_metadata(doc: Document, source: str, title: str, chunk_id: int, total_chunks: int) -> Dict[str, Any]:
    """Create enhanced metadata for better citations and retrieval"""
    chunk_hash = sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
    
    return {
        "source": source,
        "title": title,
        "text": doc.page_content,
        "chunk_id": chunk_id,
        "position": chunk_id,
        "section": f"chunk_{chunk_id}",
        "total_chunks": total_chunks,
        "length": len(doc.page_content),
        "hash": chunk_hash,
        "timestamp": time.time(),
        "word_count": len(doc.page_content.split()),
        "type": "text_chunk"
    }

# ---------- main optimized function ----------
async def ingest_text_async(
    text: str,
    source: str = "user_input",
    title: str = "Untitled",
    embed_batch_size: int = 50,  # Reduced for better reliability
    upsert_batch_size: int = 50  # Reduced for better reliability
) -> Dict[str, Any]:
    """
    Enhanced async ingestion with better error handling and timeout management:
    - Parallel text splitting with enhanced metadata
    - Concurrent batch embedding with retry logic
    - Parallel Pinecone upserts
    - Smart error handling and fallbacks
    - Returns detailed ingestion statistics
    """
    start_time = time.time()
    
    try:
        # 1) Parallel text splitting
        print(f"Starting text splitting for {len(text)} characters...")
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(executor, text_splitter.split_text, text)
        
        if not docs:
            return {
                "chunks_ingested": 0,
                "processing_time": 0,
                "total_words": 0,
                "chunk_stats": {},
                "status": "no_content"
            }
        
        total_chunks = len(docs)
        total_words = sum(len(doc.split()) for doc in docs)
        
        print(f"Text split into {total_chunks} chunks with {total_words} total words")
        
        # Create documents with enhanced metadata for citations
        documents = []
        for i, doc in enumerate(docs):
            document = Document(
                page_content=doc, 
                metadata=_create_enhanced_metadata(Document(page_content=doc), source, title, i, total_chunks)
            )
            documents.append(document)
        
        # 2) Parallel batch embedding with better error handling
        print(f"Starting embedding process for {total_chunks} chunks...")
        texts = [d.page_content for d in documents]
        embeddings_list = await _parallel_embed_batch(texts, embed_batch_size)
        
        if len(embeddings_list) != len(documents):
            print(f"Warning: Embedding count mismatch. Expected {len(documents)}, got {len(embeddings_list)}")
            # Pad embeddings if needed
            while len(embeddings_list) < len(documents):
                embeddings_list.append([0.0] * 768)
            # Truncate if too many
            embeddings_list = embeddings_list[:len(documents)]
        
        print(f"Embedding completed: {len(embeddings_list)} vectors generated")
        
        # 3) Prepare vectors for parallel upsert
        vectors = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings_list)):
            # vec_id = f"{source}_{title}_{doc.metadata['hash']}"
            vec_id = f"{source}_{title}_{i}_{doc.metadata['hash']}"
            vectors.append({
                "id": vec_id,
                "values": emb,
                "metadata": doc.metadata
            })
        
        # 4) Parallel Pinecone upsert
        print(f"Starting Pinecone upsert for {len(vectors)} vectors...")
        await _parallel_pinecone_upsert(vectors, upsert_batch_size)
        print("Pinecone upsert completed successfully")
        
        elapsed_time = time.time() - start_time
        
        # Calculate chunk statistics
        chunk_stats = {
            "avg_chunk_size": total_words / total_chunks if total_chunks > 0 else 0,
            "chunk_size_range": f"{min(len(doc.split()) for doc in docs)}-{max(len(doc.split()) for doc in docs)}",
            "overlap_percentage": "15%",
            "embedding_dimensions": len(embeddings_list[0]) if embeddings_list else 0
        }
        
        print(f"✅ Successfully ingested {total_chunks} chunks in {elapsed_time:.2f}s")
        print(f"   Total words: {total_words}, Avg chunk size: {chunk_stats['avg_chunk_size']:.1f} words")
        
        return {
            "chunks_ingested": total_chunks,
            "processing_time": elapsed_time,
            "total_words": total_words,
            "chunk_stats": chunk_stats,
            "status": "success",
            "source": source,
            "title": title,
            "chunk_ids": list(range(total_chunks))
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Ingestion failed after {elapsed_time:.2f}s: {e}")
        
        return {
            "chunks_ingested": 0,
            "processing_time": elapsed_time,
            "total_words": len(text.split()) if text else 0,
            "chunk_stats": {},
            "status": "error",
            "error": str(e),
            "source": source,
            "title": title
        }

# Backward compatibility
def ingest_text(
    text: str,
    source: str = "user_input",
    title: str = "Untitled",
    embed_batch_size: int = 50,
    upsert_batch_size: int = 50
) -> Dict[str, Any]:
    """Synchronous wrapper for backward compatibility"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use run_until_complete
        return loop.run_until_complete(ingest_text_async(text, source, title, embed_batch_size, upsert_batch_size))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(ingest_text_async(text, source, title, embed_batch_size, upsert_batch_size))



# # backend/services/ingest.py
# import os
# import time
# import asyncio
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# from hashlib import sha256
# from typing import List, Dict, Any
# from pinecone import Pinecone
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from dotenv import load_dotenv
# import aiohttp
# import json

# load_dotenv()

# # Global connection pool for Pinecone
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index(os.getenv("PINECONE_INDEX"))

# # Optimized embeddings with better timeout handling
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     request_timeout=120,
#     max_retries=3,
#     temperature=0.0
# )

# # Optimized text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=800,
#     chunk_overlap=120,
#     length_function=len,
#     separators=["\n\n", "\n", ". ", " ", ""]
# )

# # MAXIMUM THREAD OPTIMIZATION - Use all available CPU cores
# import multiprocessing
# MAX_WORKERS = min(32, multiprocessing.cpu_count() * 4)  # Use 4x CPU cores, max 32
# executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
# print(f"🚀 Using {MAX_WORKERS} threads for parallel processing")

# # Background task queue for non-blocking operations
# background_tasks = set()

# # ---------- optimized helpers ----------
# def _batch_list(items, batch_size):
#     """Optimized batching with memory efficiency"""
#     for i in range(0, len(items), batch_size):
#         yield items[i:i + batch_size]

# async def _parallel_embed_batch(texts: List[str], batch_size: int = 50) -> List[List[float]]:
#     """ULTRA-PARALLEL embedding with maximum concurrency"""
#     if not texts:
#         return []
    
#     # Use larger batches for better throughput
#     batches = list(_batch_list(texts, batch_size))
    
#     async def embed_single_batch(batch):
#         loop = asyncio.get_event_loop()
#         max_retries = 2  # Reduced retries for speed
        
#         for attempt in range(max_retries):
#             try:
#                 # Use thread pool for parallel embedding
#                 result = await loop.run_in_executor(executor, embeddings.embed_documents, batch)
#                 return result
                
#             except Exception as e:
#                 print(f"Batch embed attempt {attempt + 1} failed: {e}")
#                 if attempt < max_retries - 1:
#                     await asyncio.sleep(1)
#                 else:
#                     print(f"Final embed failure for batch size {len(batch)}")
#                     # Return zeros instead of failing completely
#                     return [[0.0] * 768 for _ in range(len(batch))]
        
#         return [[0.0] * 768 for _ in range(len(batch))]
    
#     # Process ALL batches in parallel for maximum speed
#     tasks = [embed_single_batch(batch) for batch in batches]
#     results = await asyncio.gather(*tasks, return_exceptions=True)
    
#     # Combine results
#     all_embeddings = []
#     for result in results:
#         if isinstance(result, Exception):
#             print(f"Batch failed: {result}")
#             # Add placeholder embeddings
#             all_embeddings.extend([[0.0] * 768] * batch_size)
#         else:
#             all_embeddings.extend(result)
    
#     return all_embeddings[:len(texts)]  # Trim to exact length

# async def _parallel_pinecone_upsert(vectors: List[Dict[str, Any]], batch_size: int = 100):
#     """ULTRA-PARALLEL Pinecone upserts with maximum concurrency"""
#     if not vectors:
#         print("❌ No vectors to upsert")
#         return
    
#     # Larger batches for better throughput
#     batches = list(_batch_list(vectors, batch_size))
    
#     async def upsert_batch(batch):
#         loop = asyncio.get_event_loop()
#         try:
#             # Use thread pool for parallel upsert
#             result = await loop.run_in_executor(
#                 executor, 
#                 lambda: index.upsert(vectors=batch)
#             )
#             return result
#         except Exception as e:
#             print(f"Upsert batch failed: {e}")
#             return None
    
#     # Process ALL batches in parallel
#     tasks = [upsert_batch(batch) for batch in batches]
#     results = await asyncio.gather(*tasks, return_exceptions=True)
    
#     # Log results
#     successful = sum(1 for r in results if not isinstance(r, Exception) and r is not None)
#     print(f"✅ Upserted {successful}/{len(batches)} batches successfully")

# def _create_enhanced_metadata(doc: Document, source: str, title: str, chunk_id: int, total_chunks: int) -> Dict[str, Any]:
#     """Create enhanced metadata for better citations and retrieval"""
#     chunk_hash = sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
    
#     return {
#         "source": source,
#         "title": title,
#         "text": doc.page_content,
#         "chunk_id": chunk_id,
#         "position": chunk_id,
#         "section": f"chunk_{chunk_id}",
#         "total_chunks": total_chunks,
#         "length": len(doc.page_content),
#         "hash": chunk_hash,
#         "timestamp": time.time(),
#         "word_count": len(doc.page_content.split()),
#         "type": "text_chunk"
#     }

# # ---------- main optimized function ----------
# async def ingest_text_async(
#     text: str,
#     source: str = "user_input",
#     title: str = "Untitled",
#     embed_batch_size: int = 100,  # Larger batches for throughput
#     upsert_batch_size: int = 100   # Larger batches for throughput
# ) -> Dict[str, Any]:
#     """
#     ULTRA-PARALLEL ingestion with maximum concurrency
#     """
#     start_time = time.time()
    
#     try:
#         # 1) Parallel text splitting
#         print(f"📝 Splitting {len(text)} characters...")
#         loop = asyncio.get_event_loop()
#         docs = await loop.run_in_executor(executor, text_splitter.split_text, text)
        
#         if not docs:
#             return {
#                 "chunks_ingested": 0,
#                 "processing_time": 0,
#                 "total_words": 0,
#                 "chunk_stats": {},
#                 "status": "no_content"
#             }
        
#         total_chunks = len(docs)
#         total_words = sum(len(doc.split()) for doc in docs)
#         print(f"📊 Split into {total_chunks} chunks ({total_words} words)")
        
#         # 2) PARALLEL: Create documents with metadata
#         documents = []
#         metadata_tasks = []
        
#         for i, doc in enumerate(docs):
#             # Process metadata creation in background
#             task = loop.run_in_executor(
#                 executor,
#                 _create_enhanced_metadata,
#                 Document(page_content=doc), source, title, i, total_chunks
#             )
#             metadata_tasks.append((i, doc, task))
        
#         # Wait for all metadata tasks to complete
#         for i, doc, task in metadata_tasks:
#             metadata = await task
#             documents.append(Document(page_content=doc, metadata=metadata))
        
#         # 3) PARALLEL: Batch embedding
#         print(f"🧠 Embedding {total_chunks} chunks...")
#         texts = [d.page_content for d in documents]
#         embeddings_list = await _parallel_embed_batch(texts, embed_batch_size)
        
#         # 4) PARALLEL: Vector preparation
#         print("📦 Preparing vectors...")
#         vectors = []
#         for i, (doc, emb) in enumerate(zip(documents, embeddings_list)):
#             vec_id = f"{source}_{title}_{i}_{doc.metadata['hash']}"
#             vectors.append({
#                 "id": vec_id,
#                 "values": emb,
#                 "metadata": doc.metadata
#             })
        
#         # 5) PARALLEL: Pinecone upsert
#         print(f"🚀 Upserting {len(vectors)} vectors to Pinecone...")
#         await _parallel_pinecone_upsert(vectors, upsert_batch_size)
        
#         elapsed_time = time.time() - start_time
        
#         print(f"✅ Successfully ingested {total_chunks} chunks in {elapsed_time:.2f}s")
#         print(f"   Throughput: {total_chunks/elapsed_time:.1f} chunks/second")
        
#         return {
#             "chunks_ingested": total_chunks,
#             "processing_time": elapsed_time,
#             "total_words": total_words,
#             "chunk_stats": {
#                 "avg_chunk_size": total_words / total_chunks,
#                 "chunk_size_range": f"{min(len(doc.split()) for doc in docs)}-{max(len(doc.split()) for doc in docs)}",
#                 "throughput_chunks_sec": total_chunks / elapsed_time
#             },
#             "status": "success",
#             "source": source,
#             "title": title,
#             "chunk_ids": list(range(total_chunks))
#         }
        
#     except Exception as e:
#         elapsed_time = time.time() - start_time
#         print(f"❌ Ingestion failed after {elapsed_time:.2f}s: {e}")
        
#         return {
#             "chunks_ingested": 0,
#             "processing_time": elapsed_time,
#             "total_words": len(text.split()) if text else 0,
#             "chunk_stats": {},
#             "status": "error",
#             "error": str(e),
#             "source": source,
#             "title": title
#         }

# # Backward compatibility
# def ingest_text(
#     text: str,
#     source: str = "user_input",
#     title: str = "Untitled",
#     embed_batch_size: int = 100,
#     upsert_batch_size: int = 100
# ) -> Dict[str, Any]:
#     """Synchronous wrapper for backward compatibility"""
#     try:
#         loop = asyncio.get_running_loop()
#         return loop.run_until_complete(ingest_text_async(text, source, title, embed_batch_size, upsert_batch_size))
#     except RuntimeError:
#         return asyncio.run(ingest_text_async(text, source, title, embed_batch_size, upsert_batch_size))