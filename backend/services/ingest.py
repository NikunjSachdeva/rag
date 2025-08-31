# backend/services/ingest.py
import os
import time
from hashlib import sha256
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Adjust chunk size for balance (we’ll fine-tune later if needed)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,   # smaller chunks = faster embeddings & better retrieval
    chunk_overlap=100
)


# ---------- helpers ----------
def _batch_list(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def safe_batch_embed(texts: list[str], retries: int = 3, delay: int = 2):
    """Embed a list of texts in one call, retrying on failure"""
    for attempt in range(retries):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            print(f"Batch embed attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


# ---------- main ----------
def ingest_text(
    text: str,
    source: str = "user_input",
    title: str = "Untitled",
    embed_batch_size: int = 32,
    upsert_batch_size: int = 64
):
    """
    Optimized ingestion:
    - split text into chunks
    - embed in batches
    - upsert in batches to Pinecone
    """
    # 1) split into chunks
    docs = text_splitter.split_text(text)
    documents = [
        Document(page_content=doc, metadata={"source": source, "title": title, "text": doc})
        for doc in docs
    ]
    if not documents:
        return 0

    vectors_buffer = []

    # 2) batch embedding
    for batch in _batch_list(documents, embed_batch_size):
        texts = [d.page_content for d in batch]
        embs = safe_batch_embed(texts)

        for i, emb in enumerate(embs):
            d = batch[i]
            # create stable ID using hash of text
            chunk_hash = sha256(d.page_content.encode("utf-8")).hexdigest()
            vec_id = f"{source}_{title}_{chunk_hash}"
            vectors_buffer.append({
                "id": vec_id,
                "values": emb,
                "metadata": d.metadata
            })

        # 3) upsert in Pinecone in batches
        while len(vectors_buffer) >= upsert_batch_size:
            to_upsert = vectors_buffer[:upsert_batch_size]
            index.upsert(vectors=to_upsert)
            vectors_buffer = vectors_buffer[upsert_batch_size:]

    # flush remaining
    if vectors_buffer:
        index.upsert(vectors=vectors_buffer)

    return len(documents)
