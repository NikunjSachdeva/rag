# backend/app/services/retrieve.py
import os
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


import time


def safe_embed_query(query, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return embeddings.embed_query(query)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

def retrieve(query: str, top_k: int = 10):
    if len(query) > 2000:  # safeguard for Gemini
        query = query[:2000]
    query_emb = safe_embed_query(query)
    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    return results.matches
