# backend/app/services/rerank.py
import os
import cohere
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def rerank(query: str, docs: list, top_k: int = 5):
    # Handle empty documents list
    if not docs:
        return []
    
    # docs: list of dicts with { "text": str, "metadata": {...} }
    rerank_docs = [d["metadata"]["text"] for d in docs]
    
    # Handle case where all documents might be empty strings
    if not any(rerank_docs):
        return []
    
    try:
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=rerank_docs,
            top_n=top_k
        )
        
        ranked = []
        # Access the results from the response object
        for r in response.results:
            ranked.append({
                "text": rerank_docs[r.index],
                "metadata": docs[r.index]["metadata"],
                "score": r.relevance_score
            })
        return ranked
        
    except Exception as e:
        print(f"Rerank error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return original docs if reranking fails
        return [{"text": doc["metadata"]["text"], "metadata": doc["metadata"], "score": 0.0} for doc in docs[:top_k]]