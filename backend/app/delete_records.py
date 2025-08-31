import os
from pinecone import Pinecone

# Fetch from environment
api_key = "api"

if not api_key:
    raise ValueError("❌ PINECONE_API_KEY not set in environment variables")

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Example: connect to your index
index = pc.Index("mini-rag-docs")

# Delete all vectors
index.delete(delete_all=True)

print("✅ All records deleted")
