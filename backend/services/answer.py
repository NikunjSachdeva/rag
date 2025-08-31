import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

def generate_answer(query: str, reranked_docs: list):
    if not reranked_docs:
        return {
            "answer": "I couldn’t find relevant information to answer that.",
            "citations": []
        }

    # Build context with citations
    context_parts = []
    citations = []
    for i, doc in enumerate(reranked_docs):
        context_parts.append(f"[{i+1}] {doc['text']}")
        citations.append({
            "id": i+1,
            "source": doc["metadata"].get("source", "unknown"),
            "title": doc["metadata"].get("title", "Untitled"),
            "text": doc["text"]
        })

    context = "\n\n".join(context_parts)

    # Prompt Gemini
    prompt = f"""
You are a helpful assistant. Use the provided context to answer the question.
Always include inline citations in square brackets referring to the numbers.

Question: {query}

Context:
{context}

Answer:
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
    except Exception as e:
        print(f"LLM error: {e}")
        answer = "Sorry, I had an issue generating the answer."

    return {"answer": answer, "citations": citations}
