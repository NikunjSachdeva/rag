# backend/services/answer.py
import os
import time
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import json

load_dotenv()

class AnswerService:
    """Service for generating answers with citations using Google Gemini LLM"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,  # Low temperature for consistent, factual answers
            max_output_tokens=2048,
            request_timeout=60
        )
        
        # System prompt for citation-aware answering
        self.system_prompt = """You are an expert AI assistant that provides accurate, well-cited answers based on retrieved documents.

IMPORTANT RULES:
1. ALWAYS use inline citations [1], [2], [3] etc. when referencing information from documents
2. ONLY use information from the provided documents - do not add external knowledge
3. If you cannot answer the question from the documents, say "I cannot answer this question based on the available information"
4. Be concise but comprehensive
5. Format citations as [1], [2], [3] etc. in the text
6. Always end your answer with a summary of the key points

Document format: Each document has a chunk_id, text content, and relevance score."""

        # Human prompt template
        self.human_prompt = """Question: {query}

Retrieved Documents:
{documents}

Please provide a comprehensive answer with inline citations [1], [2], [3] etc. that reference the specific documents above. Include the relevance scores in your analysis.

Answer:"""

    def _format_documents_for_citation(self, reranked_docs: List[Dict]) -> str:
        """Format documents for the LLM with citation information"""
        formatted_docs = []
        
        for i, doc in enumerate(reranked_docs):
            # Extract metadata
            metadata = doc.get("metadata", {})
            text = doc.get("text", "")
            score = doc.get("score", 0.0)
            
            # Format each document
            doc_text = f"[{i+1}] (Score: {score:.3f}, Chunk: {metadata.get('chunk_id', 'N/A')})\n"
            doc_text += f"Source: {metadata.get('source', 'Unknown')}\n"
            doc_text += f"Title: {metadata.get('title', 'Untitled')}\n"
            doc_text += f"Text: {text[:500]}{'...' if len(text) > 500 else ''}\n"
            
            formatted_docs.append(doc_text)
        
        return "\n---\n".join(formatted_docs)

    def _extract_citations_from_answer(self, answer: str) -> List[Dict]:
        """Extract citation information from the LLM answer"""
        citations = []
        
        # Find all citation markers [1], [2], etc.
        import re
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, answer)
        
        # Remove duplicates and sort
        unique_citations = sorted(list(set(matches)), key=int)
        
        return [{"citation_id": int(cid), "marker": f"[{cid}]"} for cid in unique_citations]

    def _map_citations_to_sources(self, citations: List[Dict], reranked_docs: List[Dict]) -> List[Dict]:
        """Map citation markers to actual source documents"""
        mapped_citations = []
        
        for citation in citations:
            citation_id = citation["citation_id"]
            
            # Map citation ID to document index (citation_id - 1 for 0-based indexing)
            if 1 <= citation_id <= len(reranked_docs):
                doc = reranked_docs[citation_id - 1]
                metadata = doc.get("metadata", {})
                
                mapped_citation = {
                    **citation,
                    "source": metadata.get("source", "Unknown"),
                    "title": metadata.get("title", "Untitled"),
                    "chunk_id": metadata.get("chunk_id", "N/A"),
                    "relevance_score": doc.get("score", 0.0),
                    "text_snippet": doc.get("text", "")[:200] + "..." if len(doc.get("text", "")) > 200 else doc.get("text", ""),
                    "word_count": metadata.get("word_count", 0),
                    "position": metadata.get("position", "N/A")
                }
                
                mapped_citations.append(mapped_citation)
        
        return mapped_citations

    async def generate_answer_with_citations(
        self, 
        query: str, 
        reranked_docs: List[Dict],
        include_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive answer with citations and scores
        
        Args:
            query: User's question
            reranked_docs: List of reranked documents with scores
            include_scores: Whether to include relevance scores in the answer
            
        Returns:
            Dictionary containing answer, citations, sources, and metadata
        """
        start_time = time.time()
        
        if not reranked_docs:
            return {
                "answer": "I cannot answer this question as no relevant documents were found.",
                "citations": [],
                "sources": [],
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "documents_used": 0,
                    "status": "no_documents"
                }
            }
        
        try:
            # Format documents for the LLM
            formatted_docs = self._format_documents_for_citation(reranked_docs)
            
            # Create the prompt
            prompt = self.human_prompt.format(
                query=query,
                documents=formatted_docs
            )
            
            # Generate answer using Gemini
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            answer = response.content
            
            # Extract citations
            citations = self._extract_citations_from_answer(answer)
            
            # Map citations to sources
            mapped_citations = self._map_citations_to_sources(citations, reranked_docs)
            
            # Prepare sources list
            sources = []
            for doc in reranked_docs:
                metadata = doc.get("metadata", {})
                source_info = {
                    "source": metadata.get("source", "Unknown"),
                    "title": metadata.get("title", "Untitled"),
                    "chunk_id": metadata.get("chunk_id", "N/A"),
                    "relevance_score": doc.get("score", 0.0),
                    "text": doc.get("text", ""),
                    "word_count": metadata.get("word_count", 0),
                    "position": metadata.get("position", "N/A"),
                    "timestamp": metadata.get("timestamp", "N/A")
                }
                sources.append(source_info)
            
            # Calculate statistics
            processing_time = time.time() - start_time
            avg_score = sum(doc.get("score", 0.0) for doc in reranked_docs) / len(reranked_docs) if reranked_docs else 0.0
            
            metadata = {
                "processing_time": processing_time,
                "documents_used": len(reranked_docs),
                "citations_found": len(citations),
                "average_relevance_score": avg_score,
                "query_length": len(query),
                "answer_length": len(answer),
                "status": "success"
            }
            
            return {
                "answer": answer,
                "citations": mapped_citations,
                "sources": sources,
                "metadata": metadata,
                "query": query
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error generating answer: {e}")
            
            return {
                "answer": f"I encountered an error while generating the answer: {str(e)}",
                "citations": [],
                "sources": [],
                "metadata": {
                    "processing_time": processing_time,
                    "documents_used": len(reranked_docs),
                    "error": str(e),
                    "status": "error"
                },
                "query": query
            }

    def generate_answer_with_citations_sync(
        self, 
        query: str, 
        reranked_docs: List[Dict],
        include_scores: bool = True
    ) -> Dict[str, Any]:
        """Synchronous wrapper for backward compatibility"""
        import asyncio
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop, use run_until_complete
            return loop.run_until_complete(
                self.generate_answer_with_citations(query, reranked_docs, include_scores)
            )
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(
                self.generate_answer_with_citations(query, reranked_docs, include_scores)
            )

# Global instance for easy access
answer_service = AnswerService()

# Convenience functions
async def generate_answer_with_citations(query: str, reranked_docs: List[Dict]) -> Dict[str, Any]:
    """Generate answer with citations using the global answer service"""
    return await answer_service.generate_answer_with_citations(query, reranked_docs)

def generate_answer_with_citations_sync(query: str, reranked_docs: List[Dict]) -> Dict[str, Any]:
    """Synchronous version of generate_answer_with_citations"""
    return answer_service.generate_answer_with_citations_sync(query, reranked_docs)
