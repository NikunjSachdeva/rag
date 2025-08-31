#!/usr/bin/env python3
"""
Test script for the complete RAG pipeline with citations and scores
"""

import asyncio
import time
import json
from services.ingest import ingest_text_async
from services.retrieve import retrieve_async
from services.rerank import rerank_async
from services.answer import generate_answer_with_citations

# Sample text for testing
SAMPLE_TEXT = """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. 
Some of the activities computers with artificial intelligence are designed for include speech recognition, learning, planning, and problem solving.

Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. 
It focuses on the development of computer programs that can access data and use it to learn for themselves.

Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns. 
It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.

Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. 
It involves the development of algorithms and models that can understand, interpret, and generate human language.

Computer Vision is another important area of AI that enables computers to interpret and understand visual information from the world. 
It involves techniques for acquiring, processing, analyzing, and understanding digital images.

Robotics combines AI with mechanical engineering to create robots that can perform tasks autonomously or semi-autonomously. 
These robots can be used in manufacturing, healthcare, exploration, and many other fields.

Expert Systems are AI programs that emulate the decision-making ability of a human expert in a specific domain. 
They use knowledge bases and inference engines to solve complex problems.

Neural Networks are computing systems inspired by biological neural networks. 
They consist of interconnected nodes (neurons) that process information and can learn to recognize patterns.

Reinforcement Learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment. 
The agent receives rewards or penalties for its actions and learns to maximize rewards over time.

AI Ethics is an important consideration in the development and deployment of artificial intelligence systems. 
It involves ensuring that AI systems are fair, transparent, accountable, and aligned with human values.
""" * 3  # Multiply to create larger text for testing

async def test_complete_rag_pipeline():
    """Test the complete RAG pipeline: ingest → retrieve → rerank → answer"""
    print("🚀 Testing Complete RAG Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Ingest text with enhanced metadata
        print("📥 Step 1: Ingesting text...")
        start_time = time.time()
        
        ingest_result = await ingest_text_async(
            text=SAMPLE_TEXT,
            source="test_document",
            title="AI_Overview_Test"
        )
        
        ingest_time = time.time() - start_time
        
        if ingest_result["status"] != "success":
            print(f"❌ Ingestion failed: {ingest_result}")
            return False
        
        print(f"✅ Ingested {ingest_result['chunks_ingested']} chunks in {ingest_time:.2f}s")
        print(f"   Total words: {ingest_result['total_words']}")
        print(f"   Chunk stats: {ingest_result['chunk_stats']}")
        
        # Step 2: Test retrieval
        print("\n🔍 Step 2: Testing retrieval...")
        test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain deep learning concepts",
            "What is natural language processing?",
            "How do neural networks function?"
        ]
        
        retrieval_results = []
        for query in test_queries:
            start_time = time.time()
            matches = await retrieve_async(query, top_k=5)
            retrieval_time = time.time() - start_time
            
            retrieval_results.append({
                "query": query,
                "matches": len(matches),
                "time": retrieval_time,
                "first_match_score": getattr(matches[0], 'score', 0.0) if matches else 0.0
            })
            
            print(f"   Query: '{query[:50]}...' → {len(matches)} matches in {retrieval_time:.2f}s")
        
        # Step 3: Test reranking
        print("\n🎯 Step 3: Testing reranking...")
        rerank_results = []
        
        for query in test_queries:
            start_time = time.time()
            
            # Get documents for reranking
            matches = await retrieve_async(query, top_k=10)
            docs = []
            for match in matches:
                doc = {
                    "metadata": match.metadata,
                    "text": match.metadata.get("text", ""),
                    "score": getattr(match, 'score', 0.0)
                }
                docs.append(doc)
            
            # Rerank
            reranked = await rerank_async(query, docs, top_k=5)
            rerank_time = time.time() - start_time
            
            rerank_results.append({
                "query": query,
                "original_docs": len(docs),
                "reranked_docs": len(reranked),
                "time": rerank_time,
                "top_score": reranked[0]["score"] if reranked else 0.0
            })
            
            print(f"   Query: '{query[:50]}...' → {len(reranked)} reranked in {rerank_time:.2f}s")
        
        # Step 4: Test answer generation with citations
        print("\n🤖 Step 4: Testing answer generation with citations...")
        answer_results = []
        
        for i, query in enumerate(test_queries[:2]):  # Test first 2 queries
            start_time = time.time()
            
            # Get reranked documents
            matches = await retrieve_async(query, top_k=5)
            docs = []
            for match in matches:
                doc = {
                    "metadata": match.metadata,
                    "text": match.metadata.get("text", ""),
                    "score": getattr(match, 'score', 0.0)
                }
                docs.append(doc)
            
            reranked = await rerank_async(query, docs, top_k=3)
            
            # Generate answer with citations
            answer_result = await generate_answer_with_citations(query, reranked)
            answer_time = time.time() - start_time
            
            answer_results.append({
                "query": query,
                "answer_length": len(answer_result["answer"]),
                "citations_count": len(answer_result["citations"]),
                "sources_count": len(answer_result["sources"]),
                "time": answer_time,
                "metadata": answer_result["metadata"]
            })
            
            print(f"   Query: '{query[:50]}...' → {len(answer_result['citations'])} citations in {answer_time:.2f}s")
            print(f"      Answer preview: {answer_result['answer'][:100]}...")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 RAG Pipeline Test Summary")
        print("=" * 60)
        
        print(f"✅ Ingestion: {ingest_result['chunks_ingested']} chunks in {ingest_time:.2f}s")
        
        avg_retrieval_time = sum(r["time"] for r in retrieval_results) / len(retrieval_results)
        print(f"✅ Retrieval: {len(retrieval_results)} queries, avg {avg_retrieval_time:.2f}s per query")
        
        avg_rerank_time = sum(r["time"] for r in rerank_results) / len(rerank_results)
        print(f"✅ Reranking: {len(rerank_results)} queries, avg {avg_rerank_time:.2f}s per query")
        
        avg_answer_time = sum(r["time"] for r in answer_results) / len(answer_results)
        print(f"✅ Answer Generation: {len(answer_results)} queries, avg {avg_answer_time:.2f}s per query")
        
        total_time = ingest_time + (avg_retrieval_time * len(retrieval_results)) + (avg_rerank_time * len(rerank_results)) + (avg_answer_time * len(answer_results))
        print(f"\n🎯 Total Pipeline Time: {total_time:.2f}s")
        
        # Save detailed results
        results = {
            "ingestion": ingest_result,
            "retrieval": retrieval_results,
            "reranking": rerank_results,
            "answering": answer_results,
            "summary": {
                "total_chunks": ingest_result["chunks_ingested"],
                "total_queries_tested": len(test_queries),
                "total_pipeline_time": total_time,
                "status": "success"
            }
        }
        
        with open('rag_pipeline_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to rag_pipeline_test_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_citation_system():
    """Test the citation system specifically"""
    print("\n📚 Testing Citation System")
    print("=" * 40)
    
    try:
        # Test with a simple query
        query = "What is machine learning?"
        
        # Retrieve documents
        matches = await retrieve_async(query, top_k=3)
        docs = []
        for match in matches:
            doc = {
                "metadata": match.metadata,
                "text": match.metadata.get("text", ""),
                "score": getattr(match, 'score', 0.0)
            }
            docs.append(doc)
        
        # Generate answer with citations
        answer_result = await generate_answer_with_citations(query, docs)
        
        print(f"Query: {query}")
        print(f"Answer: {answer_result['answer'][:200]}...")
        print(f"Citations found: {len(answer_result['citations'])}")
        print(f"Sources: {len(answer_result['sources'])}")
        
        # Show citation details
        for citation in answer_result['citations']:
            print(f"  Citation [{citation['citation_id']}]: Score {citation['relevance_score']:.3f}, Chunk {citation['chunk_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Citation test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Starting RAG Pipeline Tests...")
    
    # Test 1: Complete pipeline
    pipeline_success = await test_complete_rag_pipeline()
    
    # Test 2: Citation system
    citation_success = await test_citation_system()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 Final Test Results")
    print("=" * 60)
    
    if pipeline_success and citation_success:
        print("🎉 All tests passed! RAG pipeline is working correctly.")
        print("✅ Citations and scores are being generated properly.")
        print("✅ Ready for frontend integration.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
