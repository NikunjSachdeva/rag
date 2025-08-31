#!/usr/bin/env python3
"""
Performance Optimization Test Script
Tests the RAG system to ensure sub-10-second performance on first use
"""

import asyncio
import time
import json
from services.ingest import ingest_text_async
from services.retrieve import retrieve_async
from services.rerank import rerank_async
from services.answer import generate_answer_with_citations

# Test data
TEST_TEXT = """
Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly.

Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns. These neural networks are inspired by the human brain and can automatically learn representations from data such as images, text, or sound.

Natural language processing (NLP) is another important area of AI that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language in a way that is both meaningful and useful.

Computer vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects and then react to what they "see."

Reinforcement learning is a type of machine learning where an agent learns to behave in an environment by performing actions and seeing the results. The agent receives rewards or penalties for actions and learns to maximize the total reward over time.

Supervised learning involves training a model on a labeled dataset, where the correct answers are provided. The model learns to map inputs to outputs based on these examples. Common applications include image classification, spam detection, and medical diagnosis.

Unsupervised learning finds hidden patterns in data without labeled responses. The algorithm is given data and must find structure on its own. Clustering and dimensionality reduction are common unsupervised learning techniques.

Semi-supervised learning uses both labeled and unlabeled data for training. This approach is useful when labeled data is expensive or difficult to obtain, but unlabeled data is plentiful.

Transfer learning allows a model trained on one task to be applied to a related task. This is particularly useful when you have limited data for the target task but plenty of data for a related task.

The field of machine learning is constantly evolving with new algorithms, techniques, and applications being developed regularly. It has applications in virtually every industry, from healthcare and finance to transportation and entertainment.
"""

TEST_QUERIES = [
    "What is machine learning and how does it work?",
    "Explain the difference between supervised and unsupervised learning",
    "How does deep learning relate to machine learning?",
    "What are the main applications of natural language processing?",
    "Describe reinforcement learning and its applications"
]

async def test_ingestion_performance():
    """Test ingestion performance - should be under 10 seconds"""
    print("🚀 Testing Ingestion Performance...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        result = await ingest_text_async(
            text=TEST_TEXT,
            source="performance_test",
            title="Machine Learning Overview",
            embed_batch_size=16,
            upsert_batch_size=25
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"📊 Ingestion Results:")
        print(f"   Status: {result['status']}")
        print(f"   Chunks ingested: {result['chunks_ingested']}")
        print(f"   Total words: {result['total_words']}")
        print(f"   Processing time: {elapsed_time:.2f}s")
        
        if result['status'] == 'success':
            throughput = result['chunks_ingested'] / elapsed_time if elapsed_time > 0 else 0
            print(f"   Throughput: {throughput:.1f} chunks/second")
            
            # Performance check
            if elapsed_time <= 10.0:
                print("✅ SUCCESS: Ingestion completed in under 10 seconds!")
                return True
            else:
                print(f"❌ FAILED: Ingestion took {elapsed_time:.2f}s (target: ≤10s)")
                return False
        else:
            print(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Ingestion error: {e}")
        print(f"   Time taken: {elapsed_time:.2f}s")
        return False

async def test_query_performance():
    """Test query performance - should be under 10 seconds"""
    print("\n🔍 Testing Query Performance...")
    print("=" * 50)
    
    query = TEST_QUERIES[0]  # Use first query for testing
    print(f"Query: {query}")
    
    start_time = time.time()
    
    try:
        # Step 1: Retrieve
        print("   Step 1: Retrieving documents...")
        retrieve_start = time.time()
        matches = await retrieve_async(query, top_k=10)
        retrieve_time = time.time() - retrieve_start
        print(f"   Retrieved {len(matches)} documents in {retrieve_time:.2f}s")
        
        if not matches:
            print("❌ No documents retrieved")
            return False
        
        # Step 2: Prepare for reranking
        docs = []
        for match in matches:
            doc = {
                "metadata": match.metadata,
                "text": match.metadata.get("text", ""),
                "score": getattr(match, 'score', 0.0)
            }
            docs.append(doc)
        
        # Step 3: Rerank
        print("   Step 2: Reranking documents...")
        rerank_start = time.time()
        reranked = await rerank_async(query, docs, top_k=5)
        rerank_time = time.time() - rerank_start
        print(f"   Reranked {len(reranked)} documents in {rerank_time:.2f}s")
        
        # Step 4: Generate answer
        print("   Step 3: Generating answer...")
        answer_start = time.time()
        answer_result = await generate_answer_with_citations(query, reranked)
        answer_time = time.time() - answer_start
        print(f"   Generated answer in {answer_time:.2f}s")
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Query Performance Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Breakdown:")
        print(f"     - Retrieval: {retrieve_time:.2f}s")
        print(f"     - Reranking: {rerank_time:.2f}s")
        print(f"     - Answer generation: {answer_time:.2f}s")
        print(f"     - Other overhead: {total_time - retrieve_time - rerank_time - answer_time:.2f}s")
        
        # Performance check
        if total_time <= 10.0:
            print("✅ SUCCESS: Query completed in under 10 seconds!")
            return True
        else:
            print(f"❌ FAILED: Query took {total_time:.2f}s (target: ≤10s)")
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Query error: {e}")
        print(f"   Time taken: {total_time:.2f}s")
        return False

async def test_batch_operations():
    """Test batch operations for multiple queries"""
    print("\n🔄 Testing Batch Operations...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        results = []
        for i, query in enumerate(TEST_QUERIES[:3]):  # Test first 3 queries
            print(f"   Processing query {i+1}/3: {query[:50]}...")
            query_start = time.time()
            
            # Full pipeline for each query
            matches = await retrieve_async(query, top_k=5)
            if matches:
                docs = [{"metadata": m.metadata, "text": m.metadata.get("text", ""), "score": getattr(m, 'score', 0.0)} for m in matches]
                reranked = await rerank_async(query, docs, top_k=3)
                answer_result = await generate_answer_with_citations(query, reranked)
                
                query_time = time.time() - query_start
                results.append({
                    "query": query,
                    "time": query_time,
                    "status": "success"
                })
                print(f"     ✅ Completed in {query_time:.2f}s")
            else:
                query_time = time.time() - query_start
                results.append({
                    "query": query,
                    "time": query_time,
                    "status": "no_results"
                })
                print(f"     ⚠️ No results in {query_time:.2f}s")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(TEST_QUERIES[:3])
        
        print(f"\n📊 Batch Operation Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per query: {avg_time:.2f}s")
        print(f"   Queries processed: {len(results)}")
        
        # Performance check
        if avg_time <= 10.0:
            print("✅ SUCCESS: Average query time is under 10 seconds!")
            return True
        else:
            print(f"❌ FAILED: Average query time is {avg_time:.2f}s (target: ≤10s)")
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Batch operation error: {e}")
        print(f"   Time taken: {total_time:.2f}s")
        return False

async def main():
    """Run all performance tests"""
    print("🚀 RAG System Performance Optimization Test Suite")
    print("=" * 60)
    print("Target: Sub-10-second performance on first use")
    print("=" * 60)
    
    results = []
    
    # Test 1: Ingestion performance
    ingestion_success = await test_ingestion_performance()
    results.append(("Ingestion", ingestion_success))
    
    # Test 2: Query performance
    query_success = await test_query_performance()
    results.append(("Query", query_success))
    
    # Test 3: Batch operations
    batch_success = await test_batch_operations()
    results.append(("Batch Operations", batch_success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL PERFORMANCE TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System achieves sub-10-second performance!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    # Save results
    test_results = {
        "timestamp": time.time(),
        "target_performance": "sub-10-seconds",
        "results": {name: success for name, success in results},
        "summary": f"{passed}/{total} tests passed"
    }
    
    with open("performance_optimization_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to performance_optimization_results.json")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
