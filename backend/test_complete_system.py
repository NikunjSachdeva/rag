#!/usr/bin/env python3
"""
Complete RAG System Test Script
Tests the entire pipeline: ingest → retrieve → rerank → answer with citations
"""

import asyncio
import time
import json
import requests
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TEXT = """
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
""" * 2  # Multiply to create larger text

TEST_QUERIES = [
    "What is artificial intelligence?",
    "How does machine learning work?",
    "Explain deep learning concepts",
    "What is natural language processing?",
    "How do neural networks function?"
]

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Features: {', '.join(data['features'])}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_ingestion():
    """Test the ingestion endpoint"""
    print("\n📥 Testing Ingestion...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json={
                "text": TEST_TEXT,
                "source": "test_document",
                "title": "AI_Overview_Test"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                result = data['data']
                print(f"✅ Ingestion successful:")
                print(f"   Chunks: {result['chunks_ingested']}")
                print(f"   Words: {result['total_words']}")
                print(f"   Time: {result['processing_time']:.2f}s")
                print(f"   Status: {result['status']}")
                return True
            else:
                print(f"❌ Ingestion failed: {data['message']}")
                return False
        else:
            print(f"❌ Ingestion request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        return False

def test_query(query: str, top_k: int = 5):
    """Test a single query"""
    print(f"\n🔍 Testing Query: '{query[:50]}...'")
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": query,
                "top_k": top_k,
                "include_scores": True
            }
        )
        
        query_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                print(f"✅ Query successful in {query_time:.2f}s:")
                print(f"   Answer length: {len(data['answer'])} chars")
                print(f"   Citations: {len(data['citations'])}")
                print(f"   Sources: {len(data['sources'])}")
                print(f"   Processing time: {data['metadata']['total_processing_time']:.2f}s")
                
                # Show first few citations
                if data['citations']:
                    print("   Top citations:")
                    for citation in data['citations'][:3]:
                        print(f"     [{citation['citation_id']}] Score: {citation['relevance_score']:.3f}")
                
                return True, data
            elif data['status'] == 'no_results':
                print(f"⚠️ No results found for query")
                return True, data
            else:
                print(f"❌ Query failed: {data.get('message', 'Unknown error')}")
                return False, None
        else:
            print(f"❌ Query request failed: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False, None

def test_batch_queries():
    """Test multiple queries in sequence"""
    print("\n🔄 Testing Batch Queries...")
    
    successful_queries = 0
    total_time = 0
    
    for i, query in enumerate(TEST_QUERIES):
        print(f"\n--- Query {i+1}/{len(TEST_QUERIES)} ---")
        success, data = test_query(query, top_k=5)
        if success:
            successful_queries += 1
            if data and 'metadata' in data:
                total_time += data['metadata'].get('total_processing_time', 0)
    
    print(f"\n📊 Batch Query Results:")
    print(f"   Successful: {successful_queries}/{len(TEST_QUERIES)}")
    print(f"   Total processing time: {total_time:.2f}s")
    print(f"   Average time per query: {total_time/len(TEST_QUERIES):.2f}s")
    
    return successful_queries == len(TEST_QUERIES)

def test_citation_system():
    """Test the citation system specifically"""
    print("\n📚 Testing Citation System...")
    
    # Test with a specific query
    query = "What is machine learning and how does it work?"
    success, data = test_query(query, top_k=3)
    
    if success and data and data.get('status') == 'success':
        citations = data.get('citations', [])
        sources = data.get('sources', [])
        
        print(f"\n📖 Citation Analysis:")
        print(f"   Citations found: {len(citations)}")
        print(f"   Sources available: {len(sources)}")
        
        if citations:
            print("\n   Citation Details:")
            for citation in citations:
                print(f"     [{citation['citation_id']}] Score: {citation['relevance_score']:.3f}")
                print(f"        Source: {citation['source']}")
                print(f"        Chunk: {citation['chunk_id']}")
                print(f"        Text preview: {citation['text_snippet'][:100]}...")
        
        return True
    else:
        print("❌ Citation system test failed")
        return False

def test_performance_metrics():
    """Test performance metrics endpoint"""
    print("\n📊 Testing Performance Metrics...")
    try:
        response = requests.get(f"{API_BASE_URL}/performance")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Performance metrics retrieved:")
            print(f"   System info: {data['system_info']}")
            return True
        else:
            print(f"❌ Performance metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Performance metrics error: {e}")
        return False

def test_system_stats():
    """Test system statistics endpoint"""
    print("\n📈 Testing System Stats...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System stats retrieved:")
            print(f"   Vector DB: {data['vector_db_stats']}")
            print(f"   System config: {data['system_config']}")
            return True
        else:
            print(f"❌ System stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ System stats error: {e}")
        return False

def run_complete_test_suite():
    """Run the complete test suite"""
    print("🚀 Starting Complete RAG System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Ingestion", test_ingestion),
        ("Citation System", test_citation_system),
        ("Batch Queries", test_batch_queries),
        ("Performance Metrics", test_performance_metrics),
        ("System Stats", test_system_stats)
    ]
    
    results = {}
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 Summary: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("🎉 ALL TESTS PASSED! RAG system is working correctly.")
        print("✅ Ready for frontend integration and deployment.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    # Save results
    with open('complete_system_test_results.json', 'w') as f:
        json.dump({
            "test_results": results,
            "summary": {
                "total_tests": len(tests),
                "passed_tests": passed_tests,
                "failed_tests": len(tests) - passed_tests,
                "success_rate": f"{(passed_tests/len(tests)*100):.1f}%"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to complete_system_test_results.json")
    
    return passed_tests == len(tests)

if __name__ == "__main__":
    success = run_complete_test_suite()
    exit(0 if success else 1)
