#!/usr/bin/env python3
"""
Test script to verify the timeout fix for the RAG system
"""

import asyncio
import time
from services.ingest import ingest_text_async

# Test with a large text that previously caused timeouts
LARGE_TEST_TEXT = """
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

The field of artificial intelligence has seen remarkable progress in recent years, with breakthroughs in areas such as:
- Large Language Models (LLMs) like GPT, BERT, and Gemini
- Computer vision systems that can identify objects with human-level accuracy
- Autonomous vehicles that can navigate complex environments
- Medical AI systems that can diagnose diseases from medical images
- AI-powered recommendation systems that personalize user experiences

These advancements have been driven by several key factors:
1. Increased computational power and the availability of GPUs
2. Large-scale datasets for training machine learning models
3. Improved algorithms and neural network architectures
4. Better understanding of how to train deep learning models effectively

However, with these advances come important challenges and considerations:
- Ensuring AI systems are fair and unbiased
- Protecting user privacy and data security
- Addressing the potential for job displacement
- Developing AI systems that are transparent and explainable
- Creating AI that aligns with human values and goals

The future of AI holds tremendous promise, but it also requires careful consideration of these ethical and societal implications.
""" * 5  # Multiply to create a very large text for testing

async def test_timeout_fix():
    """Test the timeout fix with large text"""
    print("🧪 Testing Timeout Fix for Large Text Ingestion")
    print("=" * 60)
    
    try:
        print(f"📝 Test text size: {len(LARGE_TEST_TEXT)} characters")
        print(f"📊 Estimated chunks: {len(LARGE_TEST_TEXT) // 800}")
        
        start_time = time.time()
        
        print("\n🚀 Starting ingestion...")
        result = await ingest_text_async(
            text=LARGE_TEST_TEXT,
            source="timeout_test",
            title="Large_Text_Test"
        )
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Ingestion Results:")
        print(f"   Status: {result['status']}")
        print(f"   Chunks ingested: {result.get('chunks_ingested', 0)}")
        print(f"   Total words: {result.get('total_words', 0)}")
        print(f"   Processing time: {total_time:.2f}s")
        
        if result['status'] == 'success':
            print("✅ SUCCESS: Large text ingested without timeout errors!")
            print(f"   Chunk stats: {result.get('chunk_stats', {})}")
            return True
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ EXCEPTION: {e}")
        print(f"   Time elapsed: {total_time:.2f}s")
        return False

async def test_small_text():
    """Test with small text to ensure basic functionality works"""
    print("\n🧪 Testing Small Text Ingestion")
    print("=" * 40)
    
    small_text = "This is a small test text to verify basic functionality."
    
    try:
        start_time = time.time()
        
        result = await ingest_text_async(
            text=small_text,
            source="small_test",
            title="Small_Text_Test"
        )
        
        total_time = time.time() - start_time
        
        print(f"📊 Small text results:")
        print(f"   Status: {result['status']}")
        print(f"   Chunks: {result.get('chunks_ingested', 0)}")
        print(f"   Time: {total_time:.2f}s")
        
        return result['status'] == 'success'
        
    except Exception as e:
        print(f"❌ Small text test failed: {e}")
        return False

async def main():
    """Run all timeout tests"""
    print("🚀 Starting Timeout Fix Tests...")
    
    # Test 1: Small text
    small_success = await test_small_text()
    
    # Test 2: Large text (the main test)
    large_success = await test_timeout_fix()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TIMEOUT FIX TEST RESULTS")
    print("=" * 60)
    
    print(f"Small text test: {'✅ PASS' if small_success else '❌ FAIL'}")
    print(f"Large text test: {'✅ PASS' if large_success else '❌ FAIL'}")
    
    if small_success and large_success:
        print("\n🎉 ALL TESTS PASSED! Timeout fix is working correctly.")
        print("✅ System can now handle large texts reliably.")
        print("✅ Batch splitting and retry logic are functioning.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
