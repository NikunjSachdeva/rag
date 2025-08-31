#!/usr/bin/env python3
"""
Test script to verify the asyncio event loop fix
"""

import asyncio
import time
from services.retrieve import retrieve, retrieve_async
from services.rerank import rerank, rerank_async
from services.ingest import ingest_text, ingest_text_async

async def test_async_functions():
    """Test async functions directly"""
    print("🧪 Testing async functions...")
    
    try:
        # Test async retrieve
        print("Testing retrieve_async...")
        results = await retrieve_async("test query", top_k=5)
        print(f"✅ retrieve_async successful: {len(results)} results")
        
        # Test async rerank
        print("Testing rerank_async...")
        docs = [{"metadata": {"text": "test document"}}]
        ranked = await rerank_async("test query", docs, top_k=1)
        print(f"✅ rerank_async successful: {len(ranked)} results")
        
        return True
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        return False

def test_sync_functions():
    """Test sync functions (should not have event loop issues)"""
    print("\n🧪 Testing sync functions...")
    
    try:
        # Test sync retrieve
        print("Testing retrieve...")
        results = retrieve("test query", top_k=5)
        print(f"✅ retrieve successful: {len(results)} results")
        
        # Test sync rerank
        print("Testing rerank...")
        docs = [{"metadata": {"text": "test document"}}]
        ranked = rerank("test query", docs, top_k=1)
        print(f"✅ rerank successful: {len(ranked)} results")
        
        return True
    except Exception as e:
        print(f"❌ Sync test failed: {e}")
        return False

async def test_mixed_usage():
    """Test mixing sync and async functions"""
    print("\n🧪 Testing mixed usage...")
    
    try:
        # Use sync function inside async context
        print("Testing sync function inside async context...")
        results = retrieve("test query", top_k=5)
        print(f"✅ Mixed usage successful: {len(results)} results")
        
        return True
    except Exception as e:
        print(f"❌ Mixed usage test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting asyncio event loop fix tests...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Async functions
    if await test_async_functions():
        tests_passed += 1
    
    # Test 2: Sync functions
    if test_sync_functions():
        tests_passed += 1
    
    # Test 3: Mixed usage
    if await test_mixed_usage():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Event loop issue is fixed.")
        return True
    else:
        print("❌ Some tests failed. Event loop issue may still exist.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
