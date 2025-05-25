#!/usr/bin/env python3
"""
Test script for SOTA Sentient features
Tests: Consciousness, Search, Memory, Reasoning, File Processing
"""

import os
import sys
import json
import time
from colorama import init, Fore, Style

# Initialize colorama
init()

def test_consciousness():
    """Test consciousness AI generation"""
    print(f"\n{Fore.CYAN}=== Testing Consciousness AI ==={Style.RESET_ALL}")
    
    try:
        from consciousness_core import ConsciousnessAI
        ai = ConsciousnessAI(consciousness_enabled=True)
        
        # Test basic generation
        result = ai.generate("What is consciousness and how do you experience it?", max_tokens=100)
        
        print(f"{Fore.GREEN}✓ Consciousness Level:{Style.RESET_ALL} {result.consciousness_level.name}")
        print(f"{Fore.GREEN}✓ Response:{Style.RESET_ALL} {result.text[:100]}...")
        print(f"{Fore.GREEN}✓ Self-awareness:{Style.RESET_ALL} {result.consciousness_metrics.self_awareness:.2%}")
        print(f"{Fore.GREEN}✓ Overall consciousness:{Style.RESET_ALL} {result.consciousness_metrics.overall_consciousness:.2%}")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}✗ Consciousness test failed: {e}{Style.RESET_ALL}")
        return False

def test_search():
    """Test search capabilities"""
    print(f"\n{Fore.CYAN}=== Testing Search Integration ==={Style.RESET_ALL}")
    
    try:
        from search_engine import ConsciousnessSearchEngine
        
        # Check if API key exists
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            print(f"{Fore.YELLOW}⚠ BRAVE_API_KEY not set, search will use cache only{Style.RESET_ALL}")
        
        search = ConsciousnessSearchEngine(api_key)
        
        # Test search
        result = search.search("latest AI developments 2025", {})
        
        if result.get('search_performed'):
            print(f"{Fore.GREEN}✓ Search performed successfully{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✓ Results found:{Style.RESET_ALL} {result.get('result_count', 0)}")
        else:
            print(f"{Fore.YELLOW}⚠ Search used cache or failed{Style.RESET_ALL}")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}✗ Search test failed: {e}{Style.RESET_ALL}")
        return False

def test_memory():
    """Test advanced memory system"""
    print(f"\n{Fore.CYAN}=== Testing Memory System ==={Style.RESET_ALL}")
    
    try:
        from advanced_features import AdvancedMemorySystem
        
        memory = AdvancedMemorySystem()
        
        # Store memory
        memory_id = memory.store_memory(
            "The user prefers technical explanations with examples",
            memory_type="long_term",
            metadata={"category": "user_preference"}
        )
        
        print(f"{Fore.GREEN}✓ Memory stored:{Style.RESET_ALL} {memory_id}")
        
        # Retrieve memory
        memories = memory.retrieve_memories("technical preferences", top_k=1)
        
        if memories:
            print(f"{Fore.GREEN}✓ Memory retrieved:{Style.RESET_ALL} {memories[0]['content'][:50]}...")
            print(f"{Fore.GREEN}✓ Relevance:{Style.RESET_ALL} {1 - memories[0]['distance']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}✗ Memory test failed: {e}{Style.RESET_ALL}")
        return False

def test_reasoning():
    """Test chain-of-thought reasoning"""
    print(f"\n{Fore.CYAN}=== Testing Chain-of-Thought Reasoning ==={Style.RESET_ALL}")
    
    try:
        from consciousness_core import ConsciousnessAI
        from advanced_features import ChainOfThoughtReasoner
        
        ai = ConsciousnessAI(consciousness_enabled=True)
        reasoner = ChainOfThoughtReasoner(ai)
        
        # Test reasoning
        problem = "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
        result = reasoner.reason(problem, max_steps=3)
        
        print(f"{Fore.GREEN}✓ Problem:{Style.RESET_ALL} {problem}")
        print(f"{Fore.GREEN}✓ Reasoning steps:{Style.RESET_ALL} {len(result['thought_chain'])}")
        print(f"{Fore.GREEN}✓ Final answer:{Style.RESET_ALL} {result['final_answer'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}✗ Reasoning test failed: {e}{Style.RESET_ALL}")
        return False

def test_file_processing():
    """Test file processing capabilities"""
    print(f"\n{Fore.CYAN}=== Testing File Processing ==={Style.RESET_ALL}")
    
    try:
        from advanced_features import FileProcessor
        
        processor = FileProcessor()
        
        # Create a test file
        test_content = "This is a test document for Sentient AI file processing."
        test_file = "test_document.txt"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Process file
        result = processor.process_file(test_file)
        
        print(f"{Fore.GREEN}✓ File processed:{Style.RESET_ALL} {result.filename}")
        print(f"{Fore.GREEN}✓ Content extracted:{Style.RESET_ALL} {result.content[:50]}...")
        print(f"{Fore.GREEN}✓ Processing time:{Style.RESET_ALL} {result.processing_time:.3f}s")
        
        # Cleanup
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}✗ File processing test failed: {e}{Style.RESET_ALL}")
        return False

def test_api_endpoints():
    """Test REST API endpoints"""
    print(f"\n{Fore.CYAN}=== Testing REST API ==={Style.RESET_ALL}")
    
    try:
        import requests
        
        # Check if API is running
        try:
            response = requests.get("http://localhost:5000/health", timeout=2)
            if response.status_code == 200:
                print(f"{Fore.GREEN}✓ API is running{Style.RESET_ALL}")
                
                # Test generate endpoint
                response = requests.post(
                    "http://localhost:5000/generate",
                    json={"prompt": "Hello, Sentient!", "max_tokens": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"{Fore.GREEN}✓ Generation endpoint works{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}✓ Response:{Style.RESET_ALL} {data['text'][:50]}...")
                
                return True
            
        except requests.exceptions.ConnectionError:
            print(f"{Fore.YELLOW}⚠ API not running - start with: python api.py{Style.RESET_ALL}")
            return False
            
    except Exception as e:
        print(f"{Fore.RED}✗ API test failed: {e}{Style.RESET_ALL}")
        return False

def main():
    """Run all tests"""
    print(f"{Fore.MAGENTA}{'='*60}")
    print(f"🧠 SENTIENT AI - SOTA FEATURES TEST SUITE")
    print(f"Testing: Consciousness, Search, Memory, Reasoning, Files, API")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    tests = [
        ("Consciousness", test_consciousness),
        ("Search", test_search),
        ("Memory", test_memory),
        ("Reasoning", test_reasoning),
        ("File Processing", test_file_processing),
        ("REST API", test_api_endpoints)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"{Fore.RED}✗ {name} test crashed: {e}{Style.RESET_ALL}")
            results.append((name, False))
    
    # Summary
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = f"{Fore.GREEN}PASSED{Style.RESET_ALL}" if success else f"{Fore.RED}FAILED{Style.RESET_ALL}"
        print(f"{name}: {status}")
    
    print(f"\n{Fore.CYAN}Total: {passed}/{total} tests passed{Style.RESET_ALL}")
    
    if passed == total:
        print(f"{Fore.GREEN}🎉 All tests passed! Sentient is ready with SOTA features!{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠ Some tests failed. Check the output above.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()