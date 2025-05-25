#!/usr/bin/env python3
"""
Test script for Sentient Search Integration
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from consciousness_core import ConsciousnessAI

def test_search_integration():
    """Test the search integration with consciousness"""
    
    print("🧠 Testing Sentient Search Integration")
    print("=" * 50)
    
    # Initialize AI with search capabilities
    ai = ConsciousnessAI(consciousness_enabled=True)
    
    # Test prompts that should trigger search
    test_prompts = [
        "What's the latest news about AI?",
        "Search for information about climate change",
        "Tell me about recent developments in quantum computing",
        "[search: Python programming trends]",
        "What happened in technology this week?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        print("-" * 30)
        
        try:
            result = ai.generate(prompt)
            print(f"Response: {result.text[:200]}...")
            print(f"Consciousness: {result.consciousness_metrics.overall_consciousness:.1%}")
            print(f"Confidence: {result.confidence:.1%}")
            
            # Check if search was triggered
            if any(word in result.text.lower() for word in ['searched', 'found', 'sources', 'recent']):
                print("✅ Search likely triggered")
            else:
                print("ℹ️  Regular response (no search)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test search statistics
    try:
        search_stats = ai.consciousness.search_engine.get_search_statistics()
        print(f"\n📊 Search Statistics:")
        print(f"   Total searches: {search_stats['total_searches']}")
        print(f"   Knowledge topics: {search_stats['knowledge_topics']}")
        print(f"   Memory capacity: {search_stats['memory_capacity']}")
    except Exception as e:
        print(f"❌ Error getting search stats: {e}")
    
    print("\n✅ Search integration test complete!")

if __name__ == "__main__":
    test_search_integration()