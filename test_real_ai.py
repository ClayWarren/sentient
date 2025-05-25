#!/usr/bin/env python3
"""
Test script for Real AI Integration with Consciousness
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from consciousness_core import ConsciousnessAI

def test_real_ai_integration():
    """Test the real AI integration with consciousness"""
    
    print("🤖 Testing REAL AI Integration with Consciousness")
    print("=" * 60)
    
    # Initialize AI with real neural network generation
    ai = ConsciousnessAI(consciousness_enabled=True, ai_model="gpt2", device="cpu")
    
    # Test prompts
    test_prompts = [
        "Hello, what are you?",
        "Tell me about consciousness",
        "What's 2+2?",
        "Write a creative story about AI",
        "Search for latest AI developments"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: Real AI Generation")
        print(f"Prompt: {prompt}")
        print("-" * 40)
        
        try:
            result = ai.generate(prompt)
            print(f"Response: {result.text}")
            print(f"Consciousness: {result.consciousness_metrics.overall_consciousness:.1%}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Generation time: {result.processing_time:.3f}s")
            
            # Check if response shows signs of real AI generation
            if len(result.text) > 50 and not result.text.startswith("I'm "):
                print("✅ Real AI generation detected")
            else:
                print("⚠️  Response may be fallback")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test model info
    try:
        if hasattr(ai.consciousness, 'ai_engine'):
            model_info = ai.consciousness.ai_engine.get_model_info()
            print(f"\n🔍 Model Information:")
            print(f"   Status: {model_info['status']}")
            print(f"   Model: {model_info.get('model_type', 'Unknown')}")
            print(f"   Parameters: {model_info.get('parameters', 0):,}")
            print(f"   Device: {model_info.get('device', 'Unknown')}")
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
    
    print("\n✅ Real AI integration test complete!")

if __name__ == "__main__":
    test_real_ai_integration()