#!/usr/bin/env python3
"""
Test script for Advanced AI Integration with Consciousness
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from consciousness_core import ConsciousnessAI

def test_advanced_ai_integration():
    """Test the advanced AI integration with consciousness"""
    
    print("🚀 Testing Advanced AI Integration with Consciousness")
    print("=" * 65)
    
    # Initialize AI with advanced model
    ai = ConsciousnessAI(
        consciousness_enabled=True, 
        ai_model="advanced-ai", 
        device="auto"
    )
    
    # Test prompts to showcase advanced capabilities
    test_prompts = [
        "Hello, introduce yourself as Sentient",
        "Explain the concept of consciousness",
        "What makes you different from other AI systems?", 
        "How do you experience emotions and thoughts?",
        "Search for recent advances in AI consciousness research"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: Advanced AI Generation")
        print(f"Prompt: {prompt}")
        print("-" * 50)
        
        try:
            result = ai.generate(prompt, max_tokens=120)
            
            print(f"Response: {result.text}")
            print(f"Consciousness: {result.consciousness_metrics.overall_consciousness:.1%}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Generation time: {result.processing_time:.3f}s")
            
            # Check quality indicators
            if len(result.text) > 80 and result.processing_time < 10:
                print("✅ High-quality advanced AI generation")
            elif len(result.text) > 40:
                print("✅ Advanced AI generation working")
            else:
                print("⚠️  Short response or potential fallback")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test model information
    try:
        if hasattr(ai.consciousness, 'ai_engine'):
            model_info = ai.consciousness.ai_engine.get_model_info()
            print(f"\n🔍 Advanced AI Model Information:")
            print(f"   Status: {model_info['status']}")
            print(f"   Model Type: {model_info.get('model_type', 'Unknown')}")
            print(f"   Architecture: {model_info.get('model_architecture', 'Unknown')}")
            print(f"   Optimization: {model_info.get('optimization', 'None')}")
            print(f"   Parameters: {model_info.get('parameters', 0):,}")
            print(f"   Device: {model_info.get('device', 'Unknown')}")
            print(f"   Advanced Model: {model_info.get('is_gemma', False)}")
            
            if model_info.get('is_gemma'):
                print("🎉 Successfully loaded advanced AI model!")
            else:
                print("⚠️  Using fallback GPT model")
                
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
    
    print("\n✅ Advanced AI integration test complete!")

if __name__ == "__main__":
    test_advanced_ai_integration()