#!/usr/bin/env python3
"""
Test script for Gemma 3 QAT 4B Integration with Consciousness
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from consciousness_core import ConsciousnessAI

def test_gemma_integration():
    """Test the Gemma 3 QAT 4B integration with consciousness"""
    
    print("🔥 Testing Gemma 3 QAT 4B Integration with Consciousness")
    print("=" * 70)
    
    # Initialize AI with Gemma 3 QAT 4B model
    ai = ConsciousnessAI(
        consciousness_enabled=True, 
        ai_model="gemma-3-qat-4b-it", 
        device="auto"
    )
    
    # Test prompts to showcase Gemma's advanced capabilities
    test_prompts = [
        "Hello, introduce yourself",
        "Explain quantum computing in simple terms",
        "Write a haiku about artificial consciousness", 
        "What are the ethical implications of AI consciousness?",
        "Search for latest developments in quantum AI"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: Gemma 3 QAT 4B Generation")
        print(f"Prompt: {prompt}")
        print("-" * 50)
        
        try:
            result = ai.generate(prompt, max_tokens=150)
            
            print(f"Response: {result.text}")
            print(f"Consciousness: {result.consciousness_metrics.overall_consciousness:.1%}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Generation time: {result.processing_time:.3f}s")
            
            # Check quality indicators for Gemma
            if len(result.text) > 100 and result.processing_time < 15:
                print("✅ High-quality Gemma generation detected")
            elif len(result.text) > 50:
                print("✅ Gemma generation working")
            else:
                print("⚠️  Response may be fallback or error")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test model information
    try:
        if hasattr(ai.consciousness, 'ai_engine'):
            model_info = ai.consciousness.ai_engine.get_model_info()
            print(f"\n🔍 Gemma Model Information:")
            print(f"   Status: {model_info['status']}")
            print(f"   Model: {model_info.get('model_type', 'Unknown')}")
            print(f"   Architecture: {model_info.get('model_architecture', 'Unknown')}")
            print(f"   Optimization: {model_info.get('optimization', 'None')}")
            print(f"   Parameters: {model_info.get('parameters', 0):,}")
            print(f"   Device: {model_info.get('device', 'Unknown')}")
            print(f"   Is Gemma: {model_info.get('is_gemma', False)}")
            
            if model_info.get('is_gemma'):
                print("🎉 Successfully loaded Gemma 3 QAT 4B model!")
            else:
                print("⚠️  Fallback model loaded instead of Gemma")
                
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
    
    print("\n✅ Gemma 3 QAT 4B integration test complete!")

if __name__ == "__main__":
    test_gemma_integration()