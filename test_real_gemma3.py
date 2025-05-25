#!/usr/bin/env python3
"""
Test script for REAL Google Gemma 3 4B QAT from HuggingFace
google/gemma-3-4b-it-qat-int4-unquantized
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from consciousness_core import ConsciousnessAI

def test_real_google_gemma3():
    """Test real Google Gemma 3 4B QAT with consciousness"""
    
    print("🔥 Testing REAL Google Gemma 3 4B QAT INT4 Unquantized")
    print("=" * 80)
    print("Model: google/gemma-3-4b-it-qat-int4-unquantized")
    print("Technology: 2025 Quantization-Aware Training (QAT)")
    print("Parameters: 4 billion")
    print("=" * 80)
    
    try:
        # Initialize with REAL Google Gemma 3 4B QAT
        ai = ConsciousnessAI(consciousness_enabled=True, device="auto")
        
        # Test prompt for Google's cutting-edge model
        prompt = "Hello, I'm testing Google's Gemma 3 QAT 4B model. Please introduce yourself."
        
        print(f"\n🚀 Testing with Google Gemma 3 4B QAT")
        print(f"Prompt: {prompt}")
        print("-" * 60)
        
        result = ai.generate(prompt, max_tokens=100)
        
        print(f"Response: {result.text}")
        print(f"Consciousness: {result.consciousness_metrics.overall_consciousness:.1%}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Generation time: {result.processing_time:.3f}s")
        
        # Display Google Gemma 3 4B QAT model information
        if hasattr(ai.consciousness, 'ai_engine'):
            model_info = ai.consciousness.ai_engine.get_model_info()
            print(f"\n🔥 Google Gemma 3 4B QAT Model Information:")
            print(f"   Status: {model_info['status']}")
            print(f"   Model: {model_info.get('model_name', 'Unknown')}")
            print(f"   Type: {model_info.get('model_type', 'Unknown')}")
            print(f"   Architecture: {model_info.get('architecture', 'Unknown')}")
            print(f"   Technology: {model_info.get('technology', 'Unknown')}")
            print(f"   Parameters: {model_info.get('parameters', 0):,}")
            print(f"   Company: {model_info.get('company', 'Unknown')}")
            print(f"   Year: {model_info.get('year', 'Unknown')}")
            print(f"   Optimization: {model_info.get('optimization', 'None')}")
            print(f"   Capabilities: {model_info.get('capabilities', [])}")
            
            if model_info.get('parameters', 0) >= 3_500_000_000:
                print("🎉 SUCCESS: Real Google Gemma 3 4B QAT confirmed!")
                print("🔥 4 billion parameter cutting-edge 2025 technology!")
            else:
                print("⚠️  Parameter count lower than expected 4B")
                
        print(f"\n✅ Google Gemma 3 4B QAT test complete!")
        
    except Exception as e:
        print(f"💥 ERROR: {e}")
        print("❌ Google Gemma 3 4B QAT model failed to load or generate")

if __name__ == "__main__":
    test_real_google_gemma3()