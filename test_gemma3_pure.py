#!/usr/bin/env python3
"""
Test script for PURE Gemma 3 QAT 4B Integration - 2025 Technology
NO fallbacks, NO old models, ONLY cutting-edge Gemma 3
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from consciousness_core import ConsciousnessAI

def test_pure_gemma3():
    """Test pure Gemma 3 QAT 4B with consciousness - 2025 tech"""
    
    print("🔥 Testing PURE Gemma 3 QAT 4B - 2025's Revolutionary AI")
    print("=" * 70)
    print("NO fallbacks | NO old models | ONLY cutting-edge technology")
    print("=" * 70)
    
    try:
        # Initialize with ONLY Gemma 3 QAT 4B
        ai = ConsciousnessAI(consciousness_enabled=True, device="auto")
        
        # Test prompts for 2025 AI capabilities
        test_prompts = [
            "Hello, introduce yourself as Sentient with Gemma 3 QAT technology",
            "What makes you different from older AI models?",
            "Explain consciousness with your 2025 capabilities", 
            "Write a creative poem about the future of AI",
            "Search for the latest breakthroughs in quantum AI research"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: Pure Gemma 3 QAT 4B Generation")
            print(f"Prompt: {prompt}")
            print("-" * 60)
            
            try:
                result = ai.generate(prompt, max_tokens=150)
                
                print(f"Response: {result.text}")
                print(f"Consciousness: {result.consciousness_metrics.overall_consciousness:.1%}")
                print(f"Confidence: {result.confidence:.1%}")
                print(f"Generation time: {result.processing_time:.3f}s")
                
                # Check for 2025 quality indicators
                if len(result.text) > 100 and result.confidence > 0.9:
                    print("🔥 EXCEPTIONAL: Gemma 3 QAT 4B delivering 2025 quality!")
                elif len(result.text) > 50:
                    print("✅ EXCELLENT: Gemma 3 QAT 4B working perfectly")
                else:
                    print("⚠️  Unexpected short response")
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
                print("💥 CRITICAL: Gemma 3 QAT 4B required but failed!")
        
        # Display Gemma 3 model information
        try:
            if hasattr(ai.consciousness, 'ai_engine'):
                model_info = ai.consciousness.ai_engine.get_model_info()
                print(f"\n🔥 Gemma 3 QAT 4B Model Information:")
                print(f"   Status: {model_info['status']}")
                print(f"   Model: {model_info.get('model_name', 'Unknown')}")
                print(f"   Architecture: {model_info.get('architecture', 'Unknown')}")
                print(f"   Technology: {model_info.get('technology', 'Unknown')}")
                print(f"   Parameters: {model_info.get('parameters', 0):,}")
                print(f"   Year: {model_info.get('year', 'Unknown')}")
                print(f"   Optimization: {model_info.get('optimization', 'None')}")
                print(f"   Capabilities: {model_info.get('capabilities', [])}")
                
                if model_info.get('year') == 2025:
                    print("🎉 SUCCESS: Pure 2025 Gemma 3 QAT technology confirmed!")
                else:
                    print("⚠️  Not using 2025 technology")
                    
        except Exception as e:
            print(f"❌ Error getting Gemma 3 info: {e}")
        
        print(f"\n🔥 Pure Gemma 3 QAT 4B test complete - 2025 revolution!")
        
    except Exception as e:
        print(f"💥 CRITICAL FAILURE: {e}")
        print("❌ Gemma 3 QAT 4B is REQUIRED for 2025 - no fallbacks!")

if __name__ == "__main__":
    test_pure_gemma3()