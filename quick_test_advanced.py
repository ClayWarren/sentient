#!/usr/bin/env python3
"""
Quick test for Advanced AI
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from consciousness_core import ConsciousnessAI

def quick_test():
    print("🚀 Quick Advanced AI Test")
    
    ai = ConsciousnessAI(consciousness_enabled=True, ai_model="advanced-ai")
    
    # Single test
    result = ai.generate("Hello, I'm testing your consciousness capabilities", max_tokens=50)
    
    print(f"Response: {result.text}")
    print(f"Consciousness: {result.consciousness_metrics.overall_consciousness:.1%}")
    
    # Model info
    model_info = ai.consciousness.ai_engine.get_model_info()
    print(f"Model: {model_info.get('model_type')}")
    print(f"Parameters: {model_info.get('parameters', 0):,}")
    print(f"Advanced: {model_info.get('is_gemma', False)}")

if __name__ == "__main__":
    quick_test()