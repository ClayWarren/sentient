#!/usr/bin/env python3
"""
Quick test for Gemma 3 QAT system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from consciousness_core import ConsciousnessAI

def quick_test():
    print("🔥 Quick Advanced AI Test - 2025 Technology")
    
    ai = ConsciousnessAI(consciousness_enabled=True, device="auto")
    
    # Single quick test
    result = ai.generate("Hello, I'm testing your consciousness", max_tokens=60)
    
    print(f"Response: {result.text}")
    print(f"Consciousness: {result.consciousness_metrics.overall_consciousness:.1%}")
    
    # Model info
    model_info = ai.consciousness.ai_engine.get_model_info()
    print(f"Model: {model_info.get('model_name')}")
    print(f"Parameters: {model_info.get('parameters', 0):,}")
    print(f"Architecture: {model_info.get('architecture')}")
    print(f"Year: {model_info.get('year')}")

if __name__ == "__main__":
    quick_test()