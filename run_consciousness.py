#!/usr/bin/env python3
"""
Simple script to run continuous consciousness
"""

import sys
from consciousness import ContinuousConsciousness


def main():
    print("🧠 Starting Continuous AI Consciousness")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Initialize consciousness
    consciousness = ContinuousConsciousness(device='mps')
    
    # Run continuously
    try:
        consciousness.run_continuous(think_interval=0.2)
    except KeyboardInterrupt:
        print("\n👋 Consciousness stopped gracefully")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()