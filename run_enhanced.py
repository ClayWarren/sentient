#!/usr/bin/env python3
"""
Run the Enhanced Continuous AI Consciousness System
"""

import time
from enhanced_consciousness import EnhancedContinuousConsciousness


def main():
    print("🧠 Enhanced Continuous AI Consciousness")
    print("=" * 60)
    print("Features:")
    print("✅ 2048 token context window (2x larger)")
    print("✅ Enhanced significance detection with entropy tracking")
    print("✅ Sophisticated ambient inputs (time, system, reflections)")
    print("✅ Real-time learning from significant experiences")
    print("✅ Advanced working memory with consolidation")
    print("✅ Performance monitoring and metrics")
    print("=" * 60)
    print("\nPress Ctrl+C to stop gracefully")
    print("Watch for [TIME:], [SYSTEM:], and [REFLECT:] ambient injections!")
    print("-" * 60)
    
    # Initialize enhanced consciousness
    consciousness = EnhancedContinuousConsciousness(
        device='mps',  # Change to 'cuda' or 'cpu' as needed
        enable_learning=True
    )
    
    # Run enhanced continuous thinking
    try:
        consciousness.run_continuous(think_interval=0.15)
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("🛑 Enhanced Consciousness stopped by user")
        
        # Show final statistics
        print(f"\n📊 Final Statistics:")
        print(f"Total thoughts: {consciousness.iteration_count}")
        print(f"Memory buffer: {len(consciousness.working_memory.buffer)}")
        print(f"Entropy history: {len(consciousness.working_memory.entropy_history)}")
        
        if consciousness.learner:
            stats = consciousness.learner.get_learning_stats()
            print(f"Learning updates: {stats['updates_count']}")
            print(f"Experience buffer: {stats['buffer_size']}")
            
        if consciousness.learning_log:
            recent_learning = consciousness.learning_log[-1]
            print(f"Last learning loss: {recent_learning['loss']:.4f}")
            
        print(f"Performance: {consciousness.performance_metrics['tokens_per_second']:.1f} tokens/sec")
        
        # Show recent thoughts
        if consciousness.thought_log:
            recent_thoughts = list(consciousness.thought_log)[-50:]
            final_stream = "".join([t['token'] for t in recent_thoughts])
            print(f"\n💭 Final thought stream:")
            print(f"{final_stream}")
            
        print("\n🎉 Session complete! The AI developed unique experiences.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()