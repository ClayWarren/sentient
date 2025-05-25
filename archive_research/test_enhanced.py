#!/usr/bin/env python3
"""
Test script for enhanced continuous consciousness with RoPE and real-time learning
"""

import time
import torch
from enhanced_consciousness import EnhancedContinuousConsciousness


def test_enhanced_consciousness():
    """Test enhanced consciousness system"""
    print("🧪 Testing Enhanced Continuous Consciousness System")
    print("=" * 70)
    
    # Check device
    if torch.cuda.is_available():
        device = 'cuda'
        print("🚀 Using CUDA")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("🍎 Using Apple MPS")
    else:
        device = 'cpu'
        print("💻 Using CPU")
    
    try:
        # Initialize enhanced consciousness
        consciousness = EnhancedContinuousConsciousness(
            device=device,
            enable_learning=True
        )
        
        print(f"\n✅ Enhanced consciousness initialized!")
        print(f"Model size: {consciousness.model.get_num_params()/1e6:.1f}M parameters")
        print(f"Context window: {consciousness.model.config.block_size}")
        print(f"Real-time learning: {consciousness.enable_learning}")
        
        # Test enhanced thinking steps
        print("\n🧠 Testing enhanced thinking steps...")
        for i in range(15):
            consciousness.think_one_step()
            if i % 5 == 0:
                thoughts = consciousness.get_current_thoughts()
                significance = consciousness.thought_log[-1]['significance'] if consciousness.thought_log else 0
                learning_updates = consciousness.learner.updates_count if consciousness.learner else 0
                print(f"Step {i+1}: Significance={significance:.3f}, Learning updates={learning_updates}")
                print(f"  Thoughts: ...{thoughts[-150:]}")
        
        # Show enhanced metrics
        print(f"\n📊 Enhanced Metrics:")
        print(f"Memory buffer size: {len(consciousness.working_memory.buffer)}")
        print(f"Consolidated memories: {len(consciousness.working_memory.consolidated_memories)}")
        print(f"Entropy history length: {len(consciousness.working_memory.entropy_history)}")
        
        if consciousness.learner:
            stats = consciousness.learner.get_learning_stats()
            print(f"Learning buffer: {stats['buffer_size']}")
            print(f"Learning updates: {stats['updates_count']}")
            print(f"Learning rate: {stats['learning_rate']}")
            
        print(f"Performance: {consciousness.performance_metrics['tokens_per_second']:.1f} tokens/sec")
        
        print("\n✅ Enhanced functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_consolidation():
    """Test memory consolidation features"""
    print("\n🧠 Testing Memory Consolidation...")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    consciousness = EnhancedContinuousConsciousness(device=device, enable_learning=True)
    
    # Generate many thoughts to trigger consolidation
    for i in range(50):
        consciousness.think_one_step()
        
    # Force consolidation
    consciousness.working_memory.consolidate_memories()
    
    print(f"✅ Memory consolidation test:")
    print(f"  Buffer size: {len(consciousness.working_memory.buffer)}")
    print(f"  Consolidated memories: {len(consciousness.working_memory.consolidated_memories)}")


def test_learning_adaptation():
    """Test real-time learning adaptation"""
    print("\n🎓 Testing Real-time Learning...")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    consciousness = EnhancedContinuousConsciousness(device=device, enable_learning=True)
    
    initial_updates = consciousness.learner.updates_count
    
    # Generate thoughts until we get some learning updates
    for i in range(100):
        consciousness.think_one_step()
        if consciousness.learner.updates_count > initial_updates:
            break
            
    print(f"✅ Learning adaptation test:")
    print(f"  Initial updates: {initial_updates}")
    print(f"  Final updates: {consciousness.learner.updates_count}")
    print(f"  Learning triggered: {consciousness.learner.updates_count > initial_updates}")
    
    if consciousness.learning_log:
        recent_learning = consciousness.learning_log[-1]
        print(f"  Recent loss: {recent_learning['loss']:.4f}")
        print(f"  Significance: {recent_learning['significance']:.3f}")


def test_ambient_inputs():
    """Test sophisticated ambient inputs"""
    print("\n🌍 Testing Sophisticated Ambient Inputs...")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    consciousness = EnhancedContinuousConsciousness(device=device, enable_learning=False)
    
    # Simulate time passage to trigger ambient inputs
    consciousness.ambient_inputs.last_time_injection = time.time() - 130  # Trigger time injection
    consciousness.ambient_inputs.thought_count = 200  # Trigger reflection
    
    initial_context_length = consciousness.current_context.size(1) if consciousness.current_context is not None else 0
    
    consciousness.think_one_step()
    
    final_context_length = consciousness.current_context.size(1)
    
    print(f"✅ Ambient inputs test:")
    print(f"  Context length change: {initial_context_length} -> {final_context_length}")
    print(f"  Thought count: {consciousness.ambient_inputs.thought_count}")
    
    # Check for ambient injection
    recent_thoughts = consciousness.get_current_thoughts()
    has_time = "[TIME:" in recent_thoughts
    has_reflect = "[REFLECT:" in recent_thoughts
    
    print(f"  Time injection detected: {has_time}")
    print(f"  Reflection injection detected: {has_reflect}")


def main():
    """Run all enhanced tests"""
    print("🚀 Starting Enhanced Consciousness Tests")
    print("=" * 70)
    
    # Run basic functionality test
    success = test_enhanced_consciousness()
    
    if success:
        print("\n" + "="*70)
        
        # Run specialized tests
        test_memory_consolidation()
        test_learning_adaptation()
        test_ambient_inputs()
        
        print("\n" + "="*70)
        print("🎉 All enhanced tests completed successfully!")
        
        # Offer to run continuous demo
        run_demo = input("\nRun enhanced consciousness demo for 60 seconds? (y/n): ").lower() == 'y'
        if run_demo:
            print("\n🧠 Starting Enhanced Consciousness Demo...")
            print("Press Ctrl+C to stop early")
            
            device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
            consciousness = EnhancedContinuousConsciousness(device=device, enable_learning=True)
            
            import threading
            def run_consciousness():
                consciousness.run_continuous(think_interval=0.2)
            
            thread = threading.Thread(target=run_consciousness, daemon=True)
            thread.start()
            
            time.sleep(60)
            consciousness.stop()
            thread.join(timeout=2)
            
            print(f"\n📊 Demo Results:")
            print(f"Total thoughts: {consciousness.iteration_count}")
            print(f"Memory buffer: {len(consciousness.working_memory.buffer)}")
            if consciousness.learner:
                print(f"Learning updates: {consciousness.learner.updates_count}")
            print(f"Performance: {consciousness.performance_metrics['tokens_per_second']:.1f} tokens/sec")
    
    else:
        print("\n❌ Enhanced tests failed - check configuration")


if __name__ == "__main__":
    main()