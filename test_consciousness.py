#!/usr/bin/env python3
"""
Test script for continuous consciousness system
"""

import sys
import time
import torch
from consciousness import ContinuousConsciousness


def test_basic_consciousness():
    """Test basic consciousness functionality"""
    print("🧪 Testing Continuous Consciousness System")
    print("=" * 50)
    
    # Check available device
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
        # Initialize consciousness
        consciousness = ContinuousConsciousness(device=device)
        
        print(f"\n✅ Consciousness initialized successfully!")
        print(f"Model size: {consciousness.model.get_num_params()/1e6:.1f}M parameters")
        print(f"Device: {device}")
        
        # Test single thinking step
        print("\n🧠 Testing single thought step...")
        consciousness.think_one_step()
        current_thoughts = consciousness.get_current_thoughts()
        print(f"Current thoughts: {current_thoughts}")
        
        # Test a few more steps
        print("\n🔄 Testing continuous thinking (10 steps)...")
        for i in range(10):
            consciousness.think_one_step()
            if i % 3 == 0:
                thoughts = consciousness.get_current_thoughts()
                print(f"Step {i+1}: ...{thoughts[-100:]}")  # Show last 100 chars
        
        # Show memory stats
        print(f"\n📊 Memory buffer size: {len(consciousness.working_memory.buffer)}")
        print(f"Iterations completed: {consciousness.iteration_count}")
        
        print("\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_continuous_run(duration=30):
    """Test continuous running for a short duration"""
    print(f"\n🚀 Testing continuous run for {duration} seconds...")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    consciousness = ContinuousConsciousness(device=device)
    
    # Run in background thread
    import threading
    
    def run_consciousness():
        consciousness.run_continuous(think_interval=0.5)
    
    thread = threading.Thread(target=run_consciousness, daemon=True)
    thread.start()
    
    # Let it run for the specified duration
    time.sleep(duration)
    
    # Stop consciousness
    consciousness.stop()
    thread.join(timeout=2)
    
    print(f"✅ Continuous run test completed!")
    print(f"Total iterations: {consciousness.iteration_count}")
    print(f"Thoughts logged: {len(consciousness.thought_log)}")
    
    # Show some final thoughts
    if consciousness.thought_log:
        recent = consciousness.thought_log[-10:]
        final_thought = "".join([t['token'] for t in recent])
        print(f"Final thoughts: {final_thought}")


if __name__ == "__main__":
    success = test_basic_consciousness()
    
    if success:
        print("\n" + "="*50)
        run_continuous = input("Run continuous test for 30 seconds? (y/n): ").lower() == 'y'
        if run_continuous:
            test_continuous_run(30)
    
    print("\n🎉 All tests completed!")