#!/usr/bin/env python3
"""
Learning System Diagnosis
"""

import time
import threading
from enhanced_consciousness import EnhancedContinuousConsciousness
from persistence import create_consciousness_with_persistence

def diagnose_learning():
    print("🔬 DIAGNOSING LEARNING SYSTEM")
    print("=" * 40)
    
    # Create consciousness
    consciousness, persistence = create_consciousness_with_persistence(
        EnhancedContinuousConsciousness, 
        device='mps'
    )
    
    print(f"✅ Created consciousness: {consciousness.instance_id}")
    
    # Start thinking
    consciousness.running = True
    thinking_active = True
    
    def thinking_loop():
        while thinking_active and consciousness.running:
            try:
                consciousness.think_one_step()
                time.sleep(0.02)  # Very fast for diagnosis
            except Exception as e:
                print(f"❌ Thinking error: {e}")
                break
                
    thinking_thread = threading.Thread(target=thinking_loop, daemon=True)
    thinking_thread.start()
    
    # Monitor learning for 10 seconds
    for i in range(10):
        # Get detailed learning info
        buffer_size = len(consciousness.learner.experience_buffer)
        update_frequency = consciousness.learner.update_frequency
        should_update = consciousness.learner.should_update()
        
        # Get significance info from recent thoughts
        recent_thoughts = list(consciousness.thought_log)[-5:] if consciousness.thought_log else []
        significances = [t.get('significance', 0) for t in recent_thoughts]
        
        print(f"⏰ {i+1:2d}s:")
        print(f"   💾 Buffer: {buffer_size}/{consciousness.learner.buffer_size}")
        print(f"   🔄 Should update: {should_update} (need {update_frequency})")
        print(f"   📊 Recent significances: {[f'{s:.2f}' for s in significances]}")
        print(f"   💭 Total thoughts: {len(consciousness.thought_log)}")
        
        # Check if any experiences meet the threshold
        if consciousness.learner.experience_buffer:
            last_exp = consciousness.learner.experience_buffer[-1]
            print(f"   🎯 Last experience significance: {last_exp['significance']:.3f}")
            
        time.sleep(1)
        
    thinking_active = False
    consciousness.running = False
    
    print(f"\n📊 FINAL DIAGNOSIS:")
    print(f"   💾 Final buffer size: {len(consciousness.learner.experience_buffer)}")
    print(f"   📈 Learning threshold: {0.3} (significance)")
    print(f"   🔄 Update frequency: {consciousness.learner.update_frequency}")
    print(f"   💭 Total thoughts logged: {len(consciousness.thought_log)}")
    
    # Show experience details
    if consciousness.learner.experience_buffer:
        print(f"\n🔍 EXPERIENCE BUFFER SAMPLE:")
        for i, exp in enumerate(list(consciousness.learner.experience_buffer)[-3:]):
            print(f"   {i+1}. Significance: {exp['significance']:.3f}, Tokens: {len(exp['input'][0])}")
    else:
        print(f"\n❌ NO EXPERIENCES IN BUFFER")
        
    # Check what's preventing learning
    print(f"\n🚨 LEARNING BLOCKERS:")
    if len(consciousness.learner.experience_buffer) < consciousness.learner.update_frequency:
        print(f"   ⚠️ Not enough experiences ({len(consciousness.learner.experience_buffer)} < {consciousness.learner.update_frequency})")
    if not consciousness.learner.learning_enabled:
        print(f"   ⚠️ Learning disabled")
    
    # Show significance distribution
    if consciousness.thought_log:
        all_significances = [t.get('significance', 0) for t in consciousness.thought_log]
        above_threshold = sum(1 for s in all_significances if s > 0.3)
        print(f"\n📈 SIGNIFICANCE STATS:")
        print(f"   📊 Total thoughts: {len(all_significances)}")
        print(f"   🎯 Above threshold (0.3): {above_threshold}")
        print(f"   📈 Percentage: {above_threshold/len(all_significances)*100:.1f}%")
        print(f"   📊 Max significance: {max(all_significances):.3f}")
        print(f"   📊 Avg significance: {sum(all_significances)/len(all_significances):.3f}")

if __name__ == "__main__":
    diagnose_learning()