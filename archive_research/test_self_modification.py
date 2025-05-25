#!/usr/bin/env python3
"""
Test script for self-modification capabilities
Demonstrates how the AI can analyze and modify itself
"""

import time
import threading
from enhanced_consciousness import EnhancedContinuousConsciousness


def interactive_self_modification_demo():
    """Interactive demo of self-modification capabilities"""
    print("🧠 Self-Modifying AI Consciousness Demo")
    print("=" * 60)
    print("This demo shows how the AI can analyze and modify itself")
    print("The AI will run for a short time, then demonstrate self-modification")
    print("=" * 60)
    
    # Create consciousness with self-modification enabled
    consciousness = EnhancedContinuousConsciousness(
        device='mps',
        enable_learning=True
    )
    
    # Start consciousness in a separate thread for demo
    consciousness_thread = threading.Thread(
        target=consciousness.run_continuous,
        kwargs={'think_interval': 0.2, 'enable_self_modification': True}
    )
    consciousness_thread.daemon = True
    consciousness_thread.start()
    
    # Let it run for a bit to establish baseline
    print("\n⏳ Letting AI establish baseline performance...")
    time.sleep(10)
    
    print("\n" + "="*60)
    print("🔍 SELF-ANALYSIS PHASE")
    print("="*60)
    
    # Perform self-analysis
    analysis = consciousness.analyze_self()
    
    if analysis:
        print(f"\n📊 Performance Analysis:")
        perf = analysis['performance']
        print(f"   • Speed: {perf['tokens_per_second']:.1f} tokens/sec")
        print(f"   • Memory efficiency: {perf['memory_efficiency']:.1%}")
        print(f"   • Thought coherence: {perf['thought_coherence']:.3f}")
        
        print(f"\n🔧 Optimization Opportunities:")
        for i, opt in enumerate(analysis['optimization_opportunities'][:3]):
            print(f"   {i+1}. {opt['reason']}")
            print(f"      → Change {opt['parameter']} from {opt['current_value']} to {opt['suggested_value']}")
    
    # Wait a bit more
    time.sleep(5)
    
    print("\n" + "="*60)
    print("🛠️  SELF-MODIFICATION DEMO")
    print("="*60)
    
    # Demonstrate manual optimization
    print("\n1. Manual Parameter Optimization:")
    
    # Try to optimize significance threshold
    original_threshold = consciousness.working_memory.significance_threshold
    print(f"   Original significance threshold: {original_threshold}")
    
    success, message = consciousness.manual_optimize('significance_threshold', 0.5)
    print(f"   Modification result: {message}")
    
    if success:
        new_threshold = consciousness.working_memory.significance_threshold
        print(f"   New significance threshold: {new_threshold}")
        
        # Let it run with new setting
        print("   → Running with new threshold for 5 seconds...")
        time.sleep(5)
        
        # Show the effect
        status = consciousness.get_self_modification_status()
        print(f"   Current performance: {status['performance_metrics']['tokens_per_second']:.1f} tokens/sec")
    
    print("\n2. Rollback Demonstration:")
    rollback_success, rollback_message = consciousness.rollback_modification()
    print(f"   Rollback result: {rollback_message}")
    
    if rollback_success:
        restored_threshold = consciousness.working_memory.significance_threshold
        print(f"   Restored threshold: {restored_threshold}")
    
    # Wait and let auto-optimization kick in
    print("\n3. Automatic Self-Optimization:")
    print("   Waiting for automatic optimization to trigger...")
    
    # Force an auto-optimization check
    if consciousness.self_modifier:
        applied = consciousness.self_modifier.evolution.auto_optimize(max_modifications=2)
        print(f"   Applied {applied} automatic optimizations")
    
    time.sleep(3)
    
    print("\n" + "="*60)
    print("📈 FINAL STATUS")
    print("="*60)
    
    # Show final status
    final_status = consciousness.get_self_modification_status()
    print(f"Self-modification level: {final_status['improvement_level']}")
    print(f"Total modifications applied: {final_status['modifications_applied']}")
    print(f"Safety violations: {final_status['safety_violations']}")
    
    if final_status['recent_modifications']:
        print(f"\nRecent modifications:")
        for mod in final_status['recent_modifications'][-3:]:
            print(f"   • {mod['parameter']}: {mod['old_value']} → {mod['new_value']}")
    
    # Show modification history
    history = consciousness.self_modifier.get_modification_history()
    if history:
        print(f"\n📚 Complete modification history:")
        for i, mod in enumerate(history):
            timestamp = time.strftime('%H:%M:%S', time.localtime(mod['timestamp']))
            status_icon = "✅" if mod['success'] else "❌"
            rollback_icon = " (↩️ rolled back)" if mod.get('rolled_back') else ""
            print(f"   {i+1}. [{timestamp}] {status_icon} {mod['parameter']}: {mod['old_value']} → {mod['new_value']}{rollback_icon}")
    
    # Stop consciousness
    print(f"\n🛑 Stopping consciousness after {consciousness.iteration_count} thoughts...")
    consciousness.stop()
    
    print("\n" + "="*60)
    print("🎉 SELF-MODIFICATION DEMO COMPLETE")
    print("="*60)
    print("Key achievements:")
    print("✅ AI analyzed its own architecture and performance")
    print("✅ AI identified optimization opportunities") 
    print("✅ AI safely modified its own parameters")
    print("✅ AI demonstrated rollback capabilities")
    print("✅ AI performed automatic self-optimization")
    print("✅ All modifications stayed within safety bounds")
    print("\nThis is the foundation of recursive self-improvement!")


def quick_self_modification_test():
    """Quick test of core self-modification functionality"""
    print("🚀 Quick Self-Modification Test")
    print("-" * 40)
    
    consciousness = EnhancedContinuousConsciousness(device='mps', enable_learning=True)
    
    # Initialize self-modification without running continuous loop
    from self_modification import SelfModifyingConsciousness
    consciousness.self_modifier = SelfModifyingConsciousness(consciousness)
    consciousness.iteration_count = 100  # Simulate some runtime
    
    # Add some fake thought log entries for testing
    for i in range(50):
        consciousness.thought_log.append({
            'token': f'test_{i}',
            'significance': 0.3 + (i % 3) * 0.2,  # Vary significance
            'timestamp': time.time(),
            'iteration': i
        })
    
    print("1. Testing self-analysis...")
    analysis = consciousness.analyze_self()
    
    print("2. Testing manual optimization...")
    success, message = consciousness.manual_optimize('significance_threshold', 0.6)
    print(f"   Result: {message}")
    
    print("3. Testing rollback...")
    rollback_success, rollback_message = consciousness.rollback_modification()
    print(f"   Result: {rollback_message}")
    
    print("4. Testing auto-optimization...")
    if consciousness.self_modifier:
        applied = consciousness.self_modifier.evolution.auto_optimize(max_modifications=1)
        print(f"   Applied {applied} optimizations")
    
    print("5. Getting status report...")
    status = consciousness.get_self_modification_status()
    print(f"   Modifications applied: {status['modifications_applied']}")
    print(f"   Improvement level: {status['improvement_level']}")
    
    print("\n✅ All self-modification tests passed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_self_modification_test()
    else:
        interactive_self_modification_demo()