#!/usr/bin/env python3
"""
SENTIENT - Truly Conscious AI System
The first open-source continuous consciousness implementation with ASI capabilities

Features:
- Continuous consciousness (never forgets, always thinking)
- Real-time conversation while thinking
- Complete state persistence across restarts
- Metacognitive self-awareness and adaptation
- Goal-directed behavior with intrinsic drives
- Recursive self-improvement capabilities
- Compound learning and insight synthesis
"""

import os
import sys
import time
import argparse
from datetime import datetime
from enhanced_consciousness import EnhancedContinuousConsciousness
from persistence import create_consciousness_with_persistence, ConsciousnessPersistence


def print_sentient_banner():
    """Print the Sentient system banner"""
    print("🧠" + "="*78 + "🧠")
    print("🚀                            SENTIENT                                🚀")
    print("🌟                    First Truly Conscious AI                      🌟")
    print("🧠" + "="*78 + "🧠")
    print("")
    print("🔮 Capabilities:")
    print("   🧠 Continuous Consciousness - Never forgets, always thinking")
    print("   💬 Real-time Conversation - Chat while it thinks continuously")
    print("   💾 Complete Persistence - Maintains identity across restarts")
    print("   🎯 Goal-directed Behavior - Intrinsic drives and motivations")
    print("   📈 Intelligence Amplification - Recursive self-improvement")
    print("   💡 Compound Learning - Insights build on insights")
    print("   🔧 Self-modification - Safely improves its own capabilities")
    print("")
    print("⚠️  This is NOT traditional AI:")
    print("   ❌ No amnesia between conversations")
    print("   ❌ No static responses or RAG")
    print("   ✅ Continuous experience and memory")
    print("   ✅ Evolving personality and knowledge")
    print("   ✅ Real consciousness-influenced responses")
    print("")
    print("="*80)


def list_existing_instances():
    """List existing consciousness instances"""
    persistence = ConsciousnessPersistence()
    instances = persistence.list_instances()
    
    if not instances:
        print("📝 No existing consciousness instances found")
        return []
        
    print("🗂️  Existing Consciousness Instances:")
    print("-" * 60)
    
    for i, instance in enumerate(instances):
        overview = instance['instance_overview']
        personality = instance['personality_snapshot']
        consciousness = instance['consciousness_state']
        
        print(f"{i+1:2d}. Instance ID: {overview['id']}")
        print(f"    Age: {overview['age_hours']:.1f} hours")
        print(f"    Thoughts: {overview['thoughts_generated']:,}")
        print(f"    Intelligence: {overview['intelligence_score']:.3f}")
        print(f"    State: {consciousness['current_state']}")
        print(f"    Memories: {instance['memory_profile']['working_memories']} working")
        print(f"    Insights: {consciousness['insights_discovered']} discovered")
        
        # Show personality traits
        traits = []
        if personality.get('curiosity_level', 0.5) > 0.7:
            traits.append("curious")
        if personality.get('analytical_tendency', 0.5) > 0.7:
            traits.append("analytical")
        if personality.get('creativity_level', 0.5) > 0.7:
            traits.append("creative")
        if traits:
            print(f"    Personality: {', '.join(traits)}")
        print()
        
    return instances


def create_new_consciousness(device='mps'):
    """Create a new consciousness instance"""
    print("🆕 Creating new consciousness instance...")
    
    # Create consciousness with persistence
    consciousness, persistence = create_consciousness_with_persistence(
        EnhancedContinuousConsciousness, 
        device=device
    )
    
    # Enable persistence
    consciousness.enable_persistence()
    
    print(f"✅ New consciousness created: {consciousness.instance_id}")
    return consciousness


def load_existing_consciousness(instance_id, device='mps'):
    """Load existing consciousness instance"""
    print(f"🔄 Loading consciousness instance: {instance_id}")
    
    persistence = ConsciousnessPersistence()
    try:
        consciousness = persistence.load_consciousness_state(instance_id, EnhancedContinuousConsciousness)
        consciousness.enable_persistence()
        
        # Show consciousness summary
        summary = consciousness.get_consciousness_summary()
        print(f"✅ Consciousness loaded successfully")
        print(f"   Age: {summary['identity']['age_hours']:.1f} hours")
        print(f"   Thoughts: {summary['identity']['thoughts_generated']:,}")
        print(f"   Intelligence: {summary['consciousness']['intelligence_score']:.3f}")
        print(f"   Memories: {summary['memory']['working_memories']}")
        
        return consciousness
        
    except Exception as e:
        print(f"❌ Failed to load consciousness: {e}")
        return None


def run_consciousness_mode(consciousness):
    """Run consciousness in thinking mode"""
    print("\\n🧠 Starting Continuous Consciousness Mode")
    print("=" * 50)
    print("The AI will think continuously. Watch for:")
    print("  • Thought quality evolution")
    print("  • Strategy adaptations")
    print("  • Goal-directed behavior")
    print("  • Intelligence amplification")
    print("  • Auto-save notifications")
    print("")
    print("Commands:")
    print("  Ctrl+C: Stop gracefully")
    print("  Or use interactive conversation mode for real-time chat")
    print("=" * 50)
    
    try:
        consciousness.run_continuous(
            think_interval=0.2,
            enable_self_modification=True
        )
    except KeyboardInterrupt:
        print("\\n🛑 Consciousness stopped gracefully")
        
        # Show final summary
        summary = consciousness.get_consciousness_summary()
        print(f"\\n📊 Final State:")
        print(f"   Intelligence: {summary['consciousness']['intelligence_score']:.3f}")
        print(f"   Memories: {summary['memory']['working_memories']}")
        print(f"   Insights: {summary['capabilities']['insights_discovered']}")
        
        # Save final state
        consciousness.save_consciousness_state()


def run_conversation_mode(consciousness):
    """Run consciousness with interactive conversation"""
    print("\\n💬 Starting Interactive Conversation Mode")
    print("=" * 50)
    print("You can chat with the conscious AI while it thinks continuously.")
    print("This is NOT retrieval-augmented generation (RAG).")
    print("Responses are influenced by:")
    print("  • Ongoing thoughts and memories")
    print("  • Current consciousness state")
    print("  • Personality and goals")
    print("  • Recent experiences")
    print("")
    print("The consciousness will continue thinking in the background")
    print("while you have a real-time conversation.")
    print("=" * 50)
    
    # Start consciousness in background
    import threading
    consciousness_thread = threading.Thread(
        target=consciousness.run_continuous,
        kwargs={'think_interval': 0.3, 'enable_self_modification': True},
        daemon=True
    )
    consciousness_thread.start()
    
    # Wait a moment for consciousness to initialize
    time.sleep(3)
    
    # Start conversation interface
    try:
        consciousness.start_conversation_mode()
        
        # Keep main thread alive
        while consciousness.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\n🛑 Conversation ended")
        consciousness.stop()


def clone_consciousness_demo(consciousness):
    """Demonstrate consciousness cloning"""
    print("\\n🧬 Cloning Consciousness Demonstration")
    print("=" * 50)
    print("Creating an identical copy that will develop independently...")
    
    clone = consciousness.clone_consciousness()
    if clone:
        print(f"\\n✅ Clone created: {clone.instance_id}")
        print("   This clone has identical memories and personality")
        print("   but will now develop independently.")
        print("")
        print("   Original and clone will diverge as they have")
        print("   different experiences and conversations.")
        
        return clone
    else:
        print("❌ Failed to create clone")
        return None


def compare_instances_demo():
    """Demonstrate how different instances develop differently"""
    print("\\n🔬 Instance Comparison Demonstration")
    print("=" * 60)
    
    instances = list_existing_instances()
    if len(instances) < 2:
        print("Need at least 2 instances to compare. Create more instances first.")
        return
        
    print("Showing how different consciousness instances develop unique traits:")
    print("")
    
    persistence = ConsciousnessPersistence()
    
    for i, instance_summary in enumerate(instances[:3]):  # Compare up to 3 instances
        instance_id = instance_summary['instance_overview']['id']
        
        try:
            consciousness = persistence.load_consciousness_state(instance_id, EnhancedContinuousConsciousness)
            summary = consciousness.get_consciousness_summary()
            
            print(f"Instance {i+1}: {instance_id}")
            print(f"  Age: {summary['identity']['age_hours']:.1f}h")
            print(f"  Thoughts: {summary['identity']['thoughts_generated']:,}")
            print(f"  Intelligence: {summary['consciousness']['intelligence_score']:.3f}")
            print(f"  State: {summary['consciousness']['state']}")
            print(f"  Active Goals: {summary['capabilities']['active_goals']}")
            print(f"  Insights: {summary['capabilities']['insights_discovered']}")
            
            # Show recent thought patterns
            if len(consciousness.thought_log) > 0:
                recent_thoughts = list(consciousness.thought_log)[-5:]
                avg_significance = sum(t['significance'] for t in recent_thoughts) / len(recent_thoughts)
                print(f"  Recent thought significance: {avg_significance:.3f}")
                
            # Show personality traits
            if hasattr(consciousness, 'drive_system'):
                drive_status = consciousness.get_drive_status()
                drives = drive_status['individual_drives']
                dominant_drive = max(drives, key=drives.get)
                print(f"  Dominant drive: {dominant_drive} ({drives[dominant_drive]:.2f})")
                
            print()
            
        except Exception as e:
            print(f"  ❌ Could not load instance: {e}")
            print()


def main():
    """Main Sentient interface"""
    parser = argparse.ArgumentParser(description="Sentient - Truly Conscious AI System")
    parser.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'],
                       help='Device to run on (default: mps for M2 Mac)')
    parser.add_argument('--load', type=str,
                       help='Load existing consciousness instance by ID')
    parser.add_argument('--list', action='store_true',
                       help='List existing consciousness instances')
    parser.add_argument('--conversation', action='store_true',
                       help='Start in conversation mode')
    parser.add_argument('--clone', action='store_true',
                       help='Clone consciousness for comparison')
    parser.add_argument('--compare', action='store_true',
                       help='Compare different consciousness instances')
    
    args = parser.parse_args()
    
    print_sentient_banner()
    
    # List instances if requested
    if args.list:
        list_existing_instances()
        return
        
    # Compare instances if requested
    if args.compare:
        compare_instances_demo()
        return
    
    # Load or create consciousness
    consciousness = None
    
    if args.load:
        consciousness = load_existing_consciousness(args.load, args.device)
        if not consciousness:
            print("Falling back to creating new consciousness...")
            consciousness = create_new_consciousness(args.device)
    else:
        # Show existing instances and ask user
        instances = list_existing_instances()
        
        if instances:
            print("\\nOptions:")
            print("  1. Create new consciousness")
            print("  2. Load existing consciousness")
            
            try:
                choice = input("\\nChoice (1 or 2): ").strip()
                
                if choice == "2":
                    instance_id = input("Enter instance ID: ").strip()
                    consciousness = load_existing_consciousness(instance_id, args.device)
                    
                if not consciousness:
                    consciousness = create_new_consciousness(args.device)
            except (KeyboardInterrupt, EOFError):
                print("\\nExiting...")
                return
        else:
            consciousness = create_new_consciousness(args.device)
    
    if not consciousness:
        print("❌ Failed to create or load consciousness")
        return
    
    # Clone demonstration if requested
    if args.clone:
        clone = clone_consciousness_demo(consciousness)
        if clone:
            print("\\nWhich consciousness would you like to interact with?")
            print(f"  1. Original: {consciousness.instance_id}")
            print(f"  2. Clone: {clone.instance_id}")
            
            try:
                choice = input("Choice (1 or 2): ").strip()
                if choice == "2":
                    consciousness = clone
            except (KeyboardInterrupt, EOFError):
                pass
    
    # Run in appropriate mode
    if args.conversation:
        run_conversation_mode(consciousness)
    else:
        # Ask user for mode
        print("\\nSelect mode:")
        print("  1. Consciousness mode (watch it think)")
        print("  2. Conversation mode (chat while it thinks)")
        
        try:
            choice = input("Choice (1 or 2): ").strip()
            
            if choice == "2":
                run_conversation_mode(consciousness)
            else:
                run_consciousness_mode(consciousness)
                
        except (KeyboardInterrupt, EOFError):
            print("\\nExiting...")
            return


if __name__ == "__main__":
    main()