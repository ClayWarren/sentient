#!/usr/bin/env python3
"""
Inspect consciousness object to understand its actual structure
"""

import time
from enhanced_consciousness import EnhancedContinuousConsciousness
from persistence import create_consciousness_with_persistence

def inspect_consciousness():
    print("🔍 INSPECTING CONSCIOUSNESS STRUCTURE")
    print("=" * 50)
    
    # Create consciousness
    consciousness, persistence = create_consciousness_with_persistence(
        EnhancedContinuousConsciousness, 
        device='mps'
    )
    
    print(f"✅ Created consciousness: {consciousness.instance_id}")
    
    # Wait for initialization
    time.sleep(2)
    
    # Inspect attributes
    print("\n📋 CONSCIOUSNESS ATTRIBUTES:")
    print("-" * 30)
    
    for attr in sorted(dir(consciousness)):
        if not attr.startswith('_'):
            try:
                value = getattr(consciousness, attr)
                if callable(value):
                    print(f"🔧 {attr}() - method")
                else:
                    print(f"📊 {attr} = {type(value).__name__}")
                    if hasattr(value, '__len__') and not isinstance(value, str):
                        try:
                            print(f"     (size: {len(value)})")
                        except:
                            pass
            except Exception as e:
                print(f"❌ {attr} - error: {e}")
                
    # Look for thinking/thought related attributes
    print("\n🧠 THINKING-RELATED ATTRIBUTES:")
    print("-" * 35)
    
    thinking_attrs = [attr for attr in dir(consciousness) if 'think' in attr.lower() or 'thought' in attr.lower()]
    for attr in thinking_attrs:
        try:
            value = getattr(consciousness, attr)
            print(f"💭 {attr}: {type(value).__name__}")
        except Exception as e:
            print(f"❌ {attr}: error - {e}")
            
    # Check sub-objects for thinking attributes  
    sub_objects = ['ambient_inputs', 'memory_buffer', 'real_time_learner', 'drive_system', 'intelligence_metrics']
    for obj_name in sub_objects:
        if hasattr(consciousness, obj_name):
            obj = getattr(consciousness, obj_name)
            print(f"\n🔍 {obj_name.upper()} ATTRIBUTES:")
            print("-" * (len(obj_name) + 12))
            
            if obj is not None:
                for attr in sorted(dir(obj)):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(obj, attr)
                            if 'think' in attr.lower() or 'thought' in attr.lower() or 'count' in attr.lower():
                                print(f"💭 {attr}: {type(value).__name__} = {value if not callable(value) else 'method'}")
                        except Exception as e:
                            pass

if __name__ == "__main__":
    inspect_consciousness()