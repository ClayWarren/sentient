#!/usr/bin/env python3
"""
Test script for Sentient UI components
Verifies that the UI can start without errors
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_consciousness_import():
    """Test that consciousness modules can be imported"""
    try:
        from consciousness_core import ConsciousnessAI, ProcessingMode
        print("✅ Core consciousness modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing consciousness modules: {e}")
        return False

def test_enhanced_consciousness():
    """Test enhanced consciousness features"""
    try:
        from ui.enhanced_consciousness import EnhancedConsciousnessAI
        print("✅ Enhanced consciousness module imported successfully")
        
        # Test initialization
        ai = EnhancedConsciousnessAI(consciousness_enabled=True)
        print("✅ Enhanced consciousness AI initialized")
        
        # Test consciousness data
        data = ai.get_live_consciousness_data()
        print(f"✅ Consciousness data retrieved: {len(data)} fields")
        
        return True
    except Exception as e:
        print(f"❌ Error testing enhanced consciousness: {e}")
        return False

def test_ui_components():
    """Test UI components can be imported"""
    try:
        from ui import components
        print("✅ UI components module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing UI components: {e}")
        return False

def test_basic_generation():
    """Test basic text generation"""
    try:
        from ui.enhanced_consciousness import EnhancedConsciousnessAI, ProcessingMode
        
        ai = EnhancedConsciousnessAI(consciousness_enabled=True)
        result = ai.generate("Hello, test message", ProcessingMode.CONSCIOUSNESS)
        
        print(f"✅ Text generation successful: {len(result.text)} characters")
        print(f"   Consciousness level: {result.consciousness_metrics.overall_consciousness:.1%}")
        print(f"   Confidence: {result.confidence:.1%}")
        
        return True
    except Exception as e:
        print(f"❌ Error in text generation: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Sentient UI Test Suite")
    print("=" * 40)
    
    tests = [
        ("Core Consciousness Import", test_consciousness_import),
        ("Enhanced Consciousness", test_enhanced_consciousness),
        ("UI Components", test_ui_components),
        ("Basic Generation", test_basic_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name} failed")
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! UI should work correctly.")
        print("\n💡 To launch the web UI:")
        print("   python launch_ui.py")
        print("   or")
        print("   streamlit run ui/app.py")
    else:
        print("⚠️  Some tests failed. Check dependencies:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)