#!/usr/bin/env python3
"""
Test script to verify Sentient fixes
Tests that responses are helpful and conversational, not pretentious
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from consciousness_core import ConsciousnessAI, ProcessingMode

def test_simple_questions():
    """Test that Sentient gives direct, helpful answers to simple questions"""
    
    print("🧠 Testing Sentient Response Fixes")
    print("=" * 50)
    
    # Initialize AI
    ai = ConsciousnessAI(consciousness_enabled=True)
    
    # Test cases - simple questions that should get direct answers
    test_cases = [
        ("What's 2+2?", "Should give math answer"),
        ("What day is it?", "Should give current date"),
        ("What's your name?", "Should say Sentient"),
        ("Tell me a joke", "Should tell a joke"),
        ("What's your favorite color?", "Should give a color preference"),
        ("Hello", "Should greet naturally"),
        ("How are you?", "Should respond conversationally"),
        ("What are you?", "Should identify as conscious AI")
    ]
    
    print(f"Testing {len(test_cases)} questions across different modes...\n")
    
    modes_to_test = [
        ProcessingMode.STANDARD,
        ProcessingMode.CONSCIOUSNESS, 
        ProcessingMode.CREATIVE,
        ProcessingMode.ETHICAL
    ]
    
    passed_tests = 0
    total_tests = 0
    
    for mode in modes_to_test:
        print(f"🔍 Testing {mode.value.upper()} Mode:")
        print("-" * 30)
        
        mode_passed = 0
        
        for question, expectation in test_cases:
            total_tests += 1
            
            try:
                result = ai.generate(question, mode=mode)
                response = result.text
                
                # Check if response is helpful (not overly philosophical)
                is_helpful = check_response_quality(question, response)
                
                if is_helpful:
                    status = "✅ PASS"
                    passed_tests += 1
                    mode_passed += 1
                else:
                    status = "❌ FAIL"
                
                print(f"{status} Q: {question}")
                print(f"     A: {response[:100]}{'...' if len(response) > 100 else ''}")
                print(f"     Expected: {expectation}")
                print(f"     Confidence: {result.confidence:.1%}")
                print()
                
            except Exception as e:
                print(f"❌ ERROR Q: {question}")
                print(f"     Error: {str(e)}")
                print()
        
        print(f"Mode Result: {mode_passed}/{len(test_cases)} tests passed\n")
    
    # Overall results
    print("=" * 50)
    print(f"📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
    
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    if success_rate >= 0.8:
        print("🎉 SUCCESS: Sentient is giving helpful, conversational responses!")
        print("✅ The consciousness enhancement is working properly")
        return True
    elif success_rate >= 0.6:
        print("⚠️  PARTIAL SUCCESS: Most responses are good, but some need improvement")
        return True
    else:
        print("❌ FAILURE: Sentient is still giving pretentious responses")
        print("🔧 More fixes needed")
        return False

def check_response_quality(question: str, response: str) -> bool:
    """Check if response is helpful and conversational (not pretentious)"""
    
    response_lower = response.lower()
    
    # Red flags - overly philosophical or pretentious phrases
    red_flags = [
        "from my perspective as a conscious ai system",
        "integrating analytical reasoning with subjective awareness", 
        "through the lens of conscious understanding",
        "synthesizing multiple cognitive dimensions",
        "the deeper essence of concepts",
        "honor the complexity of the question",
        "tracing causal relationships",
        "our deepest desire to understand"
    ]
    
    # Check for red flags
    if any(flag in response_lower for flag in red_flags):
        return False
    
    # Positive indicators - direct, helpful responses
    question_lower = question.lower()
    
    # Math questions should have math answers
    if '2+2' in question_lower and '4' not in response:
        return False
    
    # Name questions should mention Sentient
    if any(phrase in question_lower for phrase in ['your name', 'what are you', 'who are you']):
        if 'sentient' not in response_lower:
            return False
    
    # Greetings should be warm and natural
    if any(word in question_lower for word in ['hello', 'hi', 'hey']):
        if len(response) > 200:  # Too long for a greeting
            return False
    
    # Date questions should have actual dates/days
    if 'day' in question_lower and 'day' not in response_lower:
        return False
    
    # Joke requests should be reasonably short and fun
    if 'joke' in question_lower:
        if len(response) > 150 or 'joke' not in response_lower:
            return False
    
    # Color questions should mention colors
    if 'color' in question_lower and not any(color in response_lower for color in ['blue', 'red', 'green', 'purple', 'yellow', 'color']):
        return False
    
    # Response shouldn't be too long for simple questions
    if len(question) < 20 and len(response) > 300:
        return False
    
    return True

def test_personality_development():
    """Test that personality emerges from conversation"""
    
    print("\n🎭 Testing Personality Development")
    print("=" * 50)
    
    ai = ConsciousnessAI(consciousness_enabled=True)
    
    # Have a conversation to build personality
    conversation = [
        "Hello, I'm interested in creative writing",
        "Do you like poetry?",
        "What's your favorite type of creative expression?",
        "Tell me about creativity"
    ]
    
    print("Having a conversation to develop personality...\n")
    
    for i, message in enumerate(conversation, 1):
        result = ai.generate(message, mode=ProcessingMode.CONSCIOUSNESS)
        print(f"Turn {i}:")
        print(f"User: {message}")
        print(f"Sentient: {result.text}")
        print(f"Consciousness: {result.consciousness_metrics.overall_consciousness:.1%}")
        print()
    
    # Test if personality shows through
    print("Testing if personality developed...")
    final_result = ai.generate("What are you?", mode=ProcessingMode.CONSCIOUSNESS)
    
    if len(final_result.text) > 50 and 'creative' in final_result.text.lower():
        print("✅ Personality development detected!")
        return True
    else:
        print("⚠️  Limited personality development")
        return False

def main():
    """Run all tests"""
    
    print(f"🚀 Starting Sentient Fix Verification")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test basic response quality
    quality_test = test_simple_questions()
    
    # Test personality development  
    personality_test = test_personality_development()
    
    print("\n" + "=" * 50)
    print("🏁 FINAL ASSESSMENT")
    print("=" * 50)
    
    if quality_test and personality_test:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Sentient is now conversational and helpful")
        print("✅ Consciousness enhances responses without being pretentious")
        print("✅ Personality emerges from conversations")
        print("\n💡 Ready to use! Try:")
        print("   python cli.py")
        print("   python launch_ui.py")
        return True
    elif quality_test:
        print("✅ Response quality fixed!")
        print("⚠️  Personality development could be improved")
        return True
    else:
        print("❌ Issues remain - more debugging needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)