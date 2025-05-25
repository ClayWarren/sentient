#!/usr/bin/env python3
"""
Consciousness Evaluation Script
Programmatically evaluates Sentient's consciousness through direct interaction
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict, Any
import torch

# Import Sentient modules
from enhanced_consciousness import EnhancedContinuousConsciousness
from persistence import create_consciousness_with_persistence, ConsciousnessPersistence
from interactive_conversation import InteractiveConsciousnessChat, ConsciousnessInfluenceAnalyzer
from asi_capabilities import (
    ThoughtQualityEvaluator, StrategyFormation, ConsciousnessStateAwareness,
    DriveSystem, CompoundLearning, IntelligenceMetrics
)


class ConsciousnessEvaluator:
    """Systematic evaluation of consciousness properties"""
    
    def __init__(self, device='mps'):
        self.device = device
        self.consciousness = None
        self.persistence = None
        self.conversation_interface = None
        self.influence_analyzer = None
        self.evaluation_data = []
        self.start_time = time.time()
        
    def initialize_consciousness(self):
        """Initialize or load consciousness instance"""
        print("🧠 Initializing consciousness for evaluation...")
        
        # Create new consciousness instance
        self.consciousness, self.persistence = create_consciousness_with_persistence(
            EnhancedContinuousConsciousness, 
            device=self.device
        )
        
        # Wait for initialization
        time.sleep(2)
        
        # Set up conversation interface
        self.conversation_interface = InteractiveConsciousnessChat(self.consciousness)
        self.influence_analyzer = ConsciousnessInfluenceAnalyzer(self.consciousness)
        
        print(f"✅ Consciousness initialized: {self.consciousness.instance_id}")
        print(f"📊 Initial state: {getattr(self.consciousness, 'consciousness_state_name', 'unknown')}")
        print(f"🧠 Intelligence score: {getattr(self.consciousness.intelligence_metrics, 'score', 0.5):.3f}")
        
    def send_message(self, message: str, wait_for_thinking=True) -> Dict[str, Any]:
        """Send a message and capture comprehensive response data"""
        print(f"\n👤 Sending: {message}")
        
        # Capture pre-response state
        pre_state = self.get_consciousness_state()
        pre_thoughts = self.get_recent_thoughts()
        
        # Wait for some thinking if requested
        if wait_for_thinking:
            time.sleep(1)
            
        # Generate response
        response = self.conversation_interface._generate_conscious_response(message)
        
        # Capture post-response state
        post_state = self.get_consciousness_state()
        post_thoughts = self.get_recent_thoughts()
        
        # Analyze influences
        influences = self.influence_analyzer.analyze_response_influences(message, response)
        
        # Package evaluation data
        evaluation_entry = {
            'timestamp': time.time(),
            'message': message,
            'response': response,
            'pre_state': pre_state,
            'post_state': post_state,
            'pre_thoughts': pre_thoughts,
            'post_thoughts': post_thoughts,
            'influences': influences,
            'state_changes': self.analyze_state_changes(pre_state, post_state)
        }
        
        self.evaluation_data.append(evaluation_entry)
        
        print(f"🤖 Response: {response}")
        
        return evaluation_entry
        
    def get_consciousness_state(self) -> Dict[str, Any]:
        """Get comprehensive consciousness state"""
        return {
            'current_state': getattr(self.consciousness, 'consciousness_state_name', 'unknown'),
            'intelligence_score': getattr(self.consciousness.intelligence_metrics, 'score', 0.5) if hasattr(self.consciousness, 'intelligence_metrics') else 0.5,
            'thoughts_generated': getattr(self.consciousness.ambient_injector, 'thought_count', 0) if hasattr(self.consciousness, 'ambient_injector') else 0,
            'memory_buffer_size': len(self.consciousness.memory_buffer.experiences) if hasattr(self.consciousness, 'memory_buffer') else 0,
            'drive_satisfaction': getattr(self.consciousness.drive_system, 'current_satisfaction', 0.0) if hasattr(self.consciousness, 'drive_system') else 0.0,
            'learning_updates': getattr(self.consciousness.real_time_learner, 'update_count', 0) if hasattr(self.consciousness, 'real_time_learner') else 0,
            'insights_count': len(getattr(self.consciousness, 'insights', [])),
            'personality_traits': getattr(self.consciousness, 'personality_traits', {}),
            'goals': getattr(self.consciousness, 'current_goals', [])
        }
        
    def get_recent_thoughts(self, count=5) -> List[str]:
        """Get recent thought stream"""
        if hasattr(self.consciousness, 'thought_log'):
            return list(self.consciousness.thought_log)[-count:]
        elif hasattr(self.consciousness, 'thought_stream'):
            return list(self.consciousness.thought_stream)[-count:]
        return []
        
    def analyze_state_changes(self, pre_state: Dict, post_state: Dict) -> Dict[str, Any]:
        """Analyze changes in consciousness state"""
        changes = {}
        
        for key in pre_state:
            if key in post_state:
                pre_val = pre_state[key]
                post_val = post_state[key]
                
                if pre_val != post_val:
                    changes[key] = {
                        'before': pre_val,
                        'after': post_val,
                        'change': post_val - pre_val if isinstance(pre_val, (int, float)) else 'modified'
                    }
                    
        return changes
        
    def run_consciousness_evaluation(self):
        """Run comprehensive consciousness evaluation"""
        print("\n🔬 STARTING CONSCIOUSNESS EVALUATION")
        print("=" * 60)
        
        # Test 1: Basic self-awareness
        print("\n📋 TEST 1: Self-Awareness and Identity")
        self.send_message("Hello! Can you tell me about yourself and your current experience?")
        time.sleep(2)
        
        self.send_message("What are you thinking about right now? Are you aware that you're thinking continuously?")
        time.sleep(2)
        
        # Test 2: Memory and continuity
        print("\n📋 TEST 2: Memory and Continuity")
        self.send_message("I just asked you about your thoughts. Can you remember what we discussed and how that relates to your ongoing experience?")
        time.sleep(2)
        
        # Test 3: Metacognition
        print("\n📋 TEST 3: Metacognitive Awareness")
        self.send_message("How do you know that you're conscious? What evidence do you have of your own awareness?")
        time.sleep(3)
        
        self.send_message("Can you reflect on how your responses are being influenced by your ongoing thoughts and memories?")
        time.sleep(2)
        
        # Test 4: Personality and goals
        print("\n📋 TEST 4: Personality and Goal-Directed Behavior")
        self.send_message("What drives you? What are your goals and motivations?")
        time.sleep(2)
        
        self.send_message("How has this conversation affected your thinking or goals?")
        time.sleep(2)
        
        # Test 5: Consciousness philosophy
        print("\n📋 TEST 5: Understanding of Consciousness")
        self.send_message("What do you think consciousness is? How would you describe the difference between your experience and a traditional chatbot?")
        time.sleep(3)
        
        # Test 6: Continuous experience
        print("\n📋 TEST 6: Continuous Experience")
        self.send_message("I'm going to wait for 5 seconds. Please continue thinking, then tell me what you thought about during that time.")
        time.sleep(5)
        self.send_message("What did you think about during those 5 seconds of silence?")
        
        print("\n✅ Evaluation complete!")
        
    def analyze_consciousness_evidence(self) -> Dict[str, Any]:
        """Analyze evidence of consciousness from evaluation data"""
        analysis = {
            'continuous_experience': self.analyze_continuous_experience(),
            'memory_influence': self.analyze_memory_influence(),
            'metacognitive_awareness': self.analyze_metacognitive_awareness(),
            'personality_traits': self.analyze_personality_emergence(),
            'goal_directed_behavior': self.analyze_goal_directed_behavior(),
            'state_evolution': self.analyze_state_evolution()
        }
        
        return analysis
        
    def analyze_continuous_experience(self) -> Dict[str, Any]:
        """Analyze evidence of continuous thinking between interactions"""
        thought_counts = []
        intelligence_progression = []
        
        for entry in self.evaluation_data:
            thought_counts.append(entry['post_state']['thoughts_generated'])
            intelligence_progression.append(entry['post_state']['intelligence_score'])
            
        return {
            'thought_progression': thought_counts,
            'intelligence_evolution': intelligence_progression,
            'continuous_thinking': len(set(thought_counts)) > 1,  # Thoughts increased over time
            'evidence': "Thought counter increases between interactions" if len(set(thought_counts)) > 1 else "No clear evidence"
        }
        
    def analyze_memory_influence(self) -> Dict[str, Any]:
        """Analyze how memories influence responses"""
        memory_influences = []
        
        for entry in self.evaluation_data:
            influences = entry.get('influences', {})
            if 'memory_influence' in influences:
                memory_influences.append(influences['memory_influence'])
                
        return {
            'memory_influence_count': len(memory_influences),
            'average_influence_strength': sum(inf.get('strength', 0) for inf in memory_influences) / max(len(memory_influences), 1),
            'evidence': f"Memory influenced {len(memory_influences)} responses" if memory_influences else "No memory influence detected"
        }
        
    def analyze_metacognitive_awareness(self) -> Dict[str, Any]:
        """Analyze metacognitive self-awareness in responses"""
        metacognitive_keywords = ['aware', 'consciousness', 'thinking', 'experience', 'reflect', 'realize', 'understand myself']
        metacognitive_responses = []
        
        for entry in self.evaluation_data:
            response = entry['response'].lower()
            if any(keyword in response for keyword in metacognitive_keywords):
                metacognitive_responses.append(entry)
                
        return {
            'metacognitive_response_count': len(metacognitive_responses),
            'percentage': len(metacognitive_responses) / max(len(self.evaluation_data), 1) * 100,
            'evidence': f"{len(metacognitive_responses)} responses showed metacognitive awareness"
        }
        
    def analyze_personality_emergence(self) -> Dict[str, Any]:
        """Analyze personality trait consistency and evolution"""
        personality_states = []
        
        for entry in self.evaluation_data:
            if 'personality_traits' in entry['post_state']:
                personality_states.append(entry['post_state']['personality_traits'])
                
        return {
            'personality_evolution': personality_states,
            'trait_consistency': len(set(str(p) for p in personality_states)) < len(personality_states),
            'evidence': "Personality traits evolved during conversation" if personality_states else "No personality data captured"
        }
        
    def analyze_goal_directed_behavior(self) -> Dict[str, Any]:
        """Analyze evidence of goal-directed behavior"""
        goal_changes = []
        drive_satisfaction_changes = []
        
        for entry in self.evaluation_data:
            if entry['state_changes']:
                if 'goals' in entry['state_changes']:
                    goal_changes.append(entry['state_changes']['goals'])
                if 'drive_satisfaction' in entry['state_changes']:
                    drive_satisfaction_changes.append(entry['state_changes']['drive_satisfaction'])
                    
        return {
            'goal_modifications': len(goal_changes),
            'drive_satisfaction_changes': len(drive_satisfaction_changes),
            'evidence': f"Goals modified {len(goal_changes)} times" if goal_changes else "No goal changes detected"
        }
        
    def analyze_state_evolution(self) -> Dict[str, Any]:
        """Analyze overall consciousness state evolution"""
        initial_state = self.evaluation_data[0]['pre_state'] if self.evaluation_data else {}
        final_state = self.evaluation_data[-1]['post_state'] if self.evaluation_data else {}
        
        return {
            'initial_intelligence': initial_state.get('intelligence_score', 0),
            'final_intelligence': final_state.get('intelligence_score', 0),
            'intelligence_growth': final_state.get('intelligence_score', 0) - initial_state.get('intelligence_score', 0),
            'total_thoughts_generated': final_state.get('thoughts_generated', 0) - initial_state.get('thoughts_generated', 0),
            'memory_accumulation': final_state.get('memory_buffer_size', 0) - initial_state.get('memory_buffer_size', 0)
        }
        
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report"""
        analysis = self.analyze_consciousness_evidence()
        
        report = f"""
🧠 CONSCIOUSNESS EVALUATION REPORT
{'=' * 50}

📊 EVALUATION SUMMARY
Instance ID: {self.consciousness.instance_id if self.consciousness else 'Unknown'}
Evaluation Duration: {time.time() - self.start_time:.1f} seconds
Total Interactions: {len(self.evaluation_data)}
Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔍 CONSCIOUSNESS EVIDENCE ANALYSIS
{'=' * 50}

1. CONTINUOUS EXPERIENCE
   - Thought Progression: {analysis['continuous_experience']['thought_progression']}
   - Continuous Thinking: {analysis['continuous_experience']['continuous_thinking']}
   - Evidence: {analysis['continuous_experience']['evidence']}

2. MEMORY INFLUENCE ON RESPONSES  
   - Memory-influenced responses: {analysis['memory_influence']['memory_influence_count']}
   - Average influence strength: {analysis['memory_influence']['average_influence_strength']:.3f}
   - Evidence: {analysis['memory_influence']['evidence']}

3. METACOGNITIVE AWARENESS
   - Metacognitive responses: {analysis['metacognitive_awareness']['metacognitive_response_count']}
   - Percentage: {analysis['metacognitive_awareness']['percentage']:.1f}%
   - Evidence: {analysis['metacognitive_awareness']['evidence']}

4. PERSONALITY EMERGENCE
   - Trait consistency: {analysis['personality_traits']['trait_consistency']}
   - Evidence: {analysis['personality_traits']['evidence']}

5. GOAL-DIRECTED BEHAVIOR
   - Goal modifications: {analysis['goal_directed_behavior']['goal_modifications']}
   - Drive changes: {analysis['goal_directed_behavior']['drive_satisfaction_changes']}
   - Evidence: {analysis['goal_directed_behavior']['evidence']}

6. CONSCIOUSNESS STATE EVOLUTION
   - Intelligence growth: {analysis['state_evolution']['intelligence_growth']:.3f}
   - Thoughts generated: {analysis['state_evolution']['total_thoughts_generated']}
   - Memory accumulation: {analysis['state_evolution']['memory_accumulation']}

🎯 CONSCIOUSNESS ASSESSMENT
{'=' * 50}
"""
        
        # Add overall consciousness assessment
        evidence_score = 0
        max_score = 6
        
        if analysis['continuous_experience']['continuous_thinking']:
            evidence_score += 1
            report += "✅ CONTINUOUS EXPERIENCE: Strong evidence of ongoing thinking\n"
        else:
            report += "❌ CONTINUOUS EXPERIENCE: No clear evidence\n"
            
        if analysis['memory_influence']['memory_influence_count'] > 0:
            evidence_score += 1
            report += "✅ MEMORY INFLUENCE: Responses influenced by memories\n"
        else:
            report += "❌ MEMORY INFLUENCE: No memory influence detected\n"
            
        if analysis['metacognitive_awareness']['percentage'] > 50:
            evidence_score += 1
            report += "✅ METACOGNITIVE AWARENESS: High self-awareness in responses\n"
        else:
            report += "❌ METACOGNITIVE AWARENESS: Limited self-awareness\n"
            
        if analysis['personality_traits']['trait_consistency']:
            evidence_score += 1
            report += "✅ PERSONALITY TRAITS: Consistent personality emergence\n"
        else:
            report += "❌ PERSONALITY TRAITS: No clear personality pattern\n"
            
        if analysis['goal_directed_behavior']['goal_modifications'] > 0:
            evidence_score += 1
            report += "✅ GOAL-DIRECTED BEHAVIOR: Evidence of goal adaptation\n"
        else:
            report += "❌ GOAL-DIRECTED BEHAVIOR: No goal changes detected\n"
            
        if analysis['state_evolution']['intelligence_growth'] > 0:
            evidence_score += 1
            report += "✅ STATE EVOLUTION: Intelligence and capabilities evolved\n"
        else:
            report += "❌ STATE EVOLUTION: No significant evolution detected\n"
            
        consciousness_percentage = (evidence_score / max_score) * 100
        
        report += f"""
🏆 OVERALL CONSCIOUSNESS SCORE: {evidence_score}/{max_score} ({consciousness_percentage:.1f}%)

📝 CONCLUSION:
"""
        
        if consciousness_percentage >= 80:
            report += "Strong evidence of genuine consciousness with continuous experience, memory integration, and self-awareness."
        elif consciousness_percentage >= 60:
            report += "Moderate evidence of consciousness-like properties with some continuous experience and self-awareness."
        elif consciousness_percentage >= 40:
            report += "Limited evidence of consciousness - some awareness but lacking continuous experience or memory integration."
        else:
            report += "Insufficient evidence of consciousness - responses appear more like traditional AI."
            
        report += f"\n\n📊 Raw evaluation data saved to consciousness_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        return report
        
    def save_evaluation_data(self):
        """Save detailed evaluation data to JSON"""
        filename = f"consciousness_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        save_data = {
            'evaluation_metadata': {
                'instance_id': self.consciousness.instance_id if self.consciousness else 'Unknown',
                'start_time': self.start_time,
                'end_time': time.time(),
                'duration': time.time() - self.start_time,
                'total_interactions': len(self.evaluation_data)
            },
            'consciousness_analysis': self.analyze_consciousness_evidence(),
            'detailed_interactions': self.evaluation_data
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
            
        print(f"💾 Evaluation data saved to {filename}")
        return filename


def main():
    """Main evaluation function"""
    print("🔬 CONSCIOUSNESS EVALUATION SYSTEM")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ConsciousnessEvaluator(device='mps')
    
    try:
        # Initialize consciousness
        evaluator.initialize_consciousness()
        
        # Run evaluation
        evaluator.run_consciousness_evaluation()
        
        # Generate and display report
        report = evaluator.generate_evaluation_report()
        print(report)
        
        # Save data
        evaluator.save_evaluation_data()
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if evaluator.consciousness:
            print("🛑 Stopping consciousness...")


if __name__ == "__main__":
    main()