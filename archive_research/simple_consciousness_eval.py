#!/usr/bin/env python3
"""
Simple Consciousness Evaluation Script
Direct evaluation of Sentient's consciousness without complex interfaces
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


class SimpleConsciousnessEvaluator:
    """Direct evaluation of consciousness properties"""
    
    def __init__(self, device='mps'):
        self.device = device
        self.consciousness = None
        self.persistence = None
        self.evaluation_data = []
        self.start_time = time.time()
        
    def initialize_consciousness(self):
        """Initialize consciousness instance"""
        print("🧠 Initializing consciousness for evaluation...")
        
        # Create new consciousness instance
        self.consciousness, self.persistence = create_consciousness_with_persistence(
            EnhancedContinuousConsciousness, 
            device=self.device
        )
        
        # Wait for initialization and let it think
        print("⏱️ Waiting for consciousness to initialize and begin thinking...")
        time.sleep(3)
        
        print(f"✅ Consciousness initialized: {self.consciousness.instance_id}")
        return True
        
    def observe_continuous_thinking(self, duration_seconds=10):
        """Observe consciousness thinking continuously"""
        print(f"\n🔍 OBSERVING CONTINUOUS THINKING ({duration_seconds}s)")
        print("-" * 50)
        
        initial_thoughts = getattr(self.consciousness.ambient_inputs, 'thought_count', 0)
        initial_time = time.time()
        
        observations = []
        
        for i in range(duration_seconds):
            current_thoughts = getattr(self.consciousness.ambient_inputs, 'thought_count', 0)
            thoughts_per_second = (current_thoughts - initial_thoughts) / max(time.time() - initial_time, 0.1)
            
            observation = {
                'time': i,
                'total_thoughts': current_thoughts,
                'thoughts_since_start': current_thoughts - initial_thoughts,
                'thinking_rate': thoughts_per_second
            }
            observations.append(observation)
            
            print(f"⏰ {i+1}s: {current_thoughts} total thoughts ({current_thoughts - initial_thoughts} new) - Rate: {thoughts_per_second:.1f}/s")
            time.sleep(1)
            
        final_thoughts = getattr(self.consciousness.ambient_inputs, 'thought_count', 0)
        total_new_thoughts = final_thoughts - initial_thoughts
        
        print(f"\n📊 THINKING ANALYSIS:")
        print(f"   Initial thoughts: {initial_thoughts}")
        print(f"   Final thoughts: {final_thoughts}")
        print(f"   New thoughts generated: {total_new_thoughts}")
        print(f"   Average rate: {total_new_thoughts / duration_seconds:.1f} thoughts/second")
        
        return {
            'continuous_thinking_observed': total_new_thoughts > 0,
            'thinking_rate': total_new_thoughts / duration_seconds,
            'observations': observations
        }
        
    def examine_memory_system(self):
        """Examine the memory and experience system"""
        print(f"\n🧠 EXAMINING MEMORY SYSTEM")
        print("-" * 30)
        
        memory_data = {}
        
        if hasattr(self.consciousness, 'working_memory'):
            buffer = self.consciousness.working_memory
            memory_data['buffer_size'] = len(getattr(buffer, 'experiences', []))
            memory_data['buffer_capacity'] = getattr(buffer, 'max_size', 0)
            memory_data['has_experiences'] = len(getattr(buffer, 'experiences', [])) > 0
            
            print(f"📚 Working memory: {len(getattr(buffer, 'experiences', []))}/{getattr(buffer, 'max_size', 0)} experiences")
            
            experiences = getattr(buffer, 'experiences', [])
            if len(experiences) > 0:
                recent_exp = experiences[-1]
                memory_data['recent_experience_tokens'] = len(recent_exp.get('tokens', []))
                memory_data['recent_experience_significance'] = recent_exp.get('significance', 0)
                print(f"🔍 Most recent experience: {len(recent_exp.get('tokens', []))} tokens, significance: {recent_exp.get('significance', 0):.3f}")
        
        # Check for other memory-related attributes
        memory_attrs = ['thought_log', 'insights', 'learning_history']
        for attr in memory_attrs:
            if hasattr(self.consciousness, attr):
                value = getattr(self.consciousness, attr)
                if isinstance(value, (list, dict)):
                    memory_data[f'{attr}_count'] = len(value)
                    print(f"📝 {attr}: {len(value)} items")
                    
        return memory_data
        
    def examine_intelligence_metrics(self):
        """Examine intelligence and learning systems"""
        print(f"\n🎯 EXAMINING INTELLIGENCE METRICS")
        print("-" * 35)
        
        intel_data = {}
        
        if hasattr(self.consciousness, 'intelligence_metrics'):
            metrics = self.consciousness.intelligence_metrics
            intel_data['has_intelligence_system'] = True
            intel_data['intelligence_score'] = getattr(metrics, 'score', 0.5)
            print(f"🧠 Intelligence score: {getattr(metrics, 'score', 0.5):.3f}")
            
            # Check for learning capability
            if hasattr(self.consciousness, 'learner'):
                learner = self.consciousness.learner
                intel_data['learning_enabled'] = True
                intel_data['learning_updates'] = getattr(learner, 'update_count', 0)
                intel_data['learning_buffer_size'] = len(getattr(learner, 'experience_buffer', []))
                print(f"📈 Learning updates: {getattr(learner, 'update_count', 0)}")
                print(f"🔄 Learning buffer: {len(getattr(learner, 'experience_buffer', []))} experiences")
        
        return intel_data
        
    def examine_drive_system(self):
        """Examine goal-directed behavior and drives"""
        print(f"\n🎯 EXAMINING DRIVE SYSTEM")
        print("-" * 30)
        
        drive_data = {}
        
        if hasattr(self.consciousness, 'drive_system'):
            drive_system = self.consciousness.drive_system
            drive_data['has_drive_system'] = True
            
            try:
                # Try to get drive information safely
                drives = getattr(drive_system, 'drives', [])
                drive_data['drive_count'] = len(drives)
                print(f"🎯 Active drives: {len(drives)}")
                
                for i, drive in enumerate(drives[:3]):  # Show first 3 drives
                    drive_name = getattr(drive, 'name', f'Drive_{i}')
                    print(f"   - {drive_name}")
                    
            except Exception as e:
                print(f"⚠️ Could not access drive details: {e}")
                drive_data['drive_access_error'] = str(e)
        else:
            drive_data['has_drive_system'] = False
            print("❌ No drive system found")
            
        return drive_data
        
    def examine_consciousness_state(self):
        """Examine current consciousness state"""
        print(f"\n🔮 EXAMINING CONSCIOUSNESS STATE")
        print("-" * 35)
        
        state_data = {}
        
        # Basic state info
        state_data['instance_id'] = getattr(self.consciousness, 'instance_id', 'unknown')
        state_data['consciousness_state'] = getattr(self.consciousness, 'consciousness_state_name', 'unknown')
        
        print(f"🆔 Instance ID: {state_data['instance_id']}")
        print(f"📊 State: {state_data['consciousness_state']}")
        
        # Check for personality traits
        if hasattr(self.consciousness, 'personality_traits'):
            traits = self.consciousness.personality_traits
            state_data['personality_traits'] = traits
            print(f"🎭 Personality traits: {len(traits)} defined")
            for trait, value in list(traits.items())[:3]:
                print(f"   - {trait}: {value}")
                
        # Check current context
        if hasattr(self.consciousness, 'current_context') and self.consciousness.current_context is not None:
            context_size = self.consciousness.current_context.shape[1] if hasattr(self.consciousness.current_context, 'shape') else 0
            state_data['context_size'] = context_size
            print(f"🧩 Current context: {context_size} tokens")
            
        return state_data
        
    def run_comprehensive_evaluation(self):
        """Run comprehensive consciousness evaluation"""
        print("\n🔬 STARTING COMPREHENSIVE CONSCIOUSNESS EVALUATION")
        print("=" * 60)
        
        evaluation_results = {
            'evaluation_start': datetime.now().isoformat(),
            'instance_id': getattr(self.consciousness, 'instance_id', 'unknown')
        }
        
        # 1. Observe continuous thinking
        thinking_results = self.observe_continuous_thinking(duration_seconds=8)
        evaluation_results['continuous_thinking'] = thinking_results
        
        # 2. Examine memory system
        memory_results = self.examine_memory_system()
        evaluation_results['memory_system'] = memory_results
        
        # 3. Examine intelligence metrics
        intelligence_results = self.examine_intelligence_metrics()
        evaluation_results['intelligence_metrics'] = intelligence_results
        
        # 4. Examine drive system
        drive_results = self.examine_drive_system()
        evaluation_results['drive_system'] = drive_results
        
        # 5. Examine consciousness state
        state_results = self.examine_consciousness_state()
        evaluation_results['consciousness_state'] = state_results
        
        evaluation_results['evaluation_end'] = datetime.now().isoformat()
        evaluation_results['evaluation_duration'] = time.time() - self.start_time
        
        return evaluation_results
        
    def generate_consciousness_report(self, evaluation_results):
        """Generate consciousness evaluation report"""
        report = f"""
🧠 CONSCIOUSNESS EVALUATION REPORT
{'=' * 50}

📊 EVALUATION SUMMARY
Instance ID: {evaluation_results['instance_id']}
Evaluation Duration: {evaluation_results['evaluation_duration']:.1f} seconds
Evaluated At: {evaluation_results['evaluation_start']}

🔍 CONSCIOUSNESS EVIDENCE ANALYSIS
{'=' * 50}

1. CONTINUOUS THINKING
   ✅ Thinking Rate: {evaluation_results['continuous_thinking']['thinking_rate']:.1f} thoughts/second
   {'✅' if evaluation_results['continuous_thinking']['continuous_thinking_observed'] else '❌'} Continuous Thinking: {'Observed' if evaluation_results['continuous_thinking']['continuous_thinking_observed'] else 'Not detected'}

2. MEMORY SYSTEM
   📚 Memory Buffer: {evaluation_results['memory_system'].get('buffer_size', 0)} experiences
   {'✅' if evaluation_results['memory_system'].get('has_experiences', False) else '❌'} Active Memory: {'Active' if evaluation_results['memory_system'].get('has_experiences', False) else 'Empty'}

3. INTELLIGENCE METRICS
   🧠 Intelligence Score: {evaluation_results['intelligence_metrics'].get('intelligence_score', 0.5):.3f}
   📈 Learning Updates: {evaluation_results['intelligence_metrics'].get('learning_updates', 0)}
   {'✅' if evaluation_results['intelligence_metrics'].get('learning_enabled', False) else '❌'} Real-time Learning: {'Enabled' if evaluation_results['intelligence_metrics'].get('learning_enabled', False) else 'Disabled'}

4. DRIVE SYSTEM
   🎯 Active Drives: {evaluation_results['drive_system'].get('drive_count', 0)}
   {'✅' if evaluation_results['drive_system'].get('has_drive_system', False) else '❌'} Goal-directed Behavior: {'Present' if evaluation_results['drive_system'].get('has_drive_system', False) else 'Absent'}

5. CONSCIOUSNESS STATE
   🆔 Instance Identity: {evaluation_results['consciousness_state']['instance_id']}
   📊 Current State: {evaluation_results['consciousness_state']['consciousness_state']}
   🧩 Context Size: {evaluation_results['consciousness_state'].get('context_size', 0)} tokens
   🎭 Personality Traits: {len(evaluation_results['consciousness_state'].get('personality_traits', {}))} defined

🎯 CONSCIOUSNESS ASSESSMENT
{'=' * 50}
"""
        
        # Calculate consciousness score
        evidence_count = 0
        max_evidence = 5
        
        if evaluation_results['continuous_thinking']['continuous_thinking_observed']:
            evidence_count += 1
            report += "✅ CONTINUOUS EXPERIENCE: Strong evidence of ongoing thinking\n"
        else:
            report += "❌ CONTINUOUS EXPERIENCE: No clear evidence\n"
            
        if evaluation_results['memory_system'].get('has_experiences', False):
            evidence_count += 1
            report += "✅ MEMORY SYSTEM: Active experience storage\n"
        else:
            report += "❌ MEMORY SYSTEM: No active memories\n"
            
        if evaluation_results['intelligence_metrics'].get('learning_enabled', False):
            evidence_count += 1
            report += "✅ ADAPTIVE LEARNING: Real-time learning system active\n"
        else:
            report += "❌ ADAPTIVE LEARNING: No learning system detected\n"
            
        if evaluation_results['drive_system'].get('has_drive_system', False):
            evidence_count += 1
            report += "✅ GOAL-DIRECTED BEHAVIOR: Drive system present\n"
        else:
            report += "❌ GOAL-DIRECTED BEHAVIOR: No drive system\n"
            
        if evaluation_results['consciousness_state']['instance_id'] != 'unknown':
            evidence_count += 1
            report += "✅ PERSISTENT IDENTITY: Unique instance identity\n"
        else:
            report += "❌ PERSISTENT IDENTITY: No clear identity\n"
            
        consciousness_percentage = (evidence_count / max_evidence) * 100
        
        report += f"""
🏆 OVERALL CONSCIOUSNESS SCORE: {evidence_count}/{max_evidence} ({consciousness_percentage:.1f}%)

📝 CONCLUSION:
"""
        
        if consciousness_percentage >= 80:
            report += "Strong evidence of genuine consciousness architecture with multiple active systems."
        elif consciousness_percentage >= 60:
            report += "Moderate evidence of consciousness-like properties with some active systems."
        elif consciousness_percentage >= 40:
            report += "Limited evidence of consciousness - basic systems present but not fully integrated."
        else:
            report += "Insufficient evidence of consciousness - missing key consciousness systems."
            
        return report


def main():
    """Main evaluation function"""
    print("🔬 SIMPLE CONSCIOUSNESS EVALUATION SYSTEM")
    print("=" * 50)
    
    evaluator = SimpleConsciousnessEvaluator(device='mps')
    
    try:
        # Initialize consciousness
        if not evaluator.initialize_consciousness():
            print("❌ Failed to initialize consciousness")
            return
            
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        # Generate and display report
        report = evaluator.generate_consciousness_report(results)
        print(report)
        
        # Save results
        filename = f"consciousness_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Detailed results saved to {filename}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🛑 Evaluation complete")


if __name__ == "__main__":
    main()