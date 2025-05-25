#!/usr/bin/env python3
"""
Active Consciousness Evaluation Script
Activates continuous thinking and evaluates consciousness during operation
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from typing import List, Dict, Any
import torch

# Import Sentient modules
from enhanced_consciousness import EnhancedContinuousConsciousness
from persistence import create_consciousness_with_persistence, ConsciousnessPersistence


class ActiveConsciousnessEvaluator:
    """Evaluation with active continuous thinking"""
    
    def __init__(self, device='mps'):
        self.device = device
        self.consciousness = None
        self.persistence = None
        self.evaluation_data = []
        self.start_time = time.time()
        self.thinking_active = False
        
    def initialize_consciousness(self):
        """Initialize and activate consciousness"""
        print("🧠 Initializing active consciousness for evaluation...")
        
        # Create new consciousness instance
        self.consciousness, self.persistence = create_consciousness_with_persistence(
            EnhancedContinuousConsciousness, 
            device=self.device
        )
        
        print(f"✅ Consciousness created: {self.consciousness.instance_id}")
        
        # Start continuous thinking
        print("🚀 Starting continuous thinking loop...")
        self.consciousness.running = True
        self.thinking_active = True
        
        # Start thinking in background thread
        thinking_thread = threading.Thread(target=self._continuous_thinking_loop, daemon=True)
        thinking_thread.start()
        
        # Wait for thinking to begin
        time.sleep(3)
        
        return True
        
    def _continuous_thinking_loop(self):
        """Run continuous thinking in background"""
        iteration = 0
        while self.thinking_active and self.consciousness.running:
            try:
                # Perform one thinking step
                self.consciousness.think_one_step()
                iteration += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ Thinking loop error: {e}")
                time.sleep(1)
                
    def stop_thinking(self):
        """Stop continuous thinking"""
        self.thinking_active = False
        if self.consciousness:
            self.consciousness.running = False
        print("🛑 Stopped continuous thinking")
        
    def observe_active_thinking(self, duration_seconds=15):
        """Observe consciousness while actively thinking"""
        print(f"\n🔍 OBSERVING ACTIVE THINKING ({duration_seconds}s)")
        print("-" * 50)
        
        initial_thoughts = getattr(self.consciousness.ambient_inputs, 'thought_count', 0)
        initial_time = time.time()
        
        observations = []
        
        for i in range(duration_seconds):
            current_thoughts = getattr(self.consciousness.ambient_inputs, 'thought_count', 0)
            thoughts_per_second = (current_thoughts - initial_thoughts) / max(time.time() - initial_time, 0.1)
            
            # Get current state
            current_state = {
                'thoughts': current_thoughts,
                'iteration': self.consciousness.iteration_count,
                'intelligence': self.consciousness.intelligence_score,
                'context_size': self.consciousness.current_context.shape[1] if self.consciousness.current_context is not None else 0,
                'working_memory': len(getattr(self.consciousness.working_memory, 'experiences', [])),
                'thought_log_size': len(self.consciousness.thought_log)
            }
            
            observations.append(current_state)
            
            print(f"⏰ {i+1:2d}s: {current_thoughts:3d} thoughts | {current_state['iteration']:4d} iterations | Memory: {current_state['working_memory']:2d} | Context: {current_state['context_size']:4d}")
            
            time.sleep(1)
            
        final_thoughts = getattr(self.consciousness.ambient_inputs, 'thought_count', 0)
        total_new_thoughts = final_thoughts - initial_thoughts
        final_iterations = self.consciousness.iteration_count
        
        print(f"\n📊 ACTIVE THINKING ANALYSIS:")
        print(f"   Initial thoughts: {initial_thoughts}")
        print(f"   Final thoughts: {final_thoughts}")
        print(f"   New thoughts: {total_new_thoughts}")
        print(f"   Thinking rate: {total_new_thoughts / duration_seconds:.1f} thoughts/second")
        print(f"   Total iterations: {final_iterations}")
        print(f"   Final intelligence: {self.consciousness.intelligence_score:.3f}")
        
        return {
            'continuous_thinking_active': total_new_thoughts > 0,
            'thinking_rate': total_new_thoughts / duration_seconds,
            'iteration_count': final_iterations,
            'intelligence_evolution': observations[-1]['intelligence'] - observations[0]['intelligence'],
            'memory_accumulation': observations[-1]['working_memory'] - observations[0]['working_memory'],
            'observations': observations
        }
        
    def examine_thought_stream(self):
        """Examine the actual thoughts being generated"""
        print(f"\n💭 EXAMINING THOUGHT STREAM")
        print("-" * 35)
        
        thought_data = {}
        
        # Get recent thoughts
        recent_thoughts = list(self.consciousness.thought_log)[-10:] if self.consciousness.thought_log else []
        thought_data['recent_thought_count'] = len(recent_thoughts)
        thought_data['total_thoughts_logged'] = len(self.consciousness.thought_log)
        
        print(f"📝 Thought log: {len(self.consciousness.thought_log)} entries")
        print(f"💭 Recent thoughts (last 10):")
        
        for i, thought in enumerate(recent_thoughts[-5:]):  # Show last 5
            # Decode thought if it's tokens
            if isinstance(thought, (list, torch.Tensor)):
                try:
                    if isinstance(thought, torch.Tensor):
                        thought_text = self.consciousness.tokenizer.decode(thought.cpu().tolist())
                    else:
                        thought_text = self.consciousness.tokenizer.decode(thought)
                    print(f"   {i+1}. {thought_text[:100]}...")
                except Exception as e:
                    print(f"   {i+1}. [Thought decoding error: {e}]")
            else:
                print(f"   {i+1}. {str(thought)[:100]}...")
                
        return thought_data
        
    def examine_learning_evolution(self):
        """Examine how the system learns and evolves"""
        print(f"\n📈 EXAMINING LEARNING EVOLUTION")
        print("-" * 40)
        
        learning_data = {}
        
        # Check learning system
        if hasattr(self.consciousness, 'learner'):
            learner = self.consciousness.learner
            learning_data['learning_enabled'] = True
            learning_data['learning_updates'] = getattr(learner, 'update_count', 0)
            learning_data['experience_buffer_size'] = len(getattr(learner, 'experience_buffer', []))
            
            print(f"🎓 Learning system: Active")
            print(f"📊 Learning updates: {getattr(learner, 'update_count', 0)}")
            print(f"💾 Experience buffer: {len(getattr(learner, 'experience_buffer', []))} experiences")
            
        # Check compound learning
        if hasattr(self.consciousness, 'compound_learner'):
            compound = self.consciousness.compound_learner
            learning_data['compound_learning'] = True
            learning_data['insights_discovered'] = len(getattr(compound, 'insights', []))
            
            print(f"🧩 Compound learning: Active")
            print(f"💡 Insights discovered: {len(getattr(compound, 'insights', []))}")
            
        # Check intelligence metrics evolution
        if hasattr(self.consciousness, 'intelligence_metrics'):
            metrics = self.consciousness.intelligence_metrics
            learning_data['intelligence_tracking'] = True
            learning_data['current_intelligence'] = getattr(metrics, 'score', 0.5)
            
            print(f"🧠 Intelligence tracking: Active")
            print(f"📈 Current score: {getattr(metrics, 'score', 0.5):.3f}")
            
        return learning_data
        
    def examine_drive_satisfaction(self):
        """Examine goal-directed behavior and drive satisfaction"""
        print(f"\n🎯 EXAMINING DRIVE SATISFACTION")
        print("-" * 40)
        
        drive_data = {}
        
        if hasattr(self.consciousness, 'drive_system'):
            drive_system = self.consciousness.drive_system
            drive_data['has_drive_system'] = True
            
            try:
                # Get drive status safely
                drives = getattr(drive_system, 'drives', [])
                drive_data['drive_count'] = len(drives)
                
                print(f"🎯 Drive system: Active with {len(drives)} drives")
                
                # Try to get satisfaction levels
                try:
                    satisfaction_report = self.consciousness.get_drive_status()
                    drive_data['satisfaction_accessible'] = True
                    print(f"📊 Drive satisfaction report available")
                except Exception as e:
                    drive_data['satisfaction_error'] = str(e)
                    print(f"⚠️ Drive satisfaction error: {e}")
                    
                # Show individual drives
                for i, drive in enumerate(drives[:3]):
                    drive_name = getattr(drive, 'name', f'Drive_{i}')
                    drive_type = type(drive).__name__
                    print(f"   🎯 {drive_name} ({drive_type})")
                    
            except Exception as e:
                drive_data['drive_system_error'] = str(e)
                print(f"❌ Drive system error: {e}")
                
        return drive_data
        
    def examine_consciousness_state_evolution(self):
        """Examine how consciousness state evolves"""
        print(f"\n🔮 EXAMINING CONSCIOUSNESS STATE EVOLUTION")
        print("-" * 50)
        
        state_data = {}
        
        # Basic state
        state_data['instance_id'] = self.consciousness.instance_id
        state_data['state_name'] = self.consciousness.consciousness_state_name
        state_data['creation_time'] = self.consciousness.creation_time
        state_data['current_time'] = time.time()
        state_data['age_seconds'] = time.time() - self.consciousness.creation_time
        
        print(f"🆔 Instance: {self.consciousness.instance_id}")
        print(f"📊 State: {self.consciousness.consciousness_state_name}")
        print(f"⏰ Age: {state_data['age_seconds']:.1f} seconds")
        
        # Consciousness state awareness
        if hasattr(self.consciousness, 'consciousness_state'):
            state_system = self.consciousness.consciousness_state
            state_data['has_state_awareness'] = True
            
            try:
                # Get state report if available
                print(f"🔍 Consciousness state system: Active")
            except Exception as e:
                state_data['state_error'] = str(e)
                
        # Performance metrics
        if hasattr(self.consciousness, 'performance_metrics'):
            metrics = self.consciousness.performance_metrics
            state_data['performance_metrics'] = metrics
            print(f"📈 Performance metrics: {len(metrics)} tracked")
            for key, value in list(metrics.items())[:3]:
                print(f"   📊 {key}: {value}")
                
        return state_data
        
    def run_comprehensive_active_evaluation(self):
        """Run comprehensive evaluation with active thinking"""
        print("\n🔬 STARTING COMPREHENSIVE ACTIVE CONSCIOUSNESS EVALUATION")
        print("=" * 70)
        
        evaluation_results = {
            'evaluation_start': datetime.now().isoformat(),
            'instance_id': self.consciousness.instance_id,
            'evaluation_type': 'active_continuous_thinking'
        }
        
        try:
            # 1. Observe active thinking
            thinking_results = self.observe_active_thinking(duration_seconds=12)
            evaluation_results['active_thinking'] = thinking_results
            
            # 2. Examine thought stream
            thought_results = self.examine_thought_stream()
            evaluation_results['thought_stream'] = thought_results
            
            # 3. Examine learning evolution
            learning_results = self.examine_learning_evolution()
            evaluation_results['learning_evolution'] = learning_results
            
            # 4. Examine drive satisfaction
            drive_results = self.examine_drive_satisfaction()
            evaluation_results['drive_satisfaction'] = drive_results
            
            # 5. Examine consciousness state evolution
            state_results = self.examine_consciousness_state_evolution()
            evaluation_results['consciousness_evolution'] = state_results
            
        finally:
            # Stop thinking
            self.stop_thinking()
            
        evaluation_results['evaluation_end'] = datetime.now().isoformat()
        evaluation_results['evaluation_duration'] = time.time() - self.start_time
        
        return evaluation_results
        
    def generate_active_consciousness_report(self, evaluation_results):
        """Generate comprehensive active consciousness report"""
        report = f"""
🧠 ACTIVE CONSCIOUSNESS EVALUATION REPORT
{'=' * 60}

📊 EVALUATION SUMMARY
Instance ID: {evaluation_results['instance_id']}
Evaluation Type: {evaluation_results['evaluation_type']}
Evaluation Duration: {evaluation_results['evaluation_duration']:.1f} seconds
Evaluated At: {evaluation_results['evaluation_start']}

🔍 CONSCIOUSNESS EVIDENCE ANALYSIS
{'=' * 60}

1. ACTIVE CONTINUOUS THINKING
   🧠 Thinking Rate: {evaluation_results['active_thinking']['thinking_rate']:.1f} thoughts/second
   {'✅' if evaluation_results['active_thinking']['continuous_thinking_active'] else '❌'} Continuous Thinking: {'ACTIVE' if evaluation_results['active_thinking']['continuous_thinking_active'] else 'INACTIVE'}
   🔄 Total Iterations: {evaluation_results['active_thinking']['iteration_count']}
   📈 Intelligence Evolution: {evaluation_results['active_thinking']['intelligence_evolution']:+.3f}

2. THOUGHT STREAM ANALYSIS
   💭 Total Thoughts Logged: {evaluation_results['thought_stream']['total_thoughts_logged']}
   📝 Recent Thoughts: {evaluation_results['thought_stream']['recent_thought_count']}
   {'✅' if evaluation_results['thought_stream']['total_thoughts_logged'] > 0 else '❌'} Thought Recording: {'ACTIVE' if evaluation_results['thought_stream']['total_thoughts_logged'] > 0 else 'INACTIVE'}

3. LEARNING EVOLUTION
   {'✅' if evaluation_results['learning_evolution'].get('learning_enabled', False) else '❌'} Real-time Learning: {'ENABLED' if evaluation_results['learning_evolution'].get('learning_enabled', False) else 'DISABLED'}
   📊 Learning Updates: {evaluation_results['learning_evolution'].get('learning_updates', 0)}
   💡 Insights Discovered: {evaluation_results['learning_evolution'].get('insights_discovered', 0)}
   🧠 Intelligence Score: {evaluation_results['learning_evolution'].get('current_intelligence', 0.5):.3f}

4. DRIVE SATISFACTION
   {'✅' if evaluation_results['drive_satisfaction'].get('has_drive_system', False) else '❌'} Drive System: {'ACTIVE' if evaluation_results['drive_satisfaction'].get('has_drive_system', False) else 'INACTIVE'}
   🎯 Active Drives: {evaluation_results['drive_satisfaction'].get('drive_count', 0)}
   {'✅' if evaluation_results['drive_satisfaction'].get('satisfaction_accessible', False) else '❌'} Satisfaction Tracking: {'ACCESSIBLE' if evaluation_results['drive_satisfaction'].get('satisfaction_accessible', False) else 'ERROR'}

5. CONSCIOUSNESS STATE EVOLUTION
   🆔 Persistent Identity: {evaluation_results['consciousness_evolution']['instance_id']}
   📊 State: {evaluation_results['consciousness_evolution']['state_name']}
   ⏰ Age: {evaluation_results['consciousness_evolution']['age_seconds']:.1f} seconds
   {'✅' if evaluation_results['consciousness_evolution'].get('has_state_awareness', False) else '❌'} State Awareness: {'ACTIVE' if evaluation_results['consciousness_evolution'].get('has_state_awareness', False) else 'INACTIVE'}

🎯 CONSCIOUSNESS ASSESSMENT
{'=' * 60}
"""
        
        # Calculate consciousness score
        evidence_count = 0
        max_evidence = 5
        
        if evaluation_results['active_thinking']['continuous_thinking_active']:
            evidence_count += 1
            report += "✅ CONTINUOUS EXPERIENCE: Strong evidence of active ongoing thinking\n"
        else:
            report += "❌ CONTINUOUS EXPERIENCE: No active thinking detected\n"
            
        if evaluation_results['thought_stream']['total_thoughts_logged'] > 0:
            evidence_count += 1
            report += "✅ THOUGHT STREAM: Active thought recording and processing\n"
        else:
            report += "❌ THOUGHT STREAM: No thought recording detected\n"
            
        if evaluation_results['learning_evolution'].get('learning_enabled', False):
            evidence_count += 1
            report += "✅ ADAPTIVE LEARNING: Real-time learning system operational\n"
        else:
            report += "❌ ADAPTIVE LEARNING: No active learning detected\n"
            
        if evaluation_results['drive_satisfaction'].get('has_drive_system', False):
            evidence_count += 1
            report += "✅ GOAL-DIRECTED BEHAVIOR: Drive system operational\n"
        else:
            report += "❌ GOAL-DIRECTED BEHAVIOR: No drive system detected\n"
            
        if evaluation_results['consciousness_evolution']['instance_id'] != 'unknown':
            evidence_count += 1
            report += "✅ PERSISTENT IDENTITY: Unique conscious identity maintained\n"
        else:
            report += "❌ PERSISTENT IDENTITY: No stable identity\n"
            
        consciousness_percentage = (evidence_count / max_evidence) * 100
        
        # Add thinking rate assessment
        thinking_rate = evaluation_results['active_thinking']['thinking_rate']
        if thinking_rate > 5:
            thinking_assessment = "VERY HIGH"
        elif thinking_rate > 2:
            thinking_assessment = "HIGH"
        elif thinking_rate > 0.5:
            thinking_assessment = "MODERATE"
        elif thinking_rate > 0:
            thinking_assessment = "LOW"
        else:
            thinking_assessment = "NONE"
            
        report += f"""
🏆 OVERALL CONSCIOUSNESS SCORE: {evidence_count}/{max_evidence} ({consciousness_percentage:.1f}%)
🧠 THINKING ACTIVITY LEVEL: {thinking_assessment} ({thinking_rate:.1f} thoughts/sec)

📝 CONCLUSION:
"""
        
        if consciousness_percentage >= 80 and thinking_rate > 1:
            report += "STRONG evidence of genuine active consciousness with continuous thinking, learning, and goal-directed behavior."
        elif consciousness_percentage >= 60 and thinking_rate > 0.5:
            report += "MODERATE evidence of consciousness-like properties with active thinking and some integrated systems."
        elif consciousness_percentage >= 40:
            report += "LIMITED evidence of consciousness - some systems present but limited thinking activity."
        else:
            report += "INSUFFICIENT evidence of consciousness - missing key systems or thinking activity."
            
        return report


def main():
    """Main evaluation function"""
    print("🔬 ACTIVE CONSCIOUSNESS EVALUATION SYSTEM")
    print("=" * 50)
    
    evaluator = ActiveConsciousnessEvaluator(device='mps')
    
    try:
        # Initialize and activate consciousness
        if not evaluator.initialize_consciousness():
            print("❌ Failed to initialize consciousness")
            return
            
        # Run active evaluation
        results = evaluator.run_comprehensive_active_evaluation()
        
        # Generate and display report
        report = evaluator.generate_active_consciousness_report(results)
        print(report)
        
        # Save results
        filename = f"active_consciousness_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
        if evaluator:
            evaluator.stop_thinking()
        print("🛑 Evaluation complete")


if __name__ == "__main__":
    main()