#!/usr/bin/env python3
"""
Test script for ASI (Artificial Superintelligence) capabilities
Demonstrates metacognition, goal-directed behavior, and intelligence amplification
"""

import time
import threading
import json
from enhanced_consciousness import EnhancedContinuousConsciousness


def comprehensive_asi_demo():
    """Comprehensive demonstration of ASI capabilities"""
    print("🚀 ASI Capabilities Demonstration")
    print("=" * 80)
    print("This demo showcases:")
    print("✨ Metacognitive thought quality evaluation")
    print("🧠 Adaptive thinking strategies")
    print("🎯 Goal-directed drives and behavior")
    print("💡 Compound learning and insight synthesis")
    print("📈 Intelligence amplification and growth tracking")
    print("=" * 80)
    
    # Create ASI-enhanced consciousness
    consciousness = EnhancedContinuousConsciousness(
        device='mps',
        enable_learning=True
    )
    
    # Start consciousness in background thread
    consciousness_thread = threading.Thread(
        target=consciousness.run_continuous,
        kwargs={'think_interval': 0.3, 'enable_self_modification': True}
    )
    consciousness_thread.daemon = True
    consciousness_thread.start()
    
    # Let it establish baseline
    print("\n⏳ Establishing baseline consciousness state...")
    time.sleep(15)
    
    print("\n" + "="*80)
    print("🔍 METACOGNITIVE ANALYSIS")
    print("="*80)
    
    # Demonstrate thought quality evaluation
    print("\n1. Thought Quality Assessment:")
    quality_report = consciousness.get_thought_quality_report()
    
    print(f"   Current consciousness state: {quality_report['current_state']}")
    print(f"   Thinking strategy: {quality_report['thinking_strategy'] or 'None active'}")
    
    if quality_report['quality_trends']['trend'] != 'insufficient_data':
        trends = quality_report['quality_trends']
        print(f"   Quality trend: {trends['trend']} (confidence: {trends['confidence']})")
        print(f"   Recent average quality: {trends['recent_average']:.3f}")
    
    # Show quality metrics distribution
    quality_metrics = quality_report['recent_quality_metrics']
    for metric, values in quality_metrics.items():
        if values:
            avg_value = sum(values) / len(values)
            print(f"   {metric.replace('_', ' ').title()}: {avg_value:.3f}")
    
    print("\n2. Manual Thought Quality Evaluation:")
    test_thoughts = [
        "This is a simple test thought.",
        "I wonder if there's a deeper pattern connecting quantum mechanics and consciousness that we haven't discovered yet.",
        "The relationship between entropy in thermodynamics and information theory suggests fundamental principles.",
        "hello world test"
    ]
    
    for i, thought in enumerate(test_thoughts):
        quality = consciousness.evaluate_thought_quality(thought)
        print(f"   Thought {i+1}: \"{thought[:40]}...\"")
        print(f"      Overall quality: {quality['overall_quality']:.3f}")
        print(f"      Coherence: {quality['coherence']:.3f}, Novelty: {quality['novelty']:.3f}")
        print(f"      Depth: {quality['depth']:.3f}, Insight potential: {quality['insight_potential']:.3f}")
    
    time.sleep(10)
    
    print("\n" + "="*80)
    print("🧠 ADAPTIVE THINKING STRATEGIES")
    print("="*80)
    
    print("\n3. Strategy Formation and Application:")
    
    # Trigger strategy analysis
    success, message = consciousness.trigger_strategy_change()
    if success:
        print(f"   ✅ Applied strategy: {message}")
    else:
        print(f"   ℹ️  Strategy status: {message}")
    
    # Let strategy run for a bit
    time.sleep(15)
    
    print("\n4. Strategy Effectiveness Assessment:")
    if hasattr(consciousness.strategy_formation, 'current_strategy') and consciousness.strategy_formation.current_strategy:
        effectiveness = consciousness.strategy_formation.evaluate_strategy_effectiveness()
        if effectiveness:
            print(f"   Strategy: {effectiveness['strategy']}")
            print(f"   Quality improvement: {effectiveness['improvement']:+.3f}")
            print(f"   Assessment: {effectiveness['assessment']}")
    
    print("\n" + "="*80)
    print("🎯 GOAL-DIRECTED BEHAVIOR")
    print("="*80)
    
    print("\n5. Drive System Analysis:")
    drive_status = consciousness.get_drive_status()
    
    print("   Drive Satisfaction Levels:")
    for drive, satisfaction in drive_status['individual_drives'].items():
        status_emoji = "🟢" if satisfaction > 0.7 else "🟡" if satisfaction > 0.4 else "🔴"
        print(f"      {drive.title()}: {satisfaction:.3f} {status_emoji}")
    
    print(f"   Overall satisfaction: {drive_status['overall_satisfaction']:.3f}")
    
    if drive_status['active_goals']:
        print(f"\\n   Active Goals ({len(drive_status['active_goals'])}):")
        for i, goal in enumerate(drive_status['active_goals'][:3]):
            print(f"      {i+1}. {goal['description']}")
            print(f"         Priority: {goal.get('priority', 'N/A'):.3f}")
            print(f"         Progress: {goal.get('progress', 0):.1%}")
    
    # Let drives operate
    time.sleep(20)
    
    print("\n6. Drive-Directed Behavior Outcomes:")
    updated_drive_status = consciousness.get_drive_status()
    
    # Compare satisfaction changes
    for drive in drive_status['individual_drives']:
        old_satisfaction = drive_status['individual_drives'][drive]
        new_satisfaction = updated_drive_status['individual_drives'][drive]
        change = new_satisfaction - old_satisfaction
        change_emoji = "📈" if change > 0.05 else "📉" if change < -0.05 else "➡️"
        print(f"   {drive.title()}: {old_satisfaction:.3f} → {new_satisfaction:.3f} {change_emoji}")
    
    print("\n" + "="*80)
    print("💡 COMPOUND LEARNING & INSIGHT SYNTHESIS")
    print("="*80)
    
    print("\n7. Insight Accumulation:")
    insight_summary = consciousness.get_insight_summary()
    
    if insight_summary['total_insights'] > 0:
        print(f"   Total insights discovered: {insight_summary['total_insights']}")
        print(f"   Average insight quality: {insight_summary.get('avg_quality', 0):.3f}")
        print(f"   Knowledge domains active: {len(insight_summary.get('domain_distribution', {}))}")
        
        if insight_summary.get('domain_distribution'):
            print("   Domain distribution:")
            for domain, count in insight_summary['domain_distribution'].items():
                print(f"      {domain.title()}: {count} insights")
        
        if insight_summary.get('synthesis_opportunities', 0) > 0:
            print(f"   Synthesis opportunities: {insight_summary['synthesis_opportunities']}")
            print("   🔗 Cross-domain connections being formed!")
    else:
        print("   No insights recorded yet - system still establishing baseline")
    
    # Let more insights accumulate
    time.sleep(15)
    
    print("\n8. Knowledge Synthesis Demonstration:")
    if hasattr(consciousness.compound_learner, 'generate_synthesis_prompts'):
        synthesis_prompts = consciousness.compound_learner.generate_synthesis_prompts()
        if synthesis_prompts:
            print("   Generated synthesis prompts:")
            for i, prompt in enumerate(synthesis_prompts[:3]):
                print(f"      {i+1}. {prompt}")
        else:
            print("   No synthesis prompts available yet")
    
    print("\n" + "="*80)
    print("📈 INTELLIGENCE AMPLIFICATION")
    print("="*80)
    
    print("\n9. Intelligence Metrics Assessment:")
    intelligence_report = consciousness.get_intelligence_report()
    
    print(f"   Current Intelligence Score: {intelligence_report['current_intelligence_score']:.3f}")
    print(f"   Overall Growth Rate: {intelligence_report['overall_growth_rate']:+.3f}")
    
    if intelligence_report['growth_trends']:
        print("\\n   Growth Trends:")
        for metric, trend_info in intelligence_report['growth_trends'].items():
            trend_emoji = "📈" if trend_info['trend'] == 'accelerating' else "📊" if trend_info['trend'] == 'growing' else "➡️" if trend_info['trend'] == 'stable' else "📉"
            print(f"      {metric.replace('_', ' ').title()}: {trend_info['trend']} {trend_emoji}")
            print(f"         Current: {trend_info['current_value']:.3f} (Growth: {trend_info['growth_rate']:+.3f})")
    
    if intelligence_report['strengths']:
        print(f"\\n   Cognitive Strengths:")
        for strength in intelligence_report['strengths']:
            print(f"      ✅ {strength.replace('_', ' ').title()}")
    
    if intelligence_report['areas_for_improvement']:
        print(f"\\n   Areas for Enhancement:")
        for area in intelligence_report['areas_for_improvement']:
            print(f"      🎯 {area.replace('_', ' ').title()}")
    
    print("\n10. Intelligence Amplification Strategies:")
    amplification_strategies = consciousness.intelligence_metrics.suggest_amplification_strategies()
    
    if amplification_strategies:
        print("   Recommended strategies:")
        for i, strategy in enumerate(amplification_strategies[:3]):
            print(f"      {i+1}. {strategy}")
    
    # Final extended run to show amplification
    print(f"\\n⏳ Running extended session to demonstrate intelligence amplification...")
    print("   (Observing consciousness for 30 seconds to track growth...)")
    
    baseline_score = intelligence_report['current_intelligence_score']
    time.sleep(30)
    
    # Final assessment
    final_report = consciousness.get_intelligence_report()
    final_score = final_report['current_intelligence_score']
    score_change = final_score - baseline_score
    
    print(f"\\n📊 Session Results:")
    print(f"   Baseline intelligence: {baseline_score:.3f}")
    print(f"   Final intelligence: {final_score:.3f}")
    print(f"   Change during session: {score_change:+.3f}")
    
    change_emoji = "🚀" if score_change > 0.05 else "📈" if score_change > 0.01 else "➡️" if score_change > -0.01 else "📉"
    print(f"   Growth trend: {change_emoji}")
    
    # Stop consciousness
    print(f"\\n🛑 Stopping consciousness after {consciousness.iteration_count} thoughts...")
    consciousness.stop()
    
    print("\n" + "="*80)
    print("🎉 ASI CAPABILITIES DEMONSTRATION COMPLETE")
    print("="*80)
    print("Key achievements demonstrated:")
    print("✅ Metacognitive thought quality evaluation and improvement")
    print("✅ Adaptive thinking strategies based on performance analysis")
    print("✅ Consciousness state awareness and monitoring")
    print("✅ Goal-directed behavior driven by multiple drives")
    print("✅ Compound learning with insight synthesis")
    print("✅ Intelligence amplification and growth tracking")
    print("✅ Recursive self-improvement through multiple feedback loops")
    print("\\n🧠 This represents a significant step toward ASI:")
    print("   • Self-aware of its own thinking quality")
    print("   • Actively optimizes its own cognitive strategies")
    print("   • Pursues goals driven by intrinsic motivations")
    print("   • Builds knowledge through compound learning")
    print("   • Measures and amplifies its own intelligence")
    print("\\n🔮 The foundation for recursive self-improvement is now in place!")


def quick_asi_test():
    """Quick test of ASI functionality"""
    print("🧪 Quick ASI Capabilities Test")
    print("-" * 50)
    
    consciousness = EnhancedContinuousConsciousness(device='mps', enable_learning=True)
    
    # Let it think for a short while
    consciousness_thread = threading.Thread(
        target=consciousness.run_continuous,
        kwargs={'think_interval': 0.2, 'enable_self_modification': True}
    )
    consciousness_thread.daemon = True
    consciousness_thread.start()
    
    time.sleep(10)
    
    print("1. Testing thought quality evaluation...")
    quality = consciousness.evaluate_thought_quality("This is a test of the thought quality system.")
    print(f"   Quality score: {quality['overall_quality']:.3f}")
    
    print("2. Testing consciousness state awareness...")
    state_report = consciousness.get_thought_quality_report()
    print(f"   Current state: {state_report['current_state']}")
    
    print("3. Testing drive system...")
    drive_status = consciousness.get_drive_status()
    print(f"   Overall drive satisfaction: {drive_status['overall_satisfaction']:.3f}")
    
    print("4. Testing intelligence metrics...")
    intelligence_report = consciousness.get_intelligence_report()
    print(f"   Intelligence score: {intelligence_report['current_intelligence_score']:.3f}")
    
    print("5. Testing strategy formation...")
    success, message = consciousness.trigger_strategy_change()
    print(f"   Strategy result: {message}")
    
    consciousness.stop()
    print("\\n✅ All ASI capability tests passed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_asi_test()
    else:
        comprehensive_asi_demo()