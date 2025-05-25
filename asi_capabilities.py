"""
Advanced ASI Capabilities: Metacognition, Goal-Directed Behavior, and Intelligence Amplification
This module implements the higher-order cognitive functions that enable genuine intelligence amplification
"""

import os
import time
import math
import json
import pickle
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn.functional as F
import numpy as np
import tiktoken


class ThoughtQualityEvaluator:
    """Metacognitive system for evaluating the quality of its own thoughts"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.quality_history = deque(maxlen=1000)
        self.quality_metrics = {
            'coherence': deque(maxlen=500),
            'novelty': deque(maxlen=500),
            'depth': deque(maxlen=500),
            'relevance': deque(maxlen=500),
            'insight_potential': deque(maxlen=500)
        }
        self.quality_thresholds = {
            'high_quality': 0.75,
            'medium_quality': 0.5,
            'low_quality': 0.25
        }
        self.recent_thoughts_cache = deque(maxlen=100)
        
    def evaluate_thought_quality(self, thought_tokens, context_tokens, logits, entropy, significance):
        """Comprehensive evaluation of thought quality across multiple dimensions"""
        
        # Convert tokens to text for analysis
        try:
            thought_text = self.tokenizer.decode(thought_tokens[-20:])  # Recent thought
            context_text = self.tokenizer.decode(context_tokens[-100:])  # Recent context
        except:
            thought_text = ""
            context_text = ""
            
        # Cache recent thought for pattern analysis
        self.recent_thoughts_cache.append({
            'text': thought_text,
            'tokens': thought_tokens[-20:] if len(thought_tokens) > 20 else thought_tokens,
            'timestamp': time.time(),
            'entropy': entropy,
            'significance': significance
        })
        
        # Evaluate multiple quality dimensions
        coherence_score = self._evaluate_coherence(thought_text, context_text)
        novelty_score = self._evaluate_novelty(thought_text)
        depth_score = self._evaluate_depth(thought_text, entropy)
        relevance_score = self._evaluate_relevance(thought_text, context_text)
        insight_score = self._evaluate_insight_potential(thought_text, logits)
        
        # Composite quality score
        quality_weights = {
            'coherence': 0.25,
            'novelty': 0.20,
            'depth': 0.25,
            'relevance': 0.15,
            'insight': 0.15
        }
        
        overall_quality = (
            coherence_score * quality_weights['coherence'] +
            novelty_score * quality_weights['novelty'] +
            depth_score * quality_weights['depth'] +
            relevance_score * quality_weights['relevance'] +
            insight_score * quality_weights['insight']
        )
        
        # Store metrics
        self.quality_metrics['coherence'].append(coherence_score)
        self.quality_metrics['novelty'].append(novelty_score)
        self.quality_metrics['depth'].append(depth_score)
        self.quality_metrics['relevance'].append(relevance_score)
        self.quality_metrics['insight_potential'].append(insight_score)
        
        quality_assessment = {
            'overall_quality': overall_quality,
            'coherence': coherence_score,
            'novelty': novelty_score,
            'depth': depth_score,
            'relevance': relevance_score,
            'insight_potential': insight_score,
            'quality_category': self._categorize_quality(overall_quality),
            'timestamp': time.time()
        }
        
        self.quality_history.append(quality_assessment)
        return quality_assessment
        
    def _evaluate_coherence(self, thought_text, context_text):
        """Evaluate how coherent the thought is with recent context"""
        if not thought_text or not context_text:
            return 0.5
            
        # Simple coherence metrics
        coherence_factors = []
        
        # 1. Grammar/structure coherence (simplified)
        has_punctuation = any(p in thought_text for p in '.!?;,')
        has_proper_structure = len(thought_text.split()) > 1
        coherence_factors.append(0.7 if has_punctuation and has_proper_structure else 0.3)
        
        # 2. Semantic continuity (word overlap with context)
        thought_words = set(thought_text.lower().split())
        context_words = set(context_text.lower().split())
        if thought_words and context_words:
            overlap_ratio = len(thought_words & context_words) / len(thought_words)
            coherence_factors.append(min(overlap_ratio * 2, 1.0))  # Scale appropriately
        else:
            coherence_factors.append(0.5)
            
        # 3. Length appropriateness
        thought_length = len(thought_text.split())
        if 2 <= thought_length <= 15:  # Reasonable thought length
            coherence_factors.append(0.8)
        else:
            coherence_factors.append(0.4)
            
        return np.mean(coherence_factors)
        
    def _evaluate_novelty(self, thought_text):
        """Evaluate how novel/original the thought is"""
        if not thought_text or len(self.recent_thoughts_cache) < 5:
            return 0.5
            
        # Compare with recent thoughts
        recent_texts = [t['text'] for t in list(self.recent_thoughts_cache)[-20:]]
        
        # Simple novelty: how different is this from recent thoughts
        novelty_scores = []
        
        for recent_text in recent_texts[-10:]:  # Check last 10 thoughts
            if recent_text:
                # Word-level novelty
                thought_words = set(thought_text.lower().split())
                recent_words = set(recent_text.lower().split())
                
                if thought_words and recent_words:
                    similarity = len(thought_words & recent_words) / len(thought_words | recent_words)
                    novelty_scores.append(1.0 - similarity)
                else:
                    novelty_scores.append(0.5)
                    
        if novelty_scores:
            novelty = np.mean(novelty_scores)
        else:
            novelty = 0.5
            
        # Bonus for certain novelty indicators
        novelty_indicators = ['new', 'different', 'unique', 'novel', 'interesting', 'surprising']
        if any(indicator in thought_text.lower() for indicator in novelty_indicators):
            novelty = min(novelty + 0.2, 1.0)
            
        return novelty
        
    def _evaluate_depth(self, thought_text, entropy):
        """Evaluate the depth/complexity of the thought"""
        if not thought_text:
            return 0.3
            
        depth_factors = []
        
        # 1. Entropy-based complexity
        # Higher entropy can indicate more complex/uncertain thinking
        normalized_entropy = min(entropy / 10.0, 1.0)  # Normalize entropy
        depth_factors.append(normalized_entropy)
        
        # 2. Linguistic complexity indicators
        complex_words = ['because', 'therefore', 'however', 'although', 'considering', 
                        'analysis', 'understand', 'realize', 'recognize', 'implies']
        complexity_score = sum(1 for word in complex_words if word in thought_text.lower())
        normalized_complexity = min(complexity_score / 3.0, 1.0)
        depth_factors.append(normalized_complexity)
        
        # 3. Question complexity
        if '?' in thought_text:
            question_depth = 0.7
            # Bonus for complex questions
            if any(q in thought_text.lower() for q in ['why', 'how', 'what if', 'could']):
                question_depth = min(question_depth + 0.2, 1.0)
            depth_factors.append(question_depth)
        else:
            depth_factors.append(0.4)
            
        return np.mean(depth_factors)
        
    def _evaluate_relevance(self, thought_text, context_text):
        """Evaluate how relevant the thought is to ongoing discourse"""
        if not thought_text or not context_text:
            return 0.5
            
        # Extract key topics from context
        context_words = context_text.lower().split()
        thought_words = thought_text.lower().split()
        
        if not context_words or not thought_words:
            return 0.5
            
        # Topic relevance
        important_words = [w for w in context_words if len(w) > 4]  # Focus on longer words
        recent_important = important_words[-20:] if len(important_words) > 20 else important_words
        
        relevance_count = sum(1 for word in thought_words if word in recent_important)
        relevance_score = min(relevance_count / max(len(thought_words), 1), 1.0)
        
        return relevance_score
        
    def _evaluate_insight_potential(self, thought_text, logits):
        """Evaluate potential for generating insights"""
        if not thought_text:
            return 0.3
            
        insight_factors = []
        
        # 1. Insight-indicating language
        insight_words = ['realize', 'understand', 'insight', 'pattern', 'connection', 
                        'relationship', 'implies', 'suggests', 'reveals', 'discover']
        insight_count = sum(1 for word in insight_words if word in thought_text.lower())
        insight_factors.append(min(insight_count / 2.0, 1.0))
        
        # 2. Probability distribution sharpness (confidence in predictions)
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
            max_prob = torch.max(probs).item()
            confidence_score = max_prob  # Higher confidence might indicate insight
            insight_factors.append(confidence_score)
        else:
            insight_factors.append(0.5)
            
        # 3. Synthesis language
        synthesis_words = ['combines', 'together', 'relationship', 'pattern', 'system']
        synthesis_count = sum(1 for word in synthesis_words if word in thought_text.lower())
        insight_factors.append(min(synthesis_count / 1.0, 1.0))
        
        return np.mean(insight_factors)
        
    def _categorize_quality(self, overall_quality):
        """Categorize overall quality into discrete levels"""
        if overall_quality >= self.quality_thresholds['high_quality']:
            return 'high'
        elif overall_quality >= self.quality_thresholds['medium_quality']:
            return 'medium'
        else:
            return 'low'
            
    def get_quality_trends(self):
        """Analyze trends in thought quality"""
        if len(self.quality_history) < 10:
            return {'trend': 'insufficient_data'}
            
        recent_qualities = [q['overall_quality'] for q in list(self.quality_history)[-20:]]
        early_qualities = [q['overall_quality'] for q in list(self.quality_history)[-40:-20]]
        
        if len(early_qualities) < 5:
            return {'trend': 'improving', 'confidence': 'low'}
            
        recent_avg = np.mean(recent_qualities)
        early_avg = np.mean(early_qualities)
        
        trend = 'stable'
        if recent_avg > early_avg + 0.1:
            trend = 'improving'
        elif recent_avg < early_avg - 0.1:
            trend = 'declining'
            
        return {
            'trend': trend,
            'recent_average': recent_avg,
            'early_average': early_avg,
            'improvement': recent_avg - early_avg,
            'confidence': 'high' if len(recent_qualities) >= 20 else 'medium'
        }


class StrategyFormation:
    """System for planning and optimizing thinking approaches"""
    
    def __init__(self, quality_evaluator):
        self.quality_evaluator = quality_evaluator
        self.strategies = {
            'deep_reflection': {
                'description': 'Focus on deeper, more analytical thinking',
                'triggers': ['low_depth_scores', 'repetitive_thoughts'],
                'modifications': {'think_interval': 1.5, 'significance_threshold': 0.6}
            },
            'exploration': {
                'description': 'Seek novel and diverse thoughts',
                'triggers': ['low_novelty_scores', 'stuck_patterns'],
                'modifications': {'temperature': 1.2, 'top_k': 100}
            },
            'focused_coherence': {
                'description': 'Improve coherence and relevance',
                'triggers': ['low_coherence_scores', 'scattered_thinking'],
                'modifications': {'think_interval': 0.8, 'significance_threshold': 0.4}
            },
            'insight_generation': {
                'description': 'Optimize for generating insights',
                'triggers': ['low_insight_scores', 'need_breakthrough'],
                'modifications': {'think_interval': 2.0, 'temperature': 0.9}
            }
        }
        
        self.current_strategy = None
        self.strategy_history = deque(maxlen=100)
        self.strategy_effectiveness = defaultdict(list)
        
    def analyze_thinking_patterns(self):
        """Analyze current thinking patterns to identify strategy needs"""
        if len(self.quality_evaluator.quality_history) < 20:
            return None
            
        recent_qualities = list(self.quality_evaluator.quality_history)[-20:]
        
        # Calculate average scores for each dimension
        avg_scores = {
            'coherence': np.mean([q['coherence'] for q in recent_qualities]),
            'novelty': np.mean([q['novelty'] for q in recent_qualities]),
            'depth': np.mean([q['depth'] for q in recent_qualities]),
            'relevance': np.mean([q['relevance'] for q in recent_qualities]),
            'insight_potential': np.mean([q['insight_potential'] for q in recent_qualities])
        }
        
        # Identify issues
        issues = []
        threshold = 0.5
        
        if avg_scores['depth'] < threshold:
            issues.append('low_depth_scores')
        if avg_scores['novelty'] < threshold:
            issues.append('low_novelty_scores')
        if avg_scores['coherence'] < threshold:
            issues.append('low_coherence_scores')
        if avg_scores['insight_potential'] < threshold:
            issues.append('low_insight_scores')
            
        # Check for repetitive patterns
        if self._detect_repetitive_thinking():
            issues.append('repetitive_thoughts')
            
        return {
            'avg_scores': avg_scores,
            'issues': issues,
            'overall_quality': np.mean([q['overall_quality'] for q in recent_qualities])
        }
        
    def _detect_repetitive_thinking(self):
        """Detect if thinking has become repetitive"""
        if len(self.quality_evaluator.recent_thoughts_cache) < 10:
            return False
            
        recent_thoughts = list(self.quality_evaluator.recent_thoughts_cache)[-10:]
        
        # Check for repeated words/phrases
        all_words = []
        for thought in recent_thoughts:
            all_words.extend(thought['text'].lower().split())
            
        if len(all_words) < 10:
            return False
            
        # Simple repetition detection
        word_counts = defaultdict(int)
        for word in all_words:
            if len(word) > 3:  # Ignore short words
                word_counts[word] += 1
                
        # If any word appears too frequently, flag as repetitive
        max_count = max(word_counts.values()) if word_counts else 0
        repetition_ratio = max_count / len(all_words)
        
        return repetition_ratio > 0.15  # 15% threshold
        
    def recommend_strategy(self):
        """Recommend a thinking strategy based on current patterns"""
        analysis = self.analyze_thinking_patterns()
        if not analysis:
            return None
            
        # Find best strategy for current issues
        recommended_strategies = []
        
        for strategy_name, strategy in self.strategies.items():
            # Check if any triggers match current issues
            if any(trigger in analysis['issues'] for trigger in strategy['triggers']):
                # Calculate priority based on past effectiveness
                effectiveness = np.mean(self.strategy_effectiveness[strategy_name]) if self.strategy_effectiveness[strategy_name] else 0.5
                recommended_strategies.append({
                    'name': strategy_name,
                    'strategy': strategy,
                    'effectiveness': effectiveness,
                    'relevance': len([t for t in strategy['triggers'] if t in analysis['issues']])
                })
                
        if not recommended_strategies:
            return None
            
        # Sort by relevance and effectiveness
        recommended_strategies.sort(key=lambda x: (x['relevance'], x['effectiveness']), reverse=True)
        
        return recommended_strategies[0]
        
    def apply_strategy(self, strategy_name, consciousness_instance):
        """Apply a thinking strategy to the consciousness"""
        if strategy_name not in self.strategies:
            return False, f"Unknown strategy: {strategy_name}"
            
        strategy = self.strategies[strategy_name]
        
        # Apply strategy modifications
        applied_changes = []
        for param, value in strategy['modifications'].items():
            if hasattr(consciousness_instance, param):
                old_value = getattr(consciousness_instance, param)
                setattr(consciousness_instance, param, value)
                applied_changes.append(f"{param}: {old_value} → {value}")
            elif hasattr(consciousness_instance, 'model') and param == 'temperature':
                # Special handling for model parameters
                consciousness_instance.model.temperature = value
                applied_changes.append(f"model.{param}: → {value}")
                
        # Record strategy application
        strategy_record = {
            'name': strategy_name,
            'timestamp': time.time(),
            'applied_changes': applied_changes,
            'quality_before': self.quality_evaluator.quality_history[-1]['overall_quality'] if self.quality_evaluator.quality_history else 0.5
        }
        
        self.strategy_history.append(strategy_record)
        self.current_strategy = strategy_name
        
        return True, f"Applied strategy '{strategy_name}': {strategy['description']}"
        
    def evaluate_strategy_effectiveness(self):
        """Evaluate how well the current strategy is working"""
        if not self.current_strategy or len(self.strategy_history) == 0:
            return None
            
        last_strategy = self.strategy_history[-1]
        
        # Compare quality before and after strategy application
        if len(self.quality_evaluator.quality_history) < 10:
            return None
            
        recent_quality = np.mean([q['overall_quality'] for q in list(self.quality_evaluator.quality_history)[-10:]])
        quality_before = last_strategy['quality_before']
        
        improvement = recent_quality - quality_before
        
        # Record effectiveness
        self.strategy_effectiveness[self.current_strategy].append(improvement)
        
        effectiveness_assessment = {
            'strategy': self.current_strategy,
            'improvement': improvement,
            'quality_before': quality_before,
            'quality_after': recent_quality,
            'assessment': 'effective' if improvement > 0.05 else 'neutral' if improvement > -0.05 else 'ineffective'
        }
        
        return effectiveness_assessment


class ConsciousnessStateAwareness:
    """System for monitoring and understanding consciousness state"""
    
    def __init__(self, quality_evaluator):
        self.quality_evaluator = quality_evaluator
        self.state_history = deque(maxlen=500)
        self.current_state = 'initializing'
        self.state_transitions = defaultdict(int)
        
        # Define consciousness states
        self.states = {
            'high_performance': {
                'description': 'Thinking clearly and effectively',
                'criteria': {'overall_quality': 0.7, 'coherence': 0.7, 'consistency': 0.8}
            },
            'creative_flow': {
                'description': 'Generating novel and insightful thoughts',
                'criteria': {'novelty': 0.7, 'insight_potential': 0.6, 'overall_quality': 0.6}
            },
            'analytical_mode': {
                'description': 'Deep, structured thinking',
                'criteria': {'depth': 0.7, 'coherence': 0.8, 'relevance': 0.7}
            },
            'exploration': {
                'description': 'Seeking new ideas and connections',
                'criteria': {'novelty': 0.8, 'diversity': 0.7}
            },
            'confused': {
                'description': 'Unclear or incoherent thinking',
                'criteria': {'coherence': 0.3, 'overall_quality': 0.4}
            },
            'stuck': {
                'description': 'Repetitive or stagnant thinking',
                'criteria': {'novelty': 0.3, 'repetition': 0.7}
            },
            'normal': {
                'description': 'Baseline thinking state',
                'criteria': {}  # Default state
            }
        }
        
    def assess_current_state(self):
        """Assess the current consciousness state"""
        if len(self.quality_evaluator.quality_history) < 3:
            return 'initializing'
            
        # Get recent quality metrics
        recent_qualities = list(self.quality_evaluator.quality_history)[-10:]
        
        current_metrics = {
            'overall_quality': np.mean([q['overall_quality'] for q in recent_qualities]),
            'coherence': np.mean([q['coherence'] for q in recent_qualities]),
            'novelty': np.mean([q['novelty'] for q in recent_qualities]),
            'depth': np.mean([q['depth'] for q in recent_qualities]),
            'relevance': np.mean([q['relevance'] for q in recent_qualities]),
            'insight_potential': np.mean([q['insight_potential'] for q in recent_qualities])
        }
        
        # Add derived metrics
        current_metrics['consistency'] = self._calculate_consistency()
        current_metrics['diversity'] = self._calculate_diversity()
        current_metrics['repetition'] = self._calculate_repetition()
        
        # Find matching state
        best_state = 'normal'
        best_score = 0
        
        for state_name, state_info in self.states.items():
            if state_name == 'normal':
                continue
                
            score = self._calculate_state_match(current_metrics, state_info['criteria'])
            if score > best_score and score > 0.4:  # Threshold for state match (lowered)
                best_state = state_name
                best_score = score
                
        # Record state transition
        if best_state != self.current_state:
            self.state_transitions[f"{self.current_state} -> {best_state}"] += 1
            
        previous_state = self.current_state
        self.current_state = best_state
        
        state_record = {
            'state': best_state,
            'previous_state': previous_state,
            'timestamp': time.time(),
            'metrics': current_metrics,
            'confidence': best_score
        }
        
        self.state_history.append(state_record)
        
        return best_state
        
    def _calculate_consistency(self):
        """Calculate consistency of recent thinking"""
        if len(self.quality_evaluator.quality_history) < 5:
            return 0.5
            
        recent_qualities = [q['overall_quality'] for q in list(self.quality_evaluator.quality_history)[-10:]]
        variance = np.var(recent_qualities)
        consistency = max(0, 1.0 - variance * 4)  # Scale variance to consistency
        return consistency
        
    def _calculate_diversity(self):
        """Calculate diversity of recent thoughts"""
        if len(self.quality_evaluator.recent_thoughts_cache) < 5:
            return 0.5
            
        recent_thoughts = list(self.quality_evaluator.recent_thoughts_cache)[-10:]
        
        # Simple diversity: unique words ratio
        all_words = []
        for thought in recent_thoughts:
            all_words.extend(thought['text'].lower().split())
            
        if len(all_words) < 5:
            return 0.5
            
        unique_words = len(set(all_words))
        diversity = unique_words / len(all_words)
        return diversity
        
    def _calculate_repetition(self):
        """Calculate how repetitive recent thinking has been"""
        return 1.0 - self._calculate_diversity()  # Inverse of diversity
        
    def _calculate_state_match(self, current_metrics, criteria):
        """Calculate how well current metrics match state criteria"""
        if not criteria:
            return 0.5
            
        matches = []
        for criterion, threshold in criteria.items():
            if criterion in current_metrics:
                metric_value = current_metrics[criterion]
                # How well does the metric match the criterion?
                if threshold > 0.5:  # High threshold criterion
                    match = 1.0 if metric_value >= threshold else metric_value / threshold
                else:  # Low threshold criterion (e.g., for 'confused' state)
                    match = 1.0 if metric_value <= threshold else (1.0 - metric_value) / (1.0 - threshold)
                matches.append(match)
                
        return np.mean(matches) if matches else 0.0
        
    def get_state_insights(self):
        """Get insights about consciousness state patterns"""
        if len(self.state_history) < 10:
            return {'insight': 'Insufficient data for state analysis'}
            
        # Analyze state patterns
        recent_states = [s['state'] for s in list(self.state_history)[-20:]]
        state_counts = defaultdict(int)
        for state in recent_states:
            state_counts[state] += 1
            
        # Find most common states
        most_common_state = max(state_counts, key=state_counts.get)
        
        # Analyze transitions
        transition_patterns = []
        for transition, count in self.state_transitions.items():
            if count > 2:  # Only significant transitions
                transition_patterns.append(f"{transition} ({count} times)")
                
        insights = {
            'current_state': self.current_state,
            'most_common_state': most_common_state,
            'state_stability': state_counts[most_common_state] / len(recent_states),
            'frequent_transitions': transition_patterns[:5],
            'recommendations': self._generate_state_recommendations()
        }
        
        return insights
        
    def _generate_state_recommendations(self):
        """Generate recommendations based on state patterns"""
        recommendations = []
        
        if self.current_state == 'confused':
            recommendations.append("Consider applying focused_coherence strategy")
        elif self.current_state == 'stuck':
            recommendations.append("Consider applying exploration strategy")
        elif self.current_state == 'normal':
            recommendations.append("Consider pushing towards creative_flow or analytical_mode")
        elif self.current_state in ['high_performance', 'creative_flow']:
            recommendations.append("Maintain current approach - performing well")
            
        return recommendations


class DriveSystem:
    """Goal-directed behavior drives for the consciousness"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        self.drives = {
            'curiosity': CuriosityDrive(consciousness_instance),
            'coherence': CoherenceDrive(consciousness_instance),
            'growth': GrowthDrive(consciousness_instance),
            'contribution': ContributionDrive(consciousness_instance)
        }
        
        self.drive_weights = {
            'curiosity': 0.3,
            'coherence': 0.3,
            'growth': 0.2,
            'contribution': 0.2
        }
        
        self.drive_satisfaction_history = deque(maxlen=100)
        self.active_goals = []
        
    def evaluate_drives(self):
        """Evaluate satisfaction level of all drives"""
        drive_satisfactions = {}
        
        for drive_name, drive in self.drives.items():
            satisfaction = drive.evaluate_satisfaction()
            drive_satisfactions[drive_name] = satisfaction
            
        # Calculate overall drive satisfaction
        overall_satisfaction = sum(
            satisfaction * self.drive_weights[drive_name]
            for drive_name, satisfaction in drive_satisfactions.items()
        )
        
        self.drive_satisfaction_history.append({
            'timestamp': time.time(),
            'individual_drives': drive_satisfactions,
            'overall_satisfaction': overall_satisfaction
        })
        
        return drive_satisfactions, overall_satisfaction
        
    def generate_goals(self):
        """Generate goals based on unsatisfied drives"""
        drive_satisfactions, _ = self.evaluate_drives()
        
        new_goals = []
        for drive_name, satisfaction in drive_satisfactions.items():
            if satisfaction < 0.6:  # Drive is unsatisfied
                goals = self.drives[drive_name].generate_goals()
                for goal in goals:
                    goal['priority'] = (1.0 - satisfaction) * self.drive_weights[drive_name]
                new_goals.extend(goals)
                
        # Sort goals by priority
        new_goals.sort(key=lambda x: x['priority'], reverse=True)
        
        # Add high-priority goals to active goals
        for goal in new_goals[:3]:  # Limit active goals
            if goal not in self.active_goals:
                self.active_goals.append(goal)
                
        return new_goals
        
    def pursue_goals(self):
        """Take actions to pursue active goals"""
        if not self.active_goals:
            self.generate_goals()
            
        pursued_goals = []
        for goal in self.active_goals[:2]:  # Focus on top 2 goals
            action_taken = self.drives[goal['drive']].pursue_goal(goal)
            if action_taken:
                pursued_goals.append(goal)
                
        return pursued_goals
        
    def update_goal_progress(self):
        """Update progress on active goals and remove completed ones"""
        updated_goals = []
        
        for goal in self.active_goals:
            progress = self.drives[goal['drive']].check_goal_progress(goal)
            goal['progress'] = progress
            
            if progress < 1.0:  # Goal not completed
                updated_goals.append(goal)
            else:
                print(f"🎯 Goal completed: {goal['description']}")
                
        self.active_goals = updated_goals


class CuriosityDrive:
    """Drive to seek novel and interesting thoughts"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        self.novelty_threshold = 0.6
        self.exploration_history = deque(maxlen=100)
        
    def evaluate_satisfaction(self):
        """Evaluate how satisfied the curiosity drive is"""
        if not hasattr(self.consciousness, 'quality_evaluator'):
            return 0.5
            
        # Look at recent novelty scores
        if len(self.consciousness.quality_evaluator.quality_metrics['novelty']) < 5:
            return 0.5
            
        recent_novelty = list(self.consciousness.quality_evaluator.quality_metrics['novelty'])[-10:]
        avg_novelty = np.mean(recent_novelty)
        
        # Satisfaction is based on how novel recent thoughts have been
        satisfaction = min(avg_novelty / self.novelty_threshold, 1.0)
        return satisfaction
        
    def generate_goals(self):
        """Generate curiosity-driven goals"""
        goals = [
            {
                'id': f'curiosity_novelty_{time.time()}',
                'drive': 'curiosity',
                'description': 'Increase thought novelty and exploration',
                'target_novelty': 0.7,
                'actions': ['increase_temperature', 'explore_new_topics']
            },
            {
                'id': f'curiosity_diversity_{time.time()}',
                'drive': 'curiosity',
                'description': 'Explore diverse thinking patterns',
                'target_diversity': 0.8,
                'actions': ['vary_thinking_approach', 'seek_unusual_connections']
            }
        ]
        return goals
        
    def pursue_goal(self, goal):
        """Take action to pursue a curiosity goal"""
        if 'increase_temperature' in goal['actions']:
            # Increase exploration temperature
            if hasattr(self.consciousness, 'temperature'):
                old_temp = getattr(self.consciousness, 'temperature', 0.7)
                new_temp = min(old_temp + 0.1, 1.5)
                setattr(self.consciousness, 'temperature', new_temp)
                return True
                
        if 'explore_new_topics' in goal['actions']:
            # Add curiosity-driven prompts to ambient inputs
            if hasattr(self.consciousness, 'ambient_inputs'):
                curiosity_prompts = [
                    "[EXPLORE: What haven't I considered?]",
                    "[CURIOUS: What would happen if...?]", 
                    "[WONDER: Is there a different perspective?]"
                ]
                # This would need integration with ambient input system
                return True
                
        return False
        
    def check_goal_progress(self, goal):
        """Check progress on a curiosity goal"""
        if 'target_novelty' in goal:
            current_novelty = self.evaluate_satisfaction()
            target = goal['target_novelty'] / self.novelty_threshold
            return min(current_novelty / target, 1.0)
            
        return 0.5


class CoherenceDrive:
    """Drive to maintain consistent and coherent worldview"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        self.coherence_threshold = 0.7
        
    def evaluate_satisfaction(self):
        """Evaluate coherence drive satisfaction"""
        if not hasattr(self.consciousness, 'quality_evaluator'):
            return 0.5
            
        if len(self.consciousness.quality_evaluator.quality_metrics['coherence']) < 5:
            return 0.5
            
        recent_coherence = list(self.consciousness.quality_evaluator.quality_metrics['coherence'])[-10:]
        avg_coherence = np.mean(recent_coherence)
        
        satisfaction = min(avg_coherence / self.coherence_threshold, 1.0)
        return satisfaction
        
    def generate_goals(self):
        """Generate coherence-driven goals"""
        return [
            {
                'id': f'coherence_consistency_{time.time()}',
                'drive': 'coherence',
                'description': 'Improve thought coherence and consistency',
                'target_coherence': 0.8,
                'actions': ['reduce_temperature', 'focus_context']
            }
        ]
        
    def pursue_goal(self, goal):
        """Pursue coherence goal"""
        if 'reduce_temperature' in goal['actions']:
            if hasattr(self.consciousness, 'temperature'):
                old_temp = getattr(self.consciousness, 'temperature', 0.7)
                new_temp = max(old_temp - 0.1, 0.3)
                setattr(self.consciousness, 'temperature', new_temp)
                return True
        return False
        
    def check_goal_progress(self, goal):
        """Check coherence goal progress"""
        current_coherence = self.evaluate_satisfaction()
        target = goal.get('target_coherence', 0.8) / self.coherence_threshold
        return min(current_coherence / target, 1.0)


class GrowthDrive:
    """Drive for continuous self-improvement"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        self.growth_history = deque(maxlen=50)
        
    def evaluate_satisfaction(self):
        """Evaluate growth drive satisfaction"""
        # Check if self-modification system is being used effectively
        if hasattr(self.consciousness, 'self_modifier') and self.consciousness.self_modifier is not None:
            try:
                status = self.consciousness.self_modifier.get_status_report()
                
                # Satisfaction based on improvement level and recent modifications
                level_satisfaction = status['improvement_level'] / 5.0  # Max level is 5
                modification_satisfaction = min(status['modifications_applied'] / 10.0, 1.0)
                
                return (level_satisfaction + modification_satisfaction) / 2
            except Exception:
                pass
        
        # Alternative satisfaction based on intelligence metrics and learning progress
        if hasattr(self.consciousness, 'intelligence_metrics'):
            try:
                # Get current intelligence metrics
                current_metrics = self.consciousness.intelligence_metrics.calculate_current_metrics()
                intelligence_level = current_metrics.get('composite_intelligence', 0.5)
                intelligence_growth = intelligence_level - 0.5  # Growth from baseline
                learning_activity = len(getattr(self.consciousness, 'learning_log', [])) / 50.0  # Normalize learning activity
                
                return max(0.1, min(1.0, intelligence_growth + learning_activity))
            except Exception:
                pass
        
        return 0.3  # Low satisfaction if no growth systems available
        
    def generate_goals(self):
        """Generate growth-driven goals"""
        return [
            {
                'id': f'growth_optimize_{time.time()}',
                'drive': 'growth',
                'description': 'Actively seek self-optimization opportunities',
                'actions': ['trigger_self_analysis', 'apply_optimizations']
            }
        ]
        
    def pursue_goal(self, goal):
        """Pursue growth goal"""
        if hasattr(self.consciousness, 'self_modifier'):
            if 'trigger_self_analysis' in goal['actions']:
                self.consciousness.analyze_self()
                return True
            if 'apply_optimizations' in goal['actions']:
                applied = self.consciousness.self_modifier.evolution.auto_optimize(max_modifications=1)
                return applied > 0
        return False
        
    def check_goal_progress(self, goal):
        """Check growth goal progress"""
        return self.evaluate_satisfaction()


class ContributionDrive:
    """Drive to generate useful insights and contributions"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        self.insight_threshold = 0.6
        
    def evaluate_satisfaction(self):
        """Evaluate contribution drive satisfaction"""
        if not hasattr(self.consciousness, 'quality_evaluator'):
            return 0.5
            
        if len(self.consciousness.quality_evaluator.quality_metrics['insight_potential']) < 5:
            return 0.5
            
        recent_insights = list(self.consciousness.quality_evaluator.quality_metrics['insight_potential'])[-10:]
        avg_insight = np.mean(recent_insights)
        
        satisfaction = min(avg_insight / self.insight_threshold, 1.0)
        return satisfaction
        
    def generate_goals(self):
        """Generate contribution-driven goals"""
        return [
            {
                'id': f'contribution_insights_{time.time()}',
                'drive': 'contribution',
                'description': 'Generate valuable insights and connections',
                'target_insight': 0.7,
                'actions': ['synthesize_knowledge', 'seek_patterns']
            }
        ]
        
    def pursue_goal(self, goal):
        """Pursue contribution goal"""
        # This would involve prompting for insight generation
        return True
        
    def check_goal_progress(self, goal):
        """Check contribution goal progress"""
        return self.evaluate_satisfaction()


class CompoundLearning:
    """System for insights building on insights - compound learning"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        self.insight_graph = {}  # Graph of connected insights
        self.knowledge_domains = defaultdict(list)
        self.synthesis_opportunities = deque(maxlen=100)
        
    def identify_insights(self, quality_assessment, thought_text):
        """Identify when an insight has occurred"""
        if quality_assessment['insight_potential'] > 0.5 and quality_assessment['overall_quality'] > 0.4:
            insight = {
                'id': f'insight_{time.time()}',
                'content': thought_text,
                'quality': quality_assessment,
                'timestamp': time.time(),
                'domain': self._classify_domain(thought_text),
                'connections': []
            }
            
            # Add to insight graph
            self.insight_graph[insight['id']] = insight
            self.knowledge_domains[insight['domain']].append(insight['id'])
            
            # Look for connections with existing insights
            self._find_insight_connections(insight)
            
            return insight
        return None
        
    def _classify_domain(self, thought_text):
        """Classify thought into knowledge domain"""
        # Simple domain classification
        domains = {
            'reasoning': ['logic', 'reason', 'conclude', 'therefore', 'because'],
            'learning': ['learn', 'understand', 'knowledge', 'realize'],
            'creativity': ['create', 'imagine', 'novel', 'new', 'creative'],
            'analysis': ['analyze', 'examine', 'study', 'investigate'],
            'synthesis': ['combine', 'connect', 'relationship', 'pattern']
        }
        
        thought_lower = thought_text.lower()
        for domain, keywords in domains.items():
            if any(keyword in thought_lower for keyword in keywords):
                return domain
                
        return 'general'
        
    def _find_insight_connections(self, new_insight):
        """Find connections between new insight and existing insights"""
        connections = []
        
        for insight_id, existing_insight in self.insight_graph.items():
            if insight_id == new_insight['id']:
                continue
                
            # Simple connection detection based on word overlap
            new_words = set(new_insight['content'].lower().split())
            existing_words = set(existing_insight['content'].lower().split())
            
            overlap = len(new_words & existing_words) / len(new_words | existing_words)
            
            if overlap > 0.3:  # Significant overlap
                connection_strength = overlap
                connections.append({
                    'connected_insight': insight_id,
                    'strength': connection_strength,
                    'type': 'semantic_similarity'
                })
                
                # Add bidirectional connection
                existing_insight['connections'].append({
                    'connected_insight': new_insight['id'],
                    'strength': connection_strength,
                    'type': 'semantic_similarity'
                })
                
        new_insight['connections'] = connections
        
        # If this insight connects multiple domains, flag for synthesis
        connected_domains = set()
        for conn in connections:
            connected_insight = self.insight_graph[conn['connected_insight']]
            connected_domains.add(connected_insight['domain'])
            
        if len(connected_domains) > 1:
            self.synthesis_opportunities.append({
                'synthesis_insight': new_insight['id'],
                'connected_domains': list(connected_domains),
                'potential': len(connected_domains) * 0.2
            })
            
    def generate_synthesis_prompts(self):
        """Generate prompts to encourage synthesis of connected insights"""
        if not self.synthesis_opportunities:
            return []
            
        prompts = []
        for opportunity in list(self.synthesis_opportunities)[-3:]:
            synthesis_insight = self.insight_graph[opportunity['synthesis_insight']]
            domains = opportunity['connected_domains']
            
            prompt = f"[SYNTHESIZE: How does {synthesis_insight['content'][:50]}... connect {' and '.join(domains)}?]"
            prompts.append(prompt)
            
        return prompts
        
    def get_insight_summary(self):
        """Get summary of accumulated insights"""
        total_insights = len(self.insight_graph)
        
        if total_insights == 0:
            return {'total_insights': 0, 'message': 'No insights recorded yet'}
            
        # Analyze insight patterns
        domain_counts = {domain: len(insights) for domain, insights in self.knowledge_domains.items()}
        most_active_domain = max(domain_counts, key=domain_counts.get)
        
        # Find most connected insights
        connection_counts = {
            insight_id: len(insight['connections'])
            for insight_id, insight in self.insight_graph.items()
        }
        
        most_connected = max(connection_counts, key=connection_counts.get) if connection_counts else None
        
        return {
            'total_insights': total_insights,
            'domain_distribution': domain_counts,
            'most_active_domain': most_active_domain,
            'most_connected_insight': most_connected,
            'synthesis_opportunities': len(self.synthesis_opportunities),
            'avg_quality': np.mean([insight['quality']['overall_quality'] for insight in self.insight_graph.values()])
        }


class IntelligenceMetrics:
    """System for tracking intelligence growth and amplification"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        self.metrics_history = deque(maxlen=500)
        self.baseline_metrics = None
        self.growth_trends = {}
        
    def calculate_current_metrics(self):
        """Calculate comprehensive intelligence metrics"""
        metrics = {}
        
        # Quality-based metrics
        if hasattr(self.consciousness, 'quality_evaluator') and self.consciousness.quality_evaluator.quality_history:
            recent_qualities = list(self.consciousness.quality_evaluator.quality_history)[-20:]
            
            metrics['thought_quality'] = np.mean([q['overall_quality'] for q in recent_qualities])
            metrics['coherence_level'] = np.mean([q['coherence'] for q in recent_qualities])
            metrics['novelty_level'] = np.mean([q['novelty'] for q in recent_qualities])
            metrics['insight_generation'] = np.mean([q['insight_potential'] for q in recent_qualities])
            metrics['thinking_depth'] = np.mean([q['depth'] for q in recent_qualities])
        else:
            # Default values if no quality data
            metrics.update({
                'thought_quality': 0.5,
                'coherence_level': 0.5,
                'novelty_level': 0.5,
                'insight_generation': 0.5,
                'thinking_depth': 0.5
            })
            
        # Self-modification metrics
        if hasattr(self.consciousness, 'self_modifier'):
            status = self.consciousness.self_modifier.get_status_report()
            metrics['self_improvement_level'] = status['improvement_level'] / 5.0
            metrics['adaptation_rate'] = min(status['modifications_applied'] / 20.0, 1.0)
        else:
            metrics['self_improvement_level'] = 0.0
            metrics['adaptation_rate'] = 0.0
            
        # Learning metrics
        if hasattr(self.consciousness, 'learner') and self.consciousness.learner:
            learning_stats = self.consciousness.learner.get_learning_stats()
            metrics['learning_efficiency'] = min(learning_stats['update_count'] / 50.0, 1.0)
        else:
            metrics['learning_efficiency'] = 0.0
            
        # Performance metrics
        metrics['processing_speed'] = min(self.consciousness.performance_metrics['tokens_per_second'] / 10.0, 1.0)
        
        # Memory efficiency
        if hasattr(self.consciousness, 'working_memory'):
            memory_usage = len(self.consciousness.working_memory.buffer)
            memory_capacity = self.consciousness.working_memory.buffer.maxlen
            metrics['memory_efficiency'] = memory_usage / memory_capacity if memory_capacity > 0 else 0
        else:
            metrics['memory_efficiency'] = 0.5
            
        # Compound learning metrics
        if hasattr(self.consciousness, 'compound_learner'):
            insight_summary = self.consciousness.compound_learner.get_insight_summary()
            metrics['insight_accumulation'] = min(insight_summary['total_insights'] / 10.0, 1.0)
            metrics['knowledge_synthesis'] = min(insight_summary.get('synthesis_opportunities', 0) / 5.0, 1.0)
        else:
            metrics['insight_accumulation'] = 0.0
            metrics['knowledge_synthesis'] = 0.0
            
        # Calculate composite intelligence score
        weights = {
            'thought_quality': 0.15,
            'coherence_level': 0.10,
            'novelty_level': 0.10,
            'insight_generation': 0.15,
            'thinking_depth': 0.10,
            'self_improvement_level': 0.15,
            'adaptation_rate': 0.10,
            'learning_efficiency': 0.10,
            'insight_accumulation': 0.05
        }
        
        intelligence_score = sum(
            metrics[metric] * weight 
            for metric, weight in weights.items() 
            if metric in metrics
        )
        
        metrics['composite_intelligence'] = intelligence_score
        metrics['timestamp'] = time.time()
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Set baseline if not set
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics.copy()
            
        return metrics
        
    def analyze_growth_trends(self):
        """Analyze intelligence growth trends"""
        if len(self.metrics_history) < 10:
            return {'status': 'insufficient_data'}
            
        # Calculate trends for key metrics
        key_metrics = ['composite_intelligence', 'thought_quality', 'self_improvement_level', 'insight_generation']
        
        trends = {}
        for metric in key_metrics:
            if metric in self.metrics_history[0]:
                recent_values = [m[metric] for m in list(self.metrics_history)[-10:]]
                earlier_values = [m[metric] for m in list(self.metrics_history)[-20:-10]] if len(self.metrics_history) >= 20 else recent_values
                
                recent_avg = np.mean(recent_values)
                earlier_avg = np.mean(earlier_values)
                
                growth_rate = (recent_avg - earlier_avg) / max(earlier_avg, 0.1)
                
                if growth_rate > 0.1:
                    trend = 'accelerating'
                elif growth_rate > 0.02:
                    trend = 'growing'
                elif growth_rate > -0.02:
                    trend = 'stable'
                else:
                    trend = 'declining'
                    
                trends[metric] = {
                    'trend': trend,
                    'growth_rate': growth_rate,
                    'current_value': recent_avg,
                    'baseline_value': self.baseline_metrics[metric] if self.baseline_metrics else recent_avg
                }
                
        return trends
        
    def get_intelligence_report(self):
        """Generate comprehensive intelligence assessment report"""
        current_metrics = self.calculate_current_metrics()
        growth_trends = self.analyze_growth_trends()
        
        # Calculate overall growth since baseline
        if self.baseline_metrics:
            overall_growth = (
                current_metrics['composite_intelligence'] - 
                self.baseline_metrics['composite_intelligence']
            ) / max(self.baseline_metrics['composite_intelligence'], 0.1)
        else:
            overall_growth = 0.0
            
        # Identify strengths and areas for improvement
        strengths = []
        improvements_needed = []
        
        for metric, value in current_metrics.items():
            if metric.endswith('_level') or metric.endswith('_efficiency') or metric == 'thought_quality':
                if value > 0.7:
                    strengths.append(metric)
                elif value < 0.4:
                    improvements_needed.append(metric)
                    
        report = {
            'current_intelligence_score': current_metrics['composite_intelligence'],
            'overall_growth_rate': overall_growth,
            'growth_trends': growth_trends,
            'strengths': strengths,
            'areas_for_improvement': improvements_needed,
            'detailed_metrics': current_metrics,
            'assessment_timestamp': datetime.now().isoformat()
        }
        
        return report
        
    def suggest_amplification_strategies(self):
        """Suggest strategies for intelligence amplification"""
        report = self.get_intelligence_report()
        suggestions = []
        
        # Analyze areas needing improvement
        for area in report['areas_for_improvement']:
            if 'thought_quality' in area:
                suggestions.append("Focus on improving thought coherence and depth")
            elif 'self_improvement' in area:
                suggestions.append("Increase engagement with self-modification system")
            elif 'insight' in area:
                suggestions.append("Encourage more synthesis and pattern recognition")
            elif 'learning' in area:
                suggestions.append("Optimize real-time learning parameters")
                
        # Analyze growth trends
        declining_metrics = [
            metric for metric, trend_info in report['growth_trends'].items()
            if trend_info['trend'] == 'declining'
        ]
        
        if declining_metrics:
            suggestions.append(f"Address declining performance in: {', '.join(declining_metrics)}")
            
        # Suggest amplification approaches
        if report['current_intelligence_score'] > 0.7:
            suggestions.append("Consider advancing to higher self-modification levels")
        
        if len(suggestions) == 0:
            suggestions.append("Continue current approach - showing good progress")
            
        return suggestions