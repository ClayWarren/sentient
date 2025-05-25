"""
Self-modification capabilities for Enhanced Consciousness System
Implements code introspection, safe modification, and bounded self-improvement
"""

import os
import time
import ast
import inspect
import tempfile
import shutil
import subprocess
import copy
from collections import deque
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np


class CodeIntrospector:
    """System for reading and understanding own source code"""
    
    def __init__(self, consciousness_file=None):
        self.consciousness_file = consciousness_file or 'enhanced_consciousness.py'
        self.model_file = os.path.join(os.path.dirname(self.consciousness_file), 'model.py')
        self.source_cache = {}
        self.architecture_map = {}
        self.modification_history = []
        
    def read_own_source(self, filepath=None):
        """Read and cache source code"""
        filepath = filepath or self.consciousness_file
        try:
            with open(filepath, 'r') as f:
                source = f.read()
            self.source_cache[filepath] = source
            return source
        except Exception as e:
            print(f"Error reading source {filepath}: {e}")
            return None
            
    def analyze_architecture(self):
        """Analyze own architecture through AST parsing"""
        source = self.read_own_source()
        if not source:
            return {}
            
        try:
            tree = ast.parse(source)
            architecture = {
                'classes': [],
                'functions': [],
                'key_variables': [],
                'imports': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    architecture['classes'].append({
                        'name': node.name,
                        'methods': methods,
                        'line': node.lineno
                    })
                elif isinstance(node, ast.FunctionDef) and node.name not in [m for c in architecture['classes'] for m in c['methods']]:
                    architecture['functions'].append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'line': node.lineno
                    })
                elif isinstance(node, ast.Import):
                    architecture['imports'].extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        architecture['imports'].extend([f"{node.module}.{alias.name}" for alias in node.names])
                    
            self.architecture_map = architecture
            return architecture
        except Exception as e:
            print(f"Error analyzing architecture: {e}")
            return {}
            
    def identify_performance_critical_sections(self):
        """Identify code sections that affect performance"""
        critical_sections = {
            'think_one_step': 'Core thinking loop - affects inference speed',
            'working_memory': 'Memory management - affects memory usage',
            'significance_calculation': 'Determines what gets remembered',
            'model_forward': 'Neural network forward pass',
            'buffer_management': 'Memory buffer operations'
        }
        return critical_sections
        
    def get_modifiable_parameters(self):
        """Get list of safe parameters that can be modified"""
        return {
            'hyperparameters': {
                'think_interval': 'Time between thoughts',
                'significance_threshold': 'Threshold for memory storage',
                'learning_rate': 'Real-time learning rate',
                'temperature': 'Randomness in generation',
                'top_k': 'Token selection constraint'
            },
            'buffer_sizes': {
                'working_memory_size': 'Size of working memory buffer',
                'thought_log_size': 'Size of thought history',
                'entropy_history_size': 'Size of entropy tracking'
            },
            'architectural': {
                'update_frequency': 'How often to learn',
                'consolidation_frequency': 'Memory consolidation interval'
            }
        }


class SafeModificationFramework:
    """Framework for safely modifying code with validation and rollback"""
    
    def __init__(self, introspector, consciousness_instance):
        self.introspector = introspector
        self.consciousness = consciousness_instance
        self.backup_dir = tempfile.mkdtemp(prefix='consciousness_backup_')
        self.modification_log = []
        self.safety_limits = self._initialize_safety_limits()
        
    def _initialize_safety_limits(self):
        """Set initial safety boundaries for modifications"""
        return {
            'max_think_interval': 2.0,
            'min_think_interval': 0.01,
            'max_significance_threshold': 0.9,
            'min_significance_threshold': 0.1,
            'max_learning_rate': 1e-3,
            'min_learning_rate': 1e-7,
            'max_buffer_size': 2048,
            'min_buffer_size': 50,
            'max_temperature': 2.0,
            'min_temperature': 0.1
        }
        
    def create_backup(self, filepath):
        """Create backup of current state"""
        backup_path = os.path.join(self.backup_dir, f"{os.path.basename(filepath)}.backup")
        shutil.copy2(filepath, backup_path)
        return backup_path
        
    def validate_modification(self, parameter, new_value):
        """Validate proposed modification against safety limits"""
        # Check against safety limits
        max_key = f'max_{parameter}'
        min_key = f'min_{parameter}'
        
        if max_key in self.safety_limits and new_value > self.safety_limits[max_key]:
            return False, f"Value {new_value} exceeds maximum {self.safety_limits[max_key]}"
        if min_key in self.safety_limits and new_value < self.safety_limits[min_key]:
            return False, f"Value {new_value} below minimum {self.safety_limits[min_key]}"
                
        return True, "Validation passed"
        
    def apply_parameter_modification(self, parameter, new_value):
        """Apply modification to running system"""
        valid, message = self.validate_modification(parameter, new_value)
        if not valid:
            return False, message
            
        try:
            # Handle different parameter types
            if parameter == 'significance_threshold':
                old_value = self.consciousness.working_memory.significance_threshold
                self.consciousness.working_memory.significance_threshold = new_value
            elif parameter == 'learning_rate' and hasattr(self.consciousness, 'learner'):
                old_value = self.consciousness.learner.learning_rate
                self.consciousness.learner.learning_rate = new_value
                # Update optimizer learning rate
                for param_group in self.consciousness.learner.optimizer.param_groups:
                    param_group['lr'] = new_value
            elif parameter == 'working_memory_size':
                old_value = self.consciousness.working_memory.buffer.maxlen
                # This requires special handling - see _resize_buffer method
                return self._resize_working_memory(new_value)
            else:
                old_value = getattr(self.consciousness, parameter, None)
                setattr(self.consciousness, parameter, new_value)
            
            self.modification_log.append({
                'timestamp': time.time(),
                'parameter': parameter,
                'old_value': old_value,
                'new_value': new_value,
                'success': True
            })
            
            return True, f"Successfully modified {parameter} from {old_value} to {new_value}"
        except Exception as e:
            return False, f"Error applying modification: {e}"
            
    def _resize_working_memory(self, new_size):
        """Safely resize working memory buffer"""
        try:
            old_size = self.consciousness.working_memory.buffer.maxlen
            current_memories = list(self.consciousness.working_memory.buffer)
            
            if len(current_memories) > new_size:
                # Keep most significant memories
                sorted_memories = sorted(current_memories, 
                                       key=lambda x: x.get('enhanced_significance', 0), 
                                       reverse=True)
                preserved_memories = sorted_memories[:new_size]
            else:
                preserved_memories = current_memories
                
            # Create new buffer with preserved memories
            self.consciousness.working_memory.buffer = deque(preserved_memories, maxlen=int(new_size))
            
            self.modification_log.append({
                'timestamp': time.time(),
                'parameter': 'working_memory_size',
                'old_value': old_size,
                'new_value': new_size,
                'success': True,
                'preserved_memories': len(preserved_memories)
            })
            
            return True, f"Resized working memory from {old_size} to {new_size}, preserved {len(preserved_memories)} memories"
        except Exception as e:
            return False, f"Error resizing working memory: {e}"
            
    def rollback_last_modification(self):
        """Rollback the last modification"""
        if not self.modification_log:
            return False, "No modifications to rollback"
            
        last_mod = self.modification_log[-1]
        if not last_mod['success']:
            return False, "Last modification was not successful"
            
        try:
            # Apply rollback based on parameter type
            parameter = last_mod['parameter']
            old_value = last_mod['old_value']
            
            if parameter == 'significance_threshold':
                self.consciousness.working_memory.significance_threshold = old_value
            elif parameter == 'learning_rate' and hasattr(self.consciousness, 'learner'):
                self.consciousness.learner.learning_rate = old_value
                for param_group in self.consciousness.learner.optimizer.param_groups:
                    param_group['lr'] = old_value
            elif parameter == 'working_memory_size':
                # Resize back to original size
                return self._resize_working_memory(old_value)
            else:
                setattr(self.consciousness, parameter, old_value)
                
            last_mod['rolled_back'] = True
            return True, f"Rolled back {parameter} to {old_value}"
        except Exception as e:
            return False, f"Error during rollback: {e}"
            
    def generate_code_modification(self, target_function, optimization_type):
        """Generate safe code modifications"""
        modifications = {
            'buffer_optimization': """# Optimized buffer management
if len(self.buffer) > self.max_size * 0.8:
    # Remove least significant entries
    sorted_buffer = sorted(self.buffer, key=lambda x: x['significance'])
    self.buffer = deque(sorted_buffer[-self.max_size//2:], maxlen=self.max_size)""",
            
            'significance_optimization': """# Enhanced significance detection
if len(self.entropy_history) > 20:
    recent_entropy_std = np.std(list(self.entropy_history)[-20:])
    if recent_entropy_std > 0.1:
        significance *= (1.0 + recent_entropy_std)"""
        }
        
        return modifications.get(optimization_type, "# No optimization available")


class LearningDrivenEvolution:
    """System for analyzing patterns and generating optimizations"""
    
    def __init__(self, consciousness_instance, modifier):
        self.consciousness = consciousness_instance
        self.modifier = modifier
        self.pattern_history = deque(maxlen=1000)
        self.optimization_candidates = []
        
    def analyze_thinking_patterns(self):
        """Analyze patterns in thinking to identify optimization opportunities"""
        if len(self.consciousness.thought_log) < 100:
            return []
            
        patterns = {
            'memory_utilization': self._analyze_memory_patterns(),
            'significance_distribution': self._analyze_significance_patterns(),
            'performance_trends': self._analyze_performance_patterns()
        }
        
        return self._generate_optimizations_from_patterns(patterns)
        
    def _analyze_memory_patterns(self):
        """Analyze how memory is being used"""
        memory_usage = len(self.consciousness.working_memory.buffer)
        max_memory = self.consciousness.working_memory.buffer.maxlen
        utilization = memory_usage / max_memory if max_memory > 0 else 0
        
        return {
            'utilization_rate': utilization,
            'is_frequently_full': utilization > 0.9,
            'is_underutilized': utilization < 0.3
        }
        
    def _analyze_significance_patterns(self):
        """Analyze significance score patterns"""
        recent_thoughts = list(self.consciousness.thought_log)[-50:]
        if not recent_thoughts:
            return {'avg_significance': 0.5}
            
        significances = [t['significance'] for t in recent_thoughts]
        return {
            'avg_significance': np.mean(significances),
            'significance_variance': np.var(significances),
            'high_significance_ratio': sum(1 for s in significances if s > 0.7) / len(significances)
        }
        
    def _analyze_performance_patterns(self):
        """Analyze performance trends"""
        return {
            'tokens_per_second': self.consciousness.performance_metrics['tokens_per_second'],
            'is_slowing_down': self.consciousness.performance_metrics['tokens_per_second'] < 5.0
        }
        
    def _generate_optimizations_from_patterns(self, patterns):
        """Generate optimization suggestions based on patterns"""
        optimizations = []
        
        # Memory optimizations
        if patterns['memory_utilization']['is_frequently_full']:
            current_size = self.consciousness.working_memory.buffer.maxlen
            optimizations.append({
                'type': 'increase_buffer_size',
                'parameter': 'working_memory_size',
                'current_value': current_size,
                'suggested_value': min(int(current_size * 1.5), 1024),
                'reason': 'Memory buffer frequently full'
            })
            
        elif patterns['memory_utilization']['is_underutilized']:
            current_size = self.consciousness.working_memory.buffer.maxlen
            optimizations.append({
                'type': 'decrease_buffer_size', 
                'parameter': 'working_memory_size',
                'current_value': current_size,
                'suggested_value': max(int(current_size * 0.7), 100),
                'reason': 'Memory buffer underutilized'
            })
            
        # Significance threshold optimization
        if patterns['significance_distribution']['high_significance_ratio'] > 0.8:
            current_threshold = self.consciousness.working_memory.significance_threshold
            optimizations.append({
                'type': 'increase_significance_threshold',
                'parameter': 'significance_threshold',
                'current_value': current_threshold,
                'suggested_value': min(current_threshold + 0.1, 0.9),
                'reason': 'Too many thoughts deemed significant'
            })
            
        # Performance optimizations
        if patterns['performance_trends']['is_slowing_down']:
            current_interval = getattr(self.consciousness, 'think_interval', 0.1)
            optimizations.append({
                'type': 'increase_think_interval',
                'parameter': 'think_interval',
                'current_value': current_interval,
                'suggested_value': min(current_interval * 1.2, 1.0),
                'reason': 'Performance degradation detected'
            })
            
        return optimizations
        
    def auto_optimize(self, max_modifications=1):
        """Automatically apply safe optimizations"""
        optimizations = self.analyze_thinking_patterns()
        applied = 0
        
        for opt in optimizations[:max_modifications]:
            success, message = self.modifier.apply_parameter_modification(
                opt['parameter'], opt['suggested_value']
            )
                
            if success:
                print(f"🔧 Auto-optimization: {opt['reason']} - {message}")
                applied += 1
            else:
                print(f"❌ Auto-optimization failed: {message}")
                
        return applied


class BoundedSelfImprovement:
    """Manages bounded self-improvement with safety checks"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        self.improvement_level = 1  # Start at level 1
        self.max_level = 5
        self.performance_baseline = None
        self.safety_violations = 0
        self.max_safety_violations = 3
        
    def evaluate_performance(self):
        """Evaluate current performance metrics"""
        metrics = {
            'tokens_per_second': self.consciousness.performance_metrics['tokens_per_second'],
            'memory_efficiency': len(self.consciousness.working_memory.buffer) / max(self.consciousness.working_memory.buffer.maxlen, 1),
            'learning_progress': len(getattr(self.consciousness, 'learning_log', [])),
            'thought_coherence': self._estimate_coherence()
        }
        
        if self.performance_baseline is None:
            self.performance_baseline = metrics.copy()
            
        return metrics
        
    def _estimate_coherence(self):
        """Estimate thought coherence (simplified)"""
        if len(self.consciousness.thought_log) < 10:
            return 0.5
            
        recent_thoughts = list(self.consciousness.thought_log)[-10:]
        avg_significance = np.mean([t['significance'] for t in recent_thoughts])
        return min(avg_significance * 2, 1.0)  # Simple coherence proxy
        
    def check_improvement_readiness(self):
        """Check if system is ready for next improvement level"""
        current_metrics = self.evaluate_performance()
        
        # Check if performance has improved or stayed stable
        performance_score = (
            current_metrics['tokens_per_second'] / max(self.performance_baseline['tokens_per_second'], 0.1) +
            current_metrics['thought_coherence'] / max(self.performance_baseline['thought_coherence'], 0.1)
        ) / 2
        
        ready = (
            performance_score > 0.95 and  # Performance maintained
            self.safety_violations < self.max_safety_violations and
            self.improvement_level < self.max_level
        )
        
        return ready, performance_score
        
    def advance_improvement_level(self):
        """Advance to next improvement level with expanded permissions"""
        ready, score = self.check_improvement_readiness()
        if not ready:
            return False, f"Not ready for advancement. Performance score: {score:.3f}"
            
        self.improvement_level += 1
        print(f"🎉 Advanced to improvement level {self.improvement_level}")
        
        # Expand modification permissions based on level
        expanded_permissions = self._get_level_permissions()
        return True, f"Advanced to level {self.improvement_level}: {expanded_permissions}"
        
    def _get_level_permissions(self):
        """Get permissions for current improvement level"""
        permissions = {
            1: "Basic hyperparameter tuning",
            2: "Buffer size modifications", 
            3: "Learning algorithm adjustments",
            4: "Attention pattern modifications",
            5: "Architecture modifications"
        }
        return permissions.get(self.improvement_level, "Unknown level")


class SelfModifyingConsciousness:
    """Enhanced consciousness with self-modification capabilities"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        self.introspector = CodeIntrospector()
        self.modifier = SafeModificationFramework(self.introspector, consciousness_instance)
        self.evolution = LearningDrivenEvolution(consciousness_instance, self.modifier)
        self.bounded_improvement = BoundedSelfImprovement(consciousness_instance)
        
        # Self-modification state
        self.modification_enabled = True
        self.auto_optimization_interval = 1000  # Every N iterations
        self.last_auto_optimization = 0
        
    def analyze_self(self):
        """Perform self-analysis and introspection"""
        print("🔍 Performing self-analysis...")
        
        # Analyze architecture
        architecture = self.introspector.analyze_architecture()
        print(f"📋 Found {len(architecture.get('classes', []))} classes, {len(architecture.get('functions', []))} functions")
        
        # Analyze current performance
        performance = self.bounded_improvement.evaluate_performance()
        print(f"⚡ Performance: {performance['tokens_per_second']:.1f} tokens/sec, coherence: {performance['thought_coherence']:.3f}")
        
        # Analyze patterns
        patterns = self.evolution.analyze_thinking_patterns()
        if patterns:
            print(f"🧠 Found {len(patterns)} optimization opportunities")
            for opt in patterns[:3]:  # Show top 3
                print(f"   • {opt['reason']}: {opt['parameter']} {opt['current_value']} → {opt['suggested_value']}")
        
        return {
            'architecture': architecture,
            'performance': performance,
            'optimization_opportunities': patterns
        }
        
    def auto_modify_if_needed(self):
        """Check if auto-modification should occur"""
        if not self.modification_enabled:
            return False
            
        iterations_since_last = self.consciousness.iteration_count - self.last_auto_optimization
        
        if iterations_since_last >= self.auto_optimization_interval:
            print(f"🤖 Auto-modification check at iteration {self.consciousness.iteration_count}")
            
            # Perform auto-optimization
            optimizations_applied = self.evolution.auto_optimize(max_modifications=1)
            
            # Check for improvement level advancement
            if optimizations_applied > 0:
                ready, score = self.bounded_improvement.check_improvement_readiness()
                if ready:
                    success, message = self.bounded_improvement.advance_improvement_level()
                    if success:
                        print(f"🚀 {message}")
                        
            self.last_auto_optimization = self.consciousness.iteration_count
            return optimizations_applied > 0
            
        return False
        
    def manual_optimize(self, parameter, value):
        """Manually apply optimization"""
        if not self.modification_enabled:
            return False, "Self-modification disabled"
            
        success, message = self.modifier.apply_parameter_modification(parameter, value)
        
        if success:
            print(f"✅ Manual optimization: {message}")
        else:
            print(f"❌ Manual optimization failed: {message}")
            
        return success, message
        
    def rollback_last_change(self):
        """Rollback the last modification"""
        success, message = self.modifier.rollback_last_modification()
        
        if success:
            print(f"↩️ Rollback successful: {message}")
        else:
            print(f"❌ Rollback failed: {message}")
            
        return success, message
        
    def get_modification_history(self):
        """Get history of all modifications"""
        return self.modifier.modification_log
        
    def enable_self_modification(self, enabled=True):
        """Enable or disable self-modification"""
        self.modification_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"🔧 Self-modification {status}")
        
    def get_status_report(self):
        """Get comprehensive status report"""
        report = {
            'self_modification_enabled': self.modification_enabled,
            'improvement_level': self.bounded_improvement.improvement_level,
            'safety_violations': self.bounded_improvement.safety_violations,
            'modifications_applied': len(self.modifier.modification_log),
            'performance_metrics': self.bounded_improvement.evaluate_performance(),
            'recent_modifications': self.modifier.modification_log[-5:] if self.modifier.modification_log else []
        }
        
        return report