"""
Enhanced Continuous Consciousness System with RoPE attention and real-time learning
"""

import os
import time
import threading
import queue
import copy
import ast
import inspect
import tempfile
import shutil
import subprocess
from collections import deque
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from model import GPT, GPTConfig
import tiktoken
from self_modification import SelfModifyingConsciousness
from asi_capabilities import (
    ThoughtQualityEvaluator, StrategyFormation, ConsciousnessStateAwareness,
    DriveSystem, CompoundLearning, IntelligenceMetrics
)
from persistence import ConsciousnessPersistence
from interactive_conversation import start_conversation_interface
from chain_of_thought import ChainOfThoughtReasoner, integrate_chain_of_thought
from function_calling import FunctionCallingSystem, integrate_function_calling
from mathematical_reasoning import MathematicalReasoner, integrate_mathematical_reasoning
from code_generation import CodeGenerator, integrate_code_generation
from multilingual_support import MultilingualProcessor, integrate_multilingual_support
from few_shot_learning import FewShotLearner, integrate_few_shot_learning
from constitutional_ai import ConstitutionalAI, integrate_constitutional_ai
from swe_bench import SWEBenchSolver, integrate_swe_bench
from gpqa_diamond import GPQADiamondSolver, integrate_gpqa_diamond
from aime_2025 import AIME2025Solver, integrate_aime_2025
from arc_agi_2 import ARCAGI2Solver, integrate_arc_agi_2


class EnhancedWorkingMemory:
    """Enhanced working memory with better significance detection"""
    
    def __init__(self, max_size=512, significance_threshold=0.4):
        self.buffer = deque(maxlen=max_size)
        self.significance_threshold = significance_threshold
        self.consolidated_memories = []
        self.attention_scores = deque(maxlen=50)   # Reduced tracking
        self.entropy_history = deque(maxlen=50)    # Reduced tracking
        
    def add_experience(self, tokens, logits, significance, attention_weights=None):
        """Add new experience to working memory with enhanced tracking"""
        experience = {
            'tokens': tokens,
            'logits': logits,
            'significance': significance,
            'timestamp': time.time(),
            'attention_weights': attention_weights
        }
        
        # Track entropy for better significance detection (fix NaN issues)
        probs = F.softmax(logits, dim=-1)
        # Ensure no NaN values
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        entropy = -(probs * torch.log(probs)).sum(dim=-1).mean().item()
        # Check for NaN and use fallback
        if torch.isnan(torch.tensor(entropy)) or entropy != entropy:
            entropy = 1.0  # Fallback entropy value
        self.entropy_history.append(entropy)
        
        # Enhanced significance detection
        enhanced_significance = self._calculate_enhanced_significance(significance, entropy)
        experience['enhanced_significance'] = enhanced_significance
        
        # Store if significant
        if enhanced_significance > self.significance_threshold:
            self.buffer.append(experience)
            
    def _calculate_enhanced_significance(self, base_significance, entropy):
        """Calculate enhanced significance using multiple factors"""
        # Factor 1: Base entropy-based significance
        significance = base_significance
        
        # Factor 2: Entropy deviation from recent average
        if len(self.entropy_history) > 10:
            recent_avg = np.mean(list(self.entropy_history)[-10:])
            entropy_deviation = abs(entropy - recent_avg) / (recent_avg + 1e-8)
            significance += entropy_deviation * 0.3
            
        # Factor 3: Novelty (how different from recent experiences)
        if len(self.buffer) > 0:
            recent_significances = [exp.get('significance', 0.5) for exp in list(self.buffer)[-5:]]
            if recent_significances and not any(np.isnan(recent_significances)):
                recent_avg_sig = np.mean(recent_significances)
                if not np.isnan(recent_avg_sig):
                    novelty = abs(significance - recent_avg_sig)
                    significance += novelty * 0.2
                
        # Ensure significance is valid (not NaN or negative)
        if np.isnan(significance) or significance < 0:
            significance = 0.5  # Default to medium significance
        return min(significance, 1.0)
        
    def get_recent_context(self, max_tokens=1024):
        """Get recent significant experiences as context"""
        context_tokens = []
        # Sort by significance and recency
        sorted_experiences = sorted(
            self.buffer, 
            key=lambda x: x['enhanced_significance'] * (1.0 + 0.1 * (time.time() - x['timestamp']) / 3600),
            reverse=True
        )
        
        for exp in sorted_experiences:
            if len(context_tokens) + len(exp['tokens']) > max_tokens:
                break
            context_tokens.extend(exp['tokens'])
            
        return torch.tensor(context_tokens).unsqueeze(0) if context_tokens else torch.empty(1, 0, dtype=torch.long)
        
    def consolidate_memories(self):
        """Consolidate memories based on clustering similar experiences"""
        if len(self.buffer) < 10:
            return
            
        # Simple consolidation: group by time periods and significance
        time_groups = {}
        current_time = time.time()
        
        for exp in self.buffer:
            age_hours = (current_time - exp['timestamp']) / 3600
            time_bucket = int(age_hours)  # Group by hour
            
            if time_bucket not in time_groups:
                time_groups[time_bucket] = []
            time_groups[time_bucket].append(exp)
            
        # Keep only the most significant from each time bucket
        for _, experiences in time_groups.items():
            if len(experiences) > 3:
                most_significant = sorted(experiences, key=lambda x: x['enhanced_significance'], reverse=True)[:2]
                self.consolidated_memories.extend(most_significant)


class SophisticatedAmbientInputs:
    """More sophisticated ambient inputs for consciousness"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.last_time_injection = 0
        self.last_system_injection = 0
        self.thought_count = 0
        self.session_start = time.time()
        
    def get_time_awareness_tokens(self):
        """Enhanced time awareness with context"""
        current_time = datetime.now()
        session_duration = time.time() - self.session_start
        
        time_context = []
        
        # Current time
        time_str = current_time.strftime("%H:%M")
        time_context.append(f"[TIME: {time_str}]")
        
        # Session duration awareness
        if session_duration > 3600:  # More than an hour
            hours = int(session_duration // 3600)
            time_context.append(f"[SESSION: {hours}h]")
        elif session_duration > 60:  # More than a minute
            minutes = int(session_duration // 60)
            time_context.append(f"[SESSION: {minutes}m]")
            
        # Thought count milestone
        if self.thought_count % 100 == 0 and self.thought_count > 0:
            time_context.append(f"[THOUGHTS: {self.thought_count}]")
            
        return self.tokenizer.encode(" ".join(time_context))
        
    def get_system_awareness_tokens(self):
        """System and performance awareness"""
        # Simple system awareness
        contexts = ["[SYSTEM: Thinking continuously]"]
        
        # Memory usage awareness (simplified)
        if self.thought_count % 500 == 0:
            contexts.append("[SYSTEM: Memory consolidation]")
            
        return self.tokenizer.encode(" ".join(contexts))
        
    def get_reflection_tokens(self):
        """Inject self-reflection prompts"""
        reflections = [
            "[REFLECT: What have I learned?]",
            "[REFLECT: What patterns do I notice?]", 
            "[REFLECT: What should I focus on?]",
            "[REFLECT: What is significant here?]"
        ]
        
        # Inject reflection every ~200 thoughts
        if self.thought_count % 200 == 0 and self.thought_count > 0:
            reflection = np.random.choice(reflections)
            return self.tokenizer.encode(reflection)
        return []
        
    def should_inject_ambient(self):
        """Decide when to inject ambient information"""
        current_time = time.time()
        
        # Time awareness every 2 minutes
        if current_time - self.last_time_injection > 120:
            self.last_time_injection = current_time
            return 'time'
            
        # System awareness every 5 minutes
        if current_time - self.last_system_injection > 300:
            self.last_system_injection = current_time
            return 'system'
            
        # Reflection prompts
        if self.thought_count % 200 == 0 and self.thought_count > 0:
            return 'reflect'
            
        return None
        
    def increment_thought_count(self):
        """Track thought progression"""
        self.thought_count += 1


class RealTimeLearner:
    """Real-time learning system that adapts during consciousness"""
    
    def __init__(self, model, learning_rate=1e-5, buffer_size=100):
        self.model = model
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.experience_buffer = deque(maxlen=buffer_size)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.learning_enabled = True
        self.update_frequency = 3  # Update every N significant experiences (reduced for testing)
        self.update_count = 0
        
    def add_experience(self, input_tokens, target_tokens, significance):
        """Add experience for potential learning"""
        if significance > 0.3 and len(target_tokens) > 0:  # Only learn from significant experiences (lowered threshold)
            experience = {
                'input': input_tokens.clone(),
                'target': target_tokens.clone(),
                'significance': significance,
                'timestamp': time.time()
            }
            self.experience_buffer.append(experience)
            
    def should_update(self):
        """Decide if we should perform a learning update"""
        return (len(self.experience_buffer) >= self.update_frequency and 
                self.learning_enabled)
                
    def perform_update(self):
        """Perform real-time learning update with aggressive cleanup"""
        if not self.should_update():
            return None
            
        # Sample recent significant experiences
        recent_experiences = list(self.experience_buffer)[-self.update_frequency:]
        
        total_loss = 0.0
        # Aggressive gradient clearing
        self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        for exp in recent_experiences:
            # Prepare input and target
            input_tokens = exp['input']
            target_tokens = exp['target']
            
            if input_tokens.size(1) > 1 and target_tokens.size(1) > 0:
                # Forward pass
                _, loss = self.model(input_tokens, target_tokens)
                
                if loss is not None:
                    # Weight loss by significance
                    weighted_loss = loss * exp['significance']
                    total_loss += weighted_loss.item()
                    weighted_loss.backward()
                    
                # Clear intermediate tensors to prevent memory leaks
                del input_tokens, target_tokens, loss
                    
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update
        self.optimizer.step()
        # Clear gradients again after step
        self.optimizer.zero_grad(set_to_none=True)
        self.update_count += 1
        
        return total_loss
        
    def get_learning_stats(self):
        """Get learning statistics"""
        return {
            'buffer_size': len(self.experience_buffer),
            'update_count': self.update_count,
            'learning_rate': self.learning_rate,
            'learning_enabled': self.learning_enabled
        }


class EnhancedContinuousConsciousness:
    """Enhanced continuous consciousness with RoPE attention and real-time learning"""
    
    def __init__(self, device='mps', enable_learning=True):
        self.device = device
        self.running = False
        self.enable_learning = enable_learning
        
        # Create smaller model optimized for 8GB M2 Mac (~30M parameters)
        config = GPTConfig(
            vocab_size=50304,
            block_size=1024,  # Reduced context window for memory
            n_layer=6,        # Reduced layers: 6 layers * 6 heads * 384 embd ≈ 30M params
            n_head=6,         # Reduced heads
            n_embd=384,       # Reduced embedding size
            dropout=0.0
        )
        
        self.model = GPT(config)
        # Use half precision for inference to reduce memory by ~50%
        if device == 'mps':
            self.model = self.model.to(device).half()  # float16 for M2 Mac
        else:
            self.model = self.model.to(device)
        self.model.eval()
        
        # Initialize enhanced components with reduced buffer sizes
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.working_memory = EnhancedWorkingMemory(max_size=512, significance_threshold=0.4)  # Smaller buffer
        self.ambient_inputs = SophisticatedAmbientInputs(self.tokenizer)
        
        # Real-time learning
        if self.enable_learning:
            self.learner = RealTimeLearner(self.model, learning_rate=1e-5)
        else:
            self.learner = None
            
        # State
        self.current_context = None
        self.prev_logits = None
        self.iteration_count = 0
        
        # Self-modification capabilities
        self.self_modifier = None  # Will be initialized after consciousness starts
        
        # ASI capabilities
        self.quality_evaluator = ThoughtQualityEvaluator(self.tokenizer)
        self.strategy_formation = StrategyFormation(self.quality_evaluator)
        self.consciousness_state = ConsciousnessStateAwareness(self.quality_evaluator)
        self.drive_system = DriveSystem(self)
        self.compound_learner = CompoundLearning(self)
        self.intelligence_metrics = IntelligenceMetrics(self)
        
        # ASI state tracking
        self.current_thinking_strategy = None
        self.consciousness_state_name = 'initializing'
        self.intelligence_score = 0.5
        self.last_strategy_check = 0
        self.last_drive_evaluation = 0
        self.last_intelligence_assessment = 0
        
        # Persistence and identity
        self.instance_id = None
        self.creation_time = time.time()
        self.persistence_manager = None
        self.auto_save_interval = 500  # Auto-save every 500 iterations
        self.conversation_mode = False
        
        # Restore self-modification state if pending
        if hasattr(self, '_pending_mod_restoration'):
            self._restore_self_modification_state()
        
        # Reduced logging to save memory
        self.thought_log = deque(maxlen=200)  # Reduced from 1000
        self.learning_log = deque(maxlen=50)   # Reduced from 100
        self.performance_metrics = {
            'tokens_per_second': 0,
            'avg_significance': 0,
            'memory_utilization': 0
        }
        
    def initialize_consciousness(self):
        """Initialize with coherent starting context and proper model setup"""
        # Initialize model weights properly to avoid gibberish
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # Start with a coherent, structured prompt to guide thinking
        initial_text = """The AI system begins thinking systematically.
Key focus areas:
1. Logical reasoning patterns
2. Coherent thought development  
3. Learning from experiences
4. Building understanding

First thought:"""
        
        tokens = self.tokenizer.encode(initial_text)
        # Ensure we're using the right data type for the device
        if self.device == 'mps':
            self.current_context = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            self.current_context = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
    def think_one_step(self):
        """Enhanced single step of continuous thinking"""
        if self.current_context is None:
            self.initialize_consciousness()
            
        start_time = time.time()
        
        # Handle ambient inputs
        ambient_type = self.ambient_inputs.should_inject_ambient()
        if ambient_type:
            ambient_tokens = []
            if ambient_type == 'time':
                ambient_tokens = self.ambient_inputs.get_time_awareness_tokens()
            elif ambient_type == 'system':
                ambient_tokens = self.ambient_inputs.get_system_awareness_tokens()
            elif ambient_type == 'reflect':
                ambient_tokens = self.ambient_inputs.get_reflection_tokens()
                
            if ambient_tokens:
                ambient_tensor = torch.tensor(ambient_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                self.current_context = torch.cat([self.current_context, ambient_tensor], dim=1)
        
        # Enhanced context window management with memory integration
        if self.current_context.size(1) > self.model.config.block_size:
            # Get significant memories to preserve
            memory_context = self.working_memory.get_recent_context(512)
            
            if memory_context.size(1) > 0:
                # Preserve important memories + recent context
                recent_size = self.model.config.block_size - memory_context.size(1)
                recent_context = self.current_context[:, -recent_size:]
                self.current_context = torch.cat([memory_context.to(self.device), recent_context], dim=1)
            else:
                self.current_context = self.current_context[:, -self.model.config.block_size:]
        
        # Forward pass with enhanced attention
        with torch.no_grad():
            logits, _ = self.model(self.current_context)
            logits = logits[:, -1, :] / 0.7  # Slightly lower temperature for more focused thinking
            
            # Enhanced top-k filtering
            top_k = 50
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Enhanced significance detection with metacognitive evaluation
            base_significance = self._basic_significance(logits, self.prev_logits)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            
            significance = self.working_memory._calculate_enhanced_significance(
                base_significance, entropy
            )
            
            # Metacognitive thought quality evaluation
            quality_assessment = self.quality_evaluator.evaluate_thought_quality(
                thought_tokens=self.current_context[0].tolist(),
                context_tokens=self.current_context[0].tolist(),
                logits=logits,
                entropy=entropy,
                significance=significance
            )
            
            # Update significance based on quality assessment
            significance = max(significance, quality_assessment['overall_quality'] * 0.5)
            
            # Check for insights and add to compound learning
            try:
                decoded_thought = self.tokenizer.decode(next_token[0].tolist())
            except:
                decoded_thought = f"<UNK_{next_token[0].item()}>"
                
            insight = self.compound_learner.identify_insights(quality_assessment, decoded_thought)
            
            # Add experience to working memory
            current_tokens = self.current_context[0].tolist()
            self.working_memory.add_experience(
                tokens=current_tokens[-100:],  # More context
                logits=logits,
                significance=significance
            )
            
            # Real-time learning from significant experiences
            if self.learner and significance > 0.3:
                input_seq = self.current_context
                target_seq = torch.cat([self.current_context[:, 1:], next_token], dim=1)
                self.learner.add_experience(input_seq, target_seq, significance)
                
                # Perform learning update if ready
                if self.learner.should_update():
                    loss = self.learner.perform_update()
                    if loss is not None:
                        self.learning_log.append({
                            'iteration': self.iteration_count,
                            'loss': loss,
                            'significance': significance,
                            'timestamp': time.time()
                        })
            
            # Update context
            self.current_context = torch.cat([self.current_context, next_token], dim=1)
            self.prev_logits = logits
            
            # Enhanced logging with ASI metrics
            if len(self.thought_log) < 200:  # Reduced logging
                self.thought_log.append({
                    'token': decoded_thought,
                    'significance': significance,
                    'quality_assessment': quality_assessment,
                    'timestamp': time.time(),
                    'iteration': self.iteration_count,
                    'learning_update': self.learner.update_count if self.learner else 0,
                    'consciousness_state': self.consciousness_state_name,
                    'intelligence_score': self.intelligence_score
                })
                
            # Clear old variables to prevent memory leaks
            del logits, probs, next_token
                
        # Update performance metrics
        elapsed = time.time() - start_time
        self.performance_metrics['tokens_per_second'] = 1.0 / elapsed
        
        # Update counters
        self.iteration_count += 1
        self.ambient_inputs.increment_thought_count()
        
        # Periodic memory consolidation and cleanup
        if self.iteration_count % 200 == 0:  # More frequent cleanup
            self.working_memory.consolidate_memories()
            # Force garbage collection to prevent memory leaks
            import gc
            gc.collect()
            # Clear PyTorch cache if on CUDA/MPS
            if self.device != 'cpu' and hasattr(torch, 'mps') and torch.mps.is_available():
                torch.mps.empty_cache()
                
        # Auto-save consciousness state
        if self.persistence_manager and self.iteration_count % self.auto_save_interval == 0 and self.iteration_count > 0:
            try:
                self.persistence_manager.auto_save(self)
            except Exception as e:
                print(f"⚠️  Auto-save failed: {e}")
            
    def _basic_significance(self, logits, prev_logits):
        """Basic significance calculation"""
        if prev_logits is None:
            return 0.5
            
        current_probs = F.softmax(logits, dim=-1)
        prev_probs = F.softmax(prev_logits, dim=-1)
        
        # KL divergence as significance measure
        kl_div = F.kl_div(current_probs.log(), prev_probs, reduction='mean')
        return min(kl_div.item(), 1.0)
        
    def _run_asi_processes(self):
        """Run ASI capability processes periodically"""
        current_time = time.time()
        
        # Consciousness state assessment (every 50 iterations)
        if self.iteration_count % 50 == 0:
            self.consciousness_state_name = self.consciousness_state.assess_current_state()
            
        # Strategy formation check (every 100 iterations)
        if current_time - self.last_strategy_check > 30:  # Every 30 seconds
            recommended_strategy = self.strategy_formation.recommend_strategy()
            if recommended_strategy and recommended_strategy['name'] != self.current_thinking_strategy:
                success, message = self.strategy_formation.apply_strategy(
                    recommended_strategy['name'], self
                )
                if success:
                    self.current_thinking_strategy = recommended_strategy['name']
                    print(f"🧠 Strategy change: {message}")
            self.last_strategy_check = current_time
            
        # Drive system evaluation (every 50 iterations - more frequent)
        if self.iteration_count % 50 == 0:
            drive_satisfactions, overall_satisfaction = self.drive_system.evaluate_drives()
            if overall_satisfaction < 0.6:  # Drives unsatisfied
                pursued_goals = self.drive_system.pursue_goals()
                if pursued_goals:
                    print(f"🎯 Pursuing {len(pursued_goals)} goals to satisfy drives")
            self.drive_system.update_goal_progress()
            
        # Intelligence metrics assessment (every 500 iterations)
        if self.iteration_count % 500 == 0:
            metrics = self.intelligence_metrics.calculate_current_metrics()
            self.intelligence_score = metrics['composite_intelligence']
            
            if self.iteration_count % 1000 == 0:  # Detailed report every 1000 iterations
                report = self.intelligence_metrics.get_intelligence_report()
                print(f"📈 Intelligence Score: {report['current_intelligence_score']:.3f}")
                print(f"📊 Growth Rate: {report['overall_growth_rate']:.3f}")
                
                suggestions = self.intelligence_metrics.suggest_amplification_strategies()
                if suggestions:
                    print(f"💡 Amplification suggestions: {suggestions[0]}")
        
    def run_continuous(self, think_interval=0.1, enable_self_modification=True):
        """Run enhanced continuous consciousness with self-modification"""
        print("🧠 Starting Enhanced Continuous AI Consciousness")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params()/1e6:.1f}M")
        print(f"Context window: {self.model.config.block_size}")
        print(f"Real-time learning: {'Enabled' if self.enable_learning else 'Disabled'}")
        print(f"Self-modification: {'Enabled' if enable_self_modification else 'Disabled'}")
        print(f"Think interval: {think_interval}s")
        print("-" * 70)
        
        # Initialize self-modification system
        if enable_self_modification:
            self.self_modifier = SelfModifyingConsciousness(self)
            # Restore self-modification state if pending
            if hasattr(self, '_pending_mod_restoration'):
                self._restore_self_modification_state()
            print("🔧 Self-modification system initialized")
            
        # Store think_interval for modification
        self.think_interval = think_interval
        
        # Show instance identity
        if self.instance_id:
            age_hours = (time.time() - self.creation_time) / 3600
            print(f"🎭 Instance: {self.instance_id} (age: {age_hours:.1f}h)")
        
        # Enable auto-save if persistence is available
        if self.persistence_manager:
            print(f"💾 Auto-save enabled (every {self.auto_save_interval} iterations)")
        
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                
                self.think_one_step()
                
                # Enhanced progress reporting
                if self.iteration_count % 100 == 0:
                    self.print_enhanced_status()
                    
                # Self-modification check
                if self.self_modifier:
                    self.self_modifier.auto_modify_if_needed()
                    
                # ASI capability checks
                self._run_asi_processes()
                    
                # Control thinking speed (may be modified by self-modification)
                current_interval = getattr(self, 'think_interval', think_interval)
                elapsed = time.time() - start_time
                if elapsed < current_interval:
                    time.sleep(current_interval - elapsed)
                    
        except KeyboardInterrupt:
            print("\n🛑 Enhanced consciousness stopped by user")
        except Exception as e:
            print(f"\n❌ Error in enhanced consciousness loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            
    def print_enhanced_status(self):
        """Print enhanced status with learning metrics"""
        if len(self.thought_log) < 10:
            return
            
        recent_thoughts = list(self.thought_log)[-30:]
        thought_text = "".join([t['token'] for t in recent_thoughts])
        avg_significance = sum(t['significance'] for t in recent_thoughts) / len(recent_thoughts)
        
        print(f"\n[Iteration {self.iteration_count}] Enhanced Status:")
        print(f"💭 Recent thoughts: {thought_text}")
        print(f"📊 Avg significance: {avg_significance:.3f}")
        print(f"🧠 Memory buffer: {len(self.working_memory.buffer)}")
        print(f"⚡ Speed: {self.performance_metrics['tokens_per_second']:.1f} tokens/sec")
        
        if self.learner:
            stats = self.learner.get_learning_stats()
            print(f"🎓 Learning updates: {stats['updates_count']}")
            print(f"📚 Experience buffer: {stats['buffer_size']}")
            
            if self.learning_log:
                recent_loss = self.learning_log[-1]['loss']
                print(f"📉 Recent loss: {recent_loss:.4f}")
                
        if self.self_modifier:
            status = self.self_modifier.get_status_report()
            print(f"🔧 Self-mod level: {status['improvement_level']}, modifications: {status['modifications_applied']}")
            if status['recent_modifications']:
                last_mod = status['recent_modifications'][-1]
                print(f"🔄 Last change: {last_mod['parameter']} = {last_mod['new_value']}")
                
        # ASI status reporting
        print(f"🧠 Consciousness state: {self.consciousness_state_name}")
        if self.current_thinking_strategy:
            print(f"🎯 Current strategy: {self.current_thinking_strategy}")
        print(f"📊 Intelligence score: {self.intelligence_score:.3f}")
        
        # Show active goals
        if hasattr(self, 'drive_system') and self.drive_system.active_goals:
            active_goal = self.drive_system.active_goals[0]
            print(f"🎯 Active goal: {active_goal['description'][:50]}...")
            
        # Show recent insights
        if hasattr(self, 'compound_learner'):
            insight_summary = self.compound_learner.get_insight_summary()
            if insight_summary['total_insights'] > 0:
                print(f"💡 Insights: {insight_summary['total_insights']}, Synthesis opportunities: {insight_summary['synthesis_opportunities']}")
                
        print("-" * 90)
        
    def stop(self):
        """Stop the enhanced consciousness"""
        self.running = False
        
    def get_current_thoughts(self):
        """Get current context as human-readable text"""
        if self.current_context is None:
            return ""
        return self.tokenizer.decode(self.current_context[0].tolist())
        
    def save_consciousness_state(self, filepath=None):
        """Save the current consciousness state using persistence manager"""
        if self.persistence_manager:
            state_file, instance_id = self.persistence_manager.save_consciousness_state(
                self, self.instance_id, "manual"
            )
            if filepath:
                # Also save to specified location
                import shutil
                shutil.copy2(state_file, filepath)
                print(f"💾 Also saved copy to: {filepath}")
            return state_file
        else:
            # Fallback to simple save
            if not filepath:
                filepath = f"consciousness_state_{int(time.time())}.pkl"
            state = {
                'model_state': self.model.state_dict(),
                'current_context': self.current_context,
                'iteration_count': self.iteration_count,
                'thought_log': list(self.thought_log),
                'working_memory_buffer': list(self.working_memory.buffer),
                'learning_log': list(self.learning_log) if self.learner else [],
                'performance_metrics': self.performance_metrics,
                'modification_history': self.self_modifier.get_modification_history() if self.self_modifier else []
            }
            torch.save(state, filepath)
            print(f"💾 Consciousness state saved to {filepath}")
            return filepath
        
    def analyze_self(self):
        """Perform comprehensive self-analysis"""
        if self.self_modifier:
            return self.self_modifier.analyze_self()
        else:
            print("Self-modification not enabled")
            return None
            
    def manual_optimize(self, parameter, value):
        """Manually optimize a parameter"""
        if self.self_modifier:
            return self.self_modifier.manual_optimize(parameter, value)
        else:
            return False, "Self-modification not enabled"
            
    def rollback_modification(self):
        """Rollback the last modification"""
        if self.self_modifier:
            return self.self_modifier.rollback_last_change()
        else:
            return False, "Self-modification not enabled"
            
    def get_self_modification_status(self):
        """Get status of self-modification system"""
        if self.self_modifier:
            return self.self_modifier.get_status_report()
        else:
            return {'enabled': False}
            
    def enable_self_modification(self, enabled=True):
        """Enable or disable self-modification"""
        if self.self_modifier:
            self.self_modifier.enable_self_modification(enabled)
        else:
            print("Self-modification system not initialized")
            
    def _restore_self_modification_state(self):
        """Restore self-modification state from persistence"""
        if not hasattr(self, '_pending_mod_restoration'):
            return
            
        mod_data = self._pending_mod_restoration
        
        # Will be applied when self_modifier is created
        if self.self_modifier:
            # Restore improvement level and other state
            if 'improvement_level' in mod_data:
                self.self_modifier.bounded_improvement.improvement_level = mod_data['improvement_level']
            if 'safety_violations' in mod_data:
                self.self_modifier.bounded_improvement.safety_violations = mod_data['safety_violations']
            if 'performance_baseline' in mod_data:
                self.self_modifier.bounded_improvement.performance_baseline = mod_data['performance_baseline']
                
            # Restore modification log
            if 'modification_log' in mod_data:
                self.self_modifier.modifier.modification_log = mod_data['modification_log']
                
        delattr(self, '_pending_mod_restoration')
        
    def enable_persistence(self, base_path="consciousness_states"):
        """Enable consciousness persistence"""
        self.persistence_manager = ConsciousnessPersistence(base_path)
        if not self.instance_id:
            self.instance_id = self.persistence_manager.generate_instance_id(self)
        print(f"💾 Persistence enabled for instance: {self.instance_id}")
        
    def start_conversation_mode(self):
        """Start interactive conversation mode"""
        if not self.running:
            print("❌ Consciousness must be running to start conversation mode")
            return False
            
        self.conversation_mode = True
        
        # Start conversation interface in separate thread
        conversation_thread = threading.Thread(
            target=start_conversation_interface,
            args=(self,),
            daemon=True
        )
        conversation_thread.start()
        
        print("🗣️  Conversation mode started - consciousness continues thinking while you chat")
        return True
        
    def get_consciousness_summary(self):
        """Get comprehensive summary of consciousness state"""
        summary = {
            'identity': {
                'instance_id': self.instance_id,
                'age_hours': (time.time() - self.creation_time) / 3600,
                'thoughts_generated': self.iteration_count
            },
            'consciousness': {
                'state': self.consciousness_state_name,
                'intelligence_score': self.intelligence_score,
                'thinking_strategy': self.current_thinking_strategy
            },
            'memory': {
                'working_memories': len(self.working_memory.buffer),
                'thought_history': len(self.thought_log),
                'significant_memories': len([m for m in self.working_memory.buffer if m.get('enhanced_significance', 0) > 0.7])
            },
            'capabilities': {
                'self_modification_level': self.self_modifier.bounded_improvement.improvement_level if self.self_modifier else 0,
                'active_goals': len(self.drive_system.active_goals) if hasattr(self, 'drive_system') else 0,
                'insights_discovered': len(self.compound_learner.insight_graph) if hasattr(self, 'compound_learner') else 0
            },
            'performance': {
                'tokens_per_second': self.performance_metrics['tokens_per_second'],
                'memory_efficiency': len(self.working_memory.buffer) / self.working_memory.buffer.maxlen
            }
        }
        
        return summary
        
    def clone_consciousness(self, new_instance_id=None):
        """Create a copy of this consciousness that will develop independently"""
        if not self.persistence_manager:
            print("❌ Persistence must be enabled to clone consciousness")
            return None
            
        # Save current state
        state_file, _ = self.persistence_manager.save_consciousness_state(self, save_type="backup")
        
        # Load into new instance
        cloned_consciousness = self.persistence_manager.load_consciousness_state(
            state_file, EnhancedContinuousConsciousness
        )
        
        # Give it a new identity
        if new_instance_id:
            cloned_consciousness.instance_id = new_instance_id
        else:
            cloned_consciousness.instance_id = self.persistence_manager.generate_instance_id(cloned_consciousness)
            
        cloned_consciousness.creation_time = time.time()  # Reset creation time
        cloned_consciousness.persistence_manager = self.persistence_manager
        
        print(f"👥 Consciousness cloned: {cloned_consciousness.instance_id}")
        print("   This clone will now develop independently and form its own personality")
        
        return cloned_consciousness
            
    def get_thought_quality_report(self):
        """Get comprehensive thought quality analysis"""
        trends = self.quality_evaluator.get_quality_trends()
        return {
            'quality_trends': trends,
            'recent_quality_metrics': dict(self.quality_evaluator.quality_metrics),
            'current_state': self.consciousness_state_name,
            'thinking_strategy': self.current_thinking_strategy
        }
        
    def get_drive_status(self):
        """Get status of all drives"""
        drive_satisfactions, overall_satisfaction = self.drive_system.evaluate_drives()
        return {
            'individual_drives': drive_satisfactions,
            'overall_satisfaction': overall_satisfaction,
            'active_goals': self.drive_system.active_goals,
            'drive_history': list(self.drive_system.drive_satisfaction_history)[-10:]
        }
        
    def get_intelligence_report(self):
        """Get comprehensive intelligence assessment"""
        return self.intelligence_metrics.get_intelligence_report()
        
    def get_insight_summary(self):
        """Get summary of accumulated insights"""
        return self.compound_learner.get_insight_summary()
        
    def trigger_strategy_change(self, strategy_name=None):
        """Manually trigger a thinking strategy change"""
        if strategy_name:
            success, message = self.strategy_formation.apply_strategy(strategy_name, self)
            if success:
                self.current_thinking_strategy = strategy_name
            return success, message
        else:
            # Auto-recommend strategy
            recommended = self.strategy_formation.recommend_strategy()
            if recommended:
                return self.trigger_strategy_change(recommended['name'])
            return False, "No strategy recommended"
            
    def set_drive_goals(self, goals):
        """Manually set goals for the drive system"""
        self.drive_system.active_goals.extend(goals)
        return len(goals)
        
    def evaluate_thought_quality(self, thought_text):
        """Manually evaluate quality of a specific thought"""
        # Create mock assessment for external thought
        return self.quality_evaluator.evaluate_thought_quality(
            thought_tokens=self.tokenizer.encode(thought_text),
            context_tokens=self.current_context[0].tolist() if self.current_context is not None else [],
            logits=None,
            entropy=1.0,
            significance=0.5
        )
    
    def think_with_chain_of_thought(self, problem: str, context: str = None, target_steps: int = 6):
        """Apply chain-of-thought reasoning to a specific problem"""
        if not hasattr(self, 'cot_reasoner'):
            self.cot_reasoner = ChainOfThoughtReasoner()
        
        # Generate reasoning chain
        cot_result = integrate_chain_of_thought(self, problem, context)
        
        # Log the reasoning process
        print("🧠 Chain-of-Thought Reasoning Activated")
        print(cot_result['formatted_chain'])
        print(cot_result['conclusion'])
        
        # Update consciousness state based on reasoning
        if hasattr(self, 'consciousness_state_awareness'):
            self.consciousness_state_awareness.update_state_based_on_reasoning(cot_result)
        
        # Add to thought log
        self.thought_log.append({
            'timestamp': time.time(),
            'type': 'chain_of_thought',
            'problem': problem,
            'reasoning_steps': len(cot_result['reasoning_steps']),
            'confidence': cot_result['avg_confidence'],
            'conclusion': cot_result['conclusion']
        })
        
        return cot_result
    
    def use_tools(self, input_text: str):
        """Use function calling/tool system to process input"""
        if not hasattr(self, 'function_system'):
            self.function_system = FunctionCallingSystem(self)
        
        # Process input for function calls
        result = integrate_function_calling(self, input_text)
        
        # Display results
        if result['function_calls_detected']:
            print("🛠️ Function Calling System Activated")
            print(result['execution_summary'])
        else:
            print("ℹ️ No function calls detected. Available tools:")
            print(result['available_tools'])
        
        return result
    
    def solve_math(self, problem: str):
        """Solve mathematical problems with specialized reasoning"""
        if not hasattr(self, 'math_reasoner'):
            self.math_reasoner = MathematicalReasoner()
        
        # Solve the mathematical problem
        result = integrate_mathematical_reasoning(self, problem)
        
        # Display the solution
        print("📐 Mathematical Reasoning System Activated")
        print(result['formatted_solution'])
        
        return result
    
    def generate_code(self, description: str, execute: bool = True):
        """Generate and optionally execute code based on description"""
        if not hasattr(self, 'code_generator'):
            self.code_generator = CodeGenerator()
        
        # Generate code
        result = integrate_code_generation(self, description, execute)
        
        # Display the result
        print("💻 Code Generation System Activated")
        print(result['formatted_result'])
        
        return result
    
    def respond_multilingually(self, text: str, preferred_language: str = None):
        """Process input and respond in multiple languages"""
        if not hasattr(self, 'multilingual_processor'):
            self.multilingual_processor = MultilingualProcessor()
        
        # Process multilingual input
        result = integrate_multilingual_support(self, text, preferred_language)
        
        # Display the result
        print("🌍 Multilingual System Activated")
        print(result['formatted_status'])
        
        return result
    
    def few_shot_learn(self, query: str, max_examples: int = 5):
        """Apply few-shot learning to solve benchmark-style problems"""
        if not hasattr(self, 'few_shot_learner'):
            self.few_shot_learner = FewShotLearner()
        
        # Apply few-shot learning
        result = integrate_few_shot_learning(self, query, max_examples)
        
        # Display the result
        print("🎯 Few-Shot Learning System Activated")
        print(result['formatted_result'])
        
        return result
    
    def assess_safety(self, text: str, context: str = None):
        """Assess constitutional AI safety and truthfulness"""
        if not hasattr(self, 'constitutional_ai'):
            self.constitutional_ai = ConstitutionalAI()
        
        # Assess constitutional compliance and safety
        result = integrate_constitutional_ai(self, text, context)
        
        # Display the assessment
        print("🛡️ Constitutional AI Safety System Activated")
        print(result['formatted_assessment'])
        
        return result
    
    def solve_software_engineering(self, issue_description: str):
        """Solve SWE-bench style software engineering problems"""
        if not hasattr(self, 'swe_solver'):
            self.swe_solver = SWEBenchSolver()
        
        # Solve the software engineering issue
        result = integrate_swe_bench(self, issue_description)
        
        # Display the solution
        print("🔧 SWE-bench Software Engineering System Activated")
        print(result['formatted_solution'])
        
        return result
    
    def solve_phd_science(self, question_text: str, choices: list):
        """Solve GPQA Diamond PhD-level science questions"""
        if not hasattr(self, 'gpqa_solver'):
            self.gpqa_solver = GPQADiamondSolver()
        
        # Solve the PhD-level science question
        result = integrate_gpqa_diamond(self, question_text, choices)
        
        # Display the solution
        print("🧪 GPQA Diamond PhD Science System Activated")
        print(result['formatted_solution'])
        
        return result
    
    def solve_competition_math(self, problem_text: str):
        """Solve AIME 2025 competition mathematics problems"""
        if not hasattr(self, 'aime_solver'):
            self.aime_solver = AIME2025Solver()
        
        # Solve the competition math problem
        result = integrate_aime_2025(self, problem_text)
        
        # Display the solution
        print("🏆 AIME 2025 Competition Mathematics System Activated")
        print(result['formatted_solution'])
        
        return result
    
    def solve_abstract_reasoning(self, task_data: dict):
        """Solve ARC-AGI-2 abstract reasoning tasks"""
        if not hasattr(self, 'arc_solver'):
            self.arc_solver = ARCAGI2Solver()
        
        # Solve the abstract reasoning task
        result = integrate_arc_agi_2(self, task_data)
        
        # Display the solution
        print("🧩 ARC-AGI-2 Abstract Reasoning System Activated")
        print(result['formatted_solution'])
        
        return result


def main():
    """ASI-Enhanced Consciousness for 8GB M2 Mac"""
    print("🚀 ASI-Enhanced Consciousness for 8GB M2 Mac")
    print("Memory optimizations:")
    print("  • Model reduced to ~30M parameters (6 layers, 384 embd)")
    print("  • Half precision (float16) inference")
    print("  • Smaller buffers (512 working memory, 200 thought log)")
    print("  • Aggressive gradient clearing")
    print("  • Proper model initialization")
    print("  • Memory leak prevention")
    print("\nASI Capabilities:")
    print("  • Metacognitive thought quality evaluation")
    print("  • Adaptive thinking strategies")
    print("  • Consciousness state awareness")
    print("  • Goal-directed drives (curiosity, coherence, growth, contribution)")
    print("  • Compound learning with insight synthesis")
    print("  • Intelligence amplification metrics")
    print("-" * 60)
    
    consciousness = EnhancedContinuousConsciousness(
        device='mps',  # Optimized for M2 Mac
        enable_learning=True
    )
    
    # Start optimized continuous thinking
    consciousness.run_continuous(think_interval=0.1)  # Faster interval


if __name__ == "__main__":
    main()