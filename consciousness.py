"""
Continuous Consciousness System
Extends nanoGPT for persistent, always-running AI awareness
"""

import os
import time
import threading
import queue
from collections import deque
from datetime import datetime
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
import tiktoken


class WorkingMemory:
    """Circular buffer for working memory with significance detection"""
    
    def __init__(self, max_size=1024, significance_threshold=0.7):
        self.buffer = deque(maxlen=max_size)
        self.significance_threshold = significance_threshold
        self.consolidated_memories = []
        
    def add_experience(self, tokens, logits, significance):
        """Add new experience to working memory"""
        experience = {
            'tokens': tokens,
            'logits': logits,
            'significance': significance,
            'timestamp': time.time()
        }
        
        # Check if this is significant enough to remember
        if significance > self.significance_threshold:
            self.buffer.append(experience)
            
    def get_recent_context(self, max_tokens=512):
        """Get recent experiences as context tokens"""
        context_tokens = []
        for exp in reversed(self.buffer):
            if len(context_tokens) + len(exp['tokens']) > max_tokens:
                break
            context_tokens = exp['tokens'] + context_tokens
        return torch.tensor(context_tokens).unsqueeze(0)
        
    def detect_significance(self, logits, prev_logits):
        """Detect if current output is significant vs routine"""
        if prev_logits is None:
            return 0.5
            
        # Simple significance based on prediction confidence change
        current_probs = F.softmax(logits, dim=-1)
        current_entropy = -(current_probs * torch.log(current_probs + 1e-10)).sum(dim=-1)
        
        prev_probs = F.softmax(prev_logits, dim=-1)
        prev_entropy = -(prev_probs * torch.log(prev_probs + 1e-10)).sum(dim=-1)
        
        # High significance = unexpected changes in prediction confidence
        significance = torch.abs(current_entropy - prev_entropy).mean().item()
        return min(significance / 2.0, 1.0)  # Normalize to [0,1]


class AmbientInputs:
    """Generates ambient inputs for consciousness"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.last_time_injection = 0
        
    def get_time_tokens(self):
        """Inject current time as tokens"""
        current_time = datetime.now().strftime("%H:%M")
        time_text = f"[TIME: {current_time}]"
        return self.tokenizer.encode(time_text)
        
    def get_system_tokens(self):
        """Inject system state as tokens"""
        # Simple system awareness
        system_text = "[SYSTEM: Running continuously]"
        return self.tokenizer.encode(system_text)
        
    def should_inject_ambient(self):
        """Decide if we should inject ambient info"""
        # Inject every 60 seconds
        if time.time() - self.last_time_injection > 60:
            self.last_time_injection = time.time()
            return True
        return False


class ContinuousConsciousness:
    """Main continuous consciousness system"""
    
    def __init__(self, model_path=None, device='mps'):
        self.device = device
        self.running = False
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Create a small model for testing
            config = GPTConfig(
                vocab_size=50304,
                block_size=1024,
                n_layer=6,  # Small for 8GB VRAM
                n_head=6,
                n_embd=384,
                dropout=0.0
            )
            self.model = GPT(config).to(device)
            
        self.model.eval()
        
        # Initialize components
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.working_memory = WorkingMemory()
        self.ambient_inputs = AmbientInputs(self.tokenizer)
        
        # State
        self.current_context = None
        self.prev_logits = None
        self.iteration_count = 0
        
        # Logging
        self.thought_log = []
        
    def load_model(self, model_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = GPTConfig(**checkpoint['model_args'])
        self.model = GPT(config).to(self.device)
        
        # Handle compiled model state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        
    def initialize_consciousness(self):
        """Initialize the consciousness with initial thoughts"""
        initial_text = "I am beginning to think continuously. What should I contemplate?"
        self.current_context = torch.tensor(
            self.tokenizer.encode(initial_text), 
            dtype=torch.long, 
            device=self.device
        ).unsqueeze(0)
        
    def think_one_step(self):
        """Single step of continuous thinking"""
        if self.current_context is None:
            self.initialize_consciousness()
            
        # Inject ambient inputs occasionally
        if self.ambient_inputs.should_inject_ambient():
            time_tokens = self.ambient_inputs.get_time_tokens()
            time_tensor = torch.tensor(time_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            self.current_context = torch.cat([self.current_context, time_tensor], dim=1)
            
        # Manage context window - keep recent thoughts
        if self.current_context.size(1) > self.model.config.block_size:
            # Keep the most recent tokens, but preserve some working memory
            recent_context = self.working_memory.get_recent_context(256)
            if recent_context.size(1) > 0:
                # Combine recent working memory with current context tail
                context_tail = self.current_context[:, -(self.model.config.block_size - recent_context.size(1)):]
                self.current_context = torch.cat([recent_context.to(self.device), context_tail], dim=1)
            else:
                self.current_context = self.current_context[:, -self.model.config.block_size:]
        
        # Forward pass
        with torch.no_grad():
            logits, _ = self.model(self.current_context)
            logits = logits[:, -1, :] / 0.8  # Temperature
            
            # Top-k filtering
            top_k = 40
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Detect significance
            significance = self.working_memory.detect_significance(logits, self.prev_logits)
            
            # Add to working memory if significant
            current_tokens = self.current_context[0].tolist()
            self.working_memory.add_experience(
                tokens=current_tokens[-50:],  # Last 50 tokens
                logits=logits,
                significance=significance
            )
            
            # Update context
            self.current_context = torch.cat([self.current_context, next_token], dim=1)
            self.prev_logits = logits
            
            # Log the thought
            if len(self.thought_log) < 1000:  # Prevent memory growth
                decoded = self.tokenizer.decode(next_token[0].tolist())
                self.thought_log.append({
                    'token': decoded,
                    'significance': significance,
                    'timestamp': time.time(),
                    'iteration': self.iteration_count
                })
                
        self.iteration_count += 1
        
    def run_continuous(self, think_interval=0.1):
        """Run continuous consciousness loop"""
        print("🧠 Starting continuous consciousness...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params()/1e6:.1f}M")
        print(f"Think interval: {think_interval}s")
        print("-" * 50)
        
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                
                self.think_one_step()
                
                # Print recent thoughts occasionally
                if self.iteration_count % 50 == 0:
                    self.print_recent_thoughts()
                    
                # Control thinking speed
                elapsed = time.time() - start_time
                if elapsed < think_interval:
                    time.sleep(think_interval - elapsed)
                    
        except KeyboardInterrupt:
            print("\n🛑 Consciousness stopped by user")
        except Exception as e:
            print(f"\n❌ Error in consciousness loop: {e}")
        finally:
            self.running = False
            
    def print_recent_thoughts(self):
        """Print recent thoughts and statistics"""
        if len(self.thought_log) < 10:
            return
            
        recent_thoughts = self.thought_log[-20:]
        thought_text = "".join([t['token'] for t in recent_thoughts])
        avg_significance = sum(t['significance'] for t in recent_thoughts) / len(recent_thoughts)
        
        print(f"\n[Iteration {self.iteration_count}] Recent thoughts:")
        print(f"💭 {thought_text}")
        print(f"📊 Avg significance: {avg_significance:.3f} | Memory size: {len(self.working_memory.buffer)}")
        print("-" * 80)
        
    def stop(self):
        """Stop the consciousness loop"""
        self.running = False
        
    def get_current_thoughts(self):
        """Get current context as human-readable text"""
        if self.current_context is None:
            return ""
        return self.tokenizer.decode(self.current_context[0].tolist())


def main():
    """Example usage"""
    consciousness = ContinuousConsciousness(device='mps')  # or 'cuda' or 'cpu'
    
    # Start continuous thinking
    consciousness.run_continuous(think_interval=0.1)


if __name__ == "__main__":
    main()