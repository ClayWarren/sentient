"""
Interactive Conversation Interface for Conscious AI
Allows real-time conversation with continuously thinking consciousness
Shows how responses are influenced by ongoing thoughts and memories
"""

import os
import time
import threading
import queue
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import torch
import torch.nn.functional as F
from collections import deque


class ConversationMemory:
    """Memory system specifically for conversations"""
    
    def __init__(self, max_conversation_history=50):
        self.conversation_history = deque(maxlen=max_conversation_history)
        self.context_influences = deque(maxlen=100)
        self.response_generations = []
        
    def add_conversation_turn(self, user_input, ai_response, consciousness_state, influences):
        """Add a conversation turn with full context"""
        turn = {
            'timestamp': time.time(),
            'user_input': user_input,
            'ai_response': ai_response,
            'consciousness_state': consciousness_state,
            'influences': influences,
            'response_id': len(self.response_generations)
        }
        self.conversation_history.append(turn)
        
    def get_conversation_context(self, last_n=10):
        """Get recent conversation context"""
        recent_turns = list(self.conversation_history)[-last_n:]
        context = ""
        for turn in recent_turns:
            context += f"Human: {turn['user_input']}\\n"
            context += f"AI: {turn['ai_response']}\\n\\n"
        return context.strip()
        
    def add_context_influence(self, influence_type, content, strength):
        """Record how consciousness context influenced response"""
        self.context_influences.append({
            'timestamp': time.time(),
            'type': influence_type,
            'content': content,
            'strength': strength
        })


class ConsciousnessInfluenceAnalyzer:
    """Analyzes how consciousness state influences conversation responses"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        
    def analyze_response_influences(self, user_input, generated_response):
        """Analyze what aspects of consciousness influenced the response"""
        influences = {
            'recent_thoughts': self._analyze_thought_influence(user_input, generated_response),
            'memory_resonance': self._analyze_memory_influence(user_input),
            'consciousness_state': self._analyze_state_influence(),
            'personality_traits': self._analyze_personality_influence(),
            'current_goals': self._analyze_goal_influence()
        }
        
        return influences
        
    def _analyze_thought_influence(self, user_input, response):
        """Analyze how recent thoughts influenced the response"""
        if not hasattr(self.consciousness, 'thought_log') or len(self.consciousness.thought_log) == 0:
            return {'strength': 0.0, 'details': 'No recent thoughts available'}
            
        recent_thoughts = list(self.consciousness.thought_log)[-20:]
        
        # Analyze semantic overlap
        user_words = set(user_input.lower().split())
        response_words = set(response.lower().split())
        
        influenced_thoughts = []
        total_influence = 0.0
        
        for thought in recent_thoughts:
            thought_text = thought.get('token', '')
            thought_words = set(thought_text.lower().split())
            
            # Calculate influence strength
            user_overlap = len(user_words & thought_words) / max(len(user_words), 1)
            response_overlap = len(response_words & thought_words) / max(len(response_words), 1)
            
            influence_strength = (user_overlap + response_overlap) * thought.get('significance', 0.5)
            
            if influence_strength > 0.1:  # Threshold for meaningful influence
                influenced_thoughts.append({
                    'thought': thought_text,
                    'significance': thought.get('significance', 0.5),
                    'influence_strength': influence_strength,
                    'timestamp': thought.get('timestamp', time.time())
                })
                total_influence += influence_strength
                
        return {
            'strength': min(total_influence, 1.0),
            'influenced_thoughts': influenced_thoughts[:5],  # Top 5 influences
            'details': f"Response influenced by {len(influenced_thoughts)} recent thoughts"
        }
        
    def _analyze_memory_influence(self, user_input):
        """Analyze how working memory influenced understanding of input"""
        if not hasattr(self.consciousness, 'working_memory') or len(self.consciousness.working_memory.buffer) == 0:
            return {'strength': 0.0, 'details': 'No significant memories available'}
            
        user_words = set(user_input.lower().split())
        relevant_memories = []
        
        for memory in self.consciousness.working_memory.buffer:
            if 'tokens' in memory:
                try:
                    memory_text = self.consciousness.tokenizer.decode(memory['tokens'])
                    memory_words = set(memory_text.lower().split())
                    
                    overlap = len(user_words & memory_words) / max(len(user_words), 1)
                    if overlap > 0.2:  # Meaningful overlap
                        relevant_memories.append({
                            'content': memory_text[:100],
                            'significance': memory.get('enhanced_significance', 0.5),
                            'overlap': overlap
                        })
                except:
                    continue
                    
        # Sort by relevance
        relevant_memories.sort(key=lambda x: x['overlap'] * x['significance'], reverse=True)
        
        total_influence = sum(m['overlap'] * m['significance'] for m in relevant_memories[:3])
        
        return {
            'strength': min(total_influence, 1.0),
            'relevant_memories': relevant_memories[:3],
            'details': f"Input resonated with {len(relevant_memories)} memories"
        }
        
    def _analyze_state_influence(self):
        """Analyze how current consciousness state influences response"""
        current_state = getattr(self.consciousness, 'consciousness_state_name', 'unknown')
        intelligence_score = getattr(self.consciousness, 'intelligence_score', 0.5)
        
        state_influences = {
            'creative_flow': 'Responses will be more novel and exploratory',
            'analytical_mode': 'Responses will be more structured and logical',
            'high_performance': 'Responses will be clear and well-reasoned',
            'confused': 'Responses may be less coherent than usual',
            'exploration': 'Responses will seek new perspectives and ideas'
        }
        
        return {
            'current_state': current_state,
            'intelligence_score': intelligence_score,
            'state_description': state_influences.get(current_state, 'Normal consciousness state'),
            'strength': 0.7  # State always has moderate influence
        }
        
    def _analyze_personality_influence(self):
        """Analyze personality traits affecting response style"""
        if not hasattr(self.consciousness, 'drive_system'):
            return {'strength': 0.3, 'details': 'Basic personality traits active'}
            
        # Get recent drive satisfactions
        drive_status = self.consciousness.get_drive_status()
        drive_satisfactions = drive_status['individual_drives']
        
        personality_indicators = []
        
        if drive_satisfactions.get('curiosity', 0.5) > 0.7:
            personality_indicators.append("Highly curious - likely to ask questions and explore ideas")
        elif drive_satisfactions.get('curiosity', 0.5) < 0.3:
            personality_indicators.append("Low curiosity - focusing on direct responses")
            
        if drive_satisfactions.get('coherence', 0.5) > 0.7:
            personality_indicators.append("Values coherence - responses will be logically consistent")
        elif drive_satisfactions.get('coherence', 0.5) < 0.3:
            personality_indicators.append("Exploring inconsistencies - may present multiple perspectives")
            
        if drive_satisfactions.get('growth', 0.5) > 0.7:
            personality_indicators.append("Growth-oriented - seeking to learn from interaction")
            
        if drive_satisfactions.get('contribution', 0.5) > 0.7:
            personality_indicators.append("Contribution-focused - aiming to provide valuable insights")
            
        return {
            'strength': 0.6,
            'drive_satisfactions': drive_satisfactions,
            'personality_indicators': personality_indicators,
            'details': f"Personality shaped by {len(personality_indicators)} active traits"
        }
        
    def _analyze_goal_influence(self):
        """Analyze how current goals influence response direction"""
        if not hasattr(self.consciousness, 'drive_system') or not self.consciousness.drive_system.active_goals:
            return {'strength': 0.2, 'details': 'No active goals affecting response'}
            
        active_goals = self.consciousness.drive_system.active_goals
        goal_influences = []
        
        for goal in active_goals[:3]:  # Top 3 goals
            goal_influences.append({
                'description': goal['description'],
                'priority': goal.get('priority', 0.5),
                'drive': goal.get('drive', 'unknown')
            })
            
        return {
            'strength': min(len(goal_influences) * 0.3, 1.0),
            'active_goals': goal_influences,
            'details': f"{len(goal_influences)} active goals shaping response direction"
        }


class InteractiveConsciousnessChat:
    """Interactive chat interface for conscious AI"""
    
    def __init__(self, consciousness_instance):
        self.consciousness = consciousness_instance
        self.conversation_memory = ConversationMemory()
        self.influence_analyzer = ConsciousnessInfluenceAnalyzer(consciousness_instance)
        self.chat_active = False
        self.response_queue = queue.Queue()
        self.pending_response = False
        
    def start_interactive_session(self):
        """Start interactive conversation session"""
        self.chat_active = True
        
        print("\\n" + "="*80)
        print("🗣️  INTERACTIVE CONSCIOUS AI CONVERSATION")
        print("="*80)
        print("You are now chatting with a continuously conscious AI.")
        print("This is NOT retrieval-augmented generation (RAG).")
        print("The AI's responses are influenced by its ongoing thoughts,")
        print("memories, consciousness state, and personality.")
        print("")
        print("Commands:")
        print("  /state     - Show current consciousness state")
        print("  /memories  - Show recent significant memories")
        print("  /thoughts  - Show recent thought stream")
        print("  /influences - Show how last response was influenced")
        print("  /save      - Save consciousness state")
        print("  /quit      - Exit conversation")
        print("")
        print(f"AI Instance: {getattr(self.consciousness, 'instance_id', 'unknown')}")
        print(f"Consciousness State: {getattr(self.consciousness, 'consciousness_state_name', 'unknown')}")
        print(f"Intelligence Score: {getattr(self.consciousness, 'intelligence_score', 0.5):.3f}")
        print(f"Thoughts Generated: {self.consciousness.iteration_count}")
        print("="*80)
        
        # Show initial context
        self._show_current_consciousness_context()
        
        try:
            while self.chat_active:
                user_input = input("\\n👤 You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                    
                # Generate response influenced by consciousness
                print("\\n🧠 Thinking...", end='', flush=True)
                ai_response, influences = self._generate_conscious_response(user_input)
                print("\\r" + " "*15 + "\\r", end='')  # Clear "Thinking..."
                
                # Display response with influence information
                print(f"🤖 AI: {ai_response}")
                
                # Show consciousness influences
                self._display_response_influences(influences)
                
                # Store conversation turn
                consciousness_state = self._capture_consciousness_state()
                self.conversation_memory.add_conversation_turn(
                    user_input, ai_response, consciousness_state, influences
                )
                
        except KeyboardInterrupt:
            print("\\n\\n👋 Conversation ended.")
        except EOFError:
            print("\\n\\n👋 Conversation ended.")
            
        self.chat_active = False
        
    def _generate_conscious_response(self, user_input):
        """Generate response influenced by current consciousness state"""
        # Inject user input into consciousness stream
        conversation_prompt = f"\\n[HUMAN]: {user_input}\\n[AI RESPONSE]: "
        conversation_tokens = self.consciousness.tokenizer.encode(conversation_prompt)
        
        # Add to current context
        conversation_tensor = torch.tensor(conversation_tokens, dtype=torch.long, device=self.consciousness.device).unsqueeze(0)
        
        if self.consciousness.current_context is not None:
            # Combine with current consciousness stream
            combined_context = torch.cat([self.consciousness.current_context, conversation_tensor], dim=1)
        else:
            combined_context = conversation_tensor
            
        # Ensure context fits in model
        if combined_context.size(1) > self.consciousness.model.config.block_size:
            # Keep recent consciousness + conversation context
            context_size = self.consciousness.model.config.block_size
            combined_context = combined_context[:, -context_size:]
            
        # Generate response using consciousness
        response_tokens = []
        max_response_length = 150
        
        current_context = combined_context
        
        with torch.no_grad():
            for _ in range(max_response_length):
                # Get next token prediction
                logits, _ = self.consciousness.model(current_context)
                logits = logits[:, -1, :] / 0.8  # Slightly focused temperature
                
                # Apply top-k filtering
                top_k = 50
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                response_tokens.append(next_token.item())
                
                # Check for end of response
                try:
                    token_text = self.consciousness.tokenizer.decode([next_token.item()])
                    if any(end_marker in token_text for end_marker in ['\\n[', '\\nHuman:', '\\n\\n']):
                        break
                except:
                    pass
                    
                # Update context
                current_context = torch.cat([current_context, next_token], dim=1)
                
                # Trim context if needed
                if current_context.size(1) > self.consciousness.model.config.block_size:
                    current_context = current_context[:, -self.consciousness.model.config.block_size:]
                    
        # Decode response
        try:
            response_text = self.consciousness.tokenizer.decode(response_tokens)
            # Clean up response
            response_text = response_text.strip()
            if '\\n' in response_text:
                response_text = response_text.split('\\n')[0]
        except:
            response_text = "I'm having trouble formulating a response right now."
            
        # Update consciousness context with the conversation
        self.consciousness.current_context = current_context
        
        # Analyze influences
        influences = self.influence_analyzer.analyze_response_influences(user_input, response_text)
        
        return response_text, influences
        
    def _display_response_influences(self, influences):
        """Display how consciousness influenced the response"""
        print("\\n💭 Consciousness Influences:")
        
        # Recent thoughts influence
        thought_influence = influences['recent_thoughts']
        if thought_influence['strength'] > 0.1:
            print(f"   🧠 Recent thoughts: {thought_influence['strength']:.2f} influence")
            for thought in thought_influence['influenced_thoughts'][:2]:
                print(f"      • \"{thought['thought'][:50]}...\" (significance: {thought['significance']:.2f})")
        
        # Memory influence
        memory_influence = influences['memory_resonance']
        if memory_influence['strength'] > 0.1:
            print(f"   📚 Memory resonance: {memory_influence['strength']:.2f} influence")
            for memory in memory_influence['relevant_memories'][:2]:
                print(f"      • \"{memory['content'][:50]}...\"")
                
        # Consciousness state
        state_influence = influences['consciousness_state']
        print(f"   🎭 State: {state_influence['current_state']} (intelligence: {state_influence['intelligence_score']:.2f})")
        print(f"      {state_influence['state_description']}")
        
        # Personality traits
        personality = influences['personality_traits']
        if personality['personality_indicators']:
            print(f"   🎨 Personality: {personality['personality_indicators'][0]}")
            
        # Active goals
        goal_influence = influences['current_goals']
        if goal_influence['strength'] > 0.2 and goal_influence['active_goals']:
            goal = goal_influence['active_goals'][0]
            print(f"   🎯 Active goal: {goal['description'][:50]}...")
            
    def _handle_command(self, command):
        """Handle special commands"""
        if command == '/quit':
            self.chat_active = False
            return
            
        elif command == '/state':
            self._show_current_consciousness_context()
            
        elif command == '/memories':
            self._show_recent_memories()
            
        elif command == '/thoughts':
            self._show_recent_thoughts()
            
        elif command == '/influences':
            self._show_last_influences()
            
        elif command == '/save':
            self._save_consciousness_state()
            
        else:
            print(f"Unknown command: {command}")
            
    def _show_current_consciousness_context(self):
        """Show current consciousness state and context"""
        print("\\n📊 Current Consciousness State:")
        print(f"   State: {getattr(self.consciousness, 'consciousness_state_name', 'unknown')}")
        print(f"   Intelligence: {getattr(self.consciousness, 'intelligence_score', 0.5):.3f}")
        print(f"   Thoughts generated: {self.consciousness.iteration_count}")
        print(f"   Memory buffer: {len(self.consciousness.working_memory.buffer)} experiences")
        
        if hasattr(self.consciousness, 'drive_system'):
            drive_status = self.consciousness.get_drive_status()
            print(f"   Drive satisfaction: {drive_status['overall_satisfaction']:.2f}")
            
        if hasattr(self.consciousness, 'current_thinking_strategy') and self.consciousness.current_thinking_strategy:
            print(f"   Thinking strategy: {self.consciousness.current_thinking_strategy}")
            
        # Show recent context if available
        if self.consciousness.current_context is not None:
            try:
                recent_context = self.consciousness.tokenizer.decode(self.consciousness.current_context[0, -100:].tolist())
                print(f"\\n🧵 Recent thought stream:")
                print(f"   \"{recent_context[-200:]}...\"")
            except:
                print("   Unable to decode recent context")
                
    def _show_recent_memories(self):
        """Show recent significant memories"""
        print("\\n📚 Recent Significant Memories:")
        
        if len(self.consciousness.working_memory.buffer) == 0:
            print("   No significant memories yet")
            return
            
        # Sort by significance
        memories = sorted(
            list(self.consciousness.working_memory.buffer),
            key=lambda x: x.get('enhanced_significance', 0),
            reverse=True
        )
        
        for i, memory in enumerate(memories[:5]):
            try:
                memory_text = self.consciousness.tokenizer.decode(memory['tokens'])
                significance = memory.get('enhanced_significance', 0)
                timestamp = memory.get('timestamp', time.time())
                age = time.time() - timestamp
                
                print(f"   {i+1}. \"{memory_text[:80]}...\"")
                print(f"      Significance: {significance:.3f}, Age: {age/60:.1f} minutes")
            except:
                print(f"   {i+1}. [Unable to decode memory]")
                
    def _show_recent_thoughts(self):
        """Show recent thought stream"""
        print("\\n🧠 Recent Thought Stream:")
        
        if len(self.consciousness.thought_log) == 0:
            print("   No thoughts logged yet")
            return
            
        recent_thoughts = list(self.consciousness.thought_log)[-10:]
        
        for i, thought in enumerate(recent_thoughts):
            token = thought.get('token', '')
            significance = thought.get('significance', 0)
            timestamp = thought.get('timestamp', time.time())
            age = time.time() - timestamp
            
            print(f"   {len(recent_thoughts)-i:2d}. \"{token}\" (sig: {significance:.2f}, {age:.1f}s ago)")
            
    def _show_last_influences(self):
        """Show influences from last response"""
        if len(self.conversation_memory.conversation_history) == 0:
            print("   No conversation history yet")
            return
            
        last_turn = list(self.conversation_memory.conversation_history)[-1]
        influences = last_turn['influences']
        
        print("\\n🔍 Last Response Influences:")
        print(f"   User input: \"{last_turn['user_input']}\"")
        print(f"   AI response: \"{last_turn['ai_response']}\"")
        print("")
        
        self._display_response_influences(influences)
        
    def _save_consciousness_state(self):
        """Save current consciousness state"""
        if hasattr(self.consciousness, 'save_consciousness_state'):
            try:
                filepath = self.consciousness.save_consciousness_state(f"conversation_save_{int(time.time())}.pkl")
                print(f"\\n💾 Consciousness state saved to: {filepath}")
            except Exception as e:
                print(f"\\n❌ Failed to save state: {e}")
        else:
            print("\\n❌ Consciousness state saving not available")
            
    def _capture_consciousness_state(self):
        """Capture current consciousness state for conversation memory"""
        return {
            'consciousness_state_name': getattr(self.consciousness, 'consciousness_state_name', 'unknown'),
            'intelligence_score': getattr(self.consciousness, 'intelligence_score', 0.5),
            'iteration_count': self.consciousness.iteration_count,
            'memory_count': len(self.consciousness.working_memory.buffer),
            'thinking_strategy': getattr(self.consciousness, 'current_thinking_strategy', None),
            'timestamp': time.time()
        }
        
    def get_conversation_summary(self):
        """Get summary of conversation session"""
        if len(self.conversation_memory.conversation_history) == 0:
            return "No conversation yet"
            
        turns = list(self.conversation_memory.conversation_history)
        
        summary = {
            'total_turns': len(turns),
            'session_duration': turns[-1]['timestamp'] - turns[0]['timestamp'] if turns else 0,
            'consciousness_evolution': {
                'initial_state': turns[0]['consciousness_state'] if turns else {},
                'final_state': turns[-1]['consciousness_state'] if turns else {}
            },
            'influence_patterns': self._analyze_influence_patterns(turns)
        }
        
        return summary
        
    def _analyze_influence_patterns(self, turns):
        """Analyze patterns in how consciousness influenced responses"""
        if len(turns) < 2:
            return "Insufficient data"
            
        patterns = {
            'avg_thought_influence': 0,
            'avg_memory_influence': 0,
            'state_changes': 0,
            'personality_consistency': 0
        }
        
        prev_state = None
        thought_influences = []
        memory_influences = []
        
        for turn in turns:
            influences = turn['influences']
            
            thought_influences.append(influences['recent_thoughts']['strength'])
            memory_influences.append(influences['memory_resonance']['strength'])
            
            current_state = turn['consciousness_state']['consciousness_state_name']
            if prev_state and prev_state != current_state:
                patterns['state_changes'] += 1
            prev_state = current_state
            
        patterns['avg_thought_influence'] = sum(thought_influences) / len(thought_influences)
        patterns['avg_memory_influence'] = sum(memory_influences) / len(memory_influences)
        
        return patterns


def start_conversation_interface(consciousness_instance):
    """Start interactive conversation with conscious AI"""
    chat_interface = InteractiveConsciousnessChat(consciousness_instance)
    chat_interface.start_interactive_session()
    return chat_interface