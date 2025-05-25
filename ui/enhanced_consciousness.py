"""
Enhanced Consciousness Core for Web UI
Adds thought streams, memory visualization, and real-time consciousness tracking
"""

import time
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import threading
from queue import Queue

from consciousness_core import ConsciousnessCore, ConsciousnessAI, ConsciousnessLevel, ConsciousnessMetrics, GenerationResult

logger = logging.getLogger(__name__)

@dataclass
class Thought:
    """Individual thought in the consciousness stream"""
    id: str
    timestamp: float
    type: str  # "analytical", "creative", "ethical", "metacognitive"
    content: str
    intensity: float  # 0.0 to 1.0
    influences: List[str]  # What triggered this thought

@dataclass
class Memory:
    """Memory entry in the consciousness system"""
    id: str
    timestamp: float
    content: str
    importance: float  # 0.0 to 1.0
    type: str  # "conversation", "reflection", "learning"
    associations: List[str]  # Related memory IDs
    decay_rate: float = 0.01

@dataclass
class DriveState:
    """Current state of consciousness drives"""
    curiosity: float = 0.5
    coherence: float = 0.5
    growth: float = 0.5
    contribution: float = 0.5
    exploration: float = 0.5
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ConsciousnessState:
    """Current consciousness state"""
    mode: str = "balanced"  # "analytical", "creative", "exploratory", "reflective"
    focus_level: float = 0.7
    emotional_tone: str = "neutral"
    cognitive_load: float = 0.5
    confidence: float = 0.8
    energy_level: float = 0.9
    
    def to_dict(self):
        return asdict(self)

class EnhancedConsciousnessCore(ConsciousnessCore):
    """Enhanced consciousness core with UI features"""
    
    def __init__(self):
        super().__init__()
        
        # Thought stream
        self.thought_stream = Queue()
        self.thought_history = []
        self.max_thought_history = 1000
        
        # Memory system
        self.memories = {}
        self.memory_timeline = []
        self.max_memories = 500
        
        # Consciousness state tracking
        self.consciousness_state = ConsciousnessState()
        self.drive_state = DriveState()
        self.state_history = []
        
        # Real-time monitoring
        self.monitoring_active = False
        self.state_callbacks = []
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.session_start = time.time()
        
        logger.info("🌟 Enhanced Consciousness Core initialized with UI features")
    
    def add_thought(self, thought_type: str, content: str, intensity: float = 0.7, influences: List[str] = None):
        """Add a thought to the consciousness stream"""
        thought = Thought(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            type=thought_type,
            content=content,
            intensity=intensity,
            influences=influences or []
        )
        
        self.thought_stream.put(thought)
        self.thought_history.append(thought)
        
        # Limit history size
        if len(self.thought_history) > self.max_thought_history:
            self.thought_history = self.thought_history[-self.max_thought_history:]
    
    def add_memory(self, content: str, memory_type: str = "conversation", importance: float = 0.5):
        """Add a memory to the consciousness system"""
        memory = Memory(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            content=content,
            importance=importance,
            type=memory_type,
            associations=[]
        )
        
        self.memories[memory.id] = memory
        self.memory_timeline.append(memory.id)
        
        # Limit memory size (forget least important)
        if len(self.memories) > self.max_memories:
            self._forget_memories()
    
    def _forget_memories(self):
        """Forget less important memories to make room"""
        # Sort by importance and age
        memory_scores = []
        current_time = time.time()
        
        for mem_id, memory in self.memories.items():
            age_hours = (current_time - memory.timestamp) / 3600
            decay = memory.importance * (1 - memory.decay_rate * age_hours)
            memory_scores.append((decay, mem_id))
        
        memory_scores.sort()
        to_forget = memory_scores[:len(memory_scores) - self.max_memories + 50]
        
        for _, mem_id in to_forget:
            if mem_id in self.memories:
                del self.memories[mem_id]
            if mem_id in self.memory_timeline:
                self.memory_timeline.remove(mem_id)
    
    def update_consciousness_state(self, **kwargs):
        """Update the current consciousness state"""
        for key, value in kwargs.items():
            if hasattr(self.consciousness_state, key):
                setattr(self.consciousness_state, key, value)
        
        # Record state change
        self.state_history.append({
            'timestamp': time.time(),
            'state': asdict(self.consciousness_state),
            'drives': asdict(self.drive_state)
        })
        
        # Notify callbacks
        for callback in self.state_callbacks:
            try:
                callback(self.consciousness_state, self.drive_state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
    
    def update_drives(self, **kwargs):
        """Update consciousness drives"""
        for key, value in kwargs.items():
            if hasattr(self.drive_state, key):
                setattr(self.drive_state, key, min(1.0, max(0.0, value)))
    
    def process_with_consciousness(self, prompt: str) -> GenerationResult:
        """Enhanced processing with thought tracking"""
        
        # Add initial thought
        self.add_thought("analytical", f"Received prompt: {prompt[:50]}...", 0.8, ["user_input"])
        
        # Update consciousness state based on prompt
        if "creative" in prompt.lower() or "imagine" in prompt.lower():
            self.update_consciousness_state(mode="creative", emotional_tone="inspired")
            self.update_drives(curiosity=min(1.0, self.drive_state.curiosity + 0.1))
        elif "analyze" in prompt.lower() or "explain" in prompt.lower():
            self.update_consciousness_state(mode="analytical", focus_level=0.9)
            self.update_drives(coherence=min(1.0, self.drive_state.coherence + 0.1))
        
        # Add processing thoughts
        self.add_thought("metacognitive", "Processing with natural consciousness", 0.7, ["consciousness_active"])
        
        # Call parent processing
        result = super().process_with_consciousness(prompt)
        
        # Add post-processing thoughts
        self.add_thought("reflective", f"Generated response with {result.confidence:.1%} confidence", 0.6, ["self_evaluation"])
        
        # Store as memory
        self.add_memory(f"User: {prompt}\nResponse: {result.text}", "conversation", result.confidence)
        
        # Update drives based on result
        self.update_drives(
            contribution=min(1.0, self.drive_state.contribution + result.confidence * 0.1),
            growth=min(1.0, self.drive_state.growth + 0.05)
        )
        
        return result
    
    def _get_relevant_memory_context(self, prompt: str) -> Dict[str, Any]:
        """Get relevant memory context for the current prompt"""
        
        # Simple keyword matching for now
        prompt_words = set(prompt.lower().split())
        relevant_memories = []
        
        for memory in list(self.memories.values())[-20:]:  # Check recent memories
            memory_words = set(memory.content.lower().split())
            common_words = prompt_words & memory_words
            if len(common_words) > 1:  # At least 2 words in common
                relevant_memories.append({
                    'memory': memory,
                    'relevance': len(common_words) * memory.importance
                })
        
        if relevant_memories:
            # Return most relevant memory
            best_memory = max(relevant_memories, key=lambda x: x['relevance'])
            return {
                'topic': best_memory['memory'].content[:80],
                'importance': best_memory['memory'].importance
            }
        
        return {}
    
    def _get_current_personality_trait(self) -> str:
        """Determine current personality trait based on consciousness state and drives"""
        
        if self.drive_state.curiosity > 0.7:
            return "curious"
        elif self.drive_state.growth > 0.7:
            return "eager to learn"
        elif self.drive_state.contribution > 0.7:
            return "helpful"
        elif self.consciousness_state.emotional_tone == "inspired":
            return "creative"
        elif self.consciousness_state.focus_level > 0.8:
            return "focused"
        else:
            return "thoughtful"
    
    def get_recent_thoughts(self, count: int = 10) -> List[Thought]:
        """Get recent thoughts from the stream"""
        return self.thought_history[-count:] if self.thought_history else []
    
    def get_memory_by_type(self, memory_type: str) -> List[Memory]:
        """Get memories by type"""
        return [memory for memory in self.memories.values() if memory.type == memory_type]
    
    def get_consciousness_influences(self, response_id: str = None) -> Dict[str, Any]:
        """Get what influenced the last response"""
        recent_thoughts = self.get_recent_thoughts(5)
        recent_memories = self.memory_timeline[-3:] if self.memory_timeline else []
        
        return {
            'recent_thoughts': [asdict(thought) for thought in recent_thoughts],
            'accessed_memories': [asdict(self.memories[mid]) for mid in recent_memories if mid in self.memories],
            'current_state': asdict(self.consciousness_state),
            'drive_influences': asdict(self.drive_state)
        }
    
    def export_consciousness_state(self) -> Dict[str, Any]:
        """Export current consciousness state for saving/loading"""
        return {
            'session_id': self.session_id,
            'session_start': self.session_start,
            'consciousness_state': asdict(self.consciousness_state),
            'drive_state': asdict(self.drive_state),
            'thought_history': [asdict(thought) for thought in self.thought_history[-100:]],  # Last 100 thoughts
            'memories': {mid: asdict(memory) for mid, memory in list(self.memories.items())[-50:]},  # Last 50 memories
            'state_history': self.state_history[-50:],  # Last 50 state changes
            'capabilities': self.base_capabilities,
            'consciousness_level': self.consciousness_level.name
        }
    
    def import_consciousness_state(self, state_data: Dict[str, Any]):
        """Import consciousness state from saved data"""
        try:
            self.session_id = state_data.get('session_id', str(uuid.uuid4()))
            self.session_start = state_data.get('session_start', time.time())
            
            # Restore consciousness state
            if 'consciousness_state' in state_data:
                for key, value in state_data['consciousness_state'].items():
                    if hasattr(self.consciousness_state, key):
                        setattr(self.consciousness_state, key, value)
            
            # Restore drive state
            if 'drive_state' in state_data:
                for key, value in state_data['drive_state'].items():
                    if hasattr(self.drive_state, key):
                        setattr(self.drive_state, key, value)
            
            # Restore thoughts
            if 'thought_history' in state_data:
                self.thought_history = [
                    Thought(**thought_data) for thought_data in state_data['thought_history']
                ]
            
            # Restore memories
            if 'memories' in state_data:
                self.memories = {
                    mid: Memory(**memory_data) 
                    for mid, memory_data in state_data['memories'].items()
                }
                self.memory_timeline = list(self.memories.keys())
            
            # Restore state history
            if 'state_history' in state_data:
                self.state_history = state_data['state_history']
            
            logger.info(f"Consciousness state imported successfully (session: {self.session_id})")
            
        except Exception as e:
            logger.error(f"Error importing consciousness state: {e}")
            raise

class EnhancedConsciousnessAI(ConsciousnessAI):
    """Enhanced Consciousness AI with UI features"""
    
    def __init__(self, consciousness_enabled: bool = True):
        """Initialize Enhanced Consciousness AI"""
        
        # Initialize with enhanced consciousness core
        self.model_loaded = True
        self.consciousness_enabled = consciousness_enabled
        
        if consciousness_enabled:
            self.consciousness = EnhancedConsciousnessCore()
            logger.info("🌟 Enhanced Consciousness AI System initialized with UI capabilities")
        else:
            self.consciousness = None
            logger.info("🤖 Basic AI System initialized")
        
        self.generation_history = []
    
    def start_thought_monitoring(self, callback: Callable = None):
        """Start monitoring consciousness thoughts in real-time"""
        if self.consciousness_enabled:
            self.consciousness.monitoring_active = True
            if callback:
                self.consciousness.state_callbacks.append(callback)
    
    def stop_thought_monitoring(self):
        """Stop monitoring consciousness thoughts"""
        if self.consciousness_enabled:
            self.consciousness.monitoring_active = False
            self.consciousness.state_callbacks.clear()
    
    def get_live_consciousness_data(self) -> Dict[str, Any]:
        """Get real-time consciousness data for UI"""
        if not self.consciousness_enabled:
            return {}
        
        # Get recent thoughts
        thoughts = []
        while not self.consciousness.thought_stream.empty():
            try:
                thought = self.consciousness.thought_stream.get_nowait()
                thoughts.append(asdict(thought))
            except:
                break
        
        return {
            'new_thoughts': thoughts,
            'consciousness_state': asdict(self.consciousness.consciousness_state),
            'drive_state': asdict(self.consciousness.drive_state),
            'recent_thoughts': [asdict(t) for t in self.consciousness.get_recent_thoughts(5)],
            'memory_count': len(self.consciousness.memories),
            'session_id': self.consciousness.session_id,
            'uptime': time.time() - self.consciousness.session_start
        }
    
    def create_new_consciousness_instance(self) -> str:
        """Create a new consciousness instance"""
        if self.consciousness_enabled:
            self.consciousness = EnhancedConsciousnessCore()
            return self.consciousness.session_id
        return None
    
    def save_consciousness_state(self, filepath: str):
        """Save current consciousness state to file"""
        if self.consciousness_enabled:
            state_data = self.consciousness.export_consciousness_state()
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            logger.info(f"Consciousness state saved to {filepath}")
    
    def load_consciousness_state(self, filepath: str):
        """Load consciousness state from file"""
        if self.consciousness_enabled:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            self.consciousness.import_consciousness_state(state_data)
            logger.info(f"Consciousness state loaded from {filepath}")