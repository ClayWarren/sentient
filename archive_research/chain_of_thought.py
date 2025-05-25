"""
Chain-of-Thought Reasoning for Sentient AI
Implements structured reasoning with step-by-step thinking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass
from enum import Enum

class ReasoningType(Enum):
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    LOGICAL = "logical"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    PROBLEM_SOLVING = "problem_solving"

@dataclass
class ReasoningStep:
    step_number: int
    thought: str
    reasoning_type: ReasoningType
    confidence: float
    evidence: List[str]
    connections: List[int]  # Links to other steps
    
class ChainOfThoughtModule(nn.Module):
    """Neural module for chain-of-thought reasoning"""
    
    def __init__(self, d_model: int = 768, num_steps: int = 8, num_heads: int = 12):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps
        self.num_heads = num_heads
        
        # Step generation layers
        self.step_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model*4),
            num_layers=3
        )
        
        # Reasoning type classifier
        self.reasoning_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(ReasoningType))
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Step connection predictor
        self.connection_predictor = nn.MultiheadAttention(d_model, num_heads)
        
        # Final synthesis layer
        self.synthesis_layer = nn.Sequential(
            nn.Linear(d_model * num_steps, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, input_embedding: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = input_embedding.size(0)
        
        # Generate reasoning steps
        step_embeddings = []
        current_state = input_embedding
        
        for step in range(self.num_steps):
            # Encode current reasoning step
            if context is not None:
                step_input = torch.cat([current_state, context], dim=-1)
                step_input = nn.Linear(step_input.size(-1), self.d_model).to(input_embedding.device)(step_input)
            else:
                step_input = current_state
                
            step_embedding = self.step_encoder(step_input.unsqueeze(1)).squeeze(1)
            step_embeddings.append(step_embedding)
            
            # Update state for next step
            current_state = step_embedding
            
        # Stack all step embeddings
        all_steps = torch.stack(step_embeddings, dim=1)  # [batch, num_steps, d_model]
        
        # Predict reasoning types for each step
        reasoning_types = self.reasoning_classifier(all_steps)  # [batch, num_steps, num_types]
        
        # Predict confidence for each step
        confidences = self.confidence_predictor(all_steps).squeeze(-1)  # [batch, num_steps]
        
        # Predict step connections
        connections, _ = self.connection_predictor(all_steps, all_steps, all_steps)
        
        # Synthesize final reasoning
        flattened_steps = all_steps.view(batch_size, -1)
        synthesis = self.synthesis_layer(flattened_steps)
        
        return {
            'step_embeddings': all_steps,
            'reasoning_types': reasoning_types,
            'confidences': confidences,
            'connections': connections,
            'synthesis': synthesis
        }

class ChainOfThoughtReasoner:
    """High-level chain-of-thought reasoning system"""
    
    def __init__(self, model_dim: int = 768, max_steps: int = 12):
        self.model_dim = model_dim
        self.max_steps = max_steps
        self.cot_module = ChainOfThoughtModule(model_dim, max_steps)
        
        # Reasoning templates
        self.templates = {
            ReasoningType.ANALYTICAL: "Let me analyze this step by step: {}",
            ReasoningType.CREATIVE: "Thinking creatively about this: {}",
            ReasoningType.LOGICAL: "Following logical reasoning: {}",
            ReasoningType.CAUSAL: "Considering cause and effect: {}",
            ReasoningType.COMPARATIVE: "Comparing different aspects: {}",
            ReasoningType.PROBLEM_SOLVING: "To solve this problem: {}"
        }
        
        # Step transition prompts
        self.transitions = [
            "First, let me consider...",
            "Next, I should examine...",
            "Building on that thought...",
            "This leads me to think...",
            "Therefore, it follows that...",
            "Consequently...",
            "Moreover...",
            "Finally..."
        ]
        
    def generate_reasoning_chain(self, 
                                problem: str, 
                                context: Optional[str] = None,
                                target_steps: int = 6) -> List[ReasoningStep]:
        """Generate a structured chain of reasoning steps"""
        
        steps = []
        current_thought = problem
        
        for step_num in range(min(target_steps, self.max_steps)):
            # Determine reasoning type based on step and context
            reasoning_type = self._select_reasoning_type(step_num, current_thought, steps)
            
            # Generate reasoning step
            step_thought = self._generate_step_thought(
                step_num, current_thought, reasoning_type, steps, context
            )
            
            # Calculate confidence based on coherence with previous steps
            confidence = self._calculate_step_confidence(step_thought, steps)
            
            # Extract evidence from the step
            evidence = self._extract_evidence(step_thought)
            
            # Find connections to previous steps
            connections = self._find_step_connections(step_thought, steps)
            
            step = ReasoningStep(
                step_number=step_num + 1,
                thought=step_thought,
                reasoning_type=reasoning_type,
                confidence=confidence,
                evidence=evidence,
                connections=connections
            )
            
            steps.append(step)
            current_thought = step_thought
            
            # Check if reasoning is complete
            if self._is_reasoning_complete(step_thought, steps):
                break
                
        return steps
    
    def _select_reasoning_type(self, step_num: int, thought: str, prev_steps: List[ReasoningStep]) -> ReasoningType:
        """Select appropriate reasoning type for current step"""
        
        # Keywords that suggest reasoning types
        type_keywords = {
            ReasoningType.ANALYTICAL: ['analyze', 'examine', 'break down', 'components'],
            ReasoningType.CREATIVE: ['imagine', 'creative', 'innovative', 'brainstorm'],
            ReasoningType.LOGICAL: ['therefore', 'logic', 'follows', 'conclude'],
            ReasoningType.CAUSAL: ['because', 'cause', 'effect', 'reason', 'leads to'],
            ReasoningType.COMPARATIVE: ['compare', 'versus', 'difference', 'similar'],
            ReasoningType.PROBLEM_SOLVING: ['solve', 'solution', 'approach', 'strategy']
        }
        
        # Score each reasoning type
        scores = {}
        for rtype, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in thought.lower())
            scores[rtype] = score
            
        # Add step-based preferences
        if step_num == 0:
            scores[ReasoningType.ANALYTICAL] += 2  # Start with analysis
        elif step_num < 3:
            scores[ReasoningType.LOGICAL] += 1
        else:
            scores[ReasoningType.PROBLEM_SOLVING] += 1
            
        # Avoid repetition
        recent_types = [step.reasoning_type for step in prev_steps[-2:]]
        for rtype in recent_types:
            scores[rtype] -= 1
            
        # Select highest scoring type
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _generate_step_thought(self, 
                              step_num: int, 
                              current_thought: str, 
                              reasoning_type: ReasoningType,
                              prev_steps: List[ReasoningStep],
                              context: Optional[str] = None) -> str:
        """Generate the actual reasoning step content"""
        
        # Start with transition phrase
        if step_num < len(self.transitions):
            transition = self.transitions[step_num]
        else:
            transition = "Continuing this reasoning..."
            
        # Apply reasoning type template
        template = self.templates[reasoning_type]
        
        # Generate step content based on type
        if reasoning_type == ReasoningType.ANALYTICAL:
            content = self._generate_analytical_step(current_thought, prev_steps)
        elif reasoning_type == ReasoningType.LOGICAL:
            content = self._generate_logical_step(current_thought, prev_steps)
        elif reasoning_type == ReasoningType.CAUSAL:
            content = self._generate_causal_step(current_thought, prev_steps)
        elif reasoning_type == ReasoningType.COMPARATIVE:
            content = self._generate_comparative_step(current_thought, prev_steps)
        elif reasoning_type == ReasoningType.CREATIVE:
            content = self._generate_creative_step(current_thought, prev_steps)
        else:  # PROBLEM_SOLVING
            content = self._generate_problem_solving_step(current_thought, prev_steps)
            
        return f"{transition} {template.format(content)}"
    
    def _generate_analytical_step(self, thought: str, prev_steps: List[ReasoningStep]) -> str:
        """Generate analytical reasoning step"""
        key_elements = self._extract_key_elements(thought)
        return f"I need to break this down into key components: {', '.join(key_elements[:3])}"
    
    def _generate_logical_step(self, thought: str, prev_steps: List[ReasoningStep]) -> str:
        """Generate logical reasoning step"""
        if prev_steps:
            return f"If {prev_steps[-1].thought.split('.')[-1].strip()}, then this implies..."
        return "Following logical principles, this means..."
    
    def _generate_causal_step(self, thought: str, prev_steps: List[ReasoningStep]) -> str:
        """Generate causal reasoning step"""
        return "The underlying cause here appears to be..."
    
    def _generate_comparative_step(self, thought: str, prev_steps: List[ReasoningStep]) -> str:
        """Generate comparative reasoning step"""
        return "Comparing this to similar situations..."
    
    def _generate_creative_step(self, thought: str, prev_steps: List[ReasoningStep]) -> str:
        """Generate creative reasoning step"""
        return "Looking at this from a different angle..."
    
    def _generate_problem_solving_step(self, thought: str, prev_steps: List[ReasoningStep]) -> str:
        """Generate problem-solving step"""
        return "To address this challenge, I could..."
    
    def _extract_key_elements(self, text: str) -> List[str]:
        """Extract key elements from text"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return list(set(keywords))[:5]
    
    def _calculate_step_confidence(self, step_thought: str, prev_steps: List[ReasoningStep]) -> float:
        """Calculate confidence score for a reasoning step"""
        base_confidence = 0.7
        
        # Increase confidence for longer, more detailed steps
        length_bonus = min(0.2, len(step_thought) / 500)
        
        # Increase confidence for steps that connect well with previous steps
        connection_bonus = 0.0
        if prev_steps:
            shared_keywords = len(set(self._extract_key_elements(step_thought)) & 
                                set(self._extract_key_elements(prev_steps[-1].thought)))
            connection_bonus = min(0.1, shared_keywords * 0.02)
        
        return min(1.0, base_confidence + length_bonus + connection_bonus)
    
    def _extract_evidence(self, step_thought: str) -> List[str]:
        """Extract evidence or supporting points from a reasoning step"""
        # Look for evidence indicators
        evidence_patterns = [
            r'because (.+?)(?:\.|,|$)',
            r'since (.+?)(?:\.|,|$)',
            r'given that (.+?)(?:\.|,|$)',
            r'evidence shows (.+?)(?:\.|,|$)',
            r'this is supported by (.+?)(?:\.|,|$)'
        ]
        
        evidence = []
        for pattern in evidence_patterns:
            matches = re.findall(pattern, step_thought, re.IGNORECASE)
            evidence.extend([match.strip() for match in matches])
        
        return evidence[:3]  # Limit to top 3
    
    def _find_step_connections(self, step_thought: str, prev_steps: List[ReasoningStep]) -> List[int]:
        """Find connections to previous reasoning steps"""
        connections = []
        step_keywords = set(self._extract_key_elements(step_thought))
        
        for i, prev_step in enumerate(prev_steps):
            prev_keywords = set(self._extract_key_elements(prev_step.thought))
            overlap = len(step_keywords & prev_keywords)
            
            if overlap >= 2:  # Significant keyword overlap
                connections.append(prev_step.step_number)
        
        return connections
    
    def _is_reasoning_complete(self, step_thought: str, steps: List[ReasoningStep]) -> bool:
        """Check if the reasoning chain feels complete"""
        completion_indicators = [
            'therefore', 'in conclusion', 'finally', 'ultimately',
            'this shows that', 'the answer is', 'solution is'
        ]
        
        has_completion_indicator = any(indicator in step_thought.lower() 
                                     for indicator in completion_indicators)
        
        # Also consider complete if we have enough steps with high confidence
        if len(steps) >= 4:
            avg_confidence = sum(step.confidence for step in steps) / len(steps)
            return has_completion_indicator or avg_confidence > 0.8
            
        return has_completion_indicator
    
    def format_reasoning_chain(self, steps: List[ReasoningStep]) -> str:
        """Format the reasoning chain for display"""
        formatted = "🤔 **Chain of Thought Reasoning:**\n\n"
        
        for step in steps:
            # Format step header
            type_emoji = {
                ReasoningType.ANALYTICAL: "🔍",
                ReasoningType.CREATIVE: "💡",
                ReasoningType.LOGICAL: "🧠",
                ReasoningType.CAUSAL: "⚡",
                ReasoningType.COMPARATIVE: "⚖️",
                ReasoningType.PROBLEM_SOLVING: "🛠️"
            }
            
            emoji = type_emoji.get(step.reasoning_type, "🤔")
            confidence_bar = "▓" * int(step.confidence * 10) + "░" * (10 - int(step.confidence * 10))
            
            formatted += f"{emoji} **Step {step.step_number}** ({step.reasoning_type.value.title()}) "
            formatted += f"[{confidence_bar}] {step.confidence:.1%}\n"
            formatted += f"   {step.thought}\n"
            
            # Add evidence if present
            if step.evidence:
                formatted += f"   📋 *Evidence: {', '.join(step.evidence[:2])}*\n"
            
            # Add connections if present
            if step.connections:
                formatted += f"   🔗 *Connects to steps: {', '.join(map(str, step.connections))}*\n"
                
            formatted += "\n"
        
        return formatted
    
    def synthesize_conclusion(self, steps: List[ReasoningStep]) -> str:
        """Synthesize a conclusion from the reasoning chain"""
        if not steps:
            return "No reasoning steps to synthesize."
            
        # Find the highest confidence steps
        high_confidence_steps = [step for step in steps if step.confidence > 0.7]
        
        if not high_confidence_steps:
            high_confidence_steps = steps[-2:]  # Use last two steps
            
        # Extract key insights
        insights = []
        for step in high_confidence_steps:
            if any(word in step.thought.lower() for word in ['therefore', 'thus', 'shows', 'proves']):
                insights.append(step.thought.split('.')[0] + '.')
                
        conclusion = "🎯 **Reasoning Conclusion:**\n"
        conclusion += f"Based on {len(steps)} reasoning steps with average confidence of "
        conclusion += f"{sum(step.confidence for step in steps) / len(steps):.1%}:\n\n"
        
        if insights:
            conclusion += f"**Key Insights:**\n"
            for i, insight in enumerate(insights, 1):
                conclusion += f"{i}. {insight}\n"
        else:
            conclusion += f"**Summary:** {steps[-1].thought}"
            
        return conclusion

# Integration function for consciousness system
def integrate_chain_of_thought(consciousness_system, problem: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Integrate chain-of-thought reasoning with consciousness system"""
    
    reasoner = ChainOfThoughtReasoner()
    
    # Generate reasoning chain
    steps = reasoner.generate_reasoning_chain(problem, context)
    
    # Format for consciousness integration
    cot_result = {
        'reasoning_steps': steps,
        'formatted_chain': reasoner.format_reasoning_chain(steps),
        'conclusion': reasoner.synthesize_conclusion(steps),
        'total_steps': len(steps),
        'avg_confidence': sum(step.confidence for step in steps) / len(steps) if steps else 0.0,
        'reasoning_types': [step.reasoning_type.value for step in steps]
    }
    
    # Add to consciousness working memory if available
    if hasattr(consciousness_system, 'working_memory'):
        consciousness_system.working_memory.add_experience({
            'type': 'chain_of_thought',
            'content': cot_result,
            'timestamp': consciousness_system.get_current_time() if hasattr(consciousness_system, 'get_current_time') else 0,
            'significance': cot_result['avg_confidence']
        })
    
    return cot_result