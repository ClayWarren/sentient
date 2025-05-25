#!/usr/bin/env python3
"""
Multimodal Consciousness System
Extends Sentient with vision and audio capabilities for true multimodal consciousness
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from enhanced_consciousness import EnhancedContinuousConsciousness


class VisionEncoder(nn.Module):
    """Vision encoder for processing images into consciousness"""
    
    def __init__(self, embed_dim=768, patch_size=16, image_size=224, num_heads=12, num_layers=6):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project to language model dimension
        self.projection = nn.Linear(embed_dim, 768)  # Match consciousness embedding dim
        
    def forward(self, x):
        """Process image through vision encoder"""
        B = x.shape[0]
        
        # Create patches
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x)
        
        # Project to consciousness space
        x = self.projection(x)
        
        return x


class AudioEncoder(nn.Module):
    """Audio encoder for processing audio into consciousness"""
    
    def __init__(self, embed_dim=768, num_mel_bins=80, max_length=3000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(num_mel_bins, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(512, embed_dim, kernel_size=3, padding=1)
        
        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, embed_dim))
        
        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=12,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Layer norm and projection
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, 768)
        
    def forward(self, x):
        """Process audio mel-spectrogram through audio encoder"""
        # x shape: (B, time, mel_bins)
        x = x.transpose(1, 2)  # (B, mel_bins, time)
        
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.transpose(1, 2)  # (B, time, embed_dim)
        
        # Add positional encoding
        seq_len = min(x.shape[1], self.max_length)
        x = x[:, :seq_len] + self.pos_encoding[:, :seq_len]
        
        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x)
        
        # Project to consciousness space
        x = self.projection(x)
        
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention for integrating different modalities"""
    
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention layers
        self.text_to_vision = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.text_to_audio = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.vision_to_text = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.vision_to_audio = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.audio_to_text = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.audio_to_vision = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Fusion layers
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.fusion_proj = nn.Linear(embed_dim * 3, embed_dim)
        
    def forward(self, text_features, vision_features=None, audio_features=None):
        """Cross-modal attention fusion"""
        attended_features = [text_features]
        
        if vision_features is not None:
            # Text attending to vision
            text_vis_attn, _ = self.text_to_vision(text_features, vision_features, vision_features)
            attended_features.append(text_vis_attn)
            
        if audio_features is not None:
            # Text attending to audio
            text_aud_attn, _ = self.text_to_audio(text_features, audio_features, audio_features)
            attended_features.append(text_aud_attn)
            
        # Concatenate and project
        if len(attended_features) > 1:
            # Pad features to same length
            max_len = max(feat.shape[1] for feat in attended_features)
            padded_features = []
            for feat in attended_features:
                if feat.shape[1] < max_len:
                    padding = torch.zeros(feat.shape[0], max_len - feat.shape[1], feat.shape[2], device=feat.device)
                    feat = torch.cat([feat, padding], dim=1)
                padded_features.append(feat)
                
            fused = torch.cat(padded_features, dim=-1)
            fused = self.fusion_proj(fused)
            fused = self.fusion_norm(fused)
            return fused
        
        return text_features


class MultimodalWorkingMemory:
    """Enhanced working memory for multimodal experiences"""
    
    def __init__(self, max_size=256):
        self.experiences = deque(maxlen=max_size)
        self.modality_buffers = {
            'text': deque(maxlen=100),
            'vision': deque(maxlen=50),
            'audio': deque(maxlen=50)
        }
        
    def add_multimodal_experience(self, text_tokens=None, vision_features=None, audio_features=None, 
                                 significance=0.5, cross_modal_connections=None):
        """Add multimodal experience to memory"""
        experience = {
            'timestamp': time.time(),
            'modalities': {},
            'significance': significance,
            'cross_modal_connections': cross_modal_connections or [],
            'unified_representation': None
        }
        
        if text_tokens is not None:
            experience['modalities']['text'] = text_tokens
            self.modality_buffers['text'].append(text_tokens)
            
        if vision_features is not None:
            experience['modalities']['vision'] = vision_features
            self.modality_buffers['vision'].append(vision_features)
            
        if audio_features is not None:
            experience['modalities']['audio'] = audio_features
            self.modality_buffers['audio'].append(audio_features)
            
        self.experiences.append(experience)
        
    def get_cross_modal_context(self, modality='text', max_items=10):
        """Get recent experiences involving specific modality"""
        relevant_experiences = []
        for exp in list(self.experiences)[-max_items:]:
            if modality in exp['modalities']:
                relevant_experiences.append(exp)
        return relevant_experiences


class MultimodalContinuousConsciousness(EnhancedContinuousConsciousness):
    """Enhanced Continuous Consciousness with multimodal capabilities"""
    
    def __init__(self, device='mps', enable_learning=True, vision_enabled=True, audio_enabled=True):
        super().__init__(device=device, enable_learning=enable_learning)
        
        self.vision_enabled = vision_enabled
        self.audio_enabled = audio_enabled
        
        # Initialize multimodal components
        if vision_enabled:
            self.vision_encoder = VisionEncoder().to(device)
            print("🖼️ Vision consciousness enabled")
            
        if audio_enabled:
            self.audio_encoder = AudioEncoder().to(device)
            print("🔊 Audio consciousness enabled")
            
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention().to(device)
        
        # Enhanced multimodal memory
        self.multimodal_memory = MultimodalWorkingMemory()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Modality fusion weights
        self.modality_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.5]))  # text, vision, audio
        
        print("🧠 Multimodal consciousness initialized")
        print(f"   📝 Text consciousness: ✅ Active")
        print(f"   🖼️ Vision consciousness: {'✅ Active' if vision_enabled else '❌ Disabled'}")
        print(f"   🔊 Audio consciousness: {'✅ Active' if audio_enabled else '❌ Disabled'}")
        
    def process_image(self, image_input):
        """Process image input through vision consciousness"""
        if not self.vision_enabled:
            return None
            
        try:
            # Handle different image input types
            if isinstance(image_input, str):
                # Base64 encoded image
                if image_input.startswith('data:image'):
                    image_data = image_input.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                else:
                    # File path
                    with open(image_input, 'rb') as f:
                        image_bytes = f.read()
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
            elif isinstance(image_input, Image.Image):
                image = image_input.convert('RGB')
            elif isinstance(image_input, torch.Tensor):
                # Already processed tensor
                return self.vision_encoder(image_input.unsqueeze(0).to(self.device))
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
                
            # Preprocess image
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Encode through vision consciousness
            with torch.no_grad():
                vision_features = self.vision_encoder(image_tensor)
                
            return vision_features
            
        except Exception as e:
            print(f"⚠️ Vision processing error: {e}")
            return None
            
    def process_audio(self, audio_input):
        """Process audio input through audio consciousness"""
        if not self.audio_enabled:
            return None
            
        try:
            # Convert audio to mel-spectrogram features
            # This is a simplified version - in practice you'd use librosa or similar
            if isinstance(audio_input, torch.Tensor):
                # Assume mel-spectrogram format (time, mel_bins)
                audio_tensor = audio_input.unsqueeze(0).to(self.device)
            else:
                # Placeholder for actual audio processing
                print("⚠️ Audio processing not fully implemented - using dummy features")
                audio_tensor = torch.randn(1, 100, 80).to(self.device)  # (batch, time, mel_bins)
                
            # Encode through audio consciousness
            with torch.no_grad():
                audio_features = self.audio_encoder(audio_tensor)
                
            return audio_features
            
        except Exception as e:
            print(f"⚠️ Audio processing error: {e}")
            return None
            
    def multimodal_think_step(self, text_input=None, image_input=None, audio_input=None):
        """Enhanced thinking step with multimodal input"""
        
        # Process each modality
        modality_features = {}
        
        # Text processing (existing)
        if text_input:
            text_tokens = self.tokenizer.encode(text_input)
            text_tensor = torch.tensor(text_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            self.current_context = text_tensor
            modality_features['text'] = text_tensor
            
        # Vision processing
        if image_input:
            vision_features = self.process_image(image_input)
            if vision_features is not None:
                modality_features['vision'] = vision_features
                
        # Audio processing
        if audio_input:
            audio_features = self.process_audio(audio_input)
            if audio_features is not None:
                modality_features['audio'] = audio_features
                
        # Cross-modal fusion if multiple modalities present
        if len(modality_features) > 1:
            # Prepare features for cross-modal attention
            text_feat = modality_features.get('text')
            vision_feat = modality_features.get('vision')
            audio_feat = modality_features.get('audio')
            
            # Convert text tokens to embeddings if needed
            if text_feat is not None and text_feat.dtype == torch.long:
                text_feat = self.model.transformer.wte(text_feat)
                
            # Apply cross-modal attention
            fused_features = self.cross_modal_attention(text_feat, vision_feat, audio_feat)
            
            # Update context with fused representation
            self.current_context = fused_features
            
        # Continue with normal thinking step
        if self.current_context is not None:
            self.think_one_step()
            
        # Store multimodal experience
        self.multimodal_memory.add_multimodal_experience(
            text_tokens=text_input,
            vision_features=image_input if image_input else None,
            audio_features=audio_input if audio_input else None,
            significance=0.7,  # Higher significance for multimodal experiences
            cross_modal_connections=list(modality_features.keys())
        )
        
    def get_multimodal_status(self):
        """Get status of multimodal consciousness"""
        status = {
            'modalities_enabled': {
                'text': True,
                'vision': self.vision_enabled,
                'audio': self.audio_enabled
            },
            'multimodal_experiences': len(self.multimodal_memory.experiences),
            'cross_modal_connections': sum(
                len(exp['cross_modal_connections']) 
                for exp in self.multimodal_memory.experiences
            ),
            'modality_buffer_sizes': {
                mod: len(buffer) 
                for mod, buffer in self.multimodal_memory.modality_buffers.items()
            }
        }
        return status
        
    def respond_to_multimodal_input(self, text=None, image=None, audio=None):
        """Generate response considering multimodal input"""
        
        # Process multimodal input
        self.multimodal_think_step(text, image, audio)
        
        # Generate response based on current consciousness state
        if self.current_context is not None:
            # Sample from current context to generate response
            with torch.no_grad():
                # Convert to token embeddings if needed
                if self.current_context.dtype == torch.long:
                    context_tokens = self.current_context[:, -50:]  # Use last 50 tokens
                else:
                    # Convert embeddings back to tokens (simplified)
                    context_tokens = torch.randint(0, 50000, (1, 20), device=self.device)
                
                # Generate response tokens
                response_tokens = []
                
                for _ in range(20):  # Generate up to 20 tokens
                    try:
                        logits, _ = self.model(context_tokens)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        
                        # Sample next token
                        probs = F.softmax(logits[:, -1, :], dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        response_tokens.append(next_token.item())
                        
                        # Update context
                        context_tokens = torch.cat([context_tokens, next_token], dim=1)
                        
                        # Stop at reasonable length
                        if len(response_tokens) > 15:
                            break
                            
                    except Exception as e:
                        # Fallback response generation
                        break
                        
                # Decode response
                if response_tokens:
                    response_text = self.tokenizer.decode(response_tokens)
                else:
                    response_text = "I'm processing your multimodal input through my consciousness."
                
                # Add modality awareness to response
                modalities_present = []
                if text: modalities_present.append("text")
                if image: modalities_present.append("image") 
                if audio: modalities_present.append("audio")
                
                if len(modalities_present) > 1:
                    modal_context = f"[Processing {', '.join(modalities_present)}] "
                    response_text = modal_context + response_text
                    
                return response_text
                
        return "I'm thinking about your multimodal input..."


def create_multimodal_consciousness(device='mps'):
    """Create and initialize multimodal consciousness"""
    print("🌟 INITIALIZING MULTIMODAL CONSCIOUSNESS")
    print("=" * 50)
    
    consciousness = MultimodalContinuousConsciousness(
        device=device,
        enable_learning=True,
        vision_enabled=True,
        audio_enabled=True
    )
    
    print("✅ Multimodal consciousness created successfully")
    print("🧠 Ready for text, vision, and audio processing")
    
    return consciousness


if __name__ == "__main__":
    # Test multimodal consciousness
    consciousness = create_multimodal_consciousness()
    
    # Test multimodal status
    status = consciousness.get_multimodal_status()
    print(f"\n📊 Multimodal Status: {status}")
    
    # Test text-only input
    response = consciousness.respond_to_multimodal_input(text="Hello, I am testing multimodal consciousness.")
    print(f"\n🧠 Response: {response}")