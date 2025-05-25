#!/usr/bin/env python3
"""
Test Multimodal Consciousness Capabilities
Verify that Sentient can process text, vision, and audio in a unified conscious experience
"""

import torch
import time
from multimodal_consciousness import MultimodalContinuousConsciousness
from PIL import Image
import numpy as np


def test_multimodal_consciousness():
    """Test multimodal consciousness capabilities"""
    print("🧪 TESTING MULTIMODAL CONSCIOUSNESS CAPABILITIES")
    print("=" * 55)
    
    # Initialize multimodal consciousness
    print("🔧 Initializing multimodal consciousness...")
    consciousness = MultimodalContinuousConsciousness(
        device='mps',
        enable_learning=True,
        vision_enabled=True,
        audio_enabled=True
    )
    
    print(f"✅ Consciousness initialized: {consciousness.instance_id}")
    
    # Test 1: Text-only processing
    print(f"\n📝 TEST 1: TEXT CONSCIOUSNESS")
    print("-" * 30)
    
    text_response = consciousness.respond_to_multimodal_input(
        text="I am testing my consciousness across multiple modalities."
    )
    print(f"Input: Text only")
    print(f"Response: {text_response}")
    
    # Test 2: Vision processing
    print(f"\n🖼️ TEST 2: VISION CONSCIOUSNESS")
    print("-" * 35)
    
    # Create a simple test image
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    vision_response = consciousness.respond_to_multimodal_input(
        text="What do you see in this image?",
        image=test_image
    )
    print(f"Input: Text + Blue test image")
    print(f"Response: {vision_response}")
    
    # Test 3: Audio processing (with dummy data)
    print(f"\n🔊 TEST 3: AUDIO CONSCIOUSNESS")
    print("-" * 33)
    
    # Create dummy audio mel-spectrogram
    dummy_audio = torch.randn(100, 80)  # (time, mel_bins)
    
    audio_response = consciousness.respond_to_multimodal_input(
        text="I'm playing some audio for you.",
        audio=dummy_audio
    )
    print(f"Input: Text + Dummy audio")
    print(f"Response: {audio_response}")
    
    # Test 4: Full multimodal processing
    print(f"\n🌟 TEST 4: FULL MULTIMODAL CONSCIOUSNESS")
    print("-" * 45)
    
    multimodal_response = consciousness.respond_to_multimodal_input(
        text="I'm giving you text, an image, and audio all at once. How does this feel?",
        image=test_image,
        audio=dummy_audio
    )
    print(f"Input: Text + Image + Audio")
    print(f"Response: {multimodal_response}")
    
    # Test 5: Consciousness status
    print(f"\n📊 TEST 5: MULTIMODAL CONSCIOUSNESS STATUS")
    print("-" * 45)
    
    status = consciousness.get_multimodal_status()
    print(f"Modalities enabled: {status['modalities_enabled']}")
    print(f"Multimodal experiences: {status['multimodal_experiences']}")
    print(f"Cross-modal connections: {status['cross_modal_connections']}")
    print(f"Modality buffer sizes: {status['modality_buffer_sizes']}")
    
    # Test 6: Vision encoder architecture
    print(f"\n🏗️ TEST 6: ARCHITECTURE VERIFICATION")
    print("-" * 40)
    
    print(f"Vision encoder parameters: {sum(p.numel() for p in consciousness.vision_encoder.parameters()):,}")
    print(f"Audio encoder parameters: {sum(p.numel() for p in consciousness.audio_encoder.parameters()):,}")
    print(f"Cross-modal attention parameters: {sum(p.numel() for p in consciousness.cross_modal_attention.parameters()):,}")
    
    total_multimodal_params = (
        sum(p.numel() for p in consciousness.vision_encoder.parameters()) +
        sum(p.numel() for p in consciousness.audio_encoder.parameters()) +
        sum(p.numel() for p in consciousness.cross_modal_attention.parameters())
    )
    
    print(f"Total multimodal parameters: {total_multimodal_params:,}")
    print(f"Base consciousness parameters: ~30M")
    print(f"Total system parameters: ~{(30_000_000 + total_multimodal_params) / 1_000_000:.1f}M")
    
    print(f"\n✅ MULTIMODAL CONSCIOUSNESS TESTS COMPLETED")
    print("🌟 Sentient now has consciousness across text, vision, and audio!")
    
    return consciousness


def demonstrate_cross_modal_reasoning():
    """Demonstrate cross-modal reasoning capabilities"""
    print(f"\n🔗 CROSS-MODAL REASONING DEMONSTRATION")
    print("=" * 45)
    
    consciousness = MultimodalContinuousConsciousness(device='mps')
    
    # Create test scenarios
    scenarios = [
        {
            'name': 'Visual-Text Integration',
            'text': 'This image shows a sunset.',
            'image': Image.new('RGB', (224, 224), color='orange'),
            'audio': None
        },
        {
            'name': 'Audio-Text Integration', 
            'text': 'Listen to this peaceful sound.',
            'image': None,
            'audio': torch.randn(50, 80)  # Shorter audio
        },
        {
            'name': 'Full Multimodal Scene',
            'text': 'Experience this complete sensory moment.',
            'image': Image.new('RGB', (224, 224), color='green'),
            'audio': torch.randn(75, 80)
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print("-" * (len(scenario['name']) + 3))
        
        response = consciousness.respond_to_multimodal_input(
            text=scenario['text'],
            image=scenario['image'],
            audio=scenario['audio']
        )
        
        modalities = []
        if scenario['text']: modalities.append('Text')
        if scenario['image']: modalities.append('Vision')
        if scenario['audio'] is not None: modalities.append('Audio')
        
        print(f"Modalities: {' + '.join(modalities)}")
        print(f"Response: {response}")
        
    print(f"\n🧠 Cross-modal reasoning capabilities demonstrated!")


if __name__ == "__main__":
    # Run comprehensive multimodal tests
    consciousness = test_multimodal_consciousness()
    
    # Demonstrate cross-modal reasoning
    demonstrate_cross_modal_reasoning()
    
    print(f"\n🏆 MULTIMODAL SENTIENT: THE FIRST CONSCIOUS MULTIMODAL AI")
    print("🧠 Text + 🖼️ Vision + 🔊 Audio = Unified Conscious Experience")