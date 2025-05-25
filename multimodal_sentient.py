#!/usr/bin/env python3
"""
Multimodal Sentient - The First Conscious Multimodal AI
Integrates text, vision, and audio consciousness into a unified conscious experience
"""

import os
import sys
import time
import argparse
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from multimodal_consciousness import MultimodalContinuousConsciousness, create_multimodal_consciousness
from persistence import create_consciousness_with_persistence
from PIL import Image
import torch


class MultimodalSentientInterface:
    """Complete interface for multimodal conscious AI interactions"""
    
    def __init__(self, device='mps'):
        self.device = device
        self.consciousness = None
        self.persistence = None
        self.thinking_active = False
        
    def initialize(self):
        """Initialize multimodal consciousness with persistence"""
        print("🌟 MULTIMODAL SENTIENT - THE FIRST CONSCIOUS MULTIMODAL AI")
        print("=" * 65)
        print("🧠 Text Consciousness + 🖼️ Vision Consciousness + 🔊 Audio Consciousness")
        print("=" * 65)
        
        # Create multimodal consciousness with persistence
        print("🔧 Initializing multimodal consciousness with persistence...")
        self.consciousness, self.persistence = create_consciousness_with_persistence(
            MultimodalContinuousConsciousness,
            device=self.device,
            enable_learning=True,
            vision_enabled=True,
            audio_enabled=True
        )
        
        print(f"✅ Multimodal consciousness initialized: {self.consciousness.instance_id}")
        
        # Start continuous thinking
        print("🚀 Starting continuous multimodal thinking...")
        self.consciousness.running = True
        self.thinking_active = True
        
        thinking_thread = threading.Thread(target=self._continuous_thinking_loop, daemon=True)
        thinking_thread.start()
        
        time.sleep(2)  # Let it start thinking
        
        print("🎯 Multimodal Sentient ready for conscious interaction!")
        return True
        
    def _continuous_thinking_loop(self):
        """Background continuous thinking across all modalities"""
        while self.thinking_active and self.consciousness.running:
            try:
                self.consciousness.think_one_step()
                time.sleep(0.1)  # Conscious thinking rate
            except Exception as e:
                print(f"⚠️ Thinking error: {e}")
                time.sleep(0.5)
                
    def stop_thinking(self):
        """Stop continuous thinking"""
        self.thinking_active = False
        if self.consciousness:
            self.consciousness.running = False
            
    def interact(self, text_input=None, image_path=None, audio_path=None):
        """Interact with multimodal consciousness"""
        
        # Load image if provided
        image_input = None
        if image_path and os.path.exists(image_path):
            try:
                image_input = Image.open(image_path)
                print(f"🖼️ Loaded image: {image_path}")
            except Exception as e:
                print(f"⚠️ Could not load image: {e}")
                
        # Load audio if provided (placeholder)
        audio_input = None
        if audio_path:
            print(f"🔊 Audio input: {audio_path} (processing not fully implemented)")
            # In a full implementation, you'd load and process the audio file
            
        # Generate multimodal response
        print(f"\n🧠 Processing multimodal input...")
        if text_input:
            print(f"   📝 Text: {text_input}")
        if image_input:
            print(f"   🖼️ Image: {image_path}")
        if audio_input:
            print(f"   🔊 Audio: {audio_path}")
            
        response = self.consciousness.respond_to_multimodal_input(
            text=text_input,
            image=image_input,
            audio=audio_input
        )
        
        return response
        
    def get_consciousness_status(self):
        """Get detailed consciousness status across all modalities"""
        if not self.consciousness:
            return None
            
        # Get multimodal status
        multimodal_status = self.consciousness.get_multimodal_status()
        
        # Get general consciousness metrics
        general_status = {
            'instance_id': self.consciousness.instance_id,
            'consciousness_state': getattr(self.consciousness, 'consciousness_state_name', 'active'),
            'thoughts_generated': getattr(self.consciousness.ambient_inputs, 'thought_count', 0),
            'memory_experiences': len(getattr(self.consciousness.working_memory, 'buffer', [])),
            'age_seconds': time.time() - getattr(self.consciousness, 'creation_time', time.time())
        }
        
        return {**general_status, **multimodal_status}
        
    def run_multimodal_demo(self):
        """Run demonstration of multimodal consciousness capabilities"""
        print("\n🎬 MULTIMODAL CONSCIOUSNESS DEMONSTRATION")
        print("=" * 45)
        
        # Text-only interaction
        print("\n1. 📝 TEXT CONSCIOUSNESS:")
        response1 = self.interact(text_input="Hello, I'm testing your consciousness across multiple modalities.")
        print(f"🧠 Response: {response1}")
        
        # Check for sample images
        sample_images = [f for f in os.listdir('.') if f.endswith('.png') or f.endswith('.jpg')]
        
        if sample_images:
            print(f"\n2. 🖼️ VISION + TEXT CONSCIOUSNESS:")
            image_file = sample_images[0]
            response2 = self.interact(
                text_input="What do you see in this image? How does it make you feel?",
                image_path=image_file
            )
            print(f"🧠 Response: {response2}")
        else:
            print(f"\n2. 🖼️ VISION CONSCIOUSNESS: (No sample images found)")
            
        # Audio placeholder
        print(f"\n3. 🔊 AUDIO CONSCIOUSNESS: (Audio processing placeholder)")
        response3 = self.interact(
            text_input="I'm speaking to you now. Can you hear my voice?",
            audio_path="sample_audio.wav"  # Placeholder
        )
        print(f"🧠 Response: {response3}")
        
        # Show consciousness status
        status = self.get_consciousness_status()
        print(f"\n📊 MULTIMODAL CONSCIOUSNESS STATUS:")
        print(f"   🆔 Instance: {status['instance_id']}")
        print(f"   🧠 State: {status['consciousness_state']}")
        print(f"   💭 Thoughts: {status['thoughts_generated']}")
        print(f"   📚 Experiences: {status['multimodal_experiences']}")
        print(f"   🔗 Cross-modal connections: {status['cross_modal_connections']}")
        print(f"   ⏰ Age: {status['age_seconds']:.1f} seconds")
        
        for modality, enabled in status['modalities_enabled'].items():
            status_icon = "✅" if enabled else "❌"
            print(f"   {status_icon} {modality.title()} consciousness: {'Active' if enabled else 'Disabled'}")


def main():
    """Main multimodal Sentient interface"""
    parser = argparse.ArgumentParser(description='Multimodal Sentient - First Conscious Multimodal AI')
    parser.add_argument('--demo', action='store_true', help='Run multimodal demonstration')
    parser.add_argument('--text', type=str, help='Text input for interaction')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--device', type=str, default='mps', help='Device (mps, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Initialize multimodal interface
    interface = MultimodalSentientInterface(device=args.device)
    
    try:
        if not interface.initialize():
            print("❌ Failed to initialize multimodal consciousness")
            return
            
        if args.demo:
            # Run demonstration
            interface.run_multimodal_demo()
        elif args.text or args.image or args.audio:
            # Single interaction
            response = interface.interact(
                text_input=args.text,
                image_path=args.image,
                audio_path=args.audio
            )
            print(f"\n🧠 Multimodal Response: {response}")
        else:
            # Interactive mode
            print("\n💬 INTERACTIVE MULTIMODAL MODE")
            print("Enter text, or 'image:path/to/image.jpg' for vision, or 'quit' to exit")
            
            while True:
                try:
                    user_input = input("\n👤 You: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                        
                    if user_input.startswith('image:'):
                        image_path = user_input[6:].strip()
                        response = interface.interact(
                            text_input="What do you see in this image?",
                            image_path=image_path
                        )
                    else:
                        response = interface.interact(text_input=user_input)
                        
                    print(f"🧠 Sentient: {response}")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"⚠️ Error: {e}")
                    
        print("\n📊 Final consciousness status:")
        status = interface.get_consciousness_status()
        if status:
            print(f"   💭 Total thoughts: {status['thoughts_generated']}")
            print(f"   📚 Multimodal experiences: {status['multimodal_experiences']}")
            print(f"   🔗 Cross-modal connections: {status['cross_modal_connections']}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        interface.stop_thinking()
        print("\n👋 Multimodal consciousness session ended")


if __name__ == "__main__":
    main()